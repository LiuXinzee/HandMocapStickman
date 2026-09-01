/*
 * useSequenceRecorder — 动态手语词的多路同步序列录制
 *
 * 三路数据源的时钟不同步且速率不同：
 *   - 左/右手套：各自 COM 口独立上报，实测 ~100Hz，两只手之间无硬件同步
 *   - 视觉：MediaPipe 回调，~30Hz，且会整帧丢手
 * 所以录制期间三路各自带 performance.now() 时间戳堆进缓冲，停止时再统一
 * 重采样到一个公共栅格（默认 50Hz —— 手语动作带宽远低于 25Hz，50Hz 足够，
 * 同时把存储压到约 64KB/条）。这与 useSyncRecorder.ts 是同一套时钟策略。
 *
 * 手套帧必须走 useDualGloveSerial 的 onLeftFrame/onRightFrame **全速回调**收，
 * 不能轮询 latestFrameRef —— 那个 ref 由 setLatestFrame 按 targetFps 节流更新，
 * 轮询会丢帧，动态词的轨迹细节正好丢在这里。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type { GloveFrame } from "@/lib/gloveProtocol";
import type { HandResult } from "./useHandTracking";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "@/lib/datasetStore";
import { analyzeSequenceImu } from "@/lib/imuHealth";
import { extractHandLandmarks } from "@/lib/visionLandmarks";

export interface SequenceRecorderOptions {
  /** 公共重采样栅格频率 */
  gridFps?: number;
  /** 单条录制的最长时长，超过自动停止（防止忘记松手把内存吃光） */
  maxDurationMs?: number;
  /**
   * 视觉最近邻的最大时间差。超过这个差值就判定该栅格点没有视觉，填 NaN。
   * 30Hz 视觉的帧间隔是 33ms，取 50ms 允许一帧抖动但不允许跨两帧硬凑。
   */
  visionMaxGapMs?: number;
  /** MediaPipe 结果 ref（录制期间由内部 rAF 循环采样） */
  handResultsRef?: React.RefObject<HandResult | null>;
}

interface GloveEntry {
  t: number;
  sensor: number[];
  quat: [number, number, number, number];
  acc: [number, number, number] | null;
  att: [number, number, number] | null;
}

interface VisionEntry {
  t: number;
  left: Float32Array | null; // 63
  right: Float32Array | null;
}

export interface RecorderLiveStats {
  elapsedMs: number;
  leftFrames: number;
  rightFrames: number;
  visionFrames: number;
}

const DEFAULTS = {
  gridFps: 50,
  maxDurationMs: 6000,
  visionMaxGapMs: 50,
};

/*
 * `extractHandLandmarks` 原来是这里的私有函数，已搬到 `lib/visionLandmarks.ts` ——
 * 句子采集页（`/collect-sentence`）也要用同一份，两边排布必须逐位相同。
 */

/**
 * 游标式最近邻查找：buffer 按时间递增，startIdx 从上次结果继续，
 * 整趟重采样是 O(N+M) 而不是 O(N·M)。同 useSyncRecorder.ts 的做法。
 */
function findNearest<T extends { t: number }>(
  buffer: T[],
  targetTime: number,
  startIdx: number
): { entry: T | null; idx: number } {
  if (buffer.length === 0) return { entry: null, idx: 0 };
  let i = Math.max(0, Math.min(startIdx, buffer.length - 1));
  while (i + 1 < buffer.length && buffer[i + 1].t <= targetTime) i++;
  // i 是最后一个 <= targetTime 的位置（或 0），比较它与下一个谁更近
  let best = i;
  if (
    i + 1 < buffer.length &&
    Math.abs(buffer[i + 1].t - targetTime) < Math.abs(buffer[i].t - targetTime)
  ) {
    best = i + 1;
  }
  return { entry: buffer[best], idx: i };
}

export function useSequenceRecorder(options: SequenceRecorderOptions = {}) {
  const gridFps = options.gridFps ?? DEFAULTS.gridFps;
  const maxDurationMs = options.maxDurationMs ?? DEFAULTS.maxDurationMs;
  const visionMaxGapMs = options.visionMaxGapMs ?? DEFAULTS.visionMaxGapMs;
  const handResultsRef = options.handResultsRef;

  const [isRecording, setIsRecording] = useState(false);
  const [stats, setStats] = useState<RecorderLiveStats>({
    elapsedMs: 0,
    leftFrames: 0,
    rightFrames: 0,
    visionFrames: 0,
  });

  const recordingRef = useRef(false);
  const startTimeRef = useRef(0);
  const leftBufRef = useRef<GloveEntry[]>([]);
  const rightBufRef = useRef<GloveEntry[]>([]);
  const visionBufRef = useRef<VisionEntry[]>([]);
  const lastVisionObjRef = useRef<HandResult | null>(null);
  const rafRef = useRef<number | null>(null);
  const statsTimerRef = useRef<number | null>(null);

  const toEntry = (frame: GloveFrame): GloveEntry => ({
    t: frame.timestamp,
    sensor: frame.mapped_data,
    quat: frame.quaternion,
    acc: frame.acceleration,
    att: frame.attitude,
  });

  const pushLeftFrame = useCallback((frame: GloveFrame) => {
    if (!recordingRef.current) return;
    leftBufRef.current.push(toEntry(frame));
  }, []);

  const pushRightFrame = useCallback((frame: GloveFrame) => {
    if (!recordingRef.current) return;
    rightBufRef.current.push(toEntry(frame));
  }, []);

  /** 视觉采样循环：按对象引用去重，MediaPipe 没出新结果就不重复记 */
  const sampleVision = useCallback(() => {
    if (!recordingRef.current) return;
    const r = handResultsRef?.current ?? null;
    if (r && r !== lastVisionObjRef.current) {
      lastVisionObjRef.current = r;
      visionBufRef.current.push({
        t: performance.now(),
        left: extractHandLandmarks(r, "Left"),
        right: extractHandLandmarks(r, "Right"),
      });
    }
    rafRef.current = requestAnimationFrame(sampleVision);
  }, [handResultsRef]);

  const start = useCallback(() => {
    if (recordingRef.current) return;
    leftBufRef.current = [];
    rightBufRef.current = [];
    visionBufRef.current = [];
    lastVisionObjRef.current = null;
    startTimeRef.current = performance.now();
    recordingRef.current = true;
    setIsRecording(true);
    rafRef.current = requestAnimationFrame(sampleVision);
    statsTimerRef.current = window.setInterval(() => {
      setStats({
        elapsedMs: performance.now() - startTimeRef.current,
        leftFrames: leftBufRef.current.length,
        rightFrames: rightBufRef.current.length,
        visionFrames: visionBufRef.current.length,
      });
    }, 100);
  }, [sampleVision]);

  const teardown = useCallback(() => {
    recordingRef.current = false;
    setIsRecording(false);
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (statsTimerRef.current !== null) clearInterval(statsTimerRef.current);
    statsTimerRef.current = null;
  }, []);

  /**
   * 停止录制并重采样成 SequenceSample。
   * 返回 null 表示这条录废了（时长太短或完全没有手套数据），调用方应提示重录。
   */
  const stop = useCallback(
    (label: string): SequenceSample | null => {
      if (!recordingRef.current) return null;
      const endTime = performance.now();
      teardown();

      const t0 = startTimeRef.current;
      const durationMs = endTime - t0;
      const dt = 1000 / gridFps;
      const T = Math.floor(durationMs / dt);

      const leftBuf = leftBufRef.current;
      const rightBuf = rightBufRef.current;
      const visionBuf = visionBufRef.current;

      if (T < 4 || (leftBuf.length === 0 && rightBuf.length === 0)) {
        return null;
      }

      const timestamps = new Float32Array(T);
      const hasLeft = leftBuf.length > 0;
      const hasRight = rightBuf.length > 0;
      const hasVision = visionBuf.length > 0;

      const leftSensor = hasLeft ? new Uint8Array(T * SEQ_SENSOR_N) : null;
      const rightSensor = hasRight ? new Uint8Array(T * SEQ_SENSOR_N) : null;
      const leftImu = hasLeft ? new Float32Array(T * SEQ_IMU_N) : null;
      const rightImu = hasRight ? new Float32Array(T * SEQ_IMU_N) : null;
      // 只要这只手有手套数据就分配视觉列（缺帧填 NaN），
      // 这样后续 fillVisionGaps 能区分"这只手没戴手套"和"这只手视觉丢了"
      const leftLandmarks =
        hasLeft && hasVision ? new Float32Array(T * SEQ_LANDMARK_N) : null;
      const rightLandmarks =
        hasRight && hasVision ? new Float32Array(T * SEQ_LANDMARK_N) : null;

      let li = 0;
      let ri = 0;
      let vi = 0;

      for (let t = 0; t < T; t++) {
        const target = t0 + t * dt;
        timestamps[t] = t * dt;

        if (leftSensor && leftImu) {
          const r = findNearest(leftBuf, target, li);
          li = r.idx;
          writeGlove(r.entry, leftSensor, leftImu, t);
        }
        if (rightSensor && rightImu) {
          const r = findNearest(rightBuf, target, ri);
          ri = r.idx;
          writeGlove(r.entry, rightSensor, rightImu, t);
        }
        if (leftLandmarks || rightLandmarks) {
          const r = findNearest(visionBuf, target, vi);
          vi = r.idx;
          const inGap =
            r.entry !== null && Math.abs(r.entry.t - target) <= visionMaxGapMs;
          if (leftLandmarks) {
            writeVision(inGap ? r.entry!.left : null, leftLandmarks, t);
          }
          if (rightLandmarks) {
            writeVision(inGap ? r.entry!.right : null, rightLandmarks, t);
          }
        }
      }

      // IMU 健康度：**事后**在已重采样好的 leftImu/rightImu 上算（每帧 acc 在 [4:7]），
      // 不在录制热路径上做实时监控——数据本来就都在这两个数组里，录制时多算一遍纯浪费。
      const imuHealth = {
        left: leftImu ? analyzeSequenceImu(leftImu, T, timestamps) : null,
        right: rightImu ? analyzeSequenceImu(rightImu, T, timestamps) : null,
      };

      return {
        segments: [{ label, startFrame: 0, endFrame: T }],
        primaryLabel: label,
        frameCount: T,
        timestamps,
        leftSensor,
        rightSensor,
        leftImu,
        rightImu,
        leftLandmarks,
        rightLandmarks,
        durationMs,
        sourceFps: gridFps,
        origin: "recorded",
        timestamp: Date.now(),
        imuHealth,
      };
    },
    [gridFps, teardown, visionMaxGapMs]
  );

  const cancel = useCallback(() => {
    if (!recordingRef.current) return;
    teardown();
    leftBufRef.current = [];
    rightBufRef.current = [];
    visionBufRef.current = [];
  }, [teardown]);

  // 超时保护：忘记松手时自动停在 maxDurationMs，缓冲不会无限增长
  useEffect(() => {
    if (!isRecording) return;
    const id = window.setTimeout(() => {
      if (recordingRef.current) teardown();
    }, maxDurationMs);
    return () => clearTimeout(id);
  }, [isRecording, maxDurationMs, teardown]);

  useEffect(() => () => teardown(), [teardown]);

  return {
    isRecording,
    stats,
    start,
    stop,
    cancel,
    pushLeftFrame,
    pushRightFrame,
  };
}

function writeGlove(
  entry: GloveEntry | null,
  sensor: Uint8Array,
  imu: Float32Array,
  t: number
): void {
  const so = t * SEQ_SENSOR_N;
  const io = t * SEQ_IMU_N;
  if (!entry) {
    // 该栅格点没有任何手套帧（只可能出现在录制起点之前），保持 0
    imu[io] = 1; // 单位四元数，避免 0 向量归一化时退化
    return;
  }
  const n = Math.min(SEQ_SENSOR_N, entry.sensor.length);
  for (let c = 0; c < n; c++) {
    const v = entry.sensor[c];
    sensor[so + c] = v < 0 ? 0 : v > 255 ? 255 : v;
  }
  imu[io] = entry.quat[0];
  imu[io + 1] = entry.quat[1];
  imu[io + 2] = entry.quat[2];
  imu[io + 3] = entry.quat[3];
  if (entry.acc) {
    imu[io + 4] = entry.acc[0];
    imu[io + 5] = entry.acc[1];
    imu[io + 6] = entry.acc[2];
  }
  if (entry.att) {
    imu[io + 7] = entry.att[0];
    imu[io + 8] = entry.att[1];
    imu[io + 9] = entry.att[2];
  }
}

function writeVision(
  lm: Float32Array | null,
  dst: Float32Array,
  t: number
): void {
  const o = t * SEQ_LANDMARK_N;
  if (!lm) {
    dst.fill(NaN, o, o + SEQ_LANDMARK_N);
    return;
  }
  dst.set(lm, o);
}
