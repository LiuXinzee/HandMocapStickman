/*
 * useSyncRecorder — 同步录制 Hook（双手）
 * 同时录制视频手部关键点和左右手手套传感器数据，统一时间轴。
 *
 * 录制策略:
 * - 视频帧: 每次 MediaPipe 检测到结果时记录（~30fps，最多两只手）
 * - 手套帧: 每次收到串口数据时记录（~100Hz/手），按 hand 路由到左/右手缓冲
 * - 三路数据各自带 performance.now() 高精度时间戳（与视觉同一时钟）
 * - 导出时按时间戳最近邻匹配（左右手各自匹配到视频帧）
 */
import { useCallback, useEffect, useRef, useState } from "react";
import type {
  HandDetectionSource,
  HandResult,
  HandSurface,
} from "./useHandTracking";
import type { GloveFrame } from "@/lib/gloveProtocol";

export interface VideoFrameRecord {
  /** 相对录制开始的时间(ms) */
  relativeTime: number;
  /** 绝对时间戳(ms) */
  absoluteTime: number;
  /** 手部关键点（可能含左右手多组） */
  landmarks: { x: number; y: number; z: number }[][] | null;
  /** 左右手标签（与 landmarks 顺序对应，"Left"/"Right"） */
  handedness: string[];
  /** 手套表面及置信度（与 landmarks 顺序对应） */
  surfaces: HandSurface[];
  surfaceConfidences: number[];
  gloveConfidences: number[];
  detectionSource: HandDetectionSource | null;
}

export interface GloveFrameRecord {
  /** 相对录制开始的时间(ms) */
  relativeTime: number;
  /** 绝对时间戳(ms) */
  absoluteTime: number;
  /** 手套原始时间戳(ms, performance.now()) */
  gloveTimestamp: number;
  /** 手别标识 0x01=左 0x02=右 */
  hand: number;
  /** 137个有效传感点（物理顺序，已重映射） */
  sensorData: number[];
  /** IMU 四元数 [w,x,y,z] */
  quaternion: [number, number, number, number];
  /** 加速度 [x,y,z]（旧手套为 null） */
  acceleration: [number, number, number] | null;
  /** 姿态角 [yaw,roll,pitch]（旧手套为 null） */
  attitude: [number, number, number] | null;
  /** 帧序号 */
  frameId: number;
}

interface RecordingStats {
  duration: number;
  videoFrames: number;
  gloveFramesLeft: number;
  gloveFramesRight: number;
  avgVideoFps: number;
  avgGloveFpsLeft: number;
  avgGloveFpsRight: number;
}

interface UseSyncRecorderReturn {
  isRecording: boolean;
  recordingDuration: number;
  videoFrameCount: number;
  /** 左右手手套帧总数（用于 UI 概览） */
  gloveFrameCount: number;
  gloveFrameCountLeft: number;
  gloveFrameCountRight: number;
  startRecording: () => void;
  stopRecording: () => void;
  recordVideoFrame: (result: HandResult | null) => void;
  recordGloveFrame: (frame: GloveFrame) => void;
  exportSyncedCSV: () => string;
  exportRawJSON: () => string;
  getStats: () => RecordingStats;
  clear: () => void;
}

export function useSyncRecorder(): UseSyncRecorderReturn {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [videoFrameCount, setVideoFrameCount] = useState(0);
  const [gloveFrameCountLeft, setGloveFrameCountLeft] = useState(0);
  const [gloveFrameCountRight, setGloveFrameCountRight] = useState(0);

  const startTimeRef = useRef<number>(0);
  const isRecordingRef = useRef(false);
  const videoBufferRef = useRef<VideoFrameRecord[]>([]);
  const gloveBufferLeftRef = useRef<GloveFrameRecord[]>([]);
  const gloveBufferRightRef = useRef<GloveFrameRecord[]>([]);
  const durationTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      isRecordingRef.current = false;
      if (durationTimerRef.current) {
        clearInterval(durationTimerRef.current);
        durationTimerRef.current = null;
      }
    };
  }, []);

  const startRecording = useCallback(() => {
    videoBufferRef.current = [];
    gloveBufferLeftRef.current = [];
    gloveBufferRightRef.current = [];
    startTimeRef.current = performance.now();
    isRecordingRef.current = true;
    setVideoFrameCount(0);
    setGloveFrameCountLeft(0);
    setGloveFrameCountRight(0);
    setRecordingDuration(0);
    setIsRecording(true);

    durationTimerRef.current = setInterval(() => {
      setRecordingDuration(performance.now() - startTimeRef.current);
    }, 100);

    console.log("[Recorder] Recording started");
  }, []);

  const stopRecording = useCallback(() => {
    isRecordingRef.current = false;
    setIsRecording(false);
    if (durationTimerRef.current) {
      clearInterval(durationTimerRef.current);
      durationTimerRef.current = null;
    }
    setRecordingDuration(performance.now() - startTimeRef.current);
    console.log(
      "[Recorder] Recording stopped. Video:",
      videoBufferRef.current.length,
      "GloveL:",
      gloveBufferLeftRef.current.length,
      "GloveR:",
      gloveBufferRightRef.current.length
    );
  }, []);

  const recordVideoFrame = useCallback((result: HandResult | null) => {
    if (!isRecordingRef.current) return;
    const now = performance.now();
    const record: VideoFrameRecord = {
      relativeTime: now - startTimeRef.current,
      absoluteTime: now,
      landmarks: result?.landmarks ?? null,
      handedness: result?.handedness ?? [],
      surfaces: result?.surfaces ?? [],
      surfaceConfidences: result?.surfaceConfidences ?? [],
      gloveConfidences: result?.gloveConfidences ?? [],
      detectionSource: result?.detectionSource ?? null,
    };
    videoBufferRef.current.push(record);
    setVideoFrameCount(videoBufferRef.current.length);
  }, []);

  const recordGloveFrame = useCallback((frame: GloveFrame) => {
    if (!isRecordingRef.current) return;
    const now = performance.now();
    const record: GloveFrameRecord = {
      relativeTime: now - startTimeRef.current,
      absoluteTime: now,
      gloveTimestamp: frame.timestamp,
      hand: frame.hand,
      sensorData: frame.mapped_data,
      quaternion: frame.quaternion,
      acceleration: frame.acceleration,
      attitude: frame.attitude,
      frameId: frame.frame_id,
    };
    if (frame.hand === 0x02) {
      gloveBufferRightRef.current.push(record);
      setGloveFrameCountRight(gloveBufferRightRef.current.length);
    } else {
      gloveBufferLeftRef.current.push(record);
      setGloveFrameCountLeft(gloveBufferLeftRef.current.length);
    }
  }, []);

  /** 在有序手套缓冲中，找与 targetTime 最近的帧（附带搜索游标优化） */
  const findNearest = (
    buffer: GloveFrameRecord[],
    targetTime: number,
    startIdx: number
  ): { rec: GloveFrameRecord | null; idx: number; delta: number } => {
    let best: GloveFrameRecord | null = null;
    let bestDistance = Infinity;
    let bestSignedDelta = Infinity;
    let bestIdx = startIdx;
    for (let i = Math.max(0, startIdx - 5); i < buffer.length; i++) {
      const signedDelta = buffer[i].relativeTime - targetTime;
      const distance = Math.abs(signedDelta);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestSignedDelta = signedDelta;
        best = buffer[i];
        bestIdx = i;
      } else if (distance > bestDistance) {
        break; // 已过最近点（缓冲时间有序）
      }
    }
    return {
      rec: best,
      idx: bestIdx,
      delta: best ? bestSignedDelta : Infinity,
    };
  };

  /** 取视频帧中指定手别的 21 关键点（按 handedness 匹配），无则 null */
  const getHandLandmarks = (
    vf: VideoFrameRecord,
    label: "Left" | "Right"
  ): { x: number; y: number; z: number }[] | null => {
    if (!vf.landmarks) return null;
    const idx = vf.handedness.findIndex(h => h === label);
    if (idx >= 0 && vf.landmarks[idx]) return vf.landmarks[idx];
    return null;
  };

  const SENSOR_N = 137;

  const exportSyncedCSV = useCallback((): string => {
    const videoFrames = videoBufferRef.current;
    if (videoFrames.length === 0) return "";

    const gloveL = gloveBufferLeftRef.current;
    const gloveR = gloveBufferRightRef.current;

    // 表头
    const headers: string[] = ["relative_time_ms"];
    // 视觉：左右手各 21×3
    for (const side of ["LH", "RH"]) {
      for (let i = 0; i < 21; i++) {
        headers.push(
          `${side}_lm_${i}_x`,
          `${side}_lm_${i}_y`,
          `${side}_lm_${i}_z`
        );
      }
    }
    // 触觉：左右手各 137 传感 + 四元数 + 加速度 + 姿态角 + 同步偏差 + 帧号
    for (const side of ["LH", "RH"]) {
      for (let i = 0; i < SENSOR_N; i++) headers.push(`${side}_sensor_${i}`);
      headers.push(
        `${side}_quat_w`,
        `${side}_quat_x`,
        `${side}_quat_y`,
        `${side}_quat_z`
      );
      headers.push(`${side}_acc_x`, `${side}_acc_y`, `${side}_acc_z`);
      headers.push(`${side}_att_yaw`, `${side}_att_roll`, `${side}_att_pitch`);
      headers.push(`${side}_sync_delta_ms`, `${side}_frame_id`);
    }
    headers.push(
      "LH_surface",
      "LH_surface_confidence",
      "LH_glove_confidence",
      "RH_surface",
      "RH_surface_confidence",
      "RH_glove_confidence",
      "vision_detection_source"
    );

    const rows: string[] = [headers.join(",")];

    let lIdx = 0;
    let rIdx = 0;

    const pushGloveCols = (
      row: string[],
      rec: GloveFrameRecord | null,
      delta: number
    ) => {
      if (rec) {
        for (let i = 0; i < SENSOR_N; i++)
          row.push(String(rec.sensorData[i] ?? 0));
        row.push(...rec.quaternion.map(v => v.toFixed(6)));
        const acc = rec.acceleration ?? [0, 0, 0];
        row.push(...acc.map(v => v.toFixed(6)));
        const att = rec.attitude ?? [0, 0, 0];
        row.push(...att.map(v => v.toFixed(6)));
        row.push(delta.toFixed(2), String(rec.frameId));
      } else {
        // 137 + 4 + 3 + 3 + 2 = 149 空列
        for (let i = 0; i < SENSOR_N + 4 + 3 + 3 + 2; i++) row.push("");
      }
    };

    const pushLandmarkCols = (
      row: string[],
      lms: { x: number; y: number; z: number }[] | null
    ) => {
      if (lms && lms.length === 21) {
        for (let i = 0; i < 21; i++) {
          row.push(
            lms[i].x.toFixed(6),
            lms[i].y.toFixed(6),
            lms[i].z.toFixed(6)
          );
        }
      } else {
        for (let i = 0; i < 63; i++) row.push("");
      }
    };

    for (const vf of videoFrames) {
      const row: string[] = [vf.relativeTime.toFixed(2)];

      // 视觉：左右手
      pushLandmarkCols(row, getHandLandmarks(vf, "Left"));
      pushLandmarkCols(row, getHandLandmarks(vf, "Right"));

      // 触觉：左右手各自最近邻
      const lMatch = findNearest(gloveL, vf.relativeTime, lIdx);
      lIdx = lMatch.idx;
      pushGloveCols(row, lMatch.rec, lMatch.delta);

      const rMatch = findNearest(gloveR, vf.relativeTime, rIdx);
      rIdx = rMatch.idx;
      pushGloveCols(row, rMatch.rec, rMatch.delta);

      for (const label of ["Left", "Right"] as const) {
        const index = vf.handedness.findIndex(hand => hand === label);
        row.push(
          index >= 0 ? (vf.surfaces[index] ?? "unknown") : "",
          index >= 0 ? (vf.surfaceConfidences[index] ?? 0).toFixed(4) : "",
          index >= 0 ? (vf.gloveConfidences[index] ?? 0).toFixed(4) : ""
        );
      }
      row.push(vf.detectionSource ?? "");

      rows.push(row.join(","));
    }

    return rows.join("\n");
  }, []);

  const exportRawJSON = useCallback((): string => {
    return JSON.stringify(
      {
        metadata: {
          recordingStartTime: startTimeRef.current,
          totalDuration: recordingDuration,
          videoFrames: videoBufferRef.current.length,
          gloveFramesLeft: gloveBufferLeftRef.current.length,
          gloveFramesRight: gloveBufferRightRef.current.length,
          exportTime: new Date().toISOString(),
        },
        videoData: videoBufferRef.current,
        gloveDataLeft: gloveBufferLeftRef.current,
        gloveDataRight: gloveBufferRightRef.current,
      },
      null,
      2
    );
  }, [recordingDuration]);

  const getStats = useCallback((): RecordingStats => {
    const duration = recordingDuration / 1000;
    const vFrames = videoBufferRef.current.length;
    const gL = gloveBufferLeftRef.current.length;
    const gR = gloveBufferRightRef.current.length;
    return {
      duration,
      videoFrames: vFrames,
      gloveFramesLeft: gL,
      gloveFramesRight: gR,
      avgVideoFps: duration > 0 ? vFrames / duration : 0,
      avgGloveFpsLeft: duration > 0 ? gL / duration : 0,
      avgGloveFpsRight: duration > 0 ? gR / duration : 0,
    };
  }, [recordingDuration]);

  const clear = useCallback(() => {
    videoBufferRef.current = [];
    gloveBufferLeftRef.current = [];
    gloveBufferRightRef.current = [];
    setVideoFrameCount(0);
    setGloveFrameCountLeft(0);
    setGloveFrameCountRight(0);
    setRecordingDuration(0);
  }, []);

  return {
    isRecording,
    recordingDuration,
    videoFrameCount,
    gloveFrameCount: gloveFrameCountLeft + gloveFrameCountRight,
    gloveFrameCountLeft,
    gloveFrameCountRight,
    startRecording,
    stopRecording,
    recordVideoFrame,
    recordGloveFrame,
    exportSyncedCSV,
    exportRawJSON,
    getStats,
    clear,
  };
}
