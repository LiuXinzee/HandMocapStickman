/*
 * sequenceWindow — 实时滑窗推理用的环形缓冲
 *
 * 与 useSequenceRecorder 的区别：录制是"先攒完整段再重采样一次"，
 * 这里是"永远只保留最近几秒，每 100ms 切一段出来推理"，所以：
 *   - 缓冲有上限，超时的帧要丢掉，否则长时间开着翻译会一直涨内存
 *   - 只做触觉（部署用的是纯触觉学生模型），不碰视觉
 *
 * 切片时重采样到与录制时相同的 50Hz 栅格。虽然 buildSequenceFeatures 之后
 * 还会再重采样到 T=32，但中间这一步不能省：左右手是两个独立 COM 口，
 * 各自的时间戳对不齐，必须先落到公共栅格上，两只手的第 t 帧才是"同一时刻"。
 */
import type { GloveFrame } from "./gloveProtocol";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  type SequenceSample,
} from "./datasetStore";

interface WindowEntry {
  t: number;
  sensor: number[];
  quat: [number, number, number, number];
  acc: [number, number, number] | null;
  att: [number, number, number] | null;
}

export interface SequenceWindowOptions {
  /** 缓冲保留时长，需要 >= 推理窗口长度 */
  bufferMs?: number;
  /** 公共重采样栅格，必须与采集端一致 */
  gridFps?: number;
}

export class SequenceWindowBuffer {
  private left: WindowEntry[] = [];
  private right: WindowEntry[] = [];
  private readonly bufferMs: number;
  private readonly gridFps: number;

  constructor(options: SequenceWindowOptions = {}) {
    this.bufferMs = options.bufferMs ?? 3000;
    this.gridFps = options.gridFps ?? 50;
  }

  push(hand: "left" | "right", frame: GloveFrame): void {
    const buf = hand === "left" ? this.left : this.right;
    buf.push({
      t: frame.timestamp,
      sensor: frame.mapped_data,
      quat: frame.quaternion,
      acc: frame.acceleration,
      att: frame.attitude,
    });
    this.trim(buf, frame.timestamp);
  }

  clear(): void {
    this.left = [];
    this.right = [];
  }

  /** 缓冲里最新一帧的时间戳，两手取较大者；空缓冲返回 null */
  latestTime(): number | null {
    const lt = this.left.length ? this.left[this.left.length - 1].t : -Infinity;
    const rt = this.right.length
      ? this.right[this.right.length - 1].t
      : -Infinity;
    const m = Math.max(lt, rt);
    return Number.isFinite(m) ? m : null;
  }

  /**
   * 取最近 windowMs 的一段，重采样成 SequenceSample。
   * 返回 null 表示这一刻数据不足以推理（刚开始翻译、或手套刚掉线）。
   */
  snapshot(windowMs: number, label = "_live"): SequenceSample | null {
    const end = this.latestTime();
    if (end === null) return null;
    const start = end - windowMs;
    const dt = 1000 / this.gridFps;
    const T = Math.floor(windowMs / dt);
    if (T < 4) return null;

    // 缓冲还没攒够一整个窗口就别推理 —— 前半段会被最近邻拉成一条直线，
    // 那是个"静止"的假动作，模型多半会输出错词
    const covers = (buf: WindowEntry[]) => buf.length > 0 && buf[0].t <= start;
    const hasLeft = covers(this.left);
    const hasRight = covers(this.right);
    if (!hasLeft && !hasRight) return null;

    const timestamps = new Float32Array(T);
    const leftSensor = hasLeft ? new Uint8Array(T * SEQ_SENSOR_N) : null;
    const rightSensor = hasRight ? new Uint8Array(T * SEQ_SENSOR_N) : null;
    const leftImu = hasLeft ? new Float32Array(T * SEQ_IMU_N) : null;
    const rightImu = hasRight ? new Float32Array(T * SEQ_IMU_N) : null;

    let li = 0;
    let ri = 0;
    for (let t = 0; t < T; t++) {
      const target = start + t * dt;
      timestamps[t] = t * dt;
      if (leftSensor && leftImu) {
        const r = nearest(this.left, target, li);
        li = r.idx;
        write(r.entry, leftSensor, leftImu, t);
      }
      if (rightSensor && rightImu) {
        const r = nearest(this.right, target, ri);
        ri = r.idx;
        write(r.entry, rightSensor, rightImu, t);
      }
    }

    return {
      segments: [{ label, startFrame: 0, endFrame: T }],
      primaryLabel: label,
      frameCount: T,
      timestamps,
      leftSensor,
      rightSensor,
      leftImu,
      rightImu,
      leftLandmarks: null,
      rightLandmarks: null,
      durationMs: windowMs,
      sourceFps: this.gridFps,
      origin: "recorded",
      timestamp: Date.now(),
    };
  }

  private trim(buf: WindowEntry[], now: number): void {
    const cutoff = now - this.bufferMs;
    let drop = 0;
    while (drop < buf.length && buf[drop].t < cutoff) drop++;
    if (drop > 0) buf.splice(0, drop);
  }
}

function nearest(
  buf: WindowEntry[],
  target: number,
  startIdx: number
): { entry: WindowEntry | null; idx: number } {
  if (buf.length === 0) return { entry: null, idx: 0 };
  let i = Math.max(0, Math.min(startIdx, buf.length - 1));
  while (i + 1 < buf.length && buf[i + 1].t <= target) i++;
  let best = i;
  if (
    i + 1 < buf.length &&
    Math.abs(buf[i + 1].t - target) < Math.abs(buf[i].t - target)
  ) {
    best = i + 1;
  }
  return { entry: buf[best], idx: i };
}

function write(
  entry: WindowEntry | null,
  sensor: Uint8Array,
  imu: Float32Array,
  t: number
): void {
  const so = t * SEQ_SENSOR_N;
  const io = t * SEQ_IMU_N;
  if (!entry) {
    imu[io] = 1; // 单位四元数，避免零向量归一化时退化
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
