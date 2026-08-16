/*
 * sequenceFeatures — 时序样本的重采样、特征构建、增强与静态样本合成
 *
 * 与单帧模型（signLanguageModel.ts）的关键差异，以及每处这么做的理由：
 *
 * 1) 四元数相对化。绝对朝向不可信：IMU 的 yaw 有固定偏置，且每次戴手套的
 *    姿态都不一样。所以每条序列取相对于**首帧**的相对四元数
 *    q_rel_t = q_0⁻¹ ⊗ q_t，既消除偏置，又正好对齐动态词的信息本质
 *    ——信息在朝向的**变化**里。
 *
 * 2) 但相对化会丢掉"手心朝上还是朝下"这类绝对倾角信息，而这对区分手势有用。
 *    重力方向在世界系里是可观测的（加速度计能测），把它转到手系
 *    g_hand_t = R(q_t)⁻¹ · [0,0,1] 得到 3 维，天然对 yaw 不变（yaw 旋转
 *    不改变重力在手系的投影），所以不受 yaw 偏置影响。这比"保留首帧绝对
 *    四元数"更干净：逐帧可用，且不需要给模型加第二个输入。
 *
 * 3) 关键点归一化。signLanguageModel.ts 直接用原始 x/y/z，手在画面里的绝对
 *    位置会污染特征。但动态词的**手腕轨迹恰恰是关键信息**。所以拆开：
 *    手腕点(landmark 0) 保留绝对 3D 坐标（承载轨迹），其余 20 点相对手腕
 *    平移、再按手长(landmark 0→9 距离)归一（承载手型）。共 63D/手不变。
 *
 * 每帧特征布局（每只手 147 维，左手在前右手在后）：
 *   [  0..136] 137 个传感点 / 255
 *   [137..140] 相对首帧的四元数 (w,x,y,z)
 *   [141..143] 重力方向在手系的投影（单位向量）
 *   [144..146] 加速度 / ACC_SCALE，截断到 [-1,1]
 * 学生（触觉）帧维度 = 147 × 2 = 294
 * 教师（融合）再拼上每手 63 维归一化关键点 = 294 + 126 = 420
 */

import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
  type TrainingSample,
  type HandSample,
} from "./datasetStore";
import {
  DEFAULT_TRIM,
  detectSignSpan,
  spanToGridBounds,
  type TrimConfig,
  type TrimSpan,
} from "./sequenceTrim";

// ===== 维度常量 =====

/** 默认重采样后的定长帧数 */
export const SEQ_LEN = 32;

/** 每只手的触觉帧维度：137 传感 + 4 相对四元数 + 3 重力方向 + 3 加速度 */
export const HAND_TACTILE_FRAME_DIM = SEQ_SENSOR_N + 4 + 3 + 3; // 147
/** 每只手的视觉帧维度：21 关键点 × 3 */
export const HAND_VISUAL_FRAME_DIM = SEQ_LANDMARK_N; // 63

/** 学生（部署）每帧维度 */
export const TACTILE_FRAME_DIM = HAND_TACTILE_FRAME_DIM * 2; // 294
/** 教师（训练）每帧维度 */
export const FUSED_FRAME_DIM = TACTILE_FRAME_DIM + HAND_VISUAL_FRAME_DIM * 2; // 420

/**
 * 加速度归一化尺度。固件下发的是裸 float，规格书没写单位（可能是 g 也可能是
 * m/s²）。除以 16 再截断到 [-1,1] 对两种单位都落在合理量程内（g: ±16g；
 * m/s²: 重力 9.8 → 0.61），不需要知道真实单位。
 */
export const ACC_SCALE = 16;

/** 手长归一化的下限，防止关键点退化（手离镜头极远）时除以接近 0 的数 */
const MIN_HAND_LENGTH = 1e-3;

// ===== 四元数工具 =====

export type Quat = [number, number, number, number]; // [w, x, y, z]

export function quatMul(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return [
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ];
}

/** 单位四元数的逆 = 共轭 */
export function quatConj(q: Quat): Quat {
  return [q[0], -q[1], -q[2], -q[3]];
}

export function quatNormalize(q: Quat): Quat {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!isFinite(n) || n < 1e-8) return [1, 0, 0, 0];
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** 用四元数旋转向量：v' = q ⊗ v ⊗ q⁻¹ */
export function rotateVec(
  q: Quat,
  v: [number, number, number]
): [number, number, number] {
  const [w, x, y, z] = q;
  // t = 2 * (q_vec × v)
  const tx = 2 * (y * v[2] - z * v[1]);
  const ty = 2 * (z * v[0] - x * v[2]);
  const tz = 2 * (x * v[1] - y * v[0]);
  return [
    v[0] + w * tx + (y * tz - z * ty),
    v[1] + w * ty + (z * tx - x * tz),
    v[2] + w * tz + (x * ty - y * tx),
  ];
}

const WORLD_GRAVITY: [number, number, number] = [0, 0, 1];

/** 重力方向在手系中的投影。对世界系 yaw 旋转不变，故不受 IMU yaw 偏置影响 */
export function gravityInHandFrame(q: Quat): [number, number, number] {
  return rotateVec(quatConj(quatNormalize(q)), WORLD_GRAVITY);
}

/** 绕单位轴旋转 angle 弧度对应的四元数 */
export function quatFromAxisAngle(
  axis: [number, number, number],
  angle: number
): Quat {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const h = angle / 2;
  const s = Math.sin(h) / n;
  return [Math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s];
}

/** 四元数球面线性插值。重采样时必须用 slerp 而非逐分量线性插值 */
export function quatSlerp(a: Quat, b: Quat, t: number): Quat {
  const qa = quatNormalize(a);
  let qb = quatNormalize(b);
  let dot = qa[0] * qb[0] + qa[1] * qb[1] + qa[2] * qb[2] + qa[3] * qb[3];
  // q 与 -q 表示同一旋转，取短弧
  if (dot < 0) {
    qb = [-qb[0], -qb[1], -qb[2], -qb[3]];
    dot = -dot;
  }
  if (dot > 0.9995) {
    // 夹角极小，线性插值后归一化即可，避免 sin(θ)→0 的数值问题
    return quatNormalize([
      qa[0] + (qb[0] - qa[0]) * t,
      qa[1] + (qb[1] - qa[1]) * t,
      qa[2] + (qb[2] - qa[2]) * t,
      qa[3] + (qb[3] - qa[3]) * t,
    ]);
  }
  const theta = Math.acos(dot);
  const s = Math.sin(theta);
  const wa = Math.sin((1 - t) * theta) / s;
  const wb = Math.sin(t * theta) / s;
  return [
    qa[0] * wa + qb[0] * wb,
    qa[1] * wa + qb[1] * wb,
    qa[2] * wa + qb[2] * wb,
    qa[3] * wa + qb[3] * wb,
  ];
}

// ===== 重采样 =====

/** 重采样后的中间表示：定长 T 帧，各通道已按帧展开 */
export interface ResampledSequence {
  frameCount: number;
  leftSensor: Float32Array | null; // [T*137]
  rightSensor: Float32Array | null;
  leftImu: Float32Array | null; // [T*10]
  rightImu: Float32Array | null;
  leftLandmarks: Float32Array | null; // [T*63]，仍可能含 NaN
  rightLandmarks: Float32Array | null;
  /** 该手是否有任何有效视觉帧 */
  leftHasVision: boolean;
  rightHasVision: boolean;
}

/** 归一化的采样位置：给定源时间戳与目标位置 u∈[0,1]，返回下标与插值权重 */
function sampleIndex(
  timestamps: Float32Array,
  u: number
): { i0: number; i1: number; w: number } {
  const n = timestamps.length;
  if (n <= 1) return { i0: 0, i1: 0, w: 0 };
  const t0 = timestamps[0];
  const t1 = timestamps[n - 1];
  const target = t0 + (t1 - t0) * u;
  // 源时间戳单调递增，二分查找
  let lo = 0;
  let hi = n - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (timestamps[mid] <= target) lo = mid;
    else hi = mid;
  }
  const span = timestamps[hi] - timestamps[lo];
  const w = span > 1e-6 ? (target - timestamps[lo]) / span : 0;
  return { i0: lo, i1: hi, w: Math.max(0, Math.min(1, w)) };
}

function lerpBlock(
  src: ArrayLike<number>,
  width: number,
  i0: number,
  i1: number,
  w: number,
  dst: Float32Array,
  dstFrame: number
): void {
  const a = i0 * width;
  const b = i1 * width;
  const o = dstFrame * width;
  for (let c = 0; c < width; c++) {
    const va = src[a + c];
    const vb = src[b + c];
    // NaN 传播：任一端是 NaN 则整帧该通道标 NaN，交给后续插值处理
    dst[o + c] = va + (vb - va) * w;
  }
}

/**
 * IMU 块的插值：前 4 维是四元数走 slerp，后 6 维（加速度+姿态角）走线性。
 * 逐分量线性插值四元数在大角度时会产生非单位、方向错误的结果。
 */
function lerpImu(
  src: ArrayLike<number>,
  i0: number,
  i1: number,
  w: number,
  dst: Float32Array,
  dstFrame: number
): void {
  const a = i0 * SEQ_IMU_N;
  const b = i1 * SEQ_IMU_N;
  const o = dstFrame * SEQ_IMU_N;
  const q = quatSlerp(
    [src[a], src[a + 1], src[a + 2], src[a + 3]],
    [src[b], src[b + 1], src[b + 2], src[b + 3]],
    w
  );
  dst[o] = q[0];
  dst[o + 1] = q[1];
  dst[o + 2] = q[2];
  dst[o + 3] = q[3];
  for (let c = 4; c < SEQ_IMU_N; c++) {
    const va = src[a + c];
    const vb = src[b + c];
    dst[o + c] = va + (vb - va) * w;
  }
}

/** 目标帧在 [0,1] 上的采样位置序列（可被增强改写） */
export type TimeGrid = Float32Array;

/** 均匀栅格 */
export function uniformGrid(T: number): TimeGrid {
  const g = new Float32Array(T);
  for (let i = 0; i < T; i++) g[i] = T === 1 ? 0 : i / (T - 1);
  return g;
}

/**
 * 把变长序列重采样到定长 T 帧。
 * grid 给出每个目标帧在源序列归一化时间 [0,1] 上的位置，默认均匀。
 */
export function resampleSequence(
  sample: SequenceSample,
  T: number = SEQ_LEN,
  grid: TimeGrid = uniformGrid(T)
): ResampledSequence {
  const ts = sample.timestamps;
  const out: ResampledSequence = {
    frameCount: T,
    leftSensor: sample.leftSensor ? new Float32Array(T * SEQ_SENSOR_N) : null,
    rightSensor: sample.rightSensor ? new Float32Array(T * SEQ_SENSOR_N) : null,
    leftImu: sample.leftImu ? new Float32Array(T * SEQ_IMU_N) : null,
    rightImu: sample.rightImu ? new Float32Array(T * SEQ_IMU_N) : null,
    leftLandmarks: sample.leftLandmarks
      ? new Float32Array(T * SEQ_LANDMARK_N)
      : null,
    rightLandmarks: sample.rightLandmarks
      ? new Float32Array(T * SEQ_LANDMARK_N)
      : null,
    leftHasVision: false,
    rightHasVision: false,
  };

  for (let t = 0; t < T; t++) {
    const { i0, i1, w } = sampleIndex(ts, grid[t]);
    if (sample.leftSensor && out.leftSensor)
      lerpBlock(sample.leftSensor, SEQ_SENSOR_N, i0, i1, w, out.leftSensor, t);
    if (sample.rightSensor && out.rightSensor)
      lerpBlock(sample.rightSensor, SEQ_SENSOR_N, i0, i1, w, out.rightSensor, t);
    if (sample.leftImu && out.leftImu)
      lerpImu(sample.leftImu, i0, i1, w, out.leftImu, t);
    if (sample.rightImu && out.rightImu)
      lerpImu(sample.rightImu, i0, i1, w, out.rightImu, t);
    // 视觉：不跨 NaN 插值，取最近邻，NaN 保留给后续 fillVisionGaps 处理
    if (sample.leftLandmarks && out.leftLandmarks)
      copyNearestVision(sample.leftLandmarks, i0, i1, w, out.leftLandmarks, t);
    if (sample.rightLandmarks && out.rightLandmarks)
      copyNearestVision(sample.rightLandmarks, i0, i1, w, out.rightLandmarks, t);
  }

  if (out.leftLandmarks) out.leftHasVision = fillVisionGaps(out.leftLandmarks);
  if (out.rightLandmarks)
    out.rightHasVision = fillVisionGaps(out.rightLandmarks);

  return out;
}

/**
 * 视觉重采样取"最近邻且该帧有效"，而不是线性插值。
 * 线性插值会让一个有效帧和一个 NaN 帧混出整帧 NaN，白白丢掉一半有效数据。
 */
function copyNearestVision(
  src: ArrayLike<number>,
  i0: number,
  i1: number,
  w: number,
  dst: Float32Array,
  dstFrame: number
): void {
  const a0 = i0 * SEQ_LANDMARK_N;
  const a1 = i1 * SEQ_LANDMARK_N;
  const v0 = isFinite(src[a0]);
  const v1 = isFinite(src[a1]);
  const o = dstFrame * SEQ_LANDMARK_N;

  if (v0 && v1) {
    for (let c = 0; c < SEQ_LANDMARK_N; c++) {
      const va = src[a0 + c];
      const vb = src[a1 + c];
      dst[o + c] = va + (vb - va) * w;
    }
    return;
  }
  const src0 = v0 ? a0 : v1 ? a1 : -1;
  if (src0 < 0) {
    dst.fill(NaN, o, o + SEQ_LANDMARK_N);
    return;
  }
  for (let c = 0; c < SEQ_LANDMARK_N; c++) dst[o + c] = src[src0 + c];
}

/**
 * 沿时间轴补齐视觉缺帧：中间空洞线性插值，首尾空洞用最近有效帧外推（hold）。
 * 返回该手是否存在任何有效帧；全程无视觉时数组置 0，由调用方置 visMask=0。
 */
export function fillVisionGaps(lm: Float32Array): boolean {
  const T = lm.length / SEQ_LANDMARK_N;
  const valid: number[] = [];
  for (let t = 0; t < T; t++) {
    if (isFinite(lm[t * SEQ_LANDMARK_N])) valid.push(t);
  }
  if (valid.length === 0) {
    lm.fill(0);
    return false;
  }
  let vi = 0;
  for (let t = 0; t < T; t++) {
    if (isFinite(lm[t * SEQ_LANDMARK_N])) continue;
    while (vi < valid.length - 1 && valid[vi + 1] < t) vi++;
    const prev = valid[vi] < t ? valid[vi] : -1;
    const next = valid.find((v) => v > t) ?? -1;
    const o = t * SEQ_LANDMARK_N;
    if (prev >= 0 && next >= 0) {
      const w = (t - prev) / (next - prev);
      const a = prev * SEQ_LANDMARK_N;
      const b = next * SEQ_LANDMARK_N;
      for (let c = 0; c < SEQ_LANDMARK_N; c++) {
        lm[o + c] = lm[a + c] + (lm[b + c] - lm[a + c]) * w;
      }
    } else {
      const s = (prev >= 0 ? prev : next) * SEQ_LANDMARK_N;
      for (let c = 0; c < SEQ_LANDMARK_N; c++) lm[o + c] = lm[s + c];
    }
  }
  return true;
}

// ===== 关键点归一化 =====

/**
 * 单帧 21 点归一化，原地写入 dst。
 * 手腕(0)保留绝对坐标承载轨迹；其余 20 点相对手腕平移并按手长(0→9)缩放承载手型。
 */
export function normalizeLandmarkFrame(
  src: ArrayLike<number>,
  srcOffset: number,
  dst: Float32Array,
  dstOffset: number
): void {
  const wx = src[srcOffset];
  const wy = src[srcOffset + 1];
  const wz = src[srcOffset + 2];
  const mx = src[srcOffset + 9 * 3];
  const my = src[srcOffset + 9 * 3 + 1];
  const mz = src[srcOffset + 9 * 3 + 2];
  const scale = Math.max(Math.hypot(mx - wx, my - wy, mz - wz), MIN_HAND_LENGTH);

  dst[dstOffset] = wx;
  dst[dstOffset + 1] = wy;
  dst[dstOffset + 2] = wz;
  for (let p = 1; p < 21; p++) {
    const s = srcOffset + p * 3;
    const d = dstOffset + p * 3;
    dst[d] = (src[s] - wx) / scale;
    dst[d + 1] = (src[s + 1] - wy) / scale;
    dst[d + 2] = (src[s + 2] - wz) / scale;
  }
}

// ===== 数据增强 =====

export interface SeqAugmentConfig {
  /** 时间扭曲强度 0~1：0 关闭。控制点最大偏移比例 */
  timeWarp: number;
  /** 随机裁剪保留比例的下限（1 = 不裁剪） */
  cropMin: number;
  /** 传感器整体幅度缩放范围半宽，如 0.1 → [0.9, 1.1]（规格书标称 ±8%） */
  amplitudeScale: number;
  /** 手套佩戴偏差角上限（度）：对四元数做右乘小旋转 */
  mountJitterDeg: number;
  /** 传感器高斯噪声标准差（归一化后单位，即 /255 之后） */
  noise: number;
}

export const DEFAULT_AUGMENT: SeqAugmentConfig = {
  timeWarp: 0.15,
  cropMin: 0.85,
  amplitudeScale: 0.1,
  mountJitterDeg: 5,
  noise: 0.01,
};

export const NO_AUGMENT: SeqAugmentConfig = {
  timeWarp: 0,
  cropMin: 1,
  amplitudeScale: 0,
  mountJitterDeg: 0,
  noise: 0,
};

export type Rng = () => number;

/**
 * 构造一条增强用的时间栅格：随机裁剪一个子区间 + 单调非线性扭曲。
 *
 * 输出帧数固定为 T，所以"打得慢"只能通过裁剪源区间来模拟；
 * 非线性扭曲则模拟一个动作内部前后段快慢不均。
 */
export function makeAugmentedGrid(
  T: number,
  cfg: SeqAugmentConfig,
  rng: Rng = Math.random
): TimeGrid {
  const ratio = cfg.cropMin >= 1 ? 1 : cfg.cropMin + (1 - cfg.cropMin) * rng();
  const start = (1 - ratio) * rng();
  const end = start + ratio;

  // 3 个内部控制点做单调分段线性扭曲
  const K = 3;
  const knots = [0];
  for (let k = 1; k <= K; k++) {
    const base = k / (K + 1);
    const jitter = cfg.timeWarp * (rng() * 2 - 1) * (1 / (K + 1));
    knots.push(base + jitter);
  }
  knots.push(1);
  knots.sort((a, b) => a - b); // 保证单调（扭曲不能让时间倒流）

  const grid = new Float32Array(T);
  for (let i = 0; i < T; i++) {
    const u = T === 1 ? 0 : i / (T - 1);
    const seg = Math.min(Math.floor(u * (K + 1)), K);
    const local = u * (K + 1) - seg;
    const warped = knots[seg] + (knots[seg + 1] - knots[seg]) * local;
    grid[i] = start + (end - start) * Math.max(0, Math.min(1, warped));
  }
  return grid;
}

// ===== 特征构建 =====

export interface BuildFeatureOptions {
  seqLen?: number;
  /** true 时输出教师的 420 维/帧，false 时输出学生的 294 维/帧 */
  includeVision?: boolean;
  augment?: SeqAugmentConfig;
  rng?: Rng;
  /**
   * 起手段裁剪配置。**缺省即启用**（`DEFAULT_TRIM`）——录制里"手从腿上抬进画面"
   * 那段不该被当成手势学，见 `sequenceTrim.ts` 顶部注释。
   * 传 `null` 显式关掉：推理路径（`predictSequence`）就是这么做的。
   */
  trim?: TrimConfig | null;
}

/** 构建结果：[T * frameDim] 的扁平特征 + 元信息 */
export interface BuiltFeature {
  data: Float32Array;
  seqLen: number;
  frameDim: number;
  /** 每只手是否有有效视觉（供 UI 统计覆盖率；特征里缺失手已置 0） */
  leftHasVision: boolean;
  rightHasVision: boolean;
  /** 起手段裁剪结果；关掉裁剪时为 null。`applied=false` 说明判据没成立、这条没裁 */
  trim: TrimSpan | null;
}

function writeHandTactile(
  dst: Float32Array,
  base: number,
  sensor: Float32Array | null,
  imu: Float32Array | null,
  t: number,
  firstQuatInv: Quat | null,
  mountJitter: Quat | null,
  ampScale: number,
  noise: number,
  rng: Rng
): void {
  if (!sensor && !imu) return; // 缺手：保持全 0

  if (sensor) {
    const s = t * SEQ_SENSOR_N;
    for (let c = 0; c < SEQ_SENSOR_N; c++) {
      let v = (sensor[s + c] / 255) * ampScale;
      if (noise > 0) v += gaussian(rng) * noise;
      dst[base + c] = v;
    }
  }

  if (imu) {
    const o = t * SEQ_IMU_N;
    let q: Quat = quatNormalize([imu[o], imu[o + 1], imu[o + 2], imu[o + 3]]);
    // 佩戴偏差 = 传感器坐标系相对手的小失配，数学上是右乘
    if (mountJitter) q = quatNormalize(quatMul(q, mountJitter));

    const rel = firstQuatInv ? quatMul(firstQuatInv, q) : q;
    dst[base + SEQ_SENSOR_N] = rel[0];
    dst[base + SEQ_SENSOR_N + 1] = rel[1];
    dst[base + SEQ_SENSOR_N + 2] = rel[2];
    dst[base + SEQ_SENSOR_N + 3] = rel[3];

    const g = gravityInHandFrame(q);
    dst[base + SEQ_SENSOR_N + 4] = g[0];
    dst[base + SEQ_SENSOR_N + 5] = g[1];
    dst[base + SEQ_SENSOR_N + 6] = g[2];

    for (let c = 0; c < 3; c++) {
      const a = imu[o + 4 + c] / ACC_SCALE;
      dst[base + SEQ_SENSOR_N + 7 + c] = Math.max(-1, Math.min(1, a));
    }
  }
}

/** Box-Muller，一次只取一个（够用且无状态） */
function gaussian(rng: Rng): number {
  const u = Math.max(rng(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rng());
}

function firstValidQuatInv(imu: Float32Array | null): Quat | null {
  if (!imu || imu.length < 4) return null;
  return quatConj(quatNormalize([imu[0], imu[1], imu[2], imu[3]]));
}

function makeMountJitter(deg: number, rng: Rng): Quat | null {
  if (deg <= 0) return null;
  const axis: [number, number, number] = [
    rng() * 2 - 1,
    rng() * 2 - 1,
    rng() * 2 - 1,
  ];
  const angle = ((rng() * 2 - 1) * deg * Math.PI) / 180;
  return quatFromAxisAngle(axis, angle);
}

/**
 * 把一条 SequenceSample 转成模型输入特征。
 *
 * 注意增强的作用位置：时间扭曲作用在重采样栅格上（必须在重采样时施加），
 * 幅度缩放/噪声/佩戴偏差作用在特征写入时。两者不能互换顺序——先重采样
 * 再时间扭曲会二次插值、丢高频。
 */
export function buildSequenceFeatures(
  sample: SequenceSample,
  opts: BuildFeatureOptions = {}
): BuiltFeature {
  const T = opts.seqLen ?? SEQ_LEN;
  const includeVision = opts.includeVision ?? false;
  const aug = opts.augment ?? NO_AUGMENT;
  const rng = opts.rng ?? Math.random;

  const frameDim = includeVision ? FUSED_FRAME_DIM : TACTILE_FRAME_DIM;
  const grid =
    aug.timeWarp > 0 || aug.cropMin < 1
      ? makeAugmentedGrid(T, aug, rng)
      : uniformGrid(T);

  // 起手段裁剪：把栅格从整条 [0,1] 压到"手在画面里"的那一段 [a,b]。
  // **顺序很重要** —— 增强的随机裁剪(cropMin)先在 [0,1] 上取好子区间，再整体映射进
  // [a,b]，于是随机裁剪是在**手势内部**取子段，而不是有一定概率恰好裁掉起手段。
  // 反过来先压再增强会让两层裁剪相乘，短样本被裁到只剩几帧。
  const trimCfg = opts.trim === undefined ? DEFAULT_TRIM : opts.trim;
  const trim = trimCfg ? detectSignSpan(sample, trimCfg) : null;
  if (trim?.applied) {
    const { a, b } = spanToGridBounds(sample, trim);
    for (let i = 0; i < T; i++) grid[i] = a + (b - a) * grid[i];
  }

  const rs = resampleSequence(sample, T, grid);
  const data = new Float32Array(T * frameDim);

  const leftQInv = firstValidQuatInv(rs.leftImu);
  const rightQInv = firstValidQuatInv(rs.rightImu);
  const leftJitter = makeMountJitter(aug.mountJitterDeg, rng);
  const rightJitter = makeMountJitter(aug.mountJitterDeg, rng);
  // 幅度缩放对整条序列是同一个系数（模拟佩戴松紧），不是逐帧随机
  const leftAmp = 1 + (rng() * 2 - 1) * aug.amplitudeScale;
  const rightAmp = 1 + (rng() * 2 - 1) * aug.amplitudeScale;

  for (let t = 0; t < T; t++) {
    const f = t * frameDim;
    writeHandTactile(
      data,
      f,
      rs.leftSensor,
      rs.leftImu,
      t,
      leftQInv,
      leftJitter,
      leftAmp,
      aug.noise,
      rng
    );
    writeHandTactile(
      data,
      f + HAND_TACTILE_FRAME_DIM,
      rs.rightSensor,
      rs.rightImu,
      t,
      rightQInv,
      rightJitter,
      rightAmp,
      aug.noise,
      rng
    );
    if (includeVision) {
      const vBase = f + TACTILE_FRAME_DIM;
      if (rs.leftLandmarks && rs.leftHasVision) {
        normalizeLandmarkFrame(
          rs.leftLandmarks,
          t * SEQ_LANDMARK_N,
          data,
          vBase
        );
      }
      if (rs.rightLandmarks && rs.rightHasVision) {
        normalizeLandmarkFrame(
          rs.rightLandmarks,
          t * SEQ_LANDMARK_N,
          data,
          vBase + HAND_VISUAL_FRAME_DIM
        );
      }
    }
  }

  return {
    data,
    seqLen: T,
    frameDim,
    leftHasVision: rs.leftHasVision,
    rightHasVision: rs.rightHasVision,
    trim,
  };
}

// ===== 运动能量（采集页用来确认真的录到了动作） =====

/**
 * 逐帧运动能量 = 相邻帧传感值的平均绝对差（归一化到 0-1 尺度）。
 * 采集页画成曲线：一条静止误录的样本会是一条贴地直线，肉眼立刻能发现。
 */
export function motionEnergy(sample: SequenceSample): Float32Array {
  const T = sample.frameCount;
  const out = new Float32Array(T);
  if (T < 2) return out;
  const hands = [sample.leftSensor, sample.rightSensor].filter(
    (h): h is Uint8Array => h !== null
  );
  if (hands.length === 0) return out;

  for (let t = 1; t < T; t++) {
    let sum = 0;
    for (const h of hands) {
      const a = (t - 1) * SEQ_SENSOR_N;
      const b = t * SEQ_SENSOR_N;
      for (let c = 0; c < SEQ_SENSOR_N; c++) sum += Math.abs(h[b + c] - h[a + c]);
    }
    out[t] = sum / (hands.length * SEQ_SENSOR_N * 255);
  }
  out[0] = out[1];
  return out;
}

/** 视觉覆盖率：有有效关键点的帧占比（两只手分别算，取有数据的那些手的均值） */
export function visionCoverage(sample: SequenceSample): number {
  const T = sample.frameCount;
  if (T === 0) return 0;
  const arrays = [sample.leftLandmarks, sample.rightLandmarks].filter(
    (a): a is Float32Array => a !== null
  );
  if (arrays.length === 0) return 0;
  let valid = 0;
  for (const lm of arrays) {
    for (let t = 0; t < T; t++) if (isFinite(lm[t * SEQ_LANDMARK_N])) valid++;
  }
  return valid / (arrays.length * T);
}

// ===== 静态单帧样本 → 合成序列 =====

export interface SynthesizeConfig {
  /** 合成序列的帧数 */
  frameCount: number;
  /** 栅格频率，与真实录制一致 */
  fps: number;
  /** 传感值抖动幅度（0-255 尺度） */
  sensorJitter: number;
  /** 四元数摆动幅度（度） */
  quatWobbleDeg: number;
  /** AR(1) 平滑系数，越大越平滑。必须接近 1，见下方说明 */
  smoothing: number;
}

export const DEFAULT_SYNTHESIZE: SynthesizeConfig = {
  frameCount: 40,
  fps: 50,
  sensorJitter: 3,
  quatWobbleDeg: 2,
  smoothing: 0.9,
};

/**
 * 低通相关随机游走：x_t = a·x_{t-1} + (1-a)·ε_t，输出方差归一化到 1。
 *
 * **必须是相关噪声而不是逐帧独立白噪声**。白噪声的一阶差分自相关是 -0.5，
 * 而真实保持不动的手其差分自相关接近 0；模型会立刻学到"帧间差分像白噪声
 * ⇒ 静态词"这个捷径，训练指标很好看，真实推理时直接崩。
 */
function correlatedWalk(n: number, a: number, rng: Rng): Float32Array {
  const out = new Float32Array(n);
  let x = 0;
  // AR(1) 稳态标准差是 (1-a)/sqrt(1-a²)，除掉它让输出方差为 1
  const norm = Math.sqrt(1 - a * a) / (1 - a);
  for (let i = 0; i < n; i++) {
    x = a * x + (1 - a) * gaussian(rng);
    out[i] = x * norm;
  }
  return out;
}

function handToColumns(
  hand: HandSample | null,
  cfg: SynthesizeConfig,
  rng: Rng
): {
  sensor: Uint8Array | null;
  imu: Float32Array | null;
  landmarks: Float32Array | null;
} {
  if (!hand) return { sensor: null, imu: null, landmarks: null };
  const T = cfg.frameCount;

  const sensor = new Uint8Array(T * SEQ_SENSOR_N);
  for (let c = 0; c < SEQ_SENSOR_N; c++) {
    const base = hand.sensor_data[c] ?? 0;
    const walk = correlatedWalk(T, cfg.smoothing, rng);
    for (let t = 0; t < T; t++) {
      const v = Math.round(base + walk[t] * cfg.sensorJitter);
      sensor[t * SEQ_SENSOR_N + c] = Math.max(0, Math.min(255, v));
    }
  }

  const imu = new Float32Array(T * SEQ_IMU_N);
  const q0 = quatNormalize(hand.quaternion as Quat);
  const axis: [number, number, number] = [
    rng() * 2 - 1,
    rng() * 2 - 1,
    rng() * 2 - 1,
  ];
  const wobble = correlatedWalk(T, cfg.smoothing, rng);
  const maxRad = (cfg.quatWobbleDeg * Math.PI) / 180;
  for (let t = 0; t < T; t++) {
    const q = quatNormalize(
      quatMul(q0, quatFromAxisAngle(axis, wobble[t] * maxRad))
    );
    const o = t * SEQ_IMU_N;
    imu[o] = q[0];
    imu[o + 1] = q[1];
    imu[o + 2] = q[2];
    imu[o + 3] = q[3];
    // 静态样本没有加速度/姿态角记录，留 0
  }

  let landmarks: Float32Array | null = null;
  if (hand.landmarks?.length === 21) {
    landmarks = new Float32Array(T * SEQ_LANDMARK_N);
    const jitters: Float32Array[] = [];
    for (let c = 0; c < SEQ_LANDMARK_N; c++) {
      jitters.push(correlatedWalk(T, cfg.smoothing, rng));
    }
    for (let t = 0; t < T; t++) {
      for (let p = 0; p < 21; p++) {
        const lp = hand.landmarks[p];
        const o = t * SEQ_LANDMARK_N + p * 3;
        // 关键点在归一化坐标系里，抖动幅度取 0.002 量级
        landmarks[o] = lp.x + jitters[p * 3][t] * 0.002;
        landmarks[o + 1] = lp.y + jitters[p * 3 + 1][t] * 0.002;
        landmarks[o + 2] = lp.z + jitters[p * 3 + 2][t] * 0.002;
      }
    }
  }

  return { sensor, imu, landmarks };
}

/**
 * 把旧的静态单帧样本扩成序列样本，用于迁移历史数据、立刻跑通整条链路。
 *
 * 已知局限：合成序列的帧间统计与真实"保持不动"的序列仍不完全一致。
 * 缓解手段是低通相关噪声（见 correlatedWalk）+ 后续用真实录制的静态词
 * 序列做校准替换（deleteSynthesizedSequences 就是给这一步用的）。
 */
export function synthesizeFromStatic(
  sample: TrainingSample,
  cfg: SynthesizeConfig = DEFAULT_SYNTHESIZE,
  rng: Rng = Math.random
): SequenceSample {
  const T = cfg.frameCount;
  const dt = 1000 / cfg.fps;
  const timestamps = new Float32Array(T);
  for (let t = 0; t < T; t++) timestamps[t] = t * dt;

  const left = handToColumns(sample.left, cfg, rng);
  const right = handToColumns(sample.right, cfg, rng);

  return {
    segments: [{ label: sample.label, startFrame: 0, endFrame: T }],
    primaryLabel: sample.label,
    frameCount: T,
    timestamps,
    leftSensor: left.sensor,
    rightSensor: right.sensor,
    leftImu: left.imu,
    rightImu: right.imu,
    leftLandmarks: left.landmarks,
    rightLandmarks: right.landmarks,
    durationMs: T * dt,
    sourceFps: cfg.fps,
    origin: "synthesized",
    timestamp: Date.now(),
  };
}
