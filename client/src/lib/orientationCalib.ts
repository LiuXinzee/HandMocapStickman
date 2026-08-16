/**
 * orientationCalib — 手套朝向标定（陀螺零位 + 轴向重映射）
 *
 * 移植自 glove_visual/cc_part2 的 glove-protocol.ts（2026-08-07 版逻辑），
 * 解决的是"我戴着手套摆的朝向和屏幕上手模的朝向对不上"。分两层：
 *
 *  1. **零位** `reference`：把某个约定姿态（竖立、手心朝屏幕）记为单位旋转，
 *     之后所有帧都用 `ref⁻¹ ⊗ q` 表示"相对那个姿态转了多少"。这一层修的是
 *     "整体差一个固定旋转"（IMU 装配朝向 + 戴法 + 六轴没有绝对 yaw 零点）。
 *
 *  2. **轴向映射** `axisMap`：3×3 矩阵，把旋转轴从传感器机体系换算到模型坐标系。
 *     这一层修的是"我抬手，手模却在左右倾"——即两个坐标系的轴不是简单对应。
 *     矩阵由两个实测动作反解：竖立→平铺 得俯仰轴，竖立→手心相对 得偏摆轴。
 *
 * **不要在解析层"修方向"**（gloveProtocol.ts 的四元数原样归一化，别动）。
 * cc_part2 在这件事上踩过坑：零位是相对旋转，依赖原始四元数；
 * 解析层任何"方向修正"都会把标定后的正确行为破坏掉。
 *
 * 四元数一律 **[w, x, y, z]**（协议原序）。传给 three.js 时才换成 (x,y,z,w)。
 */

import type { HandKey } from "./bendRange";

/** [w, x, y, z] */
export type Quat = [number, number, number, number];
export type Vec3 = [number, number, number];

export interface AxisQuality {
  /** 竖立→平铺 实测转了多少度（期望 ≈90） */
  pitchDeg: number;
  /** 竖立→手心相对 实测转了多少度（期望 ≈90） */
  swingDeg: number;
  /** 两个动作的转轴夹角（期望 ≈90，太小说明两个动作没分开） */
  separationDeg: number;
}

export interface OrientationCalib {
  /** 竖立、手心朝屏幕 的姿态四元数 = 零位 */
  reference: Quat;
  /** 平铺（手心朝上）参考，仅用于反解俯仰轴与事后核对 */
  flat?: Quat;
  /** 手心相对 参考，仅用于反解偏摆轴与事后核对 */
  palmsIn?: Quat;
  /** 3×3 轴向重映射；缺省表示只做了零位（轴向识别没通过门限） */
  axisMap?: number[][];
  axisQuality?: AxisQuality;
}

/** 轴向识别门限：两个动作各自至少转这么多度，且两轴至少分开这么多度 */
export const MIN_MOTION_DEG = 15;
export const MIN_SEPARATION_DEG = 25;

const STORE_KEY_PREFIX = "deafkit_orient_calib_v1_";

// ===== 四元数 / 向量基础 =====

export function normalizeQuat(q: Quat): Quat {
  const m = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / m, q[1] / m, q[2] / m, q[3] / m];
}

export function multiplyQuat(a: Quat, b: Quat): Quat {
  const [aw, ax, ay, az] = a;
  const [bw, bx, by, bz] = b;
  return normalizeQuat([
    aw * bw - ax * bx - ay * by - az * bz,
    aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
  ]);
}

export function conjugateQuat(q: Quat): Quat {
  return [q[0], -q[1], -q[2], -q[3]];
}

/**
 * 多帧四元数平均：先与首帧符号对齐再求和归一化。
 * 符号对齐是必须的 —— q 与 −q 表示同一个旋转，直接相加会互相抵消。
 */
export function averageQuaternions(quats: Quat[]): Quat | null {
  if (!quats.length) return null;
  const ref = quats[0];
  const sum: Quat = [0, 0, 0, 0];
  for (const q of quats) {
    const dot = q[0] * ref[0] + q[1] * ref[1] + q[2] * ref[2] + q[3] * ref[3];
    const sign = dot < 0 ? -1 : 1;
    for (let i = 0; i < 4; i++) sum[i] += q[i] * sign;
  }
  if (!Math.hypot(sum[0], sum[1], sum[2], sum[3])) return null;
  return normalizeQuat(sum);
}

const dot3 = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

const cross3 = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

const normalize3 = (a: Vec3): Vec3 | null => {
  const m = Math.hypot(a[0], a[1], a[2]);
  return m < 1e-8 ? null : [a[0] / m, a[1] / m, a[2] / m];
};

// ===== 轴向识别 =====

/**
 * 两个标定动作在**模型坐标系**里期望的转轴（以"竖立、手心朝屏幕"为零位）：
 *   俯仰 = 竖立→手心朝上平铺，绕 −X 转 90°
 *   偏摆 = 竖立→双手手心相对，左手绕 +Y、右手绕 −Y（动作本身镜像对称）
 */
export const MODEL_MOTION_AXES: Record<HandKey, { pitch: Vec3; swing: Vec3 }> = {
  LH: { pitch: [-1, 0, 0], swing: [0, 1, 0] },
  RH: { pitch: [-1, 0, 0], swing: [0, -1, 0] },
};

/** 参考姿态→目标姿态 的相对旋转，拆成转轴（在参考姿态机体系里）与转角 */
export function relativeAxisAngle(
  reference: Quat,
  target: Quat
): { axis: Vec3; angleDeg: number } | null {
  let rel = multiplyQuat(conjugateQuat(reference), target);
  // 取 w ≥ 0 的那一半，保证转角落在 0~180°
  if (rel[0] < 0) rel = [-rel[0], -rel[1], -rel[2], -rel[3]];
  const m = Math.hypot(rel[1], rel[2], rel[3]);
  if (m < 1e-6) return null; // 两个姿态几乎相同，转轴无定义
  return {
    axis: [rel[1] / m, rel[2] / m, rel[3] / m],
    angleDeg: (2 * Math.atan2(m, rel[0]) * 180) / Math.PI,
  };
}

/** 两转轴夹角，取 0~90°（轴的正负号无意义，所以取 |dot|） */
export function axisSeparationDeg(a: Vec3, b: Vec3): number {
  return (Math.acos(Math.min(1, Math.abs(dot3(a, b)))) * 180) / Math.PI;
}

/**
 * 由两组「传感器轴 → 模型轴」对应关系构建 3×3 重映射矩阵。
 * 两个实测轴不会严格正交（人手做不到），所以先 Gram-Schmidt 正交化补出第三轴，
 * 再用外积把两组正交基拼成旋转矩阵 Σ mᵢ sᵢᵀ。
 */
export function buildAxisMap(
  sensorPitch: Vec3,
  sensorSwing: Vec3,
  modelPitch: Vec3,
  modelSwing: Vec3
): number[][] | null {
  const s1 = normalize3(sensorPitch);
  const m1 = normalize3(modelPitch);
  if (!s1 || !m1) return null;
  const s2 = normalize3([
    sensorSwing[0] - dot3(sensorSwing, s1) * s1[0],
    sensorSwing[1] - dot3(sensorSwing, s1) * s1[1],
    sensorSwing[2] - dot3(sensorSwing, s1) * s1[2],
  ]);
  const m2 = normalize3([
    modelSwing[0] - dot3(modelSwing, m1) * m1[0],
    modelSwing[1] - dot3(modelSwing, m1) * m1[1],
    modelSwing[2] - dot3(modelSwing, m1) * m1[2],
  ]);
  if (!s2 || !m2) return null;
  const s3 = cross3(s1, s2);
  const m3 = cross3(m1, m2);
  const map = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const pairs: [Vec3, Vec3][] = [
    [m1, s1],
    [m2, s2],
    [m3, s3],
  ];
  for (const [m, s] of pairs) {
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) map[row][col] += m[row] * s[col];
    }
  }
  return map;
}

/**
 * 从向导四步采到的三个姿态四元数组装一份标定。
 * 零位（竖立）是必需的；平铺与手心相对齐了才尝试轴向识别，
 * 没过门限就只留零位 —— 宁可少修一层，也不要用一个乱解出来的矩阵把朝向搅得更差。
 */
export function assembleOrientationCalib(
  handKey: HandKey,
  zero: Quat | null,
  flat: Quat | null,
  palmsIn: Quat | null
): OrientationCalib | null {
  if (!zero) return null;
  const calib: OrientationCalib = { reference: zero };
  if (flat) calib.flat = flat;
  if (palmsIn) calib.palmsIn = palmsIn;
  if (!flat || !palmsIn) return calib;

  const pitchMotion = relativeAxisAngle(zero, flat);
  const swingMotion = relativeAxisAngle(zero, palmsIn);
  if (!pitchMotion || !swingMotion) return calib;
  const separationDeg = axisSeparationDeg(pitchMotion.axis, swingMotion.axis);
  const quality: AxisQuality = {
    pitchDeg: pitchMotion.angleDeg,
    swingDeg: swingMotion.angleDeg,
    separationDeg,
  };
  // 质量数一律留着显示：轴向没识别成功时，用户要能看出是哪一步做得不到位
  calib.axisQuality = quality;
  if (
    pitchMotion.angleDeg < MIN_MOTION_DEG ||
    swingMotion.angleDeg < MIN_MOTION_DEG ||
    separationDeg < MIN_SEPARATION_DEG
  ) {
    return calib;
  }
  const axes = MODEL_MOTION_AXES[handKey];
  const map = buildAxisMap(
    pitchMotion.axis,
    swingMotion.axis,
    axes.pitch,
    axes.swing
  );
  if (map) calib.axisMap = map;
  return calib;
}

/** 轴向没识别成功的原因（axisMap 缺失时用来提示用户重做哪一步） */
export function axisMapFailReason(calib: OrientationCalib): string | null {
  if (calib.axisMap) return null;
  const q = calib.axisQuality;
  if (!q) return "只采到零位，缺平铺或手心相对";
  if (q.pitchDeg < MIN_MOTION_DEG)
    return `平铺那步只转了 ${q.pitchDeg.toFixed(0)}°，前臂要真的从竖立放平到水平`;
  if (q.swingDeg < MIN_MOTION_DEG)
    return `手心相对那步只转了 ${q.swingDeg.toFixed(0)}°，手腕要真的翻到手心朝内`;
  if (q.separationDeg < MIN_SEPARATION_DEG)
    return `两个动作的转轴只差 ${q.separationDeg.toFixed(0)}°，说明做成了同一个方向`;
  return "轴向矩阵反解失败";
}

// ===== 应用 =====

/**
 * 把原始 IMU 四元数换算成模型该用的姿态：先减零位，再（若有）重映射旋转轴。
 * 重映射只动旋转轴、保持转角不变，所以 w 分量原样保留。
 */
export function applyOrientationCalib(
  q: Quat,
  calib: OrientationCalib | null
): Quat {
  if (!calib) return q;
  const rel = multiplyQuat(conjugateQuat(calib.reference), q);
  const map = calib.axisMap;
  if (!map) return rel;
  const [rw, rx, ry, rz] = rel;
  return normalizeQuat([
    rw,
    map[0][0] * rx + map[0][1] * ry + map[0][2] * rz,
    map[1][0] * rx + map[1][1] * ry + map[1][2] * rz,
    map[2][0] * rx + map[2][1] * ry + map[2][2] * rz,
  ]);
}

// ===== 持久化 =====

function isValidQuat(q: unknown): q is Quat {
  return (
    Array.isArray(q) &&
    q.length === 4 &&
    q.every((v) => typeof v === "number" && isFinite(v)) &&
    Math.hypot(q[0], q[1], q[2], q[3]) > 0.5
  );
}

function isValidMap(m: unknown): m is number[][] {
  return (
    Array.isArray(m) &&
    m.length === 3 &&
    m.every(
      (row) =>
        Array.isArray(row) &&
        row.length === 3 &&
        row.every((v) => typeof v === "number" && isFinite(v))
    )
  );
}

export function loadOrientationCalib(hand: HandKey): OrientationCalib | null {
  try {
    const raw = localStorage.getItem(STORE_KEY_PREFIX + hand);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as OrientationCalib;
    if (!isValidQuat(parsed?.reference)) return null;
    // 矩阵脏了就丢掉矩阵、保留零位：零位仍然有用，而坏矩阵会把朝向彻底搅乱
    if (parsed.axisMap && !isValidMap(parsed.axisMap)) delete parsed.axisMap;
    return parsed;
  } catch {
    return null;
  }
}

export function saveOrientationCalib(
  hand: HandKey,
  calib: OrientationCalib
): void {
  try {
    localStorage.setItem(STORE_KEY_PREFIX + hand, JSON.stringify(calib));
  } catch {
    /* 隐私模式 / 配额用尽：标定仅本次会话有效 */
  }
}

export function clearOrientationCalib(hand: HandKey): void {
  try {
    localStorage.removeItem(STORE_KEY_PREFIX + hand);
  } catch {
    /* ignore */
  }
}
