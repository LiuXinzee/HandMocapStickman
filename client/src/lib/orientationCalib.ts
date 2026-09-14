/**
 * orientationCalib — 手套朝向标定（陀螺零位 + 轴向重映射）
 *
 * 移植自 glove_visual/cc_part2 的 glove-protocol.ts（2026-08-07 版逻辑），
 * 解决的是"我戴着手套摆的朝向和屏幕上手模的朝向对不上"。分两层：
 *
 *  1. **零位** `reference`：把某个约定姿态（竖立、**手心朝自己**）记为单位旋转，
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
 *
 * ===== 零位约定：手心**朝自己**，不是朝屏幕 =====
 *
 * 这条曾经写错过，代价很实在，所以写清楚为什么：
 *
 * `MODEL_MOTION_AXES` 那张表是按"手心朝自己"推出来的（三个动作各 90°、两轴正交、
 * 转轴无符号歧义）。若零位改成手心朝屏幕，两者差 180° 绕 Y —— 于是第 ① 步平铺
 * 不再是绕 −X 的 90°，而变成绕 (0,1,1)/√2 的 **180° 复合旋转**，`buildAxisMap`
 * 会解出一个完全错的矩阵。**换零位约定就必须同时重推这张表，不能只改文字。**
 *
 * 顺带解释一个容易搞反的点：手模是**第一人称**的（屏幕上那只手就是你自己的手，
 * 模型正对着你）。所以"手心朝自己"在模型空间里是**手心朝相机**，也就是
 * `poseQuat` 为单位四元数时示意手模呈现的样子 —— 向导的示意手模一直是对的，
 * 错的只是 ② 那一步的**文字说明**（见 VirtualMocap.tsx）。
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

/**
 * 重力核对的结论（见 `validateAxisMapWithGravity`）。
 * `upY` 是"零位那一刻传感器测到的上，经矩阵换算到模型系之后的 Y 分量"，
 * 期望 +1；`corrected` 表示解出来的矩阵整体翻转了、已被自动纠正。
 */
export interface GravityCheck {
  upY: number;
  corrected: boolean;
}

export interface OrientationCalib {
  /** 竖立、**手心朝自己** 的姿态四元数 = 零位（约定见文件头，改不得） */
  reference: Quat;
  /** 平铺（手心朝上）参考，仅用于反解俯仰轴与事后核对 */
  flat?: Quat;
  /** 手心相对 参考，仅用于反解偏摆轴与事后核对 */
  palmsIn?: Quat;
  /** 3×3 轴向重映射；缺省表示只做了零位（轴向识别没通过门限） */
  axisMap?: number[][];
  axisQuality?: AxisQuality;
  /** 重力核对结论；缺省表示这次没做（手套不上报加速度，或零位那步没采到） */
  gravityCheck?: GravityCheck;
}

/**
 * 三个标定动作的**目标值**。向导要把这些数显示给用户 —— 之前只说"翻到手心朝内"，
 * 不给目标角，结果实测出现过 40°（勉强够）和 160°（转过头）同时存在。
 */
export const TARGET_MOTION_DEG = 90;
export const TARGET_SEPARATION_DEG = 90;

/**
 * 轴向识别门限：两个动作各自至少转这么多度，且两轴至少分开这么多度。
 *
 * **曾经是 15 / 25，太松了。** 提到 40 / 30（与 glove_visual 参考实现一致）的理由：
 * `buildAxisMap` 是"强制贴合"的 —— 它把实测轴硬拧到目标轴上，永远返回一个合法的
 * 旋转矩阵、从不失败。于是界面照样显示"轴向映射已写入"，用户拿到的是一个
 * 看着正常、实际拧歪的矩阵，表现为"手模跟着动，但方位分不出来"。
 *
 * 转角小 → 转轴方向被噪声主导；轴分离小 → 下面 Gram-Schmidt 会把 swing 轴在
 * pitch 轴上的投影整段扣掉（分离 60° 时约有 cos60° = 50% 的实测方向被丢弃、
 * 换成正交假设）。这两种情况都不如只留零位。
 */
export const MIN_MOTION_DEG = 40;
export const MIN_SEPARATION_DEG = 30;

/**
 * 转角超过这个值就有**符号翻转风险**：绕 n 转 180° ≡ 绕 −n 转 180°，
 * 所以在 180° 附近实测转轴的正负号是不稳定的（w = cos(θ/2)，165° 时只剩 0.13）。
 * 下一次重标可能整体反向。
 *
 * 只告警、不拒收：拒了就退回"仅零位"，那比一个略微不稳的矩阵更差。
 */
export const FLIP_RISK_DEG = 165;

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
 * 两个标定动作在**模型坐标系**里期望的转轴。
 *
 * 零位是"竖立、**手心朝自己**（手背对屏幕）"—— 这张表只在这个零位下成立，
 * 换约定必须重推，理由见文件头。在这个约定下三个动作各 90°、两轴理论正交、
 * 转轴无符号歧义：
 *   俯仰 = 竖立→前倾翻掌至手心朝上平铺、指尖朝屏幕，绕 −X 转 90°（双手相同）
 *   偏摆 = 竖立→双手手心相对，左手绕 +Y、右手绕 −Y 各 90°（动作本身镜像对称）
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

/** 3×3 矩阵作用在向量上 */
export function applyMap(map: number[][], v: Vec3): Vec3 {
  return [
    map[0][0] * v[0] + map[0][1] * v[1] + map[0][2] * v[2],
    map[1][0] * v[0] + map[1][1] * v[1] + map[1][2] * v[2],
    map[2][0] * v[0] + map[2][1] * v[1] + map[2][2] * v[2],
  ];
}

/**
 * 绕单位轴 m 转 180° 的旋转矩阵，Rodrigues 在 θ=180° 的退化形式：`R = 2mmᵀ − I`。
 * 对 m = (−1,0,0) 就是 diag(1,−1,−1)。这里按 m 通用地算，
 * 免得 `MODEL_MOTION_AXES` 哪天改了俯仰轴、这段却还写死成 diag。
 */
function rotation180About(m: Vec3): number[][] {
  const n = normalize3(m);
  if (!n) return [[1,0,0],[0,1,0],[0,0,1]];
  const out: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++) out[r][c] = 2 * n[r] * n[c] - (r === c ? 1 : 0);
  return out;
}

function matMul(a: number[][], b: number[][]): number[][] {
  const out: number[][] = [[0,0,0],[0,0,0],[0,0,0]];
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      for (let k = 0; k < 3; k++) out[r][c] += a[r][k] * b[k][c];
  return out;
}

/**
 * 判据的容差：|upY| 要过这个数才敢下结论（cos 60° = 0.5，即偏差 60° 以内）。
 * 松是故意的 —— 这一关只用来分辨"正着"和"整体翻转 180°"这两种相差极远的情况，
 * 不是用来量精度的。零位那步前臂没竖直、或人没站直，都会让 upY 离 ±1 有距离。
 */
const GRAVITY_DECISIVE = 0.5;

/**
 * 用**重力**核对轴向矩阵有没有整体翻转，翻了就纠正回来。
 *
 * ===== 为什么需要这一关 =====
 *
 * `buildAxisMap` 只吃两个**转轴**，而转轴反解不出方向的正负：③ 那步左手往内翻还是
 * 往外翻，实测转轴恰好差一个负号。往外翻的话 s₂ → −s₂、s₃ = s₁×s₂ → −s₃，于是
 *
 *   A' = m₁s₁ᵀ − m₂s₂ᵀ − m₃s₃ᵀ = R_{m₁}(180°) · A
 *
 * —— 一个**合法的旋转矩阵**，`buildAxisMap` 从不失败，三个质量数（俯仰 90°、
 * 偏摆 90°、分离 90°）也**全部正常**。界面只显示一个绿色✓，而用户看到的是：
 * 绕模型 Y/Z 的旋转全部反号、绕 X 的正确。摆"拇指朝上"显示成拇指朝下，
 * 手指方向也跟着左右镜像。这是三个质量数结构上抓不到的一类错。
 *
 * ===== 判据 =====
 *
 * 静止时加速度计测的是**比力**，方向朝上（`imuHealth.ts` 拿 `acc` 与
 * `gravityInHandFrame(q) = R⁻¹·[0,0,1]` 比夹角，两者同向，见那里的注释）。
 * 零位那一步前臂竖直、指尖朝上，模型此刻是单位旋转，所以**模型系的上就是 +Y**：
 *
 *   A · normalize(acc_零位) ≈ (0, +1, 0)
 *
 * 翻转的那份会得到 R_{m₁}(180°)·(0,1,0) = (0,−1,0)，相差整整 180° —— 这两种情况
 * 离得足够远，用一个很松的门限（`GRAVITY_DECISIVE`）就分得开。纠正就是再乘一次
 * R_{m₁}(180°)（它是自逆的）。
 *
 * ===== 这一关抓不到什么 =====
 *
 * 重力只定一个方向，所以绕**竖直轴**的误差它看不见 —— 也就是② 零位做成"手心朝
 * 屏幕"（差 180° 绕 Y）那种。那种错另有信号：① 会从 90° 变成 180° 复合旋转，
 * 被 `FLIP_RISK_DEG` 那条警告拦下。两条互补，都不能省。
 *
 * @param acc 零位那一步的平均加速度（原始机体系）。null / 近零 / 旧款手套不上报
 *   → 返回 null，表示这次没法核对（**不是**"核对通过"，调用方要区分开）。
 */
export function validateAxisMapWithGravity(
  handKey: HandKey,
  map: number[][],
  acc: Vec3 | null
): { map: number[][]; check: GravityCheck } | null {
  if (!acc) return null;
  const up = normalize3(acc);
  if (!up) return null; // 全零：这一路没数据，不是"朝下"
  const upY = applyMap(map, up)[1];
  if (upY >= GRAVITY_DECISIVE) return { map, check: { upY, corrected: false } };
  if (upY <= -GRAVITY_DECISIVE) {
    const fixed = matMul(rotation180About(MODEL_MOTION_AXES[handKey].pitch), map);
    return { map: fixed, check: { upY, corrected: true } };
  }
  // 落在中间带：零位那步大概没竖直（或者人没站直），不足以下结论。
  // 既不纠正也不算通过 —— 把 upY 原样带出去，由 axisQualityWarning 提示重做。
  return { map, check: { upY, corrected: false } };
}

/**
 * 从向导四步采到的三个姿态四元数组装一份标定。
 * 零位（竖立）是必需的；平铺与手心相对齐了才尝试轴向识别，
 * 没过门限就只留零位 —— 宁可少修一层，也不要用一个乱解出来的矩阵把朝向搅得更差。
 *
 * @param zeroAcc 零位那一步的平均加速度。给了就用重力核对矩阵有没有整体翻转
 *   （见 `validateAxisMapWithGravity`）—— 这是唯一能拦住"③ 翻反方向"的一关，
 *   三个质量数对那种错是全绿的。旧款手套不上报加速度，传 null 即可。
 */
export function assembleOrientationCalib(
  handKey: HandKey,
  zero: Quat | null,
  flat: Quat | null,
  palmsIn: Quat | null,
  zeroAcc: Vec3 | null = null
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
  if (!map) return calib;
  // 重力核对：③ 翻反方向解出来的矩阵在这里才会露出来（三个质量数是全绿的）
  const checked = validateAxisMapWithGravity(handKey, map, zeroAcc);
  calib.axisMap = checked ? checked.map : map;
  if (checked) calib.gravityCheck = checked.check;
  return calib;
}

/** 轴向没识别成功的原因（axisMap 缺失时用来提示用户重做哪一步） */
export function axisMapFailReason(calib: OrientationCalib): string | null {
  if (calib.axisMap) return null;
  const q = calib.axisQuality;
  if (!q) return "只采到零位，缺平铺或手心相对";
  if (q.pitchDeg < MIN_MOTION_DEG)
    return `平铺那步只转了 ${q.pitchDeg.toFixed(0)}°（要 ${TARGET_MOTION_DEG}°），前臂要真的从竖立放平到水平`;
  if (q.swingDeg < MIN_MOTION_DEG)
    return `手心相对那步只转了 ${q.swingDeg.toFixed(0)}°（要 ${TARGET_MOTION_DEG}°），手腕要真的翻到手心朝内`;
  if (q.separationDeg < MIN_SEPARATION_DEG)
    return `两个动作的转轴只差 ${q.separationDeg.toFixed(0)}°（要接近 ${TARGET_SEPARATION_DEG}°），说明做成了同一个方向`;
  return "轴向矩阵反解失败";
}

/**
 * 矩阵**过了门限、但仍然不可靠**的原因。与 `axisMapFailReason` 互补：
 * 那个管"没写入"，这个管"写进去了却不好使"—— 后者才是实际踩到的坑，
 * 因为过了门限界面就只显示一个绿色的✓，用户没有任何线索知道该重做。
 *
 * 四种情况，按危害排：
 *  1. 重力核对没定论 → 零位那步前臂没竖直，整体翻转这一关等于没设防（见下）
 *  2. 转角接近 180° → 转轴符号不稳（见 `FLIP_RISK_DEG`）
 *  3. 转角远离 90° → 转过头/没转够，轴向本身仍可用但精度差
 *  4. 轴分离远离 90° → Gram-Schmidt 补出来的成分占比大
 *
 * "已自动纠正翻转"**不在这里报** —— 那是个已解决的事实，不是待办警告，
 * 由向导汇总那边当普通结论显示（`gravityCheck.corrected`）。
 */
export function axisQualityWarning(calib: OrientationCalib): string | null {
  const q = calib.axisQuality;
  if (!calib.axisMap || !q) return null;
  const g = calib.gravityCheck;
  if (g && Math.abs(g.upY) < GRAVITY_DECISIVE) {
    return (
      `重力核对没定论（零位那步测到的"上"换算到模型系只有 ${g.upY.toFixed(2)}，` +
      `应接近 +1）—— 说明②竖立那步前臂没真正竖直。这一关是唯一能拦住"③翻反方向"的，` +
      `没过就等于没设防：建议站直、前臂竖直，重做一遍`
    );
  }
  const flip = [
    ["平铺", q.pitchDeg],
    ["手心相对", q.swingDeg],
  ].filter(([, deg]) => (deg as number) > FLIP_RISK_DEG);
  if (flip.length) {
    const which = flip.map(([n, d]) => `${n} ${(d as number).toFixed(0)}°`).join("、");
    return `${which} 接近 180°，转轴正负号不稳定，下次重标可能整体反向 —— 请重做，转到 ${TARGET_MOTION_DEG}° 就停`;
  }
  const off = [
    ["平铺", q.pitchDeg],
    ["手心相对", q.swingDeg],
  ].filter(([, deg]) => Math.abs((deg as number) - TARGET_MOTION_DEG) > 30);
  if (off.length) {
    const which = off.map(([n, d]) => `${n} ${(d as number).toFixed(0)}°`).join("、");
    return `${which} 离 ${TARGET_MOTION_DEG}° 偏远，方位分辨会变差 —— 建议重做这一步`;
  }
  if (q.separationDeg < 75)
    return `两个动作的转轴只分开 ${q.separationDeg.toFixed(0)}°（理想 ${TARGET_SEPARATION_DEG}°），矩阵有相当一部分是正交化补出来的`;
  return null;
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
    // 矩阵没了，重力结论就是无主的；留着会让界面报一份不存在矩阵的核对结果
    if (!parsed.axisMap) delete parsed.gravityCheck;
    if (
      parsed.gravityCheck &&
      !isFinite(parsed.gravityCheck.upY as unknown as number)
    )
      delete parsed.gravityCheck;
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
