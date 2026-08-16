/**
 * imuHealth — IMU 姿态漂移自检
 *
 * 手套用的是 ICM-42688 六轴（无磁力计），陀螺零偏会让板载姿态解算持续漂移。
 * 协议（gloveProtocol.ts 头部）里只有融合后的四元数 / 加速度 / 姿态角，
 * **没有原始角速度**，所以软件端补偿不了零偏——我们只能检测，然后提示用户
 * 短按主控按键做硬件校准（规格书 p32）。
 *
 * 判据不是"静止时转了多少度"，而是**四元数与加速度计互相打架的程度**：
 *   gExpected = gravityInHandFrame(q)     四元数认为重力轴在手系里指哪
 *   gMeasured = normalize(acc)            加速度计实测的比力方向
 * 静止（或加速度以重力为主）时这两者应当共线。四元数的 roll/pitch 一旦漂了，
 * 夹角就张开。好处是**不要求手完全静止**，录制过程中的帧也能筛出来用。
 *
 * ⚠ yaw 漂移测不出来，也不该测。六轴没有磁力计，yaw 绝对零点本就不存在
 * （之前实测到的 −53.7° "固定偏置"并不真固定，它只是上电瞬间的随机零点）。
 * 而特征层（sequenceFeatures.ts）已经只用相对四元数 q₀⁻¹⊗qₜ 与重力手系投影
 * R(q)⁻¹·[0,0,1]，两者对世界系 yaw 旋转都不变 —— 所以 yaw 漂移对识别无害。
 * 不要为 yaw 加告警，那只会制造假警报。
 */

import {
  gravityInHandFrame,
  quatConj,
  quatMul,
  quatNormalize,
  type Quat,
} from "./sequenceFeatures";

// ===== 阈值 =====

/** 倾角不一致中位数 < 此值判为健康（度） */
export const TILT_OK_DEG = 5;
/** 倾角不一致中位数 ≥ 此值判为不可用（度），介于两者之间为警告 */
export const TILT_BAD_DEG = 12;
/** 统计所需的最少可用帧数；不足则给 unknown 而不是硬下结论 */
export const MIN_USABLE_FRAMES = 10;
/** "加速度以重力为主"的判定带宽：|acc| 落在 1g×(1±此值) 内的帧才计入统计 */
export const GRAVITY_BAND = 0.15;
/** 静置自检时"算作没动"的角速度上限（度/秒），超过说明用户没按要求放着不动 */
export const STILL_RATE_LIMIT_DEG_S = 6;

// ===== 类型 =====

export interface ImuHealthSample {
  q: Quat;
  /** 旧手套（272B 帧）没有加速度字段，此处为 null */
  acc: [number, number, number] | null;
  /** 相对时间戳（ms）。缺省时 stillRotationDegPerMin 无法计算 */
  t?: number;
}

export type ImuVerdict = "ok" | "warn" | "bad" | "unknown";

export interface ImuHealthReport {
  /** 加速度可用且以重力为主的帧数 */
  usableFrames: number;
  /** 主指标：四元数重力方向与实测加速度方向的夹角中位数（度） */
  tiltInconsistencyDeg: number;
  /** 同一指标的 95 分位，用来看是否只是偶发抖动 */
  p95TiltDeg: number;
  /**
   * 相邻帧四元数夹角累计÷时长（度/分）。仅"静置不动"场景有意义：
   * 加速度不可用的旧手套只能靠这个兜底。有加速度时它只作参考。
   */
  stillRotationDegPerMin: number | null;
  verdict: ImuVerdict;
  /** 中文结论，可直接显示 */
  reason: string;
}

function emptyReport(reason: string): ImuHealthReport {
  return {
    usableFrames: 0,
    tiltInconsistencyDeg: 0,
    p95TiltDeg: 0,
    stillRotationDegPerMin: null,
    verdict: "unknown",
    reason,
  };
}

// ===== 数值工具 =====

function quantile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const pos = (sorted.length - 1) * p;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/**
 * 两个方向之间的夹角（度），且**对整体符号不敏感**：取 v 与 −v 中更近的那个。
 *
 * 为什么要吃掉符号：加速度计静止时测的是比力（specific force），读数指向"上"，
 * 与重力向量恰好反向；而 WORLD_GRAVITY 在 sequenceFeatures.ts 里写成 [0,0,1]，
 * 到底代表"重力朝向"还是"天顶朝向"取决于板载固件的约定，规格书没写。
 * 漂移量 θ（只要 <90°）在两种约定下都表现为 θ，所以取 min 就让这个检测器
 * 与符号约定无关 —— 不必赌固件，也不会因为约定猜错而全程报 180°。
 */
export function axisAngleDeg(
  a: [number, number, number],
  b: [number, number, number]
): number {
  const na = Math.hypot(a[0], a[1], a[2]);
  const nb = Math.hypot(b[0], b[1], b[2]);
  if (!isFinite(na) || !isFinite(nb) || na < 1e-8 || nb < 1e-8) return NaN;
  let dot = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb);
  dot = Math.max(-1, Math.min(1, Math.abs(dot))); // abs = 吃掉符号约定
  return (Math.acos(dot) * 180) / Math.PI;
}

/**
 * 从一批 |acc| 里估出"1g 读数"是多少。
 *
 * 协议没规定加速度单位（sequenceFeatures.ts 的 ACC_SCALE=16 暗示可能是 m/s²，
 * 也可能是 g），所以不能写死。做法：
 *  1. 取 **25 分位**而不是中位数当粗估。重力是 |acc| 的地板——手一动只会往上加，
 *     所以低分位比中位数更接近真 1g。若录的是一段大幅动作，中位数会被抬到
 *     远离 1g 的地方，把静止帧一起筛掉（这正是最初写中位数时被单测抓到的 bug）。
 *  2. 粗估落在 g 或 m/s² 的合理邻域时，**吸附到精确值**，彻底摆脱分布形状的影响。
 *  3. 两个邻域都不落，就直接用粗估并交给带宽兜着（未知量程手套）。
 */
export function estimateGravityMagnitude(sortedMags: number[]): number {
  const coarse = quantile(sortedMags, 0.25);
  if (coarse >= 0.7 && coarse <= 1.4) return 1; // 单位是 g
  if (coarse >= 7 && coarse <= 14) return 9.80665; // 单位是 m/s²
  return coarse;
}

/** 两个四元数之间的旋转角（度），已处理 q 与 −q 同解 */
export function quatAngleDeg(a: Quat, b: Quat): number {
  const rel = quatMul(quatConj(quatNormalize(a)), quatNormalize(b));
  const w = Math.min(1, Math.abs(rel[0]));
  return (2 * Math.acos(w) * 180) / Math.PI;
}

// ===== 主分析 =====

/**
 * 分析一段 IMU 采样的健康度。
 * 采样可以来自静置自检（用户把手套放平不动），也可以来自一条已录序列。
 */
export function analyzeImuHealth(samples: ImuHealthSample[]): ImuHealthReport {
  if (samples.length === 0) return emptyReport("没有采到 IMU 数据");

  // --- 静置旋转量（旧手套的兜底指标，也给新手套做参考） ---
  let stillRotationDegPerMin: number | null = null;
  const tFirst = samples[0].t;
  const tLast = samples[samples.length - 1].t;
  if (
    samples.length >= 2 &&
    typeof tFirst === "number" &&
    typeof tLast === "number" &&
    tLast > tFirst
  ) {
    let totalDeg = 0;
    for (let i = 1; i < samples.length; i++) {
      const d = quatAngleDeg(samples[i - 1].q, samples[i].q);
      if (isFinite(d)) totalDeg += d;
    }
    const minutes = (tLast - tFirst) / 60000;
    if (minutes > 1e-6) stillRotationDegPerMin = totalDeg / minutes;
  }

  // --- 倾角一致性 ---
  const withAcc = samples.filter((s) => {
    if (!s.acc) return false;
    const n = Math.hypot(s.acc[0], s.acc[1], s.acc[2]);
    return isFinite(n) && n > 1e-6;
  });

  if (withAcc.length === 0) {
    // 旧手套（272B）：只能靠静置旋转量
    if (stillRotationDegPerMin == null) {
      return {
        ...emptyReport("这只手套不上报加速度，且采样时长不足，无法判断"),
        stillRotationDegPerMin,
      };
    }
    const rate = stillRotationDegPerMin / 60; // 度/秒
    const drifting = rate > STILL_RATE_LIMIT_DEG_S;
    return {
      usableFrames: 0,
      tiltInconsistencyDeg: 0,
      p95TiltDeg: 0,
      stillRotationDegPerMin,
      verdict: drifting ? "warn" : "unknown",
      reason: drifting
        ? `这只手套不上报加速度，只能测静置旋转量：${stillRotationDegPerMin.toFixed(0)}°/分。` +
          `若采样期间手套确实没动，说明姿态在漂移，请短按主控按键校准。`
        : `这只手套不上报加速度（旧款 272B 帧），无法做倾角一致性检查。` +
          `静置旋转量 ${stillRotationDegPerMin.toFixed(0)}°/分，未见明显异常。`,
    };
  }

  const mags = withAcc
    .map((s) => Math.hypot(s.acc![0], s.acc![1], s.acc![2]))
    .sort((a, b) => a - b);
  const gRef = estimateGravityMagnitude(mags);
  const lo = gRef * (1 - GRAVITY_BAND);
  const hi = gRef * (1 + GRAVITY_BAND);

  const angles: number[] = [];
  for (const s of withAcc) {
    const acc = s.acc!;
    const mag = Math.hypot(acc[0], acc[1], acc[2]);
    // 甩手/敲击帧的加速度不以重力为主，拿它算倾角只会污染统计
    if (mag < lo || mag > hi) continue;
    const a = axisAngleDeg(gravityInHandFrame(s.q), acc);
    if (isFinite(a)) angles.push(a);
  }

  if (angles.length < MIN_USABLE_FRAMES) {
    return {
      usableFrames: angles.length,
      tiltInconsistencyDeg: 0,
      p95TiltDeg: 0,
      stillRotationDegPerMin,
      verdict: "unknown",
      reason:
        `可用帧只有 ${angles.length} 帧（需要 ≥${MIN_USABLE_FRAMES}）。` +
        `手在剧烈运动时加速度不以重力为主，无法判断姿态漂移，请把手套放稳后重试。`,
    };
  }

  angles.sort((a, b) => a - b);
  const median = quantile(angles, 0.5);
  const p95 = quantile(angles, 0.95);

  let verdict: ImuVerdict;
  let reason: string;
  if (median < TILT_OK_DEG) {
    verdict = "ok";
    reason = `姿态与加速度一致（偏差中位数 ${median.toFixed(1)}°），IMU 正常。`;
  } else if (median < TILT_BAD_DEG) {
    verdict = "warn";
    reason =
      `姿态与加速度偏差 ${median.toFixed(1)}°，偏大。` +
      `建议把手套放平不动、短按主控按键做陀螺校准后重新自检。`;
  } else {
    verdict = "bad";
    reason =
      `姿态与加速度偏差 ${median.toFixed(1)}°，四元数已明显漂移，这段数据的朝向通道不可用。` +
      `请把手套放平不动、短按主控按键做陀螺校准，然后重录。`;
  }

  return {
    usableFrames: angles.length,
    tiltInconsistencyDeg: median,
    p95TiltDeg: p95,
    stillRotationDegPerMin,
    verdict,
    reason,
  };
}

/**
 * 直接分析 SequenceSample 的 leftImu / rightImu。
 * 布局见 datasetStore.ts：[T*10]，每帧 = quat4 + acc3 + att3。
 *
 * 这是 post-hoc 计算——数据本来就都存着，不必在录制热路径上多做事，
 * 而且对**存量样本**同样能算，不用重录旧数据。
 */
export function analyzeSequenceImu(
  imu: Float32Array | null,
  frameCount: number,
  timestamps?: Float32Array | null
): ImuHealthReport {
  if (!imu || frameCount <= 0) return emptyReport("这一侧没有手套数据");
  const stride = 10;
  if (imu.length < frameCount * stride) {
    return emptyReport(`IMU 数组长度 ${imu.length} 与帧数 ${frameCount} 不匹配`);
  }

  const samples: ImuHealthSample[] = [];
  for (let i = 0; i < frameCount; i++) {
    const o = i * stride;
    const q: Quat = [imu[o], imu[o + 1], imu[o + 2], imu[o + 3]];
    if (!isFinite(q[0]) || Math.hypot(q[0], q[1], q[2], q[3]) < 1e-6) continue;
    const ax = imu[o + 4];
    const ay = imu[o + 5];
    const az = imu[o + 6];
    // 旧手套写入的是 0：三轴全 0 视为"无加速度"，而不是"零重力"
    const hasAcc =
      isFinite(ax) && isFinite(ay) && isFinite(az) && Math.hypot(ax, ay, az) > 1e-6;
    samples.push({
      q,
      acc: hasAcc ? [ax, ay, az] : null,
      t: timestamps && i < timestamps.length ? timestamps[i] : undefined,
    });
  }
  if (samples.length === 0) return emptyReport("这一侧的 IMU 数据全部无效");
  return analyzeImuHealth(samples);
}

/** verdict 是否应当在 UI 上提醒用户（供采集页 / 训练页统计"可疑样本"用） */
export function isImuSuspect(report: ImuHealthReport | null | undefined): boolean {
  return report?.verdict === "warn" || report?.verdict === "bad";
}

/** 一条序列样本的双手报告里，取更差的那个 verdict */
export function worstVerdict(
  ...reports: (ImuHealthReport | null | undefined)[]
): ImuVerdict {
  // ok < unknown < warn < bad：unknown 排在 ok 之后，因为"没测出来"比"测了没问题"更该被看见
  const order: ImuVerdict[] = ["ok", "unknown", "warn", "bad"];
  let best = -1;
  for (const r of reports) {
    if (!r) continue;
    best = Math.max(best, order.indexOf(r.verdict));
  }
  return best < 0 ? "unknown" : order[best];
}

/**
 * 从**多段独立采样**的报告里挑最差的一份原样返回（不是合成一份新的）。
 *
 * 用于四步校准向导：每一步都是货真价实的静止 3 秒，逐步各算一份报告，
 * 最后取最差的那份展示。**不能把四步的采样拼成一段再分析** ——
 * 步与步之间手在大幅转动，stillRotationDegPerMin 会算成天文数字，
 * 旧手套（无加速度）的兜底判据会直接失效。
 *
 * 排序沿用 worstVerdict 的 ok < unknown < warn < bad；同级时比倾角中位数，
 * 再同就比可用帧数更多的那份（样本更足、结论更可信）。
 */
export function worstReport(
  reports: (ImuHealthReport | null | undefined)[]
): ImuHealthReport | null {
  const order: ImuVerdict[] = ["ok", "unknown", "warn", "bad"];
  let best: ImuHealthReport | null = null;
  for (const r of reports) {
    if (!r) continue;
    if (!best) {
      best = r;
      continue;
    }
    const dRank = order.indexOf(r.verdict) - order.indexOf(best.verdict);
    if (dRank > 0) best = r;
    else if (dRank === 0 && r.tiltInconsistencyDeg > best.tiltInconsistencyDeg)
      best = r;
    else if (
      dRank === 0 &&
      r.tiltInconsistencyDeg === best.tiltInconsistencyDeg &&
      r.usableFrames > best.usableFrames
    )
      best = r;
  }
  return best;
}
