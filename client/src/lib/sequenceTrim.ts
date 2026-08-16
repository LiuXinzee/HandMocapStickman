/**
 * 起手段自动裁剪 —— 把"手从腿上抬到画面里"那段从训练特征里切掉。
 *
 * 为什么需要：录制是「按下 → 抬手入画 → 打手势 → 结束」，抬手那 0.5~1.5s 也进了样本。
 * 它带来的伤害不只是"多了一段没用的"：
 *
 * 1. **吃掉时间预算**。`resampleSequence` 把整条录制均匀重采样到 SEQ_LEN=32 帧，
 *    栅格铺在整段上。抬手占 40% 就意味着真手势只剩 60% 的帧，而抬手时长每条都不同，
 *    同一个词于是落在不同的相位和时间尺度上。
 * 2. **污染参考帧**。每帧四元数是相对**首帧**的（`firstValidQuatInv`），首帧是"手在腿上"，
 *    整条序列的朝向通道都在描述"相对腿上那一刻转了多少"。裁掉之后参考帧自动变成起势姿态。
 * 3. **可能泄漏**。抬手快慢若与词相关（换词时手放下休息更久），模型可以拿它当捷径。
 *
 * 判据分**两层**：
 *
 * **第一层 · 可见性**。手在腿上时 MediaPipe 根本看不到这只手，关键点整帧是 NaN
 * （`datasetStore.ts` 的约定），这是最直接、几乎免费的信号。
 * 注意这一层**只能切掉画面外的那半截**：手一进画面可见性就翻成 true，而"进画面之后
 * 继续抬到起势位置"那段照样是冗余，它有关键点、运动能量还很高，长得最像手势。
 *
 * **第二层 · 到位检测**（`detectArrival`，见下）。在可见段内部再找一次真正的起点。
 * 这里也**不能用运动能量** —— 抬手本身就是大幅运动，能量比很多手势都高，IMU 更是全程在转，
 * 靠"动起来了"找起点会把起点定在抬手开始处，等于什么都没裁。用的是手语研究里
 * transport / preparation phase 的标准判据：**速度谷底**（见 `detectHandArrival`）。
 *
 * 三条守则：
 * - **判不出来就不裁**（无视觉 / 没有够长的可见段 / 裁完太短），宁可留着起手段，
 *   也不要静默切错。每一种不裁的情形都有 `reason`，训练时汇总显示。
 * - **不写数据库**。这里只影响特征构建，`SequenceSample.segments` 一个字节都不动，
 *   所以存量样本不用重录、判据改了重训一次即可，也随时能整体关掉（`trim: null`）。
 * - **推理端显式不裁**。`predictSequence` 拿到的滑窗是纯触觉（`sequenceWindow.ts` 把
 *   landmarks 写成 null），判据本来就不成立；那里显式传 `trim: null` 是为了写明这件事，
 *   免得以后有人给滑窗加上视觉时，推理行为跟着悄悄变。
 */
import type { SequenceSample } from "@/lib/datasetStore";
import { SEQ_LANDMARK_N, SEQ_SENSOR_N } from "@/lib/datasetStore";

/**
 * 到位检测的判据。全部是**保守**方向的门限：任何一条不满足就退回只按可见性裁，
 * 不会因为判不出来而乱切。
 */
export interface ArrivalConfig {
  /** 前导冲程至少要向上移动这么多（归一化图像单位）。不够就当"手不是抬进来的" */
  minRiseNorm: number;
  /** 竖直位移至少是水平位移的这么多倍。横向挥手不是抬手 */
  verticalDominance: number;
  /** 平滑速度落到峰值的这个比例以下算"到位" */
  settleRatio: number;
  /** 到位点最晚不能超过可见段的这个比例，再晚就是在切手势本身了 */
  maxArrivalRatio: number;
  /** 弯折偏离初始手型这么多 ADC 就认为手已经开始动作，起点不再往后推 */
  bendOnsetAdc: number;
  /** 速度平滑窗口帧数（奇数）。关键点抖动在 30fps 下能造出假的速度谷 */
  smoothFrames: number;
}

export const DEFAULT_ARRIVAL: ArrivalConfig = {
  minRiseNorm: 0.08,
  verticalDominance: 1.5,
  settleRatio: 0.35,
  maxArrivalRatio: 0.6,
  bendOnsetAdc: 25,
  smoothFrames: 3,
};

export interface TrimConfig {
  /** 连续可见帧数下限。MediaPipe 偶发单帧误检，短于此的可见段当噪声 */
  minRunFrames: number;
  /** 起止各向外留白（ms）。入画前一小段通常已经在起势，别切到手势本身 */
  padMs: number;
  /** 裁剪后至少要留下的帧数；不够就整条不裁 */
  minKeptFrames: number;
  /** 裁剪后至少要留下的时长（ms）；不够就整条不裁 */
  minKeptMs: number;
  /** 第二层到位检测；null = 只按可见性裁 */
  arrival: ArrivalConfig | null;
}

export const DEFAULT_TRIM: TrimConfig = {
  minRunFrames: 3,
  padMs: 100,
  minKeptFrames: 6,
  minKeptMs: 250,
  arrival: DEFAULT_ARRIVAL,
};

export type TrimReason =
  /** 裁了 */
  | "applied"
  /** 两只手都没有关键点数组（没开摄像头录的） */
  | "no_vision"
  /** 有关键点但没有够长的连续可见段（全程都没看见手） */
  | "no_run"
  /** 裁完剩下太短，判据可疑，放弃 */
  | "too_short"
  /** 全程都可见，没什么可裁 */
  | "full_span";

export type ArrivalReason =
  /** 起点被推后了 */
  | "applied"
  /** 判据整个关掉了（`cfg.arrival === null`） */
  | "disabled"
  /** 可见段里有效手腕点太少，算不出速度 */
  | "no_landmarks"
  /** 前导段没有明显向上的冲程 —— 手不是抬进来的，没有 transport 段可切 */
  | "no_rise"
  /** 速度一直没落下来，切不出边界 */
  | "no_settle"
  /** 谷底落在可见段后段，再切就是在切手势 */
  | "too_late"
  /** 推后之后剩下太短 */
  | "too_short";

export interface ArrivalInfo {
  /** 最终采用的起点；未采用时等于可见段起点 */
  frame: number;
  reason: ArrivalReason;
  /** 相对可见段起点多切掉的帧数 / 毫秒 */
  droppedFrames: number;
  droppedMs: number;
  /**
   * 弯折钳位是否生效：手型在手腕到位之前就开始成形，起点被往前拉回。
   * 这是**保守**方向的修正，不是判据失败。
   */
  bendClamped: boolean;
}

export interface TrimSpan {
  /** 保留区间起始帧（含） */
  startFrame: number;
  /** 保留区间结束帧（不含） */
  endFrame: number;
  /** 是否真的要裁；false 时 start/end 就是整条 */
  applied: boolean;
  reason: TrimReason;
  /** 保留时长 / 总时长；不裁时为 1 */
  keptRatio: number;
  /** 第二层到位检测的结果；一层就退出（no_vision/no_run/T<2）时为 null */
  arrival: ArrivalInfo | null;
}

function fullSpan(
  sample: SequenceSample,
  reason: TrimReason,
  arrival: ArrivalInfo | null = null
): TrimSpan {
  return {
    startFrame: 0,
    endFrame: sample.frameCount,
    applied: false,
    reason,
    keptRatio: 1,
    arrival,
  };
}

function noArrival(frame: number, reason: ArrivalReason): ArrivalInfo {
  return { frame, reason, droppedFrames: 0, droppedMs: 0, bendClamped: false };
}

/** 该帧这只手是否有有效关键点。缺失帧是整帧 NaN，查首尾两个分量足够 */
function handVisibleAt(lm: Float32Array | null, t: number): boolean {
  if (!lm) return false;
  const o = t * SEQ_LANDMARK_N;
  if (o + SEQ_LANDMARK_N > lm.length) return false;
  return (
    Number.isFinite(lm[o]) &&
    Number.isFinite(lm[o + SEQ_LANDMARK_N - 1])
  );
}

// ===== 第二层：到位检测 =====

interface WristSample {
  frame: number;
  ms: number;
  x: number;
  y: number;
}

/** 取一只手在 [from,to) 内所有有效的手腕点（关键点 0 号 = 手腕，归一化图像坐标） */
function wristTrack(
  lm: Float32Array | null,
  ts: Float32Array,
  from: number,
  to: number
): WristSample[] {
  if (!lm) return [];
  const out: WristSample[] = [];
  for (let t = from; t < to; t++) {
    const o = t * SEQ_LANDMARK_N;
    if (o + SEQ_LANDMARK_N > lm.length) break;
    const x = lm[o];
    const y = lm[o + 1];
    if (Number.isFinite(x) && Number.isFinite(y))
      out.push({ frame: t, ms: ts[t], x, y });
  }
  return out;
}

/**
 * 一只手的到位帧。
 *
 * 判据是手语研究里 transport / preparation phase 的标准做法：**速度谷底**。
 * 把手送到起势位置是一次弹道式冲程 —— 速度先冲到峰值，到位时落下来，谷底就是手势起点。
 *
 * 三道门限缺一不可，任何一条不满足就返回不采用的理由：
 * 1. **前导段必须是向上的冲程**（`dy ≤ −minRiseNorm` 且 `|dy| ≥ verticalDominance·|dx|`）。
 *    没有这一条，横向挥手（"再见"）的第一个摆幅会被当成抬手切掉。
 *    图像坐标 y 向下为正，所以"抬起来" = y 变小 = dy 为负。
 * 2. **必须有明显的谷**（落到峰值的 `settleRatio` 以下）。一直在动说明切不出边界。
 * 3. **谷底必须落在可见段前 `maxArrivalRatio`**。再靠后就是在切手势本身了。
 */
function detectHandArrival(
  lm: Float32Array | null,
  ts: Float32Array,
  from: number,
  to: number,
  cfg: ArrivalConfig
): { frame: number; reason: ArrivalReason } {
  const w = wristTrack(lm, ts, from, to);
  // 速度要差分，平滑还要再吃掉几个点，样本太少算出来的谷是噪声
  if (w.length < Math.max(5, cfg.smoothFrames + 2))
    return { frame: from, reason: "no_landmarks" };

  // 归一化图像单位 / 秒。用时间戳而不是帧号：丢帧时按帧号算的速度是假的
  const raw = new Float64Array(w.length);
  for (let i = 1; i < w.length; i++) {
    const dt = (w[i].ms - w[i - 1].ms) / 1000;
    if (dt <= 1e-6) {
      raw[i] = raw[i - 1];
      continue;
    }
    const dx = w[i].x - w[i - 1].x;
    const dy = w[i].y - w[i - 1].y;
    raw[i] = Math.hypot(dx, dy) / dt;
  }
  raw[0] = raw.length > 1 ? raw[1] : 0;

  const half = Math.max(0, Math.floor(cfg.smoothFrames / 2));
  const v = new Float64Array(w.length);
  for (let i = 0; i < w.length; i++) {
    let sum = 0;
    let n = 0;
    for (let k = i - half; k <= i + half; k++) {
      if (k < 0 || k >= w.length) continue;
      sum += raw[k];
      n++;
    }
    v[i] = sum / n;
  }

  // 到位点的最晚允许位置（按**时间**算，与 padMs / 栅格换算保持同一口径）
  const spanMs = w[w.length - 1].ms - w[0].ms;
  if (spanMs <= 1e-6) return { frame: from, reason: "no_landmarks" };
  const limitMs = w[0].ms + spanMs * cfg.maxArrivalRatio;

  // 峰值只在允许窗口内找 —— 手势本身的最大速度经常出现在后半段，
  // 拿它当"抬手峰值"会让 settleRatio 的门限失去意义
  let peak = 0;
  let vmax = 0;
  for (let i = 0; i < w.length && w[i].ms <= limitMs; i++) {
    if (v[i] > vmax) {
      vmax = v[i];
      peak = i;
    }
  }
  if (vmax <= 1e-6) return { frame: from, reason: "no_rise" };

  // 门限 2：峰值之后第一个落到 settleRatio 以下的点
  let settle = -1;
  for (let i = peak + 1; i < w.length; i++) {
    if (v[i] <= vmax * cfg.settleRatio) {
      settle = i;
      break;
    }
  }
  if (settle < 0) return { frame: from, reason: "no_settle" };
  // 门限 3
  if (w[settle].ms > limitMs) return { frame: from, reason: "too_late" };

  // 门限 1：**要切掉的那一段整体**必须是向上的冲程。
  // 位移要量到 settle、不能量到 peak —— 匀速上抬时速度峰值就落在第一帧，
  // 量到 peak 得到的位移是 0，判据会把每一次标准抬手都否掉。
  const dy = w[settle].y - w[0].y;
  const dx = w[settle].x - w[0].x;
  if (dy > -cfg.minRiseNorm) return { frame: from, reason: "no_rise" };
  if (Math.abs(dy) < cfg.verticalDominance * Math.abs(dx))
    return { frame: from, reason: "no_rise" };

  return { frame: w[settle].frame, reason: "applied" };
}

/**
 * 弯折通道开始偏离初始手型的那一帧 —— 到位帧的**钳位**，不是与它竞争的估计。
 *
 * 手型经常在抬手途中就捏出来了；那一刻起画面里发生的事就可能已经属于手势，
 * 起点不该再往后推。取 `min(到位帧, 这一帧)`，是"宁可少裁"的方向。
 *
 * 137 维重排后弯折固定在下标 60~64（`sensorMapping.ts:50` / `:86`）。左右手这五路的
 * 手指顺序是**相反**的，但这里只求偏离幅度、不关心是哪根手指，所以不需要 canonical 化。
 */
function bendOnsetFrame(
  sensor: Uint8Array | null,
  from: number,
  to: number,
  thresholdAdc: number
): number | null {
  if (!sensor) return null;
  if (to * SEQ_SENSOR_N > sensor.length) return null;
  const BEND0 = 60;
  const BEND_N = 5;

  const baseN = Math.min(3, to - from);
  if (baseN <= 0) return null;
  const base = new Float64Array(BEND_N);
  for (let t = from; t < from + baseN; t++)
    for (let k = 0; k < BEND_N; k++)
      base[k] += sensor[t * SEQ_SENSOR_N + BEND0 + k] / baseN;

  for (let t = from + baseN; t < to; t++) {
    for (let k = 0; k < BEND_N; k++) {
      const d = Math.abs(sensor[t * SEQ_SENSOR_N + BEND0 + k] - base[k]);
      if (d > thresholdAdc) return t;
    }
  }
  return null;
}

/**
 * 可见段 [from,to) 内的真正起点。
 *
 * 双手各算一次，取**较早**的那个：任一只手已经到位并开始做事，就不该再往后切。
 * 某只手算不出来（没入画 / 没抬手）不构成否决 —— 单手词的另一只手本来就整条不动。
 */
function detectArrival(
  sample: SequenceSample,
  from: number,
  to: number,
  cfg: ArrivalConfig
): ArrivalInfo {
  const ts = sample.timestamps;
  const hands = [
    detectHandArrival(sample.leftLandmarks, ts, from, to, cfg),
    detectHandArrival(sample.rightLandmarks, ts, from, to, cfg),
  ];
  const hit = hands.filter((h) => h.reason === "applied");
  if (hit.length === 0) {
    // 两只手都没结论时报**最有信息量**的那个理由：no_landmarks 说明这只手压根没参与
    const informative =
      hands.find((h) => h.reason !== "no_landmarks") ?? hands[0];
    return noArrival(from, informative.reason);
  }

  let frame = Math.min(...hit.map((h) => h.frame));

  const clampCandidates = [
    bendOnsetFrame(sample.leftSensor, from, to, cfg.bendOnsetAdc),
    bendOnsetFrame(sample.rightSensor, from, to, cfg.bendOnsetAdc),
  ].filter((f): f is number => f !== null);
  let bendClamped = false;
  if (clampCandidates.length > 0) {
    const onset = Math.min(...clampCandidates);
    if (onset < frame) {
      frame = onset;
      bendClamped = true;
    }
  }

  if (frame <= from) return noArrival(from, "no_rise");
  return {
    frame,
    reason: "applied",
    droppedFrames: frame - from,
    droppedMs: ts[frame] - ts[from],
    bendClamped,
  };
}

/**
 * 找出"手在画面里"的那一段。
 *
 * 可见性取**双手的并集**：任一只手看得见就算已经入画。取交集会在单手词上永远为假
 * （另一只手整条都不在画面里），那样等于关掉裁剪。
 */
export function detectSignSpan(
  sample: SequenceSample,
  cfg: TrimConfig = DEFAULT_TRIM
): TrimSpan {
  const T = sample.frameCount;
  if (T < 2) return fullSpan(sample, "too_short");
  if (!sample.leftLandmarks && !sample.rightLandmarks)
    return fullSpan(sample, "no_vision");

  const ts = sample.timestamps;

  // 最长连续可见段。取最长而不是第一段：抬手途中被摄像头扫到一两帧的误检不该定为起点
  let bestStart = -1;
  let bestLen = 0;
  let runStart = -1;
  for (let t = 0; t <= T; t++) {
    const visible =
      t < T &&
      (handVisibleAt(sample.leftLandmarks, t) ||
        handVisibleAt(sample.rightLandmarks, t));
    if (visible) {
      if (runStart < 0) runStart = t;
    } else if (runStart >= 0) {
      const len = t - runStart;
      if (len > bestLen) {
        bestLen = len;
        bestStart = runStart;
      }
      runStart = -1;
    }
  }
  if (bestLen < cfg.minRunFrames) return fullSpan(sample, "no_run");

  const runFirst = bestStart;
  const runLast = bestStart + bestLen - 1;

  // 向外留白：把入画前/出画后 padMs 以内的帧也收进来
  let start = runFirst;
  while (start > 0 && ts[runFirst] - ts[start - 1] <= cfg.padMs) start--;
  let end = runLast + 1; // exclusive
  while (end < T && ts[end] - ts[runLast] <= cfg.padMs) end++;

  // 第二层：可见段内部再找真正的起点。
  // **不能因为可见性没裁到东西就跳过** —— 手已经在画面里、只是还在往上抬，
  // 恰恰是这一层唯一能处理的情形（可见性会判 full_span）。
  let arrival = cfg.arrival
    ? detectArrival(sample, start, end, cfg.arrival)
    : noArrival(start, "disabled");
  if (arrival.reason === "applied") {
    // 推后之后剩得太短就整个放弃这一层，退回只按可见性裁（不是退回整条）
    if (
      end - arrival.frame < cfg.minKeptFrames ||
      ts[end - 1] - ts[arrival.frame] < cfg.minKeptMs
    ) {
      arrival = noArrival(start, "too_short");
    } else {
      start = arrival.frame;
    }
  }

  if (start === 0 && end === T) return fullSpan(sample, "full_span", arrival);

  const keptFrames = end - start;
  const keptMs = ts[end - 1] - ts[start];
  if (keptFrames < cfg.minKeptFrames || keptMs < cfg.minKeptMs)
    return fullSpan(sample, "too_short", arrival);

  const totalMs = ts[T - 1] - ts[0];
  return {
    startFrame: start,
    endFrame: end,
    applied: true,
    reason: "applied",
    keptRatio: totalMs > 1e-6 ? keptMs / totalMs : 1,
    arrival,
  };
}

/**
 * 把保留区间换成重采样栅格的归一化边界 [a, b]。
 *
 * `resampleSequence` 的栅格是**归一化时间**（`sampleIndex` 把 u 映射到
 * `ts[0] + u·(ts[T-1] − ts[0])`），不是归一化帧下标 —— 所以这里必须按时间戳换算，
 * 按帧数比例算会在丢帧的录制上偏掉。
 */
export function spanToGridBounds(
  sample: SequenceSample,
  span: TrimSpan
): { a: number; b: number } {
  const ts = sample.timestamps;
  const T = sample.frameCount;
  if (!span.applied || T < 2) return { a: 0, b: 1 };
  const total = ts[T - 1] - ts[0];
  if (total <= 1e-6) return { a: 0, b: 1 };
  return {
    a: (ts[span.startFrame] - ts[0]) / total,
    b: (ts[span.endFrame - 1] - ts[0]) / total,
  };
}

export interface TrimStats {
  total: number;
  applied: number;
  /** 各种不裁的原因计数 */
  skipped: Record<Exclude<TrimReason, "applied">, number>;
  /** 被裁样本的平均保留比例；没有被裁的样本时为 1 */
  meanKeptRatio: number;
  /** 第二层（到位检测）生效的条数 */
  arrivalApplied: number;
  /** 第二层在可见段之外**额外**切掉的平均毫秒数；一条都没生效时为 0 */
  meanArrivalDroppedMs: number;
  /** 其中被弯折钳位往前拉回的条数 —— 这个数大说明手型都在抬手途中就捏好了 */
  arrivalBendClamped: number;
}

/** 一批样本的裁剪情况汇总 —— 训练完在页面上显示一行，免得裁了什么完全看不见 */
export function summarizeTrim(
  samples: SequenceSample[],
  cfg: TrimConfig = DEFAULT_TRIM
): TrimStats {
  const stats: TrimStats = {
    total: samples.length,
    applied: 0,
    skipped: { no_vision: 0, no_run: 0, too_short: 0, full_span: 0 },
    meanKeptRatio: 1,
    arrivalApplied: 0,
    meanArrivalDroppedMs: 0,
    arrivalBendClamped: 0,
  };
  let ratioSum = 0;
  let droppedSum = 0;
  for (const s of samples) {
    const span = detectSignSpan(s, cfg);
    if (span.applied) {
      stats.applied++;
      ratioSum += span.keptRatio;
    } else {
      stats.skipped[span.reason as Exclude<TrimReason, "applied">]++;
    }
    // 到位检测独立统计。它是 applied 的子集（起点一被推后，span 必然也算裁过了），
    // 单列是因为**它能把本来 full_span 的样本变成裁过的** —— 手全程在画面里、
    // 只是前半段还在往上抬，那正是第一层完全无能为力的情形。
    if (span.arrival?.reason === "applied") {
      stats.arrivalApplied++;
      droppedSum += span.arrival.droppedMs;
      if (span.arrival.bendClamped) stats.arrivalBendClamped++;
    }
  }
  if (stats.applied > 0) stats.meanKeptRatio = ratioSum / stats.applied;
  if (stats.arrivalApplied > 0)
    stats.meanArrivalDroppedMs = droppedSum / stats.arrivalApplied;
  return stats;
}
