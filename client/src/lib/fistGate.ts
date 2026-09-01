/*
 * fistGate —— 在「谢谢」和「难过」之间仲裁
 *
 * ===== 现在有两个判据，拇指已经不是主判据了（2026-08-31 改） =====
 *
 * 使用者指出了一个更根本的物理差别：谢谢**整只手不动**、只有拇指下压；
 * 难过是掌心朝胸口**画一个圈**，手腕朝向一路在变。量下来（2000ms 窗、348 个窗）：
 *
 *   四元数累计路径转角速率   分对 97.7%
 *   拇指单点压力峰值（下面这一整套）   分对 88.2%
 *
 * 所以顺序反过来了：**转角速率先定，只有它定不了的时候才问拇指。**
 * 判据本身、阈值怎么来的、为什么必须是"路径转角"而不是"端点夹角"、
 * 以及它有一个 `ROT_MIN_SPAN_MS` 的适用下限 —— 全在 `rotationRate.ts` 的文件头。
 *
 * 拇指判据**没有被删**，它管两件事：
 *   1. 速率落在中间带（[50,70) °/s，2000ms 窗下占 16%）时的仲裁；
 *   2. 区间短于 1500ms 时的**唯一**判据 —— 短区间下速率判据比拇指还差，
 *      句子路径的词区间大多在这个范围里，所以那条路基本还是拇指在管。
 *
 * 下面这一整段是拇指判据的原始记录，一个字没改：
 *
 * ===== 为什么需要它 =====
 *
 * 这两个词的手型差别很小：
 *   谢谢  = 竖大拇指，**大拇指下压一次**（要用力，拇指会压到食指侧面）
 *   难过  = 虚握拳（不是紧握），拳心朝内在胸口画圈 —— 拇指全程没吃力
 *
 * 模型对**录制数据**是分得开的（模拟实时滑窗 170/170 全对、置信度 0.994），
 * 但实测打的时候会互认。15 条样本的泛化缺口只能靠重录补，而使用者要求不重录。
 *
 * 所以改成不让网络判这一刀，直接用物理量判 —— 判据是**拇指 12 个压力点里的
 * 单点峰值**（不是 12 点均值）。之所以必须取单点峰值：
 *   "下压一次"是短促的、集中在一两个点上的尖峰。12 点取均值会把它摊平
 *   （均值口径下 谢谢 1.25 vs 难过 0.00，d'=0.64，区间还重叠，不可用），
 *   单点峰值口径下差距立刻出来。
 *
 * ===== 阈值 5 是量出来的 =====
 *
 * 按 2000ms 滑窗（推理口径）统计右手拇指单点峰值：
 *
 *   谢谢 thank_you   208 窗   12.2 ±2.0   中位 12.00
 *   难过 sad         170 窗    4.9 ±8.9   中位  0.00
 *   你好 hello       246 窗   11.1 ±8.6
 *
 * 阈值扫描（≥T 判为"拇指吃了力"）：
 *   T= 3   谢谢 100.0%   难过 25.9%   你好 85.4%
 *   T= 5   谢谢 100.0%   难过 25.9%   你好 85.4%
 *   T= 8   谢谢  97.1%   难过 25.9%   你好 75.6%
 *   T=12   谢谢  62.5%   难过 21.2%   你好 25.6%
 *
 * 取 5：**208 个谢谢窗没有一个低于 5**，所以「峰值 < 5 → 一定不是谢谢」这个方向
 * 在录制数据上零误伤。3 和 5 表现相同，取 5 是为了留一倍余量（3 太贴着噪声底）。
 *
 * ⚠ 反方向弱一些：难过有 25.9% 的窗峰值也 ≥5（画圈时拇指偶尔碰到手指）。
 * 这就是下面那个 top-2 约束存在的原因。
 *
 * ===== 为什么要 top-2 约束 =====
 *
 * 闸门**只在模型自己就在这两个词之间犹豫时**介入（top-1 和 top-2 恰好是这一对）。
 * 模型笃定输出难过、谢谢连前二都排不上的时候不动它 —— 否则那 25.9% 会把本来对的
 * 难过改错。这一条是"宁可不修，也不要把对的改坏"。
 *
 * ===== 这条闸门有一个没验证的前提 =====
 *
 * 上面所有数字都来自**录制数据**，而模型对录制数据本来就全对。也就是说这些数字
 * 证明了阈值在录制数据上可分，**没有**证明闸门能修好实时的问题 —— 前提是
 * "实时打的难过，拇指峰值和录制里一样低"。翻译页上有实时读数（`thumbPeak`），
 * 握拳/点拇指时能直接看见这个数，不对就改 `THUMB_PEAK_GATE` 一个常量。
 */
import { SEQ_SENSOR_N } from "./datasetStore";
import type { SequenceSample } from "./datasetStore";
import { rotationBand, type RotationReading } from "./rotationRate";

/** 右手拇指压力在 137 维物理顺序里的位置（见 sensorMapping.ts 的 RIGHT_HAND_INDEX_MAP） */
export const RH_THUMB_PRESSURE_RANGE = { start: 0, end: 12 } as const;

/**
 * 拇指"吃了力"的单点峰值阈值。
 *
 * 5 = 谢谢 208 窗全部命中、难过 170 窗只有 25.9% 命中。改这个常量前先看翻译页上的
 * 实时读数，别凭感觉调 —— 往下调会让难过更容易被认成谢谢，往上调会让谢谢丢掉。
 */
export const THUMB_PEAK_GATE = 5;

/** 闸门仲裁的那一对词。写成常量是为了让 `isGatedPair` 和测试共用同一个定义 */
export const GATED_PAIR: readonly [string, string] = ["thank_you", "sad"];

/**
 * 一个滑窗里右手拇指 12 个压力点的**单点峰值**（0-255）。
 *
 * 传进来的应该是**归一化到右手口径之后**的样本（`normalizeHandedness` 的输出）：
 * 左手的 137 维指序与右手相反，镜像之前读 0-11 会读到小拇指。
 *
 * 没有右手数据时返回 -1 而不是 0 —— 0 是合法读数（拇指没吃力），
 * 用它当"没数据"会让闸门在丢手的时候误判成"一定不是谢谢"。
 */
export function thumbPeak(sample: SequenceSample): number {
  return thumbPeakSpan(sample, 0, 1);
}

/**
 * 只看样本里 `[from01, to01)` 这一段时间的拇指峰值（归一化位置，0 = 首帧、1 = 末帧）。
 *
 * 句子路径要它:一句话里每个词各占一小段,闸门必须只看**那个词**的那几帧。
 * 拿整句一个峰值去判是错的 —— 实测「好看」全段拇指峰值中位 26,一句
 * 「你真好看」里只要解出「难过」,全局峰值就会把它误翻成「谢谢」。
 *
 * 区间空或越界时夹到至少一帧,**不返回 -1** —— -1 的含义是"没有右手数据",
 * 与"这一段太短"是两件事,混起来会让闸门在正常样本上静默失效。
 */
export function thumbPeakSpan(
  sample: SequenceSample,
  from01: number,
  to01: number
): number {
  const s = sample.rightSensor;
  if (!s) return -1;
  const n = sample.frameCount;
  if (n <= 0) return -1;
  let a = Math.floor(Math.max(0, Math.min(1, from01)) * n);
  let b = Math.ceil(Math.max(0, Math.min(1, to01)) * n);
  a = Math.max(0, Math.min(a, n - 1));
  b = Math.max(a + 1, Math.min(b, n));
  const { start, end } = RH_THUMB_PRESSURE_RANGE;
  let peak = 0;
  for (let t = a; t < b; t++) {
    const base = t * SEQ_SENSOR_N;
    for (let k = start; k < end; k++) {
      const v = s[base + k] ?? 0;
      if (v > peak) peak = v;
    }
  }
  return peak;
}

/** top-1 / top-2 恰好是被仲裁的那一对吗（顺序无关） */
export function isGatedPair(top1: string, top2: string | undefined): boolean {
  if (!top2) return false;
  const [a, b] = GATED_PAIR;
  return (
    (top1 === a && top2 === b) || (top1 === b && top2 === a)
  );
}

export interface GateResult {
  /** 修正后的标签（不介入时等于 top1） */
  label: string;
  /** 介入了吗 */
  changed: boolean;
  /** 为什么介入 / 为什么不介入 —— 调试读数用，别拿它做逻辑判断 */
  reason:
    | "not_gated_pair"
    | "no_right_hand"
    /** 转角速率 ≥ ROT_RATE_HI：手在转 → 一定不是谢谢。这一侧实测 0/193 误伤 */
    | "rotating_so_not_thanks"
    /** 转角速率 < ROT_RATE_LO：手没在转 → 不可能是画圈的难过 */
    | "still_so_thanks"
    | "thumb_idle_so_not_thanks"
    | "thumb_loaded_so_thanks"
    | "consistent";
}

/**
 * 在 谢谢/难过 之间仲裁：**先看转角速率，它定不了才看拇指峰值。**
 *
 * @param top1  模型的第一名标签
 * @param top2  模型的第二名标签（`allProbabilities[1].label`）
 * @param peak  `thumbPeak()` 的结果；-1 = 没有右手数据
 * @param gate  拇指阈值
 * @param rot   `rotationRate()` 的结果。传 null / `usable: false` 时**整条退回**
 *              到原来的纯拇指行为（区间太短、没有 IMU 的手套都走这一支）——
 *              这是刻意的：短区间下速率判据实测比拇指还差，见 rotationRate.ts。
 */
export function applyFistGate(
  top1: string,
  top2: string | undefined,
  peak: number,
  gate: number = THUMB_PEAK_GATE,
  rot?: RotationReading | null
): GateResult {
  if (!isGatedPair(top1, top2))
    return { label: top1, changed: false, reason: "not_gated_pair" };

  /*
   * 转角判据先走。它比拇指强，而且**不需要右手压力数据** —— 所以放在
   * `peak < 0` 那道 return 之前：压力阵列坏掉/丢手的时候，只要 IMU 还在，
   * 这一刀仍然判得出来。反过来放就等于把更可靠的判据挡在门外。
   */
  const band = rotationBand(rot);
  if (band === "rotating")
    return top1 === "thank_you"
      ? { label: "sad", changed: true, reason: "rotating_so_not_thanks" }
      : { label: top1, changed: false, reason: "rotating_so_not_thanks" };
  if (band === "still")
    return top1 === "sad"
      ? { label: "thank_you", changed: true, reason: "still_so_thanks" }
      : { label: top1, changed: false, reason: "still_so_thanks" };

  if (peak < 0) return { label: top1, changed: false, reason: "no_right_hand" };

  const loaded = peak >= gate;
  if (top1 === "thank_you" && !loaded)
    return {
      label: "sad",
      changed: true,
      reason: "thumb_idle_so_not_thanks",
    };
  if (top1 === "sad" && loaded)
    return {
      label: "thank_you",
      changed: true,
      reason: "thumb_loaded_so_thanks",
    };
  return { label: top1, changed: false, reason: "consistent" };
}
