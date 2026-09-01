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
 * **第三层 · 触觉静止段**（`leadingStillMs` / `trailingStillMs`）。前两层都是视觉的，
 * 于是漏了两类样本，而这两类都不是少数：
 *
 * - **手一开始就举在画面里等着**。可见性判 `full_span`，速度谷底判 `no_rise`
 *   （没有向上的冲程可找），于是一刀没裁 —— 而"举着等待"那一段照样该切。
 *   实测 364 条里 159 条落在这里（08-17 批 242 条里 167 条），`datasetAudit` 早就
 *   量过并写明「`full_span` 只意味着没找到可裁的地方，**不等于开头没有静止段**」。
 * - **没开摄像头录的那批**。两层都不成立，`no_vision` 直接返回。
 *
 * 第三层判的是**静止**（能量低于 `IDLE_ENERGY`），不是"动起来了" —— 与上面那条
 * "不能用运动能量"不矛盾：抬手途中能量高，所以它天然不会把 transport 段当静止，
 * 它只会切掉真正没人在动的那一头。判据整套照搬 `dominantHand.sampleEnergies`
 * （弯折 σ 0.55 + 指压 σ 0.25 + 转角 0.2）+ `IDLE_ENERGY`，那个门限是在 399 条真实
 * 录制上校准过的；**不要另立一套门限**。
 *
 * 第三层同时是唯一会裁**尾巴**的一层：前两层的 `end` 永远是"最后一个可见帧 + padMs"，
 * 手不出画就是整条，所以"打完了还举着等按键"那一段此前 100% 留在训练数据里。
 * 而推理端（`sentenceEnvelope.ts`）是把尾部 `SETTLE_MS` 的静止掐掉的 —— 不裁尾
 * 就等于训练和推理两种时间口径。
 *
 * 第三层需要弯折两点标定（`cfg.ranges`）才能跑。**没给就不跑**，行为与加这一层之前
 * 逐位相同（`tactile.ran === false`）—— 判不出来就不裁，见下面第一条守则。
 *
 * 三条守则：
 * - **判不出来就不裁**（无视觉 / 没有够长的可见段 / 裁完太短），宁可留着起手段，
 *   也不要静默切错。每一种不裁的情形都有 `reason`，训练时汇总显示。
 * - **不写数据库**。这里只影响特征构建，`SequenceSample.segments` 一个字节都不动，
 *   所以存量样本不用重录、判据改了重训一次即可，也随时能整体关掉（`trim: null`）。
 * - **推理端显式不裁**。`predictSequence` 拿到的滑窗是纯触觉，判据本来就不成立；
 *   那里显式传 `trim: null` 是为了写明这件事。这条守则**已经兑现过一次**：
 *   `sequenceWindow` 后来加了视觉通道（句子采集页要用），推理端靠 `vision: false`
 *   保持 landmarks 为 null，而即使那个默认值哪天被改错，`trim: null` 仍然拦着
 *   推理行为跟着变。两道闸门是有意的。
 *
 * ⚠ **句子样本只能跑第一、三层**，第二层假设"一条录制里只有一个手势"，
 * 对句子会把第一个词整个切掉。分流在 `trimSpanForExport` 里，见那儿的注释。
 */
import type { SeqTrimSpan, SequenceSample } from "@/lib/datasetStore";
import {
  isSentenceSample,
  SEQ_LANDMARK_N,
  SEQ_SENSOR_N,
} from "@/lib/datasetStore";
import {
  IDLE_ENERGY,
  sampleEnergies,
  type BendRanges,
} from "@/lib/dominantHand";

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
  /**
   * 第一层怎么把多个可见段合成一个保留区间。
   *
   * - `"longest_run"`：只留最长的那一段（**孤立词**）。词录制 1~2 秒、手全程在
   *   画面里，中途"看不见"基本只有误检，取最长最稳。
   * - `"first_to_last"`：第一段的起点 → 最后一段的终点，**中间的空洞一律保留**
   *   （**句子**）。句子 4~7 秒，MediaPipe 掉一次手是常态；取最长会把较短那半
   *   整段丢掉 —— 包括第一个词。实测 45 条真实句里 13 条被打断成 2~4 段，
   *   最坏的一条 301 个可见帧只保留了 96 个。
   *
   * 两个模式都先过 `minRunFrames` 过滤，所以单帧误检不会被拿来定起点/终点 ——
   * 那正是当初"取最长而不是第一段"想防的事，过滤已经防住了。
   */
  visibleSpan: "longest_run" | "first_to_last";
  /**
   * 第三层（触觉静止段）要用的弯折两点标定。**不给就不跑第三层**。
   *
   * 为什么是"不给就不跑"而不是"走兜底量程"：`sampleEnergies` 的兜底量程
   * （`BEND_SPAN_FALLBACK`）会让能量估高，配上在真实标定下校准的 `IDLE_ENERGY`
   * 门限，判出来的静止段会系统性偏短 —— 那是"看起来裁了、其实没裁到点上"，
   * 比不裁更难发现。宁可整层不跑（`tactile.ran === false`，界面上看得见）。
   *
   * 页面侧用 `currentBendRanges()`（`TrainSequence.tsx`）取。
   */
  ranges?: BendRanges | null;
}

export const DEFAULT_TRIM: TrimConfig = {
  minRunFrames: 3,
  padMs: 100,
  minKeptFrames: 6,
  minKeptMs: 250,
  arrival: DEFAULT_ARRIVAL,
  // 默认是词的口径。句子由 `trimSpanForExport` 显式改成 `first_to_last` ——
  // 默认值留在词上，是为了让既有的词训练/体检路径逐位不变
  visibleSpan: "longest_run",
  // 默认不带标定 → 第三层默认不跑。调用方（训练 / 导出 / 体检）显式传 ranges。
  // 这样 `DEFAULT_TRIM` 本身仍是纯函数式的默认值，不隐含"去读 localStorage"
  ranges: null,
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

/** 第三层（触觉静止段）的测量与采用情况 */
export interface TactileInfo {
  /** 判据有没有运行。false = 没给 `cfg.ranges`，这一层整层跳过 */
  ran: boolean;
  /** 量到的头部静止时长（ms）。原始测量值，未必被采用 */
  leadingMs: number;
  /** 量到的尾部静止时长（ms） */
  trailingMs: number;
  /** 起点实际被这一层往后推了多少帧（0 = 没推动） */
  headFrames: number;
  /** 终点实际被这一层往前收了多少帧 */
  tailFrames: number;
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
  /**
   * 第一层数到的可见段个数（已过 `minRunFrames`）。
   *
   * 这是"摄像头中途掉手"唯一的可观测量：`keptRatio` 低既可能是录制两头等太久、
   * 也可能是掉手把中间劈开后丢了一半，只有这个数能把两者分开。>1 时
   * `longest_run` 模式必然扔掉了可见的动作。
   *
   * 没有关键点（`no_vision`）或 T<2 提前退出时为 0。
   */
  visibleRunCount: number;
  /** 第二层到位检测的结果；T<2 就退出时为 null */
  arrival: ArrivalInfo | null;
  /** 第三层触觉静止段；T<2 就退出时为 null */
  tactile: TactileInfo | null;
}

const NO_TACTILE: TactileInfo = {
  ran: false,
  leadingMs: 0,
  trailingMs: 0,
  headFrames: 0,
  tailFrames: 0,
};

function fullSpan(
  sample: SequenceSample,
  reason: TrimReason,
  arrival: ArrivalInfo | null = null,
  tactile: TactileInfo | null = null,
  visibleRunCount = 0
): TrimSpan {
  return {
    startFrame: 0,
    endFrame: sample.frameCount,
    applied: false,
    reason,
    keptRatio: 1,
    visibleRunCount,
    arrival,
    tactile,
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

/** 一段连续可见帧。`end` 不含，与 `TrimSpan` 同口径 */
export interface VisibleRun {
  start: number;
  end: number;
}

export interface VisibleRuns {
  /** 长度 ≥ `minRunFrames` 的可见段，按时间先后排列 */
  runs: VisibleRun[];
  /** 可见帧总数。**含**被 `minRunFrames` 滤掉的那些，所以可能大于 runs 覆盖的帧数 */
  visibleFrames: number;
  totalFrames: number;
}

/**
 * 扫出"手在画面里"的各段。
 *
 * 可见性取**双手的并集**：任一只手看得见就算已经入画。取交集会在单手词上永远为假
 * （另一只手整条都不在画面里），那样等于关掉裁剪。
 *
 * 抽成导出函数是为了"一份实现、两处消费"：第一层裁剪用它定区间，界面上的
 * 裁剪条用它画掉手的位置。两边数出来的段必须是同一批，否则界面会显示
 * "看着没问题"而裁剪其实扔了东西 —— 那正是这套判据出过的事故。
 */
export function visibleRuns(
  sample: SequenceSample,
  minRunFrames: number = DEFAULT_TRIM.minRunFrames
): VisibleRuns {
  const T = sample.frameCount;
  const runs: VisibleRun[] = [];
  let visibleFrames = 0;
  let runStart = -1;
  // t === T 是收尾的哨兵位，用来关掉最后一段
  for (let t = 0; t <= T; t++) {
    const visible =
      t < T &&
      (handVisibleAt(sample.leftLandmarks, t) ||
        handVisibleAt(sample.rightLandmarks, t));
    if (visible) {
      visibleFrames++;
      if (runStart < 0) runStart = t;
    } else if (runStart >= 0) {
      if (t - runStart >= minRunFrames) runs.push({ start: runStart, end: t });
      runStart = -1;
    }
  }
  return { runs, visibleFrames, totalFrames: T };
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

// ===== 第三层：触觉静止段 =====

/**
 * 找静止段时前缀/尾段的最小长度，同时也是扫描步长（ms）。
 *
 * 这也是这项测量的分辨率。不能取太小：能量里的弯折那一路是窗口内的 σ，
 * 窗口太短时 σ 只反映几帧噪声，判不出"动了没有"。200ms @ 50Hz = 10 帧，
 * 对"头段是 0 还是 800ms"这个量级的问题够用了。
 *
 * （原先在 `datasetAudit.ts` 里，现在裁剪判据本身要用，移到这里；
 * `datasetAudit` 改为从这里 import —— 两份会漂。）
 */
export const ONSET_PROBE_MS = 200;

/**
 * 这条录制的一头有多少毫秒没人在动 —— 纯触觉判据，与视觉那两层独立。
 *
 * 判据用"整段前缀/后缀的能量"而不是"局部小窗的能量"：`IDLE_ENERGY` 是在整段尺度上
 * 校准的（399 条真实录制的主手判定），拿它去卡一个 200ms 小窗会偏严 —— 小窗里的 σ
 * 天然比整段小，会把动作的头几百毫秒也算成静止。
 *
 * 两只手取**较大**者：单手词只有一只手在动，用平均会把静止段算长。
 *
 * @param fromEnd true = 从尾往前扫（尾部静止），false = 从头往后扫
 */
function stillMs(
  sample: SequenceSample,
  ranges: BendRanges,
  fromEnd: boolean
): number {
  const T = sample.frameCount;
  if (T < 4 || !(sample.durationMs > 0)) return 0;
  const dt = sample.durationMs / (T - 1);
  const step = Math.max(3, Math.round(ONSET_PROBE_MS / dt));

  let still = 0;
  for (let n = step; n <= T; n += step) {
    const { left, right } = fromEnd
      ? sampleEnergies(sample, ranges, T - n, n)
      : sampleEnergies(sample, ranges, 0, n);
    const peak = Math.max(left?.total ?? 0, right?.total ?? 0);
    // 这一段整体还在静止门限下 → 动作至少要到 n 帧之后（之前）才开始（结束）
    if (peak >= IDLE_ENERGY) return still * dt;
    still = n;
  }
  // 扫到头都没超过门限：整条录制都没有动作（空录）。交给上层的 minKept 守则去否掉
  return sample.durationMs;
}

/** 录制开头连续静止的毫秒数。手举着等按键那一段就是它 */
export function leadingStillMs(
  sample: SequenceSample,
  ranges: BendRanges
): number {
  return stillMs(sample, ranges, false);
}

/**
 * 录制末尾连续静止的毫秒数。
 *
 * 这一项没有任何视觉替代品：可见性的 `end` 是"最后一个可见帧"，手不出画就是整条。
 * 推理端把尾部 `SETTLE_MS`(800ms) 的静止掐掉（`sentenceEnvelope.ts`），
 * 训练不裁尾就是两种时间口径。
 */
export function trailingStillMs(
  sample: SequenceSample,
  ranges: BendRanges
): number {
  return stillMs(sample, ranges, true);
}

/**
 * 找出"手在画面里"的那一段，再在里面找真正的起止。
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

  const ts = sample.timestamps;
  const ranges = cfg.ranges ?? null;

  // ===== 第一层：可见性 =====
  // 判不出来时**不再直接返回** —— 第三层（触觉）在没有视觉的样本上照样能跑，
  // 那正是 datasetAudit 里「给无视觉样本补一套纯触觉的裁剪判据」这条 flag 说的事
  let visionReason: "ok" | "no_vision" | "no_run" = "ok";
  let start = 0;
  let end = T;
  let visibleRunCount = 0;

  if (!sample.leftLandmarks && !sample.rightLandmarks) {
    visionReason = "no_vision";
  } else {
    const vis = visibleRuns(sample, cfg.minRunFrames);
    visibleRunCount = vis.runs.length;
    if (vis.runs.length === 0) {
      visionReason = "no_run";
    } else {
      let runFirst: number;
      let runLast: number;
      if (cfg.visibleSpan === "first_to_last") {
        // 中间的空洞留在区间里。掉手期间手在干什么不知道，但**它一定在打手语** ——
        // 扔掉等于扔掉词；留着最多是给 CTC 多几帧噪声，而 blank 本来就要建模停顿
        runFirst = vis.runs[0].start;
        runLast = vis.runs[vis.runs.length - 1].end - 1;
      } else {
        // 最长段。等长时取靠前的那段（与过去逐位一致）
        let best = vis.runs[0];
        for (const r of vis.runs) {
          if (r.end - r.start > best.end - best.start) best = r;
        }
        runFirst = best.start;
        runLast = best.end - 1;
      }
      // 向外留白：把入画前/出画后 padMs 以内的帧也收进来
      start = runFirst;
      while (start > 0 && ts[runFirst] - ts[start - 1] <= cfg.padMs) start--;
      end = runLast + 1; // exclusive
      while (end < T && ts[end] - ts[runLast] <= cfg.padMs) end++;
    }
  }

  // ===== 第二层：可见段内部再找真正的起点 =====
  // **不能因为可见性没裁到东西就跳过** —— 手已经在画面里、只是还在往上抬，
  // 恰恰是这一层唯一能处理的情形（可见性会判 full_span）。
  let arrival: ArrivalInfo;
  if (!cfg.arrival) {
    arrival = noArrival(start, "disabled");
  } else if (visionReason !== "ok") {
    arrival = noArrival(start, "no_landmarks");
  } else {
    arrival = detectArrival(sample, start, end, cfg.arrival);
  }
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

  // ===== 第三层：触觉静止段 =====
  //
  // 门票不是"传了 ranges 这个对象"，而是"这一条的量程真的齐" ——
  // `{}` 或只标了一只手时 `sampleEnergies` 会退到 `BEND_SPAN_FALLBACK`，
  // 那个兜底量程把能量估高，配上在真实标定下校准的 `IDLE_ENERGY`，
  // 判出来的静止段会系统性偏短：**看起来裁了、其实没裁到点上**，比不裁更难发现。
  // `calibrated` 就是 `sampleEnergies` 自己那条量程对称性守则的输出，直接问它
  const tactile: TactileInfo = { ...NO_TACTILE };
  if (ranges && sampleEnergies(sample, ranges).calibrated) {
    tactile.ran = true;
    tactile.leadingMs = leadingStillMs(sample, ranges);
    tactile.trailingMs = trailingStillMs(sample, ranges);

    // 头部：只在第二层**没有**给出结论时才用。速度谷底是更精确的估计，而且
    // `bendOnsetFrame` 已经在那一层做过"手型先成形就把起点拉回"的保守修正 ——
    // 再用触觉往后推会把那次修正抵消掉
    if (arrival.reason !== "applied" && tactile.leadingMs > cfg.padMs) {
      const target = ts[0] + tactile.leadingMs - cfg.padMs;
      let cand = start;
      while (cand < T - 1 && ts[cand] < target) cand++;
      if (cand > start) {
        tactile.headFrames = cand - start;
        start = cand;
      }
    }

    // 尾部：总是试。前两层从来不裁尾，这里没有可能被抵消的结论
    if (tactile.trailingMs > cfg.padMs) {
      const target = ts[T - 1] - tactile.trailingMs + cfg.padMs;
      let cand = end;
      while (cand > 1 && ts[cand - 1] > target) cand--;
      if (cand < end) {
        tactile.tailFrames = end - cand;
        end = cand;
      }
    }
  }

  // 不裁时的理由：只要**有一层真的跑过**，"没什么可裁"就是一个经过确认的结论
  // （`full_span`）；一层都没跑过才是 no_vision / no_run。datasetAudit 的
  // 「裁剪口径分裂」那条 flag 就是按这个分界线写的
  const skipReason: TrimReason =
    visionReason === "ok" || tactile.ran ? "full_span" : visionReason;

  if (start === 0 && end === T)
    return fullSpan(sample, skipReason, arrival, tactile, visibleRunCount);

  const keptFrames = end - start;
  const keptMs = ts[end - 1] - ts[start];
  if (keptFrames < cfg.minKeptFrames || keptMs < cfg.minKeptMs)
    return fullSpan(sample, "too_short", arrival, tactile, visibleRunCount);

  const totalMs = ts[T - 1] - ts[0];
  return {
    startFrame: start,
    endFrame: end,
    applied: true,
    reason: "applied",
    keptRatio: totalMs > 1e-6 ? keptMs / totalMs : 1,
    visibleRunCount,
    arrival,
    tactile,
  };
}

/**
 * 判出一条录制的 span 并压成导出用的扁平形状（`dataset.json` 的 `trimSpan`）。
 *
 * 这是"一份实现、两处消费"的那一份：浏览器训词模型按它切重采样栅格，
 * Python 合成句子按同一个区间切片。判据要视觉关键点 + `localStorage` 里的弯折标定，
 * 在 Python 重实现一遍就是 `load_dataset.py` 开头明令禁止的双实现。
 *
 * 不传 `ranges` 时第三层不跑，`tactileRan` 为假 —— 那是给下游的信号：
 * 收尾静止还在数据里，别当成已经裁干净。
 *
 * ⚠ **句子样本关掉第二层。** 到位检测（`detectArrival`）的整套判据建立在
 * "这一条里只有一个手势"上：它找的是第一个速度谷底，认定那之前都是抬手 transport。
 * 一条句子录制里那个谷底是**第 1 个词打完的位置** —— 采用它就等于把第一个词整个切掉。
 *
 * 这在句子采集开摄像头之前是空转的（没有关键点 → `no_landmarks`，第二层自动不跑），
 * 开了摄像头就会立刻生效。症状会是"句首照样错"，而这一轮改动的目标恰好是句首 ——
 * 那时候几乎不可能想到是裁剪把第一个词吃了。
 *
 * ⚠ **第一层也要换口径，理由是同一个。** 我原先在这里写过"第一层只切没看见手的
 * 两头，对句子安全" —— **那是错的**。第一层默认取的是**最长连续可见段**
 * （`visibleSpan: "longest_run"`），不是首尾裁剪：句子 4~7 秒，MediaPipe 中途掉
 * 一次手就把录制劈成两段，较短的那半整段被丢掉。实测 45 条真实句里 13 条被打断成
 * 2~4 段，最坏的一条 301 个可见帧只留下 96 个（前 4 秒的动作全没了）。
 * 所以句子走 `first_to_last`，掉手造成的空洞一律保留。
 *
 * 第三层（触觉静止段）对句子确实是安全的：它只切真正没人动的两头，不假设手势个数。
 */
export function trimSpanForExport(
  sample: SequenceSample,
  ranges?: BendRanges | null
): SeqTrimSpan {
  const sentence = isSentenceSample(sample);
  const span = detectSignSpan(sample, {
    ...DEFAULT_TRIM,
    arrival: sentence ? null : DEFAULT_TRIM.arrival,
    visibleSpan: sentence ? "first_to_last" : "longest_run",
    ranges: ranges ?? null,
  });
  return {
    startFrame: span.startFrame,
    endFrame: span.endFrame,
    applied: span.applied,
    reason: span.reason,
    keptRatio: span.keptRatio,
    tactileRan: !!span.tactile?.ran,
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
  /** 第三层（触觉静止段）跑过的条数。0 = 没传 `cfg.ranges`，整层没跑 */
  tactileRan: number;
  /** 第三层把**起点**往后推了的条数 —— 这些就是前两层完全裁不到的"举着等着" */
  tactileHead: number;
  /** 第三层把**终点**往前收了的条数。前两层从不裁尾，所以这个数只可能来自这一层 */
  tactileTail: number;
  /** 第三层切掉的平均头部 / 尾部时长（ms），按各自生效的条数平均；没生效时为 0 */
  meanTactileHeadMs: number;
  meanTactileTailMs: number;
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
    tactileRan: 0,
    tactileHead: 0,
    tactileTail: 0,
    meanTactileHeadMs: 0,
    meanTactileTailMs: 0,
  };
  let ratioSum = 0;
  let droppedSum = 0;
  let headMsSum = 0;
  let tailMsSum = 0;
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
    // 第三层同理独立统计。它是"最后一道"，且是**唯一**会动尾巴的一层 ——
    // tactileTail 为 0 就意味着收尾静止全都还在训练数据里
    if (span.tactile?.ran) {
      stats.tactileRan++;
      if (span.tactile.headFrames > 0) {
        stats.tactileHead++;
        headMsSum += span.tactile.leadingMs;
      }
      if (span.tactile.tailFrames > 0) {
        stats.tactileTail++;
        tailMsSum += span.tactile.trailingMs;
      }
    }
  }
  if (stats.applied > 0) stats.meanKeptRatio = ratioSum / stats.applied;
  if (stats.arrivalApplied > 0)
    stats.meanArrivalDroppedMs = droppedSum / stats.arrivalApplied;
  if (stats.tactileHead > 0) stats.meanTactileHeadMs = headMsSum / stats.tactileHead;
  if (stats.tactileTail > 0) stats.meanTactileTailMs = tailMsSum / stats.tactileTail;
  return stats;
}
