/*
 * dominantHand —— 从数据判"哪只手在做手语"，取代原来让用户手选主手的做法。
 *
 * ===== 为什么原来说"判不了"，而现在能判 =====
 *
 * `handMirror.ts` 原先的注释是「闲着那只手的槽位有静止数据，从数据上分不出'没戴'和
 * '戴着不动'，所以 auto 在双手都有数据时判不了」。那句话本身没错，但它回答的不是
 * 需要回答的问题：归一化要知道的是**哪只手在做手语**，不是哪只手戴了手套。
 * 而"戴着不动"与"正在做动作"是分得清的 —— 静止的手，弯折通道只在噪声底上抖；
 * 做手语的手，一秒内会走掉大半个标定量程。这个信号一直都在，只是以前没人去取。
 *
 * ===== 取哪三路、为什么这么归一化 =====
 *
 * 全部从推理用的同一个滑窗快照里算（`SequenceWindowBuffer.snapshot`），
 * 不额外开采集 —— 两只手的时序本来就躺在同一个窗口里，且已重采样到公共栅格。
 *
 * 1. **弯折能量**（主力，权重 0.55）：5 路弯折在窗口内的标准差，**除以这一路
 *    两点标定量到的量程**。除以量程这一步是必须的：实测左右手量程差得很多
 *    （左 51/121/138/135/62、右 40/99/176/120/74），不归一化会系统性偏向量程大的手。
 *    用 σ 而不是极差（max−min）：极差被单个尖峰噪声就能抬起来，σ 不会。
 *
 * 2. **指压能量**（权重 0.25）：60 个指压点在窗口内的 σ 的平均。两只手同型号、
 *    同样是 8 位 ADC，所以这一路左右**直接可比**，不需要各自归一化。
 *
 * 3. **朝向能量**（权重 0.20，故意压低）：窗口内相对首帧的**最大**转角。
 *    - 为什么不是累加角路径：那样会把逐帧抖动线性累加进去，帧数越多越大
 *      （`imuHealth.ts:161` 的 `stillRotationDegPerMin` 就栽在这上面）。实测左手 IMU
 *      抖动是右手的 4.5 倍（2621 vs 583 °/分），累加口径下左手会永远"更活跃"。
 *    - 为什么不是首尾净转角：手势常常转出去再转回来，净角接近 0，会漏掉真动作。
 *    - 最大偏离角对非累积噪声只多出 3~4σ，对漂移也被 1.5s 内的量级压住，两头都安全。
 *
 * 每一路都折算成"几份手势能量"，1.0 = 明显是个有意动作，见下面各 *_UNIT 常量。
 *
 * ===== 判定与滞回 =====
 *
 * 比值判定（一只手是另一只的 2.5 倍以上才算主手），比值居中判"双手都在动"= 双手词。
 * **滞回是必须的**：一个词做到一半翻转主手，模型会拿到前半段右手口径、后半段镜像口径
 * 拼起来的输入 —— 那比判错更糟，是训练集里根本不存在的东西。所以要连续
 * `FLIP_WINDOWS` 个窗口给出相反结论才切换。
 *
 * 两只手都低于绝对静止门限时**不重判**，沿用上次结果：手语词之间必然有静止间隙，
 * 在间隙里重判等于让主手在每个词之前随机翻一次。
 */
import { SEQ_IMU_N, SEQ_SENSOR_N, type SequenceSample } from "./datasetStore";
import {
  MIN_USEFUL_SPAN,
  channelSpans,
  type BendRange,
  type HandKey,
} from "./bendRange";
import { mirrorSample, type Dominance } from "./handMirror";

// ===== mapped_data(137 维) 里的分块，见 sensorMapping.ts =====

/** 60 个指压点：每指 12 点 × 5 指 */
const FINGER_PRESSURE_OFFSET = 0;
const FINGER_PRESSURE_N = 60;
/** 5 路弯折。注意这里是**物理顺序**，左手是小指→拇指、右手是拇指→小指 */
const BEND_OFFSET = 60;
const BEND_N = 5;

// ===== 能量折算单位（1.0 = 一份"明显是有意动作"）=====

/** 弯折：窗口内 σ 达到标定量程的这个比例，算一份 */
const BEND_SIGMA_UNIT = 0.1;
/**
 * 没有可用两点标定时的兜底量程（ADC）。取实测量程（40~176）里偏小的一档：
 * 宁可把活动手的能量估高，也不要因为分母太大把它压到静止手底下。
 */
const BEND_SPAN_FALLBACK = 90;
/** 指压：60 点平均 σ 达到这么多 ADC 计数，算一份 */
const PRESSURE_SIGMA_UNIT = 6;
/** 朝向：窗口内相对首帧最大转角达到这么多度，算一份 */
const ORIENT_DEG_UNIT = 30;

export const W_BEND = 0.55;
export const W_PRESSURE = 0.25;
const W_ORIENT = 0.2;

// ===== 判定门限 =====

/** 主手能量至少是另一只手的这么多倍才敢判主手；比值居中 = 双手词 */
export const DOMINANCE_RATIO = 2.5;
/** 两只手能量都低于这个值 = 都没在动，沿用上次判定而不是重判 */
export const IDLE_ENERGY = 0.15;
/** 要连续这么多个窗口给出相反结论，才真的切换主手 */
export const FLIP_WINDOWS = 3;

/** 没有任何数据、也没有历史判定时的兜底。模型是用右手采的数据训的，右手是训练口径本身 */
const DEFAULT_DOMINANCE: Dominance = "right";

export interface HandEnergy {
  /** 弯折能量（份） */
  bend: number;
  /** 指压能量（份） */
  pressure: number;
  /** 朝向能量（份） */
  orient: number;
  /** 加权总能量（份）。>= 1 基本可以认为这只手在做动作 */
  total: number;
}

export type DominanceReason =
  /** 只有左手在出数据 */
  | "only_left"
  /** 只有右手在出数据 */
  | "only_right"
  /** 两只手都没数据 */
  | "no_hand"
  /** 窗口还没攒满，暂时按"连了哪只手套"顶着 */
  | "warmup"
  /** 靠能量比值判出来的 */
  | "energy"
  /** 两只手能量接近，判为双手词 */
  | "both_active"
  /** 两只手都静止，沿用上次判定 */
  | "idle_hold"
  /** 已看到相反结论但还没连续够 FLIP_WINDOWS 个窗口，仍沿用上次判定 */
  | "pending_flip";

export interface DominanceVerdict {
  /** 判定结果，直接喂 `normalizeHandedness` */
  dominant: Dominance;
  /** 左/右手这一窗的能量；这只手没数据时为 null */
  left: HandEnergy | null;
  right: HandEnergy | null;
  reason: DominanceReason;
  /**
   * 弯折能量是否用了真实标定量程。false = 有手没做两点标定，走了兜底量程，
   * 判定仍然可用但左右量程差异没被抵消，界面该提示回第 1 步补标定。
   */
  calibrated: boolean;
}

// ===== 能量计算 =====

/** 一段等距取样序列的标准差（两趟，uint8 输入下够稳） */
function stdOf(
  data: ArrayLike<number>,
  offset: number,
  stride: number,
  count: number
): number {
  if (count < 2) return 0;
  let sum = 0;
  for (let i = 0; i < count; i++) sum += data[offset + i * stride];
  const mean = sum / count;
  let acc = 0;
  for (let i = 0; i < count; i++) {
    const d = data[offset + i * stride] - mean;
    acc += d * d;
  }
  return Math.sqrt(acc / (count - 1));
}

/** 两个四元数（[w,x,y,z]，已归一化）之间的转角，度 */
function quatAngleDeg(
  imu: Float32Array,
  ia: number,
  ib: number
): number {
  const d =
    imu[ia] * imu[ib] +
    imu[ia + 1] * imu[ib + 1] +
    imu[ia + 2] * imu[ib + 2] +
    imu[ia + 3] * imu[ib + 3];
  // |dot|：q 与 −q 是同一个旋转，不取绝对值会把 0° 判成 180°
  const c = Math.min(1, Math.abs(d));
  return (2 * Math.acos(c) * 180) / Math.PI;
}

/**
 * 一只手在一个窗口内的运动能量。
 *
 * @param hand 用来把物理顺序的弯折下标折回 canonical（左手是反的）
 * @param spans 5 路 canonical 量程；传 null 或某一路不足 `MIN_USEFUL_SPAN` 时走兜底量程
 * @param startFrame 从第几帧开始量，`frameCount` 是**从这一帧起算的帧数**。
 *   默认 0 = 量整段。给"这一段里的哪个位置开始有动作"这类扫描用
 *   （`datasetAudit` 找静止头段），朝向能量的参考帧跟着一起挪到 `startFrame`。
 */
export function handEnergy(
  sensor: Uint8Array | null,
  imu: Float32Array | null,
  frameCount: number,
  hand: HandKey,
  spans: number[] | null,
  startFrame = 0
): HandEnergy | null {
  if (!sensor || frameCount < 2) return null;
  const so = startFrame * SEQ_SENSOR_N;
  const io = startFrame * SEQ_IMU_N;

  let bendAcc = 0;
  for (let j = 0; j < BEND_N; j++) {
    // 物理槽位 j → canonical 指序：左手 LH_BEND 是小指→拇指，反过来才是拇指→小指
    const canonical = hand === "LH" ? BEND_N - 1 - j : j;
    const s = spans?.[canonical];
    const span =
      typeof s === "number" && s >= MIN_USEFUL_SPAN ? s : BEND_SPAN_FALLBACK;
    const sigma = stdOf(sensor, so + BEND_OFFSET + j, SEQ_SENSOR_N, frameCount);
    bendAcc += sigma / span;
  }
  const bend = bendAcc / BEND_N / BEND_SIGMA_UNIT;

  let pressAcc = 0;
  for (let j = 0; j < FINGER_PRESSURE_N; j++) {
    pressAcc += stdOf(
      sensor,
      so + FINGER_PRESSURE_OFFSET + j,
      SEQ_SENSOR_N,
      frameCount
    );
  }
  const pressure = pressAcc / FINGER_PRESSURE_N / PRESSURE_SIGMA_UNIT;

  let maxDeg = 0;
  if (imu && imu.length >= (startFrame + frameCount) * SEQ_IMU_N) {
    for (let t = 1; t < frameCount; t++) {
      const a = quatAngleDeg(imu, io, io + t * SEQ_IMU_N);
      if (a > maxDeg) maxDeg = a;
    }
  }
  const orient = maxDeg / ORIENT_DEG_UNIT;

  return {
    bend,
    pressure,
    orient,
    total: W_BEND * bend + W_PRESSURE * pressure + W_ORIENT * orient,
  };
}

/** 取一只手可用的 canonical 量程；标定缺失或未标定返回 null */
function spansOf(range: BendRange | null | undefined): number[] | null {
  if (!range) return null;
  try {
    return channelSpans(range);
  } catch {
    return null;
  }
}

export interface BendRanges {
  LH?: BendRange | null;
  RH?: BendRange | null;
}

export interface WindowEnergy {
  /** 这一窗左手的运动能量；左手没数据时 null */
  left: HandEnergy | null;
  right: HandEnergy | null;
  /**
   * 弯折能量是否用上了**两只手都齐**的真实两点标定。
   * false 时判定仍然可用，但左右量程差异没被抵消（走了兜底量程）。
   */
  calibrated: boolean;
}

/**
 * 量一条样本（或一个滑窗快照）里两只手各自的运动能量。
 *
 * 抽出来是因为这段有三个调用方 —— 流式主手判定、离线主手判定、推理前的动作闸门
 * （`motionGate.ts`）—— 而其中最容易被改坏的是下面那条**量程对称性守则**：
 *
 * 只有两只手的标定都齐时才用真实量程；只有一只手有标定的话，一只手除真量程、
 * 另一只除兜底量程（`BEND_SPAN_FALLBACK`），比值里就掺进了一个纯人为的系统偏差 ——
 * 那会让"哪只手在动"这个判定偏向恰好做过标定的那只手。三份拷贝里漏改一份，
 * 症状是"某台机器上主手总判错"，极难查。
 *
 * @param startFrame 子段起始帧，默认 0
 * @param count 子段帧数，默认到样本末尾。给"这一段的哪个位置开始有动作"
 *   这类扫描用（`datasetAudit` 找静止头段）
 */
export function sampleEnergies(
  sample: SequenceSample,
  ranges: BendRanges,
  startFrame = 0,
  count?: number
): WindowEnergy {
  const T = Math.min(
    count ?? sample.frameCount,
    sample.frameCount - startFrame
  );
  const lSpans = spansOf(ranges.LH);
  const rSpans = spansOf(ranges.RH);
  const hasL = !!sample.leftSensor;
  const hasR = !!sample.rightSensor;
  const calibrated =
    (!hasL || !!lSpans) && (!hasR || !!rSpans) && (hasL || hasR);
  return {
    left: handEnergy(
      sample.leftSensor,
      sample.leftImu,
      T,
      "LH",
      calibrated ? lSpans : null,
      startFrame
    ),
    right: handEnergy(
      sample.rightSensor,
      sample.rightImu,
      T,
      "RH",
      calibrated ? rSpans : null,
      startFrame
    ),
    calibrated,
  };
}

// ===== 带滞回的判定器 =====

export class DominanceTracker {
  /** 已生效的判定；null = 还没判过任何一次（此时首个结论立即生效，不走滞回） */
  private currentDominance: Dominance | null = null;
  private pending: Dominance | null = null;
  private pendingCount = 0;

  /** 当前生效的主手；一次都没判过时给训练口径（右手） */
  current(): Dominance {
    return this.currentDominance ?? DEFAULT_DOMINANCE;
  }

  reset(): void {
    this.currentDominance = null;
    this.pending = null;
    this.pendingCount = 0;
  }

  /**
   * 用一个窗口快照更新判定。
   *
   * @param sample 滑窗快照；`null` 表示窗口还没攒满 —— 此时退回"按连了哪只手套判"，
   *   也就是改造前 `auto` 的老行为，不去猜。
   * @param connected 两只手套的连接状态，`sample` 为 null 时的唯一依据
   */
  update(
    sample: SequenceSample | null,
    ranges: BendRanges,
    connected: { left: boolean; right: boolean }
  ): DominanceVerdict {
    if (!sample) {
      if (!connected.left && !connected.right) {
        return this.settle(null, "no_hand", null, null);
      }
      if (connected.left && !connected.right)
        return this.settle("left", "only_left", null, null);
      if (connected.right && !connected.left)
        return this.settle("right", "only_right", null, null);
      // 双手都连着但窗口没满：不猜，先顶着上次判定
      return this.hold("warmup", null, null);
    }

    const { left, right, calibrated: symmetric } = sampleEnergies(sample, ranges);

    if (!left && !right) return this.settle(null, "no_hand", null, null);
    if (left && !right) return this.settle("left", "only_left", left, null);
    if (right && !left) return this.settle("right", "only_right", null, right);
    // 上面四个分支已排掉所有空的组合
    const l = left as HandEnergy;
    const r = right as HandEnergy;

    // 词与词之间必然有静止间隙。在间隙里重判 = 每个词之前随机翻一次主手
    if (l.total < IDLE_ENERGY && r.total < IDLE_ENERGY) {
      return this.hold("idle_hold", l, r, symmetric);
    }

    let instant: Dominance;
    let reason: DominanceReason;
    if (r.total <= 0) {
      instant = "left";
      reason = "energy";
    } else if (l.total <= 0) {
      instant = "right";
      reason = "energy";
    } else {
      const ratio = l.total / r.total;
      if (ratio >= DOMINANCE_RATIO) {
        instant = "left";
        reason = "energy";
      } else if (ratio <= 1 / DOMINANCE_RATIO) {
        instant = "right";
        reason = "energy";
      } else {
        instant = "both";
        reason = "both_active";
      }
    }

    // 第一次判定立即生效：没有"上次"可以沿用，滞回在这里只会让首个词判错
    if (this.currentDominance === null) {
      return this.settle(instant, reason, l, r, symmetric);
    }
    if (instant === this.currentDominance) {
      this.pending = null;
      this.pendingCount = 0;
      return this.verdict(this.currentDominance, reason, l, r, symmetric);
    }
    // 滞回：一个词做到一半翻转主手，会拼出一个训练集里不存在的输入
    if (this.pending === instant) {
      this.pendingCount++;
    } else {
      this.pending = instant;
      this.pendingCount = 1;
    }
    if (this.pendingCount >= FLIP_WINDOWS) {
      return this.settle(instant, reason, l, r, symmetric);
    }
    return this.verdict(this.currentDominance, "pending_flip", l, r, symmetric);
  }

  /** 采纳一个结论并清掉待切换状态 */
  private settle(
    dominant: Dominance | null,
    reason: DominanceReason,
    left: HandEnergy | null,
    right: HandEnergy | null,
    calibrated = false
  ): DominanceVerdict {
    this.pending = null;
    this.pendingCount = 0;
    if (dominant !== null) this.currentDominance = dominant;
    return this.verdict(dominant ?? this.current(), reason, left, right, calibrated);
  }

  /** 维持现状，但**不清**待切换计数（静止/预热不该把攒到一半的翻转抹掉） */
  private hold(
    reason: DominanceReason,
    left: HandEnergy | null,
    right: HandEnergy | null,
    calibrated = false
  ): DominanceVerdict {
    return this.verdict(this.current(), reason, left, right, calibrated);
  }

  private verdict(
    dominant: Dominance,
    reason: DominanceReason,
    left: HandEnergy | null,
    right: HandEnergy | null,
    calibrated: boolean
  ): DominanceVerdict {
    return { dominant, left, right, reason, calibrated };
  }
}

// ===== 离线判定：整条已录样本的主手 =====

/**
 * 一条已录样本的主手判定结果。
 *
 * 和 `Dominance` 刻意不是同一个类型：
 * - 没有 `both`。离线镜像的目标是把整个数据集拉到同一个口径，双手词整体镜像之后
 *   仍是同一个词（`mirrorSample` 的文档），所以"两只手都在动"不是不作为的理由，
 *   只要还分得出谁主导就照判。
 * - 多了 `idle` / `no_hand`：`_idle` 伪类样本两只手本来就不动，没有主手可判，
 *   这时**不镜像**（而不是靠噪声 argmax 掷硬币），保证同一份数据每次训练的结果一致。
 */
export type SampleDominance = "left" | "right" | "idle" | "no_hand";

/**
 * 能量比落在 `[1/SAMPLE_TIE_RATIO, SAMPLE_TIE_RATIO]` 内时标 `nearTie`。
 *
 * 这个标记**只上报、不改变判定**：这个带里 argmax 基本由噪声决定，但落在这里的
 * 恰恰是左右近乎对称的双手词 —— 那种词镜像与否对模型看到的东西差别最小，
 * 判错的代价也最小。真正要靠这个数字回答的是"我的判定有多少条是蒙的"，
 * 大量样本落在这里说明该回去看采集方式，而不是在这里加规则。
 */
export const SAMPLE_TIE_RATIO = 1.3;

export interface SampleVerdict {
  dominant: SampleDominance;
  left: HandEnergy | null;
  right: HandEnergy | null;
  /** 两只手能量接近，判定基本是噪声决定的（仍照判，见 `SAMPLE_TIE_RATIO`） */
  nearTie: boolean;
  /** 弯折能量是否用上了两只手都齐的真实两点标定 */
  calibrated: boolean;
}

/**
 * 判一条**已录完**的样本由哪只手主导。
 *
 * 与 `DominanceTracker` 的区别在于这里没有滞回、没有"沿用上次"：离线是整条一起
 * 判、整条一起变换，不存在一个词做到一半翻转口径拼出畸形输入的风险，
 * 那正是流式那边整套滞回机制唯一要防的事。
 *
 * 量程的对称性守则和流式那边一致：只有两只手的标定都齐才用真实量程，
 * 否则两只手一起走兜底量程 —— 一只手除真量程、另一只除兜底量程，
 * 比值里会掺进一个纯人为的系统偏差。
 */
export function judgeSampleDominance(
  sample: SequenceSample,
  ranges: BendRanges
): SampleVerdict {
  const { left, right, calibrated: symmetric } = sampleEnergies(sample, ranges);

  const base = { left, right, nearTie: false, calibrated: symmetric };
  if (!left && !right) return { ...base, dominant: "no_hand" };
  if (left && !right) return { ...base, dominant: "left" };
  if (right && !left) return { ...base, dominant: "right" };

  const l = left as HandEnergy;
  const r = right as HandEnergy;
  // `_idle` 伪类样本落在这里。没有主手可判，就别靠噪声 argmax 掷硬币 ——
  // 同一份数据每次训练都该给出同一个结果
  if (l.total < IDLE_ENERGY && r.total < IDLE_ENERGY)
    return { ...base, dominant: "idle" };

  const nearTie =
    r.total > 0 &&
    l.total / r.total <= SAMPLE_TIE_RATIO &&
    l.total / r.total >= 1 / SAMPLE_TIE_RATIO;
  return {
    ...base,
    nearTie,
    dominant: l.total > r.total ? "left" : "right",
  };
}

export interface HandednessStats {
  total: number;
  /** 判为左手主导 = 被镜像的条数 */
  left: number;
  /** 判为右手主导 = 本来就是训练口径 */
  right: number;
  /** 两只手都低于静止门限（`_idle` 伪类样本应当全落这里），不镜像 */
  idle: number;
  /** 一只手套都没数据，不镜像 */
  noHand: number;
  /** left+right 里判定接近掷硬币的条数，见 `SAMPLE_TIE_RATIO` */
  nearTie: number;
  /** 两只手标定都齐、用上真实量程的条数 */
  calibrated: number;
}

/**
 * 把整个训练集归一化到**右手口径**：逐条判主手，判成左手的整条镜像。
 *
 * 为什么必须做：特征层给两只手各留一段独立槽位（`sequenceFeatures.ts`），
 * 左手做的那条样本写进 `[0,147)`、右手做的写进 `[147,294)`，两组在输入空间里
 * **零重叠**。同一个词一半左手一半右手采时，"信号落在哪半边"与标签完全无关，
 * 就是个纯噪声因子；网络能做的最优反应是把这些维一起忽略、退回类先验 ——
 * 表现出来就是"打什么手语都输出同一个词"。镜像不是数据增广，是把这个
 * 结构性缺口补上：单手词换手不改词义，所以左手样本本来就该按右手口径入库。
 *
 * 判为 `idle` / `no_hand` 的原样带过（镜像它们没有意义，也不该引入随机性）。
 * 推理侧早已是这个口径（`Translate.tsx` 用 `DominanceTracker` + `normalizeHandedness`），
 * 这一步是把训练侧对齐过去。
 */
export function normalizeSamplesToRight(
  samples: SequenceSample[],
  ranges: BendRanges
): { samples: SequenceSample[]; stats: HandednessStats } {
  const stats = emptyHandednessStats(samples.length);
  const out = samples.map((s) => {
    const v = judgeSampleDominance(s, ranges);
    tally(stats, v);
    return v.dominant === "left" ? mirrorSample(s) : s;
  });
  return { samples: out, stats };
}

/**
 * 只判不变换 —— 给界面显示"这批数据里左右手各占多少"用。
 *
 * 这一行是必要的，因为数据集统计里那个「手别分布」量的是**哪只手套在出数据**
 * （`datasetStore.ts` 的 `handCounts` 判的是 sensor/imu 是否为 null），
 * 两只手套都戴着采集时它恒等于"全是双手样本"，左右混采这件事在界面上完全看不见。
 */
export function summarizeHandedness(
  samples: SequenceSample[],
  ranges: BendRanges
): HandednessStats {
  const stats = emptyHandednessStats(samples.length);
  for (const s of samples) tally(stats, judgeSampleDominance(s, ranges));
  return stats;
}

function emptyHandednessStats(total: number): HandednessStats {
  return {
    total,
    left: 0,
    right: 0,
    idle: 0,
    noHand: 0,
    nearTie: 0,
    calibrated: 0,
  };
}

function tally(stats: HandednessStats, v: SampleVerdict): void {
  if (v.calibrated) stats.calibrated++;
  if (v.nearTie) stats.nearTie++;
  switch (v.dominant) {
    case "left":
      stats.left++;
      break;
    case "right":
      stats.right++;
      break;
    case "idle":
      stats.idle++;
      break;
    default:
      stats.noHand++;
  }
}

/** 归一化结果的一句话说明，直接显示在界面上 */
export function describeHandedness(s: HandednessStats): string {
  if (s.total === 0) return "没有样本";
  const parts = [`左手 ${s.left} 条已镜像到右手口径`, `右手 ${s.right} 条原样`];
  if (s.idle) parts.push(`静止 ${s.idle} 条`);
  if (s.noHand) parts.push(`无手套数据 ${s.noHand} 条`);
  if (s.nearTie) parts.push(`其中 ${s.nearTie} 条左右活动量接近、判定不确定`);
  if (s.calibrated < s.total)
    parts.push(`${s.total - s.calibrated} 条缺两点标定，走了兜底量程`);
  return parts.join("，");
}

/** 判定结果的一句话说明，直接显示在界面上 */
export function describeDominance(v: DominanceVerdict): string {
  switch (v.reason) {
    case "no_hand":
      return "没有手套在出数据";
    case "only_left":
      return "只有左手套在出数据，整体镜像到右手口径";
    case "only_right":
      return "只有右手套在出数据，本来就是训练口径";
    case "warmup":
      return "窗口还没攒满，暂按连了哪只手套顶着";
    case "energy":
      return v.dominant === "left"
        ? "左手活动量明显更大，判为主手并镜像"
        : "右手活动量明显更大，判为主手";
    case "both_active":
      return "两只手活动量接近，按双手词处理、不镜像";
    case "idle_hold":
      return "两只手都静止，沿用上次判定";
    case "pending_flip":
      return "检测到主手可能换了，确认中（防止一个词做到一半翻转）";
  }
}
