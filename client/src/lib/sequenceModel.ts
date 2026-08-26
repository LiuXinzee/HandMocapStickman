/*
 * sequenceModel — 时序手语识别模型（TCN 骨干 + 两阶段知识蒸馏）
 *
 * 与 signLanguageModel.ts 的关系：那个是单帧 MLP，看不到运动轨迹，
 * 「再见」（左右摆动）、「来」（招手）这类带轨迹的词天生无解。本模块把整条
 * 序列作为输入，静态词只是"帧间几乎不变"的序列，两类词由同一个模型统一处理
 * ——不做"运动门控 + 双模型路由"，因为门控阈值是不可靠的中间环节：静态手势
 * 摆到位的过程、手抖都有运动，一旦路由错了就是错词，后续平滑救不回来。
 *
 * 训练策略与静态模型一致：
 * - 阶段1：视觉+触觉融合特征（420D/帧）训练教师
 * - 阶段2：仅触觉（294D/帧）的学生做知识蒸馏
 * - 部署：只用学生，输入纯触觉
 *
 * 网络结构刻意做了 backbone / head 分离：
 *   backbone 输出保持 (T', C) 不在时间维塌缩，池化只放在 head 里。
 *   将来做句子级（连续手语 CTC）时换一个逐帧 head 即可复用 backbone。
 *   ——注意 tfjs 4.22 **没有** CTC loss / 解码器（在 dist/tf.js 里搜不到任何
 *   ctc 符号），句子级训练必须走 python_train/ 的 tf.keras 路径。
 */
import * as tf from "@tensorflow/tfjs";
import { isSentenceSample } from "./datasetStore";
import type { SequenceSample, SavedModel } from "./datasetStore";
import { summarizeTrim, type TrimStats } from "@/lib/sequenceTrim";
import {
  normalizeSamplesToRight,
  type BendRanges,
  type HandednessStats,
} from "@/lib/dominantHand";
import { mergeSamples, type MergeGroup } from "@/lib/labelMerge";
import {
  buildSequenceFeatures,
  DEFAULT_AUGMENT,
  NO_AUGMENT,
  SEQ_LEN,
  TACTILE_FRAME_DIM,
  FUSED_FRAME_DIM,
  type SeqAugmentConfig,
} from "./sequenceFeatures";

export type SeqBackbone = "tcn" | "tcn_bigru";

export interface SeqTrainingConfig {
  seqLen: number;
  epochs: number;
  batchSize: number;
  learningRate: number;
  /**
   * 验证集比例。**按源样本（整条录制）分组切**，不是 tf 的 `validationSplit`
   * ——见 `planTrainValSplit`。0 = 不留验证集（那样准确率没意义）。
   */
  validationSplit: number;
  distillationTemp: number;
  distillationAlpha: number;
  backbone: SeqBackbone;
  /** 每条样本额外生成几个增强副本（0 = 只用原始） */
  augmentCopies: number;
  augment: SeqAugmentConfig;
  /**
   * 训练前把左手主导的样本整条镜像到右手口径（`normalizeSamplesToRight`）。
   *
   * 默认开。关掉它只在一种情况下讲得通：确认整个数据集本来就是同一只手采的、
   * 想省掉这一遍判定。左右混采时关掉它 = 同一个词的两半落进两段零重叠的槽位，
   * 网络只能退回类先验，表现是"打什么都输出同一个词"。
   */
  normalizeHandedness: boolean;
  /**
   * 两只手的弯折两点标定，喂给主手判定当分母。
   *
   * 为什么要从外面传：判定要按各自量程归一化（实测左右量程差 40~176，不归一化会
   * 系统性偏向量程大的手），而标定存在 `localStorage` 里，本模块跑在 node 测试环境下
   * 读不到。调用方（`TrainSequence.tsx`）负责 `loadBendRange` 之后传进来。
   * 缺标定不会让判定失效，只是两只手一起走兜底量程，会在归一化汇总里报出来。
   */
  bendRanges: BendRanges;
  /**
   * 把特征层不可分的类合并（我/你/他 → 一类，我们/你们/他们 → 一类）。
   *
   * 默认开。这不是"先凑合"，是当前硬件下的正确建模：这几个词只差指向(yaw)，
   * 而六轴 IMU 没有磁力计、绝对 yaw 不可观测，实测漂移 P90 15.9°/s 已经盖过
   * 45° 的类间距（yawDrift 探针，399 条）。三个类抢同一块特征空间时 softmax
   * 只能按训练集比例乱分，逃逸的概率还会污染邻近词。
   *
   * 关掉是有意义的对照实验（想看合并到底帮了多少）。换成九轴 IMU 后应该重测
   * 漂移再决定要不要关 —— 库里的原始标签一直留着，退得回去。
   */
  mergeDegenerateLabels: boolean;
}

export interface SeqTrainingProgress {
  epoch: number;
  totalEpochs: number;
  loss: number;
  accuracy: number;
  valLoss: number;
  valAccuracy: number;
  phase: "teacher" | "student";
}

export interface PredictionResult {
  label: string;
  confidence: number;
  allProbabilities: Array<{ label: string; probability: number }>;
}

export const DEFAULT_SEQ_CONFIG: SeqTrainingConfig = {
  seqLen: SEQ_LEN,
  epochs: 60,
  batchSize: 16,
  learningRate: 0.001,
  validationSplit: 0.2,
  distillationTemp: 3.0,
  distillationAlpha: 0.5,
  backbone: "tcn",
  augmentCopies: 2,
  augment: DEFAULT_AUGMENT,
  normalizeHandedness: true,
  bendRanges: {},
  mergeDegenerateLabels: true,
};

// ===== 网络结构 =====

/**
 * TCN 骨干：卷积 + 池化堆叠，输出 (T', C)。时间塌缩仍是 head 的职责，
 * 这里的池化只用来扩感受野（见下）。
 *
 * ⚠ **不要在这里用 dilationRate > 1。** tfjs（4.22 实测）的 conv2D
 * **前向支持空洞、反向不支持**：梯度算子里直接断言
 * "dilation rates greater than 1 are not yet supported in gradients"。
 * 于是模型能建、能 predict，一进 fit 就抛。曾经的 b2 d=2 / b3 d=4 就是这么
 * 让整页训练无法启动的（报错先出现在最后一层，即 '1,4'）。
 * 改回空洞前先跑 sequenceModel.test.ts 里那条"反向传播能跑通"的测试。
 *
 * 感受野（按原始帧数算，T=32）：
 *   b1 k=5        → 5    T=32
 *   pool/2        → 6    T=16
 *   b2 k=5        → 14   T=16
 *   pool/2        → 16   T=8
 *   b3 k=5        → 32   T=8
 * 32 帧正好盖满整段，最后一层每个位置都能看到整个动作 —— 与原空洞版（30 帧）等效。
 * 代价是骨干输出的时间分辨率从 T/2 降到 T/4：孤立词后面接 GAP，无影响；
 * 将来句子级 CTC 逐帧输出会粗一档（句子输入本身长得多，比例上仍够，
 * 而且 tfjs 没有 CTC，句子级只能走 python_train/）。
 */
function tcnBackbone(
  input: tf.SymbolicTensor,
  c1: number,
  c2: number,
  backbone: SeqBackbone,
  prefix: string
): tf.SymbolicTensor {
  let x = input;

  const block = (
    t: tf.SymbolicTensor,
    filters: number,
    kernelSize: number,
    name: string
  ): tf.SymbolicTensor => {
    let y = tf.layers
      .conv1d({
        filters,
        kernelSize,
        padding: "same",
        useBias: false,
        kernelInitializer: "heNormal",
        name: `${prefix}_${name}_conv`,
      })
      .apply(t) as tf.SymbolicTensor;
    y = tf.layers
      .batchNormalization({ name: `${prefix}_${name}_bn` })
      .apply(y) as tf.SymbolicTensor;
    y = tf.layers
      .activation({ activation: "relu", name: `${prefix}_${name}_relu` })
      .apply(y) as tf.SymbolicTensor;
    return y;
  };

  x = block(x, c1, 5, "b1");
  x = tf.layers
    .maxPooling1d({ poolSize: 2, name: `${prefix}_pool1` })
    .apply(x) as tf.SymbolicTensor;
  x = block(x, c1, 5, "b2");
  x = tf.layers
    .maxPooling1d({ poolSize: 2, name: `${prefix}_pool2` })
    .apply(x) as tf.SymbolicTensor;
  x = block(x, c2, 5, "b3");

  if (backbone === "tcn_bigru") {
    // 句子阶段主要靠这一层补全局时序依赖；孤立词阶段默认不开（浏览器训练慢）
    x = tf.layers
      .bidirectional({
        layer: tf.layers.gru({
          units: Math.max(8, Math.floor(c2 / 2)),
          returnSequences: true,
        }) as never,
        mergeMode: "concat",
        name: `${prefix}_bigru`,
      })
      .apply(x) as tf.SymbolicTensor;
  }

  return x;
}

/** 孤立词 head：时间维全局平均池化 → Dense → softmax */
function isolatedWordHead(
  x: tf.SymbolicTensor,
  hidden: number,
  numClasses: number,
  dropout: number,
  prefix: string
): tf.SymbolicTensor {
  let y = tf.layers
    .globalAveragePooling1d({ name: `${prefix}_gap` })
    .apply(x) as tf.SymbolicTensor;
  y = tf.layers
    .dense({
      units: hidden,
      activation: "relu",
      kernelInitializer: "heNormal",
      name: `${prefix}_fc`,
    })
    .apply(y) as tf.SymbolicTensor;
  y = tf.layers
    .dropout({ rate: dropout, name: `${prefix}_drop` })
    .apply(y) as tf.SymbolicTensor;
  y = tf.layers
    .dense({ units: numClasses, activation: "softmax", name: `${prefix}_out` })
    .apply(y) as tf.SymbolicTensor;
  return y;
}

function buildSeqModel(
  numClasses: number,
  seqLen: number,
  frameDim: number,
  backbone: SeqBackbone,
  channels: [number, number],
  hidden: number,
  dropout: number,
  lr: number,
  prefix: string
): tf.LayersModel {
  const input = tf.input({ shape: [seqLen, frameDim], name: `${prefix}_in` });
  const feat = tcnBackbone(input, channels[0], channels[1], backbone, prefix);
  const out = isolatedWordHead(feat, hidden, numClasses, dropout, prefix);
  const model = tf.model({ inputs: input, outputs: out, name: prefix });
  model.compile({
    optimizer: tf.train.adam(lr),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export function buildSeqTeacher(
  numClasses: number,
  seqLen: number,
  backbone: SeqBackbone,
  lr: number
): tf.LayersModel {
  return buildSeqModel(
    numClasses,
    seqLen,
    FUSED_FRAME_DIM,
    backbone,
    [128, 256],
    128,
    0.3,
    lr,
    "teacher"
  );
}

export function buildSeqStudent(
  numClasses: number,
  seqLen: number,
  backbone: SeqBackbone,
  lr: number
): tf.LayersModel {
  return buildSeqModel(
    numClasses,
    seqLen,
    TACTILE_FRAME_DIM,
    backbone,
    [64, 128],
    64,
    0.2,
    lr,
    "student"
  );
}

/**
 * 句子级（CTC）head：逐帧输出 `numClasses+1` 维，**末位是 blank**，不做时间塌缩。
 *
 * 与 `python_train/train_seq.py` 的 `frame_wise_head` 一一对应。Dense 作用在最后一维，
 * 3D 输入时等于逐时间步共享同一个全连接 —— keras 和 tfjs 在这点上行为一致。
 */
function frameWiseHead(
  x: tf.SymbolicTensor,
  numClasses: number,
  prefix: string
): tf.SymbolicTensor {
  return tf.layers
    .dense({ units: numClasses + 1, activation: "softmax", name: `${prefix}_out` })
    .apply(x) as tf.SymbolicTensor;
}

/**
 * 搭出与 Python 句子模型**同构**的空网络，供 `sentenceModel.ts` 填权重。
 *
 * 为什么放在这个文件里：结构定义只能有一份。骨干（`tcnBackbone`）本来就在这里，
 * 句子模型只是换了个 head。在别处另搭一套的话，改了池化或通道数只会在
 * 加载权重时报形状不符 —— 而那时候人在排查的是"模型加载失败"，不会想到是两处结构漂了。
 *
 * **不 compile**：这个模型只做推理（CTC loss 在 tfjs 里不存在，训练在 Python 侧）。
 */
export function buildSeqSentenceStudent(
  numClasses: number,
  seqLen: number,
  backbone: SeqBackbone
): tf.LayersModel {
  const input = tf.input({
    shape: [seqLen, TACTILE_FRAME_DIM],
    name: "student_in",
  });
  const feat = tcnBackbone(input, 64, 128, backbone, "student");
  const out = frameWiseHead(feat, numClasses, "student");
  return tf.model({ inputs: input, outputs: out, name: "student" });
}

// ===== 数据准备 =====

interface PreparedData {
  xs: tf.Tensor3D;
  ys: tf.Tensor2D;
  /** 第 i 行对应的原始样本下标（增强副本共享同一个下标），用于对齐 soft labels */
  sourceIndex: number[];
}

function prepareSequenceData(
  samples: SequenceSample[],
  labels: string[],
  cfg: SeqTrainingConfig,
  includeVision: boolean,
  order: number[]
): PreparedData {
  const labelToIndex = new Map(labels.map((l, i) => [l, i]));
  const frameDim = includeVision ? FUSED_FRAME_DIM : TACTILE_FRAME_DIM;
  const rows = order.length;

  const xs = new Float32Array(rows * cfg.seqLen * frameDim);
  const ys = new Float32Array(rows * labels.length);
  const sourceIndex: number[] = [];

  for (let r = 0; r < rows; r++) {
    const encoded = order[r];
    // encoded = sampleIdx * (augmentCopies+1) + copyIdx
    const copies = cfg.augmentCopies + 1;
    const sampleIdx = Math.floor(encoded / copies);
    const copyIdx = encoded % copies;
    const sample = samples[sampleIdx];

    const built = buildSequenceFeatures(sample, {
      seqLen: cfg.seqLen,
      includeVision,
      augment: copyIdx === 0 ? NO_AUGMENT : cfg.augment,
    });
    xs.set(built.data, r * cfg.seqLen * frameDim);

    const li = labelToIndex.get(sample.primaryLabel);
    if (li !== undefined) ys[r * labels.length + li] = 1;
    sourceIndex.push(sampleIdx);
  }

  return {
    xs: tf.tensor3d(xs, [rows, cfg.seqLen, frameDim]),
    ys: tf.tensor2d(ys, [rows, labels.length]),
    sourceIndex,
  };
}

function shuffled(n: number): number[] {
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export interface SplitPlan {
  /** 训练行的 encoded 下标（原始 + 全部增强副本） */
  trainOrder: number[];
  /** 验证行的 encoded 下标（**只有原始，不含任何增强副本**） */
  valOrder: number[];
  trainSamples: number;
  valSamples: number;
}

/**
 * 按**源样本**分组划分训练/验证 —— 不能用 tf 的 `validationSplit`。
 *
 * 曾经的写法是"把 原始+增强副本 全部打乱成一个大数组，交给 validationSplit: 0.2 切尾部"。
 * 那是**数据泄漏**：同一次录制的原始版可能落在训练集、它的增强副本落在验证集，
 * 而增强副本与原始只差一点噪声/时间缩放/戴歪角度 —— 等于拿背过的题当考卷，
 * 验证准确率虚高到没有参考价值（小样本时尤其严重：5 词×5 遍只有 25 条源样本，
 * 75 行里几乎每条验证行都能在训练集里找到自己的同胞）。
 *
 * 所以：**整条录制要么全进训练、要么全进验证；且验证集一律不做增强**
 * （验证要衡量的是"没见过的一次真实动作"，增强过的验证集连分布都不对）。
 * 与 `python_train/train_seq.py` 的 `split_by_sequence` 同策略。
 *
 * 按标签分层，保证每类都有验证样本；类内只有 1 条时全部留给训练
 * （否则那一类的训练集为空，模型永远学不到它）。
 */
export function planTrainValSplit(
  samples: SequenceSample[],
  cfg: Pick<SeqTrainingConfig, "validationSplit" | "augmentCopies">
): SplitPlan {
  const copies = cfg.augmentCopies + 1;
  const byLabel = new Map<string, number[]>();
  samples.forEach((s, i) => {
    const bucket = byLabel.get(s.primaryLabel);
    if (bucket) bucket.push(i);
    else byLabel.set(s.primaryLabel, [i]);
  });

  const trainIdx: number[] = [];
  const valIdx: number[] = [];
  for (const label of Array.from(byLabel.keys()).sort()) {
    const idxs = byLabel.get(label)!;
    // 组内先打乱，避免"每类总是把最后录的几条当验证"（录制顺序常与疲劳/熟练度相关）
    const perm = shuffled(idxs.length).map((k) => idxs[k]);
    let nVal = 0;
    if (cfg.validationSplit > 0 && perm.length >= 2) {
      nVal = Math.max(
        1,
        Math.min(Math.round(perm.length * cfg.validationSplit), perm.length - 1)
      );
    }
    valIdx.push(...perm.slice(0, nVal));
    trainIdx.push(...perm.slice(nVal));
  }

  // encoded = sampleIdx * copies + copyIdx，与 prepareSequenceData 的解码一一对应；
  // copyIdx 0 恒为未增强（见 prepareSequenceData 里的 NO_AUGMENT 分支）
  const trainRows: number[] = [];
  for (const s of trainIdx) {
    for (let c = 0; c < copies; c++) trainRows.push(s * copies + c);
  }
  const trainOrder = shuffled(trainRows.length).map((k) => trainRows[k]);
  const valOrder = valIdx.map((s) => s * copies);

  return {
    trainOrder,
    valOrder,
    trainSamples: trainIdx.length,
    valSamples: valIdx.length,
  };
}

/** 该批样本中有多少比例含有效视觉 —— 低于阈值就没必要训教师做蒸馏 */
export function sequenceVisionRatio(samples: SequenceSample[]): number {
  if (samples.length === 0) return 0;
  let withVision = 0;
  for (const s of samples) {
    if (s.leftLandmarks || s.rightLandmarks) withVision++;
  }
  return withVision / samples.length;
}

// ===== 知识蒸馏 =====

/**
 * 生成温度缩放的软标签。
 *
 * 这里 `log → 除以 T → softmax` 看起来像近似，其实是**精确**的 logits 温度缩放：
 * softmax 输出满足 log(pᵢ) = zᵢ − logsumexp(z)，除以 T 后再过 softmax，
 * 那个 −logsumexp(z)/T 是与 i 无关的常数，在 softmax 里被完全约掉，
 * 结果恰好等于 softmax(z/T)。别把它"修"成别的写法。
 */
function generateSoftLabels(
  teacher: tf.LayersModel,
  fusedXs: tf.Tensor3D,
  temperature: number
): number[][] {
  return tf.tidy(() => {
    const p = teacher.predict(fusedXs) as tf.Tensor2D;
    const soft = p.log().div(tf.scalar(temperature)).softmax() as tf.Tensor2D;
    return soft.arraySync() as number[][];
  });
}

// ===== 训练 =====

export async function trainSequenceModel(
  rawSamples: SequenceSample[],
  config: Partial<SeqTrainingConfig> = {},
  onProgress?: (p: SeqTrainingProgress) => void
): Promise<{
  teacherModel: tf.LayersModel | null;
  studentModel: tf.LayersModel;
  labels: string[];
  teacherAccuracy: number;
  studentAccuracy: number;
  config: SeqTrainingConfig;
  split: { trainSamples: number; valSamples: number };
  /** 起手段自动裁剪的统计（见 sequenceTrim.ts）；给页面显示一行，不影响训练本身 */
  trim: TrimStats;
  /** 主手归一化的统计；关掉 `normalizeHandedness` 时为 null */
  handedness: HandednessStats | null;
  /** 合并掉的样本条数与命中的组。关掉合并时是 0 / 空数组 */
  merge: { merged: number; groups: MergeGroup[] };
  /** 被挡掉的句子级样本条数（多 segment，不属于孤立词训练） */
  sentencesExcluded: number;
}> {
  const cfg: SeqTrainingConfig = { ...DEFAULT_SEQ_CONFIG, ...config };
  if (rawSamples.length === 0) throw new Error("没有序列样本可供训练");

  /*
   * 挡掉句子级样本（多 segment，见 `isSentenceSample`）。
   *
   * 这是**孤立词**训练器。句子样本的 `primaryLabel` 等于它的第一个词，不挡的话
   * 一条「我 名字 王」会作为一条 `我` 进 softmax，而它的特征里还有另外两个词 ——
   * 相当于往 `我` 这一类里掺噪声。症状是"某个词莫名变差"，而条数、覆盖率、
   * 裁剪汇总全都正常，几乎无法定位。
   *
   * 放在**归一化之前**：镜像和裁剪都按"一条录制 = 一个词"的假设写的，
   * 先挡掉就不必去想它们在多词样本上是什么行为。
   */
  const wordSamples = rawSamples.filter((s) => !isSentenceSample(s));
  const sentencesExcluded = rawSamples.length - wordSamples.length;
  if (wordSamples.length === 0)
    throw new Error(
      `全部 ${rawSamples.length} 条都是句子级样本，孤立词训练没有数据。` +
        `句子模型走 python_train/train_seq.py --ctc，不在浏览器里训。`
    );

  // 归一化到右手口径**只做一次**，放在这里而不是 prepareSequenceData 里：
  // 那个函数每个增强副本、教师学生各跑一遍，同一条样本会被重复判定、重复镜像 6 次，
  // 而结论对同一条样本恒定。放在这里还有个好处 —— 切分、视觉占比、裁剪汇总
  // 看到的都是归一化后的同一批数据，不会出现"报的是 A、训的是 B"。
  const handednessResult = cfg.normalizeHandedness
    ? normalizeSamplesToRight(wordSamples, cfg.bendRanges)
    : null;
  const normalized = handednessResult?.samples ?? wordSamples;

  // 合并特征层不可分的类（我/你/他、我们/你们/他们，见 labelMerge.ts）。
  // 必须在下一行推类别表**之前**做，否则合并进来的类根本进不了 softmax。
  // 放在归一化之后：镜像判定看的是运动能量，和标签无关，两者互不影响
  const mergeResult = mergeSamples(normalized, cfg.mergeDegenerateLabels);
  const samples = mergeResult.samples;

  const labels = Array.from(new Set(samples.map((s) => s.primaryLabel))).sort();
  if (labels.length < 2) throw new Error("至少需要 2 个不同的标签才能训练");

  const numClasses = labels.length;
  // 按源样本分组切分（见 planTrainValSplit 的注释：绝不能用 validationSplit）。
  // 同一份 order 供教师/学生共用，保证两边行对行一致（蒸馏标签靠这个对齐）
  const split = planTrainValSplit(samples, cfg);
  const order = split.trainOrder;
  if (split.valSamples === 0) {
    console.warn(
      "[SeqModel] 验证集为空（每类样本太少或 validationSplit=0），下面报的验证准确率不可信"
    );
  }

  const visionRatio = sequenceVisionRatio(samples);
  const useDistillation = visionRatio >= 0.8;

  // 起手段裁剪的汇总。裁剪本身在 buildSequenceFeatures 里逐条做，这里只是**把它算一遍
  // 报出来** —— 没有 UI 的自动预处理最怕的就是"裁了什么完全看不见"，
  // 哪几条没裁成（no_vision / no_run / too_short）必须能在训练完那条消息里读到。
  const trimStats = summarizeTrim(samples);

  const teacherEpochs = useDistillation
    ? Math.max(1, Math.floor(cfg.epochs * 0.6))
    : 0;
  const studentEpochs = Math.max(cfg.epochs - teacherEpochs, 30);
  const totalEpochs = teacherEpochs + studentEpochs;

  let teacherModel: tf.LayersModel | null = null;
  let teacherAccuracy = 0;
  let softLabels: number[][] | null = null;
  let teacherSourceIndex: number[] = [];

  if (useDistillation) {
    const fused = prepareSequenceData(samples, labels, cfg, true, order);
    const fusedVal = split.valOrder.length
      ? prepareSequenceData(samples, labels, cfg, true, split.valOrder)
      : null;
    teacherModel = buildSeqTeacher(
      numClasses,
      cfg.seqLen,
      cfg.backbone,
      cfg.learningRate
    );

    const hist = await teacherModel.fit(fused.xs, fused.ys, {
      epochs: teacherEpochs,
      batchSize: cfg.batchSize,
      validationData: fusedVal ? [fusedVal.xs, fusedVal.ys] : undefined,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          onProgress?.({
            epoch: epoch + 1,
            totalEpochs,
            loss: logs?.loss ?? 0,
            accuracy: logs?.acc ?? 0,
            valLoss: logs?.val_loss ?? 0,
            valAccuracy: logs?.val_acc ?? 0,
            phase: "teacher",
          });
        },
      },
    });
    const valAcc = hist.history.val_acc as number[] | undefined;
    teacherAccuracy = valAcc?.length ? valAcc[valAcc.length - 1] : 0;

    softLabels = generateSoftLabels(
      teacherModel,
      fused.xs,
      cfg.distillationTemp
    );
    teacherSourceIndex = fused.sourceIndex;

    fused.xs.dispose();
    fused.ys.dispose();
    fusedVal?.xs.dispose();
    fusedVal?.ys.dispose();
  } else {
    console.warn(
      `[SeqModel] 仅 ${(visionRatio * 100).toFixed(1)}% 样本含视觉，跳过教师与蒸馏，直接用 hard label 训练学生`
    );
  }

  const tactile = prepareSequenceData(samples, labels, cfg, false, order);
  const tactileVal = split.valOrder.length
    ? prepareSequenceData(samples, labels, cfg, false, split.valOrder)
    : null;

  let trainingYs: tf.Tensor2D = tactile.ys;
  let mixedYs: tf.Tensor2D | null = null;
  if (softLabels) {
    // 教师与学生用的是同一个 order，行是一一对应的；这里断言一下，
    // 万一将来有人改了 order 的生成方式，能立刻炸出来而不是静默错配标签
    const aligned =
      teacherSourceIndex.length === tactile.sourceIndex.length &&
      teacherSourceIndex.every((v, i) => v === tactile.sourceIndex[i]);
    if (!aligned) throw new Error("教师/学生样本顺序不一致，蒸馏标签会错配");

    const softYs = tf.tensor2d(softLabels, [softLabels.length, numClasses]);
    // 混合标签 alpha*soft + (1-alpha)*hard 等价于 alpha*CE(soft)+(1-alpha)*CE(hard)，
    // 因为交叉熵对目标分布是线性的
    const a = cfg.distillationAlpha;
    mixedYs = softYs
      .mul(tf.scalar(a))
      .add(tactile.ys.mul(tf.scalar(1 - a))) as tf.Tensor2D;
    softYs.dispose();
    trainingYs = mixedYs;
  }

  const studentModel = buildSeqStudent(
    numClasses,
    cfg.seqLen,
    cfg.backbone,
    cfg.learningRate * 0.5
  );

  // 验证集始终喂**硬标签**：蒸馏时训练用的是 α·soft+(1−α)·hard 混合标签，
  // 拿混合标签算出来的 val_acc 不是"预测对不对"，而是"像不像教师"
  const hist = await studentModel.fit(tactile.xs, trainingYs, {
    epochs: studentEpochs,
    batchSize: cfg.batchSize,
    validationData: tactileVal ? [tactileVal.xs, tactileVal.ys] : undefined,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        onProgress?.({
          epoch: teacherEpochs + epoch + 1,
          totalEpochs,
          loss: logs?.loss ?? 0,
          accuracy: logs?.acc ?? 0,
          valLoss: logs?.val_loss ?? 0,
          valAccuracy: logs?.val_acc ?? 0,
          phase: "student",
        });
      },
    },
  });

  const valAcc = hist.history.val_acc as number[] | undefined;
  const studentAccuracy = valAcc?.length ? valAcc[valAcc.length - 1] : 0;

  tactile.xs.dispose();
  tactile.ys.dispose();
  tactileVal?.xs.dispose();
  tactileVal?.ys.dispose();
  mixedYs?.dispose();

  return {
    teacherModel,
    studentModel,
    labels,
    teacherAccuracy,
    studentAccuracy,
    config: cfg,
    split: { trainSamples: split.trainSamples, valSamples: split.valSamples },
    trim: trimStats,
    handedness: handednessResult?.stats ?? null,
    merge: { merged: mergeResult.merged, groups: mergeResult.groups },
    sentencesExcluded,
  };
}

// ===== 推理 =====

let loadedModel: tf.LayersModel | null = null;
let loadedLabels: string[] = [];
let loadedSeqLen = SEQ_LEN;
let loadedFrameDim = TACTILE_FRAME_DIM;

export function setActiveSequenceModel(
  model: tf.LayersModel,
  labels: string[],
  seqLen: number,
  frameDim: number
): void {
  loadedModel?.dispose();
  loadedModel = model;
  loadedLabels = labels;
  loadedSeqLen = seqLen;
  loadedFrameDim = frameDim;
}

export function isSequenceModelLoaded(): boolean {
  return loadedModel !== null && loadedLabels.length > 0;
}

export function getLoadedSequenceLabels(): string[] {
  return loadedLabels;
}

export function getLoadedSeqLen(): number {
  return loadedSeqLen;
}

/**
 * 对一段窗口做推理。窗口由调用方（Translate 的环形缓冲）拼成 SequenceSample，
 * 时间栅格不需要严格等间隔——buildSequenceFeatures 会按 timestamps 重采样。
 */
export function predictSequence(
  window: SequenceSample
): PredictionResult | null {
  if (!loadedModel || loadedLabels.length === 0) return null;
  if (window.frameCount < 2) return null;

  const includeVision = loadedFrameDim === FUSED_FRAME_DIM;
  const built = buildSequenceFeatures(window, {
    seqLen: loadedSeqLen,
    includeVision,
    augment: NO_AUGMENT,
    // 推理端**显式不裁**：滑窗是纯触觉（sequenceWindow 把 landmarks 写成 null），
    // 起手段判据本来就不成立。写死在这里是为了让以后给滑窗加视觉的人看见这个决定，
    // 而不是让推理行为跟着悄悄变 —— 训练裁的是"整条录制的头"，推理窗口没有"头"可言。
    trim: null,
  });

  const probs = tf.tidy(() => {
    const x = tf.tensor3d(built.data, [1, loadedSeqLen, built.frameDim]);
    const out = loadedModel!.predict(x) as tf.Tensor2D;
    return Array.from(out.dataSync());
  });

  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  const allProbabilities = loadedLabels
    .map((label, i) => ({ label, probability: probs[i] ?? 0 }))
    .sort((a, b) => b.probability - a.probability);

  return {
    label: loadedLabels[maxIdx],
    confidence: probs[maxIdx],
    allProbabilities,
  };
}

// ===== 序列化 =====

export async function serializeSequenceModel(
  model: tf.LayersModel,
  labels: string[],
  accuracy: number,
  name: string,
  modelType: "seq_fused" | "seq_tactile",
  seqLen: number,
  backbone: SeqBackbone
): Promise<SavedModel> {
  const artifacts = await new Promise<tf.io.ModelArtifacts>((resolve) => {
    model.save(
      tf.io.withSaveHandler(async (a) => {
        resolve(a);
        return {
          modelArtifactsInfo: {
            dateSaved: new Date(),
            modelTopologyType: "JSON",
          },
        };
      })
    );
  });

  let weightsBuffer: ArrayBuffer;
  if (artifacts.weightData instanceof ArrayBuffer) {
    weightsBuffer = artifacts.weightData;
  } else if (Array.isArray(artifacts.weightData)) {
    const bufs = artifacts.weightData as ArrayBuffer[];
    const total = bufs.reduce((s, b) => s + b.byteLength, 0);
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const b of bufs) {
      merged.set(new Uint8Array(b), offset);
      offset += b.byteLength;
    }
    weightsBuffer = merged.buffer;
  } else {
    weightsBuffer = new ArrayBuffer(0);
  }

  return {
    name,
    createdAt: Date.now(),
    accuracy,
    labels,
    modelJson: JSON.stringify({
      modelTopology: artifacts.modelTopology,
      weightsManifest: artifacts.weightSpecs,
    }),
    weightsData: weightsBuffer,
    modelType,
    seqLen,
    frameDim:
      modelType === "seq_fused" ? FUSED_FRAME_DIM : TACTILE_FRAME_DIM,
    backbone,
  };
}

export async function loadSequenceModelFromSaved(
  saved: SavedModel
): Promise<void> {
  const parsed = JSON.parse(saved.modelJson);
  const model = await tf.loadLayersModel(
    tf.io.fromMemory(
      parsed.modelTopology,
      parsed.weightsManifest,
      saved.weightsData
    )
  );
  setActiveSequenceModel(
    model,
    saved.labels,
    saved.seqLen ?? SEQ_LEN,
    saved.frameDim ?? TACTILE_FRAME_DIM
  );
  console.log(
    `[SeqModel] Loaded ${saved.modelType} "${saved.name}": ${saved.labels.length} 类, T=${saved.seqLen}, backbone=${saved.backbone}`
  );
}

// ===== 评估 =====

export function evaluateSequences(samples: SequenceSample[]): {
  accuracy: number;
  confusionMatrix: Record<string, Record<string, number>>;
} {
  if (!loadedModel || loadedLabels.length === 0) {
    return { accuracy: 0, confusionMatrix: {} };
  }
  const confusion: Record<string, Record<string, number>> = {};
  for (const a of loadedLabels) {
    confusion[a] = {};
    for (const b of loadedLabels) confusion[a][b] = 0;
  }

  let correct = 0;
  let counted = 0;
  for (const s of samples) {
    const r = predictSequence(s);
    if (!r) continue;
    counted++;
    if (confusion[s.primaryLabel]) {
      confusion[s.primaryLabel][r.label] =
        (confusion[s.primaryLabel][r.label] || 0) + 1;
    }
    if (r.label === s.primaryLabel) correct++;
  }
  return { accuracy: counted > 0 ? correct / counted : 0, confusionMatrix: confusion };
}
