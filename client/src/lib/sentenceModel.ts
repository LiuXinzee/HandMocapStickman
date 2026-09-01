/*
 * sentenceModel — 加载 Python 训好的句子级 CTC 模型，做整句推理。
 *
 * ===== 为什么不是 tf.loadLayersModel =====
 *
 * 原计划是走 `tensorflowjs_converter` 产出 model.json + 权重分片，再
 * `tf.loadLayersModel('/models/seq_sentence/model.json')`。这条路在 Windows 上走不通：
 * `tensorflowjs` 的转换器无条件 `import tensorflow_decision_forests`，而 TF-DF 只发
 * Linux/macOS 轮子（官方文档写明 Windows 要用 WSL）；绕过它之后还有一关是
 * tfjs 用的 `tf_keras` 是 Keras 2，读不了 Keras 3 存的 `.keras`。
 *
 * 改成**只导权重**：`python_train/export_weights.py` 输出 weights.bin + weights.json，
 * 浏览器用 `buildSeqSentenceStudent()`（结构定义在 sequenceModel.ts，与 Python 侧
 * 逐层同构，这是本仓库一直以来的硬约束）自己搭出空网络，再按层名把权重填进去。
 *
 * 这样反而更稳：结构不同构会在**加载时**按层名/形状报错并说清是哪一层。
 * 走转换器时同类问题的典型症状是加载成功但权重对错位 ——
 * 线上表现为"能跑、有置信度、每个词都错"。
 *
 * 读 manifest / 填权重那套现在在 `modelWeights.ts`，与孤立词模型
 * （`wordModel.ts`）共用一份 —— 两边导出格式相同，抄两份会漂。
 *
 * ===== 模型是静态文件，不进 IndexedDB =====
 *
 * 浏览器自己训的孤立词模型存在 IndexedDB（`datasetStore` 的 SavedModel）。
 * Python 训的模型是**构建产物**，放 `client/public/models/seq_sentence/`，
 * 随代码一起部署。所以换模型要重新部署，不是在页面上点一下。
 */
import * as tf from "@tensorflow/tfjs";
import type { SequenceSample } from "./datasetStore";
import { buildSeqSentenceStudent, type SeqBackbone } from "./sequenceModel";
import { buildSequenceFeatures, TACTILE_FRAME_DIM } from "./sequenceFeatures";
import type { DecodedSpan } from "./ctcDecode";
import { decodeToWords, greedyDecode, greedyDecodeSpans } from "./ctcDecode";
import {
  ModelMissingError,
  fetchWeights,
  fillWeights,
  modelDeployed,
  type WeightManifest,
} from "./modelWeights";

/** 默认部署位置。与 `export_weights.py --out` 的默认值对应 */
export const SENTENCE_MODEL_DIR = "/models/seq_sentence";

/** `sentence_student_meta.json` 里我们真正依赖的字段 */
export interface SentenceModelMeta {
  /** 顺序**就是** softmax 下标，不能排序、不能去重、不能在 JS 里重算 */
  labels: string[];
  /** 输入时间长度（128）。特征必须按这个长度重采样 */
  seqLen: number;
  backbone: SeqBackbone;
  frameDim: number;
  ctc: boolean;
  /** blank 下标（= labels.length）。**从这里读，不要另算** */
  blankIndex: number;
  /** 骨干输出帧数（seqLen/4）。解码要用它，不是 seqLen */
  outputFrames: number;
  /** Python 侧量出的验证集 WER；没量过是 null（不是 0） */
  valWer: number | null;
  /** true = 训练数据全是合成句子，这个 WER 不能当真实表现看 */
  synthOnly: boolean;
  numTrain?: number;
  numVal?: number;
  numReal?: number;
  numSynth?: number;
}

export interface LoadedSentenceModel {
  model: tf.LayersModel;
  meta: SentenceModelMeta;
  dispose(): void;
}

export interface SentencePrediction {
  /** 解码出来的词（原始类别 id，合并类是 `merged_pron_sg` 这种） */
  words: string[];
  /** 对应的类别下标 */
  indices: number[];
  /**
   * 每个词占的输出帧区间，与 `words` / `indices` 一一对应、等长。
   *
   * 拇指压力闸门在句子路径上靠它只看**那个词**的那几帧 —— 见
   * `greedyDecodeSpans` 的注释里那条「好看全段峰值 26」的实测。
   */
  spans: DecodedSpan[];
  /** 逐帧概率 [outputFrames * (numClasses+1)]，给 UI 画时间轴或调试用 */
  perFrame: Float32Array;
  outputFrames: number;
  numClasses: number;
  blankIndex: number;
}

/**
 * 没部署句子模型（404）。
 *
 * 现在是 `ModelMissingError` 的别名 —— 抽共享加载器时错误类型跟着共用了一个。
 * 保留这个名字是因为调用方（Translate 的探测逻辑、测试）按它做 instanceof 判断，
 * 而"404 = 没部署，别的 HTTP 错 = 坏了"这条区分本身没变。
 */
export { ModelMissingError as SentenceModelMissingError } from "./modelWeights";

const HINT = "先在 python_train/ 跑 train_seq.py --ctc，再跑 export_weights.py。";

/**
 * 加载句子模型：读 manifest → 搭同构空网络 → 按层名填权重。
 *
 * 失败一律抛错，**不做静默降级**。句子模型加载不上时唯一正确的表现是"这一档不可用"
 * 并说清原因；带着随机初始化的权重跑下去会输出看起来像话的乱句。
 */
export async function loadSentenceModel(
  dir: string = SENTENCE_MODEL_DIR
): Promise<LoadedSentenceModel> {
  const { manifest, buf } = await fetchWeights<SentenceModelMeta>(dir, HINT);
  const meta = manifest.meta;
  if (!meta?.labels?.length) {
    throw new Error(`${dir}/weights.json 里没有 labels —— 无法把输出下标翻回词`);
  }
  if (!meta.ctc) {
    throw new Error(`${dir} 里的不是 CTC 句子模型（meta.ctc 不为 true）`);
  }
  // blank 必须在末位。这条不成立说明 Python 侧的 head 变了，解码结果整句作废
  if (meta.blankIndex !== meta.labels.length) {
    throw new Error(
      `blankIndex ${meta.blankIndex} 与类别数 ${meta.labels.length} 不符 —— blank 必须在末位`
    );
  }
  if (meta.frameDim !== TACTILE_FRAME_DIM) {
    throw new Error(
      `模型要求每帧 ${meta.frameDim} 维，本端特征是 ${TACTILE_FRAME_DIM} 维 —— ` +
        `特征布局改过了，模型要重训`
    );
  }
  if (meta.outputFrames !== Math.floor(meta.seqLen / 4)) {
    throw new Error(
      `outputFrames ${meta.outputFrames} 与 seqLen ${meta.seqLen} 不符（骨干池化两次应为 T/4）`
    );
  }

  const model = buildSeqSentenceStudent(meta.labels.length, meta.seqLen, meta.backbone);
  fillWeights(model, manifest, buf, "对照 sequenceModel.ts 的 tcnBackbone");

  return { model, meta, dispose: () => model.dispose() };
}

/**
 * 整句推理：一条（长达 12s 的）录制 → 词序列。
 *
 * **关掉起手段裁剪**（`trim: null`）。词间停顿正是 CTC 用 blank 建模的东西，
 * 裁掉就把句子结构破坏了；而且裁剪判据（`sequenceTrim`）只看视觉，
 * 部署时摄像头不一定在。推理路径本来就一律不裁（与 `predictSequence` 一致）。
 */
export function predictSentence(
  loaded: LoadedSentenceModel,
  sample: SequenceSample
): SentencePrediction {
  const { model, meta } = loaded;
  const numClasses = meta.labels.length + 1;
  const feat = buildSequenceFeatures(sample, {
    seqLen: meta.seqLen,
    includeVision: false,
    trim: null,
  });

  const perFrame = tf.tidy(() => {
    const x = tf.tensor3d(feat.data, [1, meta.seqLen, meta.frameDim]);
    const y = model.predict(x) as tf.Tensor3D;
    return y.dataSync() as Float32Array;
  });

  // 解码用 outputFrames（T/4），不是 seqLen。传错的话会多读 3 倍的帧、越界抛错
  const indices = greedyDecode(perFrame, meta.outputFrames, numClasses, meta.blankIndex);
  /*
   * 区间**另算一遍**而不是从区间反推 indices：`greedyDecode` 是被
   * `sentenceCtcFixture.json` 逐条钉在 Python 解码结果上的那一条路，
   * 输出的词序列必须一直由它决定。下面这个断言保证两者没有分岔 ——
   * 分岔时区间和词就会错位一格，闸门去看别的词的帧,静默判错。
   */
  const spans = greedyDecodeSpans(perFrame, meta.outputFrames, numClasses, meta.blankIndex);
  if (spans.length !== indices.length) {
    throw new Error(
      `greedyDecodeSpans 解出 ${spans.length} 个词、greedyDecode 解出 ${indices.length} 个 —— 两条解码路径分岔了`
    );
  }
  return {
    words: decodeToWords(indices, meta.labels),
    indices,
    spans,
    perFrame,
    outputFrames: meta.outputFrames,
    numClasses,
    blankIndex: meta.blankIndex,
  };
}

/** 只探测有没有部署模型，不加载权重。给 UI 决定"连续句子"这一档能不能点。 */
export async function sentenceModelAvailable(
  dir: string = SENTENCE_MODEL_DIR
): Promise<boolean> {
  return modelDeployed(dir, HINT);
}
