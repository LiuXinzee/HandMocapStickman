/*
 * wordModel —— 加载 Python 训好的**孤立词**模型（部署在 /models/seq_student/）。
 *
 * ===== 为什么需要这个文件 =====
 *
 * 在这之前，浏览器拿孤立词模型只有一条路：`loadSequenceModelFromSaved`，从 IndexedDB
 * 读**页面内训练**出来的模型。Python 侧 `train_seq.py` 训出的那个（29 类、
 * valAccuracy 97.7%）没有任何进浏览器的途径 —— `client/public/models/` 下只有
 * `seq_sentence/`。
 *
 * 后果不是"少了个可选项"，而是**逐词档认不出新词**：08-28/29 那批新录的
 * `i / beautiful / name / resemble / smile / sun` 根本不在浏览器那个旧模型的标签表里。
 * 模型对任意输入都会输出一个已知类，于是打「像」「我」出来的是别的词，
 * 而不是"没认出来"。句子模型早就走了部署这条路（`sentenceModel.ts`），
 * 孤立词模型只是一直没接上。
 *
 * ===== 与 sentenceModel 的关系 =====
 *
 * 同一套导出格式、同一个加载器（`modelWeights.ts`），只有三处不同：
 *   1. 结构是 `buildSeqStudent`（带 GAP + fc 的孤立词 head），不是 CTC 的 frame-wise head
 *   2. meta 没有 ctc / blankIndex / outputFrames，多一个 valAccuracy
 *   3. 加载完要接进 `setActiveSequenceModel` 那套全局单例（滑窗推理走 `predictSequence`），
 *      而句子模型是自己持有 handle 的
 *
 * 第 3 点是有意的：滑窗推理链路（Translate 的 100ms 定时器）本来就读全局单例，
 * 让部署模型和 IndexedDB 模型走同一个出口，推理侧一个字都不用改。
 *
 * ===== 模型是构建产物，不进 IndexedDB =====
 *
 * 换模型要重新跑 export_weights.py + 重新部署，不是在页面上点一下。
 */
import * as tf from "@tensorflow/tfjs";
import { TACTILE_FRAME_DIM } from "./sequenceFeatures";
import {
  buildSeqStudent,
  setActiveSequenceModel,
  type SeqBackbone,
} from "./sequenceModel";
import { fetchOrThrow, fetchWeights, fillWeights, modelDeployed } from "./modelWeights";

/** 默认部署位置。与 `export_weights.py --out ../client/public/models/seq_student` 对应 */
export const WORD_MODEL_DIR = "/models/seq_student";

const HINT =
  "先在 python_train/ 跑 train_seq.py，再跑 " +
  "export_weights.py --model out/student.keras --meta out/student_meta.json " +
  "--out ../client/public/models/seq_student。";

/** `student_meta.json` 里我们真正依赖的字段 */
export interface WordModelMeta {
  /** 顺序**就是** softmax 下标，不能排序、不能去重、不能在 JS 里重算 */
  labels: string[];
  /** 输入时间长度（32）。特征必须按这个长度重采样 */
  seqLen: number;
  backbone: SeqBackbone;
  frameDim: number;
  /** Python 侧量出的验证集准确率；没量过是 undefined（不是 0） */
  valAccuracy?: number;
  /**
   * 导出时刻（epoch 毫秒）。用来和 IndexedDB 里那个"页面内训的模型"比新旧
   * —— 见 Translate 的选型逻辑。老产物没有这个字段（undefined）。
   */
  exportedAt?: number;
  numSequences?: number;
  numTrain?: number;
  numVal?: number;
  distilled?: boolean;
  modelType?: string;
}

export interface LoadedWordModel {
  model: tf.LayersModel;
  meta: WordModelMeta;
}

/**
 * 搭出模型并填权重，**不碰全局单例** —— 调用方拥有这个 model，负责 dispose。
 *
 * 与 `loadDeployedWordModel` 分成两层是因为所有权不能含糊：单例那边
 * `setActiveSequenceModel` 会 dispose 上一个模型，如果调用方也持有并 dispose 了它，
 * 下一次加载就会撞上 "Container 'student' is already disposed"。
 * 这个函数给需要拿着模型自己做点什么的人（数值对照测试、以后可能的离线评估）。
 */
export async function buildDeployedWordModel(
  dir: string = WORD_MODEL_DIR
): Promise<LoadedWordModel> {
  const { manifest, buf } = await fetchWeights<WordModelMeta>(dir, HINT);
  const meta = manifest.meta;

  if (!meta?.labels?.length) {
    throw new Error(`${dir}/weights.json 里没有 labels —— 无法把输出下标翻回词`);
  }
  // CTC 模型误放到这个目录会一路加载成功（骨干同构），只在 head 的形状上炸；
  // 早一步说清是"放错了模型"比让人去查 student_out 的形状有用
  if ((meta as { ctc?: boolean }).ctc) {
    throw new Error(`${dir} 里放的是 CTC 句子模型，不是孤立词模型 —— 目录搞反了？`);
  }
  if (meta.frameDim !== TACTILE_FRAME_DIM) {
    throw new Error(
      `模型要求每帧 ${meta.frameDim} 维，本端特征是 ${TACTILE_FRAME_DIM} 维 —— ` +
        `特征布局改过了，模型要重训`
    );
  }

  // lr 只用于 compile 的优化器，推理不碰它。这里复用 buildSeqStudent 而不是另搭一份，
  // 是因为结构定义只能有一份（同 buildSeqSentenceStudent 的注释）
  const model = buildSeqStudent(meta.labels.length, meta.seqLen, meta.backbone, 1e-3);
  fillWeights(model, manifest, buf, "对照 sequenceModel.ts 的 tcnBackbone + isolatedWordHead");

  return { model, meta };
}

/**
 * 加载部署的孤立词模型，并**接进全局单例** —— 滑窗推理（`predictSequence`）立刻用上。
 *
 * **返回值里没有 model：所有权归单例。** `setActiveSequenceModel` 会在下一次加载时
 * dispose 它，调用方再 dispose 一次就会撞上 "Container 'student' is already disposed"
 * —— 而炸的位置在**下一次**加载里，跟真正多调的那一次隔着好远。
 * 自己要拿模型的用 `buildDeployedWordModel`。
 *
 * 失败一律抛错，不做静默降级 —— 理由同 `fillWeights`。
 */
export async function loadDeployedWordModel(
  dir: string = WORD_MODEL_DIR
): Promise<WordModelMeta> {
  const { model, meta } = await buildDeployedWordModel(dir);
  setActiveSequenceModel(model, meta.labels, meta.seqLen, meta.frameDim);
  return meta;
}

/** 只探测有没有部署，不下权重（651KB）。给选型逻辑先问一句 */
export async function wordModelAvailable(dir: string = WORD_MODEL_DIR): Promise<boolean> {
  return modelDeployed(dir, HINT);
}

/**
 * 只读 meta，不下 weights.bin（651KB）。选型要用 exportedAt / labels 比新旧，
 * 而选型跑在页面挂载时 —— 那时候还不知道用不用得上这个模型。
 */
export async function fetchWordModelMeta(
  dir: string = WORD_MODEL_DIR
): Promise<WordModelMeta> {
  const res = (await (await fetchOrThrow(`${dir}/weights.json`, HINT)).json()) as {
    meta: WordModelMeta;
  };
  return res.meta;
}
