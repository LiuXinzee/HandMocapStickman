/*
 * modelWeights —— `export_weights.py` 产物（weights.bin + weights.json）的加载器。
 *
 * 这一份是从 `sentenceModel.ts` 里抽出来的，**不是新写的**：句子模型和孤立词模型
 * 走的是同一套导出格式，加载步骤逐字相同（读 manifest → 按层名取层 → 逐张量比形状
 * → 填权重 → 反向查缺层）。抄第二份的代价不是重复代码本身，而是两份会漂：
 * 修了一处的形状检查、另一处没修，症状是"某一档模型加载成功但全错"。
 *
 * 为什么不用 `tf.loadLayersModel`：见 `sentenceModel.ts` 文件头（Windows 上
 * tensorflowjs 转换器装不上，且 tfjs 的 tf_keras 读不了 Keras 3 的 .keras）。
 * 改成只导权重、浏览器自己搭同构结构再填 —— 结构不同构会在**加载时**按层名/形状
 * 报错并说清是哪一层，而不是加载成功后输出乱七八糟。
 *
 * 格式（与 export_weights.py 的 docstring 一一对应）：
 *   weights.bin  所有张量按 weights.json 的顺序，float32 小端，首尾相接
 *   weights.json { format, layers: [{name, weights: [{shape, offset, count}]}], meta }
 *
 * `offset` 是**元素**下标不是字节 —— ×4 得到 byteOffset，天然满足 Float32Array
 * 的 4 字节对齐要求。
 */
import * as tf from "@tensorflow/tfjs";

/** 本模块认识的格式版本。Python 侧写在 weights.json 的 `format` 里 */
export const WEIGHTS_FORMAT = "seq-weights-1.0";

export interface WeightEntry {
  shape: number[];
  /** 元素下标（不是字节），×4 就是 byteOffset */
  offset: number;
  count: number;
}

export interface LayerEntry {
  name: string;
  weights: WeightEntry[];
}

/** meta 的具体字段由各模型自己定义（句子模型要 ctc/blankIndex，词模型要 labels/valAccuracy） */
export interface WeightManifest<TMeta> {
  format: string;
  layers: LayerEntry[];
  meta: TMeta;
}

/**
 * 没部署模型（404）。**要和"部署了但坏了"分开** —— 前者是正常状态，UI 要给引导文案；
 * 后者（500、CORS、代理插一脚）必须让人看见原文，否则会显示"先去跑 python_train"，
 * 而用户明明已经跑过了。
 */
export class ModelMissingError extends Error {
  constructor(url: string, hint: string) {
    super(`没有找到模型（${url}）。${hint}`);
    this.name = "ModelMissingError";
  }
}

export async function fetchOrThrow(url: string, hint: string): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (e) {
    throw new Error(`读取 ${url} 失败：${String(e)}`);
  }
  if (res.status === 404) throw new ModelMissingError(url, hint);
  if (!res.ok) throw new Error(`读取 ${url} 失败：HTTP ${res.status}`);
  return res;
}

/** 读 manifest + 权重块。两个文件都缺才算"没部署"，缺一个是坏了 */
export async function fetchWeights<TMeta>(
  dir: string,
  hint: string
): Promise<{ manifest: WeightManifest<TMeta>; buf: ArrayBuffer }> {
  const manifest = (await (
    await fetchOrThrow(`${dir}/weights.json`, hint)
  ).json()) as WeightManifest<TMeta>;
  const buf = await (await fetchOrThrow(`${dir}/weights.bin`, hint)).arrayBuffer();
  return { manifest, buf };
}

/**
 * 按层名把权重填进已搭好的空网络。
 *
 * 失败一律抛错并**销毁模型**，不做静默降级：带着随机初始化的权重跑下去，
 * 线上表现是"能跑、有置信度、全错"—— 那比加载失败难查得多。
 *
 * 两个方向都要查：
 * - 文件里有、模型里没有的层 → 两边结构漂了
 * - 模型里有权重、文件里没给的层 → 漏一层 BN 不会报错，它会带着随机
 *   初始化的 gamma/beta 跑下去，输出偏一点、词全错
 */
export function fillWeights<TMeta>(
  model: tf.LayersModel,
  manifest: WeightManifest<TMeta>,
  buf: ArrayBuffer,
  structureHint: string
): void {
  const byName = new Map(model.layers.map((l) => [l.name, l]));
  const seen = new Set<string>();

  try {
    for (const entry of manifest.layers) {
      const layer = byName.get(entry.name);
      if (!layer) {
        throw new Error(`模型里没有层 "${entry.name}" —— 两边结构不同构（${structureHint}）`);
      }
      const want = layer.getWeights();
      if (want.length !== entry.weights.length) {
        throw new Error(
          `层 "${entry.name}" 权重个数不符：本端 ${want.length}，文件 ${entry.weights.length}`
        );
      }
      const tensors = entry.weights.map((w, i) => {
        const expected = want[i].shape;
        const same =
          expected.length === w.shape.length && expected.every((d, k) => d === w.shape[k]);
        if (!same) {
          throw new Error(
            `层 "${entry.name}" 第 ${i} 个权重形状不符：本端 [${expected}]，文件 [${w.shape}]`
          );
        }
        // slice 一份而不是直接引用 buf：tf.tensor 会持有这块内存，
        // 而 buf 是整份权重，逐层引用会把整块留在内存里
        const view = new Float32Array(buf, w.offset * 4, w.count).slice();
        return tf.tensor(view, w.shape);
      });
      layer.setWeights(tensors);
      tensors.forEach((t) => t.dispose());
      seen.add(entry.name);
    }

    const missing = model.layers
      .filter((l) => l.getWeights().length > 0 && !seen.has(l.name))
      .map((l) => l.name);
    if (missing.length) {
      throw new Error(`文件里缺这些层的权重：${missing.join(", ")}`);
    }
  } catch (e) {
    model.dispose();
    throw e;
  }
}

/** 只探测有没有部署，不下权重。给 UI 决定某一档能不能点 */
export async function modelDeployed(dir: string, hint: string): Promise<boolean> {
  try {
    await fetchOrThrow(`${dir}/weights.json`, hint);
    return true;
  } catch (e) {
    if (e instanceof ModelMissingError) return false;
    throw e;
  }
}
