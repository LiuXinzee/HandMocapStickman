/*
 * sequenceModel 的**反向传播**冒烟测试。
 *
 * 为什么单独为"能不能训"写测试：这一页的 TCN 骨干原来用空洞卷积，而 tfjs 的 conv2D
 * 前向支持空洞、反向直接抛 "dilation rates greater than 1 are not yet supported in
 * gradients"。结果是模型能建、能 predict、类型检查全绿、`pnpm build` 通过，
 * 只有真正跑 `fit` 才会炸 —— 于是这个 bug 一路活到用户录完 5 个词点训练那一刻。
 * 类型检查和前向推理都拦不住它，只有真跑一步反向能拦住。
 *
 * 所以这里不测精度、不测收敛，只测**梯度能算出来**：1 epoch、2 个类、极小 batch。
 * 谁改动 tcnBackbone（换层、加 dilationRate、换池化）都必须让这条继续绿。
 *
 * 环境说明：vitest 是纯 node，tfjs 回落到 cpu 后端，比浏览器 WebGL 慢很多 ——
 * 所以刻意用**学生**（294D 输入 / 64-128 通道）而不是教师（420D / 128-256），
 * 且只喂 4 条样本。教师与学生共用 buildSeqModel，结构问题两边同源。
 */
import * as tf from "@tensorflow/tfjs";
import { beforeAll, describe, expect, it } from "vitest";
import {
  buildSeqStudent,
  buildSeqTeacher,
  planTrainValSplit,
} from "./sequenceModel";
import { FUSED_FRAME_DIM, SEQ_LEN, TACTILE_FRAME_DIM } from "./sequenceFeatures";
import type { SequenceSample } from "./datasetStore";

beforeAll(async () => {
  await tf.ready();
});

/** 跑一步真训练；成功返回 null，失败返回错误信息（不 throw，方便断言里读原文） */
async function fitOneStep(model: tf.LayersModel, frameDim: number): Promise<string | null> {
  const xs = tf.randomNormal([4, SEQ_LEN, frameDim]);
  const ys = tf.oneHot(tf.tensor1d([0, 1, 0, 1], "int32"), 2);
  try {
    await model.fit(xs, ys, { epochs: 1, batchSize: 2, verbose: 0, shuffle: false });
    return null;
  } catch (e) {
    return e instanceof Error ? e.message : String(e);
  } finally {
    tf.dispose([xs, ys]);
  }
}

/** planTrainValSplit 只看 primaryLabel，别的字段给个空壳就够 */
function fakeSamples(counts: Record<string, number>): SequenceSample[] {
  const out: SequenceSample[] = [];
  for (const [label, n] of Object.entries(counts)) {
    for (let i = 0; i < n; i++) {
      out.push({ primaryLabel: label } as unknown as SequenceSample);
    }
  }
  return out;
}

/** encoded → 源样本下标（与 prepareSequenceData 的解码同一套） */
const srcOf = (encoded: number, copies: number) => Math.floor(encoded / copies);
const copyOf = (encoded: number, copies: number) => encoded % copies;

describe("训练/验证按源样本分组切分（防数据泄漏）", () => {
  const augmentCopies = 2;
  const copies = augmentCopies + 1;

  it("同一条录制绝不同时出现在训练集和验证集", () => {
    const samples = fakeSamples({ 你好: 5, 谢谢: 5, 再见: 5, _idle: 8 });
    const p = planTrainValSplit(samples, { validationSplit: 0.2, augmentCopies });

    const trainSrc = new Set(p.trainOrder.map((e) => srcOf(e, copies)));
    const valSrc = new Set(p.valOrder.map((e) => srcOf(e, copies)));
    for (const s of valSrc) expect(trainSrc.has(s)).toBe(false);
    // 每条源样本都被用上了，没有凭空丢样本
    expect(trainSrc.size + valSrc.size).toBe(samples.length);
    expect(p.trainSamples).toBe(trainSrc.size);
    expect(p.valSamples).toBe(valSrc.size);
  });

  it("验证集只含未增强的原始行（增强副本进验证集就是在考背过的题）", () => {
    const samples = fakeSamples({ 你好: 6, 谢谢: 6 });
    const p = planTrainValSplit(samples, { validationSplit: 0.5, augmentCopies });

    expect(p.valOrder.length).toBeGreaterThan(0);
    for (const e of p.valOrder) expect(copyOf(e, copies)).toBe(0);
    // 验证行数 = 验证样本数（一条录制一行，不翻倍）
    expect(p.valOrder.length).toBe(p.valSamples);
    // 训练行才是原始 + 全部副本
    expect(p.trainOrder.length).toBe(p.trainSamples * copies);
    expect(new Set(p.trainOrder).size).toBe(p.trainOrder.length);
  });

  it("每个类都在验证集里有样本（分层）", () => {
    const samples = fakeSamples({ 你好: 5, 谢谢: 5, 再见: 5 });
    const p = planTrainValSplit(samples, { validationSplit: 0.2, augmentCopies });
    const valLabels = new Set(
      p.valOrder.map((e) => samples[srcOf(e, copies)].primaryLabel)
    );
    expect(valLabels.size).toBe(3);
  });

  it("类内只有 1 条时全部留给训练（否则那一类训练集为空）", () => {
    const samples = fakeSamples({ 你好: 5, 稀有词: 1 });
    const p = planTrainValSplit(samples, { validationSplit: 0.2, augmentCopies });
    const rareIdx = samples.findIndex((s) => s.primaryLabel === "稀有词");
    expect(p.valOrder.some((e) => srcOf(e, copies) === rareIdx)).toBe(false);
    expect(p.trainOrder.some((e) => srcOf(e, copies) === rareIdx)).toBe(true);
  });

  it("validationSplit=0 → 不留验证集，且样本一条不少", () => {
    const samples = fakeSamples({ 你好: 4, 谢谢: 4 });
    const p = planTrainValSplit(samples, { validationSplit: 0, augmentCopies });
    expect(p.valOrder).toEqual([]);
    expect(p.trainSamples).toBe(8);
  });

  it("augmentCopies=0 时训练行就是原始行，编码不错位", () => {
    const samples = fakeSamples({ 你好: 4, 谢谢: 4 });
    const p = planTrainValSplit(samples, { validationSplit: 0.25, augmentCopies: 0 });
    expect(p.trainOrder.length).toBe(p.trainSamples);
    expect([...p.trainOrder, ...p.valOrder].sort((a, b) => a - b)).toEqual(
      Array.from({ length: 8 }, (_, i) => i)
    );
  });
});

describe("时序模型能反向传播", () => {
  it("学生（纯触觉）跑得动一步 fit —— 空洞卷积会在这里挂掉", async () => {
    const m = buildSeqStudent(2, SEQ_LEN, "tcn", 0.001);
    const err = await fitOneStep(m, TACTILE_FRAME_DIM);
    // 断言写成"错误信息是 null"而不是 not.toThrow：失败时能直接看到 tfjs 的原文
    expect(err).toBeNull();
    m.dispose();
  }, 120_000);

  it("教师（视觉+触觉）结构与学生同源，输入维度对得上", () => {
    const t = buildSeqTeacher(2, SEQ_LEN, "tcn", 0.001);
    expect(t.inputs[0].shape).toEqual([null, SEQ_LEN, FUSED_FRAME_DIM]);
    const s = buildSeqStudent(2, SEQ_LEN, "tcn", 0.001);
    expect(s.inputs[0].shape).toEqual([null, SEQ_LEN, TACTILE_FRAME_DIM]);
    // 蒸馏要求两者输出同形（soft label 逐类对齐）
    expect(t.outputs[0].shape).toEqual(s.outputs[0].shape);
    t.dispose();
    s.dispose();
  });

  it("骨干把时间维降到 T/4，且不把它塌缩掉（句子级换逐帧 head 要靠这个）", () => {
    const m = buildSeqStudent(2, SEQ_LEN, "tcn", 0.001);
    // head 的 GAP 层的输入就是骨干输出
    const gap = m.layers.find((l) => l.name.endsWith("_gap"));
    expect(gap).toBeDefined();
    const backboneOut = (gap!.input as tf.SymbolicTensor).shape;
    expect(backboneOut.length).toBe(3);
    expect(backboneOut[1]).toBe(SEQ_LEN / 4);
    m.dispose();
  });
});
