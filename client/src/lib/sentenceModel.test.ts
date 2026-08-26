/*
 * sentenceModel.test —— 权重加载 + 前向计算与 keras 对照
 *
 * 换成"浏览器自己搭结构 + 填权重"之后，多出一类失配是 `ctcDecode.test` 抓不到的：
 * **tfjs 的层实现与 keras 不完全一致**。最典型的是 BatchNormalization 的 epsilon
 * 默认值，其次是 conv1d `padding:"same"` 的左右补零方式。这类失配不会抛异常，
 * 只让输出偏一点 —— 偏一点就足够让某一帧的 argmax 换人，于是整句多一个词或少一个词。
 *
 * 所以这里做的是：读**真实导出的权重**（client/public/models/seq_sentence/），
 * 在 tfjs 里搭同构网络填进去，喂一条确定性输入，与 Python 侧同一条输入的输出逐位比。
 * 公式写在 fixture 的 `ramp.formula` 里，两边逐字实现同一个式子。
 *
 * 这个测试依赖构建产物存在。产物是 Python 训练的输出，不在版本控制的常规路径上，
 * 所以缺文件时 **skip 而不是 fail** —— 但 fixture 在库里，缺产物时至少解码那部分
 * （ctcDecode.test）仍然在跑。
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import fixture from "./sentenceCtcFixture.json";
import { greedyDecode } from "./ctcDecode";
import {
  SentenceModelMissingError,
  loadSentenceModel,
  sentenceModelAvailable,
} from "./sentenceModel";

const MODEL_DIR = resolve(__dirname, "../../public/models/seq_sentence");
const hasArtifacts =
  existsSync(`${MODEL_DIR}/weights.json`) && existsSync(`${MODEL_DIR}/weights.bin`);

/** 把 fetch 接到本地文件系统上。顺带覆盖 loadSentenceModel 自己的校验逻辑 */
function stubFetch(overrides: Record<string, Response | "404"> = {}) {
  vi.stubGlobal("fetch", async (url: string) => {
    if (overrides[url] === "404") return new Response(null, { status: 404 });
    if (overrides[url]) return overrides[url] as Response;
    const name = url.split("/").pop()!;
    const path = `${MODEL_DIR}/${name}`;
    if (!existsSync(path)) return new Response(null, { status: 404 });
    const buf = readFileSync(path);
    return new Response(new Uint8Array(buf), { status: 200 });
  });
}

/** x[t][d] = ((t*frameDim + d) % 97) / 97 —— 与 export_fixture.py 的 ramp_input 同一个式子 */
function rampInput(seqLen: number, frameDim: number, mod: number): Float32Array {
  const out = new Float32Array(seqLen * frameDim);
  for (let i = 0; i < out.length; i++) out[i] = (i % mod) / mod;
  return out;
}

describe.skipIf(!hasArtifacts)("loadSentenceModel（需要构建产物）", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("加载成功，meta 与 fixture 一致", async () => {
    stubFetch();
    const loaded = await loadSentenceModel("/models/seq_sentence");
    try {
      // fixture 和产物是同一次导出的两个产物；对不上说明其中一个忘了重新生成，
      // 后面所有数值对照都失去意义
      expect(loaded.meta.labels).toEqual(fixture.labels);
      expect(loaded.meta.blankIndex).toBe(fixture.blankIndex);
      expect(loaded.meta.seqLen).toBe(fixture.seqLen);
      expect(loaded.meta.outputFrames).toBe(fixture.outputFrames);
      expect(loaded.meta.ctc).toBe(true);
    } finally {
      loaded.dispose();
    }
  });

  it("确定性输入的前向输出与 keras 逐位一致", async () => {
    stubFetch();
    const loaded = await loadSentenceModel("/models/seq_sentence");
    try {
      const { seqLen, frameDim, outputFrames } = loaded.meta;
      const numClasses = loaded.meta.labels.length + 1;
      const x = rampInput(seqLen, frameDim, fixture.ramp.mod);
      const got = tf.tidy(() => {
        const t = tf.tensor3d(x, [1, seqLen, frameDim]);
        return (loaded.model.predict(t) as tf.Tensor3D).dataSync() as Float32Array;
      });

      const want = new Float32Array(fixture.ramp.probs.flat());
      expect(got.length).toBe(want.length);
      let maxDiff = 0;
      for (let i = 0; i < want.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(got[i] - want[i]));
      }
      // 实测 5.0e-6（float32 逐层累积 + fixture 存 6 位小数的量化误差）。
      // 门限取 5e-5，留 10 倍余量，同时远小于要抓的那几种失配：
      // BN epsilon 用错差 1e-2 以上，权重错位差 0.1 以上
      expect(maxDiff).toBeLessThan(5e-5);

      // 更硬的一条：解码结果必须**完全**相同。数值差再小，只要跨过某一帧的
      // argmax 边界，句子就会多词或少词 —— 这才是最终关心的东西
      expect(greedyDecode(got, outputFrames, numClasses, loaded.meta.blankIndex)).toEqual(
        fixture.ramp.decoded
      );
    } finally {
      loaded.dispose();
    }
  });

  it("每一帧概率和为 1（确认 head 的 softmax 作用在最后一维，不是整块）", async () => {
    // Dense 作用在 3D 输入的最后一维、softmax 也是。如果 tfjs 把 softmax 作用在
    // 整个 (T,C) 上，概率和会变成 1/T —— 解码结果不变（argmax 不受影响），
    // 但置信度全错，UI 上会显示成"每个词都只有 3% 置信度"
    stubFetch();
    const loaded = await loadSentenceModel("/models/seq_sentence");
    try {
      const { seqLen, frameDim, outputFrames } = loaded.meta;
      const numClasses = loaded.meta.labels.length + 1;
      const p = tf.tidy(() => {
        const t = tf.tensor3d(rampInput(seqLen, frameDim, fixture.ramp.mod), [1, seqLen, frameDim]);
        return (loaded.model.predict(t) as tf.Tensor3D).dataSync() as Float32Array;
      });
      for (let f = 0; f < outputFrames; f++) {
        let s = 0;
        for (let c = 0; c < numClasses; c++) s += p[f * numClasses + c];
        expect(s).toBeCloseTo(1, 4);
      }
    } finally {
      loaded.dispose();
    }
  });

  it("blankIndex 与类别数不符时拒绝加载", async () => {
    // 这是"Python 侧 head 改了但浏览器没跟上"的唯一防线。放过去的话
    // 每个词都会被翻译成另一个词，而且置信度看着很正常
    const bad = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf-8"));
    bad.meta.blankIndex = 3;
    stubFetch({
      "/models/seq_sentence/weights.json": new Response(JSON.stringify(bad), { status: 200 }),
    });
    await expect(loadSentenceModel("/models/seq_sentence")).rejects.toThrow(/blank 必须在末位/);
  });

  it("缺某一层权重时拒绝加载，而不是让那层带着随机初始化跑", async () => {
    const bad = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf-8"));
    const dropped = bad.layers.pop().name;
    stubFetch({
      "/models/seq_sentence/weights.json": new Response(JSON.stringify(bad), { status: 200 }),
    });
    await expect(loadSentenceModel("/models/seq_sentence")).rejects.toThrow(
      new RegExp(`缺这些层的权重.*${dropped}`)
    );
  });

  it("形状不符时报出是哪一层", async () => {
    const bad = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf-8"));
    bad.layers[0].weights[0].shape = [5, 294, 63];
    stubFetch({
      "/models/seq_sentence/weights.json": new Response(JSON.stringify(bad), { status: 200 }),
    });
    await expect(loadSentenceModel("/models/seq_sentence")).rejects.toThrow(
      /层 "student_b1_conv".*形状不符/
    );
  });
});

describe("没部署模型时的行为", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("404 抛 SentenceModelMissingError，文案里带下一步怎么做", async () => {
    vi.stubGlobal("fetch", async () => new Response(null, { status: 404 }));
    await expect(loadSentenceModel("/models/nope")).rejects.toThrow(SentenceModelMissingError);
    await expect(loadSentenceModel("/models/nope")).rejects.toThrow(/train_seq\.py --ctc/);
  });

  it("sentenceModelAvailable 对 404 返回 false（不抛）", async () => {
    vi.stubGlobal("fetch", async () => new Response(null, { status: 404 }));
    await expect(sentenceModelAvailable("/models/nope")).resolves.toBe(false);
  });

  it("500 不能被当成'没部署' —— 那是坏了，要让人看见", async () => {
    vi.stubGlobal("fetch", async () => new Response(null, { status: 500 }));
    await expect(sentenceModelAvailable("/models/nope")).rejects.toThrow(/HTTP 500/);
  });
});
