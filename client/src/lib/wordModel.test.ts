/*
 * wordModel.test —— 部署的孤立词模型：权重加载 + 前向计算与 keras 对照
 *
 * 与 `sentenceModel.test` 同一个目的：孤立词模型从今天起也走"浏览器自己搭结构 +
 * 填权重"这条路，于是继承了同一类失配 —— tfjs 与 keras 的层实现不完全一致
 * （BN 的 epsilon 默认值、conv1d `padding:"same"` 的补零方式，孤立词这边还多一个
 * GlobalAveragePooling1D 的轴）。这类失配不抛异常，只让输出偏一点 ——
 * 偏一点就足够让 29 类的 argmax 换人，线上表现是"能跑、有置信度、词全错"。
 *
 * **主断言不是 probs 而是 `student_fc` 的输出。** ramp 输入远在真实特征分布之外，
 * 模型对它必然满置信（实测各种 mod 下最大概率 0.9985~1.0，熵近 0）。softmax 饱和
 * 之后 BN epsilon 用错也只差 1e-7，逐位断言照样通过 = 测试形同没写。
 * `student_fc` 是 64 维 relu、没被压扁，上游任何失配都原样体现。
 * 详见 `python_train/export_word_fixture.py` 的文件头。
 *
 * 依赖构建产物（Python 训练的输出，不在版本控制的常规路径上），缺文件时
 * **skip 而不是 fail**。
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import fixture from "./wordModelFixture.json";
import { TACTILE_FRAME_DIM } from "./sequenceFeatures";
import {
  getLoadedSequenceLabels,
  getLoadedSeqLen,
  isSequenceModelLoaded,
} from "./sequenceModel";
import { ModelMissingError } from "./modelWeights";
import {
  buildDeployedWordModel,
  loadDeployedWordModel,
  wordModelAvailable,
  fetchWordModelMeta,
} from "./wordModel";

/*
 * 拿模型对象的用例走 `buildDeployedWordModel`（**不碰全局单例**，调用方自己 dispose）；
 * 只有验证"接进单例"那一条走 `loadDeployedWordModel`。
 *
 * 两个都用后者、再在 finally 里 dispose 的话，模型会同时被单例和用例持有：
 * 下一条用例加载时 `setActiveSequenceModel` 去 dispose 上一个已经 dispose 过的容器，
 * 抛 "Container 'student' is already disposed" —— 而报错的用例是**下一条**，
 * 与真正多调 dispose 的那条隔着好远。
 */

const MODEL_DIR = resolve(__dirname, "../../public/models/seq_student");
const hasArtifacts =
  existsSync(`${MODEL_DIR}/weights.json`) && existsSync(`${MODEL_DIR}/weights.bin`);

/** 把 fetch 接到本地文件系统上 */
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

/** x[t][d] = ((t*frameDim + d) % mod) / mod —— 与 export_word_fixture.py 的 ramp_input 同一个式子 */
function rampInput(seqLen: number, frameDim: number, mod: number): Float32Array {
  const out = new Float32Array(seqLen * frameDim);
  for (let i = 0; i < out.length; i++) out[i] = (i % mod) / mod;
  return out;
}

describe.skipIf(!hasArtifacts)("loadDeployedWordModel（需要构建产物）", () => {
  afterEach(() => vi.unstubAllGlobals());

  /**
   * **这条必须排在数值对照前面。**
   *
   * fixture 和 weights.bin 是同一次训练的两个产物，但由两个脚本分别生成
   * （export_word_fixture.py / export_weights.py）。重训之后只跑了后者的话，
   * 下面的数值对照会报一个很大的 maxDiff —— 而那条断言的注释把偏差解释成
   * "BN epsilon 用错 / 权重错位"，排查方向会被直接带偏到层实现上去。
   */
  it("fixture 与部署的权重出自同一次训练", () => {
    const meta = (
      JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf8")) as {
        meta: Record<string, unknown>;
      }
    ).meta;
    const want = fixture.provenance.trainMeta as Record<string, unknown>;
    const got = Object.fromEntries(Object.keys(want).map((k) => [k, meta[k]]));
    expect(
      got,
      "fixture 与 client/public/models/seq_student/ 的权重来自不同的训练。" +
        "重训 + 导出之后要跟着跑 `python export_word_fixture.py`（在 python_train/ 下）。" +
        "不要去查 tfjs 的层实现 —— 下面那条 maxDiff 断言此时必然失败，但那是果不是因。"
    ).toEqual(want);
  });

  /* 唯一一条走 loadDeployedWordModel 的用例 —— 模型留给单例，这里不 dispose */
  it("加载成功，并接进全局单例（滑窗推理走的就是这个出口）", async () => {
    stubFetch();
    const meta = await loadDeployedWordModel("/models/seq_student");
    expect(meta.labels).toEqual(fixture.labels);
    expect(meta.seqLen).toBe(fixture.seqLen);
    expect(meta.frameDim).toBe(TACTILE_FRAME_DIM);
    // 这才是这个文件存在的理由：predictSequence 读的是全局单例
    expect(isSequenceModelLoaded()).toBe(true);
    expect(getLoadedSequenceLabels()).toEqual(fixture.labels);
    expect(getLoadedSeqLen()).toBe(fixture.seqLen);
  });

  it("标签表含 08-28/29 那批新词（这次部署要解决的就是它们认不出来）", async () => {
    stubFetch();
    const { model, meta } = await buildDeployedWordModel("/models/seq_student");
    try {
      for (const w of ["i", "beautiful", "name", "resemble", "smile", "sun"]) {
        expect(meta.labels, `新词 ${w} 不在标签表里 —— 导出的是旧模型？`).toContain(w);
      }
      // 不是 CTC：head 把时间维塌缩掉，输出是 [1, numClasses] 而不是 [1, T, numClasses+1]
      expect(model.outputShape).toEqual([null, fixture.numClasses]);
    } finally {
      model.dispose();
    }
  });

  /**
   * 主断言。对照的是 `student_fc` 的 64 维 relu 输出，不是 probs ——
   * 理由见文件头（ramp 下 softmax 必然饱和）。
   *
   * 容差：fixture 存 6 位小数，而 fc 幅度是 O(10)（实测 max 18.75），
   * 所以量化误差本身就有 5e-7。取 1e-3 是"层实现一致但浮点顺序不同"的合理上限；
   * 真正的失配（BN epsilon、GAP 轴错、权重错位）会差 0.1 以上，不会卡在这个门限附近。
   */
  it("确定性输入下 student_fc 的输出与 keras 一致", async () => {
    stubFetch();
    const { model } = await buildDeployedWordModel("/models/seq_student");
    try {
      const fcLayer = model.getLayer(fixture.ramp.fcLayer);
      const sub = tf.model({ inputs: model.inputs, outputs: fcLayer.output as tf.SymbolicTensor });
      const got = tf.tidy(() => {
        const x = tf.tensor3d(
          rampInput(fixture.seqLen, fixture.frameDim, fixture.ramp.mod),
          [1, fixture.seqLen, fixture.frameDim]
        );
        return Array.from((sub.predict(x) as tf.Tensor2D).dataSync());
      });
      const want = fixture.ramp.fc as number[];
      expect(got.length).toBe(want.length);
      let maxDiff = 0;
      for (let i = 0; i < want.length; i++) {
        maxDiff = Math.max(maxDiff, Math.abs(got[i] - want[i]));
      }
      expect(
        maxDiff,
        `student_fc 输出与 keras 差 ${maxDiff}。差 0.1 以上时先查：` +
          `BatchNormalization 的 epsilon（keras 3 默认 1e-3，tfjs 默认 1e-3 —— 但改过就会漂）、` +
          `GlobalAveragePooling1D 的轴、conv1d padding:"same" 的补零方式、` +
          `以及 export_weights.py 的层顺序。`
      ).toBeLessThan(1e-3);
    } finally {
      model.dispose();
    }
  });

  it("softmax 输出概率和为 1，且 argmax 与 keras 同一个词", async () => {
    stubFetch();
    const { model, meta } = await buildDeployedWordModel("/models/seq_student");
    try {
      const probs = tf.tidy(() => {
        const x = tf.tensor3d(
          rampInput(fixture.seqLen, fixture.frameDim, fixture.ramp.mod),
          [1, fixture.seqLen, fixture.frameDim]
        );
        return Array.from((model.predict(x) as tf.Tensor2D).dataSync());
      });
      const sum = probs.reduce((a, b) => a + b, 0);
      expect(Math.abs(sum - 1)).toBeLessThan(1e-4);
      let top = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[top]) top = i;
      // ramp 下饱和，所以这条只是弱确认（真正的灵敏断言在上一条 fc 上）
      expect(meta.labels[top]).toBe(fixture.ramp.argmaxLabel);
    } finally {
      model.dispose();
    }
  });

  it("特征维度不符时拒绝加载，而不是让形状错位一路跑下去", async () => {
    const raw = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf8"));
    raw.meta.frameDim = 999;
    stubFetch({
      "/models/seq_student/weights.json": new Response(JSON.stringify(raw), { status: 200 }),
    });
    await expect(loadDeployedWordModel("/models/seq_student")).rejects.toThrow(/999/);
  });

  it("把 CTC 句子模型放进这个目录时说清是目录搞反了", async () => {
    const raw = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf8"));
    raw.meta.ctc = true;
    stubFetch({
      "/models/seq_student/weights.json": new Response(JSON.stringify(raw), { status: 200 }),
    });
    await expect(loadDeployedWordModel("/models/seq_student")).rejects.toThrow(/目录搞反/);
  });

  it("缺某一层权重时拒绝加载，而不是让那层带着随机初始化跑", async () => {
    const raw = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf8"));
    raw.layers = raw.layers.filter((l: { name: string }) => l.name !== "student_b2_bn");
    stubFetch({
      "/models/seq_student/weights.json": new Response(JSON.stringify(raw), { status: 200 }),
    });
    await expect(loadDeployedWordModel("/models/seq_student")).rejects.toThrow(/student_b2_bn/);
  });

  it("形状不符时报出是哪一层", async () => {
    const raw = JSON.parse(readFileSync(`${MODEL_DIR}/weights.json`, "utf8"));
    const layer = raw.layers.find((l: { name: string }) => l.name === "student_fc");
    layer.weights[0].shape = [128, 63];
    stubFetch({
      "/models/seq_student/weights.json": new Response(JSON.stringify(raw), { status: 200 }),
    });
    await expect(loadDeployedWordModel("/models/seq_student")).rejects.toThrow(/student_fc/);
  });
});

describe("没部署词模型时的行为", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("404 抛 ModelMissingError，文案里带下一步怎么做", async () => {
    stubFetch({ "/models/seq_student/weights.json": "404" });
    await expect(loadDeployedWordModel("/models/seq_student")).rejects.toThrow(
      ModelMissingError
    );
    await expect(loadDeployedWordModel("/models/seq_student")).rejects.toThrow(
      /export_weights\.py/
    );
  });

  it("wordModelAvailable 对 404 返回 false（不抛）", async () => {
    stubFetch({ "/models/seq_student/weights.json": "404" });
    await expect(wordModelAvailable("/models/seq_student")).resolves.toBe(false);
  });

  /*
   * 500 不能被当成"没部署"。那会让 UI 显示"先去跑 python_train"，
   * 而用户明明已经跑过了 —— 真正的原因（代理、CORS、静态服务挂了）被藏起来。
   */
  it("500 不能被当成'没部署' —— 那是坏了，要让人看见", async () => {
    stubFetch({
      "/models/seq_student/weights.json": new Response(null, { status: 500 }),
    });
    await expect(wordModelAvailable("/models/seq_student")).rejects.toThrow(/HTTP 500/);
  });

  it("fetchWordModelMeta 不下 weights.bin", async () => {
    let binFetched = false;
    vi.stubGlobal("fetch", async (url: string) => {
      if (url.endsWith("weights.bin")) binFetched = true;
      const name = url.split("/").pop()!;
      const path = `${MODEL_DIR}/${name}`;
      if (!existsSync(path)) return new Response(null, { status: 404 });
      return new Response(new Uint8Array(readFileSync(path)), { status: 200 });
    });
    if (!hasArtifacts) return;
    const meta = await fetchWordModelMeta("/models/seq_student");
    expect(meta.labels.length).toBeGreaterThan(0);
    expect(binFetched, "选型只需要 meta，不该顺手下 651KB 的权重").toBe(false);
  });
});
