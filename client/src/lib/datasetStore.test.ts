/*
 * datasetStore.test —— 孤立词 / 句子两条链路的分界
 *
 * `getSequenceStats` 走 IndexedDB，这里测不了（仓库里没装 fake-indexeddb）。
 * 但真正会静默出事的不是聚合，而是**判据本身**：`isSentenceSample` 是孤立词训练、
 * 词频表、CTC 训练三处共用的那一条线。它判错的后果是一条多词录制作为它的第一个词
 * 进 softmax，而条数、覆盖率、裁剪汇总全都显示正常 —— 症状只表现为"某个词莫名变差"。
 *
 * 所以这里锁两件事：判据在 0/1/多 segment 上的行为，以及孤立词训练器在"全是句子"
 * 时必须**明确报错**而不是训出一个空模型。
 */
import { describe, expect, it } from "vitest";
import { isSentenceSample } from "./datasetStore";
import { trainSequenceModel } from "./sequenceModel";
import type { SequenceSample } from "./datasetStore";

const seg = (label: string, startFrame: number, endFrame: number) => ({
  label,
  startFrame,
  endFrame,
});

/** 最小可用的样本骨架 —— 这些测试只看 segments，其他字段够类型过就行 */
function sample(segments: Array<ReturnType<typeof seg>>): SequenceSample {
  return {
    segments,
    primaryLabel: segments[0]?.label ?? "",
    frameCount: 4,
    timestamps: new Float32Array([0, 20, 40, 60]),
    leftSensor: null,
    rightSensor: new Uint8Array(4 * 137),
    leftImu: null,
    rightImu: new Float32Array(4 * 10),
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: 80,
    sourceFps: 50,
    origin: "recorded",
    timestamp: 0,
  };
}

describe("isSentenceSample", () => {
  it("单 segment 是孤立词", () => {
    expect(isSentenceSample(sample([seg("i", 0, 4)]))).toBe(false);
  });

  it("多 segment 是句子", () => {
    expect(
      isSentenceSample(sample([seg("i", 0, 2), seg("name", 2, 4)]))
    ).toBe(true);
  });

  it("空 segments 不算句子", () => {
    // 判据是"多于一个词"，不是"不等于一个词"。空数组是坏数据，
    // 判成句子会把它塞进 CTC 那条路，那里更没法处理
    expect(isSentenceSample(sample([]))).toBe(false);
  });
});

describe("孤立词训练器挡句子样本", () => {
  it("全是句子样本时明确报错，并指向 python 的 CTC 路径", async () => {
    const all = [
      sample([seg("i", 0, 2), seg("name", 2, 4)]),
      sample([seg("you", 0, 2), seg("eat", 2, 4)]),
    ];
    // 不能训出一个"2 类"的模型 —— 那两个类会是 i 和 you，而样本里各含两个词
    await expect(trainSequenceModel(all)).rejects.toThrow(/句子级样本/);
  });

  it("空数组仍走原来的报错（不要被句子过滤改掉）", async () => {
    await expect(trainSequenceModel([])).rejects.toThrow(/没有序列样本/);
  });
});
