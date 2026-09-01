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
import { isSentenceSample, sampleTemplateKey } from "./datasetStore";
import { trainSequenceModel } from "./sequenceModel";
import { BATCH1_TEMPLATES, templateKey } from "./sentenceTemplates";
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

/*
 * 采集页按句型显示 N/20 进度、并按句型列样本。两件事都靠这个 key，
 * 而 `getSentencesByTemplate` 的实现是「用 primaryLabel 索引拿候选 + isSentenceSample
 * + key 相等」。key 算错的后果是进度永远停在 0/20（人会一直采下去），
 * 或者反过来把孤立词当成句子列出来（删除按钮删到孤立词上）。
 */
describe("sampleTemplateKey", () => {
  it("和句型表用的是同一个 key 格式", () => {
    // 这一条是采集页能查到计数的**前提**：页面手里拿的是句型表的 templateKey，
    // 库里存的是样本算出来的 sampleTemplateKey，两个必须逐字节相同
    for (const t of BATCH1_TEMPLATES) {
      const s = sample(t.map((label, i) => seg(label, i, i + 1)));
      expect(sampleTemplateKey(s)).toBe(templateKey(t));
    }
  });

  it("词序不同就是不同句型", () => {
    const a = sample([seg("i", 0, 1), seg("love", 1, 2), seg("you", 2, 3)]);
    const b = sample([seg("you", 0, 1), seg("love", 1, 2), seg("i", 2, 3)]);
    expect(sampleTemplateKey(a)).not.toBe(sampleTemplateKey(b));
  });

  it("孤立词样本的 key 撞不上以它开头的句型", () => {
    // 句子的 primaryLabel 就是第一个词，所以按 primaryLabel 查「i love you」
    // 会把所有 `i` 的孤立词一起捞出来 —— 靠 key 相等把它们筛掉
    expect(sampleTemplateKey(sample([seg("i", 0, 4)]))).not.toBe(
      templateKey(["i", "love", "you"])
    );
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
