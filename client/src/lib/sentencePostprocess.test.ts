import { describe, expect, it } from "vitest";

import { SEQ_SENSOR_N } from "./datasetStore";
import type { SequenceSample } from "./datasetStore";
import { greedyDecode, greedyDecodeSpans } from "./ctcDecode";
import { mergeCompoundsInWords } from "./compoundWords";
import { THUMB_PEAK_GATE, thumbPeakSpan } from "./fistGate";
import { postprocessSentence } from "./sentencePostprocess";
import { SUPPRESSED_WORDS } from "./suppressedWords";
import type { SentencePrediction } from "./sentenceModel";
import { RH_THUMB_PRESSURE_RANGE } from "./fistGate";

/**
 * 这一批测试的目的不是覆盖率，是钉住三个**会静默出错**的地方：
 *   1. 两条解码路径（`greedyDecode` / `greedyDecodeSpans`）的词序列必须一致 ——
 *      分岔时区间错位一格，闸门去看别的词的帧，输出看着像模型判错了。
 *   2. 闸门必须**逐词**取拇指峰值。整句取一个全局峰值会把本来对的词改错。
 *   3. 先闸门后合并的顺序 —— 反了「你好」永远合不出来。
 */

const LABELS = ["you", "thank_you", "sad", "resemble", "beautiful", "eat"];
const BLANK = LABELS.length;
const C = LABELS.length + 1;

/** 造逐帧概率：`frames` 里每一项是那一帧的 argmax 下标（`BLANK` 表示空白） */
function perFrameFrom(frames: number[], strong = 0.8): Float32Array {
  const out = new Float32Array(frames.length * C);
  frames.forEach((top, t) => {
    const rest = (1 - strong) / (C - 1);
    for (let c = 0; c < C; c++) out[t * C + c] = rest;
    out[t * C + top] = strong;
  });
  return out;
}

/** 把某一帧的第二名钉成指定类别（闸门要求 top-2 恰好是 谢谢/难过 那一对） */
function setRunnerUp(pf: Float32Array, frame: number, cls: number, p = 0.15) {
  pf[frame * C + cls] = p;
}

function predFrom(frames: number[], pf?: Float32Array): SentencePrediction {
  const perFrame = pf ?? perFrameFrom(frames);
  const indices = greedyDecode(perFrame, frames.length, C, BLANK);
  const spans = greedyDecodeSpans(perFrame, frames.length, C, BLANK);
  return {
    words: indices.map((i) => LABELS[i]),
    indices,
    spans,
    perFrame,
    outputFrames: frames.length,
    numClasses: C,
    blankIndex: BLANK,
  };
}

/**
 * 造一条样本，只有右手拇指压力有值：`presses` 里每一项是 `[起帧, 止帧, 压力值]`。
 * 其余通道留 0 —— 后处理只读拇指那 12 个点。
 */
function sampleWithThumb(
  frameCount: number,
  presses: [number, number, number][]
): SequenceSample {
  const rightSensor = new Uint8Array(frameCount * SEQ_SENSOR_N);
  for (const [a, b, v] of presses) {
    for (let t = a; t < b; t++) {
      for (let k = RH_THUMB_PRESSURE_RANGE.start; k < RH_THUMB_PRESSURE_RANGE.end; k++) {
        rightSensor[t * SEQ_SENSOR_N + k] = v;
      }
    }
  }
  return {
    frameCount,
    durationMs: frameCount * 20,
    rightSensor,
    leftSensor: null,
    rightImu: null,
    leftImu: null,
    rightLandmarks: null,
    leftLandmarks: null,
    timestamps: Float32Array.from({ length: frameCount }, (_, i) => i * 20),
  } as unknown as SequenceSample;
}

describe("greedyDecodeSpans", () => {
  it("词序列与 greedyDecode 逐项相同（分岔了闸门就会看错帧）", () => {
    const cases: number[][] = [
      [],
      [BLANK, BLANK],
      [0, 0, BLANK, 0],
      [0, 1, 1, BLANK, 2],
      [3, 3, 3, 1, 1, BLANK, BLANK, 0],
      [BLANK, 0, BLANK, 1, BLANK, 2, BLANK],
    ];
    for (const frames of cases) {
      const pf = perFrameFrom(frames);
      const a = greedyDecode(pf, frames.length, C, BLANK);
      const b = greedyDecodeSpans(pf, frames.length, C, BLANK).map((s) => s.index);
      expect(b).toEqual(a);
    }
  });

  it("区间是折叠前的 argmax 连续段；blank 分隔的重复算两段", () => {
    const frames = [0, 0, BLANK, 0];
    const spans = greedyDecodeSpans(perFrameFrom(frames), frames.length, C, BLANK);
    expect(spans.map((s) => [s.index, s.startFrame, s.endFrame])).toEqual([
      [0, 0, 2],
      [0, 3, 4],
    ]);
  });

  it("peakFrame 落在段内概率最高的那一帧", () => {
    const frames = [1, 1, 1];
    const pf = perFrameFrom(frames, 0.6);
    pf[1 * C + 1] = 0.9;
    const spans = greedyDecodeSpans(pf, frames.length, C, BLANK);
    expect(spans[0].peakFrame).toBe(1);
  });
});

describe("thumbPeakSpan", () => {
  it("只看指定的那一段时间", () => {
    const s = sampleWithThumb(10, [[0, 3, 30]]);
    expect(thumbPeakSpan(s, 0, 0.3)).toBe(30);
    expect(thumbPeakSpan(s, 0.5, 1)).toBe(0);
    expect(thumbPeakSpan(s, 0, 1)).toBe(30);
  });

  it("区间退化时夹到至少一帧，而不是返回 -1（-1 的含义是没有右手数据）", () => {
    const s = sampleWithThumb(10, [[0, 10, 7]]);
    expect(thumbPeakSpan(s, 0.5, 0.5)).toBe(7);
    expect(thumbPeakSpan(s, 2, 3)).toBe(7);
  });

  it("没有右手数据时才返回 -1", () => {
    const s = sampleWithThumb(4, []);
    expect(thumbPeakSpan(s, 0, 1)).toBe(0); // 0 是合法读数：拇指没吃力
    const noHand = { ...s, rightSensor: null } as unknown as SequenceSample;
    expect(thumbPeakSpan(noHand, 0, 1)).toBe(-1);
  });
});

describe("mergeCompoundsInWords", () => {
  it("你 + 谢谢 → 你好", () => {
    expect(mergeCompoundsInWords(["you", "thank_you"])).toEqual(["hello"]);
  });

  it("像 + 谢谢 → 好看，且句子里其它词原样保留", () => {
    expect(mergeCompoundsInWords(["you", "is", "resemble", "thank_you"])).toEqual([
      "you",
      "is",
      "beautiful",
    ]);
  });

  it("合过的不再参与下一次匹配（不级联）", () => {
    expect(mergeCompoundsInWords(["you", "thank_you", "thank_you"])).toEqual([
      "hello",
      "thank_you",
    ]);
  });

  it("反序不合 —— 谢谢你 是一句真话", () => {
    expect(mergeCompoundsInWords(["thank_you", "you"])).toEqual(["thank_you", "you"]);
  });

  it("没有规则命中时原序列不变（含空序列）", () => {
    expect(mergeCompoundsInWords([])).toEqual([]);
    expect(mergeCompoundsInWords(["i", "sad"])).toEqual(["i", "sad"]);
  });
});

describe("postprocessSentence", () => {
  it("你 + 谢谢（拇指吃力）→ 你好", () => {
    // 帧 0-1 = you，帧 2-3 = thank_you；拇指在后半段吃力
    const frames = [0, 0, 1, 1];
    const pf = perFrameFrom(frames);
    setRunnerUp(pf, 2, 2); // thank_you 的第二名是 sad → 进闸门
    setRunnerUp(pf, 3, 2);
    const pred = predFrom(frames, pf);
    const sample = sampleWithThumb(8, [[4, 8, THUMB_PEAK_GATE + 5]]);
    const r = postprocessSentence(pred, sample, LABELS);
    expect(r.rawWords).toEqual(["you", "thank_you"]);
    expect(r.words).toEqual(["hello"]);
    expect(r.gated).toEqual([]); // 拇指吃力，闸门不改判
    expect(r.merged).toEqual(["hello"]);
  });

  it("你 + 难过（拇指吃力）→ 闸门扳成谢谢 → 合成你好", () => {
    const frames = [0, 0, 2, 2]; // 第二段模型判成 sad
    const pf = perFrameFrom(frames);
    setRunnerUp(pf, 2, 1);
    setRunnerUp(pf, 3, 1);
    const pred = predFrom(frames, pf);
    const sample = sampleWithThumb(8, [[4, 8, THUMB_PEAK_GATE + 5]]);
    const r = postprocessSentence(pred, sample, LABELS);
    expect(r.rawWords).toEqual(["you", "sad"]);
    expect(r.gated.map((g) => [g.from, g.to])).toEqual([["sad", "thank_you"]]);
    expect(r.words).toEqual(["hello"]);
  });

  it("拇指全程不吃力时不合成 —— 「你 难过」是一句真话", () => {
    const frames = [0, 0, 2, 2];
    const pf = perFrameFrom(frames);
    setRunnerUp(pf, 2, 1);
    setRunnerUp(pf, 3, 1);
    const pred = predFrom(frames, pf);
    const sample = sampleWithThumb(8, []);
    const r = postprocessSentence(pred, sample, LABELS);
    expect(r.gated).toEqual([]);
    expect(r.words).toEqual(["you", "sad"]);
  });

  /**
   * 这条是整个改动最要紧的一条：句子里**别的词**把拇指压满了，
   * 不能因此把「难过」翻成「谢谢」。实测「好看」全段拇指峰值中位 26。
   */
  it("闸门只看该词自己那几帧，不受句子里其它词的拇指压力影响", () => {
    // 帧 0-1 = beautiful（拇指压满），帧 2-3 = sad（拇指为 0）
    const frames = [4, 4, 2, 2];
    const pf = perFrameFrom(frames);
    setRunnerUp(pf, 2, 1);
    setRunnerUp(pf, 3, 1);
    const pred = predFrom(frames, pf);
    const sample = sampleWithThumb(8, [[0, 4, 26]]); // 只有前半段有压力
    const r = postprocessSentence(pred, sample, LABELS);
    expect(r.gated).toEqual([]);
    expect(r.words).toEqual(["beautiful", "sad"]);
    // 反证：同一条样本按整句取峰值就会误判
    expect(thumbPeakSpan(sample, 0, 1)).toBeGreaterThanOrEqual(THUMB_PEAK_GATE);
    expect(thumbPeakSpan(sample, 0.5, 1)).toBe(0);
  });

  it("top-2 不是被仲裁的那一对时不介入（第二名是 blank 也不算）", () => {
    const frames = [2, 2]; // sad，第二名默认是别的词而不是 thank_you
    const pf = perFrameFrom(frames);
    setRunnerUp(pf, 0, 5); // eat
    setRunnerUp(pf, 1, 5);
    const pred = predFrom(frames, pf);
    const sample = sampleWithThumb(4, [[0, 4, 99]]);
    const r = postprocessSentence(pred, sample, LABELS);
    expect(r.gated).toEqual([]);
    expect(r.words).toEqual(["sad"]);
  });

  it("空解码（全 blank）时原样返回空序列", () => {
    const frames = [BLANK, BLANK, BLANK];
    const pred = predFrom(frames);
    const r = postprocessSentence(pred, sampleWithThumb(6, []), LABELS);
    expect(r.words).toEqual([]);
    expect(r.gated).toEqual([]);
    expect(r.merged).toEqual([]);
  });

  it("模型直接解出的 hello 不算「合成」", () => {
    const pred: SentencePrediction = {
      ...predFrom([0]),
      words: ["hello"],
      indices: [0],
    };
    const r = postprocessSentence(pred, sampleWithThumb(2, []), LABELS);
    expect(r.words).toEqual(["hello"]);
    expect(r.merged).toEqual([]);
  });
});

/*
 * 输出屏蔽（2026-08-31 加）。`eat` 现在在 `SUPPRESSED_WORDS` 里 ——
 * 大量「你」被解成「吃」，先把它从显示里拿掉。
 * 这是**显示层**的表，不是训练端的 `UNTRAINED_WORDS`。
 */
describe("SUPPRESSED_WORDS", () => {
  it("eat 在表里（这一条是提醒：它开着）", () => {
    expect(SUPPRESSED_WORDS).toContain("eat");
  });

  it("被屏蔽的词不进 words，但要报进 suppressed", () => {
    const frames = [0, BLANK, 5]; // you, eat
    const pred = predFrom(frames);
    const r = postprocessSentence(pred, sampleWithThumb(6, []), LABELS);
    expect(r.words).toEqual(["you"]);
    expect(r.suppressed).toEqual(["eat"]);
    // rawWords 是**模型原始输出**，屏蔽不能动它 —— 那是"模型到底说了什么"的唯一记录
    expect(r.rawWords).toEqual(["you", "eat"]);
  });

  it("同一句里出现两次就报两次（不去重）", () => {
    const frames = [5, BLANK, 5];
    const pred = predFrom(frames);
    const r = postprocessSentence(pred, sampleWithThumb(6, []), LABELS);
    expect(r.words).toEqual([]);
    expect(r.suppressed).toEqual(["eat", "eat"]);
  });

  it("没有被屏蔽的词时 suppressed 是空数组，其余字段逐位不变", () => {
    const frames = [0, BLANK, 3];
    const pred = predFrom(frames);
    const r = postprocessSentence(pred, sampleWithThumb(6, []), LABELS);
    expect(r.suppressed).toEqual([]);
    expect(r.words).toEqual(["you", "resemble"]);
  });

  it("屏蔽发生在合并之后 —— 复合词不会因为某个组成部分被屏蔽而合不出来", () => {
    /*
     * 今天 `eat` 不是任何复合词的组成部分，所以这一条锁的是**顺序**本身：
     * 「你 谢谢」合成「你好」这一步必须在屏蔽之前完成。
     * 反了的话，将来往表里加一个 `thank_you` 之类的词，「你好」会静默消失。
     */
    const frames = [0, BLANK, 1, BLANK, 5]; // you, thank_you, eat
    const pred = predFrom(frames);
    // 拇指吃了力 → 闸门不会把 thank_you 扳成 sad
    const r = postprocessSentence(pred, sampleWithThumb(10, [[0, 10, 99]]), LABELS);
    expect(r.words).toEqual(["hello"]);
    expect(r.merged).toEqual(["hello"]);
    expect(r.suppressed).toEqual(["eat"]);
  });
});
