/*
 * ctcDecode.test —— CTC 解码约定
 *
 * 这里锁的不是算法（greedy 解码就三行），而是**两边的约定一致**。三处约定错一处，
 * 整句就全错，而且都不会抛异常：
 *
 *   - blank 的位置（末位 vs 首位）
 *   - 折叠与去 blank 的先后顺序
 *   - 输出帧数用 T 还是 T/4
 *
 * 前半是行为断言。后半读 `sentenceCtcFixture.json`（由
 * `python_train/export_fixture.py` 从真实训练好的模型导出）逐条对照 Python 的解码
 * 结果 —— 这是唯一能抓住"两边约定不一致"的测试：手写用例里的概率是我自己造的，
 * 造的时候用的就是我以为的约定，自证不了。
 */
import { describe, expect, it } from "vitest";
import {
  decodeToWords,
  editDistance,
  greedyDecode,
  wordErrorRate,
} from "./ctcDecode";
import fixture from "./sentenceCtcFixture.json";

/** 把类别序列摊成 one-hot 概率：0.9 给目标，其余 0.01 —— 只要 argmax 对就行 */
function onehot(seq: number[], numClasses: number): Float32Array {
  const p = new Float32Array(seq.length * numClasses).fill(0.01);
  seq.forEach((k, t) => {
    p[t * numClasses + k] = 0.9;
  });
  return p;
}

describe("greedyDecode", () => {
  const C = 4; // 3 类 + blank
  const B = 3;
  const dec = (seq: number[], blank = B) =>
    greedyDecode(onehot(seq, C), seq.length, C, blank);

  it("折叠连续重复", () => {
    expect(dec([0, 0, 0, B, 1, 1])).toEqual([0, 1]);
  });

  it("跨 blank 的重复不折叠（重复词全靠这条）", () => {
    // 写错这条的症状：「我 爱 我」永远只解出一个「我」。这正是当前模型最主要的
    // 错例形态（见 fixture 里的 merged_pron_sg love merged_pron_sg），
    // 所以必须先排除"是解码器吞掉的"这个可能
    expect(dec([0, B, 0])).toEqual([0, 0]);
    expect(dec([2, B, 2, B, 2])).toEqual([2, 2, 2]);
  });

  it("全 blank 解出空序列，而不是一串 blank 下标", () => {
    expect(dec([B, B, B])).toEqual([]);
  });

  it("两端的 blank 不影响结果", () => {
    expect(dec([B, 2, B, 2, B])).toEqual([2, 2]);
  });

  it("blank 下标配错会把真类当 blank 丢掉", () => {
    // 演示后果：把 blank 当成 0 号类，[0,B,0] 就解成了 [B] ——
    // 不报错、有输出、全错。这就是为什么 blankIndex 必须从 meta 读而不是各自算
    expect(dec([0, B, 0], 0)).toEqual([B]);
  });

  it("frames 只吃前 frames 帧，多余的概率被忽略", () => {
    // 传 T 而不是 T/4 时会多读 3 倍的帧。这里锁住"以 frames 为准"，
    // 使得长度传错在 sentenceModel 里会表现为解码变长（可发现），而不是静默读越界内存
    const p = onehot([0, 1, 2, 0], C);
    expect(greedyDecode(p, 2, C, B)).toEqual([0, 1]);
  });

  it("概率长度不够时抛错", () => {
    expect(() => greedyDecode(new Float32Array(5), 4, 4, 3)).toThrow(/长度/);
  });

  it("frames 为 0 时给空序列", () => {
    expect(greedyDecode(new Float32Array(0), 0, 4, 3)).toEqual([]);
  });

  it("并列最大值取下标最小的那个（与 numpy argmax 一致）", () => {
    // numpy 的 argmax 在并列时返回第一个。实战里概率完全相等几乎不可能，
    // 但 fixture 对照要求逐位相同，规则得一样
    const p = new Float32Array([0.5, 0.5, 0, 0]);
    expect(greedyDecode(p, 1, 4, 3)).toEqual([0]);
  });
});

describe("decodeToWords", () => {
  it("翻词", () => {
    expect(decodeToWords([1, 0], ["a", "b"])).toEqual(["b", "a"]);
  });

  it("下标越界抛错而不是给 undefined", () => {
    // 静默给 undefined 的话，UI 上会显示成 "undefined" 或者空词条，
    // 排查方向会被带到 UI 上去，而真正的原因在 blankIndex
    expect(() => decodeToWords([2], ["a", "b"])).toThrow(/超出标签表/);
  });
});

describe("editDistance / wordErrorRate", () => {
  it("编辑距离三种操作", () => {
    expect(editDistance([], [])).toBe(0);
    expect(editDistance([1, 2, 3], [1, 2, 3])).toBe(0);
    expect(editDistance([1, 2, 3], [1, 3])).toBe(1); // 删
    expect(editDistance([1, 3], [1, 2, 3])).toBe(1); // 插
    expect(editDistance([1, 2, 3], [1, 9, 3])).toBe(1); // 替
    expect(editDistance([], [1, 2])).toBe(2);
  });

  it("WER 是总编辑距离 / 总参考长度，不是逐句平均", () => {
    expect(wordErrorRate([[1, 2, 3]], [[1, 2, 3]])).toBe(0);
    expect(wordErrorRate([[1, 2, 3]], [[]])).toBe(1);
    expect(wordErrorRate([[1, 2, 3], [4]], [[1, 3], [4]])).toBeCloseTo(0.25, 9);
  });

  it("参考全空时不除零", () => {
    expect(wordErrorRate([[]], [[1]])).toBe(1);
  });
});

describe("与 Python 对照（sentenceCtcFixture.json）", () => {
  const numClasses = fixture.numClasses + 1;
  const frames = fixture.outputFrames;

  it("fixture 的元信息自洽", () => {
    // blank 必须在末位。fixture 是从训练好的模型导出的，这条不成立就说明
    // Python 侧的 head 或 blank_index 变了，浏览器这边的假设整个作废
    expect(fixture.blankIndex).toBe(fixture.labels.length);
    expect(fixture.numClasses).toBe(fixture.labels.length);
    expect(frames).toBe(fixture.seqLen / 4); // 骨干池化两次
  });

  it.each(fixture.samples.map((s, i) => [i, s.refWords.join(" ")] as const))(
    "样本 %i（%s）解码与 Python 逐位一致",
    (i) => {
      const s = fixture.samples[i];
      const probs = new Float32Array(s.probs.flat());
      expect(probs.length).toBe(frames * numClasses);
      const got = greedyDecode(probs, frames, numClasses, fixture.blankIndex);
      expect(got).toEqual(s.decoded);
      expect(decodeToWords(got, fixture.labels)).toEqual(s.decodedWords);
    }
  );

  it("fixture 里既有解对的也有解错的样本", () => {
    // 全对的样本 argmax 遥遥领先，两边差一点也不改结果，测不出失配；
    // 错例才踩在 argmax 的边界上。所以 fixture 的抽样规则本身也要锁住
    const exact = fixture.samples.filter((s) => s.exact).length;
    expect(exact).toBeGreaterThan(0);
    expect(exact).toBeLessThan(fixture.samples.length);
  });

  it("确定性输入（ramp）的解码与 Python 一致", () => {
    // 这一条只验解码。同一份 ramp 概率还会被 sentenceModel.test 用来验
    // **前向计算**（tfjs 的层实现 + 填进去的权重）是否与 keras 一致
    const probs = new Float32Array(fixture.ramp.probs.flat());
    const got = greedyDecode(probs, frames, numClasses, fixture.blankIndex);
    expect(got).toEqual(fixture.ramp.decoded);
  });

  it("每一帧的概率和为 1（确认导出的是 softmax 概率而不是 logits）", () => {
    for (const row of fixture.ramp.probs) {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
    }
  });
});
