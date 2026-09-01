/*
 * compoundWords.test —— 复合词回收规则
 *
 * 这些规则会**吞掉**两个词换成一个，所以测试要锁住的不只是"能合"，还有
 * "不该合的时候不合"。三条最要紧的：
 *
 *  1. **规则里写的是原始 id（`you`），模型输出的是合并类（`merged_pron_sg`）。**
 *     匹配时两边都过 `mergeLabel`，所以这条链一旦断（比如有人把 canon 去掉），
 *     规则会**静默失效** —— 界面照样出「你」「谢谢」两个词，没有任何报错。
 *  2. **顺序不能对称。** 谢谢 → 你 不是你好。
 *  3. **规则产物必须是词表里真实存在的 id。** 打错一个字（`hello` 写成 `Hello`）
 *     的表现是历史里出现一个原始 id，而不是崩溃。
 */
import { describe, expect, it } from "vitest";
import {
  COMPOUND_MAX_GAP_MS,
  COMPOUND_RULES,
  isCompoundWord,
  matchCompound,
} from "./compoundWords";
import { mergeLabel } from "./labelMerge";
import { getWordById, getTranslationLabel } from "./signLanguageVocab";

const T0 = 1_000_000;

describe("matchCompound", () => {
  it("你 → 谢谢 合成你好（模型输出的是合并类 id）", () => {
    const prev = mergeLabel("you"); // = merged_pron_sg，模型实际输出的就是这个
    expect(prev).not.toBe("you"); // 前提：合并确实生效，否则这条测试测不到东西
    const r = matchCompound(prev, T0, "thank_you", T0 + 400);
    expect(r?.word).toBe("hello");
  });

  it("规则里写原始 id、传进来合并 id，两种写法都匹配", () => {
    // 关掉合并做对照实验时模型输出的是原始 `you`，同一条规则仍要成立
    expect(matchCompound("you", T0, "thank_you", T0 + 400)?.word).toBe("hello");
  });

  it("像 → 谢谢 合成好看", () => {
    expect(matchCompound("resemble", T0, "thank_you", T0 + 300)?.word).toBe(
      "beautiful"
    );
  });

  it("顺序反了不合 —— 谢谢 → 你 不是你好", () => {
    expect(matchCompound("thank_you", T0, "you", T0 + 300)).toBeNull();
  });

  it("间隔超过上限不合（那更像两个独立的词）", () => {
    expect(
      matchCompound("you", T0, "thank_you", T0 + COMPOUND_MAX_GAP_MS + 1)
    ).toBeNull();
    expect(
      matchCompound("you", T0, "thank_you", T0 + COMPOUND_MAX_GAP_MS)
    ).not.toBeNull();
  });

  it("时钟回跳（间隔为负）按不确定处理，不合", () => {
    expect(matchCompound("you", T0, "thank_you", T0 - 100)).toBeNull();
  });

  it("没有上一个词时不合（lastAddedWordRef 初值是空串）", () => {
    expect(matchCompound("", T0, "thank_you", T0 + 300)).toBeNull();
    expect(matchCompound(null, T0, "thank_you", T0 + 300)).toBeNull();
  });

  it("不在规则表里的组合不合", () => {
    expect(matchCompound("love", T0, "thank_you", T0 + 300)).toBeNull();
    expect(matchCompound("you", T0, "love", T0 + 300)).toBeNull();
  });
});

describe("规则表自身的约束", () => {
  it("每条规则的产物都是词表里真实存在的 id", () => {
    for (const r of COMPOUND_RULES) {
      expect(getWordById(r.word), `规则产物 ${r.word} 不在词表里`).toBeTruthy();
    }
  });

  it("每一段也都是词表里真实存在的 id", () => {
    for (const r of COMPOUND_RULES) {
      for (const p of r.parts) {
        expect(getWordById(p), `规则里的 ${p} 不在词表里`).toBeTruthy();
      }
    }
  });

  it("同一个前缀不能对应两条规则（否则先写的那条永远赢，后写的静默失效）", () => {
    const keys = COMPOUND_RULES.map(
      (r) => `${mergeLabel(r.parts[0])}→${mergeLabel(r.parts[1])}`
    );
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("每条规则都写了 reason —— 日后有人要删它之前得先读懂为什么加", () => {
    for (const r of COMPOUND_RULES) {
      expect(r.reason.length).toBeGreaterThan(10);
    }
  });

  it("isCompoundWord 认得产物、不认零件", () => {
    expect(isCompoundWord("hello")).toBe(true);
    expect(isCompoundWord("beautiful")).toBe(true);
    expect(isCompoundWord("thank_you")).toBe(false);
  });
});

describe("getTranslationLabel：合并类在翻译输出里显示成单个词", () => {
  it("merged_pron_sg 显示成「你」，不是「你/他」", () => {
    expect(getTranslationLabel(mergeLabel("you"))).toBe("你");
    // 打「他」出来也是「你」—— 使用者要求的口径，不是 bug
    expect(getTranslationLabel(mergeLabel("he"))).toBe("你");
  });

  it("复数组显示成组里的 defaultMember，而不是原始 id", () => {
    const s = getTranslationLabel(mergeLabel("we"));
    expect(s).not.toContain("merged_");
    expect(s).toBe(getWordById("they")?.label);
  });

  it("普通词原样查词表", () => {
    expect(getTranslationLabel("hello")).toBe("你好");
  });

  it("空闲伪类不露原始 id", () => {
    expect(getTranslationLabel("_idle")).not.toContain("_idle");
  });
});
