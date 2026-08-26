/*
 * sentenceGrammar.test —— 顺句规则表
 *
 * 两类断言，作用完全不同：
 *
 * 1. **不丢词**（最要紧）。规则表没命中时必须原样拼接。吞词的失败模式是不可见的：
 *    人看到一句通顺的话，不知道模型其实还解出了别的词，会以为模型没认出来。
 * 2. **和句型表对齐**。规则表照 python_train/synth_sentences.py 的 SENTENCE_TEMPLATES
 *    写的，两份分别在两种语言里。下面把那张表原样抄过来逐条跑 —— 抄一份是故意的，
 *    Python 那边加了句型而这边忘了加时，这个测试会红。
 */
import { describe, expect, it } from "vitest";
import { mergeLabel } from "./labelMerge";
import {
  defaultPronoun,
  displayWord,
  grammarRuleNames,
  isPronoun,
  pronounChoices,
  resolveSentence,
  slotsOf,
} from "./sentenceGrammar";

/**
 * synth_sentences.py 的 SENTENCE_TEMPLATES（原始标签，未合并）。
 * 改这里之前先改那边 —— 反过来会训出顺不出汉语的句型。
 */
const TEMPLATES: string[][] = [
  ["you", "name", "what"],
  ["i", "name", "is"],
  ["hello", "you", "name", "what"],
  ["you", "eat", "what"],
  ["you", "speak", "what"],
  ["you", "work", "what"],
  ["you", "study", "what"],
  ["hello", "you"],
  ["hello", "i", "name", "is"],
  ["goodbye", "you"],
  ["thank_you", "you"],
  ["sorry", "i"],
  ["welcome", "you"],
  ["welcome", "you", "we"],
  ["i", "love", "you"],
  ["you", "love", "i"],
  ["i", "listen", "you"],
  ["i", "help", "you"],
  ["you", "help", "i"],
  ["i", "speak", "you"],
  ["i", "is_not", "happy"],
  ["i", "is_not", "sad"],
  ["i", "is_not", "angry"],
  ["i", "is_not", "eat"],
  ["you", "is_not", "listen"],
  ["i", "happy"],
  ["i", "sad"],
  ["i", "eat"],
  ["i", "drink"],
  ["i", "work"],
  ["i", "study"],
  ["we", "study"],
  ["we", "work"],
  ["they", "study"],
  ["you_pl", "listen"],
  ["he", "is_not", "work"],
  ["i", "thank_you", "you"],
  ["i", "help", "you", "study"],
  ["you", "eat", "drink"],
  // 无主语句。这一组在 synth_sentences.py 里是为了压类先验加的
  // （加之前 merged_pron_sg 独占 41% 的词位，模型一没把握就只输出代词）
  ["name", "what"],
  ["eat", "what"],
  ["drink", "what"],
  ["work", "what"],
  ["study", "what"],
  ["speak", "what"],
  ["listen", "what"],
  ["hello", "welcome"],
  ["hello", "thank_you"],
  ["goodbye", "thank_you"],
  ["help", "thank_you"],
  ["sorry", "goodbye"],
  ["is_not", "happy"],
  ["is_not", "angry"],
  ["is_not", "sad"],
  ["is_not", "eat"],
  ["is_not", "drink"],
  ["is_not", "work"],
  ["eat", "drink"],
  ["study", "work"],
  ["listen", "speak"],
  ["love", "eat"],
  ["love", "study"],
  ["name", "is"],
  ["hello", "name", "what"],
  ["welcome", "study", "work"],
  ["help", "study", "thank_you"],
];

/** 模型看到的是合并后的类别 —— 六个代词只剩两个类 */
const merged = (t: string[]) => t.map((w) => mergeLabel(w));

/**
 * 顺句允许的少数改写。**这张表必须一直很短**：它一长就说明规则表在换词而不是调语序，
 * 那时候"模型认出了哪个词"就从结果里看不出来了。
 *
 * `is_not`：手语打「不是 高兴」，汉语只能说「不高兴」，「不是高兴」不成话。
 */
const ALIASES: Record<string, string[]> = {
  is_not: ["不"],
};

/** 这个词在顺句结果里留下痕迹了吗 */
function leavesTrace(id: string, text: string): boolean {
  const cands = [displayWord(id), ...(ALIASES[id] ?? [])];
  return cands.some((c) => text.includes(c));
}

describe("sentenceGrammar", () => {
  describe("不丢词", () => {
    it("规则表没命中时原样拼接，一个词都不少", () => {
      // 模型可以解出任意序列（CTC 的输出不受句型表约束），规则表不可能覆盖全。
      // 没命中时显示全部词 >> 显示一句通顺但缺词的话
      const words = ["drink", "angry", "goodbye", "study", "what", "sorry"];
      const r = resolveSentence(words);
      expect(r.rule).toBeNull();
      for (const w of words) expect(r.text).toContain(displayWord(w));
    });

    it("生词 id（不在词表里）也不吞，原样显示", () => {
      const r = resolveSentence(["i", "zzz_unknown"]);
      expect(r.text).toContain("zzz_unknown");
    });

    it("句型表里每条句子的每个词都在结果里留痕", () => {
      for (const t of TEMPLATES) {
        const ids = merged(t);
        const r = resolveSentence(ids);
        for (let i = 0; i < ids.length; i++) {
          // 代词位看消解后的词（合并类显示成「我/你/他」，结果里是具体那个）
          const check = r.resolved[i];
          expect(
            leavesTrace(check, r.text),
            `「${t.join(" ")}」→「${r.text}」丢了 ${check}`
          ).toBe(true);
        }
      }
    });
  });

  describe("与 synth_sentences.py 的句型表对齐", () => {
    it("每条句型都命中某条规则", () => {
      const missed = TEMPLATES.filter((t) => resolveSentence(merged(t)).rule === null);
      expect(
        missed.map((t) => t.join(" ")),
        "这些句型训了但顺不出汉语 —— 规则表要补"
      ).toEqual([]);
    });

    it("每条句型顺出来的都不是空串", () => {
      for (const t of TEMPLATES) {
        expect(resolveSentence(merged(t)).text.length).toBeGreaterThan(0);
      }
    });
  });

  describe("代词默认值：位置本身信息不够", () => {
    // 这四条是"按句法角色定默认值"不成立的证据（见模块头注释）。
    // 前两条同为句首主语取的人相反，后两条同为寒暄词之后取的人也相反
    it("问名字：句首主语取「你」（问的是对面）", () => {
      const r = resolveSentence(merged(["you", "name", "what"]));
      expect(r.resolved[0]).toBe("you");
      expect(r.text).toBe("你叫什么名字？");
    });

    it("自我介绍：同样是句首主语，取「我」", () => {
      const r = resolveSentence(merged(["i", "name", "is"]));
      expect(r.resolved[0]).toBe("i");
      expect(r.text).toBe("我的名字是……");
    });

    it("问候：寒暄词之后取「你」", () => {
      expect(resolveSentence(merged(["hello", "you"])).text).toBe("你好！");
    });

    it("致歉：同样是寒暄词之后，取「我」", () => {
      const r = resolveSentence(merged(["sorry", "i"]));
      expect(r.resolved[1]).toBe("i");
      expect(r.text).toBe("对不起，我的错。");
    });

    it("主谓宾：主语「我」宾语「你」", () => {
      expect(resolveSentence(merged(["i", "love", "you"])).text).toBe("我爱你。");
    });

    it("同一句里两个代词位可以取不同的人", () => {
      // [sg, love, sg] 两个位置是同一个类别 id，取的人却不同 ——
      // 消解必须逐位置做，不能按类别做
      const ids = merged(["i", "love", "you"]);
      expect(ids[0]).toBe(ids[2]);
      const r = resolveSentence(ids);
      expect(r.resolved[0]).not.toBe(r.resolved[2]);
    });

    it("复数类取「我们/你们」而不是单数", () => {
      expect(resolveSentence(merged(["we", "study"])).text).toBe("我们学习。");
      expect(resolveSentence(merged(["welcome", "you", "we"])).text).toBe("我们欢迎你！");
    });

    it("关掉合并时原始代词不被默认值改写", () => {
      // mergeLabel(x, false) 原样返回。此时模型直接输出 i/you/he，
      // 已经是确定的人了，再套默认值等于把模型的输出改掉
      const r = resolveSentence(["he", "love", "he"]);
      expect(r.resolved).toEqual(["he", "love", "he"]);
      expect(r.text).toBe("他爱他。");
    });

    it("取不到的那个人（他/你们）只能靠手动改 —— 默认值给不出来", () => {
      const ids = merged(["he", "is_not", "work"]);
      expect(resolveSentence(ids).resolved[0]).toBe("i"); // 默认挑错了
      expect(resolveSentence(ids, { 0: "he" }).text).toBe("他不工作。");
    });
  });

  describe("手动覆盖", () => {
    it("overrides 压过规则提示", () => {
      const ids = merged(["i", "love", "you"]);
      const r = resolveSentence(ids, { 0: "he", 2: "i" });
      expect(r.text).toBe("他爱我。");
    });

    it("覆盖一个位置不影响另一个", () => {
      const r = resolveSentence(merged(["i", "love", "you"]), { 2: "he" });
      expect(r.resolved).toEqual(["i", "love", "he"]);
    });

    it("raw 永远保持未消解的形态（UI 要一直显示原始词序）", () => {
      const ids = merged(["i", "love", "you"]);
      const r = resolveSentence(ids, { 0: "he" });
      expect(r.raw).toEqual(ids);
      expect(r.raw[0]).toBe("merged_pron_sg");
      expect(displayWord(r.raw[0])).toBe("我/你/他");
    });
  });

  describe("开关", () => {
    it("关掉规则表只做代词消解，原样拼接", () => {
      const r = resolveSentence(merged(["i", "love", "you"]), {}, false);
      expect(r.rule).toBeNull();
      expect(r.text).toBe("我 爱 你");
    });

    it("关掉时代词仍然消解（否则显示成「我/你/他 爱 我/你/他」没法读）", () => {
      const r = resolveSentence(merged(["i", "love", "you"]), {}, false);
      expect(r.resolved).toEqual(["i", "love", "you"]);
    });
  });

  describe("规则匹配", () => {
    it("特例排在通例之前：问名字不落到「主谓宾」", () => {
      expect(resolveSentence(merged(["you", "name", "what"])).rule).toBe("问名字");
      expect(resolveSentence(merged(["you", "eat", "what"])).rule).toBe("问做什么");
      expect(resolveSentence(merged(["you", "eat", "drink"])).rule).toBe("主谓宾");
    });

    it("「...」能匹配 0 个尾巴", () => {
      // ["@pron","is_not","..."]：否定后面可以什么都没有
      const r = resolveSentence(merged(["i", "is_not"]));
      expect(r.text).toBe("我不。");
    });

    it("「...」能匹配多个尾巴", () => {
      expect(resolveSentence(merged(["i", "help", "you", "study"])).text).toBe(
        "我帮助你学习。"
      );
    });

    it("非代词主语也有规则兜（模型可能解出任何序列）", () => {
      expect(resolveSentence(["hello", "is_not", "work"]).rule).toBe("否定（非代词主语）");
    });

    it("规则名不重复（重名会让调试时分不清命中的是哪条）", () => {
      const names = grammarRuleNames();
      expect(new Set(names).size).toBe(names.length);
    });
  });

  describe("边界", () => {
    it("空序列返回空串，不抛", () => {
      const r = resolveSentence([]);
      expect(r).toEqual({ raw: [], resolved: [], slots: [], text: "", rule: null });
    });

    it("单个词", () => {
      expect(resolveSentence(["happy"]).text).toBe("高兴。");
      expect(resolveSentence(merged(["i"])).text).toBe("我。");
    });

    it("resolved / slots 长度与 raw 一致（UI 按下标取，长度不齐会错位）", () => {
      for (const t of TEMPLATES) {
        const ids = merged(t);
        const r = resolveSentence(ids);
        expect(r.resolved).toHaveLength(ids.length);
        expect(r.slots).toHaveLength(ids.length);
      }
    });
  });

  describe("辅助函数", () => {
    it("isPronoun 认合并类也认原始成员", () => {
      expect(isPronoun("merged_pron_sg")).toBe(true);
      expect(isPronoun("merged_pron_pl")).toBe(true);
      expect(isPronoun("i")).toBe(true);
      expect(isPronoun("they")).toBe(true);
      expect(isPronoun("eat")).toBe(false);
    });

    it("pronounChoices 只对合并类给三选", () => {
      expect(pronounChoices("merged_pron_sg")).toEqual(["i", "you", "he"]);
      expect(pronounChoices("merged_pron_pl")).toEqual(["we", "you_pl", "they"]);
      // 已经确定的人不该给三选 —— 给了等于允许把模型的确定输出改掉
      expect(pronounChoices("i")).toBeNull();
      expect(pronounChoices("eat")).toBeNull();
    });

    it("defaultPronoun 单复数各取本组的成员", () => {
      expect(defaultPronoun("merged_pron_sg", "self")).toBe("i");
      expect(defaultPronoun("merged_pron_sg", "other")).toBe("you");
      expect(defaultPronoun("merged_pron_pl", "self")).toBe("we");
      expect(defaultPronoun("merged_pron_pl", "other")).toBe("you_pl");
    });

    it("displayWord 合并类给三选文字，普通词给中文", () => {
      expect(displayWord("merged_pron_sg")).toBe("我/你/他");
      expect(displayWord("eat")).toBe("吃");
      expect(displayWord("nope")).toBe("nope");
    });

    it("slotsOf：动词之后取「你」，之前取「我」", () => {
      expect(slotsOf(["merged_pron_sg", "love", "merged_pron_sg"])).toEqual([
        "self",
        "self",
        "other",
      ]);
    });

    it("slotsOf：整句有疑问词时主语位也取「你」", () => {
      // 问句是问对面的。这条只在规则表没命中时起作用（命中时用规则的提示）
      expect(slotsOf(["merged_pron_sg", "eat", "what"])[0]).toBe("other");
      expect(slotsOf(["merged_pron_sg", "eat"])[0]).toBe("self");
    });
  });
});
