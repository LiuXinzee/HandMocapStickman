/*
 * sentenceGrammar.test —— 顺句规则表
 *
 * 两类断言，作用完全不同：
 *
 * 1. **不丢词**（最要紧）。规则表没命中时必须原样拼接。吞词的失败模式是不可见的：
 *    人看到一句通顺的话，不知道模型其实还解出了别的词，会以为模型没认出来。
 * 2. **每条句型都顺得出汉语**。句型表从 `sentenceTemplates.ts` import ——
 *    以前这里手抄了一份 66 句的常量，注释还写着"Python 那边改了这个测试会红"，
 *    **那是假的**：它比的是自己那份字面量，Python 改了它什么都不知道。
 *    真正的跨语言锁在 `sentenceTemplates.test.ts`（直接读 synth_sentences.py 解析比对）。
 *    这个文件只管"TS 这份表里的每条句型，规则表能不能顺出汉语"。
 */
import { describe, expect, it } from "vitest";
import { mergeLabel } from "./labelMerge";
import { SENTENCE_TEMPLATES } from "./sentenceTemplates";
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
 * 句型表的别名。原本这里是手抄的 66 句常量，现在指向唯一来源
 * （`sentenceTemplates.ts`，由 sentenceTemplates.test.ts 锁着与 Python 一致）。
 */
const TEMPLATES: readonly (readonly string[])[] = SENTENCE_TEMPLATES;

/**
 * 模型看到的是合并后的类别。
 *
 * ⚠ **「我」原样穿过。** 它已经从合并组里拆出去（指自己胸口、有接触，分得开 ——
 * 见 labelMerge.ts），所以 `merged(["i","love","you"])` 是
 * `["i", "love", "merged_pron_sg"]`，两个代词位**不再是同一个 id**。
 * 下面凡是要"同一个合并 id 出现两次"的测试，只能拿复数组构造。
 */
const merged = (t: readonly string[]) => t.map((w) => mergeLabel(w));

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

  // 表本身与 Python 是否一致由 sentenceTemplates.test.ts 管；这里只管"顺不顺得出来"
  describe("句型表里每条都顺得出汉语", () => {
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
    // 前两条同为句首主语取的人相反，后两条同为寒暄词之后取的人也相反。
    //
    // ⚠ 现在前两条里的「我」已经**不是猜的了** —— 它是独立类，模型直接认出来。
    // 保留这几条是因为它们仍然是复数组的判据，而且记录了一件试过并失败的事。
    it("问名字：句首主语取「你」（问的是对面）", () => {
      const r = resolveSentence(merged(["you", "name", "what"]));
      expect(r.resolved[0]).toBe("you");
      expect(r.text).toBe("你叫什么名字？");
    });

    it("自我介绍：同样是句首主语，取「我」", () => {
      // 这条现在同时锁着 `isPronoun("i")`：「我」拆出合并组之后，如果 PRONOUN_IDS
      // 还是从 MERGE_GROUPS 推的，`@pron` 就匹配不上它，这句会掉到「主谓宾」
      // 渲染成「我名字是。」—— 词没丢，所以只会被当成"顺句变差了"，极难定位
      const r = resolveSentence(merged(["i", "name", "is"]));
      expect(r.raw[0]).toBe("i"); // 「我」不是合并类，原样穿过
      expect(r.resolved[0]).toBe("i");
      expect(r.rule).toBe("自我介绍");
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

    it("「我爱你」和「你爱我」是两个不同的序列 —— 方向是量到的，不是猜的", () => {
      // 拆开「我」换来的就是这个。以前两句都解成 [sg, love, sg]，字面完全相同，
      // 方向只能靠规则表猜；猜错的症状是"方向偶尔反"，而序列是对的，没法自查。
      // 现在「我」在哪一位是模型直接输出的。
      const a = merged(["i", "love", "you"]);
      const b = merged(["you", "love", "i"]);
      expect(a).not.toEqual(b);
      expect(a).toEqual(["i", "love", "merged_pron_sg"]);
      expect(b).toEqual(["merged_pron_sg", "love", "i"]);
      expect(resolveSentence(a).text).toBe("我爱你。");
      expect(resolveSentence(b).text).toBe("你爱我。");
    });

    it("同一句里两个代词位可以取不同的人（复数组）", () => {
      // 同一个类别 id 出现两次、取的人却不同 —— 消解必须逐位置做，不能按类别做。
      //
      // 只能拿复数组构造：单数组现在是 你/他，没有 selfMember，两个 slot 都落到
      // 「你」，本来就不可能取到不同的人（模型说朝外指，那就不是「我」）。
      const ids = merged(["we", "help", "you_pl"]);
      expect(ids[0]).toBe(ids[2]);
      const r = resolveSentence(ids);
      expect(r.resolved[0]).not.toBe(r.resolved[2]);
      // 「他们」而不是「你们」：复数组的 defaultMember 已从 you_pl 改成 they
      // （you_pl 在 UNTRAINED_WORDS 里，见 sentenceTemplates.ts）
      expect(r.text).toBe("我们帮助他们。");
    });

    it("单数合并类无论哪个 slot 都取「你」，绝不取「我」", () => {
      // 「我」是独立类了。合并类还翻译成「我」的话，等于把一个**量到不是我**的
      // 输出改写成「我」—— 比挑错人更糟，它伪造了一个模型没给的结论
      const subj = resolveSentence(merged(["you", "study"])); // slot=self
      expect(subj.resolved[0]).toBe("you");
      expect(subj.text).toBe("你学习。");
      const obj = resolveSentence(merged(["i", "love", "you"])); // slot=other
      expect(obj.resolved[2]).toBe("you");
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
      // 默认值是**默认值不是判定**：组里另一个成员永远只能靠 UI 一键换。
      // 单数组现在只有两个成员，猜错的代价从 1/3 降到 1/2
      const ids = merged(["he", "is_not", "work"]);
      expect(resolveSentence(ids).resolved[0]).toBe("you"); // 默认挑错了（该是「他」）
      expect(resolveSentence(ids, { 0: "he" }).text).toBe("他不工作。");
    });
  });

  /*
   * 2026-08-29 追加的三句。锁的是**顺出来的整句汉语**，不只是"命中了规则" ——
   * 上面「每条句型都命中某条规则」那条只保证 rule ≠ null，
   * 一条 render 写错字的规则照样能让它全绿。
   */
  describe("你叫什么名字 / 你真好看 / 你笑起来像太阳", () => {
    it("你叫什么名字（name 训回来之后的第一句）", () => {
      const r = resolveSentence(merged(["you", "name", "what"]));
      expect(r.rule).toBe("问名字");
      expect(r.text).toBe("你叫什么名字？");
    });

    it("你真好看 —— 有专门规则，不落到通例「主谓」", () => {
      const r = resolveSentence(merged(["you", "beautiful"]));
      // 落到通例的话是「你好看。」：词不丢、人也看得懂，所以只断言 text 不够，
      // 必须连规则名一起锁 —— 否则规则被别人挪到 ["@pron","*"] 后面就静默失效
      expect(r.rule).toBe("赞美");
      expect(r.text).toBe("你真好看！");
    });

    it("你笑起来像太阳 —— 4 词句，缺了规则会退化成原样拼词", () => {
      const r = resolveSentence(merged(["you", "smile", "resemble", "sun"]));
      expect(r.rule).toBe("笑起来像");
      expect(r.text).toBe("你笑起来像太阳。");
    });

    it("三句的句首都消解成「你」（不是「我」）", () => {
      // `slotsOf` 的启发式在后两句上会给 self：句子里没有疑问词，
      // 而 smile/beautiful 都不在 TAKES_OTHER 里。规则表写死 other 才对。
      // 单数组没有 selfMember，所以 self 现在恰好也落到「你」—— 但那是巧合，
      // 这条测试锁的是"规则里那个 other 别被删掉"
      for (const t of [
        ["you", "name", "what"],
        ["you", "beautiful"],
        ["you", "smile", "resemble", "sun"],
      ]) {
        const r = resolveSentence(merged(t));
        expect(r.slots[0], `「${t.join(" ")}」句首 slot`).toBe("other");
        expect(r.resolved[0]).toBe("you");
      }
    });

    it("换成「他」也顺得出来（默认值是默认值不是判定）", () => {
      expect(
        resolveSentence(merged(["you", "beautiful"]), { 0: "he" }).text
      ).toBe("他真好看！");
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
      const ids = merged(["you", "love", "i"]);
      const r = resolveSentence(ids, { 0: "he" });
      expect(r.raw).toEqual(ids);
      // 第 0 位是合并类，raw 里必须还是那个未消解的 id；第 2 位的「我」是确定的词
      expect(r.raw[0]).toBe("merged_pron_sg");
      expect(displayWord(r.raw[0])).toBe("你/他");
      expect(r.raw[2]).toBe("i");
      expect(r.text).toBe("他爱我。");
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
    it("isPronoun 认合并类也认原始成员，「我」也必须认", () => {
      expect(isPronoun("merged_pron_sg")).toBe(true);
      expect(isPronoun("merged_pron_pl")).toBe(true);
      // ⚠ 「我」已经不在任何合并组里，但语法上当然还是代词。这条断言防的是
      // 把 PRONOUN_IDS 改回"从 MERGE_GROUPS 推"—— 那样每一句以「我」开头的话
      // 都会掉到通例规则，顺出来的汉语全部退化而词一个不少
      expect(isPronoun("i")).toBe(true);
      expect(isPronoun("they")).toBe(true);
      expect(isPronoun("eat")).toBe(false);
    });

    it("pronounChoices 只对合并类给候选", () => {
      // 单数只剩两选：「我」拆出去之后不该再出现在这个列表里，
      // 出现了就等于允许用户把一个量到"朝外指"的输出改写成「我」
      expect(pronounChoices("merged_pron_sg")).toEqual(["you", "he"]);
      expect(pronounChoices("merged_pron_pl")).toEqual(["we", "you_pl", "they"]);
      // 已经确定的人不该给候选 —— 给了等于允许把模型的确定输出改掉
      expect(pronounChoices("i")).toBeNull();
      expect(pronounChoices("eat")).toBeNull();
    });

    it("defaultPronoun 按组自己声明的成员取，不按下标", () => {
      // 单数组没有 selfMember，所以两个 slot 都落到 defaultMember「你」。
      // 以前这里是 `opts[0]` / `opts[1]`，把「我」从 members 里删掉那一刻
      // 默认值会静默挪一位（类型过、大半测试也过，只有翻译悄悄变了）
      expect(defaultPronoun("merged_pron_sg", "self")).toBe("you");
      expect(defaultPronoun("merged_pron_sg", "other")).toBe("you");
      expect(defaultPronoun("merged_pron_pl", "self")).toBe("we");
      // 复数组的 defaultMember：you_pl → they。you_pl 不再训练，默认值指向它
      // 等于整组翻译成一个模型从来没见过的词（见 sentenceTemplates.ts 的
      // UNTRAINED_WORDS；那边有测试锁着"defaultMember 不指向不训练的词"）。
      // 没取 `we`：它是 selfMember，再拿它当默认值 selfMember 就永远不起作用了
      expect(defaultPronoun("merged_pron_pl", "other")).toBe("they");
    });

    it("defaultPronoun 遇到非合并 id 原样返回，绝不吞词", () => {
      expect(defaultPronoun("i", "self")).toBe("i");
      expect(defaultPronoun("eat", "other")).toBe("eat");
    });

    it("displayWord 合并类给候选文字，普通词给中文", () => {
      expect(displayWord("merged_pron_sg")).toBe("你/他");
      expect(displayWord("i")).toBe("我"); // 独立类，走词表
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
