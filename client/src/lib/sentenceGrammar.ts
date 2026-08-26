/*
 * sentenceGrammar — 把 CTC 解出来的词序列顺成汉语。
 *
 * 手语语序不是汉语语序：疑问词放句尾（「你 名字 什么」）、没有「的」、
 * 「是」常常不打。所以模型解出来的词序列直接拼起来是"手语汉语"，
 * 读起来别扭但**信息是全的**。这里做的是加一层可读性，不是加一层判断。
 *
 * ===== 两条硬规则 =====
 *
 * 1. **不命中就原样拼接，绝不吞词。** 规则表宁可什么都不做。吞掉一个词的代价是
 *    信息丢失且不可见 —— 人看到一句通顺的话，不知道模型其实还解出了别的词。
 *    有测试锁着这一条。
 * 2. **原始词序永远可见。** 顺句结果是"加工品"，UI 上必须和原始词序并排显示。
 *    规则表是写死的猜测，不是模型的输出；两者混在一起会让人分不清
 *    "模型认错了"和"规则顺错了"，而这两件事的修法完全不同。
 *
 * ===== 合并类怎么定默认值 =====
 *
 * `merged_pron_sg`（我/你/他）是因为六轴 IMU 观测不到绝对 yaw 才合并的，
 * 模型**永远**分不出这三个（见 labelMerge.ts）。所以只能给个默认值。
 *
 * 默认值不是按"句法角色"定的 —— 试过，不成立。看 synth_sentences.py 的句型表：
 *   ["you", "name", "what"]   句首主语，但要取「你」（问句是问对面的）
 *   ["i",   "name", "is"]     同样是句首主语，要取「我」
 *   ["sorry", "i"]            寒暄词之后，但要取「我」（道歉是自己道）
 *   ["hello", "you"]          同样是寒暄词之后，要取「你」
 * 同一个位置能取不同的人，所以位置本身信息不够。真正决定的是**这一句在说谁**。
 *
 * 于是分两层：
 *   - 命中规则的句子：由规则自己带 `pron` 提示（句型表是已知的，直接写死最准）
 *   - 没命中的句子：退到一个粗启发式（动词后取「你」、问句主语取「你」、其余取「我」）
 *
 * 无论哪层，这都是**默认值不是判定**：UI 必须能一键换成另外两个，且原始词序里
 * 保持「我/你/他」这个未消解的形态。把默认值当结论显示的话，人会以为模型真分出来了。
 */
import { MERGE_GROUPS, getMergeGroup, isMergedLabel } from "./labelMerge";
import { getWordById } from "./signLanguageVocab";

/** 合并类不在词表里，所以不能直接用 getWordById */
export function displayWord(id: string): string {
  const g = getMergeGroup(id);
  if (g) return g.display;
  return getWordById(id)?.label ?? id;
}

const L = displayWord;

/**
 * 这个代词位默认指谁。`self` → 我/我们，`other` → 你/你们。
 *
 * 刻意不叫 subject/object：句法角色和该取谁**不是一回事**（见文件头的四个反例），
 * 用句法名字命名会诱导以后有人按主谓宾去"修正"它。
 */
export type PronounSlot = "self" | "other";

/** 单数代词三选，顺序即 UI 上的顺序（我 / 你 / 他） */
export const PRON_SG_OPTIONS = MERGE_GROUPS.find((g) => g.id === "merged_pron_sg")!.members;
/** 复数代词三选（我们 / 你们 / 他们） */
export const PRON_PL_OPTIONS = MERGE_GROUPS.find((g) => g.id === "merged_pron_pl")!.members;

/** 代词相关的全部 id：两个合并类 id + 六个原始成员。用于 `@pron` 匹配 */
const PRONOUN_IDS = new Set<string>([
  ...MERGE_GROUPS.map((g) => g.id),
  ...MERGE_GROUPS.flatMap((g) => g.members),
]);

export function isPronoun(id: string): boolean {
  return PRONOUN_IDS.has(id);
}

/** 合并类按 slot 取默认成员。返回**原始**词 id（如 `"i"`），不是合并类 id */
export function defaultPronoun(groupId: string, slot: PronounSlot): string {
  const opts = groupId === "merged_pron_pl" ? PRON_PL_OPTIONS : PRON_SG_OPTIONS;
  return slot === "self" ? opts[0] : opts[1];
}

/**
 * 哪些词后面跟的代词是"对方"。
 *
 * 只列词表里真有的（signLanguageVocab.ts）。列不全的后果是**默认值挑错**
 * （「我爱我」而不是「我爱你」），不会丢词 —— 可接受的失败模式，而且 UI 能一键改。
 */
const TAKES_OTHER = new Set([
  // 动词：宾语默认是对方
  "love", "help", "listen", "speak", "see", "know", "thank_you",
  "eat", "drink", "work", "study", "go", "come",
  // 寒暄：招呼的对象是对方
  "hello", "goodbye", "welcome", "please",
]);

/** 疑问词。问句的主语默认是对方（问的是对面那个人） */
const INTERROGATIVES = new Set(["what"]);

/**
 * 没命中规则时的粗启发式：每个位置默认指谁。
 *
 * 判据故意粗：出现过 TAKES_OTHER 里的词之后 → other；否则看整句有没有疑问词，
 * 有就 other（问句），没有就 self（陈述句默认说自己）。
 * 在 20~30 句的规模上做精细句法分析是过度工程，而且分析错了比没分析更难查。
 */
export function slotsOf(words: string[]): PronounSlot[] {
  const asking = words.some((w) => INTERROGATIVES.has(w));
  let seen = false;
  return words.map((w) => {
    const slot: PronounSlot = seen || asking ? "other" : "self";
    if (TAKES_OTHER.has(w)) seen = true;
    return slot;
  });
}

/** 解出来的一句话。`raw` 和 `resolved` 都留着 —— UI 要同时显示 */
export interface ResolvedSentence {
  /** 模型原样解出的类别 id 序列（含 `merged_pron_sg` 这种） */
  raw: string[];
  /** 消解后的词 id 序列（合并类被替换成具体成员），长度同 raw */
  resolved: string[];
  /** 每个位置最终用的 slot，长度同 raw。非代词位也有值，UI 不用特判 */
  slots: PronounSlot[];
  /** 顺句后的汉语 */
  text: string;
  /** 命中了哪条规则；null = 没命中，此时 text 是原样拼接 */
  rule: string | null;
}

/**
 * 一条顺句规则。
 *
 * `pattern` 的记号：
 *   - 具体词 id：字面匹配
 *   - `"*"`：匹配任意一个词
 *   - `"@pron"`：匹配任意代词（合并类或六个原始成员之一）
 *   - `"..."`：只能放末尾，匹配剩余全部（含 0 个）
 *
 * `pron` 是这条规则对代词位的默认值提示，键是**词序列下标**（等于 pattern 下标，
 * 因为 `"..."` 只在末尾，前面每个 token 恰好吃一个词）。
 *
 * `render` 拿到的是**消解后**的词 id 数组。
 */
interface GrammarRule {
  name: string;
  pattern: string[];
  pron?: Record<number, PronounSlot>;
  render: (ids: string[]) => string;
}

/**
 * 规则表，**从上往下第一条命中就用**，所以特例必须放在通例之前。
 *
 * 这张表是照 `python_train/synth_sentences.py` 的 SENTENCE_TEMPLATES 写的
 * （合并之后 i/you/he → merged_pron_sg，we/you_pl/they → merged_pron_pl）。
 *
 * ⚠ 这张表、synth_sentences.py 的句型表、以及阶段 4 采集页的句型表是**三份**。
 * 加句型时三处都要加，否则会出现"训了但顺不出汉语"。
 */
const RULES: GrammarRule[] = [
  // ——— 自我介绍 / 问名字。名字本身打不出来（词表里没有指语），所以只能留省略号 ———
  {
    name: "问候+问名字",
    pattern: ["hello", "@pron", "name", "what"],
    pron: { 1: "other" },
    render: ([, b]) => `你好，${L(b)}叫什么名字？`,
  },
  {
    name: "问候+自我介绍",
    // ["hello","i","name","is"]：这里的代词是**自己**（和 ["hello","you"] 相反），
    // 所以必须写死 self —— 靠"寒暄词之后取对方"的启发式会挑错
    pattern: ["hello", "@pron", "name", "is", "..."],
    pron: { 1: "self" },
    render: ([, b, , , ...rest]) =>
      `你好，${L(b)}的名字是${rest.length ? rest.map(L).join("") : "……"}`,
  },
  {
    name: "问名字",
    pattern: ["@pron", "name", "what"],
    pron: { 0: "other" },
    render: ([a]) => `${L(a)}叫什么名字？`,
  },
  {
    // 省略主语的「你好 名字 什么」。没有这条会落到「问做什么（非代词主语）」，
    // 把 hello 当成主语渲染成「你好名字什么？」
    name: "问候+问名字（省略主语）",
    pattern: ["hello", "name", "what"],
    render: () => "你好，叫什么名字？",
  },
  {
    name: "自我介绍",
    pattern: ["@pron", "name", "is", "..."],
    pron: { 0: "self" },
    render: ([a, , , ...rest]) =>
      `${L(a)}的名字是${rest.length ? rest.map(L).join("") + "。" : "……"}`,
  },
  {
    name: "名字",
    pattern: ["@pron", "name", "..."],
    pron: { 0: "self" },
    render: ([a, , ...rest]) => `${L(a)}的名字${rest.map(L).join("")}`,
  },
  {
    // 省略主语版。要排在通例「问做什么（省略主语）」之前，
    // 否则会渲染成「名字什么？」
    name: "问名字（省略主语）",
    pattern: ["name", "what"],
    render: () => "叫什么名字？",
  },
  {
    name: "名字是（省略主语）",
    pattern: ["name", "is", "..."],
    render: ([, , ...rest]) =>
      `名字是${rest.length ? rest.map(L).join("") + "。" : "……"}`,
  },
  // ——— 寒暄 ———
  {
    name: "问候",
    pattern: ["hello", "@pron"],
    pron: { 1: "other" },
    render: ([, b]) => `${L(b)}好！`,
  },
  {
    name: "道别",
    pattern: ["goodbye", "@pron"],
    pron: { 1: "other" },
    render: ([, b]) => `${L(b)}，再见！`,
  },
  {
    name: "致谢",
    pattern: ["thank_you", "@pron"],
    pron: { 1: "other" },
    render: ([, b]) => `谢谢${L(b)}！`,
  },
  {
    name: "致歉",
    // ["sorry","i"]：道歉的是自己。和上面三条同样是"寒暄词 + 代词"，取的人却相反
    pattern: ["sorry", "@pron"],
    pron: { 1: "self" },
    // 保留「对不起」原词而不是换成「很抱歉」：顺句只该调语序和补虚词，
    // 换同义词会让"模型认出了哪个词"变得看不出来
    render: ([, b]) => `对不起，${L(b)}的错。`,
  },
  {
    name: "欢迎（双方）",
    pattern: ["welcome", "@pron", "@pron"],
    pron: { 1: "other", 2: "self" },
    render: ([, b, c]) => `${L(c)}欢迎${L(b)}！`,
  },
  {
    name: "欢迎",
    pattern: ["welcome", "@pron"],
    pron: { 1: "other" },
    render: ([, b]) => `欢迎${L(b)}！`,
  },
  // ——— 否定：手语用专门的「不是」手势，位置在谓语前 ———
  {
    name: "否定",
    pattern: ["@pron", "is_not", "..."],
    pron: { 0: "self" },
    render: ([a, , ...rest]) => `${L(a)}不${rest.map(L).join("")}。`,
  },
  {
    name: "否定（非代词主语）",
    pattern: ["*", "is_not", "..."],
    render: ([a, , ...rest]) => `${L(a)}不${rest.map(L).join("")}。`,
  },
  {
    // 手语里主语常省略，句子从「不是」起手。没有这条会掉到通例「主谓」，
    // 渲染成「不是高兴。」—— 词没丢，但话不对
    name: "否定（省略主语）",
    pattern: ["is_not", "..."],
    render: ([, ...rest]) => `不${rest.map(L).join("")}。`,
  },
  // ——— 疑问词在句尾，汉语里要挪到宾语位 ———
  {
    name: "问做什么",
    pattern: ["@pron", "*", "what"],
    pron: { 0: "other" },
    render: ([a, v]) => `${L(a)}${L(v)}什么？`,
  },
  {
    name: "问做什么（非代词主语）",
    pattern: ["*", "*", "what"],
    render: ([a, v]) => `${L(a)}${L(v)}什么？`,
  },
  {
    // 同上，省略主语的疑问句：「吃 什么」。必须排在通例「主谓」之前，
    // 否则会被渲染成「吃什么。」（陈述句句号），疑问语气丢了
    name: "问做什么（省略主语）",
    pattern: ["*", "what"],
    render: ([v]) => `${L(v)}什么？`,
  },
  // ——— 主谓宾 ———
  {
    name: "帮某人做某事",
    pattern: ["@pron", "help", "@pron", "..."],
    pron: { 0: "self", 2: "other" },
    // 「帮助」写全，不缩成「帮」—— 同上，原词要在结果里认得出来
    render: ([a, , c, ...rest]) => `${L(a)}帮助${L(c)}${rest.map(L).join("")}。`,
  },
  {
    name: "主谓宾（双代词）",
    pattern: ["@pron", "*", "@pron"],
    pron: { 0: "self", 2: "other" },
    render: ([a, v, o]) => `${L(a)}${L(v)}${L(o)}。`,
  },
  {
    name: "主谓宾",
    pattern: ["*", "*", "*"],
    render: ([a, v, o]) => `${L(a)}${L(v)}${L(o)}。`,
  },
  // ——— 主谓 ———
  {
    name: "主谓（代词主语）",
    pattern: ["@pron", "*"],
    pron: { 0: "self" },
    render: ([a, b]) => `${L(a)}${L(b)}。`,
  },
  {
    name: "主谓",
    pattern: ["*", "*"],
    render: ([a, b]) => `${L(a)}${L(b)}。`,
  },
  {
    name: "单词",
    pattern: ["*"],
    render: ([a]) => `${L(a)}。`,
  },
];

function tokenMatches(token: string, id: string): boolean {
  if (token === "*") return true;
  if (token === "@pron") return isPronoun(id);
  return token === id;
}

function matches(pattern: string[], ids: string[]): boolean {
  const rest = pattern.indexOf("...");
  if (rest >= 0) {
    // "..." 之前的部分逐位对上就行，之后不管（含 0 个）
    if (ids.length < rest) return false;
    return pattern.slice(0, rest).every((p, i) => tokenMatches(p, ids[i]));
  }
  if (pattern.length !== ids.length) return false;
  return pattern.every((p, i) => tokenMatches(p, ids[i]));
}

/** 找第一条命中的规则；没有就 null */
function findRule(words: string[]): GrammarRule | null {
  for (const r of RULES) if (matches(r.pattern, words)) return r;
  return null;
}

/**
 * 词序列 → 顺句结果。
 *
 * @param words 模型解出的类别 id 序列（可含合并类）
 * @param overrides 词序列下标 → 用户手动指定的成员 id。UI 上点词条改代词时传进来。
 *                  优先级最高：压过规则提示，也压过启发式
 * @param enabled false = 只消解合并类、不套规则表（原样拼接）。
 *                规则表是猜测，要能一键关掉去看模型的原始输出
 */
export function resolveSentence(
  words: string[],
  overrides: Record<number, string> = {},
  enabled = true
): ResolvedSentence {
  // slot 要先定下来才能消解代词，而规则提示比启发式准，所以先匹配规则
  const rule = enabled ? findRule(words) : null;
  const heur = slotsOf(words);
  const slots = heur.map((s, i) => rule?.pron?.[i] ?? s);

  const resolved = words.map((w, i) => {
    const o = overrides[i];
    if (o) return o;
    if (isMergedLabel(w)) return defaultPronoun(w, slots[i]);
    return w;
  });

  if (!words.length) return { raw: words, resolved, slots, text: "", rule: null };
  if (rule) {
    return { raw: words, resolved, slots, text: rule.render(resolved), rule: rule.name };
  }
  // **没命中就原样拼接，不吞词。** 模型可以解出任何序列，规则表不可能覆盖全，
  // 那时候把词都显示出来比显示一句通顺但缺词的话有用得多
  return {
    raw: words,
    resolved,
    slots,
    text: resolved.map(L).join(" "),
    rule: null,
  };
}

/** 这个位置能不能换代词，能换成哪些。UI 用它决定词条是否可点 */
export function pronounChoices(word: string): string[] | null {
  const g = getMergeGroup(word);
  return g ? g.members : null;
}

/** 规则名清单，给设置页/调试用（顺序即匹配优先级） */
export function grammarRuleNames(): string[] {
  return RULES.map((r) => r.name);
}
