/*
 * sentenceTemplates.test —— 句型表的**跨语言锁**。
 *
 * 这是整个仓库里唯一能抓住"Python 侧改了句型表、TS 侧忘了跟"的测试。
 * 做法是直接把 `python_train/synth_sentences.py` 读进来解析 —— vitest 跑在 Node 里，
 * `fs` 可用。听起来脏，但替代方案都不成立：
 *   - 手抄一份常量：抄的时候是对的，之后就是两份互不相干的字面量
 *     （`sentenceGrammar.test.ts` 从前就是这样，它的注释声称能抓漂移，其实抓不到）
 *   - 生成代码：多一个构建步骤，而且忘了跑生成器和忘了改表是同一个失败模式
 *   - 运行 Python 导出 JSON：CI/本机不一定有那个 venv，测试会因环境而红
 *
 * 路径用 `import.meta.url` 相对定位，不用 `process.cwd()`：
 * 从仓库根跑 `vitest run` 和从 client/ 里跑，cwd 是不一样的。
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { MERGE_GROUPS } from "./labelMerge";
import {
  BATCH1_TEMPLATES,
  RECOMMENDED_PER_TEMPLATE,
  SENTENCE_TEMPLATES,
  UNTRAINED_WORDS,
  templateKey,
} from "./sentenceTemplates";

function pythonSource(): string {
  return readFileSync(
    new URL("../../../python_train/synth_sentences.py", import.meta.url),
    "utf8"
  );
}

/** 从 synth_sentences.py 里抠出 UNTRAINED_WORDS 的内容 */
function pythonUntrained(): string[] {
  const src = pythonSource();
  // 声明是 `UNTRAINED_WORDS: tuple[str, ...] = ("happy", ...)`。从赋值号之后取，
  // 不能从名字之后直接取 —— 类型标注里也有括号
  const decl = src.match(/^UNTRAINED_WORDS[^=\n]*=\s*\(([^)]*)\)/m);
  expect(decl, "synth_sentences.py 里找不到 UNTRAINED_WORDS 的赋值").not.toBeNull();
  return Array.from(decl![1].matchAll(/"([^"]+)"/g)).map((m) => m[1]);
}

/** 从 synth_sentences.py 里抠出 SENTENCE_TEMPLATES 的内容 */
function pythonTemplates(): string[][] {
  const src = pythonSource();
  const start = src.indexOf("SENTENCE_TEMPLATES");
  expect(start, "synth_sentences.py 里找不到 SENTENCE_TEMPLATES").toBeGreaterThan(-1);
  // 从 `=` 之后找 `[`，不能从名字之后直接找：声明写的是
  // `SENTENCE_TEMPLATES: list[list[str]] = [`，第一个 `[` 在类型标注里
  const eq = src.indexOf("=", start);
  const open = src.indexOf("[", eq);
  expect(open, "SENTENCE_TEMPLATES 的赋值号后面没有 [").toBeGreaterThan(-1);
  // 从 `[` 起数括号配平，找到这个 list 的结尾。表里只有字符串字面量，
  // 不会出现方括号出现在字符串里的情况，所以数括号是安全的
  let depth = 0;
  let end = -1;
  for (let i = open; i < src.length; i++) {
    if (src[i] === "[") depth++;
    else if (src[i] === "]") {
      depth--;
      if (depth === 0) {
        end = i + 1;
        break;
      }
    }
  }
  expect(end, "SENTENCE_TEMPLATES 的方括号没配平").toBeGreaterThan(-1);
  const body = src.slice(open + 1, end - 1);

  // 每个内层 [...] 就是一条句型。注释里不会出现 ["..."] 形式，
  // 但为稳妥起见先把 # 到行尾去掉
  const clean = body.replace(/#[^\n]*/g, "");
  const rows = clean.match(/\[[^\]]*\]/g) ?? [];
  return rows.map((row) =>
    Array.from(row.matchAll(/"([^"]+)"/g)).map((m) => m[1])
  );
}

describe("sentenceTemplates", () => {
  describe("与 python_train/synth_sentences.py 逐条对齐", () => {
    const py = pythonTemplates();

    it("解析出来的不是空表（解析器自己坏了要能看出来）", () => {
      // 没有这一条，正则一旦失配就会"两边都是空表"从而全绿
      expect(py.length).toBeGreaterThan(50);
      expect(py.every((t) => t.length >= 2)).toBe(true);
    });

    it("条数相同", () => {
      expect(py.length).toBe(SENTENCE_TEMPLATES.length);
    });

    it("每一条的标签序列相同，顺序也相同", () => {
      // 逐条比而不是比整个数组：红的时候要能一眼看出是哪一句不一样
      const ts = SENTENCE_TEMPLATES.map((t) => templateKey(t));
      const python = py.map((t) => templateKey(t));
      expect(
        python.filter((k) => !ts.includes(k)),
        "Python 有、TS 没有 —— 这些句型训了但顺不出汉语，也采不了"
      ).toEqual([]);
      expect(
        ts.filter((k) => !python.includes(k)),
        "TS 有、Python 没有 —— 采了也没有对应的合成句型"
      ).toEqual([]);
      // 顺序也锁：句型下标出现在 train_seq 的日志里，两边错位会让排查时对不上号
      expect(python).toEqual(ts);
    });
  });

  describe("不训练的词", () => {
    it("与 synth_sentences.py 的 UNTRAINED_WORDS 逐条相同", () => {
      // 这是"两套词表"的锁。两边不一致的后果：句子模型输出一个词模型没训过的词，
      // 不报错、合成 WER 也看不出来，只有上机演示时偶尔蹦一个奇怪的词
      expect(pythonUntrained()).toEqual([...UNTRAINED_WORDS]);
    });

    it("不是空表（解析器坏了要能看出来）", () => {
      expect(UNTRAINED_WORDS.length).toBeGreaterThan(0);
    });

    it("任何句型都不含不训练的词", () => {
      const drop = new Set(UNTRAINED_WORDS);
      const bad = SENTENCE_TEMPLATES.filter((t) => t.some((w) => drop.has(w)));
      expect(
        bad.map((t) => templateKey(t)),
        "这些句型用到了不训练的词 —— 合成时抽不到录制，句型会被静默丢掉"
      ).toEqual([]);
    });

    it("合并组的 defaultMember 不指向不训练的词", () => {
      // `merged_pron_pl` 的默认值曾经是 `you_pl`，而 you_pl 不再训练 ——
      // 顺句时整组会翻译成一个模型从来没见过的词
      const drop = new Set(UNTRAINED_WORDS);
      const bad = MERGE_GROUPS.filter(
        (g) => drop.has(g.defaultMember) || (g.selfMember && drop.has(g.selfMember))
      );
      expect(bad.map((g) => g.id), "这些合并组的默认翻译指向不训练的词").toEqual([]);
    });
  });

  describe("第一批采集清单", () => {
    it("每条都是全表的成员（防手抄错字）", () => {
      // 错字的后果很贵：采了 20 条，训练时那个句型对不上任何合成句型，
      // 而 WER 只会小幅变差，不会报错
      const all = new Set(SENTENCE_TEMPLATES.map((t) => templateKey(t)));
      const bad = BATCH1_TEMPLATES.map((t) => templateKey(t)).filter(
        (k) => !all.has(k)
      );
      expect(bad, "这些句型不在 SENTENCE_TEMPLATES 里").toEqual([]);
    });

    it("没有重复", () => {
      const keys = BATCH1_TEMPLATES.map((t) => templateKey(t));
      expect(new Set(keys).size).toBe(keys.length);
    });

    it("没有猜的成分的句子 8 条、有猜的成分 7 条", () => {
      // 判据是**这一句里有没有落在合并类上的位置**，不是"有没有代词"：
      // 「我」已经是独立类（指自己胸口、有接触，见 labelMerge.ts），
      // 所以 ["i","eat"] 和 ["eat","what"] 在验收上等价 —— 每个字都是模型认出来的。
      // 这类句子是唯一能"全对/全错"二分判断的材料；合并类位置永远是规则表给的默认值。
      //
      // 这条同时锁着一个反向的坑：要是「我」哪天被并回合并组，这里的 8 会掉到 3，
      // 测试立刻红 —— 而线上的症状只是"演示时方向偶尔反"，很难定位。
      const MERGED_MEMBERS = new Set(["you", "he", "we", "you_pl", "they"]);
      const certain = BATCH1_TEMPLATES.filter(
        (t) => !t.some((w) => MERGED_MEMBERS.has(w))
      );
      expect(certain.length).toBe(8);
      // 2026-08-29：4 → 7。新加的三句（你叫什么名字 / 你真好看 / 你笑起来像太阳）
      // 全都以「你」开头，所以三句都落在合并类上。`certain` 那个 8 没变 ——
      // 它是「我」有没有被并回去的探针，不该跟着这次改动动
      expect(BATCH1_TEMPLATES.length - certain.length).toBe(7);
    });

    it("「我」在第一批里真的出现了（拆成独立类之后才敢采）", () => {
      // 拆类之前采「我」是浪费：它会被并进 merged_pron_sg，采多少条都不会
      // 多出一个可输出的词
      expect(BATCH1_TEMPLATES.some((t) => t.includes("i"))).toBe(true);
    });

    it("采集量是个合理数（改大之前先想清楚要打多久）", () => {
      // 15 句 × 20 条 × ~10s ≈ 50 分钟净打时间
      expect(BATCH1_TEMPLATES.length * RECOMMENDED_PER_TEMPLATE).toBe(300);
    });
  });

  describe("templateKey", () => {
    it("同序列同 key、不同序列不同 key", () => {
      expect(templateKey(["i", "love", "you"])).toBe("i love you");
      // 词序是句型的全部信息（CTC 只读顺序），所以换序必须是另一个 key
      expect(templateKey(["you", "love", "i"])).not.toBe(
        templateKey(["i", "love", "you"])
      );
    });
  });
});
