/*
 * compoundWords —— 逐词滑窗把复合词拆成两个词之后，在**输出侧**把它拼回来
 *
 * ===== 为什么需要这个，以及为什么它只能是个后处理规则 =====
 *
 * 有些手语词由两段组成，而**第二段本身就是另一个已训练的词**：
 *
 *   你好  = 食指前指(你) + 竖大拇指     ← 竖大拇指向前在触觉特征上就是「谢谢」
 *   好看  = ……            + 竖大拇指     ← 实测输出是「像 谢谢」
 *
 * 部署的词模型是**纯触觉**的（`frameDim: 294`，没有视觉通道），所以它看不到手在
 * 空间里的哪个位置，只看得到手型 + 相对朝向 + 加速度。竖大拇指这个手型在哪儿做
 * 都一样 —— 模型没有任何通道能把「你好的第二段」和「谢谢」分开。这不是训得不够，
 * 是当前传感器下这两段**不可分**。
 *
 * 而且滑窗推理没有词边界的概念：`Translate` 每 100ms 拿最近 2000ms 预测一次，
 * 谁先稳定谁先出词。两段各自稳定一次，就出两个词。
 *
 * 所以修法只有三条路，这个文件是第三条：
 *   1. 换九轴 IMU / 接视觉 → 能看到手的位置，模型自己就分开了。硬件改动。
 *   2. 重录 `hello`，让训练数据里的 hello 真的包含两段（现有 15 条是 2026-08-13 的
 *      旧打法「食指中指向前点头」，**根本没有大拇指那一段** —— 见 signLanguageVocab
 *      的文件头）。最干净，但要重录。
 *   3. **在确认输出之后，按规则把相邻两个词合回去**。就是本文件。
 *
 * ===== 这个规则的代价，以及为什么使用者接受它 =====
 *
 * 规则是无条件的：只要连着确认到 你 → 谢谢（间隔够短），就一定合成 你好。
 * 所以真想说「你 谢谢」这两个词时会被吞掉。
 *
 * 使用者明确判断过这一点：**「没有『你谢谢』以及『像谢谢』这个句子」** ——
 * 中文里这两个词序不构成句子，代价是零。加新规则前先问同一个问题：
 * 那两个词连着说是不是一句合法的话？是的话就不能加。
 *
 * ===== 为什么只支持两段 =====
 *
 * 触发点用的是 `Translate` 现成的 `lastAddedWordRef` / `lastAddedTimeRef`
 * —— 只记得**最近一个**确认词。三段词要另攒一个历史队列，而现在一条三段的
 * 复合词都没有。等真有了再改，别提前泛化。
 *
 * ===== 与标签合并的关系（这一条错了整个规则就静默失效） =====
 *
 * 模型输出的是**合并后**的类别 id：打「你」出来的是 `merged_pron_sg`，不是 `you`。
 * 所以规则里的 `you` 必须先过 `mergeLabel` 才能和模型输出对上。匹配时两边都
 * 归一化，于是同一条规则在"合并开着"和"关掉做对照实验"两种情况下都成立。
 */
import { mergeLabel } from "./labelMerge";

export interface CompoundRule {
  /** 两段各自被识别成的**原始**词 id（写 `you`，匹配时自动归一到合并类） */
  parts: [string, string];
  /** 合成出来的词 id（必须是词表里真实存在的 id，有测试锁着） */
  word: string;
  /** 为什么这两段会被拆开 —— 免得日后有人当 bug 删掉 */
  reason: string;
}

/**
 * 两段之间最长间隔。超过就不合 —— 那更可能是两个独立的词。
 *
 * 1500ms 的来源：确认一个词要连续 N 个窗口一致（默认 3 个 × 100ms），
 * 两段之间还有手型切换，实测间隔在几百毫秒量级。留一倍余量。
 * **不是**从推理窗口 2000ms 推出来的，两者无关。
 */
export const COMPOUND_MAX_GAP_MS = 1500;

export const COMPOUND_RULES: CompoundRule[] = [
  {
    parts: ["you", "thank_you"],
    word: "hello",
    reason: "你好 = 食指前指 + 竖大拇指；竖大拇指向前在纯触觉特征上与「谢谢」同型",
  },
  {
    parts: ["resemble", "thank_you"],
    word: "beautiful",
    reason: "好看的第二段同样是竖大拇指 → 被认成「谢谢」；第一段实测被认成「像」",
  },
];

/** 规则里的原始 id 归一到模型实际会输出的类别 id */
function canon(label: string): string {
  return mergeLabel(label);
}

/**
 * 刚确认的 `nextLabel` 能不能和上一个确认词合成一个复合词。
 *
 * @param prevLabel 上一个进入历史的类别 id；没有就传 null
 * @param prevAt    上一个词进入历史的时刻（`Date.now()` 口径）
 * @param nextLabel 刚确认的类别 id
 * @param now       现在（同一个口径）
 * @returns 命中的规则；不命中返回 null
 */
export function matchCompound(
  prevLabel: string | null,
  prevAt: number,
  nextLabel: string,
  now: number,
  maxGapMs: number = COMPOUND_MAX_GAP_MS
): CompoundRule | null {
  if (!prevLabel) return null;
  // 间隔为负（时钟回跳）时不合 —— 与"太久"一样按不确定处理
  const gap = now - prevAt;
  if (gap < 0 || gap > maxGapMs) return null;
  const a = canon(prevLabel);
  const b = canon(nextLabel);
  return (
    COMPOUND_RULES.find(
      (r) => canon(r.parts[0]) === a && canon(r.parts[1]) === b
    ) ?? null
  );
}

/**
 * 把一整个词序列里相邻的复合词段合回去（**句子路径**用）。
 *
 * 与 `matchCompound` 的两处区别，都是刻意的：
 *
 * 1. **不判时间间隔。** 那个 1500ms 门限防的是"逐词滑窗里两个各自确认的独立词
 *    恰好排在一起"。一句话是一次连续采集解出来的，CTC 输出里相邻就是动作上相邻，
 *    没有"隔了很久的两个词"这种情况可判。硬套一个间隔门限只会引入一个没有依据的常量。
 * 2. **从左往右一次扫完，合过的不再参与下一次匹配。** 所以 `你 谢谢 谢谢` 得到
 *    `你好 谢谢`，不会级联成别的东西。
 *
 * 安全性依据（2026-08-30 实测）：56 条句型里**没有任何一条**含相邻的
 * `you → thank_you` 或 `resemble → thank_you`；有一条 `thank_you → you`（谢谢你），
 * 是反序，不受影响。加新规则前要重跑这个检查 —— 规则一旦撞上真实句型，
 * 症状是那个句型永远解不对，而且看起来像模型的错。
 */
export function mergeCompoundsInWords(words: readonly string[]): string[] {
  const out: string[] = [];
  for (let i = 0; i < words.length; i++) {
    const a = canon(words[i]);
    const b = i + 1 < words.length ? canon(words[i + 1]) : null;
    const rule =
      b === null
        ? undefined
        : COMPOUND_RULES.find(
            (r) => canon(r.parts[0]) === a && canon(r.parts[1]) === b
          );
    if (rule) {
      out.push(rule.word);
      i++; // 吃掉第二段
    } else {
      out.push(words[i]);
    }
  }
  return out;
}

/** 这个词 id 是某条复合词规则的产物吗（界面上要标出"合出来的"时用） */
export function isCompoundWord(id: string): boolean {
  return COMPOUND_RULES.some((r) => r.word === id);
}
