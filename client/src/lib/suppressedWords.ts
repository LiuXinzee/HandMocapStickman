/*
 * suppressedWords —— 输出屏蔽表（**纯显示层**）。
 *
 * ===== 和 `UNTRAINED_WORDS` 的区别 =====
 *
 * `sentenceTemplates.UNTRAINED_WORDS` 是**训练端**的清单：那些词不进句型合成、
 * 不进标签表，改了要重训，而且有跨语言测试锁着（Python 那边同名常量）。
 *
 * 这张表相反：模型照旧训练、照旧输出，只是在**送去显示之前**把它删掉。
 * 加一个词、删一个词都不需要重训，也不影响任何已存的样本或模型文件。
 *
 * ===== 为什么单独一个模块 =====
 *
 * 两条推理路径都要用它，而它们互不依赖：
 *   - 句子档：`sentencePostprocess.ts`（整句解完，从词序列里删）
 *   - 逐词档：`Translate.tsx` 的 100ms 滑窗（命中就整个丢掉这一跳的预测，
 *     与 `IDLE_LABEL` 同一种处理 —— 既不显示，也不进历史）
 * 把常量放在其中任一个文件里，另一条路就得反向依赖它。
 */

/**
 * 永远不显示的词（类别 id，不是中文）。
 *
 * `eat` —— 2026-08-31 加。现场实测：连续模式下大量「你」被解成「吃」。
 *
 * ⚠ **两条代价，加词的人必须知道：**
 *
 * 1. `eat` 是**训练过的真词**，句型表里 8 条句子用它（你吃什么 / 我吃 / 不吃 /
 *    吃喝 / 爱吃 …，见 `python_train/synth_sentences.py`）。屏蔽之后那些句子
 *    永远打不出来 —— 「我吃」会变成「我」。
 * 2. 屏蔽只藏**症状**：本来该是「你」的那个词现在不是变成「吃」，而是整个消失，
 *    句子照样缺一个词。病因（输入退化 → CTC 塌到先验最大的类）还在。
 *
 * 所以这是权宜之计。屏蔽发生时两条路径都会在界面上写「已屏蔽：吃」——
 * 不报的话下次没人记得这张表开着，会去查模型。
 */
export const SUPPRESSED_WORDS: readonly string[] = ["eat"];

/** 这个词该被屏蔽吗。`words.filter(w => !isSuppressed(w))` 就是全部用法 */
export function isSuppressed(word: string): boolean {
  return SUPPRESSED_WORDS.includes(word);
}
