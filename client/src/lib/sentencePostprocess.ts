/*
 * sentencePostprocess —— 句子模型解出词序列之后的两道后处理
 *
 * ===== 为什么这个文件要存在 =====
 *
 * 逐词滑窗那条路上有两道后处理（`fistGate` 拇指闸门 + `compoundWords` 复合词回收），
 * 句子路径**一道都没走**：`applyFistGate` / `matchCompound` 只出现在 `Translate.tsx`
 * 那个 100ms 定时器的 sequence 分支里。后果实测过（2026-08-30）：打「你好」，
 * 句子档输出「你 难过」—— 没人把难过扳成谢谢，也没人把 你+谢谢 合成你好，
 * 两个词就那么原样出来了。
 *
 * 这两道后处理为什么必要、代价是什么，分别在 `fistGate.ts` 和 `compoundWords.ts`
 * 的文件头里，**不在这里重抄**。这个文件只解决"把它们套到一个词序列上"这件事，
 * 其中只有一个决定是新的：
 *
 * ===== 逐词取区间，在这里还有第二个后果（2026-08-31 补） =====
 *
 * 闸门的主判据后来换成了四元数转角速率（`rotationRate.ts`），而那个判据有一条
 * 硬性下限：区间短于 1500ms 就**不可用**（短窗下它比拇指还差，那边有实测表）。
 * 句子里一个词常常只占 750~1500ms，所以这条路上速率判据大多数时候是关着的、
 * 仍由拇指管。这是刻意的，不是没接上 —— 逐词档的窗固定 2000ms，那边才是它的主场。
 *
 * ===== 唯一的新决定：闸门逐词看自己那几帧 =====
 *
 * 词路径上一个窗口就是一个词，`thumbPeak(整个窗口)` 天然就是"这个词的拇指峰值"。
 * 句子里一句话有 3~6 个词，整句取一个全局峰值会**制造**错误：实测「好看」全段
 * 拇指峰值中位 26，一句「你真好看」里只要解出一个「难过」，全局峰值就把它翻成
 * 「谢谢」—— 本来没错的地方被改错了。所以必须用 CTC 的逐词帧区间
 * （`greedyDecodeSpans`）把峰值限制在那个词自己的那一段里。
 *
 * ===== 顺序：先闸门、后合并 =====
 *
 * 复合词规则匹配的是 `you → thank_you`。闸门有可能把 `thank_you` 判成 `sad`
 * （拇指没吃力时），也可能反过来。**先合并再闸门**的话，`你 难过` 这一对根本
 * 不会被规则看到，「你好」永远合不出来。词路径上的顺序也是先闸门后规则
 * （Translate.tsx:762 在 :838 之前），这里保持一致。
 */
import type { SequenceSample } from "./datasetStore";
import type { SentencePrediction } from "./sentenceModel";
import { isCompoundWord, mergeCompoundsInWords } from "./compoundWords";
import { applyFistGate, isGatedPair, THUMB_PEAK_GATE, thumbPeakSpan } from "./fistGate";
import { rotationRateSpan } from "./rotationRate";
import { isSuppressed } from "./suppressedWords";

export interface SentencePostprocessResult {
  /** 后处理之后的词序列，直接拿去显示 */
  words: string[];
  /** 模型原始输出，界面上要说明"改了什么"时用 */
  rawWords: string[];
  /**
   * 闸门改判的每一处：第几个词、从什么改成什么、当时那一段的拇指峰值，
   * 以及是**哪个判据**改的（`reason`，见 `GateResult.reason`）。
   * 界面上要能区分是转角判的还是拇指判的 —— 否则调阈值时不知道该调哪一个。
   */
  gated: { at: number; from: string; to: string; peak: number; reason: string }[];
  /** 合并掉的复合词 id（`hello` / `beautiful`），没有就是空数组 */
  merged: string[];
  /**
   * 被 `SUPPRESSED_WORDS` 删掉的词（按出现顺序，重复出现就重复记）。
   *
   * **必须报出来**：不报的话线上表现是"模型莫名少解出一个词"，而真正动手的是
   * 这张表。屏蔽本来就是权宜之计，不显示出来的话下次没人记得它开着。
   */
  suppressed: string[];
}

/**
 * 在**同一帧区间**里取该词之外概率最高的类别。
 *
 * 排除 blank：闸门仲裁的是"这是哪个**词**"，blank 不是词。不排的话大多数帧的
 * 第二名都是 blank，`isGatedPair` 永远不成立，整条闸门在句子路径上静默失效。
 */
function runnerUpAt(
  pred: SentencePrediction,
  frame: number,
  selfIndex: number,
  labels: readonly string[]
): string | undefined {
  const { perFrame, numClasses, blankIndex } = pred;
  const base = frame * numClasses;
  let best = -1;
  let bestP = -1;
  for (let c = 0; c < numClasses; c++) {
    if (c === selfIndex || c === blankIndex) continue;
    const p = perFrame[base + c] ?? -1;
    if (p > bestP) {
      bestP = p;
      best = c;
    }
  }
  return best < 0 ? undefined : labels[best];
}

/**
 * 跑完两道后处理。
 *
 * @param pred   `predictSentence` 的原始返回（`spans` 必须与 `words` 等长，那边有断言）
 * @param sample 喂给模型的那条样本，**归一化到右手口径之后的**（`thumbPeakSpan` 读
 *               右手 0..12 通道，镜像之前读到的是小拇指 —— 见 fistGate.thumbPeak）
 * @param labels 类别表（`meta.labels`），把第二名的下标翻回词
 */
export function postprocessSentence(
  pred: SentencePrediction,
  sample: SequenceSample,
  labels: readonly string[]
): SentencePostprocessResult {
  const rawWords = [...pred.words];
  const gated: SentencePostprocessResult["gated"] = [];
  const afterGate = [...pred.words];

  for (let i = 0; i < afterGate.length; i++) {
    const span = pred.spans[i];
    if (!span) continue;
    const top1 = afterGate[i];
    const top2 = runnerUpAt(pred, span.peakFrame, span.index, labels);
    // 先用便宜的判据挡掉绝大多数词，别为每个词都去扫一遍传感器数组
    if (!isGatedPair(top1, top2)) continue;
    // 区间是**输出帧**（T/4）下标，转成样本里的归一化时间位置
    const from01 = span.startFrame / pred.outputFrames;
    const to01 = span.endFrame / pred.outputFrames;
    const peak = thumbPeakSpan(sample, from01, to01);
    /*
     * 转角速率也按同一个区间取。**大多数词区间会短于 `ROT_MIN_SPAN_MS`(1500ms)**
     * —— 一句 12s 的话摊成 32 个输出帧，一个词占 2~4 帧就是 750~1500ms ——
     * 那种情况 `rotationRateSpan` 返回 `usable: false`，`applyFistGate` 自动退回
     * 纯拇指判据。这不是妥协，是实测结论：短区间下速率判据比拇指还差
     * （800ms 窗 77.8% vs 拇指 84.0%），见 rotationRate.ts 的文件头。
     * 所以句子路径上这一行**平时不起作用**，只在某个词确实占了 1.5s 以上时接管。
     */
    const rot = rotationRateSpan(sample, from01, to01);
    const g = applyFistGate(top1, top2, peak, THUMB_PEAK_GATE, rot);
    if (g.changed) {
      gated.push({ at: i, from: top1, to: g.label, peak, reason: g.reason });
      afterGate[i] = g.label;
    }
  }

  const merged0 = mergeCompoundsInWords(afterGate);
  /*
   * 屏蔽放在**合并之后**：万一某个被屏蔽的词是复合词的组成部分，
   * 先删就等于把那个复合词也一起毁了。合并完再删，只影响它自己。
   */
  const suppressed = merged0.filter(isSuppressed);
  const words = suppressed.length ? merged0.filter((w) => !isSuppressed(w)) : merged0;
  /*
   * 哪些是合出来的:数"复合词产物"的**净增量**,不是直接 filter。
   * `hello` 本身就是词表里的词,模型完全可以直接解出一个 `hello`
   * (实测 12/15 条 hello 录制就是直接解出来的),那种不是合并的产物。
   */
  const before = afterGate.filter(isCompoundWord);
  // 数的是**合并之后、屏蔽之前**那一份：屏蔽掉一个复合词不该让"合成了什么"的
  // 账目跟着少一笔（今天 `eat` 不是复合词，但这张表随时会加词）
  const merged = merged0.filter(isCompoundWord);
  for (const w of before) {
    const k = merged.indexOf(w);
    if (k >= 0) merged.splice(k, 1);
  }

  return { words, rawWords, gated, merged, suppressed };
}
