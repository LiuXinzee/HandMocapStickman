/*
 * ctcDecode — CTC greedy 解码。
 *
 * 参照实现是 `python_train/ctc_decode.py`，两边必须逐字对应。对不上的症状不是报错，
 * 而是"整句解码结果和 Python 训练时报的 WER 完全不符" —— 而 WER 是唯一验收指标，
 * 对不上就等于没有验收标准了。`ctcDecode.test.ts` 里用 Python 导出的 fixture
 * （`sentenceCtcFixture.json`）逐条对照，就是为了钉住这件事。
 *
 * tfjs 4.22 **没有** CTC 解码器（dist 里搜不到任何 ctc 符号），所以只能手写。
 * 好在 greedy 解码就是三行，难的地方全在约定上：
 *
 *   1. blank 在**末位**（= 类别数）。与 python 侧 `frame_wise_head` 的
 *      `num_classes+1` 和 `tf.nn.ctc_loss(blank_index=-1)` 一致。这个下标必须从
 *      `meta.json` 的 `blankIndex` 读，不要在这里写 `labels.length` 另算一遍 ——
 *      两处独立计算就有两处会各自漂。
 *   2. 先折叠连续重复、再去 blank。顺序反了会把 `[a, blank, a]`（本该两个 a）
 *      折成一个 a，重复词永远打不出来。blank 分隔重复标签是 CTC 的全部要点。
 */

/**
 * 逐帧 argmax → 折叠连续重复 → 去掉 blank。
 *
 * @param probs 扁平的 [frames * numClasses]，行优先（第 t 帧在 [t*C, (t+1)*C)）。
 *              是 softmax 之后的概率而不是 logits —— `frame_wise_head` 自带 softmax，
 *              拿不到 logits。对 argmax 没有区别（softmax 单调）。
 * @param frames 时间步数。**是骨干输出的帧数（T/4），不是输入的 T。**
 * @param numClasses 含 blank 的总维数（= labels.length + 1）。
 * @param blank blank 的下标，从 meta 读。
 */
export function greedyDecode(
  probs: Float32Array | number[],
  frames: number,
  numClasses: number,
  blank: number
): number[] {
  if (frames <= 0 || numClasses <= 0) return [];
  if (probs.length < frames * numClasses) {
    throw new Error(
      `greedyDecode: 概率长度 ${probs.length} 不够 ${frames}×${numClasses}=${frames * numClasses}`
    );
  }
  const out: number[] = [];
  let prev = -1;
  for (let t = 0; t < frames; t++) {
    const base = t * numClasses;
    let best = 0;
    let bestP = probs[base];
    for (let c = 1; c < numClasses; c++) {
      const p = probs[base + c];
      if (p > bestP) {
        bestP = p;
        best = c;
      }
    }
    // 顺序：先判"和上一帧不同"（折叠），再判"不是 blank"（丢弃）。
    // prev 记的是**折叠前**的 argmax，所以 blank 帧也要更新 prev ——
    // 不更新的话 [a, blank, a] 里第二个 a 会被当成重复而丢掉
    if (best !== prev && best !== blank) out.push(best);
    prev = best;
  }
  return out;
}

/** `greedyDecodeSpans` 的一项：解出来的一个词，以及它占了哪几个输出帧。 */
export interface DecodedSpan {
  index: number;
  /** 这个词的 argmax 连续段，闭开区间 `[startFrame, endFrame)`，单位是**输出帧**（T/4） */
  startFrame: number;
  endFrame: number;
  /** 段内该类别概率最高的那一帧。要看"这个词最像的时刻"的逐帧概率时用它 */
  peakFrame: number;
}

/**
 * 带帧区间的 greedy 解码。**索引序列与 `greedyDecode` 必须逐项相同**
 * （`ctcDecode.test.ts` 拿同一批输入对照两者，锁死这一条）。
 *
 * 为什么需要区间：拇指压力闸门（`fistGate.ts`）要判的是"**这个词**打的时候拇指
 * 有没有吃力"。整句取一个全局峰值是错的 —— 实测「好看」全段拇指峰值中位 26，
 * 一句「你真好看」里只要出现「难过」，全局峰值就会把它翻成「谢谢」。
 * 逐词区间不是精细化，是这条闸门在句子路径上成立的前提。
 *
 * 区间取的是**折叠前的 argmax 连续段**：`[a,a,blank,a]` 解出两个 a，
 * 区间分别是 `[0,2)` 和 `[3,4)`。
 */
export function greedyDecodeSpans(
  probs: Float32Array | number[],
  frames: number,
  numClasses: number,
  blank: number
): DecodedSpan[] {
  if (frames <= 0 || numClasses <= 0) return [];
  if (probs.length < frames * numClasses) {
    throw new Error(
      `greedyDecodeSpans: 概率长度 ${probs.length} 不够 ${frames}×${numClasses}=${frames * numClasses}`
    );
  }
  const out: DecodedSpan[] = [];
  let prev = -1;
  for (let t = 0; t < frames; t++) {
    const base = t * numClasses;
    let best = 0;
    let bestP = probs[base];
    for (let c = 1; c < numClasses; c++) {
      const p = probs[base + c];
      if (p > bestP) {
        bestP = p;
        best = c;
      }
    }
    if (best === prev) {
      // 同一个 argmax 连续段：延长上一项，并在段内追峰值帧。
      // blank 段没有对应的输出项，所以只在 out 非空且末项就是这个类别时才延长
      const last = out[out.length - 1];
      if (last && last.index === best && last.endFrame === t) {
        last.endFrame = t + 1;
        const pk = probs[last.peakFrame * numClasses + best];
        if (bestP > pk) last.peakFrame = t;
      }
    } else if (best !== blank) {
      out.push({ index: best, startFrame: t, endFrame: t + 1, peakFrame: t });
    }
    prev = best;
  }
  return out;
}

/** 把解码出来的类别下标翻成词。下标越界时抛错而不是静默跳过：越界意味着 blank 约定错了。 */
export function decodeToWords(indices: number[], labels: string[]): string[] {
  return indices.map((i) => {
    const w = labels[i];
    if (w === undefined) {
      throw new Error(`解码下标 ${i} 超出标签表长度 ${labels.length} —— blankIndex 可能配错了`);
    }
    return w;
  });
}

/** Levenshtein 距离。WER 的分子。 */
export function editDistance<T>(a: readonly T[], b: readonly T[]): number {
  if (a.length === 0) return b.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = new Array<number>(b.length + 1);
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(
        prev[j] + 1,
        cur[j - 1] + 1,
        prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
    }
    prev = cur;
  }
  return prev[b.length];
}

/**
 * Σ编辑距离 / Σ参考长度。
 *
 * **句子级只能看这个，不能看逐帧准确率。** 逐帧标签里 blank 占绝大多数
 * （32 个输出帧对 3~6 个词），一个恒输出 blank 的废模型逐帧准确率就有 85%+，
 * 看着像训好了，实际一个词都解不出来。
 */
export function wordErrorRate<T>(refs: readonly T[][], hyps: readonly T[][]): number {
  let num = 0;
  let den = 0;
  const n = Math.min(refs.length, hyps.length);
  for (let i = 0; i < n; i++) {
    num += editDistance(refs[i], hyps[i]);
    den += refs[i].length;
  }
  return num / Math.max(den, 1);
}
