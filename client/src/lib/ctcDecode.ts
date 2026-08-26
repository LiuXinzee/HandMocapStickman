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
