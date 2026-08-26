"""
CTC 解码与 WER —— 纯 numpy,不依赖 tensorflow。

单独成文件有两个原因:
  1. 它是浏览器端 `client/src/lib/ctcDecode.ts` 的**参照实现**。两边必须逐字对应,
     放在训练脚本里跟一堆 TF 代码混着,改的时候很容易忘了同步另一边。
  2. 不依赖 TF 就能直接跑自检(见 __main__),不用装 300MB 的东西。
"""
from __future__ import annotations

import numpy as np


def greedy_decode(probs: np.ndarray, blank: int) -> list[int]:
    """
    CTC greedy 解码:逐帧 argmax → 折叠**连续**重复 → 去掉 blank。

    两件事的**顺序不能换**:先去 blank 再折叠,会把 [a, blank, a](本该是两个 a)
    折成一个 a —— 重复词就永远打不出来。CTC 用 blank 分隔重复标签,这是它的全部要点。

    blank 在**末位**(= 类别数),与 train_seq.py 的 frame_wise_head(num_classes+1)
    和 tf.nn.ctc_loss(blank_index=-1) 一致。这个约定错一处,整句全错。
    """
    best = probs.argmax(axis=-1)
    out: list[int] = []
    prev = -1
    for k in best:
        k = int(k)
        if k != prev and k != blank:
            out.append(k)
        prev = k
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    """Levenshtein 距离。WER 的分子。"""
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def word_error_rate(refs: list[list[int]], hyps: list[list[int]]) -> float:
    """
    Σ编辑距离 / Σ参考长度。

    **句子级只能看这个,不能看逐帧 accuracy。** 逐帧标签里 blank 占绝大多数
    (32 个输出帧对 3~6 个词),一个恒输出 blank 的废模型逐帧准确率就有 85%+,
    看着像训好了,实际一个词都解不出来。
    """
    num = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    den = sum(len(r) for r in refs)
    return num / max(den, 1)


if __name__ == "__main__":
    B = 3  # 3 类 + blank(下标 3)

    def onehot(seq: list[int], c: int = 4) -> np.ndarray:
        p = np.full((len(seq), c), 0.01, np.float32)
        for t, k in enumerate(seq):
            p[t, k] = 0.9
        return p

    # 折叠连续重复
    assert greedy_decode(onehot([0, 0, 0, B, 1, 1]), B) == [0, 1]
    # 跨 blank 的重复**不折叠** —— 重复词全靠这条
    assert greedy_decode(onehot([0, B, 0]), B) == [0, 0]
    # 全 blank → 空
    assert greedy_decode(onehot([B, B, B]), B) == []
    # blank 在两端不影响
    assert greedy_decode(onehot([B, 2, B, 2, B]), B) == [2, 2]
    # blank 不在末位就会把真类当成 blank —— 这里用错误的 blank 演示后果
    assert greedy_decode(onehot([0, B, 0]), 0) == [B]

    assert edit_distance([], []) == 0
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [1, 3]) == 1          # 删
    assert edit_distance([1, 3], [1, 2, 3]) == 1          # 插
    assert edit_distance([1, 2, 3], [1, 9, 3]) == 1       # 替
    assert edit_distance([], [1, 2]) == 2

    assert word_error_rate([[1, 2, 3]], [[1, 2, 3]]) == 0.0
    assert word_error_rate([[1, 2, 3]], [[]]) == 1.0
    assert abs(word_error_rate([[1, 2, 3], [4]], [[1, 3], [4]]) - 0.25) < 1e-9
    print("ctc_decode OK")
