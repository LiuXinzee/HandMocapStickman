"""
生成 Python↔浏览器的**数值对照 fixture**。

计划里点名要这个东西:"把 Python 侧同一条 val 样本的 logits 存成 fixture,断言 JS 解码
结果与 Python greedy 一致(这是唯一能抓住两边 blank 约定不一致的测试)"。

这里比计划多做了一件事 —— 除了解码对照,还存了一条**确定性输入**的完整前向输出。
原因是换成"浏览器自己搭结构 + 填权重"之后,多出一类新的失配:tfjs 的层实现和 keras
不完全一致(最典型的是 BatchNormalization 的 epsilon 默认值,keras 3 是 1e-3;还有
conv1d 的 padding='same' 在偶数 kernel 下左右补零不对称)。这类失配不会报错,只会让
输出偏一点 —— 偏一点就足够让 argmax 换人。只对照解码结果的话抓不到它。

===== 两组数据 =====

1. `ramp` —— 确定性输入,不用随机数:

       x[t][d] = ((t * frameDim + d) % 97) / 97

   两边逐字实现同一个公式即可,不涉及浮点 RNG 的跨语言复现问题(那件事很难做对:
   JS 的 imul/ToInt32 语义要在 Python 里手工掩位)。取 97 是质数且与 frameDim=294
   互质,保证同一时刻各通道不同、相邻时刻不重复,conv 核会被真正激励到 ——
   常数输入(全 0/全 1)测不出核里权重排布错位。
   这一组验的是**权重 + 层实现**,和数据无关。

2. `samples` —— 几条真实(合成)验证句的输出概率 + Python 的 greedy 解码结果。
   这一组验的是**解码器**,概率分布是真实形状(有犹豫、有 blank 长段)。

===== 复现验证集划分 =====

按 train_ctc 里的同一个顺序消耗 rng:没有真实句子时 rng 的第一次使用就是
`split_by_sequence(word_seqs, val_split, rng)`。所以同一个 --seed / --val-split
就能拿到同一批验证录制。

划分**万一**将来漂了也不影响测试有效性:probs 是真跑前向拿到的,ref 一起存了下来,
fixture 自身始终自洽。所以下面把 seed 和 val_split 一并写进文件,是为了可追溯,
不是测试的正确性依赖。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RAMP_MOD = 97


def ramp_input(seq_len: int, frame_dim: int) -> np.ndarray:
    """x[t][d] = ((t*frame_dim + d) % 97) / 97 —— 与 JS 侧逐字一致的确定性输入。"""
    idx = np.arange(seq_len * frame_dim, dtype=np.int64)
    return ((idx % RAMP_MOD) / RAMP_MOD).astype(np.float32).reshape(1, seq_len, frame_dim)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="生成 JS 侧对照 fixture")
    ap.add_argument("--model", default="out/sentence_student.keras")
    ap.add_argument("--meta", default="out/sentence_student_meta.json")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="../client/src/lib/sentenceCtcFixture.json")
    ap.add_argument("--samples", type=int, default=6, help="错例/正确例各一半")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--synth-per-template", type=int, default=24)
    args = ap.parse_args()

    import keras

    from ctc_decode import greedy_decode
    from load_dataset import load_dataset
    from synth_sentences import synthesize_sentences
    from train_seq import build_ctc_xy, split_by_sequence

    meta = json.loads(Path(args.meta).read_text(encoding="utf-8"))
    classes: list[str] = meta["labels"]
    blank: int = meta["blankIndex"]
    seq_len: int = meta["seqLen"]
    frame_dim: int = meta["frameDim"]
    model = keras.models.load_model(args.model, compile=False)

    # --- 1. 确定性输入的前向 ---
    xr = ramp_input(seq_len, frame_dim)
    pr = model.predict(xr, verbose=0)[0]
    print(f"ramp 前向 {pr.shape},概率和 {pr.sum(axis=-1).min():.6f}~{pr.sum(axis=-1).max():.6f}")
    ramp_dec = greedy_decode(pr, blank)

    # --- 2. 真实(合成)验证句 ---
    all_seqs, labels = load_dataset(args.data)
    word_seqs = [s for s in all_seqs if not s.is_sentence]
    sent_seqs = [s for s in all_seqs if s.is_sentence]
    rng = np.random.default_rng(args.seed)
    real_tr, real_va = split_by_sequence(sent_seqs, args.val_split, rng) if sent_seqs else ([], [])
    w_tr, w_va = split_by_sequence(word_seqs, args.val_split, rng)
    # 句型表必须跟训练时一致:训练时验证句用的是**训练集能凑出来的**那批句型
    _, used = synthesize_sentences(w_tr, args.synth_per_template, rng, verbose=False)
    syn_va, _ = synthesize_sentences(
        w_va, max(1, args.synth_per_template // 4), rng, templates=used, verbose=False
    )
    val_seqs = real_va + syn_va
    if not val_seqs:
        raise SystemExit("验证集为空,没法生成 fixture")

    xv, yv, ylv = build_ctc_xy(val_seqs, classes, seq_len)
    probs = model.predict(xv, batch_size=16, verbose=0)
    refs = [list(map(int, yv[i, : ylv[i]])) for i in range(len(xv))]
    hyps = [greedy_decode(probs[i], blank) for i in range(len(xv))]

    # 挑样本。两个要求:
    #   - 错例优先。全对的样本 argmax 遥遥领先,两边差 1e-4 也不改解码结果,测不出失配;
    #     错例才踩在 argmax 的边界上。
    #   - **按参考句去重,错例和正确例混搭。** 不去重的话会抽到 3 条一模一样的
    #     "sg love sg",4 个样本只覆盖 1 种情况 —— 看着有 4 条,实际测试面没变宽
    def _pick(pool: list[int], k: int) -> list[int]:
        seen: set[tuple] = set()
        out: list[int] = []
        for i in sorted(pool, key=lambda i: -len(refs[i])):
            key = tuple(refs[i])
            if key in seen:
                continue
            seen.add(key)
            out.append(i)
            if len(out) >= k:
                break
        return out

    bad = [i for i in range(len(xv)) if refs[i] != hyps[i]]
    good = [i for i in range(len(xv)) if refs[i] == hyps[i]]
    n_bad_pick = min(len(bad), (args.samples + 1) // 2)
    picked = _pick(bad, n_bad_pick)
    picked += _pick(good, args.samples - len(picked))
    n_bad = len(bad)
    if not picked:
        raise SystemExit("一条样本都没挑出来")

    def r6(a: np.ndarray) -> list:
        return np.round(a, 6).tolist()

    fixture = {
        "_comment": (
            "由 python_train/export_fixture.py 生成,不要手改。"
            "probs 是 softmax 之后的概率(frame_wise_head 带 softmax,没有 logits)。"
            "ramp.input 的公式见 rampInput() 注释,JS 侧要逐字实现同一个公式。"
        ),
        "labels": classes,
        "blankIndex": blank,
        "seqLen": seq_len,
        "frameDim": frame_dim,
        "outputFrames": int(pr.shape[0]),
        "numClasses": len(classes),
        "provenance": {
            "seed": args.seed,
            "valSplit": args.val_split,
            "synthPerTemplate": args.synth_per_template,
            "valSize": int(len(xv)),
            "valMismatched": int(n_bad),
        },
        "ramp": {
            "formula": f"x[t][d] = ((t * {frame_dim} + d) % {RAMP_MOD}) / {RAMP_MOD}",
            "mod": RAMP_MOD,
            "probs": r6(pr),
            "decoded": ramp_dec,
        },
        "samples": [
            {
                "ref": refs[i],
                "refWords": [classes[k] for k in refs[i]],
                "decoded": hyps[i],
                "decodedWords": [classes[k] for k in hyps[i]],
                "exact": refs[i] == hyps[i],
                "probs": r6(probs[i]),
            }
            for i in picked
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, ensure_ascii=False), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"已写 {out}({kb:.0f} KB)")
    print(f"验证集 {len(xv)} 条,不完全一致 {n_bad} 条;抽了 {len(picked)} 条"
          f"({sum(1 for i in picked if refs[i] != hyps[i])} 条是错例)")
    print(f"ramp 解码 → {[classes[k] for k in ramp_dec] or '(空)'}")
    for i in picked:
        print(f"  {' '.join(classes[k] for k in refs[i])}"
              f"  →  {' '.join(classes[k] for k in hyps[i]) or '(空)'}")


if __name__ == "__main__":
    main()
