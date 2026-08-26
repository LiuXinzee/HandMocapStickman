"""
tf.keras 版时序手语训练 —— 结构与浏览器端 sequenceModel.ts 严格同构。

为什么需要这条 Python 路径:
tfjs 4.22 **没有** CTC loss,也没有 CTC 解码器(在 node_modules/@tensorflow/tfjs/dist/tf.js
里搜不到任何 ctc 符号)。孤立词能在浏览器里训,句子级(连续手语 CSLR)不能。
所以骨干与 head 在两边都做了分离:孤立词 head 走 GAP+softmax,句子级 head 走逐帧
softmax + tf.nn.ctc_loss,backbone 完全复用,训完用 export_to_tfjs.py 转回浏览器。

选 tf.keras 而不是 PyTorch:回导 tfjs 只要一行 tensorflowjs_converter(PyTorch 要绕 ONNX),
而且 tf.nn.ctc_loss 是内置的。

用法:
    python train_seq.py --data data --epochs 80                  # 孤立词
    python train_seq.py --backbone tcn_bigru --distill           # 视觉教师蒸馏
    python train_seq.py --ctc                                    # 句子级(真实句子 + 合成句子)
    python train_seq.py --ctc --no-synth                          # 只用真实句子录制

环境:本机全局 Python 是共享科研环境,务必先建独立 venv:
    python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ctc_decode import greedy_decode, word_error_rate
from label_merge import merge_label_list, merged_class_table
from load_dataset import (
    FUSED_FRAME_DIM,
    IDLE_LABEL,
    SEQ_LEN,
    TACTILE_FRAME_DIM,
    Sequence,
    build_features,
    build_xy,
    load_dataset,
)


# ===== 网络结构(与 sequenceModel.ts 的 tcnBackbone / isolatedWordHead 一一对应) =====


def tcn_backbone(x, c1: int, c2: int, backbone: str, prefix: str):
    """
    卷积 + 池化堆叠,输出 (T', C)。时间塌缩仍是 head 的职责,池化只用来扩感受野。

    ⚠ **不要改回 dilation_rate > 1。** keras 这边空洞是能训的,但
    sequenceModel.ts 那边不行:tfjs 的 conv2D 前向支持空洞、反向直接抛
    "dilation rates greater than 1 are not yet supported in gradients"。
    两边结构必须一一对应(否则同一份数据两边训出来的结果没法比、导出的权重也对不上),
    所以浏览器能不能训是共同约束。

    感受野(按原始帧数算,T=32):
      b1 k=5 → 5 (T=32);pool/2 → 6 (T=16);b2 k=5 → 14;pool/2 → 16 (T=8);b3 k=5 → 32
    32 帧盖满整段,与原空洞版(30 帧)等效;代价是骨干输出分辨率 T/2 → T/4。
    """

    def block(t, filters, kernel_size, name):
        y = layers.Conv1D(
            filters,
            kernel_size,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
            name=f"{prefix}_{name}_conv",
        )(t)
        y = layers.BatchNormalization(name=f"{prefix}_{name}_bn")(y)
        return layers.Activation("relu", name=f"{prefix}_{name}_relu")(y)

    x = block(x, c1, 5, "b1")
    x = layers.MaxPooling1D(2, name=f"{prefix}_pool1")(x)
    x = block(x, c1, 5, "b2")
    x = layers.MaxPooling1D(2, name=f"{prefix}_pool2")(x)
    x = block(x, c2, 5, "b3")

    if backbone == "tcn_bigru":
        x = layers.Bidirectional(
            layers.GRU(max(8, c2 // 2), return_sequences=True),
            merge_mode="concat",
            name=f"{prefix}_bigru",
        )(x)
    return x


def isolated_word_head(x, hidden: int, num_classes: int, dropout: float, prefix: str):
    y = layers.GlobalAveragePooling1D(name=f"{prefix}_gap")(x)
    y = layers.Dense(hidden, activation="relu", kernel_initializer="he_normal", name=f"{prefix}_fc")(y)
    y = layers.Dropout(dropout, name=f"{prefix}_drop")(y)
    return layers.Dense(num_classes, activation="softmax", name=f"{prefix}_out")(y)


def frame_wise_head(x, num_classes: int, prefix: str):
    """句子级 head:逐帧输出 num_classes+1 维(末位是 CTC blank),不做时间塌缩。"""
    return layers.Dense(num_classes + 1, activation="softmax", name=f"{prefix}_out")(x)


def build_model(
    num_classes: int,
    seq_len: int,
    frame_dim: int,
    backbone: str,
    channels: tuple[int, int],
    hidden: int,
    dropout: float,
    prefix: str,
    ctc: bool = False,
) -> keras.Model:
    inp = keras.Input(shape=(seq_len, frame_dim), name=f"{prefix}_in")
    feat = tcn_backbone(inp, channels[0], channels[1], backbone, prefix)
    out = (
        frame_wise_head(feat, num_classes, prefix)
        if ctc
        else isolated_word_head(feat, hidden, num_classes, dropout, prefix)
    )
    return keras.Model(inp, out, name=prefix)


def build_teacher(num_classes, seq_len, backbone, ctc=False):
    return build_model(num_classes, seq_len, FUSED_FRAME_DIM, backbone, (128, 256), 128, 0.3, "teacher", ctc)


def build_student(num_classes, seq_len, backbone, ctc=False):
    return build_model(num_classes, seq_len, TACTILE_FRAME_DIM, backbone, (64, 128), 64, 0.2, "student", ctc)


# ===== 数据增强(与 sequenceFeatures.ts 的 makeAugmentedGrid 同策略) =====


def augmented_grid(seq_len: int, time_warp: float, crop_min: float, rng: np.random.Generator) -> np.ndarray:
    ratio = 1.0 if crop_min >= 1 else crop_min + (1 - crop_min) * rng.random()
    start = (1 - ratio) * rng.random()
    K = 3
    knots = [0.0]
    for k in range(1, K + 1):
        knots.append(k / (K + 1) + time_warp * (rng.random() * 2 - 1) / (K + 1))
    knots.append(1.0)
    knots = np.sort(np.array(knots))  # 单调:时间不能倒流

    u = np.linspace(0, 1, seq_len)
    seg = np.minimum((u * (K + 1)).astype(int), K)
    local = u * (K + 1) - seg
    warped = np.clip(knots[seg] + (knots[seg + 1] - knots[seg]) * local, 0, 1)
    return (start + ratio * warped).astype(np.float32)


def mount_offset_quat(max_deg: float, rng: np.random.Generator) -> np.ndarray:
    """随机小角度四元数,模拟手套每次戴上去的角度都不完全一样。"""
    axis = rng.normal(size=3)
    axis /= max(np.linalg.norm(axis), 1e-8)
    half = np.deg2rad(max_deg * (rng.random() * 2 - 1)) / 2
    return np.concatenate([[np.cos(half)], axis * np.sin(half)]).astype(np.float32)


def augment_batch(
    seqs: list[Sequence],
    labels: list[str],
    seq_len: int,
    include_vision: bool,
    copies: int,
    rng: np.random.Generator,
    time_warp: float = 0.2,
    crop_min: float = 0.85,
    amp_scale: float = 0.1,
    noise_std: float = 0.01,
    mount_deg: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    原始 + copies 份增强副本。副本 0 恒为未增强,保证原始分布始终在训练集里。

    时间扭曲与裁剪走 grid(重采样层),幅度/噪声走特征层,佩戴错位走四元数层 ——
    三者作用的层次不同,不能合并。
    """
    dim = FUSED_FRAME_DIM if include_vision else TACTILE_FRAME_DIM
    n = len(seqs) * (copies + 1)
    x = np.zeros((n, seq_len, dim), np.float32)
    y = np.zeros((n, len(labels)), np.float32)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    row = 0
    for s in seqs:
        for c in range(copies + 1):
            if c == 0:
                feat = build_features(s, seq_len, include_vision)
            else:
                grid = augmented_grid(seq_len, time_warp, crop_min, rng)
                feat = build_features(
                    s, seq_len, include_vision, grid=grid,
                    quat_offset=mount_offset_quat(mount_deg, rng),
                )
                # 幅度缩放(佩戴松紧,规格书标称 ±8%)+ 高斯噪声,只作用在触觉块上;
                # 关键点是几何量,乘一个增益等于把手凭空放大,不能一起缩放
                feat[:, :TACTILE_FRAME_DIM] *= 1 + (rng.random() * 2 - 1) * amp_scale
                feat[:, :TACTILE_FRAME_DIM] += rng.normal(
                    0, noise_std, (seq_len, TACTILE_FRAME_DIM)
                ).astype(np.float32)
            x[row] = feat
            if s.primary_label in label_to_idx:
                y[row, label_to_idx[s.primary_label]] = 1.0
            row += 1
    return x, y


# ===== 知识蒸馏 =====


def soft_labels(teacher: keras.Model, x_fused: np.ndarray, temperature: float) -> np.ndarray:
    """
    log → 除以 T → softmax 是**精确**的 logits 温度缩放,不是近似:
    softmax 输出满足 log(pᵢ) = zᵢ − logsumexp(z),除以 T 后再 softmax,
    那个 −logsumexp(z)/T 与 i 无关,在 softmax 里被完全约掉,恰好等于 softmax(z/T)。
    别把它"修"成别的写法。
    """
    p = teacher.predict(x_fused, verbose=0)
    z = np.log(np.maximum(p, 1e-12)) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ===== 划分 =====


def split_by_sequence(
    seqs: list[Sequence], val_split: float, rng: np.random.Generator
) -> tuple[list[Sequence], list[Sequence]]:
    """
    **按序列划分,不是按增强后的行划分**。

    如果先增强再让 Keras 的 validation_split 切最后 20%,同一条录制的多份增强副本
    会同时出现在训练集和验证集里 —— 那是数据泄漏,验证准确率会虚高到没有参考价值
    (增强副本之间只差一点噪声和时间缩放,等于在考已经背过的题)。

    按标签分层,保证每个类都在验证集里有样本;类内样本 <2 条时全部留给训练。
    """
    by_label: dict[str, list[int]] = {}
    for i, s in enumerate(seqs):
        by_label.setdefault(s.primary_label, []).append(i)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for _, idxs in sorted(by_label.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_split))
        if len(idxs) >= 2:
            n_val = max(1, min(n_val, len(idxs) - 1))
        else:
            n_val = 0
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]

    return [seqs[i] for i in train_idx], [seqs[i] for i in val_idx]


# ===== 句子级 CTC =====

# 句子模型的时间长度。**不能沿用孤立词的 32。**
# tcn_backbone 池化两次(见上面),输出长度是 T/4:T=32 只剩 8 个输出帧,而 CTC 要求
# 输出帧数 ≥ 标签数(相邻重复类还要 ≥2L−1),8 帧连 4 个词都编不稳。一句 6 词约 9s,
# 取 128 → 32 个输出帧 ≈ 5 帧/词。
# 想要更高分辨率的正确做法是加长 T,**不是**去掉池化 —— 池化结构必须和
# sequenceModel.ts 一一对应(tfjs 反向不支持空洞卷积,这个结构就是为此妥协出来的)。
SENT_SEQ_LEN = 128


def ctc_loss_fn(labels_dense, y_pred, input_lengths, label_lengths):
    """
    tf.nn.ctc_loss 的薄封装。y_pred 是逐帧 **softmax** (B, T', C+1),blank 在末位。

    为什么传 log(softmax) 而不是原始 logits:head 那层已经带了 softmax 激活
    (frame_wise_head),拿不到 logits。而 tf.nn.ctc_loss 内部会再做一次 log_softmax ——
    对 log(p) 再 log_softmax 是**恒等**的:log_softmax(log p)ᵢ = log pᵢ − logsumexp(log p)
    = log pᵢ − log(Σp) = log pᵢ,因为 softmax 输出恰好 Σp = 1。所以这么写在数值上
    就是正确的 CTC,别把它"修"成别的形式。

    blank_index=-1 = 末位,与 frame_wise_head 的 num_classes+1 布局一致。
    **浏览器端 ctcDecode 也必须把 blank 当末位** —— 这个约定错一处,整句全错。
    """
    return tf.nn.ctc_loss(
        labels=labels_dense,
        logits=tf.math.log(tf.maximum(y_pred, 1e-12)),
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=-1,
    )


def build_ctc_xy(
    seqs: list[Sequence],
    classes: list[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    句子序列 → X (N,T,294) + 补齐的标签 (N,Lmax) int32 + 真实标签长度 (N,)。

    标签用 merge_label 重映射(我/你/他 → merged_pron_sg 等,理由见 label_merge.py)。
    padding 位填 0 —— tf.nn.ctc_loss 只看 label_length,超出的位置根本不读,所以
    填什么都行;填 0 而不是 -1 是因为 dense labels 要求非负。

    **句子样本不做起手段裁剪。** 词间停顿正是 CTC 用 blank 去建模的东西,裁掉就
    等于把要学的东西删了。build_features 本身不裁剪(裁剪在浏览器侧的 sequenceTrim),
    这里只是把这件事写下来,免得以后有人"顺手"加上。
    """
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    targets: list[list[int]] = []
    rows: list[np.ndarray] = []
    unknown: set[str] = set()
    for s in seqs:
        mapped = merge_label_list(s.label_sequence)
        # _idle 段(如果有人在句子里标了停顿)是有意丢掉的:它不是词,不进 CTC 目标。
        # 其他丢掉的都是**异常**,必须报出来 —— 静默少一个词会让参考序列和实际手势
        # 对不上,模型学到的对齐从此是错的,而 WER 看着只是"稍微高一点"
        unknown |= {c for c in mapped if c not in cls_to_idx and c != IDLE_LABEL}
        ids = [cls_to_idx[c] for c in mapped if c in cls_to_idx]
        if not ids:
            continue
        targets.append(ids)
        rows.append(build_features(s, seq_len, include_vision=False))

    if unknown:
        print(f"⚠️  句子里出现了类别表外的标签,已从目标序列中丢弃:{sorted(unknown)}。"
              f"参考序列因此与实际手势对不上,WER 不可信。")
    if not rows:
        raise SystemExit("句子样本的标签一个都不在类别表里 —— 检查 label_merge 与词表")
    lmax = max(len(t) for t in targets)
    # CTC 的硬性前提:输出帧数 ≥ 标签长度(相邻同类还要 ≥2L−1)。不满足时
    # tf.nn.ctc_loss 返回 inf 而**不报错**,表现为"loss 一直是 inf,像是没学"
    need = max(2 * len(t) - 1 for t in targets)
    if seq_len // 4 < need:
        raise SystemExit(
            f"输出帧数 {seq_len // 4} 不够编码最长的标签序列(需要 {need})。"
            f"把 --sent-seq-len 提到 {need * 4} 以上,不要去动骨干的池化。"
        )
    y = np.zeros((len(targets), lmax), np.int32)
    ylen = np.zeros(len(targets), np.int32)
    for i, t in enumerate(targets):
        y[i, : len(t)] = t
        ylen[i] = len(t)
    return np.stack(rows), y, ylen


def transfer_backbone(dst: keras.Model, src_path: Path) -> int:
    """
    用孤立词学生的权重初始化句子模型的骨干(按层名 + 形状匹配)。

    **这一步决定收不收敛,不是可选优化。** CTC 的梯度信号比分类稀疏得多(它要
    同时学"是什么词"和"在哪一段"),从随机初始化起跑,几百条合成句子基本训不动。
    骨干层(conv/bn)与孤立词模型同名同形,直接搬;只有 `*_out` 那层维度不同
    (C vs C+1),跳过。

    卷积和 BN 都与输入长度无关,所以 T=32 训出来的权重能直接用在 T=128 上。
    """
    if not src_path.exists():
        print(f"⚠️  找不到 {src_path},骨干从随机初始化起跑 —— 合成句子这么少大概率训不动。"
              f"先跑一次孤立词训练(不带 --ctc)。")
        return 0
    src = keras.models.load_model(src_path, compile=False)
    by_name = {l.name: l for l in src.layers}
    moved = 0
    for layer in dst.layers:
        s = by_name.get(layer.name)
        if s is None or not layer.weights:
            continue
        if [tuple(w.shape) for w in s.weights] != [tuple(w.shape) for w in layer.weights]:
            continue
        layer.set_weights(s.get_weights())
        moved += 1
    print(f"  骨干迁移: {moved} 层来自 {src_path.name}")
    return moved


def train_ctc(
    sentence_seqs: list[Sequence],
    word_seqs: list[Sequence],
    labels: list[str],
    args,
    rng,
) -> None:
    """
    句子级 CTC 训练。

    数据来源两条,可叠加:
      - 真实句子录制(多 segment 样本)。有协同发音,是最终能不能用的决定因素。
      - 合成句子(孤立词首尾相接,见 synth_sentences.py)。只验证管道。
    没有真实句子时也能跑,但报出来的 WER 只说明"管道通了"。
    """
    from synth_sentences import synthesize_sentences

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = args.sent_seq_len

    # 类别表**不依赖**当前有哪些句型:从数据集全部标签推,合并后去掉 _idle 再排序。
    # 依赖句型的话,今天多录一句就会让整张表移位,而表的下标就是 softmax 下标 ——
    # 旧模型配新表 = 每个词都翻译成另一个词。_idle 不是词,不进 CTC 目标
    classes = [c for c in merged_class_table(labels) if c != IDLE_LABEL]
    blank = len(classes)
    print(f"=== 句子级 CTC:{len(classes)} 类 + blank(下标 {blank}),T={seq_len} → "
          f"{seq_len // 4} 输出帧 ===")

    real_tr, real_va = split_by_sequence(sentence_seqs, args.val_split, rng) if sentence_seqs else ([], [])
    if sentence_seqs:
        print(f"真实句子 {len(sentence_seqs)} 条 → 训练 {len(real_tr)} / 验证 {len(real_va)}")

    syn_tr: list[Sequence] = []
    syn_va: list[Sequence] = []
    if args.synth and word_seqs:
        # **先划分录制池,再分别合成。** 反过来(先合成再划分句子)会让同一条孤立词
        # 录制既进训练句又进验证句 —— 那是泄漏,验证 WER 虚低到没有参考价值
        w_tr, w_va = split_by_sequence(word_seqs, args.val_split, rng)
        print(f"合成训练句(录制池 {len(w_tr)} 条):")
        syn_tr, used = synthesize_sentences(w_tr, args.synth_per_template, rng)
        print(f"合成验证句(录制池 {len(w_va)} 条,与训练池不重叠):")
        syn_va, _ = synthesize_sentences(
            w_va, max(1, args.synth_per_template // 4), rng, templates=used
        )

    train_seqs = real_tr + syn_tr
    val_seqs = real_va + syn_va
    if len(train_seqs) < 10:
        raise SystemExit(
            "句子样本太少(合成也没凑够)。要么去 /collect-seq 录连续手语句子,"
            "要么确认孤立词数据够 synth_sentences.py 的句型表用。"
        )
    print(f"合计: 训练 {len(train_seqs)} 条 / 验证 {len(val_seqs)} 条")

    xt, yt, ylt = build_ctc_xy(train_seqs, classes, seq_len)
    print(f"特征 {xt.shape},标签最长 {yt.shape[1]} 词")
    xv = yv = ylv = None
    if val_seqs:
        xv, yv, ylv = build_ctc_xy(val_seqs, classes, seq_len)
    else:
        print("⚠️  验证集为空,下面的 WER 不可信")

    model = build_student(len(classes), seq_len, args.backbone, ctc=True)
    transfer_backbone(model, out_dir / "student.keras")
    opt = keras.optimizers.Adam(args.lr * 0.5)

    # 全 batch 同一个 input_length:X 是定长 seq_len 重采样出来的(短句被拉长、长句被
    # 压缩),所以每条的有效输出帧数都是 seq_len//4。这是重采样带来的便利,不是近似 ——
    # 真要做变长输入才需要逐条算
    frames = seq_len // 4

    @tf.function
    def train_step(bx, by, byl):
        with tf.GradientTape() as tape:
            p = model(bx, training=True)
            il = tf.fill([tf.shape(bx)[0]], frames)
            loss = tf.reduce_mean(ctc_loss_fn(by, p, il, byl))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    def eval_wer(x, y, yl) -> tuple[float, list[tuple[list[int], list[int]]]]:
        probs = model.predict(x, batch_size=args.batch, verbose=0)
        hyps = [greedy_decode(probs[i], blank) for i in range(len(x))]
        refs = [list(y[i, : yl[i]]) for i in range(len(x))]
        return word_error_rate(refs, hyps), list(zip(refs, hyps))

    n = len(xt)
    best_wer = 1e9
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(n)
        tot = 0.0
        for b in range(0, n, args.batch):
            idx = order[b : b + args.batch]
            tot += float(train_step(xt[idx], yt[idx], ylt[idx])) * len(idx)
        msg = f"epoch {epoch:>3}/{args.epochs}  loss {tot/n:.4f}"
        if xv is not None:
            wer, _ = eval_wer(xv, yv, ylv)
            msg += f"  val WER {wer*100:.1f}%"
            if wer < best_wer:
                best_wer = wer
                model.save(out_dir / "sentence_student.keras")
                msg += "  ← 已保存"
        print(msg)

    if xv is None:
        model.save(out_dir / "sentence_student.keras")
        best_wer = None  # 不能写 nan:json.dumps 会输出裸 NaN,浏览器 JSON.parse 直接抛

    # 抽几条看解码长什么样。WER 是个汇总数字,看不出"是漏词还是插词" ——
    # 漏词多半是输出帧不够(加长 T),插词多半是词间停顿被学成了词
    if xv is not None:
        # **必须先把最佳权重读回来。** 训练结束时内存里是最后一个 epoch 的权重,
        # 而盘上是 WER 最低的那个 checkpoint。不重载的话打印的抽样来自最后一个
        # epoch,而 meta 里的 valWer 是最佳 epoch 的 —— 两个不同的模型并排报出来,
        # 看着像"WER 5% 但解码错得离谱"
        model = keras.models.load_model(out_dir / "sentence_student.keras", compile=False)
        _, pairs = eval_wer(xv, yv, ylv)
        # 错的排前面。全对的抽样看不出任何东西,而错例直接告诉你是漏词还是插词
        pairs.sort(key=lambda p: p[0] == p[1])
        n_bad = sum(1 for r, h in pairs if r != h)
        print(f"\n验证集抽样(参考 → 解码),{n_bad}/{len(pairs)} 条整句不完全一致,错的排在前面:")
        for ref, hyp in pairs[: min(6, len(pairs))]:
            r = " ".join(classes[i] for i in ref)
            h = " ".join(classes[i] for i in hyp) or "(空)"
            print(f"  {r}\n  → {h}")

    meta = {
        "labels": classes,
        "seqLen": seq_len,
        "backbone": args.backbone,
        "frameDim": TACTILE_FRAME_DIM,
        "modelType": "seq_sentence",
        "ctc": True,
        # 浏览器端解码必须读这个,不要在 JS 里另算一遍 len(labels)
        "blankIndex": blank,
        "outputFrames": frames,
        # 没有验证集时是 null,不是 0 —— 0 会被读成"完美",而真相是"没量过"
        "valWer": None if best_wer is None else float(best_wer),
        "numTrain": len(train_seqs),
        "numVal": len(val_seqs),
        "numReal": len(sentence_seqs),
        "numSynth": len(syn_tr) + len(syn_va),
        # 合成占比高的时候这个 WER 不能当真实表现看,写进 meta 免得以后拿它当结论
        "synthOnly": len(sentence_seqs) == 0,
    }
    (out_dir / "sentence_student_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n已保存 {out_dir}/sentence_student.keras + _meta.json")
    if meta["synthOnly"]:
        print("⚠️  全部是合成句子:这个 WER 只说明管道是通的,**不能**预测真实连续手语"
              "的表现(合成数据里没有协同发音)。要真用得先录真实句子。")


# ===== 主流程 =====


def main():
    # Windows 控制台默认 GBK,编不出 ⚠️(U+26A0),print 会直接抛 UnicodeEncodeError ——
    # 训练跑到一半崩在一句警告上。errors="replace" 保证最坏情况只是显示成问号
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="时序手语模型训练(tf.keras)")
    ap.add_argument("--data", default="data", help="dataset.bin/json 所在目录")
    ap.add_argument("--out", default="out", help="模型输出目录")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--backbone", choices=["tcn", "tcn_bigru"], default="tcn")
    ap.add_argument("--augment-copies", type=int, default=2)
    ap.add_argument("--time-warp", type=float, default=0.2, help="时间扭曲强度 0~0.5")
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--distill", action="store_true", help="先训视觉教师再蒸馏到触觉学生")
    ap.add_argument("--temp", type=float, default=3.0, help="蒸馏温度")
    ap.add_argument("--alpha", type=float, default=0.5, help="soft/hard 标签混合比")
    ap.add_argument("--recorded-only", action="store_true", help="排除合成迁移的样本")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ctc", action="store_true", help="句子级连续手语(CTC)")
    ap.add_argument("--sent-seq-len", type=int, default=SENT_SEQ_LEN,
                    help=f"句子模型的时间长度,默认 {SENT_SEQ_LEN}(→ T/4 个输出帧)")
    ap.add_argument("--no-synth", dest="synth", action="store_false", default=True,
                    help="只用真实句子录制训练,不合成(默认会合成)")
    ap.add_argument("--synth-per-template", type=int, default=24,
                    help="每个句型合成多少条")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    all_seqs, labels = load_dataset(args.data, recorded_only=args.recorded_only)
    # 两条链路各要一半:孤立词训练只能吃单 segment,CTC 只能吃多 segment。
    # 判据是 Sequence.is_sentence,与 TS 侧 isSentenceSample 同一条
    word_seqs = [s for s in all_seqs if not s.is_sentence]
    sentence_seqs = [s for s in all_seqs if s.is_sentence]
    print(
        f"载入 {len(all_seqs)} 条序列,{len(labels)} 类"
        f"(孤立词 {len(word_seqs)} / 句子 {len(sentence_seqs)})"
    )
    if IDLE_LABEL not in labels:
        print(f"⚠️  没有 {IDLE_LABEL} 类。滑窗推理必须有空闲伪类,否则线上会持续乱吐词。")

    if args.ctc:
        train_ctc(sentence_seqs, word_seqs, labels, args, rng)
        return

    # 以下是孤立词分支。句子样本在这里一条都不能进 —— 它的 primary_label 是第一个词,
    # 混进去等于往那一类里掺另外几个词的特征
    seqs = word_seqs
    if sentence_seqs:
        print(f"排除 {len(sentence_seqs)} 条句子样本(孤立词训练不吃多 segment)")
    if len(seqs) < 10:
        raise SystemExit("孤立词样本太少,先去 /collect-seq 采集")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_seqs, val_seqs = split_by_sequence(seqs, args.val_split, rng)
    print(f"划分: 训练 {len(train_seqs)} 条 / 验证 {len(val_seqs)} 条(按序列,不是按增强副本)")
    if not val_seqs:
        print("⚠️  验证集为空,下面的准确率不可信")

    vision_ratio = sum(1 for s in seqs if s.has_vision) / len(seqs)
    use_distill = args.distill and vision_ratio >= 0.8
    if args.distill and not use_distill:
        print(f"⚠️  仅 {vision_ratio*100:.0f}% 样本含视觉(<80%),跳过教师与蒸馏")

    # 增强用独立 rng,且教师/学生各自从同一个种子起跑:两次调用的抽样次数逐次相同,
    # 所以第 k 行在两边对应"同一条录制的同一次增强",蒸馏标签才不会错配
    aug_seed = args.seed + 1

    soft = None
    if use_distill:
        print("=== 阶段 1:视觉+触觉融合教师 ===")
        xf, yf = augment_batch(
            train_seqs, labels, args.seq_len, True, args.augment_copies,
            np.random.default_rng(aug_seed), time_warp=args.time_warp,
        )
        xfv, yfv = build_xy(val_seqs, labels, args.seq_len, True)  # 验证集不增强
        teacher = build_teacher(len(labels), args.seq_len, args.backbone)
        teacher.compile(
            optimizer=keras.optimizers.Adam(args.lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        teacher.fit(
            xf, yf,
            epochs=max(1, int(args.epochs * 0.6)),
            batch_size=args.batch,
            validation_data=(xfv, yfv) if len(val_seqs) else None,
            shuffle=True,
        )
        teacher.save(out_dir / "teacher.keras")
        soft = soft_labels(teacher, xf, args.temp)
        del xf, xfv

    print("=== 阶段 2:纯触觉学生(部署用) ===")
    xt, yt = augment_batch(
        train_seqs, labels, args.seq_len, False, args.augment_copies,
        np.random.default_rng(aug_seed), time_warp=0.2,
    )
    xv, yv = build_xy(val_seqs, labels, args.seq_len, False)
    target = yt
    if soft is not None:
        if soft.shape != yt.shape:
            raise RuntimeError("教师/学生样本数不一致,蒸馏标签会错配")
        # 混合标签 α·soft+(1−α)·hard 等价于 α·CE(soft)+(1−α)·CE(hard):交叉熵对目标分布是线性的
        target = args.alpha * soft + (1 - args.alpha) * yt

    student = build_student(len(labels), args.seq_len, args.backbone)
    student.compile(
        optimizer=keras.optimizers.Adam(args.lr * 0.5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    hist = student.fit(
        xt, target,
        epochs=args.epochs if soft is None else max(30, args.epochs - int(args.epochs * 0.6)),
        batch_size=args.batch,
        validation_data=(xv, yv) if len(val_seqs) else None,
        shuffle=True,
    )
    student.save(out_dir / "student.keras")

    val_acc = 0.0
    if len(val_seqs):
        # 硬标签下的真实准确率(蒸馏时 fit 报的 loss 是对混合标签算的,不能直接看)
        _, val_acc = student.evaluate(xv, yv, verbose=0)
        print(f"学生验证准确率(未增强、按序列划分): {val_acc*100:.1f}%")

    meta = {
        "labels": labels,
        "seqLen": args.seq_len,
        "backbone": args.backbone,
        "frameDim": TACTILE_FRAME_DIM,
        "modelType": "seq_tactile",
        "distilled": soft is not None,
        "valAccuracy": float(val_acc),
        "numSequences": len(seqs),
        "numTrain": len(train_seqs),
        "numVal": len(val_seqs),
        "epochsRun": len(hist.history.get("loss", [])),
    }
    (out_dir / "student_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存到 {out_dir}/。转回浏览器:python export_to_tfjs.py")


if __name__ == "__main__":
    main()
