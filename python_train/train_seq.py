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
    python train_seq.py --ctc                                    # 句子级(需要多 segment 数据)

环境:本机全局 Python 是共享科研环境,务必先建独立 venv:
    python -m venv .venv && .venv\\Scripts\\activate && pip install -r requirements.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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


# ===== 句子级 CTC(骨架) =====


def ctc_loss_fn(y_true_sparse, y_pred, input_lengths, label_lengths):
    """
    tf.nn.ctc_loss 的薄封装。y_pred 是逐帧 softmax (B, T', C+1),blank 在末位。

    未完成:句子级训练需要的是多 segment 数据(一条序列里有多个词),当前采集页
    只产出单 segment 的孤立词。等真的开始录连续手语句子后,这里要补:
      1. 变长 batch 的 padding 与 input_lengths 计算
      2. 解码(greedy / beam)与 WER 评估
      3. 与孤立词模型的联合初始化(用孤立词权重初始化 backbone 收敛快得多)
    数据 schema(segments 词边界列表)已经准备好了,不需要重新采集。
    """
    return tf.nn.ctc_loss(
        labels=y_true_sparse,
        logits=tf.math.log(tf.maximum(y_pred, 1e-12)),
        label_length=label_lengths,
        logit_length=input_lengths,
        logits_time_major=False,
        blank_index=-1,
    )


# ===== 主流程 =====


def main():
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
    ap.add_argument("--ctc", action="store_true", help="句子级 CTC(骨架,见 ctc_loss_fn)")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    seqs, labels = load_dataset(args.data, recorded_only=args.recorded_only)
    print(f"载入 {len(seqs)} 条序列,{len(labels)} 类")
    if len(seqs) < 10:
        raise SystemExit("样本太少,先去 /collect-seq 采集")
    if IDLE_LABEL not in labels:
        print(f"⚠️  没有 {IDLE_LABEL} 类。滑窗推理必须有空闲伪类,否则线上会持续乱吐词。")

    if args.ctc:
        multi = sum(1 for s in seqs if len(s.segments) > 1)
        raise SystemExit(
            f"--ctc 尚未实现完整训练流程(见 ctc_loss_fn 的说明)。"
            f"当前数据集里多 segment 的序列有 {multi} 条,"
            f"{'可以开始接' if multi else '还全是孤立词,先录连续手语句子'}。"
        )

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
