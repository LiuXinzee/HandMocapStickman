"""
读 dataset.bin + dataset.json,产出与浏览器端逐位一致的模型输入。

"逐位一致"是硬要求:Python 训完的模型要转回 tfjs 在浏览器里跑,
两边的特征构建只要有一处不同(比如四元数没相对化、加速度没除以 16),
线下指标再好,线上也是随机输出。所以这个文件的每个函数都对应
client/src/lib/sequenceFeatures.ts 里的一个同名函数,改一边就要改另一边。

格式说明见 export_format.md。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# 与 sequenceFeatures.ts 的常量一一对应
SEQ_LEN = 32
SENSOR_N = 137
IMU_N = 10
LANDMARK_N = 63
HAND_TACTILE_DIM = SENSOR_N + 4 + 3 + 3  # 147
TACTILE_FRAME_DIM = HAND_TACTILE_DIM * 2  # 294
FUSED_FRAME_DIM = TACTILE_FRAME_DIM + LANDMARK_N * 2  # 420
ACC_SCALE = 16.0

IDLE_LABEL = "_idle"

# 读得进来的导出版本。1.1 起每条多了 trimSpan(见 export_format.md),二进制布局没变
SUPPORTED_VERSIONS = ("seq-1.0", "seq-1.1")

_DTYPE = {"uint8": np.uint8, "float32": np.float32}


@dataclass
class Sequence:
    """一条变长序列。缺失的模态是 None,不是零数组。"""

    segments: list[dict]
    primary_label: str
    frame_count: int
    duration_ms: float
    source_fps: float
    origin: str
    timestamps: np.ndarray
    left_sensor: np.ndarray | None
    right_sensor: np.ndarray | None
    left_imu: np.ndarray | None
    right_imu: np.ndarray | None
    left_landmarks: np.ndarray | None
    right_landmarks: np.ndarray | None
    # ↓ seq-1.1。浏览器端 sequenceTrim.detectSignSpan 判出的"动作真正开始/结束"区间
    trim_span: tuple[int, int] | None = None
    trim_applied: bool = False
    trim_reason: str = "not_exported"
    #: 第三层(触觉静止段)有没有跑过。False = 收尾静止还在这条数据里
    trim_tactile_ran: bool = False

    @property
    def has_vision(self) -> bool:
        return self.left_landmarks is not None or self.right_landmarks is not None

    @property
    def label_sequence(self) -> list[str]:
        """句子级 CTC 的目标标签序列。孤立词就是长度 1。"""
        return [s["label"] for s in self.segments]

    @property
    def is_sentence(self) -> bool:
        """
        是不是句子级录制(一条里连着打了多个词)。与 TS 侧 `isSentenceSample` 同一判据。

        孤立词训练**必须**用它把句子挡掉:句子的 primary_label 是它的第一个词,
        不挡的话一条「我 名字 王」会作为一条 `我` 进 softmax,等于往那一类里掺噪声,
        而条数和覆盖率看起来全是正常的。CTC 训练反过来只要它为真的样本。
        """
        return len(self.segments) > 1

    def trimmed_view(self, head_only: bool = False) -> "Sequence":
        """
        按 `trim_span` 切出"只剩动作"的那一段,返回一条新的 Sequence(不改原对象)。

        `head_only=True` 时**只切头,尾巴一帧不动**(终点强制为 frame_count)。
        逐词模型要的是这个,理由是推理口径:
          - 句子推理(`sentenceEnvelope`)会主动掐掉尾部 800ms 静止 → 训练必须跟着裁尾
          - 逐词推理只是取最近 2000ms 滑窗,**不掐尾** → 裁尾会让训练离推理更远
        实测这一刀不小:441 条要裁的词录制里,尾部中位切掉 700ms(占录制 25%),
        比头部的 520ms 还多,244/441 条尾部切得比头部多。

        用途是句子合成:把孤立词录制拼成句子之前,先去掉每个词的抬手 transport 和
        收尾静止。真实连续手语里这两段不存在,留着就是给模型一个人造的分词标记 ——
        与 synth_sentences 模块 docstring 里那个 127 倍尖峰同一类错误。

        **不裁的条件是 `trim_applied` 为假**,那时原样返回(同一个对象)。判据判不出来
        就不裁,是浏览器端那三条守则之一,这里照搬。

        时间戳重新以 0 为起点:build_features 按归一化位置 [0,1] 采样,起点不归零的话
        第一帧会落在切掉的那段里。`segments` 的帧号跟着平移并夹到新区间内。
        """
        if not self.trim_applied or self.trim_span is None:
            return self
        a, b = self.trim_span
        a = max(0, min(a, self.frame_count))
        b = self.frame_count if head_only else max(a + 1, min(b, self.frame_count))
        b = max(a + 1, b)

        def cut(arr: np.ndarray | None) -> np.ndarray | None:
            return None if arr is None else arr[a:b].copy()

        ts = self.timestamps[a:b].astype(np.float32) - float(self.timestamps[a])
        segs = [
            {
                **s,
                "startFrame": max(0, min(int(s["startFrame"]) - a, b - a)),
                "endFrame": max(0, min(int(s["endFrame"]) - a, b - a)),
            }
            for s in self.segments
        ]
        return Sequence(
            segments=segs,
            primary_label=self.primary_label,
            frame_count=b - a,
            duration_ms=float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
            source_fps=self.source_fps,
            origin=self.origin,
            timestamps=ts,
            left_sensor=cut(self.left_sensor),
            right_sensor=cut(self.right_sensor),
            left_imu=cut(self.left_imu),
            right_imu=cut(self.right_imu),
            left_landmarks=cut(self.left_landmarks),
            right_landmarks=cut(self.right_landmarks),
            # 已经裁过了。再调一次 trimmed_view 必须是恒等操作,否则重复切片
            trim_span=(0, b - a),
            trim_applied=False,
            trim_reason=self.trim_reason,
            trim_tactile_ran=self.trim_tactile_ran,
        )


def require_trim_spans(seqs: list[Sequence]) -> None:
    """
    确认这批序列真的带着裁剪区间。**要求裁剪却拿到 seq-1.0 时必须炸在这里。**

    与 `_check_pair` 同一条理由:静默不裁不会报错、形状全合法、训练照跑,
    唯一的症状是 WER 差几个点 —— 而那时候已经没有任何线索指回"数据集是旧版导出的"。

    `trim_tactile_ran` 全为假只是**警告**:视觉那两层的结论是真的,只是收尾静止还在。
    """
    missing = [s for s in seqs if s.trim_reason == "not_exported"]
    if missing:
        raise ValueError(
            f"{len(missing)}/{len(seqs)} 条没有 trimSpan —— 这份 dataset.json 是 seq-1.0 导出的,\n"
            f"里面没有裁剪区间。继续跑的话句子会带着抬手 transport 和收尾静止合成出来,\n"
            f"而推理端(sentenceEnvelope)那两段永远见不到 —— 训练与推理的时间口径就差了一截,\n"
            f"表现是句首持续丢词/插词,且合成 val WER 看不出来。\n"
            f"回浏览器 /train-seq 重新点「导出数据集给 Python」,两个文件都覆盖进 data/。"
        )
    blind = [s for s in seqs if not s.trim_tactile_ran]
    if blind:
        print(
            f"  ⚠ {len(blind)}/{len(seqs)} 条的触觉静止段判据没跑过(导出时没做弯折两点标定)。\n"
            f"    那是唯一会裁尾的一层 —— 这些条的收尾静止还在数据里。"
        )


# ===== 读取 =====


def load_dataset(
    data_dir: str | Path = "data", recorded_only: bool = False
) -> tuple[list[Sequence], list[str]]:
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    blob = np.frombuffer((data_dir / "dataset.bin").read_bytes(), dtype=np.uint8)

    if manifest.get("version") not in SUPPORTED_VERSIONS:
        raise ValueError(
            f"未知的导出版本 {manifest.get('version')!r}(本脚本读 {'/'.join(SUPPORTED_VERSIONS)}),"
            "见 export_format.md"
        )
    for key, expected in (("sensorN", SENSOR_N), ("imuN", IMU_N), ("landmarkN", LANDMARK_N)):
        if manifest[key] != expected:
            raise ValueError(
                f"{key} 不匹配:数据集是 {manifest[key]},本脚本按 {expected} 编译。"
                "改了硬件通道数就要同步改本文件的常量。"
            )

    _check_pair(manifest, blob.size)

    seqs = [s for s in _iter_sequences(manifest, blob)]
    if recorded_only:
        seqs = [s for s in seqs if s.origin == "recorded"]
    return seqs, list(manifest["labels"])


def _check_pair(manifest: dict, blob_size: int) -> None:
    """
    dataset.json 和 dataset.bin 必须是**同一次导出**的。

    这不是防手滑的小检查,是防一种会一路跑通的错误:.json 里存的是每个数组在
    .bin 里的字节偏移,两个文件配错时偏移全都落在错误的位置上 —— 读出来是别的
    样本的传感值,形状却完全合法。训练照跑、准确率照样很高(错标签配错特征,
    模型学的是另一个映射),只有实机推理是全错的,而那时候已经没有线索指回这里。

    实际踩过一次:浏览器导出后只把 .bin 覆盖进 data/,.json 忘了覆盖。
    新 .bin 比旧 .json 声明的长 145,676 字节,越界检查也不会触发(是更长而不是更短)。

    判据是**严格相等**:导出器只写它列出来的那些数组,没有对齐填充,合法的一对
    永远恰好相等。宁可误报一次让人重新导一遍,也不要静默读错。
    """
    declared = 0
    for e in manifest["sequences"]:
        for ref in e["arrays"].values():
            if not ref:  # 缺哪只手就是 null(单手录制),不占字节
                continue
            end = ref["offset"] + ref["length"] * _DTYPE[ref["dtype"]]().itemsize
            declared = max(declared, end)

    if declared != blob_size:
        diff = blob_size - declared
        raise ValueError(
            f"dataset.json 与 dataset.bin 不是同一次导出的:\n"
            f"  .json 声明需要 {declared:,} 字节({manifest['totalSequences']} 条,"
            f"导出于 {manifest.get('exportedAt')})\n"
            f"  .bin  实际  {blob_size:,} 字节(相差 {diff:+,})\n"
            f"两个文件必须一起覆盖 —— 只换一个的话字节偏移会全部错位,"
            f"而训练不会报错、准确率还很高,只有实机推理是全错的。\n"
            f"回浏览器 /train-seq 重新点「导出数据集给 Python」,两个文件都放进 data/。"
        )


def _iter_sequences(manifest: dict, blob: np.ndarray) -> Iterator[Sequence]:
    for e in manifest["sequences"]:
        arrays = e["arrays"]

        def read(key: str, per_frame: int) -> np.ndarray | None:
            ref = arrays.get(key)
            if ref is None:
                return None
            dtype = _DTYPE[ref["dtype"]]
            nbytes = ref["length"] * dtype().itemsize
            raw = blob[ref["offset"] : ref["offset"] + nbytes]
            arr = raw.view(dtype)
            return arr.reshape(-1, per_frame) if per_frame > 1 else arr.copy()

        # seq-1.0 没有这个字段。区分"没导出"和"导出了但判据说不用裁"很重要:
        # 前者要报错(require_trim_spans),后者是一个经过确认的结论
        span = e.get("trimSpan")

        yield Sequence(
            segments=e["segments"],
            primary_label=e["primaryLabel"],
            frame_count=e["frameCount"],
            duration_ms=e["durationMs"],
            source_fps=e["sourceFps"],
            origin=e["origin"],
            timestamps=read("timestamps", 1),
            left_sensor=read("leftSensor", SENSOR_N),
            right_sensor=read("rightSensor", SENSOR_N),
            left_imu=read("leftImu", IMU_N),
            right_imu=read("rightImu", IMU_N),
            left_landmarks=read("leftLandmarks", LANDMARK_N),
            right_landmarks=read("rightLandmarks", LANDMARK_N),
            trim_span=(
                (int(span["startFrame"]), int(span["endFrame"])) if span else None
            ),
            trim_applied=bool(span["applied"]) if span else False,
            trim_reason=str(span["reason"]) if span else "not_exported",
            trim_tactile_ran=bool(span.get("tactileRan")) if span else False,
        )


# ===== 四元数 =====


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(n > 1e-8, q / np.maximum(n, 1e-12), np.array([1.0, 0.0, 0.0, 0.0]))


def quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1
    return out


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def gravity_in_hand_frame(q: np.ndarray) -> np.ndarray:
    """R(q)⁻¹·[0,0,1]。绕重力轴旋转不变,所以不受 IMU 的 yaw 固定偏置影响。"""
    return quat_rotate(quat_conj(q), np.array([0.0, 0.0, 1.0]))


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    # t = 2 * (q_vec × v);  v' = v + w*t + q_vec × t
    tx = 2 * (y * vz - z * vy)
    ty = 2 * (z * vx - x * vz)
    tz = 2 * (x * vy - y * vx)
    return np.stack(
        [
            vx + w * tx + (y * tz - z * ty),
            vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx),
        ],
        axis=-1,
    )


def quat_slerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """逐分量线性插值四元数在大角度时会给出非单位、方向错误的结果,必须 slerp。"""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)  # 取短弧
    dot = np.abs(dot).clip(-1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    t = t[..., None] if t.ndim < a.ndim else t
    near = sin_theta < 1e-6
    lerp = a + (b - a) * t
    slerp = (
        a * np.sin((1 - t) * theta) / np.maximum(sin_theta, 1e-12)
        + b * np.sin(t * theta) / np.maximum(sin_theta, 1e-12)
    )
    return quat_normalize(np.where(near, lerp, slerp))


# ===== 重采样 =====


def resample_block(block: np.ndarray, timestamps: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """按归一化位置 grid∈[0,1] 线性插值到 len(grid) 帧。"""
    i0, i1, w = _sample_index(timestamps, grid)
    return block[i0] + (block[i1] - block[i0]) * w[:, None]


def resample_imu(imu: np.ndarray, timestamps: np.ndarray, grid: np.ndarray) -> np.ndarray:
    i0, i1, w = _sample_index(timestamps, grid)
    out = np.empty((len(grid), IMU_N), dtype=np.float32)
    out[:, :4] = quat_slerp(imu[i0, :4], imu[i1, :4], w)
    out[:, 4:] = imu[i0, 4:] + (imu[i1, 4:] - imu[i0, 4:]) * w[:, None]
    return out


def resample_vision(lm: np.ndarray, timestamps: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """
    不跨 NaN 插值:一个有效帧和一个 NaN 帧线性插值会混出整帧 NaN,白扔一半有效数据。
    两端都有效才插值,否则取有效的那一端。
    """
    i0, i1, w = _sample_index(timestamps, grid)
    v0 = np.isfinite(lm[i0, 0])
    v1 = np.isfinite(lm[i1, 0])
    both = v0 & v1
    out = np.full((len(grid), LANDMARK_N), np.nan, dtype=np.float32)
    out[both] = lm[i0[both]] + (lm[i1[both]] - lm[i0[both]]) * w[both, None]
    only0 = v0 & ~v1
    only1 = ~v0 & v1
    out[only0] = lm[i0[only0]]
    out[only1] = lm[i1[only1]]
    return out


def _sample_index(timestamps: np.ndarray, grid: np.ndarray):
    n = len(timestamps)
    if n <= 1:
        z = np.zeros(len(grid), dtype=np.int64)
        return z, z, np.zeros(len(grid), dtype=np.float32)
    t0, t1 = timestamps[0], timestamps[-1]
    target = t0 + (t1 - t0) * grid
    i1 = np.clip(np.searchsorted(timestamps, target, side="right"), 1, n - 1)
    i0 = i1 - 1
    span = timestamps[i1] - timestamps[i0]
    w = np.where(span > 1e-6, (target - timestamps[i0]) / np.maximum(span, 1e-12), 0.0)
    return i0, i1, np.clip(w, 0.0, 1.0).astype(np.float32)


def fill_vision_gaps(lm: np.ndarray) -> bool:
    """中间空洞线性插值,首尾 hold。全程无视觉时置 0 并返回 False。"""
    valid = np.where(np.isfinite(lm[:, 0]))[0]
    if len(valid) == 0:
        lm[:] = 0.0
        return False
    idx = np.arange(len(lm))
    for c in range(lm.shape[1]):
        lm[:, c] = np.interp(idx, valid, lm[valid, c])
    return True


def normalize_landmark_frames(lm: np.ndarray) -> np.ndarray:
    """手腕(点0)保留绝对坐标承载轨迹;其余 20 点相对手腕平移 + 按手长(0→9)缩放承载手型。"""
    pts = lm.reshape(-1, 21, 3).copy()
    wrist = pts[:, 0:1, :]
    scale = np.linalg.norm(pts[:, 9, :] - pts[:, 0, :], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-3)[:, None, :]
    out = (pts - wrist) / scale
    out[:, 0, :] = pts[:, 0, :]
    return out.reshape(-1, LANDMARK_N)


# ===== 特征构建 =====


def build_features(
    seq: Sequence,
    seq_len: int = SEQ_LEN,
    include_vision: bool = False,
    grid: np.ndarray | None = None,
    quat_offset: np.ndarray | None = None,
) -> np.ndarray:
    """
    一条序列 → (seq_len, 294 或 420)。与浏览器端 buildSequenceFeatures 等价。

    grid 是归一化采样位置 ∈[0,1],默认均匀。训练时传入时间扭曲/裁剪后的非均匀 grid
    即可做序列增强 —— 增强必须发生在重采样这一层,在成品特征上插值会把相对四元数
    的参考帧也一起插坏。

    quat_offset 是佩戴错位增强:**右乘**原始四元数(q → q⊗m),表示"手套相对手掌
    装歪了一点"。必须在这里注入而不是事后改特征块,因为相对四元数和重力投影都是
    从同一个 q 派生的,只改其中一个会得到物理上不自洽的一帧。
    """
    if grid is None:
        grid = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
    else:
        grid = np.asarray(grid, dtype=np.float32)
        seq_len = len(grid)
    ts = seq.timestamps.astype(np.float32)
    dim = FUSED_FRAME_DIM if include_vision else TACTILE_FRAME_DIM
    out = np.zeros((seq_len, dim), dtype=np.float32)

    for hand_idx, (sensor, imu, lm) in enumerate(
        (
            (seq.left_sensor, seq.left_imu, seq.left_landmarks),
            (seq.right_sensor, seq.right_imu, seq.right_landmarks),
        )
    ):
        base = hand_idx * HAND_TACTILE_DIM
        if sensor is not None:
            rs = resample_block(sensor.astype(np.float32), ts, grid)
            out[:, base : base + SENSOR_N] = rs / 255.0
        if imu is not None:
            ri = resample_imu(imu.astype(np.float32), ts, grid)
            q = quat_normalize(ri[:, :4])
            if quat_offset is not None:
                q = quat_normalize(quat_mul(q, quat_normalize(quat_offset)[None, :]))
            q0_inv = quat_conj(q[0:1])
            out[:, base + SENSOR_N : base + SENSOR_N + 4] = quat_mul(
                np.repeat(q0_inv, seq_len, axis=0), q
            )
            out[:, base + SENSOR_N + 4 : base + SENSOR_N + 7] = gravity_in_hand_frame(q)
            out[:, base + SENSOR_N + 7 : base + SENSOR_N + 10] = np.clip(
                ri[:, 4:7] / ACC_SCALE, -1.0, 1.0
            )
        if include_vision and lm is not None:
            rv = resample_vision(lm.astype(np.float32), ts, grid)
            if fill_vision_gaps(rv):
                vbase = TACTILE_FRAME_DIM + hand_idx * LANDMARK_N
                out[:, vbase : vbase + LANDMARK_N] = normalize_landmark_frames(rv)

    return out


def build_xy(
    seqs: list[Sequence],
    labels: list[str],
    seq_len: int = SEQ_LEN,
    include_vision: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """孤立词训练用:X (N, T, D) + one-hot Y (N, C)。"""
    label_to_idx = {l: i for i, l in enumerate(labels)}
    x = np.zeros((len(seqs), seq_len, FUSED_FRAME_DIM if include_vision else TACTILE_FRAME_DIM), np.float32)
    y = np.zeros((len(seqs), len(labels)), np.float32)
    for i, s in enumerate(seqs):
        x[i] = build_features(s, seq_len, include_vision)
        if s.primary_label in label_to_idx:
            y[i, label_to_idx[s.primary_label]] = 1.0
    return x, y


if __name__ == "__main__":
    import argparse
    import sys
    from collections import Counter

    # 与 train_seq.main 同一条理由:Windows 控制台默认 GBK,编不出 ⚠️(U+26A0),
    # print 会抛 UnicodeEncodeError —— 体检崩在一句警告上,而警告本身才是要看的东西
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="检查导出的数据集")
    ap.add_argument("--data", default="data")
    ap.add_argument("--recorded-only", action="store_true")
    args = ap.parse_args()

    seqs, labels = load_dataset(args.data, recorded_only=args.recorded_only)
    print(f"序列数: {len(seqs)}  标签数: {len(labels)}")
    print(f"标签: {labels}")

    origins = Counter(s.origin for s in seqs)
    print(f"来源: {dict(origins)}")
    per_label = Counter(s.primary_label for s in seqs)
    for l in labels:
        print(f"  {l:>12}: {per_label[l]:>4}")
    if IDLE_LABEL not in per_label:
        print(f"\n⚠️  没有 {IDLE_LABEL} 样本。滑窗推理必须有空闲伪类,否则手放松时会持续乱吐词。")

    vision = sum(1 for s in seqs if s.has_vision)
    print(f"含视觉: {vision}/{len(seqs)} ({vision / max(len(seqs), 1) * 100:.0f}%)")
    print(f"平均时长: {np.mean([s.duration_ms for s in seqs]):.0f}ms")

    # 裁剪区间。这是最后一处能看见"到底裁了多少"的地方 —— 下游只会用切好的片段
    reasons = Counter(s.trim_reason for s in seqs)
    cut = [s for s in seqs if s.trim_applied]
    print(f"\n裁剪区间: {dict(reasons)}")
    if cut:
        kept = np.mean([(s.trim_span[1] - s.trim_span[0]) / s.frame_count for s in cut])
        after = np.mean([s.trimmed_view().duration_ms for s in cut])
        print(f"  要裁 {len(cut)} 条,平均留下 {kept*100:.0f}% 帧,裁后平均时长 {after:.0f}ms")

    # trimmed_view 自检。**造一个区间而不是等数据集里有** —— 旧版导出里一条都没有,
    # 挂在 `if cut` 下面的断言在 seq-1.0 上永远不跑,而句子合成已经依赖这段切片了
    if seqs:
        import dataclasses
        src = max(seqs, key=lambda s: s.frame_count)
        a, b = 5, src.frame_count - 3
        probe = dataclasses.replace(src, trim_span=(a, b), trim_applied=True)
        v = probe.trimmed_view()
        assert v.frame_count == b - a, f"切片长度错: {v.frame_count} != {b - a}"
        assert len(v.timestamps) == b - a, "时间戳没跟着切"
        assert float(v.timestamps[0]) == 0.0, "裁后时间戳没归零,build_features 会采到切掉那段"
        assert np.allclose(
            v.timestamps, src.timestamps[a:b] - src.timestamps[a]
        ), "时间戳不是原区间的平移"
        for name in ("left_sensor", "right_sensor", "left_imu", "right_imu",
                     "left_landmarks", "right_landmarks"):
            orig, got = getattr(src, name), getattr(v, name)
            if orig is None:
                assert got is None, f"{name} 本来是缺失,裁完变成了数组"
            else:
                assert got.shape[0] == b - a, f"{name} 帧数没跟着切"
                # equal_nan:视觉数组里的 NaN 是"这帧丢手"的合法取值,不是错误
                same = np.array_equal(got, orig[a:b], equal_nan=got.dtype.kind == "f")
                assert same, f"{name} 切错了位置"
        assert all(0 <= s["startFrame"] <= b - a for s in v.segments), "segment 帧号越界"
        assert v.trimmed_view() is v, "trimmed_view 不幂等 —— 二次调用会重复切片"
        # build_features 必须能吃裁后的序列(帧数比 SEQ_LEN 少也要能重采样上去)
        f = build_features(v, include_vision=True)
        assert f.shape == (SEQ_LEN, FUSED_FRAME_DIM) and np.isfinite(f).all()
        print(f"  trimmed_view 自检通过(以 {src.primary_label} 的 {src.frame_count} 帧造区间)")

    try:
        require_trim_spans(seqs)
    except ValueError as err:
        print(f"\n⚠️  {err}")

    x, y = build_xy(seqs[: min(8, len(seqs))], labels, include_vision=False)
    print(f"\n触觉特征形状: {x.shape}  范围 [{x.min():.3f}, {x.max():.3f}]")
    assert np.isfinite(x).all(), "特征里有 NaN/Inf"
    xf, _ = build_xy(seqs[: min(8, len(seqs))], labels, include_vision=True)
    print(f"融合特征形状: {xf.shape}  范围 [{xf.min():.3f}, {xf.max():.3f}]")
    assert np.isfinite(xf).all(), "融合特征里有 NaN/Inf"
    print("OK")
