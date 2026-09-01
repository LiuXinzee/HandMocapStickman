"""
dominant_hand —— `client/src/lib/dominantHand.ts` 里**离线那一半**的 Python 移植。

只移植 `handEnergy` / `judgeSampleDominance` / `normalizeSamplesToRight` 三个。
**没有**移植 `DominanceTracker`(带滞回的流式判定器)—— 那是推理端的东西,
离线是整条一起判、整条一起变换,不存在"一个词做到一半翻转口径"的风险,
而滞回机制唯一要防的就是那件事。

===== 一处刻意的简化,以及为什么它是无害的 =====

TS 那边弯折能量要除以**两点标定量出来的真实量程**(`channelSpans`),标定数据存在
浏览器 localStorage 里,导出的 dataset.bin/json 里没有它。所以这里一律走
兜底量程 `BEND_SPAN_FALLBACK`。

这**不是**"因为拿不到就将就",而是与 TS 完全相同的行为:那边的量程对称性守则写着
「只有两只手的标定都齐才用真实量程,否则两只手一起走兜底量程」,理由是一只手除真
量程、另一只除兜底量程会在比值里掺进一个纯人为的系统偏差,让判定偏向恰好标定过
的那只手。Python 侧两只手都没标定 → 两只手都走兜底 → 落在守则的同一个分支上。

顺带一个结论:兜底量程下 5 路的 span 全相等,于是 `handEnergy` 里那句
「物理槽位 j → canonical 指序,左手要倒序」的重映射是个恒等变换。移植过来是为了
日后真把标定接进来时不会漏掉,不是当前生效的逻辑。
"""
from __future__ import annotations

import numpy as np

from hand_mirror import mirror_sample

# ===== 以下常量与 dominantHand.ts 逐个对应,改一边要改另一边 =====

FINGER_PRESSURE_OFFSET, FINGER_PRESSURE_N = 0, 60
BEND_OFFSET, BEND_N = 60, 5

BEND_SIGMA_UNIT = 0.1
BEND_SPAN_FALLBACK = 90.0
PRESSURE_SIGMA_UNIT = 6.0
ORIENT_DEG_UNIT = 30.0

W_BEND, W_PRESSURE, W_ORIENT = 0.55, 0.25, 0.2

#: 两只手总能量都低于这个值 = 没有主手可判(`_idle` 伪类样本落在这里)
IDLE_ENERGY = 0.15
#: 能量比落在 [1/r, r] 内时判定基本由噪声决定。**只上报、不改变判定**
SAMPLE_TIE_RATIO = 1.3


def _std(a: np.ndarray) -> np.ndarray:
    """逐列样本标准差(ddof=1),与 TS 的 `stdOf` 同一口径(除以 count−1)。"""
    return a.std(axis=0, ddof=1)


def _quat_max_angle_deg(imu: np.ndarray) -> float:
    """首帧与其余各帧之间的最大转角(度)。"""
    q = imu[:, :4].astype(np.float64)
    # |dot|:q 与 −q 是同一个旋转,不取绝对值会把 0° 判成 180°
    d = np.abs(q @ q[0])
    # 四元数理应是单位的(厂家 IMU 直出),但不归一化时 dot 可能略大于 1 → acos 变 NaN
    return float(np.degrees(2.0 * np.arccos(np.clip(d, -1.0, 1.0))).max())


def hand_energy(sensor: np.ndarray | None, imu: np.ndarray | None) -> float | None:
    """
    一只手在整条样本上的加权运动能量(份)。>= 1 基本可以认为这只手在做动作。

    返回 None = 这只手没有传感器数据,或帧数不足 2(标准差无从谈起)。
    """
    if sensor is None or len(sensor) < 2:
        return None
    a = sensor.astype(np.float32)

    # 弯折:兜底量程下 5 路 span 相同,所以左手的指序倒序是恒等变换(见模块 docstring)
    bend = float(
        (_std(a[:, BEND_OFFSET : BEND_OFFSET + BEND_N]) / BEND_SPAN_FALLBACK).sum()
    ) / BEND_N / BEND_SIGMA_UNIT

    pressure = float(
        _std(a[:, FINGER_PRESSURE_OFFSET : FINGER_PRESSURE_OFFSET + FINGER_PRESSURE_N]).sum()
    ) / FINGER_PRESSURE_N / PRESSURE_SIGMA_UNIT

    orient = 0.0
    if imu is not None and len(imu) >= len(a):
        orient = _quat_max_angle_deg(imu[: len(a)]) / ORIENT_DEG_UNIT

    return W_BEND * bend + W_PRESSURE * pressure + W_ORIENT * orient


def judge_sample_dominance(seq) -> tuple[str, bool]:
    """
    一条**已录完**的样本由哪只手主导。

    返回 `(dominant, near_tie)`,`dominant ∈ {left, right, idle, no_hand}`。
    与 TS 的 `SampleDominance` 同一个取值集合 —— 刻意**没有** `both`:离线镜像的
    目标是把整个数据集拉到同一个口径,双手词整体镜像之后仍是同一个词,所以
    "两只手都在动"不是不作为的理由,只要还分得出谁主导就照判。
    """
    l = hand_energy(seq.left_sensor, seq.left_imu)
    r = hand_energy(seq.right_sensor, seq.right_imu)

    if l is None and r is None:
        return "no_hand", False
    if r is None:
        return "left", False
    if l is None:
        return "right", False
    # `_idle` 伪类落在这里。没有主手可判就别靠噪声 argmax 掷硬币 ——
    # 同一份数据每次训练都该给出同一个结果
    if l < IDLE_ENERGY and r < IDLE_ENERGY:
        return "idle", False

    near_tie = r > 0 and (1 / SAMPLE_TIE_RATIO) <= l / r <= SAMPLE_TIE_RATIO
    return ("left" if l > r else "right"), near_tie


def normalize_samples_to_right(seqs: list) -> tuple[list, dict]:
    """
    把整批样本归一化到**右手口径**:逐条判主手,判成左手的整条镜像。

    为什么必须做(照抄 TS 那边的论证,别再重新推一遍):特征层给两只手各留一段
    独立槽位,左手做的写进 `[0,147)`、右手做的写进 `[147,294)`,两组在输入空间里
    **零重叠**。同一个词一半左手一半右手采时,"信号落在哪半边"与标签完全无关,
    就是个纯噪声因子。镜像不是数据增广,是把这个结构性缺口补上。

    判为 `idle` / `no_hand` 的原样带过(镜像它们没有意义,也不该引入随机性)。
    """
    stats = {
        "total": len(seqs), "left": 0, "right": 0,
        "idle": 0, "no_hand": 0, "near_tie": 0,
    }
    out = []
    for s in seqs:
        d, tie = judge_sample_dominance(s)
        stats["no_hand" if d == "no_hand" else d] += 1
        if tie:
            stats["near_tie"] += 1
        out.append(mirror_sample(s) if d == "left" else s)
    return out, stats


def format_stats(stats: dict) -> str:
    """一行日志。`near_tie` 偏高说明该回去看采集方式,不是在这里加规则。"""
    return (
        f"手别归一化: {stats['total']} 条 → 镜像 {stats['left']} 条"
        f"(本来就是右手 {stats['right']} / 静止 {stats['idle']} / 无手 {stats['no_hand']});"
        f"其中左右能量接近、判定基本靠蒙的 {stats['near_tie']} 条"
    )
