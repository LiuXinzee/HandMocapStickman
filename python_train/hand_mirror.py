"""
hand_mirror —— `client/src/lib/handMirror.ts` 的 Python 移植。

**为什么这个文件要存在**:训练搬到 Python 之后,手别归一化这一步丢了。
浏览器端 `dominantHand.ts` 的 `normalizeSamplesToRight` 是为页面内训练写的,
Python 侧 `grep mirror|dominan` 是零命中 —— 于是:

    推理侧把左手打的整条镜像到右手口径(`Translate.tsx` 的 normalizeHandedness),
    训练侧原样入库。

后果实测过:库里 31 个词有 15 个是左右手混采的(`eat` 最严重,18 条里 10 条左手),
而句子模型 95% 的训练数据是拿这些词录制合成的。手别与标签无关时网络只能学
手别无关的特征,代价是判据变宽 —— `eat` 因此变成一个"什么都收"的吸引子:
原样 18/18 判对、**镜像后还有 14/18 仍是 eat**、词挤到窗口前 40% / 后 40% 也还是 eat。
而 `smile` 反过来只认右手槽位,镜像后 10/15 直接塌成「你」。
「你笑起来像太阳」被翻成「你吃像太阳」就是这两件事撞在一起。

置换表的推导、IMU 反射共轭的推导、以及"镜面法向取 x 轴未经硬件验证"这一条,
**都在 `handMirror.ts` 的文件头**,不在这里重抄一遍 —— 抄两份的下场是改了一份。
这里只保证一件事:**与那边逐位相同**。`handMirror.test.ts` 会解析本文件的
`MIRROR_PERM_137` 与 TS 那张表逐项比对,任何一边改了另一边没跟上,那条测试会红。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

# 与 handMirror.ts 的 MIRROR_PERM_137 逐项相同(有跨语言测试锁着)。
# 语义:dst[MIRROR_PERM_137[k]] = src[k]。**自逆**,所以左→右与右→左共用一张表。
MIRROR_PERM_137: tuple[int, ...] = (
     50,  49,  48,  53,  52,  51,  56,  55,  54,  59,  58,  57,
     38,  37,  36,  41,  40,  39,  44,  43,  42,  47,  46,  45,
     26,  25,  24,  29,  28,  27,  32,  31,  30,  35,  34,  33,
     14,  13,  12,  17,  16,  15,  20,  19,  18,  23,  22,  21,
      2,   1,   0,   5,   4,   3,   8,   7,   6,  11,  10,   9,
     64,  63,  62,  61,  60,  76,  75,  74,  73,  72,  71,  70,
     69,  68,  67,  66,  65,  91,  90,  89,  88,  87,  86,  85,
     84,  83,  82,  81,  80,  79,  78,  77, 106, 105, 104, 103,
    102, 101, 100,  99,  98,  97,  96,  95,  94,  93,  92, 121,
    120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, 109,
    108, 107, 136, 135, 134, 133, 132, 131, 130, 129, 128, 127,
    126, 125, 124, 123, 122,
)

_PERM = np.asarray(MIRROR_PERM_137, dtype=np.int64)

# 表写死成 137 维。长度不符时静默错位比崩掉危险得多 —— 照抄 TS 那边的守则
if len(MIRROR_PERM_137) != 137:
    raise AssertionError(f"MIRROR_PERM_137 长度 {len(MIRROR_PERM_137)} != 137")
if sorted(MIRROR_PERM_137) != list(range(137)):
    raise AssertionError("MIRROR_PERM_137 不是双射")
if not (_PERM[_PERM] == np.arange(137)).all():
    raise AssertionError("MIRROR_PERM_137 不自逆 —— 左→右与右→左就不能共用一张表了")

# 每帧 10 个 IMU 槽位的符号,与 mirrorImuSeries 逐位对应:
#   [0:4] 四元数 (w,x,y,z) → (w,x,−y,−z)     反射共轭
#   [4:7] 加速度 (x,y,z)   → (−x,y,z)         真矢量
#   [7:10] 姿态角 (yaw,roll,pitch) → (yaw,−roll,−pitch)
# ⚠ 实测这副手套的 [4:10] 六路**全是零**(62619 帧里唯一值只有 1),所以这一段
# 目前不影响任何数字。保留是为了与 TS 侧逐位一致,别因为"反正是零"就删掉。
IMU_SIGN = np.array([1, 1, -1, -1, -1, 1, 1, 1, -1, -1], dtype=np.float32)


def mirror_sensor_series(a: np.ndarray | None) -> np.ndarray | None:
    """`[T,137]` 传感器序列镜像。返回新数组(置换不是原地可做的)。"""
    if a is None:
        return None
    out = np.empty_like(a)
    out[:, _PERM] = a  # dst[PERM[k]] = src[k]
    return out


def mirror_imu_series(a: np.ndarray | None) -> np.ndarray | None:
    """`[T,10]` IMU 序列镜像 —— 逐通道翻号,见 `IMU_SIGN`。"""
    return None if a is None else (a.astype(np.float32) * IMU_SIGN)


def mirror_landmark_series(a: np.ndarray | None) -> np.ndarray | None:
    """
    `[T,63]` 关键点序列镜像:21 个点逐点 `x → 1 − x`,y / z 不动。

    `1 − x` 而不是 `−x`,因为 MediaPipe 关键点是归一化图像坐标 x∈[0,1]。
    缺失帧是整帧 NaN,`1 − NaN = NaN`,可见性判据不受影响 —— 所以这里
    **不能**用 `np.nan_to_num` 之类的"顺手清理"。
    """
    if a is None:
        return None
    out = a.astype(np.float32).copy()
    out[:, 0::3] = 1.0 - out[:, 0::3]
    return out


def mirror_sample(seq):
    """
    整条样本镜像 **并互换左右槽位** —— 与 TS 的 `mirrorSample` 一一对应。

    互换是重点:左手做的动作镜像之后就是右手做同一个动作,所以左手那段数据要
    写进右手槽位。只镜像不换槽(或只换槽不镜像)都会得到一条物理上不存在的样本。
    """
    return replace(
        seq,
        left_sensor=mirror_sensor_series(seq.right_sensor),
        right_sensor=mirror_sensor_series(seq.left_sensor),
        left_imu=mirror_imu_series(seq.right_imu),
        right_imu=mirror_imu_series(seq.left_imu),
        left_landmarks=mirror_landmark_series(seq.right_landmarks),
        right_landmarks=mirror_landmark_series(seq.left_landmarks),
    )


if __name__ == "__main__":
    # 自检:置换的三条不变量在 import 时已经断言过,这里补几条结构性的
    perm = _PERM
    assert (perm[60:65] == [64, 63, 62, 61, 60]).all(), "弯折 5 路应当整体倒序"
    assert all(perm[k] < 60 for k in range(60)), "手指块必须仍落在手指块"
    assert all(perm[k] >= 65 for k in range(65, 137)), "手掌块必须仍落在手掌块"

    # 镜像两次 = 原样(自逆 + 换槽两次)
    class _S:
        pass

    rng = np.random.default_rng(0)
    s = np.asarray(rng.integers(0, 256, (7, 137)), dtype=np.uint8)
    assert (mirror_sensor_series(mirror_sensor_series(s)) == s).all()
    q = rng.standard_normal((7, 10)).astype(np.float32)
    assert np.allclose(mirror_imu_series(mirror_imu_series(q)), q)
    lm = rng.random((7, 63)).astype(np.float32)
    assert np.allclose(mirror_landmark_series(mirror_landmark_series(lm)), lm)
    print("hand_mirror OK")
