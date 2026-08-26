"""
label_merge —— client/src/lib/labelMerge.ts 的 Python 镜像。

**两边必须一字不差地对应。** 类别表的顺序就是 softmax 下标,Python 训完的模型转回
tfjs 在浏览器里跑,只要合并规则或 id 有一处不同,每个词都会翻译成另一个词 ——
而且是"能跑、有置信度、就是全错"的那种,极难发现。

为什么要合并(照抄 labelMerge.ts 的结论,别再重新调研一遍):
  我/你/他 是同一手型指不同方向,区别几乎纯在 yaw。特征里两条 IMU 通路都对 yaw
  不变(相对首帧四元数消掉绝对朝向、重力投影天生 yaw 不变),所以这三个词在特征
  空间里是同一个点。手套是 ICM-42688 六轴无磁力计,绝对 yaw 零点在硬件层就不存在。
  实测(yawDrift 探针,399 条):静止时 yaw 漂移 P90 15.92°/s → 归零 10s 后累计 159°,
  而类间距只有 45°;归零点本身 P90 也有 43.4°。软件绕不过去。

CTC 这条路同样要合并:句子里的「我」和「你」在触觉特征上仍然是同一个东西,
不合并的话 CTC 会在这两个类之间随机分配概率,连带把整句的对齐搅乱。
"""
from __future__ import annotations

IDLE_LABEL = "_idle"

_YAW_REASON = (
    "同手型、区别纯在指向(yaw);六轴 IMU 无磁力计,绝对 yaw 不可观测"
    "(实测漂移 P90 15.9°/s,10s 累计 159°,类间距仅 45°)"
)

# (合并后 id, 显示名, 成员, 理由)。id 带 merged_ 前缀,避免和词表里的真实词撞车
MERGE_GROUPS: list[tuple[str, str, list[str], str]] = [
    ("merged_pron_sg", "我/你/他", ["i", "you", "he"], _YAW_REASON),
    # 复数是"同手型 + 一段横向弧线",那段弧线在手系里也是同一个旋转,一样简并
    ("merged_pron_pl", "我们/你们/他们", ["we", "you_pl", "they"], _YAW_REASON),
]

_MEMBER_TO_GROUP: dict[str, str] = {
    m: gid for gid, _, members, _ in MERGE_GROUPS for m in members
}
DISPLAY: dict[str, str] = {gid: disp for gid, disp, _, _ in MERGE_GROUPS}


def merge_label(label: str, enabled: bool = True) -> str:
    """一个原始标签 → 训练用的类别名。enabled=False 时原样返回(对照实验的基线)。"""
    if not enabled:
        return label
    return _MEMBER_TO_GROUP.get(label, label)


def merge_label_list(labels: list[str], enabled: bool = True) -> list[str]:
    """一条句子的标签序列整体重映射。顺序不动 —— CTC 的目标就是这个顺序。"""
    return [merge_label(l, enabled) for l in labels]


def merged_class_table(labels: list[str], enabled: bool = True) -> list[str]:
    """
    原始标签表 → 合并后的类别表,**排序后返回**。

    排序方式与 TS 侧 `Array.from(new Set(...)).sort()` 一致(JS 默认字符串排序 =
    按 UTF-16 码位升序,ASCII 范围内与 Python 的 sorted() 相同;这些 id 全是 ASCII)。
    顺序就是 softmax 下标,两边不一致 = 全部翻译错。
    """
    return sorted({merge_label(l, enabled) for l in labels})
