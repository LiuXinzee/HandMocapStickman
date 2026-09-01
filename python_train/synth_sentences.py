"""
用现有的孤立词录制**合成**句子级序列,给 CTC 流水线做通路验证。

为什么要合成:
CTC 那条链路(逐帧 head → ctc_loss → greedy 解码 → 导出 tfjs → 浏览器解码)有五个
环节,每一个都能静默出错(blank 位置、标签顺序、时间分辨率、input_length 算法……)。
真实句子采集是几百条 × 10s 的活,不该拿它去调这些管道错误。合成数据能把整条管道
跑通、量出一个 WER 数字,采集之前就把管道错误清掉。

**合成数据不能替代真实数据。** 真实连续手语有协同发音:上一个词还没收手就开始
过渡到下一个词的手型。孤立词录制里根本不存在这种过渡,这里只能用**交叉淡化**去近似
(见 `crossfade_sequences`)。所以合成 val 的 WER 只说明"管道是对的",
完全不能预测真实表现。

===== 为什么不能用"词 + 静止帧 + 词"直接拼 =====

第一版是这么拼的:复制上一个词的最后一帧填 300~600ms 当停顿,然后直接接上下一个词
的第一帧。下一个词是**另一条录制**,手型/朝向/压力全不一样,于是接缝处出现一帧硬跳变。
量出来:

    词边界那一帧的跳变  0.0246
    词内相邻帧的变化    0.0002        ← 127 倍

模型因此根本没学"怎么切词",它学的是"看到 127 倍尖峰 + 一段死帧就换词"。合成 val
WER 4.4% 是真的(词池确实先划分再合成,没泄漏),但它量的是"能不能顺着标记读词"。
戴上手套连着打一句真实手语,尖峰一个都没有,切词能力直接归零 —— CTC 退回类先验,
输出永远是那几个最高频的词(代词)。

所以现在**没有静止段**:相邻两个词用一段平滑过渡衔接。过渡区的帧间变化会比词内大
几倍 —— 这是对的,真人在两个词之间手也移动得快;要消灭的是**单帧瞬移**,不是
"过渡处动得快"这件事本身。

===== 静止段由裁剪去掉,不由接缝去掉 =====

孤立词录制两端各有一段静止(抬手到位之前 + 打完收手之后),真实连续手语里不存在。
第一版靠加长交叉淡化(300~600ms)顺手吃掉它们,但**接缝吃不到整句的两端**
(`crossfade_sequences` 里 `lo = ... if i > 0 else 0`),于是句首的抬手 transport
和句尾的收尾静止一直留着。而推理端 `sentenceEnvelope.ts` 的起点是"第一次判到手在动"、
终点砍掉尾部 800ms —— 那两段推理时**永远见不到**。症状:合成 val WER 6.3% 看着不错,
396 条错例里 61 条全部错在句首。

现在改成:浏览器端 `sequenceTrim.detectSignSpan` 判出"动作真正开始/结束"的帧区间随
数据集导出(seq-1.1),这里用 `Sequence.trimmed_view()` 切掉静止再拼;接缝相应缩短到
100~250ms,只负责协同发音的过渡。**两件事分开做,各自能验证。**

**不写文件。** 这个模块在内存里造 `Sequence` 对象直接交给训练脚本。原本考虑追加进
dataset.bin,放弃了:那份文件是浏览器 IndexedDB 的唯一备份(删了没有撤销),
往里写合成数据的收益只是省一次几秒的重算,风险是把唯一备份搞坏。
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from label_merge import merge_label
from load_dataset import IDLE_LABEL, Sequence, quat_slerp, require_trim_spans

# ===== 本轮不训练的词 =====
#
# 这几个词的录制**一条都不进训练**。浏览器侧 `TrainSequence` 的默认排除清单是同一份
# (`client/src/lib/sentenceTemplates.ts` 的 `UNTRAINED_WORDS`,由
# `sentenceTemplates.test.ts` 逐条锁着)。
#
# 在这之前是**两套词表并存**:浏览器词模型 22 类、句子模型 23 类 —— 句子模型会输出
# 词模型从来没训过的词。症状只是"演示时偶尔蹦一个奇怪的词",不报错、WER 也看不出来。
#
# 两个词各自的理由不同,**在类别表上的后果也不同**:
#   happy   只有 3 条录制。合成句里同 3 条反复出现,学到的是这 3 条的噪声
#   you_pl  与 we/they 只差 yaw,合并后本来就不是一个独立类(见 label_merge)
# happy 合并后还是自己 → 类别表**少一类**(23 → 22)。you_pl 合并进 merged_pron_pl,
# 而 we/they 还在 → **那个类不会消失**,只是不再拿 you_pl 的录制去训它。
#
# 所以过滤必须"先合并、再看这个类还有没有活着的成员",不能拿名字直接从类别表里删 ——
# 直接删 you_pl 会把 we/they 一起带走,而症状是 softmax 下标整体错位、每个词都翻错。
#
# 过滤在 `train_seq.main()` 一处做:**先从原始标签里删成员,再合并**。
#
# ===== name 已于 2026-08-29 移出本清单 =====
# 它当初进来的理由是"17 条里 4 条裁完不足 250ms(too_short 整条不裁)→
# 同一类里两种时间包络"。现在要训「你叫什么名字」,所以训回来。
# ⚠ 旧的 17 条录制在浏览器 IndexedDB 里,**要先删再重录**,不能直接放行 ——
# 那 4 条坏包络还在,放行等于把当初的问题原样请回来,而症状只是「名字」准确率偏低。
UNTRAINED_WORDS: tuple[str, ...] = ("happy", "you_pl")

# ===== 句型表 =====
#
# 只用词表里真有的 id(见 client/src/lib/signLanguageVocab.ts)。运行时还会按"这个词
# 到底有没有录到够条数"再筛一遍,词不够的句型会被丢掉并打印出来 —— 不打印的话
# 会出现"某几句悄悄没参与训练,而 WER 看着还不错"。
#
# 语序按**手语**写,不是汉语:手语没有「的」,「是」很多时候也不打。顺句成汉语是
# 阶段 3 的 sentenceGrammar.ts 的活,这里只管标签序列。
#
# ⚠ 这张表和阶段 4 采集页里的句型表、以及 sentenceGrammar.ts 的规则表是**三份**。
# 加句型时三处都要加,否则会出现"训了但采不了 / 采了但顺不出汉语"。
# 下面「无主语」那一组只加在这里:它们的作用是压类先验(见本组注释),
# sentenceGrammar.ts 没命中规则时会原样拼词不吞词,所以缺规则只是文案难看一点。
SENTENCE_TEMPLATES: list[list[str]] = [
    # 问答
    # ⚠ 含 happy 的 3 条已删(见 UNTRAINED_WORDS)。删的是句型,
    # 类别表另有 trainable_class_table 管 —— 类别是从**数据集标签**推的,不是从这张表
    #
    # 2026-08-29 加回第一条:name 已移出 UNTRAINED_WORDS。含 name 的另外 6 条没加回来
    # —— name 这一轮只服务「你叫什么名字」这一句,句型加多了等于把采集预算摊薄
    ["you", "name", "what"],
    ["you", "eat", "what"],
    ["you", "speak", "what"],
    ["you", "work", "what"],
    ["you", "study", "what"],
    # 寒暄
    ["hello", "you"],
    ["goodbye", "you"],
    ["thank_you", "you"],
    ["sorry", "i"],
    ["welcome", "you"],
    ["welcome", "you", "we"],
    # 主谓宾。合并后是 [sg, love, sg] —— 同一个类在一句里出现两次(中间隔着动词,
    # 不相邻)。解码器把两次出现折成一次的话这类句子就会少一个词,值得多放几条。
    # 真正相邻的重复类(如 [sg, sg])在自然手语里不出现,所以这里没有 ——
    # 也就是说 2L−1 这个上界在当前句型表里其实用不满
    ["i", "love", "you"],
    ["you", "love", "i"],
    ["i", "listen", "you"],
    ["i", "help", "you"],
    ["you", "help", "i"],
    ["i", "speak", "you"],
    # 否定
    ["i", "is_not", "sad"],
    ["i", "is_not", "angry"],
    ["i", "is_not", "eat"],
    ["you", "is_not", "listen"],
    # 陈述
    ["i", "sad"],
    ["i", "eat"],
    ["i", "drink"],
    ["i", "work"],
    ["i", "study"],
    ["we", "study"],
    ["we", "work"],
    ["they", "study"],
    # 原来是 ["you_pl","listen"] —— you_pl 已不训练。换成 they 而不是删掉:
    # 复数类(merged_pron_pl)本来就只有合成数据支撑,少一句就更薄
    ["they", "listen"],
    ["he", "is_not", "work"],
    ["i", "thank_you", "you"],
    ["i", "help", "you", "study"],
    ["you", "eat", "drink"],
    # ===== 赞美 / 比喻(2026-08-29 追加)=====
    #
    # 用到四个新词(beautiful/smile/resemble/sun,见 client/src/lib/signLanguageVocab.ts
    # 末尾那批),**一条录制都还没有**。所以本轮跑起来这两句会被下面
    # synthesize_sentences 的"缺词"检查打印「跳过 … —— 缺 …」直接丢掉 ——
    # 那不是报错,是提醒还没采。采齐之后自动参与,不需要再改这张表。
    ["you", "beautiful"],
    # 全表第二条 4 词句(此前只有 ["i","help","you","study"])。4 词是最吃 32 个输出帧
    # 预算的一档(T=128 池化两次),多一条支撑对那一档是纯赚
    ["you", "smile", "resemble", "sun"],
    # ===== 无主语句(手语里主语常省略/靠指位表达)=====
    #
    # 加这一组是为了**压类先验**,不是为了句型更全。上面 39 句**每一句都带代词**,
    # 合并后 merged_pron_sg 独占 41% 的词位、第二名只有 5.7%。CTC 一旦对输入没把握
    # 就输出最高频的类 —— 线上表现就是"打什么都只蹦代词"。只靠给句型加权压不下来
    # (没有一句是不带代词的),必须真的加无代词的句子。
    #
    # 目标是把代词两类合计从 45.8% 压到 30% 上下,而不是压平:真实手语里代词本来
    # 就高频,压过头会变成另一个方向的偏。
    ["eat", "what"],
    ["drink", "what"],
    ["work", "what"],
    ["study", "what"],
    ["speak", "what"],
    ["listen", "what"],
    ["hello", "welcome"],
    ["hello", "thank_you"],
    ["goodbye", "thank_you"],
    ["help", "thank_you"],
    ["sorry", "goodbye"],
    ["is_not", "angry"],
    ["is_not", "sad"],
    ["is_not", "eat"],
    ["is_not", "drink"],
    ["is_not", "work"],
    ["eat", "drink"],
    ["listen", "speak"],
    ["love", "eat"],
    ["love", "study"],
    # ⚠ 三条"词堆"已删:["study","work"] / ["welcome","study","work"] /
    # ["help","study","thank_you"]。它们不是句子,是词的堆叠 —— 采集页会把它当
    # "请打这句话"提示、演示时也会读出来,读出来就是坏的。
    # 留着的理由曾是**压类先验**,那个理由随「我」拆成独立类已经失效:实测最高频类
    # 从 13.6% 只涨到 14.3%,离 _PRIOR_WARN_SHARE(0.35)还有一倍多余量。
    # 原先没删是因为两边有测试锁着、要同步改 —— 现在两边一起改了。
]

# 一条录制的"有哪些模态"签名。只在**签名相同**的录制之间拼句子
_SIG_KEYS = ("left_sensor", "right_sensor", "left_imu", "right_imu")


def hand_signature(s: Sequence) -> tuple[bool, ...]:
    """
    这条录制带了哪几路数据(左/右 × 触觉/IMU)。

    拼句子必须签名一致。不一致的话某个词那一段只能填零,而"全零的左手"在特征里
    等于"左手不存在" —— 模型会学到"句子中段左手会凭空消失",这是纯人造的规律,
    真实录制里不会出现。宁可丢掉少数派签名的录制,也不要造这种假信号。
    """
    return tuple(getattr(s, k) is not None for k in _SIG_KEYS)


def _median_dt(ts: np.ndarray) -> float:
    """帧间隔中位数。用中位数而不是均值:掉帧造成的大间隔会把均值拉偏。"""
    if len(ts) < 2:
        return 20.0
    d = np.diff(ts.astype(np.float64))
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 20.0


# 一个接缝最多吃掉相邻两词各 30% 的帧。再多就开始啃到词本身的动作了 ——
# 表现是模型学到的"词"少了收尾那一截,而真实录制里那一截是在的
_OVERLAP_MAX_FRAC = 0.30


def _smoothstep(k: int) -> np.ndarray:
    """
    过渡权重 0→1,两端导数为 0。

    用 smoothstep 而不是线性:线性权重在过渡区的**两个接头**上有速度突变
    (前一帧还是纯 A,后一帧突然带上 1/k 的 B),那还是个小尖峰,只是从 127 倍
    降到十几倍。smoothstep 把突变挪到过渡区中段并摊平,两个接头是 C¹ 连续的。
    """
    u = (np.arange(k, dtype=np.float32) + 1.0) / (k + 1.0)
    return u * u * (3.0 - 2.0 * u)


def _blend(key: str, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a 的尾段与 b 的头段按 w(0=全 a,1=全 b)混成过渡段,dtype 与输入一致。"""
    af = a.astype(np.float32)
    bf = b.astype(np.float32)
    if key.endswith("_imu"):
        out = np.empty_like(af)
        # 四元数**必须 slerp**。逐分量线性插值在大角度下给出非单位、方向错误的旋转,
        # 而两个词之间手腕朝向差 90° 以上很常见 —— 那会混出一个物理上不存在的姿态,
        # 并且 build_features 的重力投影是从同一个 q 派生的,会跟着一起错
        out[:, :4] = quat_slerp(af[:, :4], bf[:, :4], w)
        out[:, 4:] = af[:, 4:] + (bf[:, 4:] - af[:, 4:]) * w[:, None]
    else:
        out = af + (bf - af) * w[:, None]
    if np.issubdtype(a.dtype, np.integer):
        info = np.iinfo(a.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(a.dtype)


def _plan_overlaps(n: list[int], dts: list[float], overlap_ms: list[float]) -> list[int]:
    """每个接缝吃掉多少帧。受 _OVERLAP_MAX_FRAC 和"每个词至少留 1 帧自己的"两条约束。"""
    k = []
    for i in range(len(n) - 1):
        dt = 0.5 * (dts[i] + dts[i + 1])
        want = int(round(overlap_ms[i] / dt))
        cap = int(min(n[i], n[i + 1]) * _OVERLAP_MAX_FRAC)
        k.append(max(1, min(want, cap)))
    # 极短录制(如 10 帧的 _idle)可能被前后两个接缝一起吃光。从左往右只缩不涨,
    # 缩小 k[i-1] 不会让已经满足约束的词 i-1 再违约,所以一遍就收敛
    for i in range(len(n)):
        lo = k[i - 1] if i > 0 else 0
        hi = k[i] if i < len(k) else 0
        while lo + hi > n[i] - 1 and (lo > 0 or hi > 0):
            if hi >= lo and hi > 0:
                hi -= 1
            else:
                lo -= 1
        if i > 0:
            k[i - 1] = lo
        if i < len(k):
            k[i] = hi
    return k


def crossfade_sequences(parts: list[Sequence], overlap_ms: list[float]) -> Sequence:
    """
    把 N 条孤立词录制交叉淡化成一条句子级序列。

    每个接缝吃掉前词末尾 k 帧和后词开头 k 帧,混成 k 帧过渡。**不插静止段**,
    原因见模块 docstring 里那个 127 倍。

    k 该多大取决于**词有没有先裁过**。本函数只管按给定的 `overlap_ms` 混,判断留给
    调用方:`synthesize_sentences(trim_words=True)` 会先用 `trimmed_view()` 去掉
    起手/收尾静止,那时接缝只需覆盖协同发音的过渡时长(100~250ms);不裁的话接缝
    还要额外吃掉两侧各 ~250ms 静止,才需要旧的 300~600ms。用错组合不会报错 ——
    要么句首多一段抬手,要么接缝啃进动作本身。

    **注意接缝吃不到句首和句尾**:下面 `lo = k[i-1] if i > 0 else 0`、
    `hi = k[i] if i < len(k) else 0` —— 第一个词的开头和最后一个词的结尾一帧不动。
    所以整句两端的静止只能靠 `trimmed_view()` 去掉,交叉淡化在这件事上帮不上忙。
    这正是句首错误集中(61/396 全在句首)的直接原因。

    时间轴:词内帧间隔照抄原录制。绝对时间戳没有意义(各次录制的本地时钟),但帧间隔
    有意义 —— build_features 按时间戳插值,间隔改了等于把这个词放慢或加速。过渡段用
    前后两词 dt 的均值。

    `segments` 的 startFrame/endFrame 现在是**近似边界**:过渡区那 k 帧两个词都算不上,
    这里把它划在两个 segment 之间(既不属于前词也不属于后词)。CTC 不用词边界,
    所以这不影响训练;但别拿它当对齐结果用。
    """
    if not parts:
        raise ValueError("没有词可拼")
    if len(overlap_ms) != len(parts) - 1:
        raise ValueError(f"{len(parts)} 个词需要 {len(parts)-1} 个过渡,给了 {len(overlap_ms)}")
    sig = hand_signature(parts[0])
    for p in parts[1:]:
        if hand_signature(p) != sig:
            raise ValueError("模态签名不一致的录制不能拼(见 hand_signature)")

    n = [len(p.timestamps) for p in parts]
    dts = [_median_dt(p.timestamps.astype(np.float32)) for p in parts]
    k = _plan_overlaps(n, dts, overlap_ms)

    dt_chunks: list[np.ndarray] = []  # 每帧相对前一帧的间隔,最后 cumsum 成时间戳
    blocks: dict[str, list[np.ndarray]] = {key: [] for key in _SIG_KEYS}
    segments: list[dict] = []
    frame = 0

    for i, p in enumerate(parts):
        lo = k[i - 1] if i > 0 else 0
        hi = k[i] if i < len(k) else 0
        own = slice(lo, n[i] - hi)
        m = own.stop - own.start
        t = p.timestamps.astype(np.float32)
        inner = np.diff(t[own]) if m > 1 else np.zeros(0, np.float32)
        first_dt = 0.0 if i == 0 else 0.5 * (dts[i - 1] + dts[i])
        dt_chunks.append(np.concatenate([[first_dt], inner]).astype(np.float32))
        for key in _SIG_KEYS:
            arr = getattr(p, key)
            if arr is not None:
                blocks[key].append(arr[own])
        segments.append(
            {"label": p.primary_label, "startFrame": frame, "endFrame": frame + m}
        )
        frame += m

        if i == len(parts) - 1:
            break
        kk = k[i]
        nxt = parts[i + 1]
        w = _smoothstep(kk)
        dt_chunks.append(np.full(kk, 0.5 * (dts[i] + dts[i + 1]), np.float32))
        for key in _SIG_KEYS:
            a, b = getattr(p, key), getattr(nxt, key)
            if a is not None and b is not None:
                blocks[key].append(_blend(key, a[n[i] - kk :], b[:kk], w))
        frame += kk

    timestamps = np.cumsum(np.concatenate(dt_chunks)).astype(np.float32)
    merged = {
        key: (np.concatenate(blocks[key], axis=0) if blocks[key] else None)
        for key in _SIG_KEYS
    }
    return Sequence(
        segments=segments,
        # primary_label 是第一个词 —— 与浏览器端 datasetStore 的约定一致。
        # 正是这个字段会让句子冒充孤立词,所以 is_sentence(len(segments)>1) 必须为真
        primary_label=parts[0].primary_label,
        frame_count=len(timestamps),
        duration_ms=float(timestamps[-1] - timestamps[0]),
        source_fps=parts[0].source_fps,
        origin="synthesized",
        timestamps=timestamps,
        left_sensor=merged["left_sensor"],
        right_sensor=merged["right_sensor"],
        left_imu=merged["left_imu"],
        right_imu=merged["right_imu"],
        # 视觉一律丢掉:句子模型训的是纯触觉学生(部署时摄像头不一定在)。留着只会
        # 让特征维度在 294/420 之间摇摆,而 CTC 这条路根本不做视觉蒸馏
        left_landmarks=None,
        right_landmarks=None,
    )


def synthesize_sentences(
    word_seqs: list[Sequence],
    per_template: int,
    rng: np.random.Generator,
    templates: list[list[str]] | None = None,
    # 每个接缝吃掉多长。**这个区间是跟着 trim_words 一起定的,别单独改。**
    #
    # 旧默认是 (300, 600),那是在"词还带着起手/收尾静止"的世界里定的:接缝要既
    # 完成过渡、又顺手吃掉两侧各 ~250ms 的静止(见 crossfade_sequences docstring)。
    # trim_words=True 之后那两段静止已经在裁剪阶段去掉了,照用 300~600ms 会直接
    # **啃进动作本身** —— 模型学到的"词"缺了收尾那一截,而真实录制里那一截是在的。
    # `_OVERLAP_MAX_FRAC`(30%)只是个上限,不是保护:2.6s 的词 30% 就是 780ms,
    # 想吃 600ms 完全够额度,一帧都不会被它挡下来。
    #
    # 新值只需覆盖真人协同发音的过渡时长(上一个词还没收手就开始转向下一个手型),
    # 100~250ms 对应 5~13 帧 @50Hz。改这个值必须重跑 __main__ 的集中度自检。
    overlap_ms: tuple[float, float] = (100.0, 250.0),
    trim_words: bool = True,
    verbose: bool = True,
) -> tuple[list[Sequence], list[list[str]]]:
    """
    孤立词录制池 → 合成句子列表 + 实际用上的句型表。

    **调用方必须先把录制池划成训练/验证两半再分别调用本函数。** 否则同一条孤立词
    录制会既出现在训练句里、又出现在验证句里 —— 那是泄漏,验证 WER 会虚低到没有
    参考价值(等于在考已经背过的题,只是词序换了)。

    只在同一模态签名内拼(见 hand_signature),少数派签名的录制会被丢掉并报数。

    `trim_words=True` 时先对每条录制调 `Sequence.trimmed_view()`,拼的是"只剩动作"
    的那一段。这是为了对齐推理端的时间包络:`sentenceEnvelope.ts` 的起点是**第一次
    判到手在动**(不是按下按钮那一刻),终点砍掉尾部 800ms 静止 —— 也就是说句首的
    抬手 transport 和句尾的收尾静止,推理时**永远见不到**。不裁的话句首那一段就是
    一个只存在于训练集里的人造前缀,与模块 docstring 里那个 127 倍尖峰同一类错误。
    没有 trimSpan 的数据集会在这里报错(`require_trim_spans`),不会静默不裁。
    """
    templates = templates if templates is not None else SENTENCE_TEMPLATES

    # 在建池之前拦:要求裁剪却拿到 seq-1.0 时,静默不裁的唯一症状是 WER 差几个点
    if trim_words:
        require_trim_spans(word_seqs)

    pool_all: dict[tuple, dict[str, list[Sequence]]] = {}
    for s in word_seqs:
        if s.primary_label == IDLE_LABEL:
            continue  # _idle 不是词,不进句子目标
        pool_all.setdefault(hand_signature(s), {}).setdefault(s.primary_label, []).append(s)
    if not pool_all:
        return [], []

    # 取录制条数最多的那个签名。多签名混训是可以的(分别合成再合并),但会让
    # "哪些词有数据"这件事变成按签名不同 —— 排查起来很绕,收益又很小
    sig = max(pool_all, key=lambda k: sum(len(v) for v in pool_all[k].values()))
    pool_src = pool_all[sig]
    kept = sum(len(v) for v in pool_src.values())
    dropped = sum(
        len(v) for k, d in pool_all.items() if k != sig for v in d.values()
    )
    if verbose:
        names = [k for k, on in zip(_SIG_KEYS, sig) if on]
        print(f"  合成用模态签名 {names},可用 {kept} 条" + (f",丢弃其他签名 {dropped} 条" if dropped else ""))
        # 报在裁剪之前:trimmed_view 会把 trim_applied 置回 False(为了幂等),
        # 裁完就再也看不出"到底裁了多少"了。这里是最后一处出口
        _report_trim([s for v in pool_src.values() for s in v], trim_words)

    # trimmed_view 在 trim_applied 为假时原样返回同一个对象,所以这一层对
    # "判据说不用裁"的录制是零成本的,不会白复制一遍数组
    pool = (
        {w: [s.trimmed_view() for s in v] for w, v in pool_src.items()}
        if trim_words
        else pool_src
    )

    used: list[list[str]] = []
    skipped: list[tuple[list[str], list[str]]] = []
    out: list[Sequence] = []
    for tpl in templates:
        missing = [w for w in tpl if not pool.get(w)]
        if missing:
            skipped.append((tpl, sorted(set(missing))))
            continue
        used.append(tpl)
        for _ in range(per_template):
            parts = [pool[w][int(rng.integers(len(pool[w])))] for w in tpl]
            laps = list(rng.uniform(overlap_ms[0], overlap_ms[1], size=len(tpl) - 1))
            out.append(crossfade_sequences(parts, laps))

    if verbose:
        print(f"  句型 {len(used)}/{len(templates)} 可用 → 合成 {len(out)} 条句子")
        if skipped:
            for tpl, miss in skipped[:8]:
                print(f"    跳过 {' '.join(tpl)} —— 缺 {','.join(miss)}")
            if len(skipped) > 8:
                print(f"    …… 另有 {len(skipped)-8} 句被跳过")
        _report_prior(used)
    return out, used


# 类先验失衡到什么程度就该管:最高频类超过这个份额,CTC 一遇到没把握的输入就会
# 无脑输出它。曾经 merged_pron_sg 占 41%,线上表现就是"打什么都只蹦代词"
#
# 「我」拆成独立类之后这条压力小了一半(实测,当前 66 句表 / 162 词位):
#   拆分前 merged_pron_sg 26.5%  →  拆分后 merged_pron_sg 13.6% + i 13.0%
# 同一批词位被劈成两个类,最高频类直接对折。**报警线不要因此调低** ——
# 它防的是"以后加句型把某一类堆起来",与当前余量多少无关。
_PRIOR_WARN_SHARE = 0.35


def _report_trim(pool: list[Sequence], trim_words: bool) -> None:
    """
    裁剪覆盖率。**必须打印**,和 `_report_prior` 同一条理由:裁多了裁少了都不报错,
    只在真实句子上表现为句首丢词 —— 合成 val WER 完全看不出来。

    这里是整条流水线上最后一处能看见裁剪读数的地方:下游只拿到切好的片段,
    连"这条被裁过没有"都不知道(trimmed_view 有意把 trim_applied 置回 False)。
    """
    if not pool:
        return
    if not trim_words:
        print("  ⚠ trim_words=False —— 句首带着抬手 transport,与推理端时间包络不一致")
        return
    reasons = Counter(s.trim_reason for s in pool)
    cut = [s for s in pool if s.trim_applied]
    body = "、".join(f"{r} {c}" for r, c in reasons.most_common())
    print(f"  裁剪:{len(cut)}/{len(pool)} 条要裁({body})")
    if cut:
        kept = float(
            np.mean([(s.trim_span[1] - s.trim_span[0]) / s.frame_count for s in cut])
        )
        eaten = float(np.mean([s.duration_ms for s in cut])) - float(
            np.mean([s.trimmed_view().duration_ms for s in cut])
        )
        print(f"    平均留下 {kept*100:.0f}% 帧,平均去掉 {eaten:.0f}ms 头尾静止")
    blind = sum(1 for s in pool if not s.trim_tactile_ran)
    if blind:
        # 第三层是唯一会裁尾的一层。它没跑 = 收尾静止还在,顶在合成句的句尾,
        # 而推理端 SETTLE_MS(800ms) 会把尾部静止掐掉 —— 口径又差回去了
        print(f"    ⚠ {blind}/{len(pool)} 条没跑过触觉静止段判据 → 句尾静止仍在(导出时没做弯折两点标定)")


def _report_prior(used: list[list[str]]) -> None:
    """打印合并后的词位分布。**必须打印**:先验失衡是静默的,WER 看不出来。"""
    tok = Counter(merge_label(w) for tpl in used for w in tpl)
    total = sum(tok.values())
    if not total:
        return
    top = tok.most_common(3)
    body = "、".join(f"{name} {100 * c / total:.0f}%" for name, c in top)
    print(f"  合并后词位分布(共 {total} 个词位,{len(tok)} 类):{body}")
    share = top[0][1] / total
    if share > _PRIOR_WARN_SHARE:
        print(
            f"    ⚠ {top[0][0]} 独占 {share*100:.0f}% —— 模型对输入没把握时会退回输出它。"
            f"加不含该词的句型来压,给句型加权压不动"
        )


def seam_concentration(s: Sequence) -> float | None:
    """
    每个接缝的"变化集中度" = 过渡区内最大单帧变化 ÷ 过渡区平均单帧变化。
    返回这条句子里最差的那个接缝。None = 只有一个词。

    **为什么是这个量而不是"过渡 ÷ 词内"。** 先写的是后者,`is_not happy` 这类句子
    能量到 49 倍 —— 不是接缝坏,是 `happy` 本身几乎不动,分母塌了。词与词之间该有
    多大变化,取决于两个词的手型差多少,没有一个跟"词内动多少"可比的基准。

    集中度是自封闭的:两个词差多少都会被分母吸收,量的纯粹是**这段变化摊开了没有**。
    - 一帧瞬移 + k 帧冻结 → 集中度 ≈ k(第一版就是这样,而且总变化全在那一帧)
    - smoothstep 均匀过渡    → 集中度 ≈ 1.5(峰值是均值的 1.5 倍)

    **读数随窗长 k 变**:上界就是 k,所以 `overlap_ms` 一改这里的所有数字都要重量。
    `__main__` 现场量对照组(`control_concentration`)正是为此 —— 不要把下面的数字
    当常数用。

    实测(`overlap_ms=(100,250)`,`trim_words=True`,378/395 条被裁、平均留下 54%
    → 接缝中位 8 帧,窗长 9,上界 9×):
        过渡区      中位 3.25  最差 6.80
        对照组      中位 1.89  p99 3.66  最大 9.00   ← 真实录制内部同长滑窗
    过渡区落在真实动作分布的偏上部分,不是瞬移。

    **三种配置下过渡区相对对照组 p99 的位置几乎不变**:
        旧 21 帧接缝 / 未裁词      4.70 / 5.40 = 0.87
        9 帧接缝  / 未裁词         3.34 / 3.88 = 0.86
        8 帧接缝  / 已裁词(当前)   3.25 / 3.66 = 0.89
    也就是说"接缝缩短 + 词先裁静止"这两件事都没有把过渡变成尖峰 —— 绝对读数变小
    只是因为量程随窗长缩了,不是接缝变好了。**别拿这三行里的任意一个数当常数用。**

    比理论的 1.5 高,是因为过渡区吃进来的是前词的收手和后词的起手,那两截本身就是
    整条录制里动得最快的地方。裁剪之后这一点更明显:静止边缘没了,接缝混的直接
    就是动作边缘。

    在**原始帧**上算,不走 build_features:重采样到 128 帧会把尖峰抹平,
    正好把要查的东西抹掉。
    """
    if len(s.segments) < 2:
        return None
    d = _frame_deltas(s)
    worst = None
    for a, b in zip(s.segments[:-1], s.segments[1:]):
        # d[j] 是 x[j]→x[j+1]。过渡区是 [endFrame, startFrame),再往前带一帧
        # 把"最后一帧自己的 → 第一帧过渡"这个接头也算进来
        lo = max(0, a["endFrame"] - 1)
        hi = min(len(d), b["startFrame"])
        seg = d[lo:hi]
        if len(seg) < 2 or seg.mean() <= 0:
            continue
        c = float(seg.max() / seg.mean())
        worst = c if worst is None else max(worst, c)
    return worst


def _frame_deltas(s: Sequence) -> np.ndarray:
    """逐帧平均绝对变化(uint8 归一化到 [0,1] 以便与 float 通道同权)。"""
    cols = [getattr(s, key) for key in _SIG_KEYS]
    x = np.concatenate(
        [
            (c.astype(np.float32) / 255.0 if c.dtype == np.uint8 else c.astype(np.float32))
            for c in cols
            if c is not None
        ],
        axis=1,
    )
    return np.abs(np.diff(x, axis=0)).mean(axis=1)


def control_concentration(words: list[Sequence], k: int) -> np.ndarray:
    """
    对照组:在**真实孤立词录制内部**、同样 k 帧的滑窗上算同一个集中度指标。

    没有它 `seam_concentration` 的读数没法解释:集中度多大算"过渡区正常地动得快"、
    多大算"瞬移",取决于真人动作本身在同长窗口里有多集中。

    **窗长必须跟着接缝长度走。** max/mean 的上界恰好是 k(全部变化挤在一帧),
    所以这个指标的量程随 k 缩小 —— 拿 21 帧时量的对照去解释 9 帧的接缝是错的。
    这就是 `overlap_ms` 一改就必须重跑本函数的原因。
    """
    out: list[float] = []
    for s in words:
        d = _frame_deltas(s)
        for i in range(0, len(d) - k + 1):
            seg = d[i : i + k]
            if seg.mean() > 0:
                out.append(float(seg.max() / seg.mean()))
    return np.asarray(out, dtype=np.float64)


if __name__ == "__main__":
    import argparse
    import sys

    from load_dataset import build_features, load_dataset

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="自检:从现有孤立词合成句子并看形状")
    ap.add_argument("--data", default="data")
    ap.add_argument("--per-template", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    # 只为了能在 seq-1.0 的旧导出上跑这套自检。**训练不要用它** ——
    # 不裁就是句首带抬手段,正是本轮要修的那个口径差
    ap.add_argument("--no-trim", action="store_true", help="不按 trimSpan 裁词(旧数据集)")
    args = ap.parse_args()

    seqs, labels = load_dataset(args.data)
    words = [s for s in seqs if not s.is_sentence]
    print(f"孤立词 {len(words)} 条")
    sents, used = synthesize_sentences(
        words,
        args.per_template,
        np.random.default_rng(args.seed),
        trim_words=not args.no_trim,
    )
    if not sents:
        raise SystemExit("一条也没合成出来 —— 先去 /collect-seq 采孤立词")

    lens = Counter(len(s.segments) for s in sents)
    print(f"词数分布: {dict(sorted(lens.items()))}")
    print(f"时长: 中位 {np.median([s.duration_ms for s in sents]):.0f}ms  "
          f"最长 {max(s.duration_ms for s in sents):.0f}ms")
    print(f"帧数: 中位 {np.median([s.frame_count for s in sents]):.0f}")

    x = np.stack([build_features(s, args.seq_len) for s in sents[:8]])
    print(f"特征形状 {x.shape} 范围 [{x.min():.3f}, {x.max():.3f}]")
    assert np.isfinite(x).all(), "特征里有 NaN/Inf"

    # 时间戳必须严格递增 —— 拼接时游标算错会让 _sample_index 的 searchsorted 乱掉,
    # 症状是特征看着正常但时间顺序被打乱,几乎不可能靠肉眼发现
    for s in sents:
        d = np.diff(s.timestamps)
        assert (d > 0).all(), f"时间戳非递增: min dt = {d.min()}"
        assert s.segments[-1]["endFrame"] <= s.frame_count, "segment 越界"

    # 词边界不能有单帧瞬移。第一版拿"冻结帧 + 硬接缝"拼,集中度就等于过渡帧数
    # (变化全挤在一帧里)—— 模型直接把这个尖峰当分词标记,换成真实连续手语
    # (没有尖峰)就完全切不动词,退回类先验,线上表现是"打什么都只蹦代词"。
    # 这条断言是那个 bug 的回归测试。**断言中位数而不是最大值**:真实动作爆发也能
    # 让单个窗口的集中度很高(见对照组的最大值),拿最大值当判据会误伤。
    #
    # 阈值**不是常数**:max/mean 的上界恰好等于窗长 k,所以 overlap_ms 一缩短,
    # 既有的读数和判据都要跟着重算(旧值是 21 帧窗口下的 中位 2.3 / p99 5.4)。
    # 这里现场量对照组,再把阈值定在"对照组 p99 与硬接缝上界 k 之间"。
    seam_ks = [
        b["startFrame"] - a["endFrame"]
        for s in sents
        for a, b in zip(s.segments[:-1], s.segments[1:])
    ]
    k_med = int(np.median(seam_ks))
    cons = [c for c in (seam_concentration(s) for s in sents) if c is not None]
    assert cons, "一条多词句子都没有,过渡区无从检查"
    med = float(np.median(cons))

    ctrl = control_concentration(words[:60], k_med + 1)  # +1:接缝多带一帧接头
    c_med, c_p99, c_max = (
        float(np.median(ctrl)),
        float(np.percentile(ctrl, 99)),
        float(ctrl.max()),
    )
    print(
        f"接缝长度: 中位 {k_med} 帧（{min(seam_ks)}~{max(seam_ks)}）;"
        f"硬接缝上界 {k_med + 1:.0f}×"
    )
    print(
        f"过渡区变化集中度: 中位 {med:.2f}×  最差 {max(cons):.2f}×  "
        f"（真实录制内同长窗口 中位 {c_med:.2f} / p99 {c_p99:.2f} / 最大 {c_max:.2f}）"
    )

    # 上界 = 硬接缝会读到的数。它和对照组 p99 之间必须还有判别空间,否则这条回归
    # 测试已经失效 —— 接缝短到"瞬移"和"正常动作爆发"读数重叠,再断言什么都没意义
    ceiling = k_med + 1
    assert c_p99 < ceiling * 0.85, (
        f"接缝只有 {k_med} 帧,对照组 p99({c_p99:.2f})已经贴到硬接缝上界({ceiling})——"
        f"集中度不再能区分瞬移和真实动作。把 overlap_ms 调长,或换一个不随窗长缩量程的判据"
    )
    thresh = round(0.5 * (c_p99 + ceiling), 1)
    print(f"判据阈值 {thresh}×（对照组 p99 与上界的中点）")
    assert med < thresh, (
        f"集中度 {med:.2f}× ≥ {thresh}× —— 过渡区又变回硬接缝了。模型会拿这个尖峰"
        f"当分词标记,线上遇到真实连续手语就只会输出类先验"
    )
    print("OK")
