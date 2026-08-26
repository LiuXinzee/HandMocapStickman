/*
 * motionGate —— 滑窗推理前的一道闸门：**这个窗口里到底有没有人在做动作？**
 *
 * ===== 为什么需要它 =====
 *
 * 分类模型的输出是"在这些词里选一个"，它没有"什么都不是"这个选项。滑窗推理每
 * 100ms 就把最近一个窗口喂进去一次，其中绝大多数窗口里**根本没有手语**——手放着、
 * 手正抬起来、两个词之间的间隙。这些窗口一样会得到一个 argmax 和一个置信度。
 *
 * 原来指望 `_idle` 伪类接住这些窗口（`Translate.tsx` 命中 idle 就清空平滑缓冲、
 * 不出词）。但那条路要求 `_idle` 真的被训过：实测数据集里 `_idle` 只有 4 条、
 * 中位时长 211ms，对着 2000ms 的推理窗口等于没有这个类，闸门形同虚设。
 *
 * 后果不是"偶尔多吐一个词"，而是**静止时最容易吐词**，这一点很反直觉：
 * 下游确认一个词要求「连续 N 次结论相同」+「置信度过门限」。手放着不动时，
 * 相邻窗口的内容几乎一样 → 结论必然连续相同；恒定输入上的 softmax 又往往很尖 →
 * 置信度也过得了门限。于是"没在打手语"这个状态反而是最稳、最自信的误报源。
 * 表现出来就是刚点开始翻译、还没起手，历史里就先蹦出几个固定的词。
 *
 * ===== 判据与门限 =====
 *
 * 用的就是主手判定那三路运动能量（`handEnergy`：弯折 σ÷标定量程、指压 σ、
 * 相对首帧最大转角），取**两只手里较大的那个** —— 单手词只有一只手动，
 * 用平均或用两手都超门限会把单手词全部误杀。
 *
 * 门限直接复用 `IDLE_ENERGY`。这不是图省事，是因为两处问的是同一个物理问题
 * （"这只手在动吗"），而且这个数在真实数据上有实测依据：数据体检把
 * `judgeSampleDominance` 用同一个门限跑过全部 399 条真实录制，判成 `idle` 的只有
 * 7 条 —— 4 条 `_idle` 伪类 + 3 条确认的空录，**392 条真手势没有一条落到门限以下**。
 * 也就是说这个门限在"真手势"和"什么都没发生"之间是干净分开的。
 *
 * ===== 刻意没有做的事 =====
 *
 * 没有滞回。主手判定需要滞回，因为中途翻转会拼出训练集里不存在的输入；闸门不会，
 * 它只是"这一跳不出词"，误关一跳的代价仅仅是晚 100ms 出词，而误开一跳就是一个错词
 * 进历史。这里刻意做成**宁可晚出词**。
 *
 * 也不管静态单帧模式。那条路每次只读最新一帧，窗口内运动能量对它没有定义；
 * 单帧模型本来也识别不了动态词，是另一个问题。
 */
import {
  IDLE_ENERGY,
  sampleEnergies,
  type BendRanges,
  type HandEnergy,
} from "./dominantHand";
import type { SequenceSample } from "./datasetStore";

export interface MotionGateVerdict {
  /** true = 这一窗有动作，放行推理；false = 静止，这一跳不该出词 */
  moving: boolean;
  /** 两只手里较大的那个能量（份）。1.0 ≈ 一个明显的有意动作 */
  peak: number;
  left: HandEnergy | null;
  right: HandEnergy | null;
  /** 弯折能量是否用上了两只手都齐的真实两点标定，见 `sampleEnergies` */
  calibrated: boolean;
}

/**
 * 判一个滑窗快照里有没有动作。
 *
 * 要传**归一化（镜像）之前**的原始快照：能量要除以各自那只手的标定量程，
 * 镜像之后左手的数据配的是右手的量程，分母就错了。
 */
export function judgeWindowMotion(
  sample: SequenceSample,
  ranges: BendRanges
): MotionGateVerdict {
  const { left, right, calibrated } = sampleEnergies(sample, ranges);
  // 取较大者而非平均：单手词只有一只手在动，平均会把它对半砍到门限以下
  const peak = Math.max(left?.total ?? 0, right?.total ?? 0);
  return { moving: peak >= IDLE_ENERGY, peak, left, right, calibrated };
}
