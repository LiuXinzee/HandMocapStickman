/*
 * sentenceEnvelope —— 把「环形缓冲 + 收句状态机」绑成一件东西。
 *
 * ===== 为什么需要这个模块 =====
 *
 * 一条句子样本的**时间包络**（从哪一刻算起、到哪一刻为止）由三处协同决定：
 *   1. 起点：`SentenceCapture` 在 `armed` 状态下见到第一次 `moving` 才开始计时
 *      （不是按按钮那一刻 —— 那会把起手前的静止一起录进去）
 *   2. 终点：停手收句时要把尾部 `SETTLE_MS` 的静止掐掉
 *   3. 取数：`snapshotAll(captureSpanMs(act), label, settleDropMs(act))`
 *
 * 第 3 条的两个参数**必须成对传**。缓冲区间是贴着尾部对齐的，只调小 span 砍掉的是
 * 句子**开头**那几个词，尾部静止照样留着（见 `sentenceCapture.ts` 的
 * `settleDropMs` 注释）。这是个能编译、能跑、结果看着正常的错误。
 *
 * `/translate`（推理）和 `/collect-sentence`（采集）必须产生**同一种包络**：
 * 采集时多录进去的一段静止，会在重采样到定长 T 时把每个词在归一化时间轴上的位置
 * 整体挪位 —— 和之前修掉的 127× 接缝尖峰是同一类系统性人造特征，合成/真实 val
 * 都看不出来，只有戴上手套才发现打什么都不准。
 *
 * 所以两边不能各写一份 tick + snapshotAll，而要共用**这一个** `take()`：
 * 参数对不上的可能性从"靠人记得"变成"编译期不存在"。
 *
 * ===== 为什么是 class 而不是 React hook =====
 *
 * `/translate` 的 100ms 循环还要做主手判定和另外两档的滑窗推理，缓冲是三档共用的
 * （所以这里**不创建**缓冲，只接管一个现成的）。把定时器和 state 也吞进来的话，
 * 那个循环就得拆开重排，风险远大于收益。这里只抽"纯时序逻辑"这一层，
 * 它不碰 React、不碰 DOM，可以直接单测。
 */
import { judgeWindowMotion } from "./motionGate";
import type { BendRanges } from "./dominantHand";
import {
  SentenceCapture,
  type CaptureAction,
  type CaptureStatus,
} from "./sentenceCapture";
import type { SequenceWindowBuffer } from "./sequenceWindow";
import type { SequenceSample } from "./datasetStore";

/**
 * 判"手还在动吗"用的滑窗长度。**比逐词档的推理窗口（2000ms）短得多，这是必需的。**
 *
 * 句尾判据是「连续静止 `SETTLE_MS`（800ms）」。如果拿 2000ms 的推理窗口去判，
 * 窗口里那 800ms 静止会被前面 1200ms 的动作盖住 —— `judgeWindowMotion` 量的是
 * 整窗的活动量，永远判不出句尾，表现为"打完不收句、一直录到 12s 上限"。
 *
 * 反过来也不能太短：窗口是滑动重叠的，600ms 窗要"整窗都静止"才判静止，
 * 于是实际收句延迟约 600+800 = 1.4s，比 0.8s 长。这个方向是安全的（宁可晚收句，
 * 不要把人打到一半掐掉），所以选择接受它，而不是去调低 `IDLE_ENERGY` 门限 ——
 * 那个门限是两条老链路共用的，为句子模式动它会顺带改坏逐词识别。
 *
 * 同一个滞后也出现在起手那头：探测窗攒满之前判不出 `moving`，所以句子开头约 600ms
 * 不会进样本。采集端和推理端走同一条路、滞后一样多，所以不是包络错位
 * （`sentenceEnvelope.test.ts` 里量的是两种取法之差，不是绝对值）。
 */
export const SENTENCE_MOTION_WINDOW_MS = 600;

/**
 * 用法：
 *   const env = new SentenceEnvelope(buf);
 *   env.arm(performance.now());                       // 清缓冲 + 进 armed
 *   const act = env.tick(performance.now(), ranges);  // 每 100ms
 *   if (act.kind === "decode") { const s = env.take(act, "_sentence"); ... }
 */
export class SentenceEnvelope {
  readonly capture: SentenceCapture;

  /**
   * @param buf 环形缓冲。**由调用方持有**（`/translate` 三档共用一个），
   *            所以这里只借用不创建。`bufferMs` 应当 ≥ `MAX_UTTERANCE_MS`。
   */
  constructor(
    private readonly buf: SequenceWindowBuffer,
    capture: SentenceCapture = new SentenceCapture()
  ) {
    this.capture = capture;
  }

  /**
   * 开始一句。
   *
   * **缓冲必须清**，而且必须在这里清、不能交给调用方：`snapshotAll` 是"有多少取多少"，
   * 不清的话上一句的尾巴会被接到这一句前面一起解码/存库。
   */
  arm(now: number): void {
    this.buf.clear();
    this.capture.arm(now);
  }

  /**
   * **连续模式**：这一句解完了，重新等下一句。
   *
   * 就是 `arm()`（**必须清缓冲**，理由见上）。单独一个名字是为了让调用点读得出
   * "这是自动的、不是用户点的" —— 手动那条路（`armSentence`）还会顺手
   * `stopSpeaking()` + 重置主手判定，这两件事在连续模式下**都不能做**：
   *   - `stopSpeaking()` 会把这一句刚开始的朗读掐掉；
   *   - 重置主手判定会让每句开头几百毫秒退回 `?? "right"` 兜底，
   *     左手用户每句话的句首都按右手口径归一化。
   * 所以别把那两行抄进这条路。
   */
  rearm(now: number): void {
    this.arm(now);
  }

  /**
   * **连续模式**：起手超时了，继续等，不当成错误。
   *
   * **不清缓冲** —— 状态机从未离开 `armed`，缓冲里只有静止，没有上一句的尾巴要防。
   * 清了反而每 8 秒造一个 600ms 的探测盲区（窗没攒满判不出 `moving`），
   * 正好在那时起手，句首会被吃掉。
   */
  keepWaiting(now: number): void {
    this.capture.arm(now);
  }

  /** 改收句门限（连续模式开关）。见 `SentenceCapture.setSettle` */
  setSettle(baseMs: number, adaptive: boolean): void {
    this.capture.setSettle(baseMs, adaptive);
  }

  /** 丢弃当前这句（切档、停止、手套掉线）。不清缓冲 —— 调用方可能还要用它做别的 */
  cancel(): void {
    this.capture.cancel();
  }

  /** 手动收句。返回 `decode` 时要立刻 `take()`：状态机已经回到 idle，下一跳不会再给 */
  finish(): CaptureAction {
    return this.capture.finish();
  }

  status(): CaptureStatus {
    return this.capture.status();
  }

  /**
   * 推进一跳。
   *
   * `now` **必须是 `performance.now()`**：帧时间戳就是 performance.now()
   * （见 `gloveProtocol.ts`），混用 `Date.now()` 会让状态机和缓冲差着一个任意大的常数。
   *
   * `ranges` 传**镜像之前**的量程（`judgeWindowMotion` 的 docstring 说明了原因：
   * 镜像后左手数据配的是右手量程，能量的分母就错了）。
   */
  tick(now: number, ranges: BendRanges): CaptureAction {
    const probe = this.buf.snapshot(SENTENCE_MOTION_WINDOW_MS, "_sentmotion");
    // 窗口还没攒满（刚 arm）时算"没动"。armed 状态下这是对的：它就是在等第一次动作，
    // 早一点晚一点只影响起手判定的那 600ms
    const moving = probe ? judgeWindowMotion(probe, ranges).moving : false;
    return this.capture.tick(now, moving);
  }

  /**
   * 取出这一句的数据段。**这是整个模块存在的理由** —— span 与 dropTail 在这里成对算出，
   * 调用方没有机会把它们拆开。
   *
   * 返回的样本：
   *   - `segments` 只有一个覆盖整段的占位（label = 传入的 label）。
   *     采集端要把它换成句型的完整标签序列，推理端不看这个字段。
   *   - `leftLandmarks` / `rightLandmarks` 是 `null`（缓冲不存视觉）。
   *     句子模型是纯触觉学生，`synth_sentences.py` 合成时也一律丢掉视觉，所以这是对的。
   *   - **没有做手别镜像归一**。推理端要在这之后自己调 `normalizeHandedness`
   *     （模型是右手口径训的）；采集端**不要**调 —— 数据集存的必须是原始录制，
   *     镜像发生在建特征那一层，存进库里就不可逆了。
   *
   * @returns null = 这一段取不出可用数据（太短，或两只手都没有帧）
   */
  take(act: CaptureAction, label: string): SequenceSample | null {
    if (act.kind !== "decode") return null;
    return this.buf.snapshotAll(
      this.capture.captureSpanMs(act),
      label,
      this.capture.settleDropMs(act)
    );
  }
}
