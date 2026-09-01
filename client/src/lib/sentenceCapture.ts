/*
 * sentenceCapture — 整句捕获的状态机。
 *
 * 连续手语不做 100ms 滑窗推理，而是"整段捕获 + 一次性解码"：整句打完再解，
 * CTC 才能看到完整的词序列。所以需要判断"这一句什么时候结束"。
 *
 * 抽成独立模块而不是写在 Translate.tsx 里，是因为这里全是时序判据（要先见到动作、
 * 静止多久算收句、上限多长），埋在 1248 行的页面组件里没法测，而它每一条判错的
 * 表现都是"整句丢了"或者"刚起手就被收掉"，靠手动试很难复现。
 *
 * ===== 四个状态 =====
 *
 *   idle      → 没在录
 *   armed     → 按了「开始一句」，但**还没见到动作**。这个状态必须存在：
 *               按下按钮的那一刻手一定是静止的，没有 armed 的话
 *               "静止 800ms 即收句"会在起手之前就把句子收掉，永远录不到东西。
 *   capturing → 见过动作了，正在录。静止计时从这里开始才有意义。
 *   settling  → 正在静止，但还没到收句门限。中途又动了就退回 capturing
 *               （词与词之间的过渡有短暂停顿，不能一停就收）。
 *
 * ===== 为什么不复用逐词模式那套平滑窗口 =====
 *
 * 逐词模式是「连续 N 次判定相同 + 同词至少间隔 2 秒」。那套在句子里是错的：
 * 间隔 2 秒的去重让「我 爱 我」打不出来，而 CTC 本来就靠 blank 区分重复词。
 * 这个模块只管切段，词序列完全交给 CTC。
 */

/**
 * 收句判据：连续静止多久算一句结束。
 *
 * 不要改它 —— `/collect-sentence` 靠它决定录制的时间包络，改了之后新录的样本
 * 和存量 90 条句子就是两种口径。
 */
export const SETTLE_MS = 800;

/**
 * **连续模式**的收句基线（一句接一句、不用点按钮）。
 *
 * 明显长于 `SETTLE_MS`：一句一次模式里"收早了"最多是这一句被切两半，用户看得见、
 * 再点一次就行；连续模式下没人盯着按钮，切错会连着错下去。而且判"静止"本身滞后一个
 * 600ms 探测窗（见 `sentenceEnvelope.SENTENCE_MOTION_WINDOW_MS`），所以体感收句延迟
 * ≈ 1200 + 600 = 1.8s。宁可晚收句 —— 早收句是丢词，晚收句只是多等一会儿。
 *
 * ⚠ 1200 是**估的**，没有实测支撑。面板上会把实测的静止时长和生效门限一起显示出来，
 * 按现场读数再定。
 */
export const CONTINUOUS_SETTLE_MS = 1200;

/**
 * 自适应能把门限抬到的上限。再长不如让人点「结束」。
 *
 * 只封住**自适应加出来的那部分**，不封基线：`effSettleMs` 里基线永远是下界，
 * 否则谁把基线设得比这个大，门限会被无声地调小。
 */
export const SETTLE_CEIL_MS = 3000;

/**
 * 单句上限。超过就强制收句。
 *
 * 12s 与 `SequenceWindowBuffer` 的 `bufferMs` 一致：缓冲装不下的部分会被丢掉，
 * 让上限大于缓冲只会让开头那几秒静默消失（而不是报错），比截断更难发现。
 */
export const MAX_UTTERANCE_MS = 12000;

/**
 * 起手超时。armed 状态等这么久还没见到动作就自动放弃。
 *
 * 没有这个的话，按了「开始一句」再走开，缓冲会一直转，回来后按「结束」会拿到
 * 一段十几秒的静止 —— 解出来是空句或者乱句。
 */
export const ARM_TIMEOUT_MS = 8000;

export type CaptureState = "idle" | "armed" | "capturing" | "settling";

/** tick 的结论。`decode` 是唯一会触发推理的那个 */
export type CaptureAction =
  | { kind: "none" }
  /** 该收句了：调用方去 snapshotAll + predictSentence */
  | { kind: "decode"; reason: "settled" | "maxLength" | "manual" }
  /** 放弃这一句（没录到任何动作），调用方给个提示 */
  | { kind: "abort"; reason: "armTimeout" | "manual" | "noMotion" };

export interface CaptureStatus {
  state: CaptureState;
  /** 已录时长（ms）；idle/armed 时为 0 */
  elapsedMs: number;
  /** settling 状态下已静止多久；其余状态为 0 */
  stillMs: number;
  /** 距离收句还差多少（settling 时用来画进度条）；其余状态为 null */
  settleRemainMs: number | null;
  /**
   * **当前生效**的收句门限（自适应之后）。进度条和读数必须用它，不能用常量 ——
   * 门限抬高之后拿常量画的条会填满后卡在 100% 干等，看着像卡死了。
   */
  settleMs: number;
  /** 本句里见过的最长"假句尾"（词间犹豫）；0 = 一次都没犹豫过 */
  maxIntraStillMs: number;
}

export interface SentenceCaptureOptions {
  settleMs?: number;
  maxUtteranceMs?: number;
  armTimeoutMs?: number;
  /** 见 `setSettle`。默认 false —— 采集页和一句一次模式的行为逐位不变 */
  adaptiveSettle?: boolean;
}

/**
 * 用法：`arm(now)` → 每 100ms `tick(now, moving)` → 拿到 `decode` 就去解码。
 *
 * **不持有任何数据**，只看 `moving` 这一个布尔量（来自 `judgeWindowMotion`）。
 * 数据在 `SequenceWindowBuffer` 里，收句时由调用方一次性取出。
 */
export class SentenceCapture {
  private state: CaptureState = "idle";
  private armedAt = 0;
  private startedAt = 0;
  private stillSince = 0;
  private lastNow = 0;
  /** 收句门限的**基线**。不再 readonly —— 连续模式开关要在运行时改（见 `setSettle`） */
  private settleMs: number;
  private adaptive: boolean;
  private readonly maxMs: number;
  private readonly armTimeoutMs: number;
  /**
   * 本句里见过的最长"假句尾"：进了 `settling` 又因为动作回到 `capturing` 的那段时长。
   * 句尾门限要比它更长，否则同样的犹豫会把下一句切两半。
   *
   * 只反映**真正的长犹豫**：`moving` 有 600ms 探测滞后，短于那个窗的词间保持
   * 根本进不了 `settling`，所以不会被这里记上。
   */
  private maxIntraStillMs = 0;
  /**
   * 上一次 settled 收句**实际用掉**的门限，`settleDropMs` 读它。
   *
   * 为什么不能直接返回 `this.settleMs`：门限一旦是动态的，尾部就会**少砍**一段静止。
   * 整段被重采样到定长 128 帧，尾部静止越长，每个词分到的帧越少 —— 这个 bug
   * 能编译、能跑，只表现为"准确率稍微差一点"。
   *
   * 初值给基线而不是 0：万一 `settleDropMs` 拿到一个不是 tick 返回的 action
   * （手工构造），砍基线等于今天的行为，砍 0 才是那个静默 bug。
   */
  private lastSettleUsedMs: number;

  constructor(opts: SentenceCaptureOptions = {}) {
    this.settleMs = opts.settleMs ?? SETTLE_MS;
    this.adaptive = opts.adaptiveSettle ?? false;
    this.maxMs = opts.maxUtteranceMs ?? MAX_UTTERANCE_MS;
    this.armTimeoutMs = opts.armTimeoutMs ?? ARM_TIMEOUT_MS;
    this.lastSettleUsedMs = this.settleMs;
  }

  /**
   * 运行时改收句门限。**只影响下一跳的判据，不打断当前这一句、不清 `maxIntraStillMs`。**
   *
   * 需要它是因为连续模式是个 checkbox，而 `SentenceEnvelope` 在页面里只 new 一次。
   * 不清 `maxIntraStillMs`：那是"本句"的观测量，中途改开关不该让它失忆。
   */
  setSettle(baseMs: number, adaptive: boolean): void {
    this.settleMs = baseMs;
    this.adaptive = adaptive;
  }

  /**
   * 当前生效的门限。
   *
   * 自适应规则：句尾静止要比**本句里已经见过的最长假句尾**更长（×1.4 + 200ms 余量）。
   * 基线永远是下界，上限只封自适应加出来的那部分。
   */
  private effSettleMs(): number {
    if (!this.adaptive) return this.settleMs;
    const want = Math.min(SETTLE_CEIL_MS, this.maxIntraStillMs * 1.4 + 200);
    return Math.max(this.settleMs, want);
  }

  /** 按「开始一句」。已经在录的话重新开始（等于放弃当前这句）。 */
  arm(now: number): void {
    this.state = "armed";
    this.armedAt = now;
    this.startedAt = 0;
    this.stillSince = 0;
    this.lastNow = now;
    // 上一句的犹豫不该抬高下一句的门限：不同句子的停顿结构没有关系
    this.maxIntraStillMs = 0;
    this.lastSettleUsedMs = this.settleMs;
  }

  /** 按「结束」手动收句。没录到动作时返回 abort —— 空段送去解码只会解出乱句 */
  finish(): CaptureAction {
    const had = this.state === "capturing" || this.state === "settling";
    this.state = "idle";
    return had
      ? { kind: "decode", reason: "manual" }
      : { kind: "abort", reason: "noMotion" };
  }

  /** 直接丢弃当前这句（切 MODE、停止翻译、手套掉线） */
  cancel(): void {
    this.state = "idle";
    this.startedAt = 0;
    this.stillSince = 0;
  }

  /**
   * 推进一跳。`moving` 来自 `judgeWindowMotion`（看最近一小段有没有动作）。
   *
   * 返回 `decode` 时状态**已经回到 idle**：调用方拿到之后去解码，
   * 不需要（也不应该）再调 finish。重复触发解码会让同一句解两遍。
   */
  tick(now: number, moving: boolean): CaptureAction {
    this.lastNow = now;
    switch (this.state) {
      case "idle":
        return { kind: "none" };

      case "armed":
        if (moving) {
          // 录制起点是**见到动作的这一刻**，不是按按钮的那一刻。
          // 用按按钮的时刻会把起手前的静止一起送进去，那段在训练数据里不存在
          this.state = "capturing";
          this.startedAt = now;
          return { kind: "none" };
        }
        if (now - this.armedAt >= this.armTimeoutMs) {
          this.state = "idle";
          return { kind: "abort", reason: "armTimeout" };
        }
        return { kind: "none" };

      case "capturing":
        if (now - this.startedAt >= this.maxMs) {
          this.state = "idle";
          return { kind: "decode", reason: "maxLength" };
        }
        if (!moving) {
          this.state = "settling";
          this.stillSince = now;
        }
        return { kind: "none" };

      case "settling":
        if (moving) {
          // 又动了 —— 刚才那段静止是词间过渡，不是句尾。计时作废，
          // 但**记一笔**：本句里出现过这么长的假句尾，句尾门限要比它更长
          this.maxIntraStillMs = Math.max(
            this.maxIntraStillMs,
            now - this.stillSince
          );
          this.state = "capturing";
          this.stillSince = 0;
          return { kind: "none" };
        }
        if (now - this.stillSince >= this.effSettleMs()) {
          // latch：`settleDropMs` 必须砍掉**实际用掉**的这个值，不是基线
          this.lastSettleUsedMs = this.effSettleMs();
          this.state = "idle";
          return { kind: "decode", reason: "settled" };
        }
        // 上限判据在 settling 里也要查：一直静止不动也不能无限等下去
        if (now - this.startedAt >= this.maxMs) {
          this.state = "idle";
          return { kind: "decode", reason: "maxLength" };
        }
        return { kind: "none" };
    }
  }

  status(): CaptureStatus {
    const capturing = this.state === "capturing" || this.state === "settling";
    const eff = this.effSettleMs();
    return {
      state: this.state,
      elapsedMs: capturing ? Math.max(0, this.lastNow - this.startedAt) : 0,
      stillMs:
        this.state === "settling" ? Math.max(0, this.lastNow - this.stillSince) : 0,
      settleRemainMs:
        this.state === "settling"
          ? Math.max(0, eff - (this.lastNow - this.stillSince))
          : null,
      settleMs: eff,
      maxIntraStillMs: this.maxIntraStillMs,
    };
  }

  /**
   * 收句时该向缓冲要多长的一段。
   *
   * **要把收句用掉的那段静止减掉**：settling 判定成立时，最后 `settleMs` 是纯静止，
   * 送进模型只会占掉输出帧（T=128 → 32 输出帧，800ms 静止约占 2 帧）。
   * 更要紧的是它挤掉了真正有词的时间比例 —— 整段被重采样到定长 128 帧，
   * 尾部静止越长，每个词分到的帧越少。
   */
  captureSpanMs(action: CaptureAction): number {
    const raw = Math.max(0, this.lastNow - this.startedAt);
    return Math.min(this.maxMs, Math.max(0, raw - this.settleDropMs(action)));
  }

  /**
   * 那段静止要从缓冲的**尾部**砍掉多少。
   *
   * 必须和 `captureSpanMs` 配对用：`snapshotAll(captureSpanMs(a), label, settleDropMs(a))`。
   * 只调小 span 是不够的 —— 缓冲的区间是贴着**尾部**对齐的，调小 span 砍掉的是句子
   * 开头那几个词，而尾部那段静止照样留着。两个值一起传才是"掐掉尾巴、保住开头"。
   */
  settleDropMs(action: CaptureAction): number {
    return action.kind === "decode" && action.reason === "settled"
      ? this.lastSettleUsedMs
      : 0;
  }
}
