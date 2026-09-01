/*
 * signTransitions —— 从视觉关键点量出一条真实句子录制里的「保持段 / 过渡段」
 *
 * ===== 这个模块要回答的问题 =====
 *
 * 合成句子是把孤立词录制交叉淡化拼起来的，接缝长度由 `synth_sentences.py` 的
 * `overlap_ms` 决定。它现在是 **(100, 250)ms**，而这个区间是**估的** —— 批次 A 里
 * 把它从 (300,600) 改小时的理由只是"旧值是为吃掉起手/收尾静止定的，静止已经在裁剪
 * 阶段去掉了"，新值写的是"对应真人协同发音的过渡时长"，但那时候一条真实句子都没有，
 * 没有任何测量支撑。
 *
 * 真人连着打一句话时，手在"打第 k 个词的位置/手型"上相对稳住一小会儿，然后**移动**
 * 到第 k+1 个词 —— 那段移动就是协同发音，也就是 `overlap_ms` 该对齐的东西。
 * 关键点看得见这段移动，触觉看不见（弯折和压力在过渡期照样在变，分不出"在打词"
 * 还是"在换词"）。这是句子采集开摄像头的**主要**用途。
 *
 * ===== 为什么不是"找 K−1 个峰" =====
 *
 * 已知句型有 K 个词，看起来应该找 K−1 个速度峰当词边界。不这么做的理由：
 * 那等于**先假定测量结果**。真人可能把两个词连成一个动作、也可能在中间多顿一下，
 * 强行取最大的 K−1 个峰永远能给出答案，且看不出答案是错的。
 *
 * 这里改成**无先验的保持/过渡分段**，再把段数和句型词数**对照**：
 *   - 保持段数 ≈ 词数  → 分段可信，过渡时长可以拿去校准 `overlap_ms`
 *   - 对不上          → 如实报"对不上"，这一条不参与校准
 * 对不上本身也是有用的读数：保持段偏少 = 词粘在一起（可能漏词），偏多 = 中间停顿了。
 *
 * ===== 三条守则 =====
 *
 * - **不进特征。** 这里算出来的东西只用于界面读数和标定 `overlap_ms`，
 *   一个数都不会进模型输入（`buildSequenceFeatures` 的 `includeVision` 默认 false，
 *   `synth_sentences.py` 合成时把视觉整个丢掉）。部署端没有摄像头，进了就上不了机。
 * - **不写数据库。** 与 `sequenceTrim` 同一条：判据改了重算一遍即可。
 * - **量不出来就说量不出来**，不给近似值。每种失败都有 `reason`，界面上照实显示。
 */
import { SEQ_LANDMARK_N, type SequenceSample } from "@/lib/datasetStore";

export interface SignTransitionConfig {
  /**
   * 过渡门限在速度分布里的位置：`thr = p10 + ratio·(p90 − p10)`。
   *
   * 用分位数而不是绝对速度：手离摄像头远近、画面裁切都会整体缩放归一化坐标里的
   * 位移，绝对门限会随人坐得远近而失效。用 p10/p90 而不是 min/max：单帧误检的
   * 关键点会把 max 拉到天上，门限跟着废掉。
   */
  moveRatio: number;
  /** 短于此的保持段并进相邻过渡段（关键点抖动会在真过渡中间造出一两帧假"静止"） */
  minHoldMs: number;
  /** 短于此的过渡段并进相邻保持段 */
  minMoveMs: number;
  /** 速度平滑窗口帧数（奇数）。30fps 下关键点抖动足以造出假峰谷 */
  smoothFrames: number;
  /**
   * p90/p10 的比值下限。低于此说明整条速度曲线没有动态范围 ——
   * 要么整条都在动（一路划过去，没有保持段），要么整条都没动（空录）。
   * 这种条上做分段只是在切噪声，直接判 `no_contrast`。
   */
  minContrast: number;
}

export const DEFAULT_TRANSITION_CONFIG: SignTransitionConfig = {
  moveRatio: 0.35,
  minHoldMs: 80,
  minMoveMs: 60,
  smoothFrames: 3,
  minContrast: 1.8,
};

export type TransitionReason =
  | "ok"
  /** 两只手都没有关键点数组（这一条录的时候没开摄像头） */
  | "no_vision"
  /** 有关键点数组，但连续看得见手的帧太少，算不出速度曲线 */
  | "no_track"
  /** 速度曲线没有动态范围，分段没有意义 */
  | "no_contrast";

export interface Segment {
  startFrame: number;
  /** 不含 */
  endFrame: number;
  startMs: number;
  endMs: number;
  durationMs: number;
}

export interface SignTransitions {
  reason: TransitionReason;
  /**
   * 逐帧速度（单位：手宽/秒）。第 0 帧恒为 0（没有前一帧）。
   * 看不见手的帧填 NaN —— 那些帧不参与分段，也不该被画成 0（0 是"没动"）。
   */
  speed: Float32Array;
  /** 相对稳住的段。词大致落在这些段上 */
  holds: Segment[];
  /** 段间移动。**这就是协同发音的过渡**，`overlap_ms` 要对齐的量 */
  moves: Segment[];
  /** 过渡段时长中位数；没有过渡段时为 0 */
  medianMoveMs: number;
  /** 有效（看得见手）的帧数 */
  trackedFrames: number;
}

const EMPTY = (reason: TransitionReason, speed: Float32Array, tracked: number) =>
  ({
    reason,
    speed,
    holds: [],
    moves: [],
    medianMoveMs: 0,
    trackedFrames: tracked,
  }) satisfies SignTransitions;

/**
 * 一帧里"手有多大"—— 手腕(0) 到中指根(9) 的距离。
 *
 * 速度要除以它：同一个动作，手离摄像头近时归一化位移大、远时小，不除就成了
 * "坐得越近打得越快"。除以手自身的尺度之后，单位是**手宽/秒**，跨录制可比。
 */
function handScale(lm: Float32Array, o: number): number {
  const dx = lm[o + 9 * 3] - lm[o];
  const dy = lm[o + 9 * 3 + 1] - lm[o + 1];
  return Math.hypot(dx, dy);
}

/** 这一帧这只手看得见吗（与 `sequenceTrim.handVisibleAt` 同一条判据：首尾有限） */
function visibleAt(lm: Float32Array | null, t: number): boolean {
  if (!lm) return false;
  const o = t * SEQ_LANDMARK_N;
  if (o + SEQ_LANDMARK_N > lm.length) return false;
  return Number.isFinite(lm[o]) && Number.isFinite(lm[o + SEQ_LANDMARK_N - 1]);
}

/**
 * 一只手相邻两帧之间的平均关键点位移 / 手宽。看不见（任一帧）就返回 NaN。
 *
 * 取 **21 点的平均**而不是只看手腕：手语里有"位置不动、只换手型"的相邻词
 * （比如同一个位置上从张开变成捏合），只看手腕会把那种过渡整个漏掉。
 */
function frameStep(lm: Float32Array, t: number): number {
  const a = (t - 1) * SEQ_LANDMARK_N;
  const b = t * SEQ_LANDMARK_N;
  const scale = (handScale(lm, a) + handScale(lm, b)) / 2;
  if (!(scale > 1e-6)) return NaN;
  let sum = 0;
  for (let i = 0; i < 21; i++) {
    const dx = lm[b + i * 3] - lm[a + i * 3];
    const dy = lm[b + i * 3 + 1] - lm[a + i * 3 + 1];
    // z 不参与：MediaPipe 的 z 是相对深度、量级与 xy 不同源，混进来只是加噪声
    sum += Math.hypot(dx, dy);
  }
  return sum / 21 / scale;
}

/** 分位数（线性插值）。入参会被排序，传副本进来 */
function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return quantile(s, 0.5);
}

/**
 * 逐帧速度（手宽/秒）。双手取**较大者** —— 单手词里闲着的那只手会把平均值压下去，
 * 而"至少有一只手在动"才是过渡的判据。
 */
export function landmarkSpeed(
  sample: SequenceSample,
  smoothFrames = DEFAULT_TRANSITION_CONFIG.smoothFrames
): { speed: Float32Array; trackedFrames: number } {
  const T = sample.frameCount;
  const ts = sample.timestamps;
  const raw = new Float32Array(T).fill(NaN);
  let tracked = 0;

  for (let t = 0; t < T; t++) {
    const lv = visibleAt(sample.leftLandmarks, t);
    const rv = visibleAt(sample.rightLandmarks, t);
    if (lv || rv) tracked++;
    if (t === 0) {
      if (lv || rv) raw[0] = 0;
      continue;
    }
    const dtMs = ts[t] - ts[t - 1];
    if (!(dtMs > 1e-6)) continue;
    let best = NaN;
    if (lv && visibleAt(sample.leftLandmarks, t - 1)) {
      best = frameStep(sample.leftLandmarks!, t);
    }
    if (rv && visibleAt(sample.rightLandmarks, t - 1)) {
      const r = frameStep(sample.rightLandmarks!, t);
      best = Number.isFinite(best) ? Math.max(best, r) : r;
    }
    if (Number.isFinite(best)) raw[t] = (best * 1000) / dtMs;
  }

  // 平滑：只在**都有效**的邻居上做，NaN 不参与也不传染
  const half = Math.max(0, Math.floor(smoothFrames / 2));
  if (half === 0) return { speed: raw, trackedFrames: tracked };
  const out = new Float32Array(T).fill(NaN);
  for (let t = 0; t < T; t++) {
    if (!Number.isFinite(raw[t])) continue;
    let sum = 0;
    let n = 0;
    for (let k = t - half; k <= t + half; k++) {
      if (k < 0 || k >= T || !Number.isFinite(raw[k])) continue;
      sum += raw[k];
      n++;
    }
    out[t] = sum / n;
  }
  return { speed: out, trackedFrames: tracked };
}

/**
 * 把一条录制切成保持段与过渡段。
 *
 * 分段只在**连续可见**的区间内进行：中间视觉丢了的那些帧既不算保持也不算过渡
 * （不知道那时候手在干什么，猜一个就是编数据）。所以段落可能不连成一片，
 * 这是有意的 —— `holds.length + moves.length` 不保证等于总段数。
 */
export function detectSignTransitions(
  sample: SequenceSample,
  cfg: SignTransitionConfig = DEFAULT_TRANSITION_CONFIG
): SignTransitions {
  const T = sample.frameCount;
  const { speed, trackedFrames } = landmarkSpeed(sample, cfg.smoothFrames);

  if (!sample.leftLandmarks && !sample.rightLandmarks) {
    return EMPTY("no_vision", speed, 0);
  }
  const valid: number[] = [];
  for (let t = 0; t < T; t++) if (Number.isFinite(speed[t])) valid.push(speed[t]);
  // 4 帧以下连速度曲线都算不出，谈不上分段
  if (valid.length < 4) return EMPTY("no_track", speed, trackedFrames);

  const sorted = [...valid].sort((a, b) => a - b);
  const p10 = quantile(sorted, 0.1);
  const p90 = quantile(sorted, 0.9);
  // p10 可能是 0（真的完全静止），比值判据要防除零
  if (!(p90 > 1e-6) || p90 / Math.max(p10, 1e-6) < cfg.minContrast) {
    return EMPTY("no_contrast", speed, trackedFrames);
  }
  const thr = p10 + cfg.moveRatio * (p90 - p10);

  // ===== 一趟扫出交替的段落 =====
  const ts = sample.timestamps;
  type Raw = { moving: boolean; start: number; end: number };
  const segs: Raw[] = [];
  let cur: Raw | null = null;
  for (let t = 0; t < T; t++) {
    if (!Number.isFinite(speed[t])) {
      cur = null; // 视觉断了：段落到此为止，不跨过空洞连起来
      continue;
    }
    const moving = speed[t] > thr;
    if (cur && cur.moving === moving && cur.end === t) {
      cur.end = t + 1;
    } else {
      cur = { moving, start: t, end: t + 1 };
      segs.push(cur);
    }
  }

  // ===== 并掉太短的段 =====
  // 反复扫到不再变化为止：并掉一段之后，它两侧的同类段会连起来，可能又够长了
  const msOf = (s: Raw) => ts[Math.min(s.end, T - 1)] - ts[s.start];
  let merged = segs;
  for (let pass = 0; pass < 8; pass++) {
    const tooShort = merged.findIndex((s) =>
      msOf(s) < (s.moving ? cfg.minMoveMs : cfg.minHoldMs)
    );
    if (tooShort < 0) break;
    const next: Raw[] = [];
    for (let i = 0; i < merged.length; i++) {
      const s = merged[i];
      if (i === tooShort) {
        // 并进上一段（没有上一段就翻转自己的类别交给下一轮合并）
        const prev = next[next.length - 1];
        if (prev && prev.end === s.start) {
          prev.end = s.end;
          continue;
        }
        next.push({ ...s, moving: !s.moving });
        continue;
      }
      const prev = next[next.length - 1];
      if (prev && prev.moving === s.moving && prev.end === s.start) {
        prev.end = s.end;
        continue;
      }
      next.push({ ...s });
    }
    merged = next;
  }

  const toSegment = (s: Raw): Segment => ({
    startFrame: s.start,
    endFrame: s.end,
    startMs: ts[s.start],
    endMs: ts[Math.min(s.end, T - 1)],
    durationMs: msOf(s),
  });
  const holds = merged.filter((s) => !s.moving).map(toSegment);
  const moves = merged.filter((s) => s.moving).map(toSegment);

  return {
    reason: "ok",
    speed,
    holds,
    moves,
    medianMoveMs: median(moves.map((m) => m.durationMs)),
    trackedFrames,
  };
}

export type TransitionVerdictKind =
  /** 保持段数与词数一致，过渡时长可信 */
  | "match"
  /** 保持段比词少 —— 词粘在一起了，可能漏词 */
  | "fewer"
  /** 保持段比词多 —— 中间多顿了一下 */
  | "more"
  /** 量不出来（见 `SignTransitions.reason`） */
  | "unmeasured";

export interface TransitionVerdict {
  kind: TransitionVerdictKind;
  holdCount: number;
  wordCount: number;
  medianMoveMs: number;
  /** 界面上直接显示的一句话；`match` 时也有（报过渡时长） */
  note: string;
}

/**
 * 把分段结果与句型词数对照。**这是这个模块唯一的验收判据** ——
 * 过渡时长只有在保持段数对得上的那些条上才拿去校准 `overlap_ms`。
 *
 * 允许差 1 段：句尾最后一个词后面通常不再有移动，句首起手也可能直接落在词上，
 * 两端各差一段是常态，不该报成异常。
 */
export function transitionVerdict(
  tr: SignTransitions,
  wordCount: number
): TransitionVerdict {
  const holdCount = tr.holds.length;
  const base = { holdCount, wordCount, medianMoveMs: tr.medianMoveMs };
  if (tr.reason !== "ok") {
    const why =
      tr.reason === "no_vision"
        ? "这一条没有视觉（录的时候没开摄像头）"
        : tr.reason === "no_track"
          ? "视觉里几乎没检出手，算不出速度曲线"
          : "速度曲线没有动态范围 —— 要么一路划过去没有停顿，要么整条都没动";
    return { ...base, kind: "unmeasured", note: `过渡时长量不出来：${why}` };
  }
  if (Math.abs(holdCount - wordCount) <= 1) {
    return {
      ...base,
      kind: "match",
      note: `保持段 ${holdCount} / 词 ${wordCount}，过渡中位 ${Math.round(tr.medianMoveMs)}ms`,
    };
  }
  if (holdCount < wordCount) {
    return {
      ...base,
      kind: "fewer",
      note: `只看到 ${holdCount} 个保持段、句型有 ${wordCount} 个词 —— 词粘在一起了，可能漏词`,
    };
  }
  return {
    ...base,
    kind: "more",
    note: `看到 ${holdCount} 个保持段、句型只有 ${wordCount} 个词 —— 中间多顿了一下，连续手语不该有停顿`,
  };
}

/**
 * 一批录制的过渡时长汇总 —— **`overlap_ms` 的实测依据就是这个数**。
 *
 * 只吃 `kind === "match"` 的条：分段与词数对不上时，那条的过渡时长是在切噪声。
 * 返回 `usable` 让调用方能如实说"N 条里只有 M 条可用"，而不是把中位数当成全体。
 */
export function summarizeTransitions(
  items: readonly { transitions: SignTransitions; wordCount: number }[]
): {
  total: number;
  usable: number;
  medianMoveMs: number;
  p10MoveMs: number;
  p90MoveMs: number;
} {
  const all: number[] = [];
  let usable = 0;
  for (const it of items) {
    if (transitionVerdict(it.transitions, it.wordCount).kind !== "match") continue;
    usable++;
    for (const m of it.transitions.moves) all.push(m.durationMs);
  }
  const sorted = all.sort((a, b) => a - b);
  return {
    total: items.length,
    usable,
    medianMoveMs: sorted.length ? quantile(sorted, 0.5) : 0,
    p10MoveMs: sorted.length ? quantile(sorted, 0.1) : 0,
    p90MoveMs: sorted.length ? quantile(sorted, 0.9) : 0,
  };
}
