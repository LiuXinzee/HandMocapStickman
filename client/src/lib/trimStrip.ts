/**
 * 裁剪条的数据模型 + 帧→像素换算。
 *
 * 一条句子录制画成一根横条：底色是总帧长，上面叠两层
 *   1. **可见段**（MediaPipe 看得见手的那几段），落在保留区间外的标红
 *   2. **保留区间**（`trimSpan`，真正喂进模型的那一段）
 *
 * 原本还有第三层「词边界」，已经去掉：句子录制的词边界是占位值，画出来全部堆在
 * 最左边。见 `StripGeometry.segmentsArePlaceholder`。
 *
 * 为什么值得单独一个模块：第一层裁剪吃掉整个词的那个 bug，藏了整整一轮改动 ——
 * 而它在这根条上是**肉眼一秒可见**的（可见段 4 段，保留区间只盖住最后一段）。
 * 数字读数（`keptRatio` 0.23）看不出这件事：等太久和吃掉了词给出同一个数字。
 *
 * 换算抽成纯函数是为了能测。条画错的症状是"看着没问题"，那种错没有任何
 * 运行时报错，只有断言能拦住。
 */
import type { SequenceSegment, SeqTrimSpan } from "./datasetStore";
import type { VisibleRun } from "./sequenceTrim";

/** 一段横条上的矩形。`x`/`w` 是像素 */
export interface StripBand {
  x: number;
  w: number;
}

export interface SegmentBand extends StripBand {
  label: string;
}

export interface VisibleBand extends StripBand {
  /**
   * 这一段可见区与保留区间有没有交集。
   *
   * `false` = **整段动作被扔了**，这正是第一层取 `longest_run` 时那个 bug 的形状。
   * 部分重叠算 covered：头尾各切掉几帧是裁剪本来就该做的事。
   */
  covered: boolean;
}

/**
 * 帧下标 → 像素 x。
 *
 * `totalFrames <= 0` 时返回 0 而不是 NaN：空样本理论上进不了这里，但一个 NaN
 * 会让整个 `<svg>` 静默不渲染（属性非法直接被丢弃），而不是报错。
 */
export function frameToX(frame: number, totalFrames: number, width: number): number {
  if (totalFrames <= 0) return 0;
  const clamped = Math.max(0, Math.min(totalFrames, frame));
  return (clamped / totalFrames) * width;
}

/**
 * 半开区间 `[start, end)` → 矩形。
 *
 * **最小宽度 1 像素。** 一段只有 1~2 帧的可见段在 600px / 330 帧下宽度不到 4px，
 * 但四舍五入到 0 的话它就彻底消失了 —— 而"有一段极短的可见段"恰好是要看见的信息
 * （它说明手在这里闪进画面一下，是掉手而不是全程不可见）。
 */
export function bandFor(
  start: number,
  end: number,
  totalFrames: number,
  width: number
): StripBand {
  const x = frameToX(start, totalFrames, width);
  const x2 = frameToX(end, totalFrames, width);
  return { x, w: Math.max(1, x2 - x) };
}

export interface StripRow {
  /** IndexedDB 里的 id，作 React key */
  id: number;
  /** 句子文本，词之间空格分隔 */
  text: string;
  totalFrames: number;
  durationMs: number;
  visibleRuns: VisibleRun[];
  /** 可见帧总数（含被 minRunFrames 滤掉的），用来算"可见率" */
  visibleFrames: number;
  trimSpan: SeqTrimSpan | null;
  segments: SequenceSegment[];
  /**
   * 需要人看一眼。>1 段可见意味着摄像头中途掉了手 —— 修好之后这些条的保留区间
   * 应该跨过空洞；没跨过就是分流没生效
   */
  multiRun: boolean;
}

export interface StripGeometry {
  width: number;
  visible: VisibleBand[];
  /** `trimSpan` 为 null（导出时没算）时是 null，界面上要显式说明"没算过" */
  trim: StripBand | null;
  /** `segmentsArePlaceholder` 为 true 时是**空数组** —— 见下 */
  segments: SegmentBand[];
  /**
   * 词边界是不是占位值。
   *
   * `CollectSentence.tsx` 录句子时把每个词的 `startFrame/endFrame` 全填成
   * `[0, frameCount)`（注释里写明「这不是真词边界，只是占位」—— CTC 只读 label
   * 的顺序，不读帧下标，所以填整段是合法的训练目标）。
   *
   * 后果是**任何基于词边界的读数在句子上都测不出东西**：每个 segment 都横跨整条，
   * 永远与保留区间相交。所以这里检测出来之后 `segments` 直接给空数组 ——
   * 画出来的竖线会全部堆在最左边，比不画更误导。
   */
  segmentsArePlaceholder: boolean;
}

/** 全部 segment 都覆盖整条 → 帧下标不含信息 */
function isPlaceholderSegments(row: StripRow): boolean {
  return (
    row.segments.length > 0 &&
    row.segments.every((s) => s.startFrame <= 0 && s.endFrame >= row.totalFrames)
  );
}

export function stripGeometry(row: StripRow, width: number): StripGeometry {
  const T = row.totalFrames;
  const span = row.trimSpan;
  const trim = span ? bandFor(span.startFrame, span.endFrame, T, width) : null;
  const placeholder = isPlaceholderSegments(row);
  return {
    width,
    visible: row.visibleRuns.map((r) => ({
      ...bandFor(r.start, r.end, T, width),
      // 没算过 trimSpan 时一律算 covered：那是"不知道"，不是"被扔了"
      covered: span ? r.start < span.endFrame && r.end > span.startFrame : true,
    })),
    trim,
    segments: placeholder
      ? []
      : row.segments.map((s) => ({
          ...bandFor(s.startFrame, s.endFrame, T, width),
          label: s.label,
        })),
    segmentsArePlaceholder: placeholder,
  };
}

/** 完全落在保留区间外的可见段。这是"有整段动作被扔了"的判据 */
export function uncoveredRunsOf(row: StripRow): VisibleRun[] {
  const span = row.trimSpan;
  if (!span) return [];
  return row.visibleRuns.filter(
    (r) => r.end <= span.startFrame || r.start >= span.endFrame
  );
}

/**
 * 一批条的汇总。界面顶上那行读数用它。
 *
 * ## 为什么不是"被裁掉的词数"
 *
 * 这里原本报的是 `droppedWords`（与保留区间不相交的 segment 数），并被当成本轮
 * 改动的验收指标。**它恒为 0，测不出任何东西** —— 句子录制的词边界是占位值
 * （见 `StripGeometry.segmentsArePlaceholder`），每个 segment 都横跨整条，
 * 永远相交。一个恒为 0 的指标比没有指标更糟：它读起来像"已验证干净"。
 *
 * 换成 `uncoveredRuns`：**完全落在保留区间外的可见段数**。它不依赖词边界，
 * 只依赖 MediaPipe 的可见掩码和 `trimSpan` —— 两个都是真实测出来的量。
 * 第一层取 `longest_run` 时，T=329 那条的前 3 段可见区全在区间外；
 * 改成 `first_to_last` 之后（起点=第一段起点，终点=最后一段终点）它必须是 0。
 */
export interface StripSummary {
  total: number;
  multiRun: number;
  /** **验收指标，必须 0。** 整段可见区落在保留区间外的段数（跨所有条累计） */
  uncoveredRuns: number;
  /**
   * 落在保留区间外的可见帧数，含**部分**溢出。
   *
   * 这只是参考量，不是验收指标：`padMs` 之外的头尾静止帧被切掉是裁剪本来就该做的，
   * 所以它正常情况下也不为 0。真正要盯的是上面那个 `uncoveredRuns`。
   */
  uncoveredFrames: number;
  /** 词边界是占位值的条数。等于"这些条上任何词级读数都不成立" */
  placeholderRows: number;
  /** trimSpan 为 null 的条数（导出时没传标定 → 根本没算） */
  notComputed: number;
  reasons: Record<string, number>;
  meanKeptRatio: number | null;
}

export function summarizeStrips(rows: StripRow[]): StripSummary {
  const reasons: Record<string, number> = {};
  let uncoveredRuns = 0;
  let uncoveredFrames = 0;
  let placeholderRows = 0;
  let notComputed = 0;
  let ratioSum = 0;
  let ratioN = 0;
  for (const r of rows) {
    if (isPlaceholderSegments(r)) placeholderRows++;
    if (!r.trimSpan) {
      notComputed++;
      continue;
    }
    const span = r.trimSpan;
    reasons[span.reason] = (reasons[span.reason] ?? 0) + 1;
    ratioSum += span.keptRatio;
    ratioN++;
    uncoveredRuns += uncoveredRunsOf(r).length;
    for (const run of r.visibleRuns) {
      // 两头各算一次：一段可能左右都溢出（保留区间落在它内部）
      uncoveredFrames += Math.max(0, Math.min(run.end, span.startFrame) - run.start);
      uncoveredFrames += Math.max(0, run.end - Math.max(run.start, span.endFrame));
    }
  }
  return {
    total: rows.length,
    multiRun: rows.filter((r) => r.multiRun).length,
    uncoveredRuns,
    uncoveredFrames,
    placeholderRows,
    notComputed,
    reasons,
    // 一条都没算过时是 null，不是 0 —— 0 会被读成"全都被裁光了"
    meanKeptRatio: ratioN ? ratioSum / ratioN : null,
  };
}

/**
 * 排序：要看的排前面。
 *
 * 顺序是 有整段可见区被扔掉 → 多段可见 → 其余。翻页找问题条是这页最常做的事，
 * 按录制时间排的话那 13 条会散在 45 条里。
 */
export function sortStripRows(rows: StripRow[]): StripRow[] {
  const rank = (r: StripRow) => {
    if (uncoveredRunsOf(r).length > 0) return 0;
    if (r.multiRun) return 1;
    return 2;
  };
  // 同级内按 id 保持稳定顺序 —— 每次刷新条的次序都变会让人无法比对
  return [...rows].sort((a, b) => rank(a) - rank(b) || a.id - b.id);
}
