/*
 * datasetAudit —— 序列数据集体检。**只读，不改任何数据。**
 *
 * 为什么需要一份单独的报告，而不是在页面上多加几个数字：
 *
 * 左栏那些统计是**全局聚合**的，而这个数据集是分批采的、两批的采集方式不一样
 * （一批有冗余预备动作、一批没有；一批可能没开摄像头、一批开了）。全局平均会把
 * 这种差异整个抹平 —— 平均时长 2625ms 既可能是"两批都 2625"，也可能是
 * "老的 3600 + 新的 1600"，而这两种情况要做的清洗完全不同。
 * 所以体检必须**按批次**和**按词**分开报。
 *
 * 三件只能在这里看到的事：
 *
 * 1. **裁剪判据在每一批上分别是什么结果。** 起手段裁剪是纯视觉的
 *    （`sequenceTrim.detectSignSpan` 开头两只手都没关键点就立刻 `no_vision` 返回），
 *    没开摄像头录的那批**一刀没裁、预备动作全在数据里**，而开了摄像头的那批裁掉了。
 *    同一个词于是有两种时间口径。这是最贵的一种脏，且在全局统计里完全看不见。
 *
 * 2. **主手归一化到底有没有生效到每个词上。** 全局只报"左 N 条已镜像"，
 *    看不出某个词的 15 遍是不是真的都落进了右手口径 —— 判定不确定的那些
 *    （`nearTie`）恰恰会散在个别词上。
 *
 * 3. **时长离群。** 同一个词的 15 遍里若有几遍时长偏离中位数一倍以上，
 *    那几遍在特征空间里是"另一个速度的词"。按词看中位数才有意义，
 *    全局中位数会被词与词之间本身的长短差异淹掉。
 */
import type { SequenceSample } from "./datasetStore";
import { SEQ_SENSOR_N } from "./datasetStore";
import {
  detectSignSpan,
  leadingStillMs,
  DEFAULT_TRIM,
  type TrimReason,
} from "./sequenceTrim";
import {
  judgeSampleDominance,
  type BendRanges,
  type SampleDominance,
} from "./dominantHand";
import { isImuSuspect, analyzeSequenceImu } from "./imuHealth";
import { IDLE_LABEL } from "./signLanguageVocab";

/** 时长偏离该词中位数超过这个倍数 → 记为离群。两倍速的同一个词在特征空间里不是一个东西 */
export const DURATION_OUTLIER_RATIO = 1.5;

export type TrimReasonCounts = Record<TrimReason, number>;
export type DominanceCounts = Record<SampleDominance, number>;

export interface PerSampleAudit {
  index: number;
  label: string;
  /** 录制时刻（ms）。合成样本沿用被合成源的时间戳，所以要连 origin 一起看 */
  timestamp: number;
  /** 采集日，`YYYY-MM-DD`，批次分组用 */
  day: string;
  origin: string;
  durationMs: number;
  frameCount: number;
  hasVision: boolean;
  trimReason: TrimReason;
  /** 裁剪后保留的时长（ms）—— 这才是特征层真正铺 32 帧的那一段 */
  keptMs: number;
  /** 开头没人在动的时长（ms），纯触觉判据，见 `leadingStillMs` */
  headStillMs: number;
  dominant: SampleDominance;
  nearTie: boolean;
  imuSuspect: boolean;
  /** 有传感器数据但整段弯折一动不动 = 手套中途冻住/掉线（不是"手闲着"，闲着也有噪声） */
  frozenHands: string[];
}

export interface GroupAudit {
  key: string;
  count: number;
  medianDurationMs: number;
  p10DurationMs: number;
  p90DurationMs: number;
  medianKeptMs: number;
  /** 静止头段的中位长度（ms）与占本词时长的比例，见 `leadingStillMs` */
  medianHeadStillMs: number;
  medianHeadStillRatio: number;
  visionRatio: number;
  trim: TrimReasonCounts;
  dominance: DominanceCounts;
  nearTie: number;
  imuSuspect: number;
  frozen: number;
  /** 时长离群条数（相对**本组**中位数） */
  durationOutliers: number;
}

export interface DatasetAudit {
  total: number;
  recorded: number;
  synthesized: number;
  labels: number;
  hasIdle: boolean;
  overall: GroupAudit;
  byDay: GroupAudit[];
  byLabel: GroupAudit[];
  samples: PerSampleAudit[];
  /** 需要人过一眼的条目，已按严重程度排好 */
  flags: string[];
}

// ===== 逐条体检 =====

function dayOf(timestamp: number): string {
  if (!Number.isFinite(timestamp) || timestamp <= 0) return "未知";
  const d = new Date(timestamp);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

/**
 * 找出"有数据但整段冻住"的手。
 *
 * 判据是弯折 5 路的极差全为 0。为什么不用能量门限：闲着的手也有 ADC 噪声，
 * 能量低但不为 0；**精确的 0** 只可能来自"同一帧被反复写入"，也就是掉线后
 * 最近邻重采样把最后一帧铺满了整条。这条判据不会把"手闲着"误判成掉线。
 */
function frozenHandsOf(sample: SequenceSample): string[] {
  const out: string[] = [];
  const check = (sensor: Uint8Array | null, name: string) => {
    if (!sensor || sample.frameCount < 2) return;
    let moved = false;
    for (let j = 60; j < 65 && !moved; j++) {
      const first = sensor[j];
      for (let t = 1; t < sample.frameCount; t++) {
        if (sensor[t * SEQ_SENSOR_N + j] !== first) {
          moved = true;
          break;
        }
      }
    }
    if (!moved) out.push(name);
  };
  check(sample.leftSensor, "左");
  check(sample.rightSensor, "右");
  return out;
}

function auditSample(
  sample: SequenceSample,
  index: number,
  ranges: BendRanges
): PerSampleAudit {
  // 体检要看的是**训练实际会用的那个口径**，所以标定要传进去 —— 不传的话
  // 第三层（触觉静止段）整层不跑，报告里的 trimReason 会比训练时乐观
  const trim = detectSignSpan(sample, { ...DEFAULT_TRIM, ranges });
  const dom = judgeSampleDominance(sample, ranges);
  const health =
    sample.imuHealth ??
    {
      left: sample.leftImu
        ? analyzeSequenceImu(sample.leftImu, sample.frameCount, sample.timestamps)
        : null,
      right: sample.rightImu
        ? analyzeSequenceImu(sample.rightImu, sample.frameCount, sample.timestamps)
        : null,
    };
  return {
    index,
    label: sample.primaryLabel,
    timestamp: sample.timestamp,
    day: dayOf(sample.timestamp),
    origin: sample.origin,
    durationMs: sample.durationMs,
    frameCount: sample.frameCount,
    hasVision: !!(sample.leftLandmarks || sample.rightLandmarks),
    trimReason: trim.reason,
    keptMs: sample.durationMs * trim.keptRatio,
    headStillMs: leadingStillMs(sample, ranges),
    dominant: dom.dominant,
    nearTie: dom.nearTie,
    // 合成样本的 IMU 是从静态单帧复制出来的，没有真加速度，判不出漂移
    imuSuspect:
      sample.origin === "recorded" &&
      (isImuSuspect(health.left) || isImuSuspect(health.right)),
    frozenHands: frozenHandsOf(sample),
  };
}

// ===== 分组统计 =====

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const i = (sorted.length - 1) * q;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

function emptyTrimCounts(): TrimReasonCounts {
  return { applied: 0, no_vision: 0, no_run: 0, too_short: 0, full_span: 0 };
}

function emptyDominanceCounts(): DominanceCounts {
  return { left: 0, right: 0, idle: 0, no_hand: 0 };
}

function groupOf(key: string, rows: PerSampleAudit[]): GroupAudit {
  const durs = rows.map((r) => r.durationMs).sort((a, b) => a - b);
  const kept = rows.map((r) => r.keptMs).sort((a, b) => a - b);
  const head = rows.map((r) => r.headStillMs).sort((a, b) => a - b);
  // 比例逐条算再取中位，不是"中位头段 ÷ 中位时长" —— 后者在时长散得开的词上会失真
  const headRatio = rows
    .map((r) => (r.durationMs > 0 ? r.headStillMs / r.durationMs : 0))
    .sort((a, b) => a - b);
  const median = quantile(durs, 0.5);
  const trim = emptyTrimCounts();
  const dominance = emptyDominanceCounts();
  let vision = 0;
  let nearTie = 0;
  let imuSuspect = 0;
  let frozen = 0;
  let durationOutliers = 0;
  for (const r of rows) {
    trim[r.trimReason]++;
    dominance[r.dominant]++;
    if (r.hasVision) vision++;
    if (r.nearTie) nearTie++;
    if (r.imuSuspect) imuSuspect++;
    if (r.frozenHands.length) frozen++;
    if (
      median > 0 &&
      (r.durationMs > median * DURATION_OUTLIER_RATIO ||
        r.durationMs < median / DURATION_OUTLIER_RATIO)
    )
      durationOutliers++;
  }
  return {
    key,
    count: rows.length,
    medianDurationMs: median,
    p10DurationMs: quantile(durs, 0.1),
    p90DurationMs: quantile(durs, 0.9),
    medianKeptMs: quantile(kept, 0.5),
    medianHeadStillMs: quantile(head, 0.5),
    medianHeadStillRatio: quantile(headRatio, 0.5),
    visionRatio: rows.length ? vision / rows.length : 0,
    trim,
    dominance,
    nearTie,
    imuSuspect,
    frozen,
    durationOutliers,
  };
}

function grouped(
  rows: PerSampleAudit[],
  keyOf: (r: PerSampleAudit) => string
): GroupAudit[] {
  const map = new Map<string, PerSampleAudit[]>();
  for (const r of rows) {
    const k = keyOf(r);
    const arr = map.get(k);
    if (arr) arr.push(r);
    else map.set(k, [r]);
  }
  // Array.from 而非展开：tsconfig 的 target 下 Map 迭代器展开需要 downlevelIteration
  return Array.from(map.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([k, v]) => groupOf(k, v));
}

export function auditDataset(
  samples: SequenceSample[],
  ranges: BendRanges
): DatasetAudit {
  const rows = samples.map((s, i) => auditSample(s, i, ranges));
  const byDay = grouped(rows, (r) => `${r.day}${r.origin === "recorded" ? "" : "（合成）"}`);
  const byLabel = grouped(rows, (r) => r.label);
  const overall = groupOf("全部", rows);

  return {
    total: rows.length,
    recorded: rows.filter((r) => r.origin === "recorded").length,
    synthesized: rows.filter((r) => r.origin !== "recorded").length,
    labels: byLabel.length,
    hasIdle: rows.some((r) => r.label === IDLE_LABEL),
    overall,
    byDay,
    byLabel,
    samples: rows,
    flags: collectFlags(rows, byDay, byLabel, overall),
  };
}

// ===== 结论 =====

/**
 * 把"该动手的地方"挑出来。这里**只报事实和它的后果**，不自动清洗 ——
 * 每一条对应的处理都会丢数据或改数据，阈值该定多少要看具体分布，不是库该替人决定的。
 */
function collectFlags(
  rows: PerSampleAudit[],
  byDay: GroupAudit[],
  byLabel: GroupAudit[],
  overall: GroupAudit
): string[] {
  const flags: string[] = [];

  // 最贵的一种脏：两批数据的裁剪**判据有没有跑过**不一样 → 同一个词两种时间口径。
  //
  // 分界线是 no_vision 和其余，不是 applied 和其余：`full_span` 意味着判据跑了、
  // 结论是"整段都在做动作、没有起手段可裁"—— 那是一个**经过确认**的口径。
  // `no_vision` 是判据根本没运行，那一批里有多少预备动作**完全未知**。
  // 这两者混在一个数据集里同样致命。
  //
  // 补了第三层（触觉静止段）之后，`no_vision` 只在**没有弯折标定**时还会出现 ——
  // 有标定的话无视觉样本也会被判过一遍，落到 applied / full_span。所以这条 flag
  // 现在同时兼任"标定没做"的告警，下面那句"修法"照旧适用（去做两点标定）。
  const judgedDays = byDay.filter((d) => d.trim.applied + d.trim.full_span > 0);
  const blindDays = byDay.filter(
    (d) => d.trim.applied + d.trim.full_span === 0 && d.trim.no_vision > 0
  );
  if (judgedDays.length > 0 && blindDays.length > 0) {
    const cut = judgedDays.reduce((a, d) => a + d.trim.applied, 0);
    const full = judgedDays.reduce((a, d) => a + d.trim.full_span, 0);
    flags.push(
      `⚠ 裁剪口径分裂：${judgedDays.map((d) => d.key).join("/")} 有视觉，起手段判据跑过了` +
        `（裁掉 ${cut} 条 / 确认全程有效 ${full} 条）；` +
        `而 ${blindDays.map((d) => d.key).join("/")} 全是 no_vision（没开摄像头），` +
        `判据一次都没运行，那一批有多少预备动作完全未知。` +
        `同一个词于是有两种时间口径 —— 这是最该先修的一项。` +
        `无视觉样本的触觉兜底判据（sequenceTrim 第三层）需要弯折两点标定，` +
        `回第 1 步把两只手都标一遍，这一批就能被判过。`
    );
  } else if (overall.trim.no_vision === overall.count && overall.count > 0) {
    flags.push(
      `全部 ${overall.count} 条都是 no_vision（没开摄像头录的），起手段裁剪从未生效，` +
        `预备动作全在数据里。同时视觉覆盖 0% 意味着教师与蒸馏一直被跳过。`
    );
  }

  // 时长跨度：直接决定"训练归一化的时间尺度"散得有多开
  if (overall.p10DurationMs > 0) {
    const spread = overall.p90DurationMs / overall.p10DurationMs;
    if (spread > 2) {
      flags.push(
        `时长跨度大：P10 ${overall.p10DurationMs.toFixed(0)}ms → P90 ${overall.p90DurationMs.toFixed(0)}ms（${spread.toFixed(1)} 倍）。` +
          `训练是把每条**各自的时长**归一化到 32 帧的，跨度越大，同一个词的速度在特征里越散。`
      );
    }
  }

  /*
   * 静止头段排行。回答的是一个非常具体的问题：
   * **滑窗推理时手放着不动，模型会输出哪个词？**
   *
   * 答案就是这份排行的头部。头段占比最大的那些词，它们的 32 帧里有相当一部分
   * 是"什么都没发生"，等于把这些词的标签贴到了静止上。推理时模型对着一个静止
   * 窗口输出它们，是**按它学到的东西正确作答** —— 这种错查不出来是因为看模型、
   * 看特征、看窗口长度都是对的，错在标签。
   *
   * 与 `trimReason` **不再独立**：这个测量（`leadingStillMs`）已经被提进
   * `sequenceTrim` 当第三层判据了，所以有标定时这里报的头段基本会被真的裁掉。
   * 这份排行现在读作"**如果**第三层没跑（没标定），静止会贴到哪些词上"，
   * 以及第三层实际切了多少的对照 —— 排行还很高就说明标定没生效。
   */
  const headRanked = byLabel
    .filter((l) => l.key !== IDLE_LABEL)
    .slice()
    .sort((a, b) => b.medianHeadStillRatio - a.medianHeadStillRatio);
  if (headRanked.length >= 2 && headRanked[0].medianHeadStillRatio >= 0.15) {
    const top = headRanked.slice(0, 6);
    const clean = headRanked[headRanked.length - 1];
    flags.push(
      `静止头段排行（前 ${top.length} 名）：` +
        top
          .map(
            (l) =>
              `${l.key} ${ms(l.medianHeadStillMs)}/${pct(l.medianHeadStillRatio)}`
          )
          .join("  ") +
        `。最干净的是 ${clean.key}（${ms(clean.medianHeadStillMs)}/${pct(clean.medianHeadStillRatio)}）。` +
        `头部这几个词的标签有相当比例贴在"没人在动"上 —— 滑窗推理时手放着不动的窗口` +
        `会被判成它们，而且是模型按学到的东西正确作答，改窗口长度/重训都动不了。` +
        `修法是按触觉判据把头段裁掉（现有裁剪是纯视觉的，手已举在画面里时裁不动），` +
        `裁下来的那些段正好是 \`_idle\` 缺的训练数据。`
    );
  }

  const outlierLabels = byLabel.filter((l) => l.durationOutliers > 0);
  if (outlierLabels.length) {
    const n = outlierLabels.reduce((a, l) => a + l.durationOutliers, 0);
    flags.push(
      `时长离群 ${n} 条，散在 ${outlierLabels.length} 个词上（偏离该词中位数 ${DURATION_OUTLIER_RATIO} 倍以上）：` +
        outlierLabels
          .slice(0, 8)
          .map((l) => `${l.key}×${l.durationOutliers}`)
          .join(" ") +
        (outlierLabels.length > 8 ? " …" : "")
    );
  }

  // 主手：某个词的样本在归一化后仍然分裂，说明那个词的判定不稳
  const tieLabels = byLabel.filter(
    (l) => l.nearTie > 0 && l.nearTie >= l.count * 0.34
  );
  if (tieLabels.length) {
    flags.push(
      `主手判定不确定的词 ${tieLabels.length} 个（该词三分之一以上的样本左右活动量接近）：` +
        tieLabels.map((l) => `${l.key} ${l.nearTie}/${l.count}`).join(" ") +
        `。左右对称的双手词落在这里无害（镜像近似恒等）；单手词落在这里说明闲着那只手动得太多。`
    );
  }

  if (overall.dominance.idle > 0) {
    const idleRows = rows.filter(
      (r) => r.dominant === "idle" && r.label !== IDLE_LABEL
    );
    if (idleRows.length) {
      flags.push(
        `${idleRows.length} 条挂着词标签、但两只手活动量都低于静止门限 —— 疑似空录/漏做动作：` +
          idleRows
            .slice(0, 10)
            .map((r) => `#${r.index} ${r.label}`)
            .join(" ") +
          (idleRows.length > 10 ? " …" : "")
      );
    }
  }

  if (overall.frozen > 0) {
    flags.push(
      `${overall.frozen} 条有手套整段弯折读数一动不动 —— 手套中途冻住/掉线，` +
        `这些条的那只手是假数据：` +
        rows
          .filter((r) => r.frozenHands.length)
          .slice(0, 10)
          .map((r) => `#${r.index} ${r.label}(${r.frozenHands.join("")})`)
          .join(" ")
    );
  }

  if (overall.imuSuspect > 0) {
    flags.push(
      `${overall.imuSuspect} 条 IMU 漂移可疑 —— 四元数通道不可信（弯折/压力仍可用）。`
    );
  }

  const thin = byLabel.filter((l) => l.count < 4);
  if (thin.length) {
    flags.push(
      `样本过少的词 ${thin.length} 个（少于 4 条，切不出验证集）：` +
        thin.map((l) => `${l.key}×${l.count}`).join(" ")
    );
  }

  if (!rows.some((r) => r.label === IDLE_LABEL)) {
    flags.push(
      `没有 \`_idle\` 样本。滑窗推理下模型对任意窗口都必须输出一个词，手放松时会持续乱吐。`
    );
  }

  return flags;
}

// ===== 文本报告 =====

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;
const ms = (v: number) => `${v.toFixed(0)}ms`;

function trimLine(t: TrimReasonCounts): string {
  return (
    [
      ["applied", "已裁"],
      ["full_span", "全程在画面"],
      ["no_vision", "无视觉"],
      ["no_run", "没看见手"],
      ["too_short", "裁完太短"],
    ] as const
  )
    .filter(([k]) => t[k] > 0)
    .map(([k, label]) => `${label} ${t[k]}`)
    .join(" / ") || "—";
}

function domLine(d: DominanceCounts): string {
  return `左 ${d.left} 右 ${d.right} 静止 ${d.idle} 无手 ${d.no_hand}`;
}

function groupBlock(g: GroupAudit, indent = ""): string[] {
  return [
    `${indent}${g.key}  共 ${g.count} 条`,
    `${indent}  时长 中位 ${ms(g.medianDurationMs)}（P10 ${ms(g.p10DurationMs)} / P90 ${ms(g.p90DurationMs)}）` +
      `，裁剪后中位 ${ms(g.medianKeptMs)}`,
    `${indent}  静止头段 中位 ${ms(g.medianHeadStillMs)}（占时长 ${pct(g.medianHeadStillRatio)}）`,
    `${indent}  视觉 ${pct(g.visionRatio)}   裁剪 ${trimLine(g.trim)}`,
    `${indent}  主手 ${domLine(g.dominance)}` +
      (g.nearTie ? `（其中 ${g.nearTie} 条判定不确定）` : ""),
    ...(g.imuSuspect || g.frozen || g.durationOutliers
      ? [
          `${indent}  可疑 IMU漂移 ${g.imuSuspect} / 冻结 ${g.frozen} / 时长离群 ${g.durationOutliers}`,
        ]
      : []),
  ];
}

/** 一份可直接复制走的纯文本报告 */
export function formatAuditReport(a: DatasetAudit): string {
  const lines: string[] = [];
  lines.push("===== 序列数据集体检 =====");
  lines.push(
    `总计 ${a.total} 条（真实 ${a.recorded} / 合成 ${a.synthesized}），${a.labels} 个词，` +
      `_idle ${a.hasIdle ? "有" : "无"}`
  );
  lines.push("");
  lines.push("--- 全体 ---");
  lines.push(...groupBlock(a.overall));

  lines.push("");
  lines.push("--- 按采集批次 ---");
  for (const g of a.byDay) lines.push(...groupBlock(g), "");

  lines.push("--- 按词 ---");
  for (const g of a.byLabel) {
    lines.push(
      `${g.key}  ${g.count} 条 | 主手 ${domLine(g.dominance)}` +
        (g.nearTie ? ` (不确定 ${g.nearTie})` : "") +
        ` | 时长中位 ${ms(g.medianDurationMs)}` +
        ` | 静止头段 ${ms(g.medianHeadStillMs)}(${pct(g.medianHeadStillRatio)})` +
        (g.durationOutliers ? ` | 离群 ${g.durationOutliers}` : "") +
        ` | 裁剪 ${trimLine(g.trim)}`
    );
  }

  lines.push("");
  lines.push("--- 需要处理的 ---");
  if (a.flags.length === 0) lines.push("没发现明显问题");
  else a.flags.forEach((f, i) => lines.push(`${i + 1}. ${f}`));

  return lines.join("\n");
}
