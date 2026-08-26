/*
 * yawDrift —— 「能不能给 yaw 建一个体坐标系参考」的可行性探针
 *
 * 为什么需要它：我/你/他 是同一个手型指不同方向，区别几乎纯粹在绕竖直轴的
 * 朝向(yaw)。而学生模型的特征里两条 IMU 通路**都对 yaw 不变**，且都是故意的
 * （sequenceFeatures.ts 开头：相对首帧的四元数消掉绝对朝向；重力在手系的投影
 * 天生 yaw 不变）。所以这三个词在特征空间里是同一个点 —— 不是训得不够，是
 * 信息不在里面。
 *
 * 唯一的纯手套出路是给 yaw 找一个零点、用陀螺积分维持。但手套是 ICM-42688
 * 六轴无磁力计（imuHealth.ts:4），绝对 yaw 没有零点且持续漂。所以这条路成不
 * 成立，押在两个从没量过的数上：
 *
 *   1) **漂多快** —— 手静止时 yaw 每秒漂几度。决定归零之后能撑多久。
 *   2) **零点稳不稳** —— 每次手回到静止位，朝向是不是同一个。漂得再慢，
 *      零点本身跳来跳去也白搭，而这一项和漂移是两回事：它还包含人每次把手
 *      放回"准备位"的姿势差异。
 *
 * 两个数都能从**已经采到的数据**里量出来，不用新采一条。先量再写，免得把
 * 参考系建完了才发现累计误差比类间距还大。
 *
 * 这个模块只读，一个字节都不改。
 */
import type { SequenceSample } from "./datasetStore";
import { SEQ_IMU_N } from "./datasetStore";
import {
  sampleEnergies,
  IDLE_ENERGY,
  W_BEND,
  W_PRESSURE,
  type BendRanges,
} from "./dominantHand";
import {
  quatConj,
  quatMul,
  quatNormalize,
  rotateVec,
  type Quat,
} from "./sequenceFeatures";

// ===== 判据常数 =====

/**
 * 触觉静止门限（份）。
 *
 * 复用 `IDLE_ENERGY` 的口径，但**只算弯折+指压**。朝向那一路在这里绝对不能用：
 * 我们要量的就是"手不动时朝向变了多少"，拿朝向当静止判据等于只挑漂移恰好为 0
 * 的那些段，量出来的漂移率必然偏小 —— 而这个探针是拿来做 go/no-go 的，偏小
 * 就意味着照着它去建参考系，然后发现不好使。
 *
 * dominantHand 里 total = 0.55·bend + 0.25·pressure + 0.2·orient，手真静止时
 * orient 项接近 0，所以 total ≈ (W_BEND+W_PRESSURE)·触觉部分。按同一比例折算，
 * 与整体口径一致。
 */
export const TACTILE_STILL_ENERGY = IDLE_ENERGY * (W_BEND + W_PRESSURE);

/** 判静止的子窗长度（ms）。太短时弯折 σ 只反映几帧噪声，判不出"动了没有" */
export const STILL_PROBE_MS = 200;

/**
 * 量漂移至少需要这么长的静止段（ms）。
 *
 * 短段量出来的是噪声除以一个很小的分母：0.5° 的姿态抖动摊到 200ms 上是
 * 2.5°/s，摊到 800ms 上是 0.6°/s。取 800ms 让真实漂移能盖过抖动。
 */
export const MIN_STILL_MS = 800;

/**
 * 可容忍的累计朝向误差（度）。
 *
 * 我/你/他 里最近的一对是「你(指对方)」和「他(指侧前方)」，大约差 45°。
 * 留三分之一余量 = 15°：超过这个数，三个类的分布就开始互相咬。
 */
export const TOLERABLE_ERROR_DEG = 15;

/**
 * 折算累计误差时假设的重新归零间隔（秒）。
 *
 * 部署时的归零事件是"手回到静止"（动作闸门已经能判）。正常连着打手语时，
 * 词与词之间的停顿大约就是这个量级。漂移率 × 这个数 = 一次归零能撑到下次
 * 归零时的误差。
 */
export const REZERO_INTERVAL_S = 10;

/** 时间间隔超过这么久就算换了一次采集会话（陀螺重启、手套摘下重戴） */
export const SESSION_GAP_MS = 10 * 60 * 1000;

/** 单数三个代词 —— 这个探针要救的就是它们 */
export const SINGULAR_PRONOUNS = ["i", "you", "he"];
/** 复数三个。同手型 + 同一段横向弧线，在手系里也是同一个旋转，一样简并 */
export const PLURAL_PRONOUNS = ["we", "you_pl", "they"];

// ===== 几何 =====

/**
 * 两个姿态之间绕**世界竖直轴**的转角（度），取绝对值。
 *
 * 不走欧拉角：yaw 的欧拉定义在 pitch 接近 ±90° 时退化，而且要先约定旋转次序，
 * 板载解算用的哪一种没有文档。这里把相对旋转转成旋转向量、投影到世界重力轴
 * （sequenceFeatures 的 WORLD_GRAVITY = [0,0,1]）上，无约定、无奇点。
 */
export function yawDeltaDeg(qa: Quat, qb: Quat): number {
  const a = quatNormalize(qa);
  const rel = quatMul(quatConj(a), quatNormalize(qb));
  // q 和 -q 是同一个旋转。w<0 时整体翻符号取最短弧，否则角度会算成 2π-θ
  const flip = rel[0] < 0 ? -1 : 1;
  const w = Math.min(1, Math.max(-1, flip * rel[0]));
  const s = Math.sqrt(Math.max(0, 1 - w * w));
  if (s < 1e-9) return 0; // 没转，旋转轴无定义
  const ang = 2 * Math.acos(w);
  // 旋转向量表达在**手系 a** 下，要先转到世界系才能取竖直分量
  const world = rotateVec(a, [
    (flip * rel[1]) / s,
    (flip * rel[2]) / s,
    (flip * rel[3]) / s,
  ]);
  return Math.abs((ang * world[2] * 180) / Math.PI);
}

type Side = "left" | "right";

function imuOf(sample: SequenceSample, side: Side): Float32Array | null {
  return side === "left" ? sample.leftImu : sample.rightImu;
}

/** 取第 t 帧的四元数。全零 = 那一帧没写进去，返回 null 而不是当成合法姿态 */
function quatAt(imu: Float32Array, t: number): Quat | null {
  const o = t * SEQ_IMU_N;
  if (o + 3 >= imu.length) return null;
  const q: Quat = [imu[o], imu[o + 1], imu[o + 2], imu[o + 3]];
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return n > 1e-6 ? q : null;
}

export interface StillRun {
  /** 帧号，含首不含尾 */
  start: number;
  end: number;
  ms: number;
}

/**
 * 找一条录制里**最长的触觉静止段**。
 *
 * 用最长段而不是开头那段（datasetAudit 的 `leadingStillMs`）：那个量的是
 * "动作什么时候开始"，判据挂在整段口径上；这里要的是"尽量长的一段确定没动
 * 的时间"，越长漂移量越准，而且末尾停下来的那段一样能用。
 */
export function longestStillRun(
  sample: SequenceSample,
  ranges: BendRanges,
  side: Side
): StillRun | null {
  const T = sample.frameCount;
  if (T < 4 || !(sample.durationMs > 0)) return null;
  const dt = sample.durationMs / (T - 1);
  const step = Math.max(3, Math.round(STILL_PROBE_MS / dt));

  let best: StillRun | null = null;
  let runStart = -1;
  for (let i = 0; i + step <= T; i += step) {
    const e = sampleEnergies(sample, ranges, i, step);
    const h = side === "left" ? e.left : e.right;
    const tactile = h ? W_BEND * h.bend + W_PRESSURE * h.pressure : Infinity;
    const still = tactile < TACTILE_STILL_ENERGY;

    if (still) {
      if (runStart < 0) runStart = i;
      const end = i + step;
      const ms = (end - 1 - runStart) * dt;
      if (!best || ms > best.ms) best = { start: runStart, end, ms };
    } else {
      runStart = -1;
    }
  }
  return best;
}

/** 一条录制上量到的漂移率（°/s）。静止段不够长或 IMU 缺帧时 null */
export function driftDegPerSec(
  sample: SequenceSample,
  ranges: BendRanges,
  side: Side
): number | null {
  const run = longestStillRun(sample, ranges, side);
  if (!run || run.ms < MIN_STILL_MS) return null;
  const imu = imuOf(sample, side);
  if (!imu) return null;
  const a = quatAt(imu, run.start);
  const b = quatAt(imu, run.end - 1);
  if (!a || !b) return null;
  return yawDeltaDeg(a, b) / (run.ms / 1000);
}

// ===== 汇总 =====

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[i];
}

export type YawVerdict = "viable" | "marginal" | "dead" | "unknown";

export interface YawProbe {
  /** 量到漂移的录制条数（静止段够长的那些） */
  driftSamples: number;
  medianDriftDegPerSec: number;
  p90DriftDegPerSec: number;
  /**
   * 归零点漂移：同一会话里相邻两条录制的静止位之间差了多少度。
   * 这个数**同时包含**陀螺漂移和人把手放回准备位的姿势差异 —— 两者都会让
   * "拿静止当零点"失准，所以要一起量
   */
  restShiftSamples: number;
  medianRestShiftDeg: number;
  p90RestShiftDeg: number;
  /** 会话数（间隔超过 SESSION_GAP_MS 算换一次） */
  sessions: number;
  /** 一次归零能撑到下次归零时的预计累计误差（度） */
  projectedErrorDeg: number;
  /** 库里六个代词各有多少条 */
  pronounCounts: Record<string, number>;
  verdict: YawVerdict;
  notes: string[];
}

/**
 * 跑一遍可行性探针。
 *
 * 只读：不改样本、不写库、不碰模型。
 */
export function probeYawReference(
  samples: SequenceSample[],
  ranges: BendRanges
): YawProbe {
  const notes: string[] = [];

  // ---- 1) 漂移率 ----
  const drifts: number[] = [];
  for (const s of samples) {
    if (s.origin !== "recorded") continue; // 合成样本的 IMU 是单帧复制的，漂移恒为 0
    for (const side of ["left", "right"] as const) {
      const d = driftDegPerSec(s, ranges, side);
      if (d !== null) drifts.push(d);
    }
  }
  drifts.sort((a, b) => a - b);

  // ---- 2) 归零点漂移 ----
  // 按时间排序后切会话，会话内取相邻两条的静止位比朝向
  const rec = samples
    .filter((s) => s.origin === "recorded" && s.timestamp > 0)
    .slice()
    .sort((a, b) => a.timestamp - b.timestamp);

  const shifts: number[] = [];
  let sessions = rec.length > 0 ? 1 : 0;
  let prev: { q: Quat; side: Side } | null = null;

  for (let i = 0; i < rec.length; i++) {
    const s = rec[i];
    if (i > 0 && s.timestamp - rec[i - 1].timestamp > SESSION_GAP_MS) {
      sessions++;
      prev = null; // 换会话 = 陀螺可能重启过，跨会话比朝向没有意义
    }
    // 右手优先：训练口径已经归一化到右手，右手的样本量也大得多
    let cur: { q: Quat; side: Side } | null = null;
    for (const side of ["right", "left"] as const) {
      const run = longestStillRun(s, ranges, side);
      const imu = imuOf(s, side);
      if (!run || !imu) continue;
      const q = quatAt(imu, run.start);
      if (q) {
        cur = { q, side };
        break;
      }
    }
    if (cur && prev && prev.side === cur.side) {
      shifts.push(yawDeltaDeg(prev.q, cur.q));
    }
    if (cur) prev = cur;
  }
  shifts.sort((a, b) => a - b);

  // ---- 3) 代词存量 ----
  const pronounCounts: Record<string, number> = {};
  for (const p of [...SINGULAR_PRONOUNS, ...PLURAL_PRONOUNS]) pronounCounts[p] = 0;
  for (const s of samples) {
    if (s.primaryLabel in pronounCounts) pronounCounts[s.primaryLabel]++;
  }

  const medianDrift = quantile(drifts, 0.5);
  const p90Drift = quantile(drifts, 0.9);
  const medianShift = quantile(shifts, 0.5);
  const p90Shift = quantile(shifts, 0.9);
  // 用 P90 而不是中位数来判：偶尔串一次类，用起来就是"时好时坏"，
  // 那比稳定地分不开更难查
  const projectedErrorDeg = p90Drift * REZERO_INTERVAL_S;

  let verdict: YawVerdict = "unknown";
  if (drifts.length < 10) {
    notes.push(
      `只有 ${drifts.length} 条录制量到了够长的静止段（需要 ≥${MIN_STILL_MS}ms），` +
        `样本太少，这个结论不作数。`
    );
  } else if (
    projectedErrorDeg <= TOLERABLE_ERROR_DEG &&
    p90Shift <= TOLERABLE_ERROR_DEG
  ) {
    verdict = "viable";
  } else if (
    projectedErrorDeg <= TOLERABLE_ERROR_DEG * 3 &&
    p90Shift <= TOLERABLE_ERROR_DEG * 3
  ) {
    verdict = "marginal";
  } else {
    verdict = "dead";
  }

  if (verdict !== "unknown" && p90Shift > TOLERABLE_ERROR_DEG) {
    notes.push(
      `归零点本身在跳（P90 ${p90Shift.toFixed(1)}°）。这一项不全是陀螺漂移 —— ` +
        `也可能是每次把手放回"准备位"的姿势不一致。若漂移率本身合格，` +
        `可以靠"归零时要求一个固定姿势"把它压下去。`
    );
  }
  const missing = SINGULAR_PRONOUNS.filter((p) => pronounCounts[p] === 0);
  if (missing.length > 0) {
    notes.push(
      `库里没有 ${missing.join("/")} 的录制 —— 就算参考系建起来，也没有数据能验证` +
        `三个代词到底分不分得开。`
    );
  }

  return {
    driftSamples: drifts.length,
    medianDriftDegPerSec: medianDrift,
    p90DriftDegPerSec: p90Drift,
    restShiftSamples: shifts.length,
    medianRestShiftDeg: medianShift,
    p90RestShiftDeg: p90Shift,
    sessions,
    projectedErrorDeg,
    pronounCounts,
    verdict,
    notes,
  };
}

const VERDICT_TEXT: Record<YawVerdict, string> = {
  viable: "可行 —— 建参考系这条路走得通，可以动手",
  marginal: "勉强 —— 需要更频繁的归零，或者接受偶尔串类",
  dead: "不可行 —— 漂移已经盖过类间距，别建了，走合并方案",
  unknown: "数据不足，判不了",
};

/**
 * 判"可行"之后的下一步，写死在报告里。
 *
 * 缺的**不是**轴向映射。绕世界竖直轴的转角与 IMU 在手套里怎么装无关：参考
 * 姿态和当前姿态都经过同一个（未知的）装配旋转，而 `yawDeltaDeg` 在世界系里
 * 取竖直分量，装配旋转在这一步被消掉了。所以 yaw 特征不需要知道机体轴。
 *
 * 缺的是**一个专门采集的零位姿势**。现在的特征把参考取在"每个窗口的首帧"
 * （sequenceFeatures.ts:6-9），而那个参考每一帧都在动 —— 指自己和指对方各自
 * 相对自己的首帧都接近 0，所以三个词还是同一个点。要让 yaw 有意义，参考必须
 * 是一个**固定的、和身体绑定的姿势**。
 *
 * glove_visual/glove_final_code 有现成的这一步：校准向导里的「竖立·手心朝自己
 * （陀螺仪零位）」采 `quaternionReference`，`calibrateQuaternion` 用
 * `ref⁻¹ ⊗ q` 应用它（glove-protocol.ts:471）。deaf-kit 的采集/推理链路里
 * 没有任何对应的东西，这块要新建。
 */
const NEXT_STEP_IF_VIABLE =
  "下一步是补一个**固定零位姿势**的采集与应用 —— 现在的参考是每个窗口的首帧，" +
  "它自己在动，所以指自己和指对方相对各自首帧都接近 0，三个词还是同一个点。" +
  "glove_visual/glove_final_code 的「陀螺仪零位」步骤 + `calibrateQuaternion`" +
  "（ref⁻¹⊗q）是现成的，deaf-kit 链路里没有对应物。" +
  "注意不需要轴向映射：绕世界竖直轴的转角与 IMU 装配朝向无关。";

export function formatYawProbe(p: YawProbe): string {
  const L: string[] = [];
  L.push("=== YAW 参考系可行性 ===");
  L.push(`结论：${VERDICT_TEXT[p.verdict]}`);
  L.push("");
  L.push(
    `静止时 yaw 漂移：中位 ${p.medianDriftDegPerSec.toFixed(2)}°/s，` +
      `P90 ${p.p90DriftDegPerSec.toFixed(2)}°/s（${p.driftSamples} 段）`
  );
  L.push(
    `归零 ${REZERO_INTERVAL_S}s 后的预计累计误差：${p.projectedErrorDeg.toFixed(1)}°` +
      `（可容忍 ≤${TOLERABLE_ERROR_DEG}°，这是「你」和「他」间距的三分之一）`
  );
  L.push(
    `归零点漂移：中位 ${p.medianRestShiftDeg.toFixed(1)}°，` +
      `P90 ${p.p90RestShiftDeg.toFixed(1)}°（${p.restShiftSamples} 对相邻录制，${p.sessions} 个会话）`
  );
  L.push("");
  const pr = [...SINGULAR_PRONOUNS, ...PLURAL_PRONOUNS]
    .map((k) => `${k} ${p.pronounCounts[k] ?? 0}`)
    .join(" / ");
  L.push(`代词存量：${pr}`);
  if (p.notes.length > 0) {
    L.push("");
    for (const n of p.notes) L.push(`⚠ ${n}`);
  }
  if (p.verdict === "viable" || p.verdict === "marginal") {
    L.push("");
    L.push(`→ ${NEXT_STEP_IF_VIABLE}`);
  }
  return L.join("\n");
}
