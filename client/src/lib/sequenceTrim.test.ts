import { describe, it, expect } from "vitest";
import {
  DEFAULT_TRIM,
  detectSignSpan,
  leadingStillMs,
  spanToGridBounds,
  summarizeTrim,
  trailingStillMs,
  trimSpanForExport,
  visibleRuns,
  type TrimConfig,
} from "./sequenceTrim";
import type { BendRange } from "./bendRange";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "./datasetStore";

// ===== 夹具 =====

/**
 * `visible` 传 null 表示那只手整条没有关键点数组（= 没开摄像头 / 只戴一只手套）；
 * 传函数则逐帧决定看不看得见，看不见的帧整帧写 NaN —— 这就是 datasetStore 的约定，
 * 也是 detectSignSpan 唯一依赖的信号。
 */
interface FixtureOptions {
  T: number;
  fps?: number;
  leftVisible?: ((t: number) => boolean) | null;
  rightVisible?: ((t: number) => boolean) | null;
  /** 手腕轨迹（归一化图像坐标，**y 向下为正**）。不给就是不动 —— 到位检测判 no_rise */
  leftWrist?: (t: number) => { x: number; y: number };
  rightWrist?: (t: number) => { x: number; y: number };
  /** 五路弯折 ADC。不给就全 0，弯折钳位不会生效 */
  leftBend?: (t: number) => number;
}

const STILL = () => ({ x: 0.5, y: 0.5 });

function makeLandmarks(
  T: number,
  visible: (t: number) => boolean,
  wrist: (t: number) => { x: number; y: number }
): Float32Array {
  const lm = new Float32Array(T * SEQ_LANDMARK_N);
  for (let t = 0; t < T; t++) {
    const o = t * SEQ_LANDMARK_N;
    if (visible(t)) {
      for (let i = 0; i < SEQ_LANDMARK_N; i++) lm[o + i] = 0.5 + i * 0.001;
      // 0 号关键点是手腕，到位检测只看它的 x/y
      const { x, y } = wrist(t);
      lm[o] = x;
      lm[o + 1] = y;
    } else {
      lm.fill(NaN, o, o + SEQ_LANDMARK_N);
    }
  }
  return lm;
}

/** 137 维重排后弯折固定在 60~64（sensorMapping.ts:50） */
const BEND0 = 60;

function makeSample(opts: FixtureOptions): SequenceSample {
  const { T, fps = 30 } = opts;
  const dt = 1000 / fps;
  const timestamps = new Float32Array(T);
  for (let t = 0; t < T; t++) timestamps[t] = t * dt;

  const imu = new Float32Array(T * SEQ_IMU_N);
  for (let t = 0; t < T; t++) imu[t * SEQ_IMU_N] = 1; // 单位四元数

  const leftSensor = new Uint8Array(T * SEQ_SENSOR_N);
  if (opts.leftBend) {
    for (let t = 0; t < T; t++) {
      const v = opts.leftBend(t);
      for (let k = 0; k < 5; k++) leftSensor[t * SEQ_SENSOR_N + BEND0 + k] = v;
    }
  }

  return {
    segments: [{ label: "test", startFrame: 0, endFrame: T }],
    primaryLabel: "test",
    frameCount: T,
    timestamps,
    leftSensor,
    rightSensor: null,
    leftImu: imu,
    rightImu: null,
    leftLandmarks: opts.leftVisible
      ? makeLandmarks(T, opts.leftVisible, opts.leftWrist ?? STILL)
      : null,
    rightLandmarks: opts.rightVisible
      ? makeLandmarks(T, opts.rightVisible, opts.rightWrist ?? STILL)
      : null,
    durationMs: T * dt,
    sourceFps: fps,
    origin: "recorded",
    timestamp: 0,
  };
}

/** 默认判据但不留白 —— 留白会让"裁到哪一帧"依赖 fps，多数断言里是噪声 */
const NO_PAD: TrimConfig = { ...DEFAULT_TRIM, padMs: 0 };

/*
 * 第三层（触觉静止段）的门票：**量程真的齐**。
 *
 * 夹具只给左手传感器（`rightSensor: null`），所以只标左手就满足
 * `sampleEnergies().calibrated` 的量程对称性守则 —— 没有数据的那只手不需要标定。
 * 只要不传 `ranges`，下面所有旧断言里第三层都不参与。
 */
const CAL: { LH: BendRange } = {
  LH: { open: [40, 40, 40, 40, 40], fist: [160, 160, 160, 160, 160] },
};
/** 带标定的判据。三层都开 */
const WITH_CAL: TrimConfig = { ...NO_PAD, ranges: CAL };

// ===== detectSignSpan =====

describe("detectSignSpan", () => {
  it("开头的 NaN 段被切掉，只留下可见区间", () => {
    // 30 帧 @30fps；前 10 帧手还在腿上
    const s = makeSample({ T: 30, leftVisible: (t) => t >= 10 });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(true);
    expect(span.reason).toBe("applied");
    expect(span.startFrame).toBe(10);
    expect(span.endFrame).toBe(30);
    // 保留 ts[29]-ts[10] = 19 帧间隔 / 总 29 帧间隔
    expect(span.keptRatio).toBeCloseTo(19 / 29, 6);
  });

  it("两只手的关键点数组都是 null → no_vision，不裁", () => {
    const s = makeSample({ T: 30 });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("no_vision");
    expect(span.startFrame).toBe(0);
    expect(span.endFrame).toBe(30);
    expect(span.keptRatio).toBe(1);
  });

  it("有关键点但整条都是 NaN → no_run，不裁", () => {
    const s = makeSample({ T: 30, leftVisible: () => false });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("no_run");
  });

  it("只有一帧误检（短于 minRunFrames）不足以定起点 → no_run", () => {
    const s = makeSample({ T: 30, leftVisible: (t) => t === 5 });
    const span = detectSignSpan(s, { ...NO_PAD, minRunFrames: 3 });
    expect(span.reason).toBe("no_run");
  });

  it("两段可见时取最长的那段（抬手途中被扫到几帧不该定为起点）", () => {
    const s = makeSample({
      T: 40,
      leftVisible: (t) => (t >= 2 && t <= 4) || (t >= 15 && t <= 35),
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(true);
    expect(span.startFrame).toBe(15);
    expect(span.endFrame).toBe(36);
  });

  it("只有右手看得见也照样裁（可见性取并集，不是交集）", () => {
    // 左手整条 NaN（单手词的另一只手），右手后半段入画
    const s = makeSample({
      T: 30,
      leftVisible: () => false,
      rightVisible: (t) => t >= 12,
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(true);
    expect(span.startFrame).toBe(12);
    expect(span.endFrame).toBe(30);
  });

  it("并集覆盖两只手各自的可见段", () => {
    const s = makeSample({
      T: 40,
      leftVisible: (t) => t >= 5 && t <= 20,
      rightVisible: (t) => t >= 18 && t <= 30,
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.startFrame).toBe(5);
    expect(span.endFrame).toBe(31);
  });

  it("全程可见 → full_span，applied=false", () => {
    const s = makeSample({ T: 30, leftVisible: () => true });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("full_span");
    expect(span.startFrame).toBe(0);
    expect(span.endFrame).toBe(30);
  });

  it("可见段太短（帧数不够）→ too_short，宁可不裁", () => {
    const s = makeSample({ T: 30, leftVisible: (t) => t >= 10 && t <= 13 });
    const span = detectSignSpan(s, { ...NO_PAD, minRunFrames: 3 });
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("too_short");
  });

  it("可见段帧数够但时长不够 → too_short", () => {
    // 200fps 下 8 帧只有 35ms，低于 minKeptMs=250
    const s = makeSample({
      T: 60,
      fps: 200,
      leftVisible: (t) => t >= 20 && t <= 27,
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("too_short");
  });

  it("padMs 按时间戳向外扩，不按帧数", () => {
    // 30fps → dt≈33.3ms；padMs=100 正好往回走 3 帧
    const s = makeSample({ T: 30, leftVisible: (t) => t >= 10 && t <= 25 });
    // 用 105 而不是 100：时间戳是 float32，3×33.33 存回来是 100.000016，
    // 卡在 100 的边界上会随浮点舍入翻边
    const span = detectSignSpan(s, { ...DEFAULT_TRIM, padMs: 105 });
    expect(span.startFrame).toBe(7);
    expect(span.endFrame).toBe(29); // 25 之后含 26,27,28（+33/67/100ms）
  });

  it("留白把区间撑到整条时视为 full_span", () => {
    const s = makeSample({ T: 12, leftVisible: (t) => t >= 3 && t <= 8 });
    const span = detectSignSpan(s, { ...DEFAULT_TRIM, padMs: 200 });
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("full_span");
  });

  it("帧数少于 2 直接判 too_short，不崩", () => {
    const s = makeSample({ T: 1, leftVisible: () => true });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(false);
    expect(span.reason).toBe("too_short");
  });
});

// ===== 第二层：到位检测 =====
//
// 第一层只能切掉**画面外**那半截。手一进画面可见性就翻成 true，而"进画面之后继续抬到
// 起势位置"那段照样是冗余 —— 它有关键点、运动能量还很高。这一层就是补那个洞的。

/** 匀速上抬到 `settleAt` 帧、之后停住不动。y 向下为正，所以抬起来是 y 变小 */
function riseThenHold(
  from: number,
  settleAt: number,
  y0 = 0.9,
  y1 = 0.4,
  x = 0.5
) {
  return (t: number) => {
    if (t <= from) return { x, y: y0 };
    if (t >= settleAt) return { x, y: y1 };
    return { x, y: y0 + ((y1 - y0) * (t - from)) / (settleAt - from) };
  };
}

describe("detectSignSpan —— 到位检测", () => {
  it("入画之后还在上抬的那段也被切掉（第一层看不见的部分）", () => {
    // 前 10 帧手在画面外；入画后一路抬到第 25 帧才到位
    const s = makeSample({
      T: 40,
      leftVisible: (t) => t >= 10,
      leftWrist: riseThenHold(10, 25),
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(true);
    expect(span.arrival?.reason).toBe("applied");
    // 平滑窗口会有一帧量级的滞后，锁区间而不是锁具体帧号
    expect(span.startFrame).toBeGreaterThanOrEqual(24);
    expect(span.startFrame).toBeLessThanOrEqual(27);
    // 只按可见性会停在 10；多切掉的这十几帧正是用户说的那段冗余
    expect(span.arrival!.droppedFrames).toBeGreaterThanOrEqual(14);
    expect(span.arrival!.bendClamped).toBe(false);
  });

  it("手全程在画面里、只是前半段在上抬 —— 从 full_span 变成裁得掉", () => {
    // 这条是这一层存在的理由：第一层对它完全无能为力（会判 full_span）
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16),
    });
    expect(detectSignSpan(s, { ...NO_PAD, arrival: null }).reason).toBe(
      "full_span"
    );

    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(true);
    expect(span.reason).toBe("applied");
    expect(span.startFrame).toBeGreaterThanOrEqual(15);
    expect(span.startFrame).toBeLessThanOrEqual(18);
    expect(span.endFrame).toBe(40);
  });

  it("横向挥手不算抬手 —— 竖直分量不占优就 no_rise", () => {
    // "再见"是左右摆动；把第一个摆幅当抬手切掉就等于吃掉手势的开头
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: (t) => ({ x: t >= 16 ? 0.9 : 0.5 + (0.4 * t) / 16, y: 0.5 }),
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.arrival?.reason).toBe("no_rise");
    expect(span.reason).toBe("full_span");
  });

  it("上抬幅度太小（手本来就举着）→ no_rise", () => {
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16, 0.5, 0.47), // 只上移 0.03，低于 minRiseNorm=0.08
    });
    expect(detectSignSpan(s, NO_PAD).arrival?.reason).toBe("no_rise");
  });

  it("速度一直不落下来 → no_settle，不裁", () => {
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 39, 0.95, 0.1),
    });
    expect(detectSignSpan(s, NO_PAD).arrival?.reason).toBe("no_settle");
  });

  it("谷底落在可见段后段 → too_late，宁可不切", () => {
    // 抬到 80% 才停，超过 maxArrivalRatio=0.6；再切就是在切手势本身
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 32, 0.9, 0.3),
    });
    expect(detectSignSpan(s, NO_PAD).arrival?.reason).toBe("too_late");
  });

  it("手型在到位之前就成形 → 起点被弯折钳位往前拉，不往后推", () => {
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16),
      leftBend: (t) => (t >= 5 ? 200 : 10), // 第 5 帧捏出手型
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.arrival?.reason).toBe("applied");
    expect(span.arrival!.bendClamped).toBe(true);
    // 到位帧在 16 附近，但手型 5 帧就变了 —— 取较早的那个
    expect(span.startFrame).toBe(5);
  });

  it("手型全程不变时钳位不生效（弯折没动就不该影响起点）", () => {
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16),
      leftBend: () => 128,
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.arrival!.bendClamped).toBe(false);
    expect(span.startFrame).toBeGreaterThanOrEqual(15);
  });

  it("arrival: null 时整层关掉，行为退回只按可见性", () => {
    const s = makeSample({
      T: 40,
      leftVisible: (t) => t >= 10,
      leftWrist: riseThenHold(10, 25),
    });
    const span = detectSignSpan(s, { ...NO_PAD, arrival: null });
    expect(span.arrival?.reason).toBe("disabled");
    expect(span.startFrame).toBe(10);
  });

  it("推后之后剩太短 → 退回**只按可见性**裁，不是整条不裁", () => {
    const s = makeSample({
      T: 40,
      leftVisible: (t) => t >= 10,
      leftWrist: riseThenHold(10, 25),
    });
    // 到位后只剩约 14 帧，够不上 20
    const span = detectSignSpan(s, { ...NO_PAD, minKeptFrames: 20 });
    expect(span.arrival?.reason).toBe("too_short");
    expect(span.applied).toBe(true);
    expect(span.startFrame).toBe(10); // 第一层的结论保住了
  });

  it("双手取较早到位的那只（另一只已经开始做事就不能再往后切）", () => {
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 22),
      rightVisible: () => true,
      rightWrist: riseThenHold(0, 10),
    });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.arrival?.reason).toBe("applied");
    expect(span.startFrame).toBeLessThanOrEqual(13);
  });

  it("整条没有视觉时这一层判 no_landmarks 而不是硬报错（第三层还要接着跑）", () => {
    // 补了第三层之后，detectSignSpan 不再在"没有关键点"时提前返回 ——
    // 提前返回会让触觉兜底永远没机会运行。所以这里 arrival 是一个**有理由的结论**，
    // 不是 null（null 专门留给 `cfg.arrival === null`，即整层被显式关掉）
    const span = detectSignSpan(makeSample({ T: 30 }), NO_PAD);
    expect(span.reason).toBe("no_vision");
    expect(span.arrival?.reason).toBe("no_landmarks");
    expect(span.arrival!.droppedFrames).toBe(0);
    expect(span.startFrame).toBe(0);
    expect(span.endFrame).toBe(30);
  });

  it("丢帧时速度按时间戳算 —— 帧号口径会把慢动作误判成到位", () => {
    // 同一段上抬，但后半段每帧间隔从 33ms 拉到 100ms（丢帧）。
    // 按帧号算速度会在丢帧处看到"速度不变"，按时间戳算才知道手确实慢下来了。
    const T = 40;
    const s = makeSample({
      T,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16),
    });
    const fast = detectSignSpan(s, NO_PAD);
    for (let t = 20; t < T; t++) s.timestamps[t] = s.timestamps[19] + (t - 19) * 100;
    const slow = detectSignSpan(s, NO_PAD);
    // 拉长尾部不该改变前半段的到位判断
    expect(slow.startFrame).toBe(fast.startFrame);
  });
});

// ===== 第三层：触觉静止段 =====

/*
 * 这一层存在的理由是两个视觉判据判不出来的缺口：
 *  - 手全程在画面里（159/364 条 `full_span`）—— 没有入画时刻，也没有向上冲程；
 *  - 整批没开摄像头 —— 视觉那两层一行都跑不了。
 * 而它是**唯一**会动尾巴的一层：可见段的 `end` 是"最后一个可见帧"，手不出画就是整条。
 */
describe("detectSignSpan —— 触觉静止段", () => {
  /** 前 `holdTo` 帧弯折恒定（静止），之后按 6 ADC/帧线性拉开 */
  const stillThenMove = (holdTo: number) => (t: number) =>
    t <= holdTo ? 20 : Math.min(255, 20 + (t - holdTo) * 6);

  it("trailingStillMs：末尾弯折恒定 → 量到那一段；一路动到最后 → 0", () => {
    const tail = makeSample({
      T: 60,
      leftBend: (t) => (t < 30 ? 10 + t * 6 : 190),
    });
    // 后 30 帧恒定（≈1000ms）。扫描步长 ONSET_PROBE_MS，所以是量级相符而非精确值
    expect(trailingStillMs(tail, CAL)).toBeGreaterThan(800);
    expect(trailingStillMs(tail, CAL)).toBeLessThan(1200);

    const moving = makeSample({ T: 60, leftBend: (t) => 10 + t * 3 });
    expect(trailingStillMs(moving, CAL)).toBe(0);
  });

  it("leadingStillMs 与 trailingStillMs 在时间上互为镜像", () => {
    const head = makeSample({ T: 60, leftBend: stillThenMove(29) });
    const tail = makeSample({ T: 60, leftBend: (t) => stillThenMove(29)(59 - t) });
    expect(leadingStillMs(head, CAL)).toBeCloseTo(trailingStillMs(tail, CAL), 5);
    expect(trailingStillMs(head, CAL)).toBe(0);
    expect(leadingStillMs(tail, CAL)).toBe(0);
  });

  it("整段一动不动 → 量到的静止段就是整条时长（交给 minKept 守则去否掉）", () => {
    const s = makeSample({ T: 60, leftBend: () => 128 });
    expect(leadingStillMs(s, CAL)).toBeCloseTo(s.durationMs, 5);
    expect(trailingStillMs(s, CAL)).toBeCloseTo(s.durationMs, 5);
  });

  it("「手举着等着」型录制：视觉两层都判不出来，第三层把起点推后", () => {
    // 手全程在画面里且不动 → 可见性无从下手（full_span）、到位检测判 no_rise。
    // 这正是那 159 条的形状
    const s = makeSample({
      T: 60,
      leftVisible: () => true,
      leftBend: stillThenMove(20),
    });

    expect(detectSignSpan(s, NO_PAD).reason).toBe("full_span"); // 没标定 = 今天的行为

    const span = detectSignSpan(s, WITH_CAL);
    expect(span.arrival?.reason).toBe("no_rise");
    expect(span.tactile!.ran).toBe(true);
    expect(span.applied).toBe(true);
    expect(span.tactile!.headFrames).toBeGreaterThan(0);
    expect(span.startFrame).toBeGreaterThan(14);
    expect(span.endFrame).toBe(60); // 一路动到最后，尾巴没得裁
  });

  it("无视觉样本靠触觉裁到 —— 那 3 条 no_vision 的修法", () => {
    const s = makeSample({ T: 60, leftBend: stillThenMove(20) });
    expect(detectSignSpan(s, NO_PAD).reason).toBe("no_vision");

    const span = detectSignSpan(s, WITH_CAL);
    expect(span.applied).toBe(true);
    expect(span.reason).toBe("applied");
    expect(span.arrival?.reason).toBe("no_landmarks");
    expect(span.tactile!.headFrames).toBeGreaterThan(0);
    expect(span.startFrame).toBeGreaterThan(14);
  });

  it("尾部静止被收回 —— 前两层从来不裁尾", () => {
    const s = makeSample({
      T: 60,
      leftVisible: () => true, // 手不出画：可见段的 end 恒等于整条
      leftBend: (t) => (t < 30 ? 10 + t * 6 : 190),
    });
    expect(detectSignSpan(s, NO_PAD).endFrame).toBe(60);

    const span = detectSignSpan(s, WITH_CAL);
    expect(span.tactile!.tailFrames).toBeGreaterThan(0);
    expect(span.endFrame).toBeLessThan(45);
  });

  it("第二层已给出结论时不再用触觉推头（速度谷底更准，且钳位已做过保守修正）", () => {
    const s = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16),
      leftBend: stillThenMove(8),
    });
    const span = detectSignSpan(s, WITH_CAL);
    expect(span.arrival?.reason).toBe("applied");
    expect(span.tactile!.ran).toBe(true);
    expect(span.tactile!.leadingMs).toBeGreaterThan(0); // 量到了
    expect(span.tactile!.headFrames).toBe(0); // 但没采用
    expect(span.startFrame).toBe(span.arrival!.frame);
  });

  it("两端都裁完剩不下 minKeptMs → 整条不裁，但测量值照样报出来", () => {
    // 中间只有 4 帧在动的录制。裁完只剩约 100ms，低于 minKeptMs=250
    const s = makeSample({
      T: 40,
      leftBend: (t) => (t >= 18 && t <= 21 ? 200 : 20),
    });
    const span = detectSignSpan(s, WITH_CAL);
    expect(span.reason).toBe("too_short");
    expect(span.applied).toBe(false);
    expect(span.startFrame).toBe(0);
    expect(span.endFrame).toBe(40);
    // 判据跑过这件事必须留痕 —— 否则界面上分不清"不该裁"和"没跑"
    expect(span.tactile!.ran).toBe(true);
    expect(span.tactile!.leadingMs).toBeGreaterThan(0);
    expect(span.tactile!.trailingMs).toBeGreaterThan(0);
  });

  it("不给 ranges 时与今天逐位相同（防默认值漂移）", () => {
    const cases: SequenceSample[] = [
      makeSample({ T: 60, leftBend: stillThenMove(20) }),
      makeSample({ T: 60, leftVisible: () => true, leftBend: stillThenMove(20) }),
      makeSample({ T: 40, leftVisible: (t) => t >= 10, leftWrist: riseThenHold(10, 25) }),
    ];
    for (const s of cases) {
      const off = detectSignSpan(s, { ...NO_PAD, ranges: null });
      const dflt = detectSignSpan(s, NO_PAD);
      expect(dflt.startFrame).toBe(off.startFrame);
      expect(dflt.endFrame).toBe(off.endFrame);
      expect(dflt.reason).toBe(off.reason);
      expect(dflt.tactile!.ran).toBe(false);
      expect(dflt.tactile!.headFrames).toBe(0);
      expect(dflt.tactile!.tailFrames).toBe(0);
    }
  });

  it("只标了一只手但两只手都有数据 → 整层不跑（兜底量程会让静止段偏短）", () => {
    const s = makeSample({ T: 60, leftBend: stillThenMove(20) });
    // 右手也有数据了，但 CAL 只有 LH → calibrated 为假
    s.rightSensor = s.leftSensor;
    const span = detectSignSpan(s, WITH_CAL);
    expect(span.tactile!.ran).toBe(false);
    expect(span.reason).toBe("no_vision");
  });
});

// ===== visibleRuns =====

describe("visibleRuns", () => {
  it("全程可见 = 一段，覆盖整条", () => {
    const v = visibleRuns(makeSample({ T: 30, leftVisible: () => true }));
    expect(v.runs).toEqual([{ start: 0, end: 30 }]);
    expect(v.visibleFrames).toBe(30);
    expect(v.totalFrames).toBe(30);
  });

  it("中途掉手切成多段，段界是半开区间", () => {
    // 可见 [0,10) + [20,30)，中间 10 帧掉手
    const v = visibleRuns(
      makeSample({ T: 30, leftVisible: (t) => t < 10 || t >= 20 })
    );
    expect(v.runs).toEqual([
      { start: 0, end: 10 },
      { start: 20, end: 30 },
    ]);
    expect(v.visibleFrames).toBe(20);
  });

  it("短于 minRunFrames 的段被滤掉，但仍计入 visibleFrames", () => {
    // visibleFrames 统计的是"摄像头到底看见了多少帧"，滤掉的是"够不够格定边界"。
    // 两个数混成一个的话，界面上就分不出"掉手很多"和"误检很多"
    const s = makeSample({ T: 30, leftVisible: (t) => t < 2 || t >= 20 });
    const v = visibleRuns(s, 3);
    expect(v.runs).toEqual([{ start: 20, end: 30 }]);
    expect(v.visibleFrames).toBe(12); // 2 + 10
  });

  it("双手取并集（单手词的另一只手整条不可见，取交集等于关掉裁剪）", () => {
    const s = makeSample({
      T: 30,
      leftVisible: (t) => t < 10,
      rightVisible: (t) => t >= 20,
    });
    expect(visibleRuns(s).runs).toEqual([
      { start: 0, end: 10 },
      { start: 20, end: 30 },
    ]);
  });

  it("两只手都没有关键点数组 → 零段", () => {
    const s = makeSample({ T: 30, leftVisible: null, rightVisible: null });
    const v = visibleRuns(s);
    expect(v.runs).toEqual([]);
    expect(v.visibleFrames).toBe(0);
    expect(v.totalFrames).toBe(30);
  });
});

// ===== trimSpanForExport：句子样本的两处分流 =====

/*
 * 句子采集页开了摄像头之后，句子录制第一次带上关键点 —— 于是**第一层和第二层
 * 都第一次在句子上真的跑起来**，而两层的默认口径都假设「这一条里只有一个手势」：
 *
 * - 第二层（到位检测）找第一个速度谷底，认定那之前都是抬手 transport。
 *   一条句子里那个谷底是**第 1 个词打完的位置**，采用它等于把第一个词整个切掉。
 * - 第一层默认取**最长连续可见段**（`longest_run`）。句子 4~7 秒，MediaPipe 中途
 *   掉一次手就把录制劈成两段，较短的那半整段被丢掉。实测 45 条真实句里 13 条被
 *   打断成 2~4 段，最坏的一条 301 个可见帧只留下 96 个 —— 前 4 秒的动作全没了。
 *   （最初这里写的是「第一层对句子是安全的，只切没看见手的两头」，那是错的：
 *   它不是首尾裁剪。这段注释本身就曾经掩护过这个 bug。）
 *
 * 两者的症状都是「句首照样错」—— 与这一轮改动的靶子（句首错误）完全重合，
 * 事后几乎不可能想到是裁剪把第一个词吃掉了。所以这里把两处分流都锁死。
 *
 * 第三层（触觉静止段）对句子确实安全：只切真正没人动的两头，不假设手势个数。
 */
describe("trimSpanForExport —— 句子样本的两处分流", () => {
  /** 同一条录制，只把 segments 换成多词 —— `isSentenceSample` 只看这一个字段 */
  const asSentence = (s: SequenceSample): SequenceSample => ({
    ...s,
    segments: [
      { label: "i", startFrame: 0, endFrame: 13 },
      { label: "help", startFrame: 13, endFrame: 26 },
      { label: "you", startFrame: 26, endFrame: s.frameCount },
    ],
  });

  it("同一条录制：当词裁掉抬手段，当句子一帧不裁", () => {
    // 手全程在画面里、前 16 帧在上抬 —— 第二层唯一能处理的形状。
    // 当成句子时那 16 帧就是「第 1 个词」，切掉它就是本模块最怕的那种静默错误
    const word = makeSample({
      T: 40,
      leftVisible: () => true,
      leftWrist: riseThenHold(0, 16),
    });

    const w = trimSpanForExport(word);
    expect(w.applied).toBe(true);
    expect(w.reason).toBe("applied");
    expect(w.startFrame).toBeGreaterThanOrEqual(15);

    const s = trimSpanForExport(asSentence(word));
    expect(s.startFrame).toBe(0);
    expect(s.endFrame).toBe(40);
    expect(s.applied).toBe(false);
    // full_span 而不是 no_vision —— 第一层确实跑过，只是这条没什么可裁
    expect(s.reason).toBe("full_span");
  });

  it("句子仍然吃第一层：画面外那半截照样裁掉", () => {
    // 关掉的只有第二层。可见性不假设手势个数，对句子照样成立
    const s = asSentence(
      makeSample({ T: 40, leftVisible: (t) => t >= 10, leftWrist: riseThenHold(10, 25) })
    );
    const span = trimSpanForExport(s);
    expect(span.applied).toBe(true);
    // padMs=100 @30fps ≈ 3 帧留白，所以停在 10 之前几帧而不是 10。
    // 恰好 100.0 的那一帧取不取，取决于 Float32 时间戳的舍入 —— 锁区间不锁帧号
    expect(span.startFrame).toBeGreaterThanOrEqual(7);
    expect(span.startFrame).toBeLessThanOrEqual(8);
    expect(span.endFrame).toBe(40);
  });

  it("句子仍然吃第三层：尾部静止照样裁掉（唯一裁尾的一层）", () => {
    // 「打完了还举着等按键」那一段在句子上尤其长（要等收句判据），不裁就是
    // 训练和 sentenceEnvelope 两种时间口径
    const base = makeSample({
      T: 60,
      leftVisible: () => true,
      leftBend: (t) => (t < 30 ? 10 + t * 6 : 190),
    });
    const s = asSentence(base);
    expect(trimSpanForExport(s).endFrame).toBe(60); // 不给标定 → 第三层不跑
    const span = trimSpanForExport(s, CAL);
    expect(span.tactileRan).toBe(true);
    expect(span.applied).toBe(true);
    expect(span.endFrame).toBeLessThan(50);
    expect(span.startFrame).toBe(0); // 头上没有静止段，别顺手裁头
  });

  /*
   * 下面三条锁的是第一层的分流。夹具形状都是「入画晚 + 中途掉一次手」，
   * 这是 45 条真实句里 13 条的真实形状，不是编出来的边角情形。
   *
   * padMs=100 @30fps ≈ 3 帧，而恰好 100.0 的那一帧取不取取决于 Float32 时间戳的
   * 舍入 —— 所以起止都锁区间不锁帧号（与本文件其它留白断言同口径）。
   */

  it("掉手把录制劈开：词只留最长段，句子跨过空洞", () => {
    // 可见 [10,27) 17 帧 + 掉手 10 帧 + 可见 [37,60) 23 帧。
    // longest_run 会选后一段，把前 17 帧**可见的打手语动作**整段丢掉 ——
    // 那 17 帧里装着句子的第 1 个词
    const dropout = makeSample({
      T: 60,
      leftVisible: (t) => (t >= 10 && t < 27) || t >= 37,
    });

    const w = trimSpanForExport(dropout);
    expect(w.applied).toBe(true);
    expect(w.startFrame).toBeGreaterThanOrEqual(34); // 前一段整个不要了
    expect(w.endFrame).toBe(60);

    const s = trimSpanForExport(asSentence(dropout));
    expect(s.applied).toBe(true);
    expect(s.reason).toBe("applied");
    expect(s.startFrame).toBeGreaterThanOrEqual(7);
    expect(s.startFrame).toBeLessThanOrEqual(8);
    expect(s.endFrame).toBe(60); // 空洞留在区间里
  });

  it("短段在后也一样（不依赖哪一段更长的先后）", () => {
    // 与上一条把两段长度对调：longest_run 这次砍的是**尾**。
    // `bestLen > len` 的严格大于让等长时取靠前那段，方向依赖必须两头都锁
    const dropout = makeSample({
      T: 60,
      leftVisible: (t) => (t >= 10 && t < 33) || t >= 43,
    });

    const w = trimSpanForExport(dropout);
    expect(w.applied).toBe(true);
    expect(w.endFrame).toBeLessThanOrEqual(36); // 后一段整个不要了

    const s = trimSpanForExport(asSentence(dropout));
    expect(s.startFrame).toBeGreaterThanOrEqual(7);
    expect(s.startFrame).toBeLessThanOrEqual(8);
    expect(s.endFrame).toBe(60);
  });

  it("句首的孤立误检帧不能定起点", () => {
    // `first_to_last` 取的是"第一段"，所以必须先过 minRunFrames —— 否则抬手途中
    // 被摄像头扫到的两帧就成了句子起点，等于整条不裁。这正是当初"取最长而不是
    // 第一段"想防的事，靠过滤防住，不靠取最长
    const s = asSentence(
      makeSample({ T: 60, leftVisible: (t) => t < 2 || t >= 15 })
    );
    const span = trimSpanForExport(s);
    expect(span.applied).toBe(true);
    expect(span.startFrame).toBeGreaterThanOrEqual(12);
    expect(span.startFrame).toBeLessThanOrEqual(13);
  });
});

// ===== visibleSpan / visibleRunCount =====

describe("detectSignSpan —— visibleSpan 与 visibleRunCount", () => {
  const dropout = () =>
    makeSample({ T: 60, leftVisible: (t) => t < 15 || (t >= 25 && t < 40) });

  it("默认值仍是 longest_run", () => {
    // 翻掉这个默认值会静默改变**全部 399 条词录制**的裁剪，从而改变词模型准确率，
    // 而症状只是"这次训出来的数字和上次不一样"
    expect(DEFAULT_TRIM.visibleSpan).toBe("longest_run");
  });

  it("同一条录制，两种模式给出不同区间", () => {
    // 可见 [0,15) 与 [25,40)，两段等长 → longest_run 取靠前那段
    const longest = detectSignSpan(dropout(), NO_PAD);
    expect(longest.startFrame).toBe(0);
    expect(longest.endFrame).toBe(15);

    const spanning = detectSignSpan(dropout(), {
      ...NO_PAD,
      visibleSpan: "first_to_last",
    });
    expect(spanning.startFrame).toBe(0);
    expect(spanning.endFrame).toBe(40);
  });

  it("visibleRunCount 报出可见段个数（掉手的唯一可观测量）", () => {
    expect(detectSignSpan(dropout(), NO_PAD).visibleRunCount).toBe(2);
    expect(
      detectSignSpan(makeSample({ T: 30, leftVisible: () => true }), NO_PAD)
        .visibleRunCount
    ).toBe(1);
  });

  it("没有关键点时 visibleRunCount 为 0（不是 1）", () => {
    // 0 = "第一层没跑过"，1 = "跑过，手全程在画面里"。混成一个数的话，
    // 界面上就没法区分"没开摄像头"和"录得很干净"
    const s = makeSample({ T: 30, leftVisible: null, rightVisible: null });
    const span = detectSignSpan(s, NO_PAD);
    expect(span.visibleRunCount).toBe(0);
    expect(span.reason).toBe("no_vision");
  });
});

// ===== spanToGridBounds =====

describe("spanToGridBounds", () => {
  it("按时间戳换算成归一化边界", () => {
    const s = makeSample({ T: 30, leftVisible: (t) => t >= 10 });
    const span = detectSignSpan(s, NO_PAD);
    const { a, b } = spanToGridBounds(s, span);
    expect(a).toBeCloseTo(10 / 29, 6);
    expect(b).toBeCloseTo(1, 6);
  });

  it("丢帧的录制上按时间戳算，与按帧数比例算不同", () => {
    const T = 14;
    const s = makeSample({ T, leftVisible: (t) => t >= 5 });
    // 人为制造丢帧：入画之后每帧间隔变成 100ms
    for (let t = 5; t < T; t++)
      s.timestamps[t] = s.timestamps[4] + (t - 4) * 100;
    const span = detectSignSpan(s, NO_PAD);
    expect(span.applied).toBe(true);
    const { a } = spanToGridBounds(s, span);
    const total = s.timestamps[T - 1] - s.timestamps[0];
    expect(a).toBeCloseTo((s.timestamps[5] - s.timestamps[0]) / total, 6);
    // 按帧下标比例会得到 5/13 ≈ 0.385，与按时间戳的 ≈0.226 差得很远 ——
    // 这条就是为了锁住"必须按时间戳换算"
    expect(Math.abs(a - 5 / (T - 1))).toBeGreaterThan(0.1);
  });

  it("不裁时给出整条 [0,1]", () => {
    const s = makeSample({ T: 30, leftVisible: () => true });
    const span = detectSignSpan(s, NO_PAD);
    expect(spanToGridBounds(s, span)).toEqual({ a: 0, b: 1 });
  });
});

// ===== summarizeTrim =====

describe("summarizeTrim", () => {
  it("分别统计裁成的条数与各种没裁成的原因", () => {
    const samples = [
      makeSample({ T: 30, leftVisible: (t) => t >= 10 }), // applied
      makeSample({ T: 30, leftVisible: (t) => t >= 15 }), // applied
      makeSample({ T: 30 }), // no_vision
      makeSample({ T: 30, leftVisible: () => false }), // no_run
      makeSample({ T: 30, leftVisible: () => true }), // full_span
    ];
    const st = summarizeTrim(samples, NO_PAD);
    expect(st.total).toBe(5);
    expect(st.applied).toBe(2);
    expect(st.skipped).toEqual({
      no_vision: 1,
      no_run: 1,
      too_short: 0,
      full_span: 1,
    });
    // 只对被裁的两条求平均：19/29 与 14/29
    expect(st.meanKeptRatio).toBeCloseTo((19 / 29 + 14 / 29) / 2, 6);
    // 这批夹具手腕不动，第二层一条都不生效
    expect(st.arrivalApplied).toBe(0);
    expect(st.meanArrivalDroppedMs).toBe(0);
  });

  it("到位检测单列统计（它能把本来 full_span 的样本变成裁过的）", () => {
    const samples = [
      makeSample({
        T: 40,
        leftVisible: () => true,
        leftWrist: riseThenHold(0, 16),
      }),
      makeSample({
        T: 40,
        leftVisible: () => true,
        leftWrist: riseThenHold(0, 16),
        leftBend: (t) => (t >= 5 ? 200 : 10),
      }),
      makeSample({ T: 40, leftVisible: () => true }), // 手腕不动 → full_span
    ];
    const st = summarizeTrim(samples, NO_PAD);
    expect(st.applied).toBe(2);
    expect(st.skipped.full_span).toBe(1);
    expect(st.arrivalApplied).toBe(2);
    expect(st.arrivalBendClamped).toBe(1);
    expect(st.meanArrivalDroppedMs).toBeGreaterThan(0);
  });

  it("一条都没裁成时 meanKeptRatio 为 1", () => {
    const st = summarizeTrim([makeSample({ T: 30 })], NO_PAD);
    expect(st.applied).toBe(0);
    expect(st.meanKeptRatio).toBe(1);
  });

  it("空数据集不崩", () => {
    const st = summarizeTrim([], NO_PAD);
    expect(st.total).toBe(0);
    expect(st.applied).toBe(0);
    expect(st.meanKeptRatio).toBe(1);
  });
});
