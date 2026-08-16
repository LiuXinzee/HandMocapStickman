import { describe, it, expect } from "vitest";
import {
  DEFAULT_TRIM,
  detectSignSpan,
  spanToGridBounds,
  summarizeTrim,
  type TrimConfig,
} from "./sequenceTrim";
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

  it("整条没有视觉时这一层不参与（arrival 为 null，不是硬报错）", () => {
    const span = detectSignSpan(makeSample({ T: 30 }), NO_PAD);
    expect(span.reason).toBe("no_vision");
    expect(span.arrival).toBeNull();
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
