/*
 * rotationRate.test —— 谢谢/难过 那条闸门的**主判据**
 *
 * 这批测试要钉住四个会静默出错的地方：
 *
 *  1. **算的必须是累计路径长度，不是端点夹角。** 转回原位的一圈是"难过"的
 *     全部特征；算成端点夹角的话它读出接近 0，闸门会把每一次难过都判成谢谢。
 *     实测这两种算法差 71.0% vs 97.7%。
 *  2. **q 与 −q 同解必须处理。** IMU 会随机跳符号，不取 |w| 的话静止的手每跳
 *     一次符号就记 180°，读数飙到几千度 —— 症状是"谢谢总被判成难过"。
 *  3. **`usable: false` 不等于"没在转"。** 没 IMU / 区间太短 / 四元数全零都归到
 *     这一档，闸门必须整条退回拇指判据，而不是当成"静止 → 谢谢"。
 *  4. **短区间必须自己关掉。** `ROT_MIN_SPAN_MS` 是从实测交叉点定的（1200~1500ms
 *     之间速率判据由劣于拇指转为优于拇指）。这一条丢了，句子路径上大量 750ms
 *     的词区间会用一个已知更差的判据去改模型的输出。
 */
import { describe, expect, it } from "vitest";
import {
  ROT_MIN_SPAN_MS,
  ROT_RATE_HI,
  ROT_RATE_LO,
  rotationBand,
  rotationRate,
  rotationRateSpan,
  type RotationReading,
} from "./rotationRate";
import { SEQ_IMU_N, type SequenceSample } from "./datasetStore";
import { quatFromAxisAngle, quatMul, type Quat } from "./sequenceFeatures";

const FPS = 50;
const DT_MS = 1000 / FPS;

/** 造一条只有右手 IMU 四元数的样本；`quatAt(t)` 给第 t 帧的姿态 */
function sample(
  T: number,
  quatAt: (t: number) => Quat,
  opts: { noImu?: boolean; dtMs?: number } = {}
): SequenceSample {
  const dt = opts.dtMs ?? DT_MS;
  const imu = new Float32Array(T * SEQ_IMU_N);
  for (let t = 0; t < T; t++) {
    const q = quatAt(t);
    for (let k = 0; k < 4; k++) imu[t * SEQ_IMU_N + k] = q[k];
  }
  return {
    segments: [{ label: "x", startFrame: 0, endFrame: T }],
    primaryLabel: "x",
    frameCount: T,
    timestamps: Float32Array.from({ length: T }, (_, i) => i * dt),
    leftSensor: null,
    rightSensor: null,
    leftImu: null,
    rightImu: opts.noImu ? null : imu,
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: (T - 1) * dt,
    sourceFps: FPS,
    origin: "recorded",
    timestamp: 0,
  } as SequenceSample;
}

const IDENTITY: Quat = [1, 0, 0, 0];

/** 绕 Z 轴匀速转，每帧 `degPerFrame` 度 */
function spin(degPerFrame: number): (t: number) => Quat {
  return (t) => quatFromAxisAngle([0, 0, 1], (t * degPerFrame * Math.PI) / 180);
}

/** 帧数够 2 秒（满足 ROT_MIN_SPAN_MS） */
const T2S = FPS * 2 + 1;

describe("rotationRateSpan：算的是累计路径长度", () => {
  it("匀速自转：速率 = 每帧度数 × 帧率", () => {
    const r = rotationRate(sample(T2S, spin(2))); // 2°/帧 × 50fps = 100°/s
    expect(r.usable).toBe(true);
    expect(r.ratePerSec).toBeCloseTo(100, 0);
    expect(r.totalDeg).toBeCloseTo(2 * (T2S - 1), 0);
  });

  it("转出去再转回原位 —— 路径转角照记，端点夹角是 0", () => {
    // 前一半绕 Z 转到 +90°，后一半转回 0°。首末帧姿态相同
    const half = Math.floor((T2S - 1) / 2);
    const s = sample(T2S, (t) => {
      const deg = t <= half ? (90 * t) / half : (90 * (T2S - 1 - t)) / (T2S - 1 - half);
      return quatFromAxisAngle([0, 0, 1], (deg * Math.PI) / 180);
    });
    const r = rotationRate(s);
    // 走过 90 + 90 = 180 度；如果错写成端点夹角，这里会是 ~0
    expect(r.totalDeg).toBeCloseTo(180, 0);
    expect(r.ratePerSec).toBeGreaterThan(ROT_RATE_HI);
  });

  it("完全静止 → 0°/s（不是 -1，静止是合法读数）", () => {
    const r = rotationRate(sample(T2S, () => IDENTITY));
    expect(r.usable).toBe(true);
    expect(r.ratePerSec).toBe(0);
  });

  it("四元数随机跳符号（q 与 −q 同解）时读数不受影响", () => {
    const base = spin(2);
    const flipped = sample(T2S, (t) => {
      const q = base(t);
      // 每隔几帧整个四元数取负 —— 真实 IMU 会这么干
      return t % 3 === 0 ? (q.map((v) => -v) as Quat) : q;
    });
    const r = rotationRate(flipped);
    // 不处理同解的话每次跳符号记 180°，这里会读到 3000+ °/s
    expect(r.ratePerSec).toBeCloseTo(100, 0);
  });

  it("绕不同轴的复合转动也按路径累加", () => {
    const s = sample(T2S, (t) =>
      quatMul(
        quatFromAxisAngle([0, 0, 1], (t * 1 * Math.PI) / 180),
        quatFromAxisAngle([1, 0, 0], (t * 1 * Math.PI) / 180)
      )
    );
    const r = rotationRate(s);
    // 两个 1°/帧 的分量，合成单帧转角在 1~2° 之间 → 50~100 °/s
    expect(r.ratePerSec).toBeGreaterThan(50);
    expect(r.ratePerSec).toBeLessThan(100);
  });
});

describe("rotationRateSpan：什么时候读数不可用", () => {
  it("没有右手 IMU → usable false，且速率不是 0", () => {
    const r = rotationRate(sample(T2S, spin(2), { noImu: true }));
    expect(r.usable).toBe(false);
    expect(r.ratePerSec).toBe(-1); // 给 0 会被当成"静止"
  });

  it("四元数全零（槽位没数据）→ usable false", () => {
    const r = rotationRate(sample(T2S, () => [0, 0, 0, 0]));
    expect(r.usable).toBe(false);
  });

  it("四元数有 NaN → usable false，而不是算出 NaN 速率", () => {
    const r = rotationRate(sample(T2S, (t) => (t === 5 ? [NaN, 0, 0, 0] : IDENTITY)));
    expect(r.usable).toBe(false);
    expect(Number.isNaN(r.ratePerSec)).toBe(false);
  });

  it("区间短于 ROT_MIN_SPAN_MS → usable false，但速率照样算出来（给界面读）", () => {
    // 1 秒的样本：转得飞快，但区间不够长，判据必须自己关掉
    const s = sample(FPS + 1, spin(4));
    const r = rotationRate(s);
    expect(r.spanMs).toBeLessThan(ROT_MIN_SPAN_MS);
    expect(r.usable).toBe(false);
    expect(r.ratePerSec).toBeCloseTo(200, 0);
  });

  it("刚好够 ROT_MIN_SPAN_MS 就可用（边界是 >=，不是 >）", () => {
    const T = Math.round(ROT_MIN_SPAN_MS / DT_MS) + 1;
    expect(rotationRate(sample(T, spin(1))).usable).toBe(true);
    expect(rotationRate(sample(T - 1, spin(1))).usable).toBe(false);
  });

  it("只有 1 帧、或区间退化成 1 帧 → usable false（一帧算不出转角）", () => {
    expect(rotationRate(sample(1, () => IDENTITY)).usable).toBe(false);
    // 与 thumbPeakSpan 不同：那边会把退化区间夹到一帧并给出峰值，这边不能
    expect(rotationRateSpan(sample(T2S, spin(2)), 0.5, 0.5).usable).toBe(false);
  });
});

describe("rotationRateSpan：只看指定的那一段", () => {
  it("前一半在转、后一半静止 —— 分段读数不同", () => {
    const half = Math.floor(T2S / 2);
    const q0 = quatFromAxisAngle([0, 0, 1], (half * 2 * Math.PI) / 180);
    const s = sample(T2S * 2, (t) =>
      t < half ? quatFromAxisAngle([0, 0, 1], (t * 2 * Math.PI) / 180) : q0
    );
    const front = rotationRateSpan(s, 0, 0.25);
    const back = rotationRateSpan(s, 0.5, 1);
    expect(front.ratePerSec).toBeGreaterThan(back.ratePerSec);
    /*
     * 不是 toBeCloseTo(0, 5)：四元数在 `SequenceSample` 里存的是 **float32**，
     * 归一化后 |w| 与 1 有 ~1e-9 的舍入残差，acos 把它放大成每帧 ~1e-6 度，
     * 累加一百帧再除以时长 → 一只**完全不动**的手读出 ~1e-4 °/s 的底噪。
     * 阈值是 50 / 70，底噪离得极远、不影响判断 —— 但别把它当成精确的 0 来断言。
     */
    expect(back.ratePerSec).toBeLessThan(0.01);
  });

  it("区间越界时夹进合法范围，不抛错", () => {
    const s = sample(T2S, spin(2));
    expect(rotationRateSpan(s, -1, 2).ratePerSec).toBeCloseTo(100, 0);
  });
});

describe("rotationBand", () => {
  const reading = (rate: number, usable = true): RotationReading => ({
    ratePerSec: rate,
    totalDeg: rate * 2,
    spanMs: 2000,
    usable,
  });

  it("按 LO / HI 分三段", () => {
    expect(rotationBand(reading(ROT_RATE_LO - 1))).toBe("still");
    expect(rotationBand(reading(ROT_RATE_HI))).toBe("rotating");
    expect(rotationBand(reading(ROT_RATE_LO))).toBe("unknown"); // 中间带
    expect(rotationBand(reading(ROT_RATE_HI - 1))).toBe("unknown");
  });

  it("读数不可用 / null / undefined 一律 unknown —— 不能当成 still", () => {
    // 这一条是全文件最要紧的一条：判成 still 就等于"没量到 → 一律谢谢"
    expect(rotationBand(reading(0, false))).toBe("unknown");
    expect(rotationBand(null)).toBe("unknown");
    expect(rotationBand(undefined)).toBe("unknown");
  });

  it("LO < HI，且两个阈值都落在实测重叠区里（改常量时这条会提醒你）", () => {
    // 2000ms 窗实测：谢谢 max 56.5 °/s、难过 min 46.2 °/s
    expect(ROT_RATE_LO).toBeLessThan(ROT_RATE_HI);
    expect(ROT_RATE_LO).toBeGreaterThanOrEqual(46);
    expect(ROT_RATE_HI).toBeGreaterThanOrEqual(57); // 高于谢谢的最大值 → 这一侧干净
  });
});
