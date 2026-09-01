/*
 * fistGate.test —— 拇指压力闸门
 *
 * 这条闸门会**改掉模型的输出**，所以测试要锁的重点不是"能改对"，而是
 * "不该改的时候一动不动"。三条最要紧的：
 *
 *  1. **top-2 约束不能失效。** 少了它，难过那 25.9% 峰值 ≥5 的窗会把本来
 *     正确的难过改成谢谢 —— 症状是难过时好时坏，且看不出是闸门干的。
 *  2. **没有右手数据时必须不介入。** `thumbPeak` 返回 -1 而不是 0，因为 0 是
 *     合法读数（拇指没吃力）。这两者混掉的后果是丢手套时所有谢谢都变难过。
 *  3. **峰值必须是单点峰值、不是 12 点均值。** 均值口径下这两个词不可分
 *     （d'=0.64），所以有一条测试专门锁住"12 个点里只有 1 个高也算"。
 */
import { describe, expect, it } from "vitest";
import {
  applyFistGate,
  isGatedPair,
  thumbPeak,
  THUMB_PEAK_GATE,
  GATED_PAIR,
} from "./fistGate";
import { SEQ_SENSOR_N, type SequenceSample } from "./datasetStore";
import { getWordById } from "./signLanguageVocab";
import { ROT_RATE_HI, ROT_RATE_LO, type RotationReading } from "./rotationRate";

/** 造一条 T 帧的样本，右手拇指 12 点按 `thumbValues(t, k)` 填 */
function sample(
  T: number,
  thumbValues: (t: number, k: number) => number,
  opts: { noRight?: boolean } = {}
): SequenceSample {
  const right = new Uint8Array(T * SEQ_SENSOR_N);
  for (let t = 0; t < T; t++)
    for (let k = 0; k < 12; k++)
      right[t * SEQ_SENSOR_N + k] = thumbValues(t, k);
  return {
    segments: [{ label: "x", startFrame: 0, endFrame: T }],
    primaryLabel: "x",
    frameCount: T,
    timestamps: Float32Array.from({ length: T }, (_, i) => i * 20),
    leftSensor: null,
    rightSensor: opts.noRight ? null : right,
    leftImu: null,
    rightImu: null,
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: T * 20,
    sourceFps: 50,
    origin: "recorded",
    timestamp: 0,
  } as SequenceSample;
}

describe("thumbPeak", () => {
  it("取的是单点峰值 —— 12 个点里只有 1 个高也要算出来", () => {
    // 这一条锁的是"别改回 12 点均值"：均值口径下 12/12 = 1.0，够不到阈值 5
    const s = sample(30, (_t, k) => (k === 3 ? 12 : 0));
    expect(thumbPeak(s)).toBe(12);
    expect(thumbPeak(s)).toBeGreaterThanOrEqual(THUMB_PEAK_GATE);
  });

  it("取的是整个窗口的峰值 —— 只在一帧上出现的尖峰也要算出来", () => {
    // "下压一次"就是这种形状：30 帧里只有 1 帧有力
    expect(thumbPeak(sample(30, (t, k) => (t === 17 && k === 5 ? 20 : 0)))).toBe(20);
  });

  it("拇指全程没吃力就是 0", () => {
    expect(thumbPeak(sample(30, () => 0))).toBe(0);
  });

  it("没有右手数据返回 -1，不是 0", () => {
    // 0 是合法读数，用它当"没数据"会让丢手套时所有谢谢都被改成难过
    expect(thumbPeak(sample(30, () => 12, { noRight: true }))).toBe(-1);
  });

  it("只读 0-11，不会串到食指去", () => {
    const s = sample(10, () => 0);
    s.rightSensor![12] = 200; // 食指压力第一个点
    expect(thumbPeak(s)).toBe(0);
  });
});

describe("isGatedPair", () => {
  it("这一对，两种顺序都算", () => {
    expect(isGatedPair("thank_you", "sad")).toBe(true);
    expect(isGatedPair("sad", "thank_you")).toBe(true);
  });

  it("只有一个在对里不算", () => {
    expect(isGatedPair("thank_you", "hello")).toBe(false);
    expect(isGatedPair("sad", "angry")).toBe(false);
  });

  it("没有第二名不算（27 类模型总有第二名，这是防御性的）", () => {
    expect(isGatedPair("thank_you", undefined)).toBe(false);
  });
});

describe("applyFistGate", () => {
  const HI = THUMB_PEAK_GATE + 7; // 谢谢实测中位 12
  const LO = 0; // 难过实测中位 0

  it("模型说谢谢、拇指没吃力 → 改判难过", () => {
    const r = applyFistGate("thank_you", "sad", LO);
    expect(r.label).toBe("sad");
    expect(r.changed).toBe(true);
    expect(r.reason).toBe("thumb_idle_so_not_thanks");
  });

  it("模型说难过、拇指吃了力 → 改判谢谢", () => {
    const r = applyFistGate("sad", "thank_you", HI);
    expect(r.label).toBe("thank_you");
    expect(r.changed).toBe(true);
  });

  it("模型和拇指一致时一动不动", () => {
    expect(applyFistGate("thank_you", "sad", HI).changed).toBe(false);
    expect(applyFistGate("sad", "thank_you", LO).changed).toBe(false);
  });

  it("**top-2 不是这一对时绝不介入** —— 少了这条，难过会时好时坏", () => {
    // 难过实测有 25.9% 的窗峰值 ≥5（画圈时拇指碰到手指）。模型笃定输出难过、
    // 谢谢连前二都排不上时，这些窗一律不能动
    const r = applyFistGate("sad", "angry", HI);
    expect(r.label).toBe("sad");
    expect(r.changed).toBe(false);
    expect(r.reason).toBe("not_gated_pair");
  });

  it("别的词一律原样返回", () => {
    for (const l of ["hello", "you", "love", "angry", "_idle"]) {
      expect(applyFistGate(l, "sad", HI).label).toBe(l);
      expect(applyFistGate(l, "thank_you", LO).label).toBe(l);
    }
  });

  it("没有右手数据时不介入", () => {
    const r = applyFistGate("thank_you", "sad", -1);
    expect(r.changed).toBe(false);
    expect(r.reason).toBe("no_right_hand");
  });

  it("阈值边界：恰好等于阈值算「吃了力」", () => {
    expect(applyFistGate("thank_you", "sad", THUMB_PEAK_GATE).changed).toBe(false);
    expect(applyFistGate("thank_you", "sad", THUMB_PEAK_GATE - 1).changed).toBe(true);
  });

  it("阈值可以按实时读数覆盖", () => {
    expect(applyFistGate("thank_you", "sad", 8, 20).label).toBe("sad");
    expect(applyFistGate("thank_you", "sad", 8, 5).label).toBe("thank_you");
  });
});

/*
 * 转角速率成了主判据之后（2026-08-31），要锁的是**两个判据的优先级**，
 * 以及"速率读数不可用"时必须原封不动地退回上面那一整套拇指行为 ——
 * 上面那 8 条测试一个字没改，它们现在同时也是"退回路径"的回归测试。
 */
describe("applyFistGate：转角速率是主判据", () => {
  const HI = THUMB_PEAK_GATE + 7;
  const LO = 0;
  const rot = (rate: number, usable = true): RotationReading => ({
    ratePerSec: rate,
    totalDeg: rate * 2,
    spanMs: 2000,
    usable,
  });
  /** 手在画圈 / 手基本不动，两个都取到实测分布的中位附近 */
  const ROTATING = rot(90);
  const STILL = rot(33);

  it("手在转 → 一定不是谢谢，**即使拇指吃了力**", () => {
    // 这一条是整个改动的目的：难过画圈时拇指偶尔会碰到手指（实测 25.9% 的窗
    // 峰值 ≥5），原来那些窗会被拇指判成谢谢。现在转角说了算
    const r = applyFistGate("sad", "thank_you", HI, THUMB_PEAK_GATE, ROTATING);
    expect(r.label).toBe("sad");
    expect(r.changed).toBe(false);
    expect(r.reason).toBe("rotating_so_not_thanks");
  });

  it("手在转、模型说谢谢 → 改判难过", () => {
    const r = applyFistGate("thank_you", "sad", HI, THUMB_PEAK_GATE, ROTATING);
    expect(r.label).toBe("sad");
    expect(r.changed).toBe(true);
    expect(r.reason).toBe("rotating_so_not_thanks");
  });

  it("手没在转、模型说难过 → 改判谢谢，**即使拇指没吃力**", () => {
    // 使用者报的症状正是这个：打「谢谢」只竖拇指没下压 → 拇指读数低 →
    // 原来的闸门必然判成难过。转角判据在同样情况下判对
    const r = applyFistGate("sad", "thank_you", LO, THUMB_PEAK_GATE, STILL);
    expect(r.label).toBe("thank_you");
    expect(r.changed).toBe(true);
    expect(r.reason).toBe("still_so_thanks");
  });

  it("速率落在中间带 → 交给拇指，行为与没有速率读数时一致", () => {
    const MID = rot((ROT_RATE_LO + ROT_RATE_HI) / 2);
    for (const [t1, t2, peak] of [
      ["thank_you", "sad", LO],
      ["sad", "thank_you", HI],
      ["thank_you", "sad", HI],
      ["sad", "thank_you", LO],
    ] as const) {
      const withRot = applyFistGate(t1, t2, peak, THUMB_PEAK_GATE, MID);
      const without = applyFistGate(t1, t2, peak);
      expect(withRot).toEqual(without);
    }
  });

  it("**读数不可用时整条退回拇指** —— 不能当成「静止 → 谢谢」", () => {
    // usable:false 的三种来源（没 IMU / 区间太短 / 四元数全零）都走这里。
    // 当成 still 的后果是没有 IMU 的手套上所有难过都变谢谢
    for (const r of [rot(0, false), rot(90, false), null, undefined]) {
      expect(applyFistGate("thank_you", "sad", LO, THUMB_PEAK_GATE, r).label).toBe("sad");
      expect(applyFistGate("sad", "thank_you", HI, THUMB_PEAK_GATE, r).label).toBe(
        "thank_you"
      );
    }
  });

  it("转角判据不需要压力数据 —— 丢手/压力阵列坏掉时仍然判得出", () => {
    // peak = -1（没有右手压力）。原来这里直接 return no_right_hand，
    // 把更可靠的判据挡在门外了
    expect(applyFistGate("thank_you", "sad", -1, THUMB_PEAK_GATE, ROTATING).label).toBe(
      "sad"
    );
    expect(applyFistGate("sad", "thank_you", -1, THUMB_PEAK_GATE, STILL).label).toBe(
      "thank_you"
    );
    // 但速率也不可用时仍然是 no_right_hand
    expect(
      applyFistGate("thank_you", "sad", -1, THUMB_PEAK_GATE, rot(90, false)).reason
    ).toBe("no_right_hand");
  });

  it("top-2 约束优先于转角 —— 模型笃定时转角也不许动它", () => {
    const r = applyFistGate("sad", "angry", HI, THUMB_PEAK_GATE, STILL);
    expect(r.changed).toBe(false);
    expect(r.reason).toBe("not_gated_pair");
  });

  it("别的词一律原样返回（转角说什么都不管）", () => {
    for (const l of ["hello", "you", "love", "_idle"]) {
      expect(applyFistGate(l, "sad", LO, THUMB_PEAK_GATE, ROTATING).label).toBe(l);
      expect(applyFistGate(l, "thank_you", HI, THUMB_PEAK_GATE, STILL).label).toBe(l);
    }
  });
});

describe("被仲裁的那一对必须是词表里真实存在的 id", () => {
  it("打错一个字的表现是闸门静默失效，不是崩溃", () => {
    for (const id of GATED_PAIR) {
      expect(getWordById(id), `${id} 不在词表里`).toBeTruthy();
    }
  });
});
