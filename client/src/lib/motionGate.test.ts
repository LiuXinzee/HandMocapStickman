/*
 * motionGate.test —— 推理前的动作闸门
 *
 * 这道闸门的两种错法代价完全不对称，测试也按这个来：
 *
 * - **误开**（静止时放行）= 一个错词进历史。这是当初要修的 bug 本身。
 * - **误关**（有动作却拦掉）= 整个词识别不出来，比错词更糟。而最容易误关的
 *   恰恰是**单手词**：只有一只手在动，如果拿两只手的平均去比门限，
 *   动作幅度正常的单手词会被对半砍到门限以下，一个都识别不出来。
 *
 * 所以下面既锁"静止必须拦住"，也锁"单手动作必须放行"。
 */
import { describe, expect, it } from "vitest";
import { judgeWindowMotion } from "./motionGate";
import { IDLE_ENERGY } from "./dominantHand";
import {
  SEQ_IMU_N,
  SEQ_SENSOR_N,
  type SequenceSample,
} from "./datasetStore";
import type { BendRange } from "./bendRange";

/** 2000ms @ 50Hz = 推理窗口的真实帧数 */
const T = 100;
const BEND_OFFSET = 60;

const range = (span: number): BendRange => ({
  open: [40, 40, 40, 40, 40],
  fist: [40 + span, 40 + span, 40 + span, 40 + span, 40 + span],
});
const RANGES = { LH: range(120), RH: range(120) };

/** 确定性伪随机，免得测试变成偶发失败 */
function noise(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000 - 0.5;
  };
}

/**
 * 造一只手的一窗数据。
 * @param active true = 在做动作；false = 戴着不动（只有 ADC 噪声）
 */
function buildHand(active: boolean, seed: number) {
  const sensor = new Uint8Array(T * SEQ_SENSOR_N);
  const imu = new Float32Array(T * SEQ_IMU_N);
  const rnd = noise(seed);
  for (let t = 0; t < T; t++) {
    // 半个正弦 = 张开→握拳→张开一次
    const phase = Math.sin((Math.PI * t) / (T - 1));
    for (let j = 0; j < 5; j++) {
      const v = 40 + (active ? 0.6 * 120 * phase : 0) + rnd() * 3;
      sensor[t * SEQ_SENSOR_N + BEND_OFFSET + j] = Math.max(
        0,
        Math.min(255, Math.round(v))
      );
    }
    for (let j = 0; j < 60; j++) {
      const v = 20 + (active ? 40 * phase : 0) + rnd() * 3;
      sensor[t * SEQ_SENSOR_N + j] = Math.max(0, Math.min(255, Math.round(v)));
    }
    const half = active ? ((60 * phase) / 2 / 180) * Math.PI : 0;
    const o = t * SEQ_IMU_N;
    imu[o] = Math.cos(half);
    imu[o + 1] = Math.sin(half);
  }
  return { sensor, imu };
}

/** @param left/@param right true=动 false=戴着不动 null=没数据 */
function snap(left: boolean | null, right: boolean | null): SequenceSample {
  const l = left === null ? null : buildHand(left, 12345);
  const r = right === null ? null : buildHand(right, 98765);
  return {
    segments: [{ label: "_live", startFrame: 0, endFrame: T }],
    primaryLabel: "_live",
    frameCount: T,
    timestamps: new Float32Array(T),
    leftSensor: l?.sensor ?? null,
    rightSensor: r?.sensor ?? null,
    leftImu: l?.imu ?? null,
    rightImu: r?.imu ?? null,
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: 2000,
    sourceFps: 50,
    origin: "recorded",
    timestamp: 0,
  };
}

describe("judgeWindowMotion：拦住静止", () => {
  it("两只手都戴着不动 → 拦住（这就是「还没起手就蹦出词」的那个窗口）", () => {
    const v = judgeWindowMotion(snap(false, false), RANGES);
    expect(v.moving).toBe(false);
    expect(v.peak).toBeLessThan(IDLE_ENERGY);
  });

  it("两只手都没数据 → 拦住，peak=0（防御性：snapshot 本该先返回 null）", () => {
    const v = judgeWindowMotion(snap(null, null), RANGES);
    expect(v.moving).toBe(false);
    expect(v.peak).toBe(0);
    expect(v.left).toBeNull();
    expect(v.right).toBeNull();
  });
});

describe("judgeWindowMotion：放行动作", () => {
  it("右手在做动作、左手戴着不动 → 放行（单手词，绝不能拦）", () => {
    const v = judgeWindowMotion(snap(false, true), RANGES);
    expect(v.moving).toBe(true);
  });

  it("左手在做动作、右手戴着不动 → 放行（镜像之前就该判出来）", () => {
    const v = judgeWindowMotion(snap(true, false), RANGES);
    expect(v.moving).toBe(true);
  });

  it("两只手都在动 → 放行（双手词）", () => {
    expect(judgeWindowMotion(snap(true, true), RANGES).moving).toBe(true);
  });

  it("只戴一只手套且在动 → 放行", () => {
    expect(judgeWindowMotion(snap(null, true), RANGES).moving).toBe(true);
    expect(judgeWindowMotion(snap(true, null), RANGES).moving).toBe(true);
  });

  it("单手动作的余量足够大 —— 不是勉强擦过门限", () => {
    // 若哪天有人把 max 改成 avg，这条会连同上面的单手词一起挂掉
    const v = judgeWindowMotion(snap(false, true), RANGES);
    expect(v.peak).toBeGreaterThan(IDLE_ENERGY * 4);
  });
});

describe("judgeWindowMotion：peak 与透传", () => {
  it("peak 取两只手里较大的那个，不是平均", () => {
    const v = judgeWindowMotion(snap(false, true), RANGES);
    expect(v.peak).toBe(Math.max(v.left!.total, v.right!.total));
    expect(v.peak).toBe(v.right!.total);
    // 平均会明显更小 —— 这是两种实现的分水岭
    expect((v.left!.total + v.right!.total) / 2).toBeLessThan(v.peak);
  });

  it("只有一只手做过两点标定 → calibrated=false，但判定仍然成立", () => {
    const v = judgeWindowMotion(snap(false, true), { LH: range(120), RH: null });
    expect(v.calibrated).toBe(false);
    expect(v.moving).toBe(true);
  });

  it("不改传进来的快照（同一个对象紧接着要喂进模型）", () => {
    const s = snap(false, true);
    const before = {
      left: [...s.leftSensor!],
      right: [...s.rightSensor!],
      imu: [...s.rightImu!],
    };
    judgeWindowMotion(s, RANGES);
    expect([...s.leftSensor!]).toEqual(before.left);
    expect([...s.rightSensor!]).toEqual(before.right);
    expect([...s.rightImu!]).toEqual(before.imu);
  });
});
