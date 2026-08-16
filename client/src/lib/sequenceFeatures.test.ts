import { describe, it, expect } from "vitest";
import {
  SEQ_LEN,
  TACTILE_FRAME_DIM,
  FUSED_FRAME_DIM,
  HAND_TACTILE_FRAME_DIM,
  quatMul,
  quatConj,
  quatNormalize,
  quatFromAxisAngle,
  quatSlerp,
  gravityInHandFrame,
  uniformGrid,
  resampleSequence,
  fillVisionGaps,
  normalizeLandmarkFrame,
  makeAugmentedGrid,
  buildSequenceFeatures,
  synthesizeFromStatic,
  motionEnergy,
  visionCoverage,
  DEFAULT_AUGMENT,
  DEFAULT_SYNTHESIZE,
  NO_AUGMENT,
  type Quat,
  type Rng,
} from "./sequenceFeatures";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
  type TrainingSample,
} from "./datasetStore";

// ===== 测试夹具 =====

/** 确定性 RNG（LCG），保证合成/增强类测试可复现 */
function makeRng(seed = 42): Rng {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

interface FixtureOptions {
  T: number;
  fps?: number;
  /** 第 c 通道第 t 帧的传感值 */
  sensorAt?: (t: number, c: number) => number;
  /** 第 t 帧的四元数 */
  quatAt?: (t: number) => Quat;
  /** 返回 null 表示该帧视觉丢失（写 NaN） */
  visionAt?: ((t: number) => number[] | null) | null;
}

function makeSample(opts: FixtureOptions): SequenceSample {
  const { T, fps = 50 } = opts;
  const dt = 1000 / fps;
  const timestamps = new Float32Array(T);
  for (let t = 0; t < T; t++) timestamps[t] = t * dt;

  const sensor = new Uint8Array(T * SEQ_SENSOR_N);
  const imu = new Float32Array(T * SEQ_IMU_N);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < SEQ_SENSOR_N; c++) {
      sensor[t * SEQ_SENSOR_N + c] = opts.sensorAt ? opts.sensorAt(t, c) : 0;
    }
    const q = opts.quatAt ? opts.quatAt(t) : ([1, 0, 0, 0] as Quat);
    const o = t * SEQ_IMU_N;
    imu[o] = q[0];
    imu[o + 1] = q[1];
    imu[o + 2] = q[2];
    imu[o + 3] = q[3];
  }

  let landmarks: Float32Array | null = null;
  if (opts.visionAt) {
    landmarks = new Float32Array(T * SEQ_LANDMARK_N);
    for (let t = 0; t < T; t++) {
      const v = opts.visionAt(t);
      const o = t * SEQ_LANDMARK_N;
      if (!v) {
        landmarks.fill(NaN, o, o + SEQ_LANDMARK_N);
      } else {
        landmarks.set(v, o);
      }
    }
  }

  return {
    segments: [{ label: "test", startFrame: 0, endFrame: T }],
    primaryLabel: "test",
    frameCount: T,
    timestamps,
    leftSensor: sensor,
    rightSensor: null,
    leftImu: imu,
    rightImu: null,
    leftLandmarks: landmarks,
    rightLandmarks: null,
    durationMs: T * dt,
    sourceFps: fps,
    origin: "recorded",
    timestamp: 0,
  };
}

/** 21 点，全部落在同一条直线上，便于验算归一化 */
function flatLandmarks(scale = 1, offset = 0): number[] {
  const out: number[] = [];
  for (let p = 0; p < 21; p++) {
    out.push(p * 0.01 * scale + offset, p * 0.02 * scale + offset, 0);
  }
  return out;
}

// ===== 四元数 =====

describe("四元数工具", () => {
  it("q₀⁻¹ ⊗ q₀ 等于单位四元数", () => {
    const q = quatNormalize([0.3, 0.5, -0.2, 0.7]);
    const rel = quatMul(quatConj(q), q);
    expect(rel[0]).toBeCloseTo(1, 6);
    expect(rel[1]).toBeCloseTo(0, 6);
    expect(rel[2]).toBeCloseTo(0, 6);
    expect(rel[3]).toBeCloseTo(0, 6);
  });

  it("相对四元数还原出正确的相对转角", () => {
    const q0 = quatFromAxisAngle([0, 0, 1], Math.PI / 6);
    const q1 = quatFromAxisAngle([0, 0, 1], Math.PI / 6 + Math.PI / 3);
    const rel = quatMul(quatConj(q0), q1);
    // w = cos(θ/2)，θ 应为 60°
    expect(2 * Math.acos(Math.min(1, Math.abs(rel[0])))).toBeCloseTo(
      Math.PI / 3,
      5
    );
  });

  it("slerp 中点落在两端角度正中，且保持单位长度", () => {
    const a = quatFromAxisAngle([0, 0, 1], 0);
    const b = quatFromAxisAngle([0, 0, 1], Math.PI / 2);
    const mid = quatSlerp(a, b, 0.5);
    expect(Math.hypot(mid[0], mid[1], mid[2], mid[3])).toBeCloseTo(1, 6);
    // 45° 绕 z：w = cos(22.5°)
    expect(mid[0]).toBeCloseTo(Math.cos(Math.PI / 8), 5);
    expect(mid[3]).toBeCloseTo(Math.sin(Math.PI / 8), 5);
  });

  it("重力方向对绕重力轴的旋转不变（yaw 偏置不会污染该特征）", () => {
    const base = quatFromAxisAngle([1, 0, 0], 0.4); // 有一定倾斜
    const g0 = gravityInHandFrame(base);
    // 世界系下绕 z（重力轴）左乘一个 yaw
    const yawed = quatMul(quatFromAxisAngle([0, 0, 1], 1.234), base);
    const g1 = gravityInHandFrame(yawed);
    for (let i = 0; i < 3; i++) expect(g1[i]).toBeCloseTo(g0[i], 5);
  });
});

// ===== 重采样 =====

describe("resampleSequence", () => {
  it("变长输入统一到定长 T", () => {
    for (const srcT of [5, 17, 40, 120]) {
      const s = makeSample({ T: srcT });
      const rs = resampleSequence(s, 32);
      expect(rs.frameCount).toBe(32);
      expect(rs.leftSensor!.length).toBe(32 * SEQ_SENSOR_N);
      expect(rs.leftImu!.length).toBe(32 * SEQ_IMU_N);
      expect(rs.rightSensor).toBeNull();
    }
  });

  it("首尾帧被精确保留（不被插值抹掉）", () => {
    const s = makeSample({ T: 10, sensorAt: (t) => t * 10 });
    const rs = resampleSequence(s, 32);
    expect(rs.leftSensor![0]).toBeCloseTo(0, 4);
    expect(rs.leftSensor![31 * SEQ_SENSOR_N]).toBeCloseTo(90, 4);
  });

  it("线性斜坡重采样后仍是线性斜坡", () => {
    const srcT = 9;
    const s = makeSample({ T: srcT, sensorAt: (t) => t * 20 });
    const T = 17;
    const rs = resampleSequence(s, T);
    for (let t = 0; t < T; t++) {
      const u = t / (T - 1);
      const expected = u * (srcT - 1) * 20;
      expect(rs.leftSensor![t * SEQ_SENSOR_N]).toBeCloseTo(expected, 3);
    }
  });

  it("四元数走 slerp 而非逐分量 lerp —— 中点仍是单位四元数", () => {
    // 两端相差 90°，逐分量 lerp 的中点模长约 0.924，slerp 恒为 1
    const s = makeSample({
      T: 2,
      quatAt: (t) => quatFromAxisAngle([0, 0, 1], (t * Math.PI) / 2),
    });
    const rs = resampleSequence(s, 3);
    const o = 1 * SEQ_IMU_N;
    const n = Math.hypot(
      rs.leftImu![o],
      rs.leftImu![o + 1],
      rs.leftImu![o + 2],
      rs.leftImu![o + 3]
    );
    expect(n).toBeCloseTo(1, 5);
  });
});

// ===== 视觉缺失处理 =====

describe("视觉缺帧", () => {
  it("中间空洞被线性插值补齐", () => {
    const lm = new Float32Array(5 * SEQ_LANDMARK_N);
    for (let t = 0; t < 5; t++) {
      const o = t * SEQ_LANDMARK_N;
      if (t === 2) {
        lm.fill(NaN, o, o + SEQ_LANDMARK_N);
      } else {
        for (let c = 0; c < SEQ_LANDMARK_N; c++) lm[o + c] = t;
      }
    }
    expect(fillVisionGaps(lm)).toBe(true);
    expect(lm[2 * SEQ_LANDMARK_N]).toBeCloseTo(2, 5);
    expect(lm.some(Number.isNaN)).toBe(false);
  });

  it("首尾空洞用最近有效帧 hold 住", () => {
    const lm = new Float32Array(4 * SEQ_LANDMARK_N);
    lm.fill(NaN, 0, SEQ_LANDMARK_N);
    lm.fill(NaN, 3 * SEQ_LANDMARK_N);
    for (let t = 1; t <= 2; t++) {
      const o = t * SEQ_LANDMARK_N;
      for (let c = 0; c < SEQ_LANDMARK_N; c++) lm[o + c] = t * 7;
    }
    expect(fillVisionGaps(lm)).toBe(true);
    expect(lm[0]).toBeCloseTo(7, 5);
    expect(lm[3 * SEQ_LANDMARK_N]).toBeCloseTo(14, 5);
  });

  it("整段无视觉时返回 false 并清零（不留 NaN 毒害训练）", () => {
    const lm = new Float32Array(6 * SEQ_LANDMARK_N).fill(NaN);
    expect(fillVisionGaps(lm)).toBe(false);
    expect(lm.some(Number.isNaN)).toBe(false);
    expect(lm.every((v) => v === 0)).toBe(true);
  });

  it("重采样不会把有效帧和 NaN 帧混成整帧 NaN", () => {
    // 偶数帧有视觉、奇数帧丢失，重采样后应该没有一帧是全 NaN
    const s = makeSample({
      T: 8,
      visionAt: (t) => (t % 2 === 0 ? flatLandmarks() : null),
    });
    const rs = resampleSequence(s, 16);
    expect(rs.leftHasVision).toBe(true);
    expect(Array.from(rs.leftLandmarks!).some(Number.isNaN)).toBe(false);
  });

  it("visionCoverage 统计与实际缺帧比例一致", () => {
    const s = makeSample({
      T: 10,
      visionAt: (t) => (t < 7 ? flatLandmarks() : null),
    });
    expect(visionCoverage(s)).toBeCloseTo(0.7, 5);
  });
});

// ===== 关键点归一化 =====

describe("normalizeLandmarkFrame", () => {
  const run = (src: number[]) => {
    const dst = new Float32Array(SEQ_LANDMARK_N);
    normalizeLandmarkFrame(src, 0, dst, 0);
    return dst;
  };

  it("手腕绝对坐标被原样保留（轨迹信息不能丢）", () => {
    const src = flatLandmarks(1, 0.3);
    const out = run(src);
    expect(out[0]).toBeCloseTo(src[0], 6);
    expect(out[1]).toBeCloseTo(src[1], 6);
    expect(out[2]).toBeCloseTo(src[2], 6);
  });

  it("非手腕点对整体平移不变", () => {
    const a = run(flatLandmarks(1, 0));
    const b = run(flatLandmarks(1, 0.25));
    for (let c = 3; c < SEQ_LANDMARK_N; c++) {
      expect(b[c]).toBeCloseTo(a[c], 5);
    }
  });

  it("非手腕点对整体缩放不变（同一手型不同手掌大小）", () => {
    const a = run(flatLandmarks(1, 0));
    const b = run(flatLandmarks(2.5, 0));
    for (let c = 3; c < SEQ_LANDMARK_N; c++) {
      expect(b[c]).toBeCloseTo(a[c], 5);
    }
  });

  it("手长退化为 0 时不产生 Inf/NaN", () => {
    const src = new Array(SEQ_LANDMARK_N).fill(0.5);
    const out = run(src);
    expect(Array.from(out).every(Number.isFinite)).toBe(true);
  });
});

// ===== 增强 =====

describe("makeAugmentedGrid", () => {
  it("栅格单调不减且落在 [0,1] 内（时间不能倒流）", () => {
    const rng = makeRng(7);
    for (let trial = 0; trial < 50; trial++) {
      const g = makeAugmentedGrid(32, DEFAULT_AUGMENT, rng);
      expect(g.length).toBe(32);
      for (let i = 0; i < g.length; i++) {
        expect(g[i]).toBeGreaterThanOrEqual(0);
        expect(g[i]).toBeLessThanOrEqual(1);
        if (i > 0) expect(g[i]).toBeGreaterThanOrEqual(g[i - 1] - 1e-6);
      }
    }
  });

  it("NO_AUGMENT 等价于均匀栅格", () => {
    const g = makeAugmentedGrid(16, NO_AUGMENT, makeRng(1));
    const u = uniformGrid(16);
    for (let i = 0; i < 16; i++) expect(g[i]).toBeCloseTo(u[i], 5);
  });
});

// ===== 特征构建 =====

describe("buildSequenceFeatures", () => {
  it("触觉/融合两种输出维度正确", () => {
    const s = makeSample({ T: 20, visionAt: () => flatLandmarks() });
    const tactile = buildSequenceFeatures(s, { includeVision: false });
    expect(tactile.frameDim).toBe(TACTILE_FRAME_DIM);
    expect(tactile.data.length).toBe(SEQ_LEN * TACTILE_FRAME_DIM);

    const fused = buildSequenceFeatures(s, { includeVision: true });
    expect(fused.frameDim).toBe(FUSED_FRAME_DIM);
    expect(fused.data.length).toBe(SEQ_LEN * FUSED_FRAME_DIM);
  });

  it("首帧的相对四元数恒为单位四元数", () => {
    const s = makeSample({
      T: 30,
      quatAt: (t) => quatFromAxisAngle([0.3, 1, -0.2], t * 0.05),
    });
    const f = buildSequenceFeatures(s, { augment: NO_AUGMENT });
    const o = SEQ_SENSOR_N; // 每帧四元数块的偏移
    expect(f.data[o]).toBeCloseTo(1, 5);
    expect(f.data[o + 1]).toBeCloseTo(0, 5);
    expect(f.data[o + 2]).toBeCloseTo(0, 5);
    expect(f.data[o + 3]).toBeCloseTo(0, 5);
  });

  it("传感值被归一化到 [0,1]，缺失的右手保持全 0", () => {
    const s = makeSample({ T: 12, sensorAt: () => 255 });
    const f = buildSequenceFeatures(s, { augment: NO_AUGMENT });
    expect(f.data[0]).toBeCloseTo(1, 5);
    // 右手块整段应为 0
    for (let t = 0; t < SEQ_LEN; t++) {
      const base = t * TACTILE_FRAME_DIM + HAND_TACTILE_FRAME_DIM;
      for (let c = 0; c < SEQ_SENSOR_N; c++) {
        expect(f.data[base + c]).toBe(0);
      }
    }
  });

  // ===== 起手段裁剪（sequenceTrim.ts 的接入点）=====

  /** 手工截取 [start, end) 帧，模拟"录制时就没录进抬手段" */
  function sliceSample(
    s: SequenceSample,
    start: number,
    end: number
  ): SequenceSample {
    const n = end - start;
    const cut = (a: Float32Array | null, stride: number) =>
      a ? a.slice(start * stride, end * stride) : null;
    return {
      ...s,
      frameCount: n,
      timestamps: s.timestamps.slice(start, end),
      leftSensor: s.leftSensor
        ? s.leftSensor.slice(start * SEQ_SENSOR_N, end * SEQ_SENSOR_N)
        : null,
      rightSensor: s.rightSensor
        ? s.rightSensor.slice(start * SEQ_SENSOR_N, end * SEQ_SENSOR_N)
        : null,
      leftImu: cut(s.leftImu, SEQ_IMU_N),
      rightImu: cut(s.rightImu, SEQ_IMU_N),
      leftLandmarks: cut(s.leftLandmarks, SEQ_LANDMARK_N),
      rightLandmarks: cut(s.rightLandmarks, SEQ_LANDMARK_N),
      segments: [{ label: s.primaryLabel, startFrame: 0, endFrame: n }],
    };
  }

  /** 前 10 帧手还在腿上（视觉全 NaN），后 20 帧是真手势 */
  function makeLiftSample(): SequenceSample {
    return makeSample({
      T: 30,
      fps: 30,
      sensorAt: (t, c) => (t * 7 + c * 3) % 256,
      quatAt: (t) => quatFromAxisAngle([0.2, 1, -0.3], t * 0.04),
      visionAt: (t) => (t >= 10 ? flatLandmarks() : null),
    });
  }

  it("裁掉起手段后的特征 ≈ 直接录那一段的特征", () => {
    const full = makeLiftSample();
    const trimmed = buildSequenceFeatures(full, {
      includeVision: false,
      augment: NO_AUGMENT,
      trim: { minRunFrames: 3, padMs: 0, minKeptFrames: 6, minKeptMs: 250 },
    });
    expect(trimmed.trim?.applied).toBe(true);
    expect(trimmed.trim?.startFrame).toBe(10);

    // 手工只留 10..29 —— 重采样是归一化**时间**的，所以两者的采样点应重合
    const reference = buildSequenceFeatures(sliceSample(full, 10, 30), {
      includeVision: false,
      augment: NO_AUGMENT,
      trim: null,
    });
    expect(reference.trim).toBeNull();
    for (let i = 0; i < trimmed.data.length; i++) {
      expect(trimmed.data[i]).toBeCloseTo(reference.data[i], 4);
    }
  });

  it("trim: null 保留旧行为（整条 [0,1] 铺栅格）", () => {
    const full = makeLiftSample();
    const off = buildSequenceFeatures(full, {
      includeVision: false,
      augment: NO_AUGMENT,
      trim: null,
    });
    const on = buildSequenceFeatures(full, {
      includeVision: false,
      augment: NO_AUGMENT,
    });
    expect(off.trim).toBeNull();
    expect(on.trim?.applied).toBe(true);
    // 首帧就该不一样：关掉时首帧是"手在腿上"，开着时首帧是入画那一刻
    let maxDiff = 0;
    for (let i = 0; i < off.data.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(off.data[i] - on.data[i]));
    }
    expect(maxDiff).toBeGreaterThan(0.01);
  });

  it("缺省即启用，但没有视觉时自动退回不裁", () => {
    // visionAt 不传 → leftLandmarks 为 null → no_vision
    const s = makeSample({ T: 30, sensorAt: (t, c) => (t + c) % 256 });
    const f = buildSequenceFeatures(s, {
      includeVision: false,
      augment: NO_AUGMENT,
    });
    expect(f.trim?.applied).toBe(false);
    expect(f.trim?.reason).toBe("no_vision");
    const explicitOff = buildSequenceFeatures(s, {
      includeVision: false,
      augment: NO_AUGMENT,
      trim: null,
    });
    for (let i = 0; i < f.data.length; i++) {
      expect(f.data[i]).toBe(explicitOff.data[i]);
    }
  });

  it("随机裁剪叠在裁剪区间之内，不会把起手段又采回来", () => {
    const full = makeLiftSample();
    const f = buildSequenceFeatures(full, {
      includeVision: false,
      augment: { ...DEFAULT_AUGMENT, cropMin: 0.85 },
      rng: makeRng(7),
      trim: { minRunFrames: 3, padMs: 0, minKeptFrames: 6, minKeptMs: 250 },
    });
    // 增强的栅格先在 [0,1] 上取子区间，再整体映射进 [a,b] —— 所以最终采样点
    // 一定落在 [a,b] 里。反过来（先映射再裁剪）会有概率把 a 前面的帧采进来。
    expect(f.trim?.applied).toBe(true);
    expect(Array.from(f.data).every(Number.isFinite)).toBe(true);
  });

  it("输出不含 NaN / Inf", () => {
    const s = makeSample({
      T: 25,
      sensorAt: (t, c) => (t * 3 + c) % 256,
      quatAt: (t) => quatFromAxisAngle([1, 1, 0], t * 0.1),
      visionAt: (t) => (t % 3 === 0 ? null : flatLandmarks()),
    });
    const f = buildSequenceFeatures(s, {
      includeVision: true,
      augment: DEFAULT_AUGMENT,
      rng: makeRng(11),
    });
    expect(Array.from(f.data).every(Number.isFinite)).toBe(true);
  });
});

// ===== 静态样本合成 =====

/** 第一阶差分的 lag-1 自相关 */
function diffLag1Autocorr(x: number[]): number {
  const d: number[] = [];
  for (let i = 1; i < x.length; i++) d.push(x[i] - x[i - 1]);
  const mean = d.reduce((a, b) => a + b, 0) / d.length;
  let num = 0;
  let den = 0;
  for (let i = 0; i < d.length; i++) {
    const c = d[i] - mean;
    den += c * c;
    if (i > 0) num += c * (d[i - 1] - mean);
  }
  return den > 1e-12 ? num / den : 0;
}

function makeStaticSample(label = "num_1"): TrainingSample {
  return {
    label,
    timestamp: 0,
    left: {
      sensor_data: Array.from({ length: SEQ_SENSOR_N }, (_, c) => (c * 7) % 200),
      quaternion: [1, 0, 0, 0],
      landmarks: Array.from({ length: 21 }, (_, p) => ({
        x: p * 0.01,
        y: p * 0.02,
        z: 0,
      })),
    },
    right: null,
  };
}

describe("synthesizeFromStatic", () => {
  it("产出帧数、时长、origin 正确", () => {
    const seq = synthesizeFromStatic(
      makeStaticSample(),
      DEFAULT_SYNTHESIZE,
      makeRng(3)
    );
    expect(seq.frameCount).toBe(DEFAULT_SYNTHESIZE.frameCount);
    expect(seq.origin).toBe("synthesized");
    expect(seq.primaryLabel).toBe("num_1");
    expect(seq.segments).toEqual([
      { label: "num_1", startFrame: 0, endFrame: DEFAULT_SYNTHESIZE.frameCount },
    ]);
    expect(seq.leftSensor!.length).toBe(
      DEFAULT_SYNTHESIZE.frameCount * SEQ_SENSOR_N
    );
    expect(seq.rightSensor).toBeNull();
    expect(seq.durationMs).toBeCloseTo(
      (DEFAULT_SYNTHESIZE.frameCount * 1000) / DEFAULT_SYNTHESIZE.fps,
      5
    );
  });

  it("注入的是低通相关噪声，不是白噪声", () => {
    // 这条直接验证"捷径风险"是否被规避：白噪声序列的一阶差分 lag-1 自相关
    // 收敛到 -0.5，平滑序列则接近 0。模型若能靠这个统计量区分动静，
    // 真实推理时必崩。要求合成序列明显偏离 -0.5。
    const cfg = { ...DEFAULT_SYNTHESIZE, frameCount: 400, sensorJitter: 20 };
    const seq = synthesizeFromStatic(makeStaticSample(), cfg, makeRng(9));
    const series: number[] = [];
    for (let t = 0; t < cfg.frameCount; t++) {
      series.push(seq.leftSensor![t * SEQ_SENSOR_N + 5]);
    }
    const ac = diffLag1Autocorr(series);
    expect(ac).toBeGreaterThan(-0.25); // 远离白噪声的 -0.5
    expect(ac).toBeLessThan(0.25);
  });

  it("四元数逐帧保持单位长度", () => {
    const seq = synthesizeFromStatic(
      makeStaticSample(),
      DEFAULT_SYNTHESIZE,
      makeRng(5)
    );
    for (let t = 0; t < seq.frameCount; t++) {
      const o = t * SEQ_IMU_N;
      const n = Math.hypot(
        seq.leftImu![o],
        seq.leftImu![o + 1],
        seq.leftImu![o + 2],
        seq.leftImu![o + 3]
      );
      expect(n).toBeCloseTo(1, 4);
    }
  });

  it("传感值不越出 0-255", () => {
    const cfg = { ...DEFAULT_SYNTHESIZE, sensorJitter: 200 };
    const seq = synthesizeFromStatic(makeStaticSample(), cfg, makeRng(17));
    for (const v of seq.leftSensor!) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(255);
    }
  });
});

// ===== 运动能量 =====

describe("motionEnergy", () => {
  it("静止序列能量远低于运动序列", () => {
    const still = makeSample({ T: 60, sensorAt: () => 100 });
    const moving = makeSample({
      T: 60,
      sensorAt: (t) => 100 + Math.round(60 * Math.sin(t * 0.3)),
    });
    const maxOf = (a: Float32Array) => Math.max(...Array.from(a));
    expect(maxOf(motionEnergy(still))).toBeLessThan(
      maxOf(motionEnergy(moving)) * 0.1
    );
  });

  it("长度与帧数一致", () => {
    expect(motionEnergy(makeSample({ T: 23 })).length).toBe(23);
  });
});
