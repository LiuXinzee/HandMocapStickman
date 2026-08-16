import { describe, expect, it } from "vitest";
import {
  analyzeImuHealth,
  analyzeSequenceImu,
  axisAngleDeg,
  estimateGravityMagnitude,
  isImuSuspect,
  MIN_USABLE_FRAMES,
  quatAngleDeg,
  worstReport,
  worstVerdict,
  type ImuHealthReport,
  type ImuHealthSample,
} from "./imuHealth";
import {
  gravityInHandFrame,
  quatFromAxisAngle,
  quatMul,
  type Quat,
} from "./sequenceFeatures";

const IDENTITY: Quat = [1, 0, 0, 0];
const N = MIN_USABLE_FRAMES * 3;

/** 造一段"四元数与加速度完全自洽"的采样：acc 直接取 q 推出的重力方向 */
function consistent(q: Quat, count = N, gMag = 9.8): ImuHealthSample[] {
  const g = gravityInHandFrame(q);
  const acc: [number, number, number] = [g[0] * gMag, g[1] * gMag, g[2] * gMag];
  return Array.from({ length: count }, (_, i) => ({ q, acc, t: i * 20 }));
}

describe("axisAngleDeg", () => {
  it("同向为 0°，正交为 90°", () => {
    expect(axisAngleDeg([0, 0, 1], [0, 0, 5])).toBeCloseTo(0, 6);
    expect(axisAngleDeg([1, 0, 0], [0, 1, 0])).toBeCloseTo(90, 6);
  });

  it("反向也算 0° —— 加速度计比力与重力向量的符号约定被吃掉", () => {
    expect(axisAngleDeg([0, 0, 1], [0, 0, -3])).toBeCloseTo(0, 6);
  });

  it("零向量返回 NaN 而不是 0", () => {
    expect(Number.isNaN(axisAngleDeg([0, 0, 0], [0, 0, 1]))).toBe(true);
  });
});

describe("estimateGravityMagnitude", () => {
  const sorted = (xs: number[]) => xs.slice().sort((a, b) => a - b);

  it("识别 g 单位并吸附到 1", () => {
    expect(estimateGravityMagnitude(sorted([0.98, 1.0, 1.02, 1.3]))).toBe(1);
  });

  it("识别 m/s² 单位并吸附到 9.80665", () => {
    expect(estimateGravityMagnitude(sorted([9.7, 9.8, 9.9, 12]))).toBeCloseTo(9.80665, 5);
  });

  it("走 25 分位而非中位数：一半是大幅动作帧时仍认出 1g", () => {
    const mags = sorted([9.8, 9.8, 9.8, 9.8, 49, 49, 49, 49]);
    expect(estimateGravityMagnitude(mags)).toBeCloseTo(9.80665, 5);
  });

  it("量程完全未知时回落到 25 分位本身", () => {
    expect(estimateGravityMagnitude(sorted([300, 320, 340, 360]))).toBeGreaterThan(300);
  });
});

describe("quatAngleDeg", () => {
  it("同一旋转为 0°，且 q 与 -q 等价", () => {
    expect(quatAngleDeg(IDENTITY, IDENTITY)).toBeCloseTo(0, 6);
    expect(quatAngleDeg(IDENTITY, [-1, 0, 0, 0])).toBeCloseTo(0, 4);
  });

  it("绕轴 30° 得到 30°", () => {
    const q = quatFromAxisAngle([0, 1, 0], (30 * Math.PI) / 180);
    expect(quatAngleDeg(IDENTITY, q)).toBeCloseTo(30, 4);
  });
});

describe("analyzeImuHealth 倾角一致性", () => {
  it("完全自洽的采样 → 夹角≈0，verdict ok", () => {
    const r = analyzeImuHealth(consistent(IDENTITY));
    expect(r.verdict).toBe("ok");
    expect(r.tiltInconsistencyDeg).toBeCloseTo(0, 4);
    expect(r.p95TiltDeg).toBeCloseTo(0, 4);
    expect(r.usableFrames).toBe(N);
    expect(isImuSuspect(r)).toBe(false);
  });

  it("加速度单位换成 g（≈1）也一样 —— 判定与单位无关", () => {
    const r = analyzeImuHealth(consistent(IDENTITY, N, 1));
    expect(r.verdict).toBe("ok");
    expect(r.tiltInconsistencyDeg).toBeCloseTo(0, 4);
  });

  it("给四元数叠加绕水平轴的已知偏置 → 夹角≈该偏置角", () => {
    // 真实姿态是单位四元数，但板载解算多转了 20°（绕手系 x 轴，与重力不共线）
    const bias = quatFromAxisAngle([1, 0, 0], (20 * Math.PI) / 180);
    const truth = consistent(IDENTITY);
    const drifted = truth.map((s) => ({ ...s, q: quatMul(bias, s.q) }));
    const r = analyzeImuHealth(drifted);
    expect(r.tiltInconsistencyDeg).toBeCloseTo(20, 3);
    expect(r.verdict).toBe("bad");
    expect(isImuSuspect(r)).toBe(true);
  });

  it("8° 偏置落在 warn 区间", () => {
    const bias = quatFromAxisAngle([1, 0, 0], (8 * Math.PI) / 180);
    const drifted = consistent(IDENTITY).map((s) => ({ ...s, q: quatMul(bias, s.q) }));
    const r = analyzeImuHealth(drifted);
    expect(r.tiltInconsistencyDeg).toBeCloseTo(8, 3);
    expect(r.verdict).toBe("warn");
  });

  it("绕重力轴（yaw）旋转四元数 → 夹角仍≈0，不报警", () => {
    // 这条锁住设计：六轴 IMU 的绝对 yaw 没有零点，yaw 漂移测不出来也对识别无害
    // （特征层只用相对四元数 + 重力手系投影，两者对世界 yaw 都不变）。
    // 谁把 axisAngleDeg 改成能感知 yaw，这条测试就会红。
    const truth = consistent(IDENTITY);
    const spun = truth.map((s, i) => ({
      ...s,
      q: quatMul(quatFromAxisAngle([0, 0, 1], (i * 7 * Math.PI) / 180), s.q),
    }));
    const r = analyzeImuHealth(spun);
    expect(r.tiltInconsistencyDeg).toBeCloseTo(0, 3);
    expect(r.verdict).toBe("ok");
  });
});

describe("analyzeImuHealth 边界", () => {
  it("空输入 → unknown，不抛", () => {
    const r = analyzeImuHealth([]);
    expect(r.verdict).toBe("unknown");
    expect(r.usableFrames).toBe(0);
  });

  it("加速度全为 null（旧 272B 手套）→ unknown，走静置旋转量分支", () => {
    const samples: ImuHealthSample[] = Array.from({ length: N }, (_, i) => ({
      q: IDENTITY,
      acc: null,
      t: i * 20,
    }));
    const r = analyzeImuHealth(samples);
    expect(r.verdict).toBe("unknown");
    expect(r.stillRotationDegPerMin).toBeCloseTo(0, 4);
    expect(r.reason).toContain("不上报加速度");
  });

  it("旧手套静置却在转 → warn", () => {
    const samples: ImuHealthSample[] = Array.from({ length: 100 }, (_, i) => ({
      // 每 20ms 转 1°，即 50°/s，远超静置门限
      q: quatFromAxisAngle([0, 1, 0], (i * 1 * Math.PI) / 180),
      acc: null,
      t: i * 20,
    }));
    const r = analyzeImuHealth(samples);
    expect(r.verdict).toBe("warn");
    expect(r.stillRotationDegPerMin!).toBeGreaterThan(1000);
  });

  it("|acc| 远离重力量级的帧被排除在统计外", () => {
    // 一半是自洽的静止帧，一半是幅度 5 倍的甩手帧（方向故意错 90°）。
    // 甩手帧占到 50% 时，若拿全体中位数当 1g 参考会把两边一起筛光——
    // 这就是 estimateGravityMagnitude 要走低分位 + 单位吸附的原因。
    const still = consistent(IDENTITY, N);
    const swung: ImuHealthSample[] = Array.from({ length: N }, (_, i) => ({
      q: IDENTITY,
      acc: [49, 0, 0] as [number, number, number], // 与重力方向正交且量级 5 倍
      t: (N + i) * 20,
    }));
    const r = analyzeImuHealth([...still, ...swung]);
    expect(r.usableFrames).toBe(N); // 甩手帧全被剔掉
    expect(r.tiltInconsistencyDeg).toBeCloseTo(0, 4);
    expect(r.verdict).toBe("ok");
  });

  it("可用帧不足 → unknown 而不是硬下结论", () => {
    const r = analyzeImuHealth(consistent(IDENTITY, MIN_USABLE_FRAMES - 1));
    expect(r.verdict).toBe("unknown");
    expect(r.reason).toContain("可用帧只有");
  });
});

describe("analyzeSequenceImu", () => {
  const stride = 10;

  function pack(samples: ImuHealthSample[]): Float32Array {
    const out = new Float32Array(samples.length * stride);
    samples.forEach((s, i) => {
      const o = i * stride;
      out[o] = s.q[0];
      out[o + 1] = s.q[1];
      out[o + 2] = s.q[2];
      out[o + 3] = s.q[3];
      out[o + 4] = s.acc ? s.acc[0] : 0;
      out[o + 5] = s.acc ? s.acc[1] : 0;
      out[o + 6] = s.acc ? s.acc[2] : 0;
      // [7..9] = 姿态角，本检查不用
    });
    return out;
  }

  it("stride-10 切片下标正确：自洽序列判为 ok", () => {
    const samples = consistent(quatFromAxisAngle([1, 0, 0], 0.4));
    const r = analyzeSequenceImu(pack(samples), samples.length);
    expect(r.verdict).toBe("ok");
    expect(r.usableFrames).toBe(samples.length);
    expect(r.tiltInconsistencyDeg).toBeCloseTo(0, 3);
  });

  it("能识别出打包序列里的倾角漂移", () => {
    const bias = quatFromAxisAngle([1, 0, 0], (25 * Math.PI) / 180);
    const drifted = consistent(IDENTITY).map((s) => ({ ...s, q: quatMul(bias, s.q) }));
    const r = analyzeSequenceImu(pack(drifted), drifted.length);
    expect(r.verdict).toBe("bad");
    expect(r.tiltInconsistencyDeg).toBeCloseTo(25, 2);
  });

  it("null / 长度不匹配 / 全零 四元数 都返回 unknown 且不抛", () => {
    expect(analyzeSequenceImu(null, 10).verdict).toBe("unknown");
    expect(analyzeSequenceImu(new Float32Array(30), 10).verdict).toBe("unknown");
    const short = pack(consistent(IDENTITY, 5));
    expect(analyzeSequenceImu(short, 50).verdict).toBe("unknown");
  });

  it("加速度三轴全零视为『无加速度』而不是『零重力』", () => {
    const samples: ImuHealthSample[] = Array.from({ length: N }, (_, i) => ({
      q: IDENTITY,
      acc: null,
      t: i * 20,
    }));
    const r = analyzeSequenceImu(pack(samples), samples.length);
    expect(r.verdict).toBe("unknown");
    expect(r.reason).toContain("不上报加速度");
  });

  it("时间戳传入后能算出静置旋转量", () => {
    const samples = consistent(IDENTITY, N);
    const ts = new Float32Array(samples.map((_, i) => i * 20));
    const r = analyzeSequenceImu(pack(samples), samples.length, ts);
    expect(r.stillRotationDegPerMin).toBeCloseTo(0, 3);
  });
});

describe("worstVerdict", () => {
  it("取最差；ok < unknown < warn < bad", () => {
    const mk = (v: "ok" | "warn" | "bad" | "unknown") =>
      ({ verdict: v } as ReturnType<typeof analyzeImuHealth>);
    expect(worstVerdict(mk("ok"), mk("ok"))).toBe("ok");
    expect(worstVerdict(mk("ok"), mk("unknown"))).toBe("unknown");
    expect(worstVerdict(mk("ok"), mk("warn"))).toBe("warn");
    expect(worstVerdict(mk("bad"), mk("warn"))).toBe("bad");
    expect(worstVerdict(mk("ok"), null)).toBe("ok");
    expect(worstVerdict(null, undefined)).toBe("unknown");
  });
});

describe("worstReport", () => {
  const mk = (
    verdict: ImuHealthReport["verdict"],
    tiltInconsistencyDeg = 0,
    usableFrames = 30
  ): ImuHealthReport => ({
    usableFrames,
    tiltInconsistencyDeg,
    p95TiltDeg: tiltInconsistencyDeg,
    stillRotationDegPerMin: null,
    verdict,
    reason: `${verdict}@${tiltInconsistencyDeg}`,
  });

  it("挑出最差的那一份，并原样返回（不合成新报告）", () => {
    const bad = mk("bad", 20);
    // 四步向导的典型输入：三步正常、一步漂了
    expect(worstReport([mk("ok", 1), mk("ok", 2), bad, mk("warn", 7)])).toBe(bad);
    // 同级时比倾角
    const worseOk = mk("ok", 4.5);
    expect(worstReport([mk("ok", 1), worseOk])).toBe(worseOk);
    // unknown 比 ok 差；null 被跳过
    expect(worstReport([mk("ok", 4), null, mk("unknown")])?.verdict).toBe(
      "unknown"
    );
  });

  it("空数组 / 全 null 给 null", () => {
    expect(worstReport([])).toBeNull();
    expect(worstReport([null, undefined])).toBeNull();
  });
});
