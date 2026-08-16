import { describe, it, expect } from "vitest";
import {
  applyOrientationCalib,
  assembleOrientationCalib,
  averageQuaternions,
  axisMapFailReason,
  axisSeparationDeg,
  buildAxisMap,
  loadOrientationCalib,
  multiplyQuat,
  relativeAxisAngle,
  type Quat,
  type Vec3,
} from "./orientationCalib";

const DEG = Math.PI / 180;

/** 绕任意轴转 deg 度的四元数 [w,x,y,z] */
function quatAbout(axis: Vec3, deg: number): Quat {
  const half = (deg * DEG) / 2;
  const m = Math.hypot(...axis) || 1;
  const s = Math.sin(half) / m;
  return [Math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s];
}

const IDENTITY: Quat = [1, 0, 0, 0];
const X: Vec3 = [1, 0, 0];
const Y: Vec3 = [0, 1, 0];
const Z: Vec3 = [0, 0, 1];

describe("averageQuaternions", () => {
  it("符号相反的同一旋转不会互相抵消", () => {
    const q = quatAbout(Z, 40);
    const flipped = q.map((v) => -v) as Quat;
    const avg = averageQuaternions([q, flipped, q]);
    expect(avg).not.toBeNull();
    // 与 q 表示同一旋转 → |点积| ≈ 1
    const dot = Math.abs(
      avg![0] * q[0] + avg![1] * q[1] + avg![2] * q[2] + avg![3] * q[3]
    );
    expect(dot).toBeCloseTo(1, 6);
  });

  it("空输入返回 null", () => {
    expect(averageQuaternions([])).toBeNull();
  });
});

describe("relativeAxisAngle", () => {
  it("从零位转 90° 能反解出转轴与转角", () => {
    const motion = relativeAxisAngle(IDENTITY, quatAbout(Y, 90));
    expect(motion).not.toBeNull();
    expect(motion!.angleDeg).toBeCloseTo(90, 4);
    expect(motion!.axis[1]).toBeCloseTo(1, 6);
  });

  it("参考姿态本身不是单位旋转时也成立（算的是相对量）", () => {
    const ref = quatAbout([1, 2, -3], 57);
    const motion = relativeAxisAngle(ref, multiplyQuat(ref, quatAbout(X, 30)));
    expect(motion!.angleDeg).toBeCloseTo(30, 4);
    expect(motion!.axis[0]).toBeCloseTo(1, 6);
  });

  it("两个姿态相同时转轴无定义，返回 null 而不是 NaN", () => {
    expect(relativeAxisAngle(IDENTITY, IDENTITY)).toBeNull();
  });
});

describe("axisSeparationDeg", () => {
  it("正交两轴 = 90°，同向/反向都算 0°（轴的符号无意义）", () => {
    expect(axisSeparationDeg(X, Y)).toBeCloseTo(90, 6);
    expect(axisSeparationDeg(X, X)).toBeCloseTo(0, 6);
    expect(axisSeparationDeg(X, [-1, 0, 0])).toBeCloseTo(0, 6);
  });
});

describe("buildAxisMap", () => {
  it("传感器轴与模型轴一致时得到单位矩阵", () => {
    const map = buildAxisMap(X, Y, X, Y)!;
    expect(map[0]).toEqual([1, 0, 0]);
    expect(map[1][1]).toBeCloseTo(1, 10);
    expect(map[2][2]).toBeCloseTo(1, 10);
  });

  it("两个实测轴共线时无法定第三轴，返回 null", () => {
    expect(buildAxisMap(X, [2, 0, 0], X, Y)).toBeNull();
  });
});

describe("applyOrientationCalib", () => {
  it("没有标定时原样返回", () => {
    const q = quatAbout(Z, 33);
    expect(applyOrientationCalib(q, null)).toEqual(q);
  });

  it("零位那一帧被映射成单位旋转", () => {
    const ref = quatAbout([0.3, -0.7, 0.2], 88);
    const out = applyOrientationCalib(ref, { reference: ref });
    expect(out[0]).toBeCloseTo(1, 6);
    expect(Math.hypot(out[1], out[2], out[3])).toBeCloseTo(0, 6);
  });

  it("零位只减固定旋转、不改变相对转角", () => {
    const ref = quatAbout(Y, 120);
    const q = multiplyQuat(ref, quatAbout(X, 45));
    const out = applyOrientationCalib(q, { reference: ref });
    const motion = relativeAxisAngle(IDENTITY, out)!;
    expect(motion.angleDeg).toBeCloseTo(45, 4);
    expect(motion.axis[0]).toBeCloseTo(1, 6);
  });

  it("轴向映射把转轴换算到模型轴、转角不变", () => {
    // 传感器绕 +Z 对应模型的俯仰轴 −X；绕 +Y 对应模型的偏摆轴 +Y
    const map = buildAxisMap(Z, Y, [-1, 0, 0], Y)!;
    const out = applyOrientationCalib(quatAbout(Z, 30), {
      reference: IDENTITY,
      axisMap: map,
    });
    const motion = relativeAxisAngle(IDENTITY, out)!;
    expect(motion.angleDeg).toBeCloseTo(30, 4); // 转角守恒
    expect(motion.axis[0]).toBeCloseTo(-1, 5); // 轴换成了 −X
    expect(motion.axis[1]).toBeCloseTo(0, 5);
    expect(motion.axis[2]).toBeCloseTo(0, 5);
  });
});

describe("assembleOrientationCalib", () => {
  /** 传感器机体系：竖立→平铺 绕 +Z，竖立→手心相对 绕 +Y（一组正交的合格动作） */
  const zero = IDENTITY;
  const flat = quatAbout(Z, 90);
  const palms = quatAbout(Y, 90);

  it("没有零位就整份作废（零位是唯一必需项）", () => {
    expect(assembleOrientationCalib("LH", null, flat, palms)).toBeNull();
  });

  it("只有零位时给出零位、不给轴向", () => {
    const calib = assembleOrientationCalib("RH", zero, null, null)!;
    expect(calib.reference).toEqual(zero);
    expect(calib.axisMap).toBeUndefined();
    expect(calib.axisQuality).toBeUndefined();
    expect(axisMapFailReason(calib)).toMatch(/零位/);
  });

  it("两个动作都到位时写入轴向映射与质量数", () => {
    const calib = assembleOrientationCalib("LH", zero, flat, palms)!;
    expect(calib.axisMap).toBeDefined();
    expect(calib.axisQuality!.pitchDeg).toBeCloseTo(90, 3);
    expect(calib.axisQuality!.swingDeg).toBeCloseTo(90, 3);
    expect(calib.axisQuality!.separationDeg).toBeCloseTo(90, 3);
    expect(axisMapFailReason(calib)).toBeNull();
  });

  it("左右手的偏摆轴期望相反，所以同样的实测动作会解出不同矩阵", () => {
    const lh = assembleOrientationCalib("LH", zero, flat, palms)!;
    const rh = assembleOrientationCalib("RH", zero, flat, palms)!;
    expect(lh.axisMap).not.toEqual(rh.axisMap);
  });

  it("动作幅度不够时只留零位，并说明是哪一步", () => {
    const small = assembleOrientationCalib("LH", zero, quatAbout(Z, 6), palms)!;
    expect(small.axisMap).toBeUndefined();
    expect(small.axisQuality!.pitchDeg).toBeCloseTo(6, 3);
    expect(axisMapFailReason(small)).toMatch(/平铺/);

    const smallSwing = assembleOrientationCalib(
      "LH",
      zero,
      flat,
      quatAbout(Y, 8)
    )!;
    expect(smallSwing.axisMap).toBeUndefined();
    expect(axisMapFailReason(smallSwing)).toMatch(/手心相对/);
  });

  it("两个动作转轴几乎重合时只留零位（做成了同一个方向）", () => {
    const calib = assembleOrientationCalib(
      "LH",
      zero,
      quatAbout(Z, 90),
      quatAbout([0, 0.2, 1], 90)
    )!;
    expect(calib.axisMap).toBeUndefined();
    expect(calib.axisQuality!.separationDeg).toBeLessThan(25);
    expect(axisMapFailReason(calib)).toMatch(/转轴/);
  });
});

describe("持久化", () => {
  it("没有 localStorage 的环境（node 测试环境）当作未标定，不抛异常", () => {
    expect(() => loadOrientationCalib("LH")).not.toThrow();
    expect(loadOrientationCalib("LH")).toBeNull();
  });
});
