import { describe, it, expect } from "vitest";
import {
  applyOrientationCalib,
  assembleOrientationCalib,
  averageQuaternions,
  axisMapFailReason,
  axisQualityWarning,
  axisSeparationDeg,
  buildAxisMap,
  FLIP_RISK_DEG,
  loadOrientationCalib,
  MIN_MOTION_DEG,
  MIN_SEPARATION_DEG,
  multiplyQuat,
  relativeAxisAngle,
  TARGET_MOTION_DEG,
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

  /*
   * 这条锁的是"门限不能再被调松"。曾经是 15°/25°，实测放行过一份
   * 偏摆 40° / 分离 61° 的标定 —— 界面显示✓、手模跟着动，但方位分不出来。
   * `buildAxisMap` 是强制贴合的，永远返回合法矩阵，所以门限是唯一的防线。
   */
  it("门限不低于参考实现的 40° / 30°", () => {
    expect(MIN_MOTION_DEG).toBeGreaterThanOrEqual(40);
    expect(MIN_SEPARATION_DEG).toBeGreaterThanOrEqual(30);
    expect(FLIP_RISK_DEG).toBeLessThan(180);
  });

  it("旧门限（15~40°）之间的动作现在会被拒收", () => {
    const marginal = assembleOrientationCalib("LH", zero, flat, quatAbout(Y, 30))!;
    expect(marginal.axisQuality!.swingDeg).toBeCloseTo(30, 3);
    expect(marginal.axisMap).toBeUndefined();
    expect(axisMapFailReason(marginal)).toMatch(/手心相对/);
  });

  it("两个动作转轴几乎重合时只留零位（做成了同一个方向）", () => {
    const calib = assembleOrientationCalib(
      "LH",
      zero,
      quatAbout(Z, 90),
      quatAbout([0, 0.2, 1], 90)
    )!;
    expect(calib.axisMap).toBeUndefined();
    expect(calib.axisQuality!.separationDeg).toBeLessThan(MIN_SEPARATION_DEG);
    expect(axisMapFailReason(calib)).toMatch(/转轴/);
  });
});

/*
 * `axisQualityWarning` 管的是"过了门限但仍不好使" —— 这是实际踩到的坑：
 * 过门限后界面只显示一个绿色✓，一份拧歪的矩阵没有任何线索能被发现。
 * 与 `axisMapFailReason` 互斥：矩阵没写入时它必须闭嘴，否则两条提示会打架。
 */
describe("axisQualityWarning", () => {
  const zero = IDENTITY;
  const flat = quatAbout(Z, 90);

  it("三项都接近 90° 时不报警", () => {
    const good = assembleOrientationCalib("LH", zero, flat, quatAbout(Y, 90))!;
    expect(good.axisMap).toBeDefined();
    expect(axisQualityWarning(good)).toBeNull();
  });

  it("转角接近 180° 时报符号不稳 —— 绕 n 转 180° ≡ 绕 −n 转 180°", () => {
    const flip = assembleOrientationCalib("LH", zero, flat, quatAbout(Y, 172))!;
    expect(flip.axisMap).toBeDefined(); // 过了门限，确实写进去了
    expect(axisQualityWarning(flip)).toMatch(/180°|正负号/);
  });

  it("转过头但还没到翻转带时，报的是精度而不是符号", () => {
    const over = assembleOrientationCalib("LH", zero, flat, quatAbout(Y, 150))!;
    expect(over.axisQuality!.swingDeg).toBeGreaterThan(TARGET_MOTION_DEG + 30);
    expect(over.axisQuality!.swingDeg).toBeLessThan(FLIP_RISK_DEG);
    const warn = axisQualityWarning(over)!;
    expect(warn).toMatch(/偏远/);
    expect(warn).not.toMatch(/正负号/);
  });

  it("轴分离偏小时点出矩阵有一部分是补出来的", () => {
    // 分离约 45°：过了 30° 门限，但 Gram-Schmidt 要扣掉相当一部分实测方向
    const skew = assembleOrientationCalib(
      "LH",
      zero,
      flat,
      quatAbout([0, 1, 1], 90)
    )!;
    expect(skew.axisMap).toBeDefined();
    expect(axisQualityWarning(skew)).toMatch(/正交化|分开/);
  });

  it("矩阵没写入时不报警（该由 axisMapFailReason 说话，两条提示不能打架）", () => {
    const failed = assembleOrientationCalib("LH", zero, flat, quatAbout(Y, 8))!;
    expect(failed.axisMap).toBeUndefined();
    expect(axisQualityWarning(failed)).toBeNull();
    expect(axisMapFailReason(failed)).not.toBeNull();
  });
});

/*
 * ===== 重力核对：拦住"③ 翻反方向" =====
 *
 * 这一组锁的是实际报上来的故障：**左手摆拇指朝上，手模显示拇指朝下**。
 * 根因不是手模镜像（`HandModel.tsx` 的 ±0.78 与 glb 手性核对过，是对的），
 * 也不是 `MODEL_MOTION_AXES`（三个目标值按 glb 零位反推过，是对的），
 * 而是 ③ 那步往外翻代替了往内翻 —— 解出来的矩阵合法、三个质量数全绿。
 *
 * 下面先用一条端到端的测试**复现**这个症状，再验证重力核对能纠正它。
 */
describe("③ 翻反方向：症状复现与重力核对", () => {
  /**
   * 造一份合成的左手标定输入。传感器机体系刻意取一个**歪的装配朝向** `mount`，
   * 这样测的就不是"恰好对齐时碰巧能过"。
   *
   * 模型系的三个动作（已按 glb 实测核对，见 orientationCalib.ts 文件头）：
   *   零位  指尖 +Y、掌心 +Z、拇指 −X（左手）
   *   平铺  绕 −X 转 90°
   *   相对  左手绕 +Y 转 90°（**往内** = 掌心转向使用者右边）
   */
  const mount = quatAbout([0.37, -0.62, 0.19], 47); // 机体系→模型系的装配旋转
  /** 把模型系的转动搬到机体系：机体系里看到的就是 mount⁻¹ ⊗ R ⊗ mount */
  const inBody = (axis: Vec3, deg: number): Quat =>
    multiplyQuat(
      multiplyQuat([mount[0], -mount[1], -mount[2], -mount[3]], quatAbout(axis, deg)),
      mount
    );

  const zero = IDENTITY;
  const flat = inBody([-1, 0, 0], 90);
  const palmsInward = inBody(Y, 90); // ③ 做对：左手绕 +Y
  const palmsOutward = inBody(Y, -90); // ③ 做反：往外翻

  /** 用四元数 [w,x,y,z] 旋转一个向量 */
  const rotByQuat = (q: Quat, v: Vec3): Vec3 => {
    const [w, x, y, z] = q;
    const tx = 2 * (y * v[2] - z * v[1]);
    const ty = 2 * (z * v[0] - x * v[2]);
    const tz = 2 * (x * v[1] - y * v[0]);
    return [
      v[0] + w * tx + (y * tz - z * ty),
      v[1] + w * ty + (z * tx - x * tz),
      v[2] + w * tz + (x * ty - y * tx),
    ];
  };
  const conj = (q: Quat): Quat => [q[0], -q[1], -q[2], -q[3]];
  /** 模型系的方向搬到机体系（零位那一刻） */
  const toBody = (v: Vec3, scale = 9.81): Vec3 =>
    rotByQuat(conj(mount), v).map((c) => c * scale) as Vec3;

  /**
   * 零位那一刻加速度计在机体系里测到的"上"。
   * 静止时 acc 指上（`imuHealth.ts:213` 拿它与 `gravityInHandFrame` 比夹角，同向），
   * 而零位的模型系上就是 +Y。乘 9.81 顺带证明这一关只看方向、不在乎量纲。
   */
  const accUp = toBody([0, 1, 0]);

  /** 摆"拇指朝上"：左手零位拇指在 −X，要转到 +Y，就是绕 Z 转 −90° */
  const thumbUpBody = inBody(Z, -90);

  /** 标定后手模的拇指指向（模型系）。左手零位拇指 = −X */
  const thumbDir = (calib: { reference: Quat; axisMap?: number[][] }): Vec3 =>
    rotByQuat(applyOrientationCalib(thumbUpBody, calib), [-1, 0, 0]);

  it("③ 做对时，拇指朝上就显示朝上（基线）", () => {
    const calib = assembleOrientationCalib("LH", zero, flat, palmsInward, accUp)!;
    expect(calib.axisMap).toBeDefined();
    expect(thumbDir(calib)[1]).toBeCloseTo(1, 4); // +Y = 朝上
  });

  it("③ 翻反方向：三个质量数全部正常，但拇指朝上显示成朝下 —— 这就是报上来的症状", () => {
    // 不传 acc = 老行为（没有重力核对这一关）
    const bad = assembleOrientationCalib("LH", zero, flat, palmsOutward)!;
    expect(bad.axisMap).toBeDefined();
    // 三个数看起来完美，所以界面只会显示一个 ✓
    expect(bad.axisQuality!.pitchDeg).toBeCloseTo(90, 3);
    expect(bad.axisQuality!.swingDeg).toBeCloseTo(90, 3);
    expect(bad.axisQuality!.separationDeg).toBeCloseTo(90, 3);
    expect(axisMapFailReason(bad)).toBeNull();
    expect(axisQualityWarning(bad)).toBeNull();
    // 而拇指是**反的**
    expect(thumbDir(bad)[1]).toBeCloseTo(-1, 4); // −Y = 朝下
  });

  it("同一份错误输入，给了加速度就被自动纠正回朝上", () => {
    const fixed = assembleOrientationCalib(
      "LH",
      zero,
      flat,
      palmsOutward,
      accUp
    )!;
    expect(fixed.gravityCheck!.corrected).toBe(true);
    expect(fixed.gravityCheck!.upY).toBeLessThan(-0.5);
    expect(thumbDir(fixed)[1]).toBeCloseTo(1, 4); // 已经朝上
  });

  it("③ 做对的那份不会被反向纠正坏掉（这一关必须是幂等的）", () => {
    const good = assembleOrientationCalib("LH", zero, flat, palmsInward, accUp)!;
    expect(good.gravityCheck!.corrected).toBe(false);
    expect(good.gravityCheck!.upY).toBeCloseTo(1, 3);
    const noAcc = assembleOrientationCalib("LH", zero, flat, palmsInward)!;
    // 本来就对的，加不加这一关矩阵都一样
    expect(good.axisMap).toEqual(noAcc.axisMap);
  });

  it("右手同样成立（偏摆轴期望相反，纠正轴也跟着走）", () => {
    const rhOutward = inBody(Y, 90); // 右手往内是绕 −Y，这里给成 +Y = 反的
    const bad = assembleOrientationCalib("RH", zero, flat, rhOutward)!;
    const fixed = assembleOrientationCalib("RH", zero, flat, rhOutward, accUp)!;
    expect(bad.gravityCheck).toBeUndefined();
    expect(fixed.gravityCheck!.corrected).toBe(true);
    expect(fixed.axisMap).not.toEqual(bad.axisMap);
  });

  it("旧款手套没有加速度：不核对、也不谎报通过", () => {
    const calib = assembleOrientationCalib("LH", zero, flat, palmsInward, null)!;
    expect(calib.axisMap).toBeDefined();
    expect(calib.gravityCheck).toBeUndefined(); // 缺席 ≠ 通过
  });

  it("acc 全零（那一路没数据）当作没核对，不会被当成朝下去乱纠正", () => {
    const calib = assembleOrientationCalib("LH", zero, flat, palmsInward, [0, 0, 0])!;
    expect(calib.gravityCheck).toBeUndefined();
    expect(calib.axisMap).toEqual(
      assembleOrientationCalib("LH", zero, flat, palmsInward)!.axisMap
    );
  });

  it("零位那步前臂没竖直 → 没定论：不纠正，并且报出来", () => {
    // "上"偏离模型系 +Y 整整 70°，落在 ±60° 的判决带之外。
    // 注意要在**模型系**里先把 +Y 真转过去（绕 X 转 70° → (0, cos70, sin70)），
    // 再整体搬回机体系；直接对机体系向量按分量缩放不是旋转，偏角会对不上。
    const tilted = toBody([0, Math.cos(70 * DEG), Math.sin(70 * DEG)]);
    const calib = assembleOrientationCalib("LH", zero, flat, palmsInward, tilted)!;
    expect(calib.gravityCheck!.corrected).toBe(false);
    expect(Math.abs(calib.gravityCheck!.upY)).toBeLessThan(0.5);
    // 这种情况必须说出来：这一关等于没设防
    expect(axisQualityWarning(calib)).toMatch(/重力核对没定论|竖直/);
  });

  it("重力核对只看方向，不受加速度量纲影响", () => {
    const scaled = accUp.map((v) => v * 137) as Vec3;
    const a = assembleOrientationCalib("LH", zero, flat, palmsOutward, accUp)!;
    const b = assembleOrientationCalib("LH", zero, flat, palmsOutward, scaled)!;
    expect(b.axisMap).toEqual(a.axisMap);
    expect(b.gravityCheck!.upY).toBeCloseTo(a.gravityCheck!.upY, 6);
  });

  it("轴向矩阵没写入时不做核对（没有矩阵可核）", () => {
    const noMap = assembleOrientationCalib("LH", zero, flat, inBody(Y, 8), accUp)!;
    expect(noMap.axisMap).toBeUndefined();
    expect(noMap.gravityCheck).toBeUndefined();
  });
});

describe("持久化", () => {
  it("没有 localStorage 的环境（node 测试环境）当作未标定，不抛异常", () => {
    expect(() => loadOrientationCalib("LH")).not.toThrow();
    expect(loadOrientationCalib("LH")).toBeNull();
  });
});
