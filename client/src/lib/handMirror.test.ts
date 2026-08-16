import { describe, it, expect } from "vitest";
import {
  MIRROR_PERM_137,
  mirrorSensorSeries,
  mirrorQuat,
  mirrorVec3,
  mirrorImuSeries,
  normalizeHandedness,
  mirrorSample,
  mirrorStaticInputs,
  mirrorHandInput,
} from "./handMirror";
import {
  LEFT_HAND_INDEX_MAP,
  RIGHT_HAND_INDEX_MAP,
} from "./sensorMapping";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  type SequenceSample,
} from "./datasetStore";
import {
  quatMul,
  quatNormalize,
  gravityInHandFrame,
  type Quat,
} from "./sequenceFeatures";

// ===== 置换表 =====

describe("MIRROR_PERM_137", () => {
  it("能从 sensorMapping 的两张索引表独立重算出来", () => {
    // 这一条是整张表的**来源锁**：置换不是手写的，而是
    // 「1-based byte n ↔ 257−n」（0-based b ↔ 255−b）在 137 维槽位上的像。
    // 规格书 V1.1 p10 的 16x16 点阵与厂家另一份实现的解剖学网格都给出同一个关系，
    // 所以只要 sensorMapping 的两张表还是那两张表，这里就必须重算得一致。
    const leftSlotOf = new Map<number, number>();
    LEFT_HAND_INDEX_MAP.forEach((b, k) => leftSlotOf.set(b, k));
    const recomputed = RIGHT_HAND_INDEX_MAP.map((b) => {
      const slot = leftSlotOf.get(255 - b);
      expect(slot, `右手 byte ${b} 的镜像 ${255 - b} 不在左手表里`).toBeDefined();
      return slot!;
    });
    expect([...MIRROR_PERM_137]).toEqual(recomputed);
  });

  it("是 137 维上的双射", () => {
    expect(MIRROR_PERM_137).toHaveLength(SEQ_SENSOR_N);
    expect([...MIRROR_PERM_137].sort((a, b) => a - b)).toEqual(
      Array.from({ length: SEQ_SENSOR_N }, (_, i) => i)
    );
  });

  it("自逆 —— 左→右和右→左是同一张表", () => {
    for (let k = 0; k < SEQ_SENSOR_N; k++) {
      expect(MIRROR_PERM_137[MIRROR_PERM_137[k]]).toBe(k);
    }
  });

  it("不跨块：手指落在手指、弯折落在弯折、手掌落在手掌", () => {
    for (let k = 0; k < 60; k++) expect(MIRROR_PERM_137[k]).toBeLessThan(60);
    for (let k = 60; k < 65; k++) {
      expect(MIRROR_PERM_137[k]).toBeGreaterThanOrEqual(60);
      expect(MIRROR_PERM_137[k]).toBeLessThan(65);
    }
    for (let k = 65; k < SEQ_SENSOR_N; k++)
      expect(MIRROR_PERM_137[k]).toBeGreaterThanOrEqual(65);
  });

  it("弯折五路整体翻号 —— 左手 [小,无,中,食,拇] 对上右手 [拇,食,中,无,小]", () => {
    expect([...MIRROR_PERM_137].slice(60, 65)).toEqual([64, 63, 62, 61, 60]);
  });

  it("同一根物理手指的压力块整体对应（拇指↔拇指、小指↔小指）", () => {
    // 右手块序是 拇,食,中,无,小；左手块序是 小,无,中,食,拇 —— 应当整块反过来
    for (let blk = 0; blk < 5; blk++) {
      const targetBlk = 4 - blk;
      for (let i = 0; i < 12; i++) {
        const dst = MIRROR_PERM_137[blk * 12 + i];
        expect(Math.floor(dst / 12)).toBe(targetBlk);
      }
    }
  });
});

// ===== 传感器序列 =====

describe("mirrorSensorSeries", () => {
  const T = 4;
  const src = new Uint8Array(T * SEQ_SENSOR_N);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < SEQ_SENSOR_N; c++)
      src[t * SEQ_SENSOR_N + c] = (t * 31 + c * 7) % 256;

  it("镜像两次回到原值（自逆表的直接后果）", () => {
    const twice = mirrorSensorSeries(mirrorSensorSeries(src, T), T);
    expect([...twice]).toEqual([...src]);
  });

  it("逐帧按表搬，不串帧", () => {
    const out = mirrorSensorSeries(src, T);
    for (let t = 0; t < T; t++)
      for (let c = 0; c < SEQ_SENSOR_N; c++)
        expect(out[t * SEQ_SENSOR_N + MIRROR_PERM_137[c]]).toBe(
          src[t * SEQ_SENSOR_N + c]
        );
  });

  it("左手小指弯折落到右手小指弯折那一路", () => {
    // 左手 60 是小指弯折、右手 64 是小指弯折。这一条是"换手之后模型看到的是
    // 同一根手指"这件事最短的证明
    const one = new Uint8Array(SEQ_SENSOR_N);
    one[60] = 200; // 左手：小指弯折
    const out = mirrorSensorSeries(one, 1);
    expect(out[64]).toBe(200); // 右手：小指弯折
    expect(out[60]).toBe(0); // 不该落在右手拇指弯折上
  });
});

// ===== IMU =====

describe("mirrorQuat / mirrorVec3", () => {
  const rot = (axis: [number, number, number], deg: number): Quat => {
    const r = (deg * Math.PI) / 360; // 半角
    const s = Math.sin(r);
    return quatNormalize([Math.cos(r), axis[0] * s, axis[1] * s, axis[2] * s]);
  };

  it("绕 x（镜面法向）的转动镜像后不变", () => {
    const q = rot([1, 0, 0], 40);
    const m = mirrorQuat(q as [number, number, number, number]);
    m.forEach((v, i) => expect(v).toBeCloseTo(q[i], 10));
  });

  it("绕 y / 绕 z 的转动镜像后反向", () => {
    for (const axis of [[0, 1, 0], [0, 0, 1]] as [number, number, number][]) {
      const q = rot(axis, 55);
      const m = mirrorQuat(q as [number, number, number, number]);
      const back = rot(axis, -55);
      m.forEach((v, i) => expect(v).toBeCloseTo(back[i], 10));
    }
  });

  it("镜像两次是恒等", () => {
    const q = rot([0.3, -0.5, 0.8], 77);
    const twice = mirrorQuat(
      mirrorQuat(q as [number, number, number, number])
    );
    twice.forEach((v, i) => expect(v).toBeCloseTo(q[i], 12));
  });

  it("对四元数乘法是同态 —— 相对四元数会自动跟着镜像", () => {
    // 特征层用的是 q₀⁻¹⊗qₜ。这一条保证「镜像每帧原始四元数」等价于
    // 「镜像相对四元数」，所以不需要在特征层再补一次
    const a = rot([0.2, 0.9, -0.3], 34);
    const b = rot([-0.7, 0.1, 0.7], 61);
    const lhs = mirrorQuat(quatMul(a, b) as [number, number, number, number]);
    const rhs = quatMul(
      mirrorQuat(a as [number, number, number, number]),
      mirrorQuat(b as [number, number, number, number])
    );
    lhs.forEach((v, i) => expect(v).toBeCloseTo(rhs[i], 10));
  });

  it("重力在手系的投影自动变成镜像后的重力", () => {
    // gravityInHandFrame 是 q 的派生量，所以镜像 q 之后它应当正好等于
    // mirrorVec3(原来的重力)。这条锁住"镜面法向 = x"与 mirrorVec3 的一致性
    const q = rot([0.4, 0.5, -0.76], 88);
    const g = gravityInHandFrame(q);
    const gm = gravityInHandFrame(
      mirrorQuat(q as [number, number, number, number])
    );
    const expected = mirrorVec3(g);
    gm.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 10));
  });
});

describe("mirrorImuSeries", () => {
  it("quat 翻 y/z、acc 翻 x、att 翻 pitch/yaw，其余不动", () => {
    const imu = new Float32Array(SEQ_IMU_N);
    for (let i = 0; i < SEQ_IMU_N; i++) imu[i] = i + 1;
    const out = mirrorImuSeries(imu, 1);
    expect([...out]).toEqual([1, 2, -3, -4, -5, 6, 7, 8, -9, -10]);
  });

  it("镜像两次回到原值", () => {
    const T = 3;
    const imu = new Float32Array(T * SEQ_IMU_N);
    for (let i = 0; i < imu.length; i++) imu[i] = Math.sin(i) * 2;
    const twice = mirrorImuSeries(mirrorImuSeries(imu, T), T);
    twice.forEach((v, i) => expect(v).toBeCloseTo(imu[i], 6));
  });
});

// ===== mirrorSample / normalizeHandedness =====

function makeSample(which: "left" | "right" | "both" | "none"): SequenceSample {
  const T = 4;
  // 两只手内容必须**不同**，否则"互换"这件事在测试里看不出来
  const sensor = (seed: number) => {
    const s = new Uint8Array(T * SEQ_SENSOR_N);
    for (let i = 0; i < s.length; i++) s[i] = (i * 13 + seed) % 256;
    return s;
  };
  const imu = (seed: number) => {
    const a = new Float32Array(T * SEQ_IMU_N);
    for (let i = 0; i < a.length; i++) a[i] = Math.sin(i + seed);
    return a;
  };
  const hasL = which === "left" || which === "both";
  const hasR = which === "right" || which === "both";
  return {
    segments: [{ label: "_live", startFrame: 0, endFrame: T }],
    primaryLabel: "_live",
    frameCount: T,
    timestamps: new Float32Array([0, 20, 40, 60]),
    leftSensor: hasL ? sensor(0) : null,
    rightSensor: hasR ? sensor(101) : null,
    leftImu: hasL ? imu(0) : null,
    rightImu: hasR ? imu(7) : null,
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: 80,
    sourceFps: 50,
    origin: "recorded",
    timestamp: 0,
  };
}

describe("mirrorSample", () => {
  it("两只手一起镜像 + 槽位互换", () => {
    const s = makeSample("both");
    const m = mirrorSample(s);
    expect([...m.rightSensor!]).toEqual([
      ...mirrorSensorSeries(s.leftSensor!, s.frameCount),
    ]);
    expect([...m.leftSensor!]).toEqual([
      ...mirrorSensorSeries(s.rightSensor!, s.frameCount),
    ]);
    m.rightImu!.forEach((v, i) =>
      expect(v).toBeCloseTo(mirrorImuSeries(s.leftImu!, s.frameCount)[i], 6)
    );
  });

  it("自逆 —— 镜像两次回到原样本", () => {
    const s = makeSample("both");
    const twice = mirrorSample(mirrorSample(s));
    expect([...twice.leftSensor!]).toEqual([...s.leftSensor!]);
    expect([...twice.rightSensor!]).toEqual([...s.rightSensor!]);
    twice.leftImu!.forEach((v, i) => expect(v).toBeCloseTo(s.leftImu![i], 5));
  });

  it("空的一路保持空 —— 只有左手时退化成'搬到右手槽位'", () => {
    const s = makeSample("left");
    const m = mirrorSample(s);
    expect(m.leftSensor).toBeNull();
    expect(m.leftImu).toBeNull();
    expect([...m.rightSensor!]).toEqual([
      ...mirrorSensorSeries(s.leftSensor!, s.frameCount),
    ]);
  });

  it("不动原样本（推理滑窗的快照可能还被别处引用）", () => {
    const s = makeSample("both");
    const before = [...s.leftSensor!];
    mirrorSample(s);
    expect([...s.leftSensor!]).toEqual(before);
  });

  it("视觉通道原样带过（滑窗本来就是纯触觉两路都 null）", () => {
    const m = mirrorSample(makeSample("both"));
    expect(m.leftLandmarks).toBeNull();
    expect(m.rightLandmarks).toBeNull();
  });
});

describe("normalizeHandedness", () => {
  it("指定左手 + 双手套都连着 → 照样镜像（这是唯一能覆盖那种用法的路径）", () => {
    const s = makeSample("both");
    const r = normalizeHandedness(s, "left");
    expect(r.mirrored).toBe(true);
    expect(r.reason).toBe("mirrored");
    expect([...r.sample.rightSensor!]).toEqual([
      ...mirrorSensorSeries(s.leftSensor!, s.frameCount),
    ]);
    // 闲着那只手不是被清零，而是镜像到对面 —— 训练时它也在出静止数据
    expect(r.sample.leftSensor).not.toBeNull();
  });

  it("指定右手 → 原样返回，哪怕左手套也连着", () => {
    const s = makeSample("both");
    const r = normalizeHandedness(s, "right");
    expect(r.mirrored).toBe(false);
    expect(r.reason).toBe("already_right");
    expect(r.sample).toBe(s); // 同一个对象，没有多余拷贝
  });

  it("auto + 只有左手 → 镜像", () => {
    const s = makeSample("left");
    const r = normalizeHandedness(s, "auto");
    expect(r.mirrored).toBe(true);
    expect(r.reason).toBe("mirrored");
    expect(r.sample.leftSensor).toBeNull();
  });

  it("auto + 只有右手 → 不动（本来就是训练口径）", () => {
    const s = makeSample("right");
    const r = normalizeHandedness(s, "auto");
    expect(r.mirrored).toBe(false);
    expect(r.reason).toBe("already_right");
    expect(r.sample).toBe(s);
  });

  it("auto + 双手都有 → 判不了，报 both_hands 且不动", () => {
    // 闲着那只手也在出静止数据，从数据上分不出"没戴"和"戴着不动"，
    // 所以这里必须交给界面上的主手选择器
    const s = makeSample("both");
    const r = normalizeHandedness(s, "auto");
    expect(r.mirrored).toBe(false);
    expect(r.reason).toBe("both_hands");
    expect(r.sample).toBe(s);
  });

  it("默认参数就是 auto", () => {
    expect(normalizeHandedness(makeSample("both")).reason).toBe("both_hands");
    expect(normalizeHandedness(makeSample("left")).reason).toBe("mirrored");
  });

  it("两只手都没有 → no_hand，指定了主手也一样不崩", () => {
    for (const d of ["auto", "left", "right"] as const) {
      const r = normalizeHandedness(makeSample("none"), d);
      expect(r.mirrored).toBe(false);
      expect(r.reason).toBe("no_hand");
    }
  });
});

describe("mirrorStaticInputs", () => {
  const mk = (seed: number) => ({
    sensor_data: Array.from({ length: SEQ_SENSOR_N }, (_, i) => (i * 3 + seed) % 256),
    quaternion: [0.5, 0.5, 0.5, 0.5] as [number, number, number, number],
  });

  it("互换 + 各自镜像", () => {
    const l = mk(0);
    const r = mk(50);
    const out = mirrorStaticInputs(l, r);
    for (let k = 0; k < SEQ_SENSOR_N; k++) {
      expect(out.right!.sensor_data[MIRROR_PERM_137[k]]).toBe(l.sensor_data[k]);
      expect(out.left!.sensor_data[MIRROR_PERM_137[k]]).toBe(r.sensor_data[k]);
    }
  });

  it("只有左手时右手仍是 null（不凭空造一只手出来）", () => {
    const out = mirrorStaticInputs(mk(0), null);
    expect(out.left).toBeNull();
    expect(out.right).not.toBeNull();
  });

  it("两边都 null 不崩", () => {
    expect(mirrorStaticInputs(null, null)).toEqual({ left: null, right: null });
  });
});

// ===== mirrorHandInput（静态单帧那条路） =====

describe("mirrorHandInput", () => {
  it("137 维按同一张表搬，四元数按同一规则镜像", () => {
    const sensor = Array.from({ length: SEQ_SENSOR_N }, (_, i) => (i * 5) % 256);
    const q: [number, number, number, number] = [0.5, 0.5, 0.5, 0.5];
    const out = mirrorHandInput({ sensor_data: sensor, quaternion: q });
    for (let k = 0; k < SEQ_SENSOR_N; k++)
      expect(out.sensor_data[MIRROR_PERM_137[k]]).toBe(sensor[k]);
    expect(out.quaternion).toEqual([0.5, 0.5, -0.5, -0.5]);
  });

  it("输入短于 137 时补 0，不越界", () => {
    const out = mirrorHandInput({
      sensor_data: [7, 8, 9],
      quaternion: [1, 0, 0, 0],
    });
    expect(out.sensor_data).toHaveLength(SEQ_SENSOR_N);
    expect(out.sensor_data[MIRROR_PERM_137[0]]).toBe(7);
    expect(out.sensor_data.reduce((a, b) => a + b, 0)).toBe(24);
  });

  it("镜像两次回到原值", () => {
    const sensor = Array.from({ length: SEQ_SENSOR_N }, (_, i) => (i * 11) % 256);
    const q: [number, number, number, number] = [0.1, -0.2, 0.3, 0.9];
    const twice = mirrorHandInput(mirrorHandInput({ sensor_data: sensor, quaternion: q }));
    expect(twice.sensor_data).toEqual(sensor);
    twice.quaternion.forEach((v, i) => expect(v).toBeCloseTo(q[i], 12));
  });
});
