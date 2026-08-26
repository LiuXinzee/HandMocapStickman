/*
 * dominantHand.test —— 主手自动判定
 *
 * 这里锁的是四件"错了会很难查"的事：
 *  1. 静止手 vs 活动手能分开（这是整个改造的前提，原代码断言它分不开）；
 *  2. 左右手量程悬殊时不会因为量程大就被判成主手（实测左右量程差 40~176）；
 *  3. 弯折下标左右手是**镜像**的，取值不能两只手都按同一个顺序算；
 *  4. 滞回真的在拦翻转，且静止间隙不会把攒到一半的翻转抹掉。
 */
import { describe, expect, it } from "vitest";
import {
  DOMINANCE_RATIO,
  DominanceTracker,
  FLIP_WINDOWS,
  IDLE_ENERGY,
  handEnergy,
  judgeSampleDominance,
  normalizeSamplesToRight,
  summarizeHandedness,
} from "./dominantHand";
import { mirrorSensorSeries } from "./handMirror";
import {
  SEQ_IMU_N,
  SEQ_SENSOR_N,
  type SequenceSample,
} from "./datasetStore";
import type { BendRange } from "./bendRange";

const T = 50; // 1s @ 50Hz，与 DOMINANCE_WINDOW_MS 一致

const BEND_OFFSET = 60;

/** 确定性伪随机，避免用 Math.random 让测试变成偶发失败 */
function noise(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000 - 0.5; // −0.5 ~ 0.5
  };
}

interface HandSpec {
  /** 弯折各路在窗口内扫过的量程比例：0 = 完全不动，1 = 从张开扫到握拳 */
  sweep: number;
  /** 弯折的物理量程（ADC），用来生成读数 */
  span: number;
  /** 指压变化幅度（ADC） */
  pressAmp: number;
  /** 窗口内最大转角（度） */
  rotDeg: number;
  /** 噪声幅度（ADC 峰峰值） */
  noiseAmp?: number;
}

const STILL: HandSpec = { sweep: 0, span: 120, pressAmp: 0, rotDeg: 0, noiseAmp: 3 };
const ACTIVE: HandSpec = { sweep: 0.6, span: 120, pressAmp: 40, rotDeg: 60 };

function buildHand(
  spec: HandSpec,
  seed: number
): { sensor: Uint8Array; imu: Float32Array } {
  const sensor = new Uint8Array(T * SEQ_SENSOR_N);
  const imu = new Float32Array(T * SEQ_IMU_N);
  const rnd = noise(seed);
  const amp = spec.noiseAmp ?? 1;
  for (let t = 0; t < T; t++) {
    // 半个正弦：0 → 峰 → 0，就是"张开→握拳→张开"一次
    const phase = Math.sin((Math.PI * t) / (T - 1));
    for (let j = 0; j < 5; j++) {
      const v = 40 + spec.sweep * spec.span * phase + rnd() * amp;
      sensor[t * SEQ_SENSOR_N + BEND_OFFSET + j] = Math.max(
        0,
        Math.min(255, Math.round(v))
      );
    }
    for (let j = 0; j < 60; j++) {
      const v = 20 + spec.pressAmp * phase + rnd() * amp;
      sensor[t * SEQ_SENSOR_N + j] = Math.max(0, Math.min(255, Math.round(v)));
    }
    // 绕 x 轴转 rotDeg，走"出去再回来"，所以净转角是 0 —— 能量必须靠最大偏离量到
    const half = ((spec.rotDeg * phase) / 2 / 180) * Math.PI;
    const o = t * SEQ_IMU_N;
    imu[o] = Math.cos(half);
    imu[o + 1] = Math.sin(half);
  }
  return { sensor, imu };
}

function makeSample(
  left: HandSpec | null,
  right: HandSpec | null
): SequenceSample {
  const l = left ? buildHand(left, 12345) : null;
  const r = right ? buildHand(right, 98765) : null;
  return {
    segments: [{ label: "_dom", startFrame: 0, endFrame: T }],
    primaryLabel: "_dom",
    frameCount: T,
    timestamps: new Float32Array(T),
    leftSensor: l?.sensor ?? null,
    rightSensor: r?.sensor ?? null,
    leftImu: l?.imu ?? null,
    rightImu: r?.imu ?? null,
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: 1000,
    sourceFps: 50,
    origin: "recorded",
    timestamp: 0,
  };
}

const range = (span: number): BendRange => ({
  open: [40, 40, 40, 40, 40],
  fist: [40 + span, 40 + span, 40 + span, 40 + span, 40 + span],
});

const BOTH_ON = { left: true, right: true };

describe("handEnergy", () => {
  it("静止的手能量低于静止门限，做动作的手远高于 1", () => {
    const still = buildHand(STILL, 1);
    const active = buildHand(ACTIVE, 2);
    const eStill = handEnergy(still.sensor, still.imu, T, "RH", [120, 120, 120, 120, 120])!;
    const eActive = handEnergy(active.sensor, active.imu, T, "RH", [120, 120, 120, 120, 120])!;
    expect(eStill.total).toBeLessThan(IDLE_ENERGY);
    expect(eActive.total).toBeGreaterThan(1);
    // 这是整个改造成立的前提：两者要拉开得比判定门限还远
    expect(eActive.total / eStill.total).toBeGreaterThan(DOMINANCE_RATIO);
  });

  it("朝向能量取最大偏离角，出去再回来的动作不会被算成 0", () => {
    const a = buildHand({ ...STILL, rotDeg: 60, noiseAmp: 0 }, 3);
    // 首尾都是单位四元数（净转角 0），最大偏离是 60° = 2 份
    const e = handEnergy(a.sensor, a.imu, T, "RH", null)!;
    expect(e.orient).toBeGreaterThan(1.5);
  });

  it("除以各自量程：量程差一倍，同样的扫动比例给出同样的弯折能量", () => {
    const wide = buildHand({ ...ACTIVE, span: 176, noiseAmp: 0 }, 4);
    const narrow = buildHand({ ...ACTIVE, span: 88, noiseAmp: 0 }, 4);
    const eWide = handEnergy(wide.sensor, wide.imu, T, "RH", [176, 176, 176, 176, 176])!;
    const eNarrow = handEnergy(narrow.sensor, narrow.imu, T, "RH", [88, 88, 88, 88, 88])!;
    expect(eWide.bend).toBeCloseTo(eNarrow.bend, 1);
  });

  it("左手弯折槽位是镜像的：量程数组按 canonical 顺序对齐", () => {
    // 只让物理第 0 路动。左手物理 0 = 小指 = canonical 4，右手物理 0 = 拇指 = canonical 0
    const a = buildHand({ ...STILL, noiseAmp: 0 }, 5);
    for (let t = 0; t < T; t++) {
      const phase = Math.sin((Math.PI * t) / (T - 1));
      a.sensor[t * SEQ_SENSOR_N + BEND_OFFSET] = Math.round(40 + 100 * phase);
    }
    // canonical 量程：拇指那一路给 50、小指那一路给 200，其余不可用
    const spans = [50, 10, 10, 10, 200];
    const lh = handEnergy(a.sensor, a.imu, T, "LH", spans)!;
    const rh = handEnergy(a.sensor, a.imu, T, "RH", spans)!;
    // 同一份数据：右手该除 50、左手该除 200，所以右手的弯折能量必须明显更大
    expect(rh.bend).toBeGreaterThan(lh.bend * 3);
  });

  it("没数据 / 帧数不足 → null", () => {
    expect(handEnergy(null, null, T, "RH", null)).toBeNull();
    const a = buildHand(STILL, 6);
    expect(handEnergy(a.sensor, a.imu, 1, "RH", null)).toBeNull();
  });
});

describe("judgeSampleDominance（离线，整条样本）", () => {
  const ranges = { LH: range(120), RH: range(120) };

  it("左手做动作、右手戴着不动 → left", () => {
    const v = judgeSampleDominance(makeSample(ACTIVE, STILL), ranges);
    expect(v.dominant).toBe("left");
    expect(v.nearTie).toBe(false);
    expect(v.calibrated).toBe(true);
  });

  it("右手做动作 → right", () => {
    expect(judgeSampleDominance(makeSample(STILL, ACTIVE), ranges).dominant).toBe(
      "right"
    );
  });

  it("两只手都静止 → idle，不镜像（`_idle` 伪类样本走这条）", () => {
    const v = judgeSampleDominance(makeSample(STILL, STILL), ranges);
    expect(v.dominant).toBe("idle");
  });

  it("没有 both —— 两只手都在动仍要给出一个主手（离线镜像不怕拼接）", () => {
    // 流式那边这里会判 both 并放弃镜像；离线是整条一起变换，没有中途翻转的风险，
    // 所以照判。这是两条路径**刻意**的差异
    const v = judgeSampleDominance(makeSample(ACTIVE, ACTIVE), ranges);
    expect(["left", "right"]).toContain(v.dominant);
    expect(v.nearTie).toBe(true); // 但要如实报出"这条基本是蒙的"
  });

  it("只有一只手有数据 → 直接判那只手", () => {
    expect(judgeSampleDominance(makeSample(ACTIVE, null), ranges).dominant).toBe(
      "left"
    );
    expect(judgeSampleDominance(makeSample(null, ACTIVE), ranges).dominant).toBe(
      "right"
    );
  });

  it("两只手都没数据 → no_hand", () => {
    expect(judgeSampleDominance(makeSample(null, null), ranges).dominant).toBe(
      "no_hand"
    );
  });

  it("只有一只手标定 → calibrated=false，两只手一起走兜底量程", () => {
    const v = judgeSampleDominance(makeSample(ACTIVE, STILL), {
      LH: range(120),
      RH: null,
    });
    expect(v.calibrated).toBe(false);
    expect(v.dominant).toBe("left"); // 判定本身仍然成立
  });

  it("量程悬殊但扫动比例相同 → 不偏向量程大的手（判定落在 nearTie 带里）", () => {
    const v = judgeSampleDominance(
      makeSample({ ...ACTIVE, span: 60 }, { ...ACTIVE, span: 176 }),
      { LH: range(60), RH: range(176) }
    );
    expect(v.nearTie).toBe(true);
  });
});

describe("normalizeSamplesToRight", () => {
  const ranges = { LH: range(120), RH: range(120) };

  it("左手主导的整条镜像、右手主导的原样返回同一个对象", () => {
    const left = makeSample(ACTIVE, STILL);
    const right = makeSample(STILL, ACTIVE);
    const { samples, stats } = normalizeSamplesToRight([left, right], ranges);
    expect(stats).toMatchObject({ total: 2, left: 1, right: 1, idle: 0 });
    // 镜像后：原左手的数据搬到右槽并按镜像表换序
    expect([...samples[0].rightSensor!]).toEqual([
      ...mirrorSensorSeries(left.leftSensor!, left.frameCount),
    ]);
    expect(samples[1]).toBe(right); // 没有多余拷贝
  });

  it("静止样本原样带过 —— 不靠噪声 argmax 掷硬币", () => {
    const idle = makeSample(STILL, STILL);
    const { samples, stats } = normalizeSamplesToRight([idle], ranges);
    expect(stats.idle).toBe(1);
    expect(samples[0]).toBe(idle);
  });

  it("同一批数据跑两遍结果一致（训练可复现）", () => {
    const batch = [
      makeSample(ACTIVE, STILL),
      makeSample(STILL, ACTIVE),
      makeSample(STILL, STILL),
    ];
    const a = normalizeSamplesToRight(batch, ranges).stats;
    const b = normalizeSamplesToRight(batch, ranges).stats;
    expect(a).toEqual(b);
  });

  it("不改原样本（调用方可能还拿着同一批引用）", () => {
    const s = makeSample(ACTIVE, STILL);
    const before = [...s.leftSensor!];
    normalizeSamplesToRight([s], ranges);
    expect([...s.leftSensor!]).toEqual(before);
  });

  it("归一化之后左手槽位不再承载主手 —— 这是整件事要达到的效果", () => {
    // 左右混采的两条同词样本，归一化前信号分别落在两段零重叠的槽位里，
    // 归一化后必须都落在右槽
    const mixed = [makeSample(ACTIVE, STILL), makeSample(STILL, ACTIVE)];
    const { samples } = normalizeSamplesToRight(mixed, ranges);
    for (const s of samples) {
      const l = handEnergy(s.leftSensor, s.leftImu, s.frameCount, "LH", [
        120, 120, 120, 120, 120,
      ])!;
      const r = handEnergy(s.rightSensor, s.rightImu, s.frameCount, "RH", [
        120, 120, 120, 120, 120,
      ])!;
      expect(r.total).toBeGreaterThan(l.total * DOMINANCE_RATIO);
    }
  });
});

describe("summarizeHandedness", () => {
  const ranges = { LH: range(120), RH: range(120) };

  it("与 normalizeSamplesToRight 的统计完全一致（同一套判定）", () => {
    const batch = [
      makeSample(ACTIVE, STILL),
      makeSample(ACTIVE, STILL),
      makeSample(STILL, ACTIVE),
      makeSample(STILL, STILL),
      makeSample(null, null),
    ];
    expect(summarizeHandedness(batch, ranges)).toEqual(
      normalizeSamplesToRight(batch, ranges).stats
    );
  });

  it("空数据集不崩", () => {
    expect(summarizeHandedness([], ranges).total).toBe(0);
  });
});

describe("DominanceTracker：单手与预热", () => {
  it("窗口没攒满时按连了哪只手套判（退回改造前的老行为）", () => {
    const t = new DominanceTracker();
    expect(t.update(null, {}, { left: true, right: false })).toMatchObject({
      dominant: "left",
      reason: "only_left",
    });
    expect(t.update(null, {}, { left: false, right: true })).toMatchObject({
      dominant: "right",
      reason: "only_right",
    });
  });

  it("双手都连着但窗口没满 → 不猜，reason=warmup", () => {
    const t = new DominanceTracker();
    const v = t.update(null, {}, BOTH_ON);
    expect(v.reason).toBe("warmup");
    expect(v.dominant).toBe("right"); // 没有历史判定时给训练口径
  });

  it("一只手套都没连 → no_hand", () => {
    const t = new DominanceTracker();
    expect(t.update(null, {}, { left: false, right: false }).reason).toBe("no_hand");
  });

  it("快照里只有一只手 → 直接判那只手，不看能量", () => {
    const t = new DominanceTracker();
    const v = t.update(makeSample(STILL, null), {}, BOTH_ON);
    expect(v.dominant).toBe("left");
    expect(v.reason).toBe("only_left");
    expect(v.right).toBeNull();
  });
});

describe("DominanceTracker：能量判定", () => {
  const ranges = { LH: range(120), RH: range(120) };

  it("左手做动作、右手戴着不动 → 判左手（原代码在这里放弃判定）", () => {
    const t = new DominanceTracker();
    const v = t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON);
    expect(v.dominant).toBe("left");
    expect(v.reason).toBe("energy");
    expect(v.calibrated).toBe(true);
  });

  it("右手做动作、左手不动 → 判右手", () => {
    const t = new DominanceTracker();
    const v = t.update(makeSample(STILL, ACTIVE), ranges, BOTH_ON);
    expect(v.dominant).toBe("right");
    expect(v.reason).toBe("energy");
  });

  it("两只手都在动 → both（双手词），不镜像", () => {
    const t = new DominanceTracker();
    const v = t.update(makeSample(ACTIVE, ACTIVE), ranges, BOTH_ON);
    expect(v.dominant).toBe("both");
    expect(v.reason).toBe("both_active");
  });

  it("量程悬殊但扫动比例相同 → 仍判成双手词，不偏向量程大的手", () => {
    const t = new DominanceTracker();
    const v = t.update(
      makeSample({ ...ACTIVE, span: 60 }, { ...ACTIVE, span: 176 }),
      { LH: range(60), RH: range(176) },
      BOTH_ON
    );
    expect(v.dominant).toBe("both");
  });

  it("两只手都静止 → 沿用上次判定并报 idle_hold", () => {
    const t = new DominanceTracker();
    t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON); // 先判成左手
    const v = t.update(makeSample(STILL, STILL), ranges, BOTH_ON);
    expect(v.reason).toBe("idle_hold");
    expect(v.dominant).toBe("left"); // 间隙里不重判
  });

  it("有手没标定 → calibrated=false，两只手一起走兜底量程（保持可比）", () => {
    const t = new DominanceTracker();
    const v = t.update(makeSample(ACTIVE, STILL), { LH: range(120), RH: null }, BOTH_ON);
    expect(v.calibrated).toBe(false);
    expect(v.dominant).toBe("left"); // 判定本身仍然成立
  });
});

describe("DominanceTracker：滞回", () => {
  const ranges = { LH: range(120), RH: range(120) };

  it("首个判定立即生效，不等滞回", () => {
    const t = new DominanceTracker();
    expect(t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON).dominant).toBe("left");
  });

  it("相反结论要连续 FLIP_WINDOWS 个窗口才切换", () => {
    const t = new DominanceTracker();
    t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON); // 判成左手
    const flip = makeSample(STILL, ACTIVE);
    for (let i = 1; i < FLIP_WINDOWS; i++) {
      const v = t.update(flip, ranges, BOTH_ON);
      expect(v.dominant).toBe("left"); // 还没切，仍喂旧口径
      expect(v.reason).toBe("pending_flip");
    }
    const v = t.update(flip, ranges, BOTH_ON);
    expect(v.dominant).toBe("right");
    expect(v.reason).toBe("energy");
  });

  it("相反结论断了 → 计数重来，不会攒够就切", () => {
    const t = new DominanceTracker();
    t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON);
    const flip = makeSample(STILL, ACTIVE);
    const back = makeSample(ACTIVE, STILL);
    for (let i = 0; i < FLIP_WINDOWS * 3; i++) {
      // 一次相反、一次相同，交替。永远攒不满连续 FLIP_WINDOWS 次
      expect(t.update(flip, ranges, BOTH_ON).dominant).toBe("left");
      expect(t.update(back, ranges, BOTH_ON).dominant).toBe("left");
    }
  });

  it("静止间隙不清掉攒到一半的翻转（词之间必然有间隙）", () => {
    const t = new DominanceTracker();
    t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON);
    const flip = makeSample(STILL, ACTIVE);
    const idle = makeSample(STILL, STILL);
    for (let i = 1; i < FLIP_WINDOWS; i++) {
      t.update(flip, ranges, BOTH_ON);
      expect(t.update(idle, ranges, BOTH_ON).reason).toBe("idle_hold");
    }
    // 间隙没把计数抹掉，所以最后一次相反结论就该切换
    expect(t.update(flip, ranges, BOTH_ON).dominant).toBe("right");
  });

  it("reset 之后回到「没判过」，下一个结论立即生效", () => {
    const t = new DominanceTracker();
    t.update(makeSample(ACTIVE, STILL), ranges, BOTH_ON);
    expect(t.current()).toBe("left");
    t.reset();
    expect(t.current()).toBe("right"); // 兜底 = 训练口径
    expect(t.update(makeSample(STILL, ACTIVE), ranges, BOTH_ON).dominant).toBe("right");
  });
});
