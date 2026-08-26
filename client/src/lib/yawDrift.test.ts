/*
 * yawDrift.test —— YAW 参考系可行性探针
 *
 * 这个探针的输出会直接决定一件事：**要不要花力气去建 yaw 参考系**。判错的
 * 代价是把两周写进一条死路（或者反过来，把一条走得通的路提前砍掉）。所以
 * 这里锁的是那几条会让结论翻号的判据：
 *
 *  1. 量的必须是绕**世界竖直轴**的转角 —— 手翻个跟头（俯仰/横滚）不算 yaw 漂移，
 *     否则每条录制都会量出巨大的"漂移"，把可行的路判成死路；
 *  2. 手已经倾斜时也要量对 —— 旋转向量在手系里，不转到世界系就会算错；
 *  3. **判静止只能用触觉**。用朝向判静止是循环论证：那等于只挑漂移为 0 的段，
 *     量出来的漂移率必然偏小，把死路判成可行 —— 这是最贵的一种错；
 *  4. 静止段太短不出数（短段量的是抖动除以小分母，不是漂移）；
 *  5. 只读。
 */
import { describe, expect, it } from "vitest";
import {
  yawDeltaDeg,
  longestStillRun,
  driftDegPerSec,
  probeYawReference,
  formatYawProbe,
  MIN_STILL_MS,
  REZERO_INTERVAL_S,
  TOLERABLE_ERROR_DEG,
} from "./yawDrift";
import {
  SEQ_IMU_N,
  SEQ_SENSOR_N,
  type SequenceSample,
} from "./datasetStore";
import {
  quatFromAxisAngle,
  quatMul,
  type Quat,
} from "./sequenceFeatures";
import type { BendRange } from "./bendRange";

const BEND_OFFSET = 60;
const D2R = Math.PI / 180;

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

const zRot = (deg: number): Quat => quatFromAxisAngle([0, 0, 1], deg * D2R);
const xRot = (deg: number): Quat => quatFromAxisAngle([1, 0, 0], deg * D2R);

interface Opts {
  frames?: number;
  durationMs?: number;
  /** 触觉静止的帧区间 [start, end)。区间外是大幅度动作 */
  stillSpan?: [number, number];
  /** 整段的 yaw 漂移率（°/s），叠在四元数上 */
  drift?: number;
  /** 这条录制起始时的 yaw（度）—— 用来造"归零点在跳" */
  yaw0?: number;
  label?: string;
  timestamp?: number;
  origin?: "recorded" | "synthesized";
}

function makeSample(o: Opts = {}): SequenceSample {
  const T = o.frames ?? 100;
  const dur = o.durationMs ?? 2000;
  const [ss, se] = o.stillSpan ?? [0, T];
  const rnd = noise(11);

  const sensor = new Uint8Array(T * SEQ_SENSOR_N);
  const imu = new Float32Array(T * SEQ_IMU_N);
  const q0 = zRot(o.yaw0 ?? 0);

  for (let t = 0; t < T; t++) {
    const still = t >= ss && t < se;
    // 动作段用大斜率线性扫，保证任何子窗里的 σ 都远超静止门限；
    // 静止段只有 ADC 噪声（幅度 2，σ≈0.58 —— 这是"戴着不动"，不是掉线）
    const ramp = still ? 0 : (120 * t) / Math.max(1, T - 1);
    for (let j = 0; j < 5; j++) {
      const v = 40 + ramp + rnd() * 2;
      sensor[t * SEQ_SENSOR_N + BEND_OFFSET + j] = Math.max(
        0,
        Math.min(255, Math.round(v))
      );
    }
    for (let j = 0; j < 60; j++) {
      const v = 20 + ramp * 0.4 + rnd() * 2;
      sensor[t * SEQ_SENSOR_N + j] = Math.max(0, Math.min(255, Math.round(v)));
    }

    const sec = (dur * t) / Math.max(1, T - 1) / 1000;
    // 漂移是**世界系**里绕竖直轴的转动，所以左乘
    const q = quatMul(zRot((o.drift ?? 0) * sec), q0);
    const io = t * SEQ_IMU_N;
    imu[io] = q[0];
    imu[io + 1] = q[1];
    imu[io + 2] = q[2];
    imu[io + 3] = q[3];
  }

  const timestamps = new Float32Array(T);
  for (let t = 0; t < T; t++) timestamps[t] = (dur * t) / Math.max(1, T - 1);

  return {
    segments: [{ label: o.label ?? "i", startFrame: 0, endFrame: T }],
    primaryLabel: o.label ?? "i",
    frameCount: T,
    timestamps,
    leftSensor: null,
    rightSensor: sensor,
    leftImu: null,
    rightImu: imu,
    leftLandmarks: null,
    rightLandmarks: null,
    durationMs: dur,
    sourceFps: 50,
    origin: o.origin ?? "recorded",
    timestamp: o.timestamp ?? new Date("2026-08-17T10:00:00Z").getTime(),
  };
}

describe("yawDeltaDeg：量的是绕世界竖直轴的转角", () => {
  it("同一个姿态 → 0", () => {
    expect(yawDeltaDeg(zRot(0), zRot(0))).toBeCloseTo(0, 6);
  });

  it("纯 yaw 转 30° → 30°", () => {
    expect(yawDeltaDeg(zRot(0), zRot(30))).toBeCloseTo(30, 4);
  });

  it("纯俯仰翻 90° → **0°**（手翻个跟头不是 yaw 漂移）", () => {
    // 这一条是判据的分界线：若这里返回 90，每条录制都会量出巨大的假漂移，
    // 一条本来走得通的路会被直接判死
    expect(yawDeltaDeg(zRot(0), xRot(90))).toBeCloseTo(0, 4);
  });

  it("手已经倾斜 90° 时，再绕世界竖直轴转 30° 仍然是 30°", () => {
    // 旋转向量表达在手系里，忘了转到世界系的话这里会算成别的数
    const tilted = xRot(90);
    const after = quatMul(zRot(30), tilted); // 世界系左乘
    expect(yawDeltaDeg(tilted, after)).toBeCloseTo(30, 4);
  });

  it("转 350° 报 10°（取最短弧，否则漂移率会虚高 35 倍）", () => {
    expect(yawDeltaDeg(zRot(0), zRot(350))).toBeCloseTo(10, 4);
  });

  it("正反方向同幅度给同一个数（返回绝对值）", () => {
    expect(yawDeltaDeg(zRot(0), zRot(-25))).toBeCloseTo(
      yawDeltaDeg(zRot(0), zRot(25)),
      4
    );
  });
});

describe("longestStillRun：只用触觉判静止", () => {
  it("手一直静止但朝向在快速漂 → 仍然判成静止段", () => {
    // 最重要的一条。若判据里掺了朝向能量，20°/s 的漂移会把整段判成"在动"，
    // 探针就只会挑出漂移≈0 的段，把死路量成可行
    const run = longestStillRun(
      makeSample({ drift: 20, stillSpan: [0, 100] }),
      RANGES,
      "right"
    );
    expect(run).not.toBeNull();
    expect(run!.ms).toBeGreaterThan(1800);
  });

  it("挑出的是静止那一段，不是动作那一段", () => {
    // 前 60 帧静止（0~1200ms），后 40 帧大幅动作
    const run = longestStillRun(
      makeSample({ stillSpan: [0, 60] }),
      RANGES,
      "right"
    );
    expect(run).not.toBeNull();
    expect(run!.start).toBe(0);
    expect(run!.ms).toBeGreaterThan(900);
    expect(run!.ms).toBeLessThan(1400);
  });

  it("末尾停下来的那段一样能用（不是只看开头）", () => {
    const run = longestStillRun(
      makeSample({ stillSpan: [40, 100] }),
      RANGES,
      "right"
    );
    expect(run).not.toBeNull();
    expect(run!.start).toBeGreaterThanOrEqual(38);
  });

  it("整段都在动 → 没有静止段", () => {
    expect(longestStillRun(makeSample({ stillSpan: [0, 0] }), RANGES, "right")).toBeNull();
  });

  it("那只手没数据 → null（没戴 ≠ 静止）", () => {
    expect(longestStillRun(makeSample(), RANGES, "left")).toBeNull();
  });
});

describe("driftDegPerSec：量出来的就是造进去的那个数", () => {
  it("2°/s 的漂移量回 2°/s", () => {
    const d = driftDegPerSec(makeSample({ drift: 2 }), RANGES, "right");
    expect(d).not.toBeNull();
    expect(d!).toBeCloseTo(2, 1);
  });

  it("15°/s 的漂移量回 15°/s（大漂移不能被最短弧折回去）", () => {
    const d = driftDegPerSec(makeSample({ drift: 15 }), RANGES, "right");
    expect(d!).toBeCloseTo(15, 1);
  });

  it("完全不漂 → 0", () => {
    expect(driftDegPerSec(makeSample({ drift: 0 }), RANGES, "right")!).toBeLessThan(0.01);
  });

  it("静止段短于门限 → 不出数（短段量的是抖动除以小分母）", () => {
    // 2000ms / 100 帧，只留 30 帧静止 = 600ms < MIN_STILL_MS
    expect(MIN_STILL_MS).toBeGreaterThan(600);
    expect(
      driftDegPerSec(makeSample({ stillSpan: [0, 30], drift: 2 }), RANGES, "right")
    ).toBeNull();
  });
});

describe("probeYawReference：结论", () => {
  const many = (n: number, o: Opts) =>
    Array.from({ length: n }, (_, i) =>
      makeSample({
        ...o,
        // 同一会话内每分钟一条
        timestamp: new Date("2026-08-17T10:00:00Z").getTime() + i * 60_000,
      })
    );

  it("漂得很慢、归零点也稳 → 可行", () => {
    const p = probeYawReference(many(12, { drift: 0.5 }), RANGES);
    expect(p.driftSamples).toBe(12);
    expect(p.verdict).toBe("viable");
    expect(p.projectedErrorDeg).toBeLessThan(TOLERABLE_ERROR_DEG);
  });

  it("漂得比类间距还快 → 不可行（这时候就该去做合并方案）", () => {
    const p = probeYawReference(many(12, { drift: 10 }), RANGES);
    expect(p.verdict).toBe("dead");
    expect(p.projectedErrorDeg).toBeGreaterThan(10 * REZERO_INTERVAL_S * 0.9);
  });

  it("漂移合格但归零点在跳 → 不判可行，且点名这一项", () => {
    // 每条录制的起始 yaw 差 40°，模拟"每次把手放回准备位的姿势不一样"
    const samples = Array.from({ length: 12 }, (_, i) =>
      makeSample({
        drift: 0.5,
        yaw0: i * 40,
        timestamp: new Date("2026-08-17T10:00:00Z").getTime() + i * 60_000,
      })
    );
    const p = probeYawReference(samples, RANGES);
    expect(p.p90RestShiftDeg).toBeGreaterThan(TOLERABLE_ERROR_DEG);
    expect(p.verdict).not.toBe("viable");
    expect(p.notes.join("\n")).toContain("归零点");
  });

  it("样本太少 → 明说判不了，而不是给个看起来很确定的结论", () => {
    const p = probeYawReference(many(3, { drift: 0.5 }), RANGES);
    expect(p.verdict).toBe("unknown");
    expect(p.notes.join("\n")).toContain("不作数");
  });

  it("跨会话不比朝向（陀螺可能重启过，比了是无意义的大数）", () => {
    const base = new Date("2026-08-17T10:00:00Z").getTime();
    const p = probeYawReference(
      [
        makeSample({ drift: 0.5, yaw0: 0, timestamp: base }),
        // 隔了一天 + 起始朝向差 90°：不该产生一条 90° 的归零点漂移
        makeSample({ drift: 0.5, yaw0: 90, timestamp: base + 24 * 3600_000 }),
      ],
      RANGES
    );
    expect(p.sessions).toBe(2);
    expect(p.restShiftSamples).toBe(0);
  });

  it("合成样本不参与漂移统计（它们的 IMU 是单帧复制的，漂移恒为 0）", () => {
    const p = probeYawReference(many(12, { drift: 2, origin: "synthesized" }), RANGES);
    expect(p.driftSamples).toBe(0);
  });

  it("报出六个代词的存量；一个都没有时点名说没法验证", () => {
    const p = probeYawReference(many(12, { drift: 0.5, label: "hello" }), RANGES);
    expect(p.pronounCounts.i).toBe(0);
    expect(p.notes.join("\n")).toContain("没有");
  });

  it("有代词录制时如实计数", () => {
    const p = probeYawReference(
      [...many(4, { label: "i" }), ...many(4, { label: "you" })],
      RANGES
    );
    expect(p.pronounCounts.i).toBe(4);
    expect(p.pronounCounts.you).toBe(4);
  });

  it("空数据集不崩", () => {
    const p = probeYawReference([], RANGES);
    expect(p.verdict).toBe("unknown");
    expect(() => formatYawProbe(p)).not.toThrow();
  });

  it("只读 —— 一个字节都不许改", () => {
    const s = makeSample({ drift: 2 });
    const before = {
      sensor: [...s.rightSensor!],
      imu: [...s.rightImu!],
      dur: s.durationMs,
    };
    probeYawReference([s], RANGES);
    expect([...s.rightSensor!]).toEqual(before.sensor);
    expect([...s.rightImu!]).toEqual(before.imu);
    expect(s.durationMs).toBe(before.dur);
  });
});

describe("formatYawProbe", () => {
  it("把结论、两个数、和可容忍上限都写进正文（复制给人看的东西不能漏行）", () => {
    const t = formatYawProbe(
      probeYawReference(
        Array.from({ length: 12 }, (_, i) =>
          makeSample({
            drift: 0.5,
            timestamp: new Date("2026-08-17T10:00:00Z").getTime() + i * 60_000,
          })
        ),
        RANGES
      )
    );
    expect(t).toContain("YAW 参考系可行性");
    expect(t).toContain("可行");
    expect(t).toContain("°/s");
    expect(t).toContain("归零点漂移");
    expect(t).toContain(String(TOLERABLE_ERROR_DEG));
    expect(t).toContain("代词存量");
  });

  it("判可行时把「先补固定零位姿势」写进报告 —— 这一步最容易被跳过", () => {
    const t = formatYawProbe(
      probeYawReference(
        Array.from({ length: 12 }, (_, i) =>
          makeSample({
            drift: 0.5,
            timestamp: new Date("2026-08-17T10:00:00Z").getTime() + i * 60_000,
          })
        ),
        RANGES
      )
    );
    expect(t).toContain("固定零位姿势");
    expect(t).toContain("calibrateQuaternion");
    // 并且要明说不需要轴向映射 —— 绕世界竖直轴的转角与 IMU 装配朝向无关，
    // 照着"先做轴向映射"去建会白花一轮
    expect(t).toContain("不需要轴向映射");
  });

  it("判不可行时不写下一步（那时候该走合并方案，别指错路）", () => {
    const t = formatYawProbe(
      probeYawReference(
        Array.from({ length: 12 }, (_, i) =>
          makeSample({
            drift: 10,
            timestamp: new Date("2026-08-17T10:00:00Z").getTime() + i * 60_000,
          })
        ),
        RANGES
      )
    );
    expect(t).not.toContain("固定零位姿势");
  });
});
