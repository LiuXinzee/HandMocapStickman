/*
 * datasetAudit.test —— 数据体检
 *
 * 这份报告的用途是**决定要删/要改哪些数据**，所以它算错的代价不是显示难看，
 * 而是照着它去删对的数据。这里锁的就是那几条判据：
 *  1. 只读 —— 一个字节都不许改（下游会拿着同一批引用去训练）；
 *  2. 按批次分组不能把两次采集混成一个平均值（这是做这份报告的全部理由）；
 *  3. "裁剪口径分裂"必须在两批裁剪结果不同时报出来，且**只**在那时报；
 *  4. 冻结判据不能把"戴着不动的手"误判成掉线（闲手有噪声、掉线是精确常数）；
 *  5. 时长离群按**本词**中位数算，不能拿全局中位数去卡长词/短词。
 */
import { describe, expect, it } from "vitest";
import {
  auditDataset,
  formatAuditReport,
  DURATION_OUTLIER_RATIO,
} from "./datasetAudit";
import {
  SEQ_IMU_N,
  SEQ_SENSOR_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "./datasetStore";
import type { BendRange } from "./bendRange";

const BEND_OFFSET = 60;

const range = (span: number): BendRange => ({
  open: [40, 40, 40, 40, 40],
  fist: [40 + span, 40 + span, 40 + span, 40 + span, 40 + span],
});
const RANGES = { LH: range(120), RH: range(120) };
/**
 * 没做弯折标定。`sequenceTrim` 第三层（触觉静止段）只在量程齐全时才跑，
 * 所以这就是"只有视觉判据"的那个旧世界 —— 无视觉样本会真的落到 `no_vision`。
 *
 * 「裁剪口径分裂」那几条必须用它：给了标定的话无视觉样本也会被判过一遍，
 * 分裂本身就不成立了（那正是补第三层的目的）。
 */
const NO_CAL = {};

/** 确定性伪随机，免得测试变成偶发失败 */
function noise(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000 - 0.5;
  };
}

interface Opts {
  label?: string;
  /** 两只手都不做动作（空录 / `_idle` 伪类）。弯折、指压、朝向三路一起静下来 */
  still?: boolean;
  /** 开头有多少比例的帧是静止的（预备动作/还没起手），动作压缩到剩下那段里 */
  headStill?: number;
  /** 弯折在整段里扫过的量程比例 */
  sweep?: number;
  /** 0 = 整段恒定常数（掉线/冻结），>0 = 有 ADC 噪声（戴着不动） */
  noiseAmp?: number;
  frames?: number;
  durationMs?: number;
  /** 有无摄像头关键点 —— 直接决定裁剪走 applied 还是 no_vision */
  vision?: boolean;
  /** 只给这一只手数据 */
  hands?: "left" | "right" | "both";
  day?: string;
  origin?: "recorded" | "synthesized";
}

function makeSample(o: Opts = {}): SequenceSample {
  const T = o.frames ?? 40;
  const sweep = o.sweep ?? 0.6;
  const amp = o.noiseAmp ?? 3;
  const hands = o.hands ?? "both";
  const rnd = noise(7);

  // 开头 headStill 比例的帧完全静止，动作被压进剩下那段（phase 在头段恒为 0）
  const head = Math.round((o.headStill ?? 0) * T);

  const hand = (active: boolean) => {
    const sensor = new Uint8Array(T * SEQ_SENSOR_N);
    const imu = new Float32Array(T * SEQ_IMU_N);
    for (let t = 0; t < T; t++) {
      const phase =
        t < head
          ? 0
          : Math.sin((Math.PI * (t - head)) / Math.max(1, T - 1 - head));
      for (let j = 0; j < 5; j++) {
        const v = 40 + (active ? sweep * 120 * phase : 0) + rnd() * amp;
        sensor[t * SEQ_SENSOR_N + BEND_OFFSET + j] = Math.max(
          0,
          Math.min(255, Math.round(v))
        );
      }
      for (let j = 0; j < 60; j++) {
        const v = 20 + (active ? 40 * phase : 0) + rnd() * amp;
        sensor[t * SEQ_SENSOR_N + j] = Math.max(0, Math.min(255, Math.round(v)));
      }
      const half = active ? ((60 * phase) / 2 / 180) * Math.PI : 0;
      const io = t * SEQ_IMU_N;
      imu[io] = Math.cos(half);
      imu[io + 1] = Math.sin(half);
    }
    return { sensor, imu };
  };

  // 主手固定放右手，让"主手分布"这一列有个确定答案
  const still = o.still ?? false;
  const r = hands === "left" ? null : hand(!still);
  const l = hands === "right" ? null : hand(!still && hands === "left");

  const marks = () => {
    const a = new Float32Array(T * SEQ_LANDMARK_N);
    for (let i = 0; i < a.length; i++) a[i] = ((i * 3) % 100) / 100;
    return a;
  };
  const timestamps = new Float32Array(T);
  const dur = o.durationMs ?? 2000;
  for (let t = 0; t < T; t++) timestamps[t] = (dur * t) / Math.max(1, T - 1);

  return {
    segments: [{ label: o.label ?? "hello", startFrame: 0, endFrame: T }],
    primaryLabel: o.label ?? "hello",
    frameCount: T,
    timestamps,
    leftSensor: l?.sensor ?? null,
    rightSensor: r?.sensor ?? null,
    leftImu: l?.imu ?? null,
    rightImu: r?.imu ?? null,
    leftLandmarks: o.vision && l ? marks() : null,
    rightLandmarks: o.vision && r ? marks() : null,
    durationMs: dur,
    sourceFps: 50,
    origin: o.origin ?? "recorded",
    timestamp: new Date(`${o.day ?? "2026-08-17"}T10:00:00Z`).getTime(),
  };
}

describe("auditDataset：只读", () => {
  it("不改任何一个字节（下游拿着同一批引用去训练）", () => {
    const s = makeSample();
    const before = {
      left: [...s.leftSensor!],
      right: [...s.rightSensor!],
      imu: [...s.rightImu!],
      dur: s.durationMs,
    };
    auditDataset([s], RANGES);
    expect([...s.leftSensor!]).toEqual(before.left);
    expect([...s.rightSensor!]).toEqual(before.right);
    expect([...s.rightImu!]).toEqual(before.imu);
    expect(s.durationMs).toBe(before.dur);
  });

  it("空数据集不崩", () => {
    const a = auditDataset([], RANGES);
    expect(a.total).toBe(0);
    expect(a.byDay).toEqual([]);
    expect(() => formatAuditReport(a)).not.toThrow();
  });
});

describe("auditDataset：按批次分组", () => {
  it("两天的数据分成两组，各自报自己的中位时长 —— 不是一个被抹平的平均值", () => {
    // 做这份报告的全部理由就在这一条：老批 3600ms、新批 1600ms，
    // 全局平均给出 2600ms，那个数字在两批数据上都不成立
    const a = auditDataset(
      [
        makeSample({ day: "2026-08-10", durationMs: 3600 }),
        makeSample({ day: "2026-08-10", durationMs: 3600 }),
        makeSample({ day: "2026-08-17", durationMs: 1600 }),
        makeSample({ day: "2026-08-17", durationMs: 1600 }),
      ],
      RANGES
    );
    expect(a.byDay).toHaveLength(2);
    expect(a.byDay[0].medianDurationMs).toBe(3600);
    expect(a.byDay[1].medianDurationMs).toBe(1600);
    expect(a.overall.medianDurationMs).toBe(2600); // 全局确实是这个没用的数字
  });

  it("合成样本单独成组 —— 它们的帧间统计与真实录制不是一回事", () => {
    const a = auditDataset(
      [
        makeSample({ day: "2026-08-17" }),
        makeSample({ day: "2026-08-17", origin: "synthesized" }),
      ],
      RANGES
    );
    expect(a.byDay).toHaveLength(2);
    expect(a.recorded).toBe(1);
    expect(a.synthesized).toBe(1);
    expect(a.byDay.some((d) => d.key.includes("合成"))).toBe(true);
  });

  it("时间戳缺失的归到「未知」批次，不算成 1970 年", () => {
    const s = { ...makeSample(), timestamp: 0 };
    expect(auditDataset([s], RANGES).byDay[0].key).toBe("未知");
  });
});

describe("auditDataset：裁剪口径", () => {
  it("没标定时：有视觉的批走 applied/full_span、无视觉的批走 no_vision", () => {
    const a = auditDataset(
      [
        makeSample({ day: "2026-08-10", vision: false }),
        makeSample({ day: "2026-08-17", vision: true }),
      ],
      NO_CAL
    );
    const old = a.byDay.find((d) => d.key.startsWith("2026-08-10"))!;
    const now = a.byDay.find((d) => d.key.startsWith("2026-08-17"))!;
    expect(old.trim.no_vision).toBe(1);
    expect(old.visionRatio).toBe(0);
    expect(now.trim.no_vision).toBe(0);
    expect(now.visionRatio).toBe(1);
  });

  it("**有标定**时无视觉样本也被判过一遍 —— no_vision 归零，口径不再分裂", () => {
    // 这是补第三层要达到的效果本身：同一批数据，唯一的差别是有没有弯折两点标定
    const samples = [
      makeSample({ day: "2026-08-10", vision: false }),
      makeSample({ day: "2026-08-17", vision: true }),
    ];
    expect(auditDataset(samples, NO_CAL).overall.trim.no_vision).toBe(1);
    const a = auditDataset(samples, RANGES);
    expect(a.overall.trim.no_vision).toBe(0);
    expect(a.flags.join("\n")).not.toContain("口径分裂");
  });

  it("无视觉样本的 keptMs = 原时长（没标定 → 一刀没裁，预备动作全在里面）", () => {
    const a = auditDataset([makeSample({ vision: false, durationMs: 3000 })], NO_CAL);
    expect(a.samples[0].trimReason).toBe("no_vision");
    expect(a.samples[0].keptMs).toBeCloseTo(3000, 5);
  });

  it("没标定且全都无视觉时报「裁剪从未生效」而不是「口径分裂」", () => {
    const a = auditDataset(
      [makeSample({ vision: false }), makeSample({ vision: false })],
      NO_CAL
    );
    const joined = a.flags.join("\n");
    expect(joined).toContain("裁剪从未生效");
    expect(joined).not.toContain("口径分裂");
  });

  it("全都有视觉时既不报分裂也不报从未生效", () => {
    const a = auditDataset(
      [makeSample({ vision: true }), makeSample({ vision: true })],
      RANGES
    );
    const joined = a.flags.join("\n");
    expect(joined).not.toContain("口径分裂");
    expect(joined).not.toContain("裁剪从未生效");
  });
});

describe("auditDataset：冻结手", () => {
  it("整段弯折恒定 → 记为冻结（掉线后最近邻重采样把最后一帧铺满了）", () => {
    const a = auditDataset([makeSample({ noiseAmp: 0, still: true })], RANGES);
    expect(a.samples[0].frozenHands.length).toBeGreaterThan(0);
    expect(a.flags.join("\n")).toContain("一动不动");
  });

  it("戴着不动但有 ADC 噪声 → **不算**冻结（这是最容易误杀的一种）", () => {
    // 闲着的手能量低但不为 0；精确的常数才是掉线。判据必须分得开这两者
    const a = auditDataset([makeSample({ still: true, noiseAmp: 4 })], RANGES);
    expect(a.samples[0].frozenHands).toEqual([]);
  });

  it("只有一只手冻住时只报那一只", () => {
    const s = makeSample({ hands: "both" });
    // 把左手整段抹成常数，右手保持活动
    for (let t = 0; t < s.frameCount; t++)
      for (let j = 60; j < 65; j++) s.leftSensor![t * SEQ_SENSOR_N + j] = 55;
    const a = auditDataset([s], RANGES);
    expect(a.samples[0].frozenHands).toEqual(["左"]);
  });

  it("那只手根本没数据 → 不算冻结（没戴 ≠ 掉线）", () => {
    const a = auditDataset([makeSample({ hands: "right" })], RANGES);
    expect(a.samples[0].frozenHands).not.toContain("左");
  });
});

describe("auditDataset：时长离群", () => {
  it("按**本词**中位数判，不拿全局中位数卡长词短词", () => {
    // 短词 800ms × 3、长词 3000ms × 3：两个词各自都很整齐，一条离群都不该有。
    // 若用全局中位数（1900ms），六条会全部被判成离群
    const samples = [
      ...Array.from({ length: 3 }, () => makeSample({ label: "a", durationMs: 800 })),
      ...Array.from({ length: 3 }, () => makeSample({ label: "b", durationMs: 3000 })),
    ];
    const a = auditDataset(samples, RANGES);
    for (const l of a.byLabel) expect(l.durationOutliers).toBe(0);
    expect(a.flags.join("\n")).not.toContain("时长离群");
  });

  it("同一个词里明显偏长的那条被挑出来", () => {
    const samples = [
      makeSample({ label: "a", durationMs: 2000 }),
      makeSample({ label: "a", durationMs: 2000 }),
      makeSample({ label: "a", durationMs: 2000 }),
      makeSample({ label: "a", durationMs: 2000 * DURATION_OUTLIER_RATIO + 500 }),
    ];
    const a = auditDataset(samples, RANGES);
    expect(a.byLabel[0].durationOutliers).toBe(1);
    expect(a.flags.join("\n")).toContain("时长离群");
  });

  it("刚好等于门限倍数不算离群（边界不含）", () => {
    const samples = [
      makeSample({ label: "a", durationMs: 2000 }),
      makeSample({ label: "a", durationMs: 2000 }),
      makeSample({ label: "a", durationMs: 2000 }),
      makeSample({ label: "a", durationMs: 2000 * DURATION_OUTLIER_RATIO }),
    ];
    expect(auditDataset(samples, RANGES).byLabel[0].durationOutliers).toBe(0);
  });
});

describe("auditDataset：静止头段", () => {
  // 这一组回答的是"手放着不动时模型会输出哪个词" —— 答案是头段占比最大的那些词。
  // 现有的视觉裁剪判据答不了这个问题（手已举在画面里时它返回 full_span），
  // 所以这里的判据必须是纯触觉的、且与 trimReason 无关

  it("整段都在做动作 → 头段接近 0", () => {
    const a = auditDataset([makeSample({ frames: 100, durationMs: 2000 })], RANGES);
    expect(a.samples[0].headStillMs).toBeLessThan(300);
  });

  it("开头一半静止 → 头段约占一半（这就是预备动作没被裁掉的那种数据）", () => {
    const a = auditDataset(
      [makeSample({ frames: 100, durationMs: 2000, headStill: 0.5 })],
      RANGES
    );
    const r = a.samples[0].headStillMs / 2000;
    expect(r).toBeGreaterThan(0.3);
    expect(r).toBeLessThan(0.7);
  });

  it("有视觉、判据判成 full_span 的样本一样能量出头段 —— 两者相互独立", () => {
    // full_span = "视觉判据跑过了但没找到可裁的地方"，不等于"开头没有静止段"。
    // 08-17 批 242 条里 167 条是 full_span，这一条锁的就是那 167 条也有数字
    const a = auditDataset(
      [makeSample({ frames: 100, durationMs: 2000, headStill: 0.5, vision: true })],
      RANGES
    );
    expect(a.samples[0].trimReason).not.toBe("no_vision");
    expect(a.samples[0].headStillMs).toBeGreaterThan(400);
  });

  it("整条都没动 → 头段 = 全时长（空录）", () => {
    const a = auditDataset(
      [makeSample({ frames: 100, durationMs: 2000, still: true })],
      RANGES
    );
    expect(a.samples[0].headStillMs).toBe(2000);
  });

  it("按词排行报出来，头段长的词在前、干净的词在后", () => {
    const samples = [
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "dirty", frames: 100, durationMs: 2000, headStill: 0.5 })
      ),
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "clean", frames: 100, durationMs: 2000 })
      ),
    ];
    const a = auditDataset(samples, RANGES);
    const dirty = a.byLabel.find((l) => l.key === "dirty")!;
    const cleanG = a.byLabel.find((l) => l.key === "clean")!;
    expect(dirty.medianHeadStillRatio).toBeGreaterThan(
      cleanG.medianHeadStillRatio + 0.2
    );
    const joined = a.flags.join("\n");
    expect(joined).toContain("静止头段排行");
    // 排行里 dirty 必须排在 clean 前面
    expect(joined.indexOf("dirty")).toBeLessThan(joined.indexOf("clean"));
  });

  it("全都干净时不报这条排行（每个词都 0% 的排行没有信息量）", () => {
    const samples = [
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "a", frames: 100, durationMs: 2000 })
      ),
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "b", frames: 100, durationMs: 2000 })
      ),
    ];
    expect(auditDataset(samples, RANGES).flags.join("\n")).not.toContain(
      "静止头段排行"
    );
  });

  it("`_idle` 不进排行 —— 它本来就该是静止的，排在第一名毫无意义", () => {
    const samples = [
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "_idle", frames: 100, durationMs: 2000, still: true })
      ),
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "dirty", frames: 100, durationMs: 2000, headStill: 0.5 })
      ),
      ...Array.from({ length: 4 }, () =>
        makeSample({ label: "clean", frames: 100, durationMs: 2000 })
      ),
    ];
    const line = auditDataset(samples, RANGES).flags.find((f) =>
      f.includes("静止头段排行")
    )!;
    // 只看排行本身那一段（后面的修法说明里正当地提到了 `_idle`）
    const rank = line.slice(0, line.indexOf("。最干净的是"));
    expect(rank).toContain("dirty");
    expect(rank).not.toContain("_idle");
  });
});

describe("auditDataset：主手与空录", () => {
  it("按词报主手分布 —— 归一化后该全落在右手", () => {
    const a = auditDataset(
      [makeSample({ label: "a" }), makeSample({ label: "a" })],
      RANGES
    );
    expect(a.byLabel[0].dominance.right).toBe(2);
    expect(a.byLabel[0].dominance.left).toBe(0);
  });

  it("左手主导的样本被如实报成 left（说明这条还没归一化）", () => {
    const a = auditDataset([makeSample({ hands: "left" })], RANGES);
    expect(a.samples[0].dominant).toBe("left");
  });

  it("挂着词标签但两手都静止 → 报疑似空录", () => {
    const a = auditDataset([makeSample({ label: "hello", still: true })], RANGES);
    expect(a.samples[0].dominant).toBe("idle");
    expect(a.flags.join("\n")).toContain("疑似空录");
  });

  it("`_idle` 样本静止是应该的，不报空录", () => {
    const a = auditDataset([makeSample({ label: "_idle", still: true })], RANGES);
    expect(a.samples[0].dominant).toBe("idle");
    expect(a.flags.join("\n")).not.toContain("疑似空录");
    expect(a.hasIdle).toBe(true);
  });

  it("没有 `_idle` 时报出来（滑窗推理下必须有空闲类）", () => {
    const a = auditDataset([makeSample()], RANGES);
    expect(a.hasIdle).toBe(false);
    expect(a.flags.join("\n")).toContain("_idle");
  });
});

describe("auditDataset：IMU", () => {
  it("合成样本不参与 IMU 漂移判定（它们的 IMU 是单帧复制的，判不出漂移）", () => {
    const a = auditDataset([makeSample({ origin: "synthesized" })], RANGES);
    expect(a.samples[0].imuSuspect).toBe(false);
  });
});

describe("formatAuditReport", () => {
  const a = auditDataset(
    [
      makeSample({ day: "2026-08-10", label: "a", vision: false, durationMs: 3600 }),
      makeSample({ day: "2026-08-17", label: "a", vision: true, durationMs: 1600 }),
      makeSample({ day: "2026-08-17", label: "b", vision: true, durationMs: 1600 }),
    ],
    // NO_CAL：要让「口径分裂」那条真的出现在正文里，就得让无视觉那批停在 no_vision
    NO_CAL
  );
  const text = formatAuditReport(a);

  it("每个批次和每个词都出现在报告里（复制给人看的东西不能漏行）", () => {
    expect(text).toContain("2026-08-10");
    expect(text).toContain("2026-08-17");
    expect(text).toContain("a");
    expect(text).toContain("b");
  });

  it("带上三段固定小标题，便于粘贴后定位", () => {
    expect(text).toContain("按采集批次");
    expect(text).toContain("按词");
    expect(text).toContain("需要处理的");
  });

  it("裁剪口径分裂这一条出现在报告正文里", () => {
    expect(text).toContain("口径分裂");
  });

  it("没问题的数据集也给一句明确结论，不是空白", () => {
    const clean = auditDataset(
      [
        makeSample({ label: "_idle", still: true, vision: true }),
        ...Array.from({ length: 5 }, () => makeSample({ label: "a", vision: true })),
      ],
      RANGES
    );
    const t = formatAuditReport(clean);
    expect(t).toContain("需要处理的");
    expect(t.length).toBeGreaterThan(100);
  });
});
