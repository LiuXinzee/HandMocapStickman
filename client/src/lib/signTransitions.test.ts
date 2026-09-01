import { describe, it, expect } from "vitest";
import {
  DEFAULT_TRANSITION_CONFIG,
  detectSignTransitions,
  landmarkSpeed,
  summarizeTransitions,
  transitionVerdict,
} from "./signTransitions";
import {
  SEQ_SENSOR_N,
  SEQ_IMU_N,
  SEQ_LANDMARK_N,
  type SequenceSample,
} from "./datasetStore";

// ===== 夹具 =====

/**
 * 一只手 = 21 个点的刚体，整只手按 `offset(t)` 平移。
 *
 * 关键点 0（手腕）在原点、关键点 9（中指根）在 (0, 0.1) —— 于是 `handScale` 恒为
 * **0.1**，速度的单位换算是确定的：位移 d（归一化坐标）→ d/0.1 手宽 → ×1000/dt 手宽/秒。
 * 判据里所有门限都是相对分位数的，绝对值不影响分段，但把它钉死能让断言算得出来。
 */
function makeHand(
  T: number,
  offset: (t: number) => { x: number; y: number },
  visible: (t: number) => boolean
): Float32Array {
  const lm = new Float32Array(T * SEQ_LANDMARK_N);
  for (let t = 0; t < T; t++) {
    const o = t * SEQ_LANDMARK_N;
    if (!visible(t)) {
      lm.fill(NaN, o, o + SEQ_LANDMARK_N);
      continue;
    }
    const { x, y } = offset(t);
    for (let i = 0; i < 21; i++) {
      // 除 0 和 9 之外的点摊在一条线上，位置固定 —— 它们只贡献"整只手平移了多少"
      lm[o + i * 3] = x + (i === 9 ? 0 : i * 0.002);
      lm[o + i * 3 + 1] = y + (i === 9 ? 0.1 : 0);
      lm[o + i * 3 + 2] = 0;
    }
  }
  return lm;
}

const ALWAYS = () => true;

function makeSample(
  T: number,
  offset: (t: number) => { x: number; y: number },
  opts: { fps?: number; visible?: (t: number) => boolean; words?: number } = {}
): SequenceSample {
  const fps = opts.fps ?? 50;
  const dt = 1000 / fps;
  const timestamps = new Float32Array(T);
  for (let t = 0; t < T; t++) timestamps[t] = t * dt;
  const imu = new Float32Array(T * SEQ_IMU_N);
  for (let t = 0; t < T; t++) imu[t * SEQ_IMU_N] = 1;
  const nWords = opts.words ?? 1;
  return {
    segments: Array.from({ length: nWords }, (_, i) => ({
      label: `w${i}`,
      startFrame: 0,
      endFrame: T,
    })),
    primaryLabel: "w0",
    frameCount: T,
    timestamps,
    leftSensor: new Uint8Array(T * SEQ_SENSOR_N),
    rightSensor: null,
    leftImu: imu,
    rightImu: null,
    leftLandmarks: makeHand(T, offset, opts.visible ?? ALWAYS),
    rightLandmarks: null,
    durationMs: T * dt,
    sourceFps: fps,
    origin: "recorded",
    timestamp: 0,
  };
}

/**
 * 「保持 300ms → 移动 200ms → 保持 300ms → 移动 200ms → 保持 300ms」，50Hz 共 65 帧。
 * 这就是一条打得干净的三词句子应有的形状：三个词各稳住一会儿，中间两段过渡。
 */
const HOLD_A = 15; // 帧 0..14
const MOVE_A = 10; // 帧 15..24
const HOLD_B = 15; // 帧 25..39
const MOVE_B = 10; // 帧 40..49
const HOLD_C = 15; // 帧 50..64
const THREE_WORD_T = HOLD_A + MOVE_A + HOLD_B + MOVE_B + HOLD_C;

/** 每个"移动"帧平移 0.02（= 0.2 手宽 → 50Hz 下 10 手宽/秒），保持帧平移 0 */
function threeWordOffset(t: number): { x: number; y: number } {
  let x = 0;
  for (let k = 1; k <= t; k++) {
    const movingIntoK =
      (k >= HOLD_A && k < HOLD_A + MOVE_A) ||
      (k >= HOLD_A + MOVE_A + HOLD_B && k < HOLD_A + MOVE_A + HOLD_B + MOVE_B);
    if (movingIntoK) x += 0.02;
  }
  return { x, y: 0 };
}

// ===== landmarkSpeed =====

describe("landmarkSpeed", () => {
  it("速度按手宽归一化（手离摄像头远近不影响读数）", () => {
    const big = makeSample(10, (t) => ({ x: t * 0.02, y: 0 }));
    // 整只手连轨迹缩到一半 = 人坐远一倍：手宽 0.1→0.05、位移 0.02→0.01，比值不变。
    // 不除手宽的话这条会读成"坐得越近打得越快"
    const small: SequenceSample = {
      ...big,
      leftLandmarks: new Float32Array(big.leftLandmarks!).map((v) => v / 2),
    };
    // 0.02 位移 / 0.1 手宽 = 0.2 手宽/帧，50Hz → 10 手宽/秒
    expect(landmarkSpeed(big, 1).speed[5]).toBeCloseTo(10, 4);
    expect(landmarkSpeed(small, 1).speed[5]).toBeCloseTo(10, 4);
  });

  it("看不见手的帧是 NaN，不是 0（0 会被读成「没动」）", () => {
    const s = makeSample(20, (t) => ({ x: t * 0.02, y: 0 }), {
      visible: (t) => t < 10,
    });
    const { speed, trackedFrames } = landmarkSpeed(s, 1);
    expect(trackedFrames).toBe(10);
    expect(speed[5]).toBeCloseTo(10, 4);
    expect(Number.isNaN(speed[15])).toBe(true);
    // 空洞的第一帧也必须是 NaN：前一帧看得见、这一帧看不见，算不出位移
    expect(Number.isNaN(speed[10])).toBe(true);
  });

  it("第 0 帧是 0 而不是 NaN（有手，只是没有前一帧）", () => {
    const s = makeSample(10, (t) => ({ x: t * 0.02, y: 0 }));
    expect(landmarkSpeed(s, 1).speed[0]).toBe(0);
  });

  it("平滑不跨过 NaN 空洞（否则空洞两侧会互相污染）", () => {
    const s = makeSample(20, (t) => ({ x: t * 0.02, y: 0 }), {
      visible: (t) => t < 8 || t > 12,
    });
    const { speed } = landmarkSpeed(s, 3);
    for (let t = 8; t <= 13; t++) expect(Number.isNaN(speed[t])).toBe(true);
    expect(speed[5]).toBeCloseTo(10, 4);
  });
});

// ===== detectSignTransitions =====

describe("detectSignTransitions", () => {
  it("三词句子切出 3 个保持段 + 2 个过渡段，过渡时长量对", () => {
    const s = makeSample(THREE_WORD_T, threeWordOffset, { words: 3 });
    const tr = detectSignTransitions(s);
    expect(tr.reason).toBe("ok");
    expect(tr.holds.length).toBe(3);
    expect(tr.moves.length).toBe(2);
    expect(tr.medianMoveMs).toBeCloseTo(200, 6);
    expect(tr.holds[0].durationMs).toBeCloseTo(300, 6);
    expect(tr.moves[0].startFrame).toBe(HOLD_A);
    expect(tr.moves[0].endFrame).toBe(HOLD_A + MOVE_A);
    expect(tr.trackedFrames).toBe(THREE_WORD_T);
  });

  it("段落覆盖整条录制、不重叠", () => {
    const s = makeSample(THREE_WORD_T, threeWordOffset, { words: 3 });
    const tr = detectSignTransitions(s);
    const all = [...tr.holds, ...tr.moves].sort((a, b) => a.startFrame - b.startFrame);
    expect(all[0].startFrame).toBe(0);
    expect(all[all.length - 1].endFrame).toBe(THREE_WORD_T);
    for (let i = 1; i < all.length; i++) {
      expect(all[i].startFrame).toBe(all[i - 1].endFrame);
    }
  });

  it("没有关键点数组时判 no_vision（不是编一个分段出来）", () => {
    const s = makeSample(THREE_WORD_T, threeWordOffset, { words: 3 });
    const tr = detectSignTransitions({
      ...s,
      leftLandmarks: null,
      rightLandmarks: null,
    });
    expect(tr.reason).toBe("no_vision");
    expect(tr.holds).toEqual([]);
    expect(tr.moves).toEqual([]);
    expect(tr.medianMoveMs).toBe(0);
  });

  it("一路匀速划过去（没有保持段）判 no_contrast", () => {
    // 这种条上取门限只是在切噪声：p90 ≈ p10，任何阈值都会给出随机分段
    const s = makeSample(40, (t) => ({ x: t * 0.02, y: 0 }), { words: 3 });
    expect(detectSignTransitions(s).reason).toBe("no_contrast");
  });

  it("整条没动也判 no_contrast（不是「一个 3 秒的保持段」）", () => {
    const s = makeSample(40, () => ({ x: 0.3, y: 0.3 }), { words: 3 });
    expect(detectSignTransitions(s).reason).toBe("no_contrast");
  });

  it("几乎没检出手时判 no_track", () => {
    const s = makeSample(40, threeWordOffset, { visible: (t) => t < 2 });
    expect(detectSignTransitions(s).reason).toBe("no_track");
  });

  it("视觉空洞不被跨过来连成一段", () => {
    // 中间 10 帧丢手：不知道那时候手在干什么，猜一个就是编数据
    const s = makeSample(THREE_WORD_T, threeWordOffset, {
      words: 3,
      visible: (t) => t < 28 || t > 37,
    });
    const tr = detectSignTransitions(s);
    expect(tr.reason).toBe("ok");
    // 没有任何段跨过空洞 [28,38]
    for (const seg of [...tr.holds, ...tr.moves]) {
      expect(seg.endFrame <= 28 || seg.startFrame >= 39).toBe(true);
    }
    expect(tr.trackedFrames).toBe(THREE_WORD_T - 10);
  });

  it("保持段里的短暂抖动被并掉，不报成一个 40ms 的「过渡」", () => {
    /*
     * 「保持 400ms（其中 t=8,9 两帧关键点抖动）→ 过渡 200ms → 保持」。
     * 抖动那两帧的速度与真过渡同量级（关键点在真实录制里就是这样跳的），
     * 光靠阈值分不掉，必须靠 minMoveMs 的合并 —— 否则过渡时长的中位数会被
     * 一堆 40ms 的假过渡压下去，而那正是要拿去定 overlap_ms 的数。
     */
    const T = 60;
    const step = (k: number) =>
      k === 8 || k === 9 || (k >= 20 && k < 30) ? 0.02 : 0;
    const offset = (t: number) => {
      let x = 0;
      for (let k = 1; k <= t; k++) x += step(k);
      return { x, y: 0 };
    };
    const tr = detectSignTransitions(makeSample(T, offset, { words: 2 }));
    expect(tr.reason).toBe("ok");
    expect(tr.moves.length).toBe(1);
    expect(tr.moves[0].durationMs).toBeCloseTo(200, 6);
    expect(tr.holds.length).toBe(2);
    for (const m of tr.moves) {
      expect(m.durationMs).toBeGreaterThanOrEqual(DEFAULT_TRANSITION_CONFIG.minMoveMs);
    }
  });
});

// ===== transitionVerdict =====

describe("transitionVerdict", () => {
  const three = () =>
    detectSignTransitions(makeSample(THREE_WORD_T, threeWordOffset, { words: 3 }));

  it("保持段数与词数一致时 match，并报出过渡时长", () => {
    const v = transitionVerdict(three(), 3);
    expect(v.kind).toBe("match");
    expect(v.holdCount).toBe(3);
    expect(Math.round(v.medianMoveMs)).toBe(200);
    expect(v.note).toContain("200ms");
  });

  it("差 1 段仍算 match（句尾最后一个词后面没有移动）", () => {
    expect(transitionVerdict(three(), 4).kind).toBe("match");
    expect(transitionVerdict(three(), 2).kind).toBe("match");
  });

  it("保持段明显少于词数 → fewer（词粘在一起，可能漏词）", () => {
    const v = transitionVerdict(three(), 6);
    expect(v.kind).toBe("fewer");
    expect(v.note).toContain("漏词");
  });

  it("保持段明显多于词数 → more（中间顿了）", () => {
    const v = transitionVerdict(three(), 1);
    expect(v.kind).toBe("more");
  });

  it("量不出来时 unmeasured，并说清是哪一种量不出来", () => {
    const s = makeSample(THREE_WORD_T, threeWordOffset, { words: 3 });
    const noVision = detectSignTransitions({
      ...s,
      leftLandmarks: null,
      rightLandmarks: null,
    });
    const v = transitionVerdict(noVision, 3);
    expect(v.kind).toBe("unmeasured");
    expect(v.note).toContain("没有视觉");
  });
});

// ===== summarizeTransitions =====

describe("summarizeTransitions", () => {
  it("只统计 match 的条，并把 usable/total 都报出来", () => {
    const good = detectSignTransitions(
      makeSample(THREE_WORD_T, threeWordOffset, { words: 3 })
    );
    const noVision = detectSignTransitions(
      makeSample(THREE_WORD_T, threeWordOffset, { words: 3, visible: () => false })
    );
    const sum = summarizeTransitions([
      { transitions: good, wordCount: 3 },
      { transitions: good, wordCount: 3 },
      { transitions: noVision, wordCount: 3 }, // 没视觉 → 不参与
      { transitions: good, wordCount: 9 }, // 分段与词数对不上 → 不参与
    ]);
    expect(sum.total).toBe(4);
    expect(sum.usable).toBe(2);
    expect(sum.medianMoveMs).toBeCloseTo(200, 6);
    expect(sum.p10MoveMs).toBeCloseTo(200, 6);
    expect(sum.p90MoveMs).toBeCloseTo(200, 6);
  });

  it("一条都不可用时返回 0 而不是 NaN", () => {
    const sum = summarizeTransitions([]);
    expect(sum.total).toBe(0);
    expect(sum.usable).toBe(0);
    expect(sum.medianMoveMs).toBe(0);
  });
});
