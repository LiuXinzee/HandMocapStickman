import { describe, expect, it } from "vitest";
import {
  averageBends,
  bendDegrees,
  bendRatios,
  canonicalBendRaw,
  channelSpans,
  handKeyOf,
  HAND_LEFT,
  HAND_RIGHT,
  isCalibrated,
  MIN_USEFUL_SPAN,
  weakChannels,
  type BendRange,
} from "./bendRange";

// sensorMapping.ts:31 / :67 —— 1-based 表减 1
const LH_BEND_IDX = [222, 219, 216, 213, 210].map((i) => i - 1); // 小指→拇指
const RH_BEND_IDX = [47, 44, 41, 38, 35].map((i) => i - 1); // 拇指→小指

/** 按 canonical 顺序（拇指→小指）把 5 个值塞进 256 字节原始帧 */
function makeRaw(sensorType: number, canonical: number[]): number[] {
  const raw = new Array(256).fill(0);
  if (sensorType === HAND_LEFT) {
    // 左手物理顺序是小指→拇指，所以 canonical 要反着写进去
    LH_BEND_IDX.forEach((idx, physical) => {
      raw[idx] = canonical[4 - physical];
    });
  } else {
    RH_BEND_IDX.forEach((idx, physical) => {
      raw[idx] = canonical[physical];
    });
  }
  return raw;
}

const CANON = [10, 20, 30, 40, 50];

describe("canonicalBendRaw", () => {
  it("右手：物理顺序已是拇指→小指，原样取出", () => {
    expect(canonicalBendRaw(makeRaw(HAND_RIGHT, CANON), HAND_RIGHT)).toEqual(CANON);
  });

  it("左手：物理顺序是小指→拇指，取出后必须反转成拇指→小指", () => {
    expect(canonicalBendRaw(makeRaw(HAND_LEFT, CANON), HAND_LEFT)).toEqual(CANON);
  });

  it("左右手同一根物理手指落在同一个 canonical 下标上", () => {
    // 只让"拇指"有值，两只手都应当只在下标 0 出现
    const onlyThumb = [200, 0, 0, 0, 0];
    const left = canonicalBendRaw(makeRaw(HAND_LEFT, onlyThumb), HAND_LEFT);
    const right = canonicalBendRaw(makeRaw(HAND_RIGHT, onlyThumb), HAND_RIGHT);
    expect(left[0]).toBe(200);
    expect(right[0]).toBe(200);
    expect(left.slice(1)).toEqual([0, 0, 0, 0]);
    expect(right.slice(1)).toEqual([0, 0, 0, 0]);
  });

  it("越界/短数组不抛，缺失位补 0", () => {
    expect(canonicalBendRaw([], HAND_LEFT)).toEqual([0, 0, 0, 0, 0]);
  });
});

describe("bendRatios 极性", () => {
  const openVals = [40, 40, 40, 40, 40];
  const fistVals = [200, 200, 200, 200, 200];

  it("正极性（越弯值越大）：张开→0，握拳→1，中点→0.5", () => {
    const range: BendRange = { open: openVals, fist: fistVals };
    expect(bendRatios(makeRaw(HAND_RIGHT, openVals), HAND_RIGHT, range)).toEqual([0, 0, 0, 0, 0]);
    expect(bendRatios(makeRaw(HAND_RIGHT, fistVals), HAND_RIGHT, range)).toEqual([1, 1, 1, 1, 1]);
    const mid = bendRatios(makeRaw(HAND_RIGHT, [120, 120, 120, 120, 120]), HAND_RIGHT, range);
    mid.forEach((v) => expect(v).toBeCloseTo(0.5, 6));
  });

  it("反极性（越弯值越小）：分母为负，比例依然 0→1 单调", () => {
    // 这条锁住"让两点标定自己吸收符号"的设计：不要加判方向的分支
    const range: BendRange = { open: fistVals, fist: openVals };
    expect(bendRatios(makeRaw(HAND_RIGHT, fistVals), HAND_RIGHT, range)).toEqual([0, 0, 0, 0, 0]);
    expect(bendRatios(makeRaw(HAND_RIGHT, openVals), HAND_RIGHT, range)).toEqual([1, 1, 1, 1, 1]);
    const mid = bendRatios(makeRaw(HAND_RIGHT, [120, 120, 120, 120, 120]), HAND_RIGHT, range);
    mid.forEach((v) => expect(v).toBeCloseTo(0.5, 6));
  });

  it("超出标定区间被钳到 [0,1]", () => {
    const range: BendRange = { open: openVals, fist: fistVals };
    const below = bendRatios(makeRaw(HAND_RIGHT, [0, 0, 0, 0, 0]), HAND_RIGHT, range);
    const above = bendRatios(makeRaw(HAND_RIGHT, [255, 255, 255, 255, 255]), HAND_RIGHT, range);
    below.forEach((v) => expect(v).toBe(0));
    above.forEach((v) => expect(v).toBe(1));
  });

  it("左手标定与右手互不干扰：同样的 canonical 输入给出同样的比例", () => {
    const range: BendRange = { open: openVals, fist: fistVals };
    const canon = [40, 80, 120, 160, 200];
    const l = bendRatios(makeRaw(HAND_LEFT, canon), HAND_LEFT, range);
    const r = bendRatios(makeRaw(HAND_RIGHT, canon), HAND_RIGHT, range);
    expect(l).toEqual(r);
    // 且是单调递增的（canon 本身单调）
    for (let i = 1; i < 5; i++) expect(l[i]).toBeGreaterThan(l[i - 1]);
  });
});

describe("bendRatios 未标定回落", () => {
  it("range 为 null 时走柔和预览，不越界", () => {
    const r = bendRatios(makeRaw(HAND_RIGHT, [0, 64, 128, 192, 255]), HAND_RIGHT, null);
    r.forEach((v) => {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    });
    expect(r[0]).toBe(0);
    expect(r[4]).toBeCloseTo(0.42, 6); // 预览增益上限
  });

  it("只捕捉了一个点（open 有 fist 无）也回落到预览，不崩", () => {
    const half = { open: [40, 40, 40, 40, 40] } as unknown as BendRange;
    const r = bendRatios(makeRaw(HAND_RIGHT, [200, 200, 200, 200, 200]), HAND_RIGHT, half);
    r.forEach((v) => {
      expect(Number.isFinite(v)).toBe(true);
      expect(v).toBeLessThanOrEqual(0.42);
    });
  });

  it("死通道（两次标定读数相同，分母为 0）回落到预览而不是除零", () => {
    const range: BendRange = { open: [100, 40, 40, 40, 40], fist: [100, 200, 200, 200, 200] };
    const r = bendRatios(makeRaw(HAND_RIGHT, [100, 200, 200, 200, 200]), HAND_RIGHT, range);
    expect(Number.isFinite(r[0])).toBe(true);
    expect(r[0]).toBeCloseTo((100 / 255) * 0.42, 6);
    expect(r[1]).toBe(1);
  });
});

describe("channelSpans / weakChannels", () => {
  it("跨度取绝对值，反极性通道不会被误判为坏", () => {
    const range: BendRange = { open: [200, 40, 40, 40, 40], fist: [40, 200, 200, 200, 200] };
    expect(channelSpans(range)).toEqual([160, 160, 160, 160, 160]);
    expect(weakChannels(range)).toEqual([]);
  });

  it("挑出零跨度与跨度不足的通道", () => {
    const range: BendRange = {
      open: [100, 40, 40, 40, 40],
      fist: [100, 50, 200, 200, 200], // 0 号跨度 0，1 号跨度 10
    };
    expect(weakChannels(range)).toEqual([0, 1]);
  });

  it("恰好等于门限判为可用，差 1 判为不足", () => {
    const at: BendRange = { open: [0, 0, 0, 0, 0], fist: new Array(5).fill(MIN_USEFUL_SPAN) };
    const below: BendRange = { open: [0, 0, 0, 0, 0], fist: new Array(5).fill(MIN_USEFUL_SPAN - 1) };
    expect(weakChannels(at)).toEqual([]);
    expect(weakChannels(below)).toEqual([0, 1, 2, 3, 4]);
  });

  it("range 为 null 时不报任何弱通道", () => {
    expect(weakChannels(null)).toEqual([]);
  });
});

describe("isCalibrated / averageBends / bendDegrees / handKeyOf", () => {
  it("isCalibrated 要求两点齐备且长度为 5", () => {
    expect(isCalibrated(null)).toBe(false);
    expect(isCalibrated({ open: [1, 2, 3, 4, 5], fist: [6, 7, 8, 9, 10] })).toBe(true);
    expect(isCalibrated({ open: [1, 2, 3], fist: [6, 7, 8, 9, 10] })).toBe(false);
    expect(isCalibrated({ open: [1, 2, 3, 4, NaN], fist: [6, 7, 8, 9, 10] })).toBe(false);
  });

  it("averageBends 逐路平均，空输入返回 null", () => {
    expect(averageBends([])).toBeNull();
    expect(
      averageBends([
        [10, 20, 30, 40, 50],
        [20, 30, 40, 50, 60],
      ])
    ).toEqual([15, 25, 35, 45, 55]);
  });

  it("averageBends 忽略长度不对的帧", () => {
    expect(averageBends([[1, 2, 3], [10, 20, 30, 40, 50]])).toEqual([10, 20, 30, 40, 50]);
  });

  it("bendDegrees 把 0~1 折算到 0~90 度并钳位", () => {
    expect(bendDegrees([0, 0.5, 1, 2, -1])).toEqual([0, 45, 90, 90, 0]);
  });

  it("handKeyOf 按协议 sensorType 映射", () => {
    expect(handKeyOf(HAND_LEFT)).toBe("LH");
    expect(handKeyOf(HAND_RIGHT)).toBe("RH");
  });
});
