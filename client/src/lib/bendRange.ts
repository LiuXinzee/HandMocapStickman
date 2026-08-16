/**
 * bendRange — 弯折通道两点标定（张开 / 握拳）
 *
 * 手套每指只有 1 路弯折传感器，输出是 8 位 ADC，**不是角度**。规格书标称重复性
 * ±8%，而且极性未知（不同批次可能是"越弯值越大"或反过来）。所以要把 ADC 变成
 * 可用的 0~1 弯曲度，只能靠两点标定：录一次张开、录一次握拳，剩下的线性内插。
 *
 * 关键设计：**分母允许为负**。`(cur−open)/(fist−open)` 里如果这一路是反极性，
 * 分子分母同时变号，比例仍然正确 —— 让两点标定自己吸收符号，不要另写一套
 * "判方向"的逻辑，也不要为两种极性各定一套阈值（那是错的：σ 和噪声百分比
 * 互为倒数，两套门限必然自相矛盾）。
 *
 * 指序：本模块对外一律用 **canonical 顺序 = 拇指→小指**。原始取值走
 * sensorMapping.getBendValues()，它返回的是**物理顺序**，左右手相反
 * （左手 LH_BEND 是小指→拇指、右手 RH_BEND 是拇指→小指，见 sensorMapping.ts:30/:66）。
 * 千万不要自己按 137 维下标索引五指 —— 左手一定会错位。
 */

import { getBendValues } from "./sensorMapping";

/** 手别常量，与协议的 sensorType 一致 */
export const HAND_LEFT = 0x01;
export const HAND_RIGHT = 0x02;

/** canonical 手指名，与返回数组下标对应 */
export const FINGER_NAMES = ["拇指", "食指", "中指", "无名指", "小指"] as const;

/**
 * 通道可用的最小 ADC 跨度。规格书重复性 ±8%，跨度小于这个值时
 * 标定区间已经被噪声吃掉，判为该路没贴好或已损坏。
 */
export const MIN_USEFUL_SPAN = 25;

/** 未标定时的柔和预览系数（沿用 cc_part2 glove-protocol.ts:394 的做法） */
const PREVIEW_GAIN = 0.42;

const STORE_KEY_PREFIX = "deafkit_bend_range_v1_";

export interface BendRange {
  /** 张开姿态下的 5 路 ADC，canonical 拇指→小指 */
  open: number[];
  /** 握拳姿态下的 5 路 ADC，canonical 拇指→小指 */
  fist: number[];
}

export type HandKey = "LH" | "RH";

export function handKeyOf(sensorType: number): HandKey {
  return sensorType === HAND_LEFT ? "LH" : "RH";
}

function clamp01(v: number): number {
  if (!isFinite(v)) return 0;
  // 用 <= 0 而不是 < 0：反极性时 (v-open)/负分母 会算出 -0，
  // 它虽然数值等于 0，但会渲染成 "-0%" 这种宽度字符串。
  return v <= 0 ? 0 : v > 1 ? 1 : v;
}

/**
 * 取 5 路弯折 ADC，转成 canonical 拇指→小指 顺序。
 * @param raw 原始 256 字节传感器数据（GloveFrame.sensor_data）
 * @param sensorType 0x01=左手 0x02=右手
 */
export function canonicalBendRaw(raw: number[], sensorType: number): number[] {
  const physical = getBendValues(raw, sensorType);
  // 左手 getBendValues 返回 [小指, 无名指, 中指, 食指, 拇指]，反过来才是 canonical
  return sensorType === HAND_LEFT ? physical.slice().reverse() : physical.slice();
}

/**
 * ADC → 0~1 弯曲度。
 * @param range null 或跨度为 0 时回落到柔和预览（`v/255*0.42`），并不代表真实角度
 */
export function bendRatios(
  raw: number[],
  sensorType: number,
  range: BendRange | null
): number[] {
  const values = canonicalBendRaw(raw, sensorType);
  return values.map((v, i) => {
    const open = range?.open?.[i];
    const fist = range?.fist?.[i];
    if (typeof open !== "number" || typeof fist !== "number") {
      return clamp01(v / 255) * PREVIEW_GAIN;
    }
    const denominator = fist - open;
    // 分母为 0 说明这一路两次标定读数相同（死通道），退回预览而不是除零
    if (Math.abs(denominator) < 1e-6) return clamp01(v / 255) * PREVIEW_GAIN;
    // 分母为负是合法情况：反极性由这里自动吸收
    return clamp01((v - open) / denominator);
  });
}

/** 0~1 弯曲度 → 度数（满量程按各指节约 90° 折算，与 cc_part2 同口径） */
export function bendDegrees(ratios: number[]): number[] {
  return ratios.map((r) => clamp01(r) * 90);
}

/** 各路的标定跨度 |fist−open|，用来判断通道是否可用 */
export function channelSpans(range: BendRange): number[] {
  return range.open.map((open, i) => Math.abs((range.fist[i] ?? open) - open));
}

/** 跨度不足的通道下标（对应 FINGER_NAMES） */
export function weakChannels(range: BendRange | null): number[] {
  if (!range) return [];
  const spans = channelSpans(range);
  const out: number[] = [];
  spans.forEach((s, i) => {
    if (s < MIN_USEFUL_SPAN) out.push(i);
  });
  return out;
}

/** 两点是否都已捕捉（open / fist 各 5 项且为有限数） */
export function isCalibrated(range: BendRange | null): boolean {
  if (!range) return false;
  const ok = (a: number[] | undefined) =>
    Array.isArray(a) && a.length === 5 && a.every((v) => typeof v === "number" && isFinite(v));
  return ok(range.open) && ok(range.fist);
}

/**
 * 多帧平均，用于"捕捉张开/握拳"按钮：单帧受 ±8% 重复性影响太大。
 * @param frames 每项都是 canonical 顺序的 5 路 ADC
 */
export function averageBends(frames: number[][]): number[] | null {
  const valid = frames.filter((f) => Array.isArray(f) && f.length === 5);
  if (valid.length === 0) return null;
  const sum = [0, 0, 0, 0, 0];
  for (const f of valid) for (let i = 0; i < 5; i++) sum[i] += f[i] ?? 0;
  return sum.map((s) => s / valid.length);
}

// ===== 持久化 =====

export function loadBendRange(hand: HandKey): BendRange | null {
  try {
    const raw = localStorage.getItem(STORE_KEY_PREFIX + hand);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as BendRange;
    return isCalibrated(parsed) ? parsed : null;
  } catch {
    // 隐私模式 / 配额用尽 / 脏数据：当作未标定
    return null;
  }
}

export function saveBendRange(hand: HandKey, range: BendRange): void {
  try {
    localStorage.setItem(STORE_KEY_PREFIX + hand, JSON.stringify(range));
  } catch {
    /* 标定仅在本次会话有效 */
  }
}

export function clearBendRange(hand: HandKey): void {
  try {
    localStorage.removeItem(STORE_KEY_PREFIX + hand);
  } catch {
    /* ignore */
  }
}
