/*
 * sensorMapping.ts — 手套传感器索引映射
 *
 * 矩侨精密 织物电子皮肤 高频版 V2.3 规格书中的索引是 1-based，
 * 代码中数组索引是 0-based，所以所有索引需要 -1。
 *
 * 原始数据流中 256 字节的排列顺序与传感器物理位置不一致，
 * 需要按照规格书的映射表重新排列为物理顺序。
 *
 * 每只手有 137 个有效传感点:
 *   - 60 个手指压力点 (每指12个)
 *   - 5 个弯折传感器
 *   - 72 个手掌压力点
 */

// ===== 左手索引映射 (sensor_type = 0x01) =====
// 规格书中为 1-based，这里转为 0-based

/** 左手小拇指压力 (12点) */
const LH_PINKY_PRESSURE = [31, 30, 29, 15, 14, 13, 255, 254, 253, 239, 238, 237].map(i => i - 1);
/** 左手无名指压力 (12点) */
const LH_RING_PRESSURE = [28, 27, 26, 12, 11, 10, 252, 251, 250, 236, 235, 234].map(i => i - 1);
/** 左手中指压力 (12点) */
const LH_MIDDLE_PRESSURE = [25, 24, 23, 9, 8, 7, 249, 248, 247, 233, 232, 231].map(i => i - 1);
/** 左手食指压力 (12点) */
const LH_INDEX_PRESSURE = [22, 21, 20, 6, 5, 4, 246, 245, 244, 230, 229, 228].map(i => i - 1);
/** 左手大拇指压力 (12点) */
const LH_THUMB_PRESSURE = [19, 18, 17, 3, 2, 1, 243, 242, 241, 227, 226, 225].map(i => i - 1);

/** 左手弯折传感器 [小拇指, 无名指, 中指, 食指, 大拇指] */
const LH_BEND = [222, 219, 216, 213, 210].map(i => i - 1);

/** 左手手掌 (72点) */
const LH_PALM = [
  207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196,
  191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180,
  179, 178, 177, 175, 174, 173, 172, 171, 170, 169, 168, 167,
  166, 165, 164, 163, 162, 161, 159, 158, 157, 156, 155, 154,
  153, 152, 151, 150, 149, 148, 147, 146, 145, 143, 142, 141,
  140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129
].map(i => i - 1);

/** 左手完整映射 (137点，按物理顺序) */
export const LEFT_HAND_INDEX_MAP: number[] = [
  ...LH_PINKY_PRESSURE,   // 0-11: 小拇指压力
  ...LH_RING_PRESSURE,    // 12-23: 无名指压力
  ...LH_MIDDLE_PRESSURE,  // 24-35: 中指压力
  ...LH_INDEX_PRESSURE,   // 36-47: 食指压力
  ...LH_THUMB_PRESSURE,   // 48-59: 大拇指压力
  ...LH_BEND,             // 60-64: 弯折 [小拇指,无名指,中指,食指,大拇指]
  ...LH_PALM,             // 65-136: 手掌
];

// ===== 右手索引映射 (sensor_type = 0x02) =====

/** 右手大拇指压力 (12点) */
const RH_THUMB_PRESSURE = [240, 239, 238, 256, 255, 254, 16, 15, 14, 32, 31, 30].map(i => i - 1);
/** 右手食指压力 (12点) */
const RH_INDEX_PRESSURE = [237, 236, 235, 253, 252, 251, 13, 12, 11, 29, 28, 27].map(i => i - 1);
/** 右手中指压力 (12点) */
const RH_MIDDLE_PRESSURE = [234, 233, 232, 250, 249, 248, 10, 9, 8, 26, 25, 24].map(i => i - 1);
/** 右手无名指压力 (12点) */
const RH_RING_PRESSURE = [231, 230, 229, 247, 246, 245, 7, 6, 5, 23, 22, 21].map(i => i - 1);
/** 右手小拇指压力 (12点) */
const RH_PINKY_PRESSURE = [228, 227, 226, 244, 243, 242, 4, 3, 2, 20, 19, 18].map(i => i - 1);

/** 右手弯折传感器 [大拇指, 食指, 中指, 无名指, 小拇指] */
const RH_BEND = [47, 44, 41, 38, 35].map(i => i - 1);

/** 右手手掌 (72点) */
const RH_PALM = [
  61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50,
  80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
  96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82,
  112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98,
  128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114
].map(i => i - 1);

/** 右手完整映射 (137点，按物理顺序) */
export const RIGHT_HAND_INDEX_MAP: number[] = [
  ...RH_THUMB_PRESSURE,   // 0-11: 大拇指压力
  ...RH_INDEX_PRESSURE,   // 12-23: 食指压力
  ...RH_MIDDLE_PRESSURE,  // 24-35: 中指压力
  ...RH_RING_PRESSURE,    // 36-47: 无名指压力
  ...RH_PINKY_PRESSURE,   // 48-59: 小拇指压力
  ...RH_BEND,             // 60-64: 弯折 [大拇指,食指,中指,无名指,小拇指]
  ...RH_PALM,             // 65-136: 手掌
];

// ===== 工具函数 =====

/**
 * 将原始 256 字节数据按物理索引映射重新排列
 * @param rawData 原始 256 字节传感器数据
 * @param sensorType 传感器类型 (0x01=LH, 0x02=RH)
 * @returns 重新排列后的 137 个有效传感器值（物理顺序）
 */
export function remapSensorData(rawData: number[], sensorType: number): number[] {
  const indexMap = sensorType === 0x01 ? LEFT_HAND_INDEX_MAP : RIGHT_HAND_INDEX_MAP;
  return indexMap.map(idx => rawData[idx] ?? 0);
}

/**
 * 获取弯折传感器值
 * @param rawData 原始 256 字节传感器数据
 * @param sensorType 传感器类型
 * @returns 5个弯折值 [拇指/小拇指, 食指/无名指, 中指, 无名指/食指, 小拇指/大拇指]
 */
export function getBendValues(rawData: number[], sensorType: number): number[] {
  if (sensorType === 0x01) {
    // 左手: [小拇指, 无名指, 中指, 食指, 大拇指]
    return LH_BEND.map(idx => rawData[idx] ?? 0);
  } else {
    // 右手: [大拇指, 食指, 中指, 无名指, 小拇指]
    return RH_BEND.map(idx => rawData[idx] ?? 0);
  }
}

/**
 * 获取各手指压力值（每指12点的平均值）
 * @param rawData 原始 256 字节传感器数据
 * @param sensorType 传感器类型
 * @returns 5个手指平均压力值
 */
export function getFingerPressures(rawData: number[], sensorType: number): {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
} {
  const avg = (indices: number[]) => {
    const values = indices.map(idx => rawData[idx] ?? 0);
    return values.reduce((a, b) => a + b, 0) / values.length;
  };

  if (sensorType === 0x01) {
    return {
      thumb: avg(LH_THUMB_PRESSURE),
      index: avg(LH_INDEX_PRESSURE),
      middle: avg(LH_MIDDLE_PRESSURE),
      ring: avg(LH_RING_PRESSURE),
      pinky: avg(LH_PINKY_PRESSURE),
    };
  } else {
    return {
      thumb: avg(RH_THUMB_PRESSURE),
      index: avg(RH_INDEX_PRESSURE),
      middle: avg(RH_MIDDLE_PRESSURE),
      ring: avg(RH_RING_PRESSURE),
      pinky: avg(RH_PINKY_PRESSURE),
    };
  }
}

/**
 * 获取手掌压力值
 * @param rawData 原始 256 字节传感器数据
 * @param sensorType 传感器类型
 * @returns 72个手掌压力值（按物理顺序）
 */
export function getPalmPressures(rawData: number[], sensorType: number): number[] {
  const palmIndices = sensorType === 0x01 ? LH_PALM : RH_PALM;
  return palmIndices.map(idx => rawData[idx] ?? 0);
}

// ===== 区域标签 =====

export interface SensorRegion {
  name: string;
  startIdx: number; // 在 remapped 数组中的起始索引
  endIdx: number;   // 在 remapped 数组中的结束索引（不含）
  color: string;    // 显示颜色
}

export const LEFT_HAND_REGIONS: SensorRegion[] = [
  { name: "小拇指", startIdx: 0, endIdx: 12, color: "#ff6b6b" },
  { name: "无名指", startIdx: 12, endIdx: 24, color: "#ffa94d" },
  { name: "中指", startIdx: 24, endIdx: 36, color: "#ffd43b" },
  { name: "食指", startIdx: 36, endIdx: 48, color: "#69db7c" },
  { name: "大拇指", startIdx: 48, endIdx: 60, color: "#4dabf7" },
  { name: "弯折", startIdx: 60, endIdx: 65, color: "#da77f2" },
  { name: "手掌", startIdx: 65, endIdx: 137, color: "#868e96" },
];

export const RIGHT_HAND_REGIONS: SensorRegion[] = [
  { name: "大拇指", startIdx: 0, endIdx: 12, color: "#4dabf7" },
  { name: "食指", startIdx: 12, endIdx: 24, color: "#69db7c" },
  { name: "中指", startIdx: 24, endIdx: 36, color: "#ffd43b" },
  { name: "无名指", startIdx: 36, endIdx: 48, color: "#ffa94d" },
  { name: "小拇指", startIdx: 48, endIdx: 60, color: "#ff6b6b" },
  { name: "弯折", startIdx: 60, endIdx: 65, color: "#da77f2" },
  { name: "手掌", startIdx: 65, endIdx: 137, color: "#868e96" },
];
