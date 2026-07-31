/*
 * gestureDetector — 基于关键点的手势识别
 * 通过分析指尖与掌根的相对位置判断手势
 */

import type { HandLandmark } from "@/hooks/useHandTracking";

export type GestureType =
  | "OPEN_PALM"
  | "FIST"
  | "POINTING"
  | "PEACE"
  | "THUMBS_UP"
  | "OK"
  | "ROCK"
  | "UNKNOWN";

interface GestureResult {
  gesture: GestureType;
  confidence: number;
  label: string;
}

// 计算两点距离
function distance(a: HandLandmark, b: HandLandmark): number {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

// 判断手指是否伸展
function isFingerExtended(
  landmarks: HandLandmark[],
  tipIdx: number,
  pipIdx: number,
  mcpIdx: number
): boolean {
  const tipToMcp = distance(landmarks[tipIdx], landmarks[mcpIdx]);
  const pipToMcp = distance(landmarks[pipIdx], landmarks[mcpIdx]);
  return tipToMcp > pipToMcp * 1.1;
}

// 判断拇指是否伸展（特殊处理）
function isThumbExtended(landmarks: HandLandmark[]): boolean {
  const tipToWrist = distance(landmarks[4], landmarks[0]);
  const mcpToWrist = distance(landmarks[2], landmarks[0]);
  return tipToWrist > mcpToWrist * 1.2;
}

export function detectGesture(landmarks: HandLandmark[]): GestureResult {
  if (!landmarks || landmarks.length < 21) {
    return { gesture: "UNKNOWN", confidence: 0, label: "未知" };
  }

  const thumb = isThumbExtended(landmarks);
  const index = isFingerExtended(landmarks, 8, 6, 5);
  const middle = isFingerExtended(landmarks, 12, 10, 9);
  const ring = isFingerExtended(landmarks, 16, 14, 13);
  const pinky = isFingerExtended(landmarks, 20, 18, 17);

  const extendedCount = [thumb, index, middle, ring, pinky].filter(Boolean).length;

  // 张开手掌
  if (extendedCount >= 4) {
    return { gesture: "OPEN_PALM", confidence: 0.9, label: "张开手掌" };
  }

  // 握拳
  if (extendedCount === 0) {
    return { gesture: "FIST", confidence: 0.85, label: "握拳" };
  }

  // 竖大拇指
  if (thumb && !index && !middle && !ring && !pinky) {
    return { gesture: "THUMBS_UP", confidence: 0.85, label: "竖大拇指" };
  }

  // 指向（食指伸展）
  if (!thumb && index && !middle && !ring && !pinky) {
    return { gesture: "POINTING", confidence: 0.85, label: "指向" };
  }

  // 剪刀手 / 和平
  if (index && middle && !ring && !pinky) {
    return { gesture: "PEACE", confidence: 0.85, label: "剪刀手" };
  }

  // OK 手势（拇指和食指接近）
  const thumbTipToIndexTip = distance(landmarks[4], landmarks[8]);
  if (thumbTipToIndexTip < 0.06 && middle && ring && pinky) {
    return { gesture: "OK", confidence: 0.8, label: "OK" };
  }

  // 摇滚手势
  if (index && !middle && !ring && pinky) {
    return { gesture: "ROCK", confidence: 0.8, label: "摇滚" };
  }

  return { gesture: "UNKNOWN", confidence: 0.5, label: "未知" };
}

// 计算手指弯曲角度（0-180度）
export function getFingerAngles(landmarks: HandLandmark[]): Record<string, number> {
  if (!landmarks || landmarks.length < 21) {
    return { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 };
  }

  const calcAngle = (a: HandLandmark, b: HandLandmark, c: HandLandmark): number => {
    const ab = { x: a.x - b.x, y: a.y - b.y };
    const cb = { x: c.x - b.x, y: c.y - b.y };
    const dot = ab.x * cb.x + ab.y * cb.y;
    const magAB = Math.sqrt(ab.x ** 2 + ab.y ** 2);
    const magCB = Math.sqrt(cb.x ** 2 + cb.y ** 2);
    if (magAB === 0 || magCB === 0) return 0;
    const cosAngle = Math.max(-1, Math.min(1, dot / (magAB * magCB)));
    return (Math.acos(cosAngle) * 180) / Math.PI;
  };

  return {
    thumb: calcAngle(landmarks[1], landmarks[2], landmarks[4]),
    index: calcAngle(landmarks[5], landmarks[6], landmarks[8]),
    middle: calcAngle(landmarks[9], landmarks[10], landmarks[12]),
    ring: calcAngle(landmarks[13], landmarks[14], landmarks[16]),
    pinky: calcAngle(landmarks[17], landmarks[18], landmarks[20]),
  };
}
