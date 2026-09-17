import {
  averageQuaternions,
  normalizeQuat,
  type OrientationCalib,
  type Quat,
} from "./orientationCalib";
import {
  AXES,
  fitPosePairs,
  pairError,
  poseAngle,
  validPose,
  type MotionAxis,
  type PosePair,
  type PoseSample,
} from "./motionCalibration";

export const MANUAL_MOTIONS: Record<
  MotionAxis,
  {
    title: string;
    positive: string;
    negative: string;
    description: string;
  }
> = {
  X: {
    title: "前后倾斜",
    positive: "指尖朝自己倾",
    negative: "指尖远离自己倾",
    description:
      "从竖立零位出发，让指尖朝自己或远离自己倾斜，避免同时左右侧倾。",
  },
  Y: {
    title: "左右转掌",
    positive: "掌心转向画面右侧",
    negative: "掌心转向画面左侧",
    description: "指尖保持向上，绕竖直方向转动掌心；正反方向以目标示意为准。",
  },
  Z: {
    title: "左右侧倾",
    positive: "指尖倾向画面左侧",
    negative: "指尖倾向画面右侧",
    description: "掌面朝向保持不变，让指尖像钟摆一样向两侧倾斜。",
  },
};
export const MANUAL_AXES = Object.keys(AXES) as MotionAxis[];
export function manualTargetLabel(axis: MotionAxis, degrees: number) {
  const motion = MANUAL_MOTIONS[axis];
  return (
    (degrees > 0 ? motion.positive : motion.negative) +
    " " +
    Math.abs(degrees) +
    "°"
  );
}
export function posePairKey(pair: Pick<PosePair, "axis" | "degrees">) {
  return pair.axis + ":" + pair.degrees;
}
export function upsertPosePair(pairs: PosePair[], pair: PosePair): PosePair[] {
  return [
    ...pairs.filter(p => posePairKey(p) !== posePairKey(pair)),
    { ...pair, q: [...pair.q] as Quat },
  ];
}

/** User presses Record after posing. Short averaging does not wait for stillness. */
export function captureManualPose(samples: PoseSample[], now: number) {
  const recent = samples.filter(s => s.t >= now - 150);
  const failed = (reason: string) => ({
    ok: false as const,
    reason,
    q: null,
    spread: null,
  });
  if (!recent.length || now - recent.at(-1)!.t >= 100)
    return failed("没有新鲜数据，请检查手套连接后重试。");
  if (recent.some(s => s.t > now || !validPose(s.q)))
    return failed("当前姿态数据无效，未记录，请重试。");
  if (recent.length < 3 || recent.at(-1)!.t - recent[0].t < 30)
    return failed("正在接收数据，请稍后再点记录。");
  if (
    recent.some(
      (s, i) => i > 0 && (s.t < recent[i - 1].t || s.t - recent[i - 1].t >= 100)
    )
  )
    return failed("采样发生中断，未记录，请重试。");
  const q = averageQuaternions(recent.map(s => normalizeQuat(s.q)))!;
  const spread = Math.max(...recent.map(s => poseAngle(q, s.q)));
  return {
    ok: true as const,
    reason: spread > 3 ? "采样时角度变化较大，可重采这一点。" : "",
    q,
    spread,
  };
}

export function evaluateManualCalibration(reference: Quat, pairs: PosePair[]) {
  const missing = MANUAL_AXES.flatMap(axis =>
    [-1, 1]
      .filter(
        sign =>
          !pairs.some(p => p.axis === axis && Math.sign(p.degrees) === sign)
      )
      .map(sign => manualTargetLabel(axis, sign * 30))
  );
  const measurements = pairs.map(p => ({
    key: posePairKey(p),
    measuredDeg: poseAngle(reference, p.q),
    angleError: Math.abs(poseAngle(reference, p.q) - Math.abs(p.degrees)),
  }));
  const bad = measurements.filter(p => p.angleError > 15);
  // Manual labels are accepted even when sensor magnitudes or axes disagree.
  // Quality remains visible; the user decides whether to save the preview.
  const fit = fitPosePairs(reference, pairs, { allowLargeErrors: true });
  const errors = fit
    ? pairs.map(p => ({
        key: posePairKey(p),
        ...pairError(reference, fit.map, p),
      }))
    : [];
  let reason = "";
  if (missing.length)
    reason =
      "还需记录 " +
      missing.length +
      " 个方向：" +
      missing.join("、") +
      "（角度可自行选择）。";
  else if (!fit)
    reason =
      "当前数据无法算出有效的三轴映射（例如转轴无法分开、相互抵消或姿态数据无效），不是角度差值超过 15° 被拦截。采样仍已记录，可在历史页对比。";
  else {
    reason =
      "已生成可试用的方向映射，平均转轴偏差 " + fit.rmsDeg.toFixed(1) + "°。";
    if (bad.length)
      reason +=
        "有 " +
        bad.length +
        " 个姿态的手套报告转角与人工标注相差超过 15°，仅供参考，不阻止试用或保存。";
    if (fit.maxDeg > 15)
      reason += "转轴拟合也存在较大偏差，请观察实际预览效果。";
    reason += "当前算法只调整方向，不会把手套报告的角度大小变成目标值。";
  }
  return { fit, missing, measurements, errors, reason };
}

export function buildManualCalibration(
  reference: Quat,
  pairs: PosePair[],
  confirmed: boolean
): OrientationCalib | null {
  if (!confirmed) return null;
  const { fit } = evaluateManualCalibration(reference, pairs);
  if (!fit) return null;
  return {
    reference: normalizeQuat(reference),
    axisMap: fit.map,
    method: "manual-angle",
    sampleCount: pairs.length,
    updatedAt: Date.now(),
    fitErrorDeg: fit.rmsDeg,
  };
}
