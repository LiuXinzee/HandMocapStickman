import { closestProperRotation } from "./rotationFit";
/** Display calibration and diagnostic measurements. No gesture model is trained here. */
import {
  applyMap,
  averageQuaternions,
  conjugateQuat,
  multiplyQuat,
  normalizeQuat,
  relativeAxisAngle,
  type OrientationCalib,
  type Quat,
  type Vec3,
} from "./orientationCalib";

export interface PoseSample {
  q: Quat;
  t: number;
}
export const AXES: Record<"X" | "Y" | "Z", Vec3> = {
  X: [1, 0, 0],
  Y: [0, 1, 0],
  Z: [0, 0, 1],
};
export type MotionAxis = keyof typeof AXES;
export function validPose(q: Quat): boolean {
  return (
    q.every(Number.isFinite) && Math.hypot(...q) > 0.5 && Math.hypot(...q) < 2
  );
}
export function poseAngle(a: Quat, b: Quat): number {
  return relativeAxisAngle(a, b)?.angleDeg ?? 0;
}
function percentile(values: number[], p: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return (
    sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p))] ??
    0
  );
}

/** A hold is judged by spread, duration and continuity, never cumulative path length. */
export function checkPoseHold(
  samples: PoseSample[],
  minMs = 900,
  now?: number
) {
  const valid = samples.filter(s => validPose(s.q));
  const mean = averageQuaternions(valid.map(s => normalizeQuat(s.q)));
  const duration = valid.length ? valid.at(-1)!.t - valid[0].t : 0;
  const spread = mean
    ? Math.max(...valid.map(s => poseAngle(mean, s.q)))
    : Infinity;
  const continuous = valid.every(
    (s, i) => !i || (s.t >= valid[i - 1].t && s.t - valid[i - 1].t < 250)
  );
  const fresh =
    now === undefined || (valid.length > 0 && now - valid.at(-1)!.t < 250);
  const ok =
    valid.length === samples.length &&
    valid.length >= 8 &&
    duration >= minMs &&
    continuous &&
    fresh &&
    spread <= 3;
  return {
    ok,
    mean,
    spread,
    duration,
    reason:
      !fresh || !continuous
        ? "数据中断，请检查连接"
        : valid.length !== samples.length
          ? "姿态数据无效"
          : duration < minMs || valid.length < 8
            ? "等待连续数据"
            : spread > 3
              ? "姿态还在变化，请保持不动"
              : "已稳定",
  };
}

/**
 * Diagnostics must record unstable orientation too; only calibration requires a stable pose.
 * Physical stillness is established by the user, not inferred from the value being tested.
 */
export function checkCaptureStart(
  kind: "still" | "pose",
  samples: PoseSample[],
  startedAt: number,
  now: number
): { ready: boolean; mean: Quat | null; reason: string } {
  if (kind === "still") {
    const remaining = Math.max(0, Math.ceil((3000 - (now - startedAt)) / 1000));
    return {
      ready: remaining === 0,
      mean: null,
      reason:
        remaining > 0
          ? `请固定实物，${remaining} 秒后开始记录；不要求角度读数稳定。`
          : "保持实物不动，正在记录 10 秒。",
    };
  }
  const hold = checkPoseHold(samples, 900, now);
  return { ready: hold.ok, mean: hold.mean, reason: hold.reason };
}

export function stationaryReport(samples: PoseSample[]) {
  const valid = samples.filter(s => validPose(s.q));
  if (valid.length < 8) return null;
  const first = valid[0].t,
    last = valid.at(-1)!.t,
    seconds = (last - first) / 1000;
  if (seconds < 1) return null;
  const start = averageQuaternions(
    valid.filter(s => s.t <= first + 500).map(s => normalizeQuat(s.q))
  )!;
  const end = averageQuaternions(
    valid.filter(s => s.t >= last - 500).map(s => normalizeQuat(s.q))
  )!;
  // Interpolate the endpoint trend, leaving short-term residual motion as a separate metric.
  const signedEnd =
    start.reduce((n, v, i) => n + v * end[i], 0) < 0
      ? (end.map(v => -v) as Quat)
      : end;
  let path = 0,
    jumps = 0,
    gaps = 0;
  const residuals = valid.map((s, i) => {
    if (i) {
      const dt = s.t - valid[i - 1].t;
      const angle = poseAngle(valid[i - 1].q, s.q);
      if (dt < 0 || dt >= 250) gaps++;
      else {
        path += angle;
        if (angle > 20 && dt < 100) jumps++;
      }
    }
    const u = (s.t - first) / (last - first);
    const trend = normalizeQuat(
      start.map((v, k) => v * (1 - u) + signedEnd[k] * u) as Quat
    );
    return poseAngle(trend, s.q);
  });
  return {
    seconds,
    netDeg: poseAngle(start, end),
    residualP95Deg: percentile(residuals, 0.95),
    pathDegPerMin: (path / seconds) * 60,
    jumps,
    gaps,
    invalid: samples.length - valid.length,
  };
}

export interface PosePair {
  axis: MotionAxis;
  degrees: number;
  q: Quat;
}
export interface AxisFit {
  map: number[][];
  rmsDeg: number;
  maxDeg: number;
  errors: number[];
}
const dot = (a: Vec3, b: Vec3) => a.reduce((n, v, i) => n + v * b[i], 0);
function inverse3(m: number[][]): number[][] | null {
  const [a, b, c] = m[0],
    [d, e, f] = m[1],
    [g, h, i] = m[2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (!Number.isFinite(det) || det <= 1e-8) return null;
  return [
    [e * i - f * h, c * h - b * i, b * f - c * e],
    [f * g - d * i, a * i - c * g, c * d - a * f],
    [d * h - e * g, b * g - a * h, a * e - b * d],
  ].map(row => row.map(v => v / det));
}
export function pairError(reference: Quat, map: number[][], pair: PosePair) {
  const motion = relativeAxisAngle(reference, pair.q);
  if (!motion) return { angleError: Math.abs(pair.degrees), axisError: 180 };
  const expected = AXES[pair.axis].map(
    v => v * Math.sign(pair.degrees)
  ) as Vec3;
  return {
    angleError: Math.abs(motion.angleDeg - Math.abs(pair.degrees)),
    axisError:
      (Math.acos(
        Math.max(-1, Math.min(1, dot(applyMap(map, motion.axis), expected)))
      ) *
        180) /
      Math.PI,
  };
}

/** Rotate axes, never stretch angles. Manual fitting allows inconsistent observations. */
export function fitPosePairs(
  reference: Quat,
  pairs: PosePair[],
  options: { allowLargeErrors?: boolean } = {}
): AxisFit | null {
  if (pairs.length < 6 || !validPose(reference)) return null;
  if (
    !Object.keys(AXES).every(axis =>
      [-1, 1].every(sign =>
        pairs.some(p => p.axis === axis && Math.sign(p.degrees) === sign)
      )
    )
  )
    return null;
  let map = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const p of pairs) {
    if (
      !validPose(p.q) ||
      !Number.isFinite(p.degrees) ||
      !(p.axis in AXES) ||
      Math.abs(p.degrees) < 25 ||
      Math.abs(p.degrees) > 110
    )
      return null;
    const motion = relativeAxisAngle(reference, p.q);
    if (
      !motion ||
      (!options.allowLargeErrors &&
        Math.abs(motion.angleDeg - Math.abs(p.degrees)) > 15)
    )
      return null;
    const expected = AXES[p.axis].map(v => v * Math.sign(p.degrees));
    for (let r = 0; r < 3; r++)
      for (let c = 0; c < 3; c++) map[r][c] += expected[r] * motion.axis[c];
  }
  if (options.allowLargeErrors) {
    const proper = closestProperRotation(map);
    if (!proper) return null;
    map = proper;
  } else
    for (let k = 0; k < 30; k++) {
      const inv = inverse3(map);
      if (!inv) return null;
      const next = map.map((row, r) => row.map((v, c) => (v + inv[c][r]) / 2));
      const change = Math.max(
        ...next.flatMap((row, r) => row.map((v, c) => Math.abs(v - map[r][c])))
      );
      map = next;
      if (change < 1e-10) break;
    }
  const errors = pairs.map(p => pairError(reference, map, p).axisError);
  const maxDeg = Math.max(...errors);
  if (!options.allowLargeErrors && maxDeg > 15) return null;
  return {
    map,
    errors,
    maxDeg,
    rmsDeg: Math.sqrt(errors.reduce((n, e) => n + e * e, 0) / errors.length),
  };
}

/** Re-zero after power-on, retaining the sensor-to-hand mounting transform. */
export function rebaseCalibration(
  calib: OrientationCalib | null,
  reference: Quat
): OrientationCalib {
  const next = {
    ...calib,
    reference: normalizeQuat(reference),
    updatedAt: Date.now(),
  };
  // Move saved reference poses into the new session's world frame, preserving relative angles.
  if (calib) {
    for (const key of ["flat", "palmsIn"] as const) {
      if (calib[key])
        next[key] = multiplyQuat(
          next.reference,
          multiplyQuat(conjugateQuat(calib.reference), calib[key]!)
        );
    }
  }
  return next;
}
