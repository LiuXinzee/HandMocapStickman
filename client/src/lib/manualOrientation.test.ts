import { describe, expect, it } from "vitest";
import {
  buildManualCalibration,
  captureManualPose,
  evaluateManualCalibration,
  upsertPosePair,
} from "./manualOrientation";
import { AXES, poseAngle, type PosePair } from "./motionCalibration";
import {
  applyOrientationCalib,
  conjugateQuat,
  multiplyQuat,
  type Quat,
  type Vec3,
} from "./orientationCalib";
const I: Quat = [1, 0, 0, 0];
const rotation = (axis: Vec3, angle: number): Quat => {
  const half = (angle * Math.PI) / 360;
  return [Math.cos(half), ...axis.map(v => v * Math.sin(half))] as Quat;
};
const samples = (startAngle = 0) =>
  Array.from({ length: 16 }, (_, i) => ({
    t: 1000 + i * 10,
    q: rotation(AXES.Y, startAngle + i * 0.2),
  }));
const pairs: PosePair[] = (Object.keys(AXES) as (keyof typeof AXES)[]).flatMap(
  axis =>
    [-60, -30, 30, 60].map(degrees => ({
      axis,
      degrees,
      q: rotation(AXES[axis], degrees),
    }))
);

describe("manual angle recording", () => {
  it("records at the click without requiring a one-second stable hold", () => {
    const captured = captureManualPose(samples(70), 1150);
    expect(captured.ok).toBe(true);
    expect(poseAngle(I, captured.q!)).toBeCloseTo(71.5, 3);
    const moving = samples().map((s, i) => ({ ...s, q: rotation(AXES.X, i) }));
    const large = captureManualPose(moving, 1150);
    expect(large.ok).toBe(true);
    expect(large.reason).toContain("变化较大");
  });
  it("rejects stale, missing, invalid and discontinuous data", () => {
    expect(captureManualPose([], 1150).ok).toBe(false);
    expect(captureManualPose(samples(), 1400).ok).toBe(false);
    expect(captureManualPose(samples().slice(-1), 1150).ok).toBe(false);
    expect(
      captureManualPose(
        [...samples().slice(0, -1), { t: 1150, q: [NaN, 0, 0, 0] }],
        1150
      ).ok
    ).toBe(false);
    expect(
      captureManualPose(
        [
          { t: 1000, q: I },
          { t: 1010, q: I },
          { t: 1150, q: I },
        ],
        1150
      ).ok
    ).toBe(false);
    expect(captureManualPose(samples().reverse(), 1150).ok).toBe(false);
    expect(captureManualPose(samples(), 1100).ok).toBe(false);
  });
  it("averages equivalent quaternion signs without cancellation", () => {
    const same = samples().map((s, i) => ({
      t: s.t,
      q: rotation(AXES.Z, 45).map(v => (i % 2 ? -v : v)) as Quat,
    }));
    const recorded = captureManualPose(same, 1150);
    expect(recorded.ok).toBe(true);
    expect(poseAngle(recorded.q!, rotation(AXES.Z, 45))).toBeLessThan(0.001);
  });
  it("replaces only the chosen signed target without mutating the original data", () => {
    const replacement: PosePair = {
      axis: "X",
      degrees: 30,
      q: rotation(AXES.X, 33),
    };
    const changed = upsertPosePair(pairs, replacement);
    expect(changed).toHaveLength(12);
    expect(changed.find(p => p.axis === "X" && p.degrees === -30)!.q).toEqual(
      rotation(AXES.X, -30)
    );
    expect(pairs.find(p => p.axis === "X" && p.degrees === 30)!.q).toEqual(
      rotation(AXES.X, 30)
    );
    replacement.q[0] = 0;
    expect(changed.at(-1)!.q[0]).not.toBe(0);
  });
});

describe("manual direction fit and save", () => {
  it("accepts twelve selected angles, preserves arbitrary rotation size and requires visual confirmation", () => {
    expect(buildManualCalibration(I, pairs, false)).toBeNull();
    const saved = buildManualCalibration(I, pairs, true)!;
    expect(saved.method).toBe("manual-angle");
    expect(saved.sampleCount).toBe(12);
    expect(
      poseAngle(I, applyOrientationCalib(rotation(AXES.Y, 47), saved))
    ).toBeCloseTo(47);
  });
  it("recovers a nontrivial mounting orientation and reference on unseen combined motion", () => {
    const mounting = rotation(
      [1 / Math.sqrt(3), 1 / Math.sqrt(3), 1 / Math.sqrt(3)],
      37
    );
    const reference = rotation(AXES.Z, 71);
    const intoSensor = (model: Quat) =>
      multiplyQuat(
        reference,
        multiplyQuat(conjugateQuat(mounting), multiplyQuat(model, mounting))
      );
    const actual = pairs.map(p => ({ ...p, q: intoSensor(p.q) }));
    const calibration = buildManualCalibration(reference, actual, true)!;
    expect(calibration).not.toBeNull();
    const heldOut = multiplyQuat(rotation(AXES.X, -45), rotation(AXES.Y, 28));
    expect(
      poseAngle(
        applyOrientationCalib(intoSensor(heldOut), calibration),
        heldOut
      )
    ).toBeLessThan(0.001);
  });
  it("reports missing directions but allows large angle differences without stretching them", () => {
    const partial = evaluateManualCalibration(
      I,
      pairs.filter(p => p.axis !== "Y")
    );
    expect(partial.fit).toBeNull();
    expect(partial.missing).toHaveLength(2);
    const bad = upsertPosePair(pairs, {
      axis: "Z",
      degrees: 60,
      q: rotation(AXES.Z, 20),
    });
    const report = evaluateManualCalibration(I, bad);
    expect(report.fit).not.toBeNull();
    expect(report.reason).toContain("不阻止试用或保存");
    expect(
      report.measurements.find(m => m.key === "Z:60")!.angleError
    ).toBeCloseTo(40);
    const saved = buildManualCalibration(I, bad, true)!;
    expect(saved).not.toBeNull();
    expect(
      poseAngle(I, applyOrientationCalib(rotation(AXES.Z, 20), saved))
    ).toBeCloseTo(20);
  });
  it("refuses conflicting axes and remains independent across left and right calibration", () => {
    const conflict = pairs.map(p => ({ ...p, q: rotation(AXES.X, p.degrees) }));
    expect(evaluateManualCalibration(I, conflict).reason).toContain(
      "无法算出有效"
    );
    expect(buildManualCalibration(I, conflict, true)).toBeNull();
    const left = buildManualCalibration(I, pairs, true)!;
    const rightReference = rotation(AXES.X, 42);
    const right = buildManualCalibration(
      rightReference,
      pairs.map(p => ({
        ...p,
        q: multiplyQuat(rightReference, p.q),
      })),
      true
    )!;
    expect(left.reference).toEqual(I);
    expect(right.reference).not.toEqual(left.reference);
    expect(
      poseAngle(I, applyOrientationCalib(rightReference, right))
    ).toBeLessThan(0.001);
  });
  it("allows large axis residuals in manual preview but still requires user confirmation", () => {
    const altered = upsertPosePair(pairs, {
      axis: "X",
      degrees: 60,
      q: rotation([Math.cos(Math.PI / 3), Math.sin(Math.PI / 3), 0], 175),
    });
    const result = evaluateManualCalibration(I, altered);
    expect(result.fit).not.toBeNull();
    expect(result.fit!.maxDeg).toBeGreaterThan(15);
    expect(result.reason).toContain("较大偏差");
    expect(buildManualCalibration(I, altered, false)).toBeNull();
    expect(buildManualCalibration(I, altered, true)).not.toBeNull();
  });
  it("lets inconsistent handedness be previewed as the closest proper rotation", () => {
    const observations: PosePair[] = [
      ...[-60, 60].map(degrees => ({
        axis: "X" as const,
        degrees,
        q: rotation(AXES.X, -degrees),
      })),
      ...[-60, -30, 30, 60].map(degrees => ({
        axis: "Y" as const,
        degrees,
        q: rotation(AXES.Y, degrees),
      })),
      ...[-90, -60, -30, 30, 60, 90].map(degrees => ({
        axis: "Z" as const,
        degrees,
        q: rotation(AXES.Z, degrees),
      })),
    ];
    const r = evaluateManualCalibration(I, observations);
    expect(r.fit).not.toBeNull();
    expect(r.fit!.maxDeg).toBeCloseTo(180);
    expect(r.reason).toContain("较大偏差");
    expect(buildManualCalibration(I, observations, false)).toBeNull();
    expect(
      poseAngle(
        I,
        applyOrientationCalib(
          rotation(AXES.Y, 47),
          buildManualCalibration(I, observations, true)!
        )
      )
    ).toBeCloseTo(47);
  });
});
