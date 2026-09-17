import { describe, expect, it } from "vitest";
import {
  AXES,
  checkPoseHold,
  checkCaptureStart,
  fitPosePairs,
  pairError,
  poseAngle,
  rebaseCalibration,
  stationaryReport,
  type PosePair,
} from "./motionCalibration";
import {
  applyOrientationCalib,
  multiplyQuat,
  type Quat,
  type Vec3,
} from "./orientationCalib";
const I: Quat = [1, 0, 0, 0];
const q = (a: Vec3, d: number): Quat =>
  [
    Math.cos((d * Math.PI) / 360),
    ...a.map(v => v * Math.sin((d * Math.PI) / 360)),
  ] as Quat;
describe("diagnostics separate path from drift", () => {
  it("records a continuously changing sensor after the countdown without allowing calibration", () => {
    const changing = Array.from({ length: 101 }, (_, i) => ({
      t: 2000 + i * 10,
      q: q(AXES.Y, 60 + i * 0.6),
    }));
    expect(checkCaptureStart("still", changing, 0, 2999).ready).toBe(false);
    expect(checkCaptureStart("still", changing, 0, 3000).ready).toBe(true);
    expect(checkCaptureStart("pose", changing, 0, 3000).ready).toBe(false);
    // Diagnostic recording must expose bad or missing samples, rather than wait forever.
    expect(checkCaptureStart("still", [], 0, 3000).ready).toBe(true);
    expect(checkCaptureStart("pose", [], 0, 3000).ready).toBe(false);
    const invalid = changing.map(s => ({ ...s, q: [NaN, 0, 0, 0] as Quat }));
    expect(checkCaptureStart("still", invalid, 0, 3000).ready).toBe(true);
    expect(checkCaptureStart("pose", invalid, 0, 3000).ready).toBe(false);
    expect(stationaryReport(invalid)).toBeNull();
  });
  it("oscillation can have a high path rate but zero endpoint offset", () => {
    const samples = Array.from({ length: 201 }, (_, i) => ({
      t: i * 50,
      q: q(AXES.Y, i % 2 ? 0.1 : -0.1),
    }));
    const r = stationaryReport(samples)!;
    expect(r.netDeg).toBeLessThan(0.03);
    expect(r.residualP95Deg).toBeLessThan(0.2);
    expect(r.pathDegPerMin).toBeGreaterThan(200);
    expect(checkPoseHold(samples.slice(-21), 900, 10000).ok).toBe(true);
  });
  it("accepts USB batches sharing arrival timestamps but rejects reversed time", () => {
    const samples = Array.from({ length: 41 }, (_, i) => ({
      t: Math.floor(i / 2) * 50,
      q: I,
    }));
    expect(checkPoseHold(samples, 900, 1000).ok).toBe(true);
    expect(stationaryReport(samples)!.gaps).toBe(0);
    const reversed = [
      ...samples.slice(0, 20),
      { t: 0, q: I },
      ...samples.slice(20),
    ];
    expect(checkPoseHold(reversed).ok).toBe(false);
    expect(stationaryReport(reversed)!.gaps).toBeGreaterThan(0);
  });
  it("separates slow net rotation from short-term jitter and flags a sudden jump", () => {
    const slow = Array.from({ length: 201 }, (_, i) => ({
      t: i * 50,
      q: q(AXES.Y, i * 0.01),
    }));
    const drift = stationaryReport(slow)!;
    expect(drift.netDeg).toBeCloseTo(1.9, 1);
    expect(drift.residualP95Deg).toBeLessThan(0.06);
    const jump = slow.map((s, i) => ({ ...s, q: q(AXES.Y, i < 100 ? 0 : 35) }));
    expect(stationaryReport(jump)!.jumps).toBe(1);
  });
  it("rejects movement, missing data, invalid samples and a stale hold", () => {
    const s = Array.from({ length: 61 }, (_, i) => ({
      t: i * 50,
      q: q(AXES.X, i),
    }));
    expect(checkPoseHold(s).ok).toBe(false);
    const still = s.map(p => ({ ...p, q: I }));
    expect(checkPoseHold(still, 900, 4000).ok).toBe(false);
    expect(checkPoseHold(still.filter((_, i) => i < 10 || i > 30)).ok).toBe(
      false
    );
    expect(checkPoseHold([...still, { t: 3050, q: [NaN, 0, 0, 0] }]).ok).toBe(
      false
    );
  });
});
describe("multiple pose alignment", () => {
  const pairs: PosePair[] = (
    Object.keys(AXES) as (keyof typeof AXES)[]
  ).flatMap(axis =>
    [-60, 60].map(degrees => ({ axis, degrees, q: q(AXES[axis], degrees) }))
  );
  it("fits all three axes and checks an independent 45 degree pose without angle gain", () => {
    const fit = fitPosePairs(I, pairs)!;
    expect(fit).not.toBeNull();
    expect(fit.rmsDeg).toBeLessThan(0.001);
    expect(
      pairError(I, fit.map, { axis: "Y", degrees: 45, q: q(AXES.Y, 45) })
        .axisError
    ).toBeLessThan(0.001);
    expect(
      poseAngle(
        I,
        applyOrientationCalib(q(AXES.Z, 38), { reference: I, axisMap: fit.map })
      )
    ).toBeCloseTo(38);
  });
  it("recovers a rotated sensor basis", () => {
    const sensor: Record<string, Vec3> = {
      X: [0, 1, 0],
      Y: [0, 0, 1],
      Z: [1, 0, 0],
    };
    const turned = pairs.map(p => ({ ...p, q: q(sensor[p.axis], p.degrees) }));
    const fit = fitPosePairs(I, turned)!;
    expect(fit.rmsDeg).toBeLessThan(0.001);
    expect(
      pairError(I, fit.map, { axis: "X", degrees: 45, q: q(sensor.X, 45) })
        .axisError
    ).toBeLessThan(0.001);
  });
  it("refuses wrong magnitude, inconsistent axes and insufficient coverage", () => {
    expect(fitPosePairs(I, pairs.slice(0, 4))).toBeNull();
    expect(
      fitPosePairs(
        I,
        pairs.map(p => ({ ...p, q: q(AXES[p.axis], p.degrees / 3) }))
      )
    ).toBeNull();
    expect(
      fitPosePairs(
        I,
        pairs.map(p => ({ ...p, q: q(AXES.X, p.degrees) }))
      )
    ).toBeNull();
  });
  it("quick zero retains axis mapping and relative calibration poses", () => {
    const fit = fitPosePairs(I, pairs)!;
    const old = { reference: I, axisMap: fit.map, flat: q(AXES.X, -90) };
    const next = rebaseCalibration(old, q(AXES.Y, 37));
    expect(next.axisMap).toBe(old.axisMap);
    expect(
      poseAngle(I, applyOrientationCalib(next.reference, next))
    ).toBeLessThan(0.001);
    expect(
      poseAngle(
        I,
        applyOrientationCalib(multiplyQuat(next.reference, q(AXES.Z, 30)), next)
      )
    ).toBeCloseTo(30);
    expect(poseAngle(next.reference, next.flat!)).toBeCloseTo(90);
  });
});
