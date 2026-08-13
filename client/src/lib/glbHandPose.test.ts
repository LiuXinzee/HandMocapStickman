import { describe, expect, it } from "vitest";
import { expandGlbFingerBends, normalizeGlbQuaternion } from "./glbHandPose";

describe("expandGlbFingerBends", () => {
  it("repeats five per-finger angles across the four finger bones", () => {
    expect(expandGlbFingerBends([10, 20, 30, 40, 50])).toEqual([
      [10, 10, 10, 10],
      [20, 20, 20, 20],
      [30, 30, 30, 30],
      [40, 40, 40, 40],
      [50, 50, 50, 50],
    ]);
  });

  it("keeps 20 per-bone angles and clamps unsafe values", () => {
    const values = Array.from({ length: 20 }, (_, index) => index * 10);
    values[1] = -5;
    values[2] = Number.NaN;

    expect(expandGlbFingerBends(values)[0]).toEqual([0, 0, 0, 30]);
    expect(expandGlbFingerBends(values)[4]).toEqual([120, 120, 120, 120]);
  });
});

describe("normalizeGlbQuaternion", () => {
  it("normalizes the glove protocol's wxyz quaternion", () => {
    expect(normalizeGlbQuaternion([2, 0, 0, 0])).toEqual([1, 0, 0, 0]);
  });

  it("rejects zero and non-finite quaternions", () => {
    expect(normalizeGlbQuaternion([0, 0, 0, 0])).toBeNull();
    expect(normalizeGlbQuaternion([1, Number.NaN, 0, 0])).toBeNull();
  });
});
