import { describe, expect, it } from "vitest";
import type { GloveVisionLandmark } from "./gloveVision";
import {
  excludeDetectedGloveRegions,
  GloveRoiTracker,
} from "./gloveRoiTracker";

function makeHand(centerX: number, centerY = 0.58): GloveVisionLandmark[] {
  const points: ReadonlyArray<readonly [number, number]> = [
    [0, 0.07],
    [-0.04, 0.035],
    [-0.07, 0.005],
    [-0.09, -0.02],
    [-0.11, -0.045],
    [-0.045, 0],
    [-0.05, -0.055],
    [-0.052, -0.11],
    [-0.055, -0.165],
    [0, -0.01],
    [0, -0.075],
    [0, -0.14],
    [0, -0.205],
    [0.04, 0],
    [0.045, -0.06],
    [0.048, -0.12],
    [0.052, -0.18],
    [0.075, 0.02],
    [0.085, -0.03],
    [0.095, -0.08],
    [0.105, -0.13],
  ];
  return points.map(([x, y], index) => ({
    x: centerX + x,
    y: centerY + y,
    z: -index * 0.001,
  }));
}

describe("GloveRoiTracker", () => {
  it("retains two separate retry crops when both hands disappear at a crossing", () => {
    const tracker = new GloveRoiTracker();
    tracker.update(
      ["left-track", "right-track"],
      [makeHand(0.3), makeHand(0.7)],
      0
    );
    tracker.update(
      ["left-track", "right-track"],
      [makeHand(0.43), makeHand(0.57)],
      100
    );
    tracker.update([], [], 180);

    const regions = tracker.predict(320, 180, 180);
    expect(regions).toHaveLength(2);
    expect(regions.map(region => region.trackingId)).toEqual([
      "left-track",
      "right-track",
    ]);
    expect(regions[0].x + regions[0].width / 2).toBeGreaterThan(0.43 * 320);
    expect(regions[1].x + regions[1].width / 2).toBeLessThan(0.57 * 320);
    expect(regions.every(region => region.width === region.height)).toBe(true);
  });

  it("keeps a side-on hand through a short miss and expires stale history", () => {
    const tracker = new GloveRoiTracker({ maxAgeMs: 500 });
    tracker.update(["side"], [makeHand(0.08, 0.52)], 1_000);

    const held = tracker.predict(320, 180, 1_300);
    expect(held).toHaveLength(1);
    expect(held[0].x).toBe(0);
    expect(held[0].width).toBeLessThanOrEqual(180);
    expect(tracker.predict(320, 180, 1_501)).toEqual([]);
  });

  it("matches detections to overlapping predictions one-to-one", () => {
    const tracker = new GloveRoiTracker();
    tracker.update(["first", "second"], [makeHand(0.47), makeHand(0.53)], 0);
    const predictions = tracker.predict(320, 180, 40);

    const missing = excludeDetectedGloveRegions(
      predictions,
      [{ landmarks: makeHand(0.48) }],
      320,
      180
    );
    expect(missing).toHaveLength(1);
    expect(missing[0].trackingId).toBe("second");
  });

  it("rekeys a nearby returning hand without leaving a third stale track", () => {
    const tracker = new GloveRoiTracker();
    tracker.update(["old-left", "right"], [makeHand(0.3), makeHand(0.7)], 0);
    tracker.update(
      ["new-left", "right"],
      [makeHand(0.32), makeHand(0.68)],
      250
    );

    const regions = tracker.predict(320, 180, 250);
    expect(regions).toHaveLength(2);
    expect(regions.map(region => region.trackingId)).toEqual([
      "new-left",
      "right",
    ]);
  });

  it("validates options and ignores malformed observations", () => {
    expect(() => new GloveRoiTracker({ maxAgeMs: 0 })).toThrow(RangeError);
    expect(() => new GloveRoiTracker({ predictionHorizonMs: -1 })).toThrow(
      RangeError
    );
    expect(() => new GloveRoiTracker({ paddingScale: 0.9 })).toThrow(
      RangeError
    );

    const tracker = new GloveRoiTracker();
    tracker.update(["bad"], [[{ x: 0.5, y: 0.5 }]], 0);
    expect(tracker.predict(320, 180, 0)).toEqual([]);
    expect(tracker.predict(0, 180, 0)).toEqual([]);
  });
});
