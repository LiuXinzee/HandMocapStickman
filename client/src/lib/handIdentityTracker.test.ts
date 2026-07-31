import { describe, expect, it } from "vitest";
import {
  HandIdentityTracker,
  type HandIdentityLandmark,
  type StableHandIdentityFrame,
} from "./handIdentityTracker";

type DetectionSource = "standard" | "glove-enhanced";

function makeHand(
  centerX: number,
  centerY = 0.58,
  shapeBias = 0
): HandIdentityLandmark[] {
  const template: ReadonlyArray<readonly [number, number]> = [
    [0, 0.06],
    [-0.035, 0.025],
    [-0.06, 0],
    [-0.08, -0.02],
    [-0.1, -0.04],
    [-0.04, -0.005],
    [-0.045, -0.055],
    [-0.047, -0.105],
    [-0.05, -0.155],
    [0, -0.015],
    [0, -0.075],
    [0, -0.135],
    [0, -0.195],
    [0.038, -0.005],
    [0.043, -0.06],
    [0.046, -0.115],
    [0.05, -0.165],
    [0.07, 0.015],
    [0.08, -0.03],
    [0.09, -0.075],
    [0.1, -0.115],
  ];

  return template.map(([x, y], index) => ({
    x: centerX + x + (index >= 17 ? shapeBias : 0),
    y: centerY + y,
    z: -index * 0.001,
  }));
}

function centerX(landmarks: ReadonlyArray<HandIdentityLandmark>): number {
  return landmarks[0].x;
}

function update(
  tracker: HandIdentityTracker<DetectionSource>,
  hands: Array<{
    x: number;
    label: string;
    source?: DetectionSource;
    shapeBias?: number;
  }>
): StableHandIdentityFrame<DetectionSource> {
  return tracker.update({
    landmarks: hands.map(hand => makeHand(hand.x, 0.58, hand.shapeBias ?? 0)),
    handedness: hands.map(hand => hand.label),
    detectionSources: hands.map(hand => hand.source ?? "standard"),
  });
}

describe("HandIdentityTracker", () => {
  it("keeps stable ordering and metadata when detector input order reverses", () => {
    const tracker = new HandIdentityTracker<DetectionSource>();
    const first = update(tracker, [
      { x: 0.25, label: "Left", source: "standard" },
      { x: 0.75, label: "Right", source: "glove-enhanced" },
    ]);

    const reversed = update(tracker, [
      { x: 0.73, label: "Right", source: "standard" },
      { x: 0.27, label: "Left", source: "glove-enhanced" },
    ]);

    expect(reversed.trackingIds).toEqual(first.trackingIds);
    expect(reversed.handedness).toEqual(["Left", "Right"]);
    expect(reversed.inputIndices).toEqual([1, 0]);
    expect(reversed.detectionSources).toEqual(["glove-enhanced", "standard"]);
    expect(centerX(reversed.landmarks[0])).toBeCloseTo(0.27, 2);
    expect(centerX(reversed.landmarks[1])).toBeCloseTo(0.73, 2);
  });

  it("ignores a one-frame handedness flip without changing identities", () => {
    const tracker = new HandIdentityTracker<DetectionSource>();
    const first = update(tracker, [
      { x: 0.28, label: "Left" },
      { x: 0.72, label: "Right" },
    ]);
    update(tracker, [
      { x: 0.3, label: "Left" },
      { x: 0.7, label: "Right" },
    ]);

    const flipped = update(tracker, [
      { x: 0.32, label: "Right" },
      { x: 0.68, label: "Left" },
    ]);
    expect(flipped.trackingIds).toEqual(first.trackingIds);
    expect(flipped.handedness).toEqual(["Left", "Right"]);

    const recovered = update(tracker, [
      { x: 0.34, label: "Left" },
      { x: 0.66, label: "Right" },
    ]);
    expect(recovered.trackingIds).toEqual(first.trackingIds);
    expect(recovered.handedness).toEqual(["Left", "Right"]);
  });

  it("keeps identities while close hands receive two simultaneous label flips", () => {
    const tracker = new HandIdentityTracker<DetectionSource>();
    const first = update(tracker, [
      { x: 0.42, label: "Left" },
      { x: 0.58, label: "Right" },
    ]);
    update(tracker, [
      { x: 0.45, label: "Left" },
      { x: 0.55, label: "Right" },
    ]);

    const close = update(tracker, [
      { x: 0.48, label: "Right" },
      { x: 0.52, label: "Left" },
    ]);
    expect(close.trackingIds).toEqual(first.trackingIds);
    expect(close.handedness).toEqual(["Left", "Right"]);
    expect(centerX(close.landmarks[0])).toBeCloseTo(0.48, 2);
    expect(centerX(close.landmarks[1])).toBeCloseTo(0.52, 2);

    const crossed = update(tracker, [
      { x: 0.49, label: "Left" },
      { x: 0.51, label: "Right" },
    ]);
    expect(crossed.trackingIds).toEqual(first.trackingIds);
    expect(crossed.handedness).toEqual(["Left", "Right"]);
    expect(centerX(crossed.landmarks[0])).toBeCloseTo(0.51, 2);
    expect(centerX(crossed.landmarks[1])).toBeCloseTo(0.49, 2);
  });

  it("requires nine consecutive opposite labels before switching handedness", () => {
    const tracker = new HandIdentityTracker<DetectionSource>();
    const first = update(tracker, [{ x: 0.35, label: "Left" }]);

    for (let frame = 0; frame < 8; frame++) {
      const result = update(tracker, [{ x: 0.35, label: "Right" }]);
      expect(result.trackingIds).toEqual(first.trackingIds);
      expect(result.handedness).toEqual(["Left"]);
    }

    const switched = update(tracker, [{ x: 0.35, label: "Right" }]);
    expect(switched.trackingIds).toEqual(first.trackingIds);
    expect(switched.handedness).toEqual(["Right"]);
  });

  it("follows motion through a crossing even when input is spatially reordered", () => {
    const tracker = new HandIdentityTracker<DetectionSource>();
    const first = update(tracker, [
      { x: 0.22, label: "Left", shapeBias: -0.003 },
      { x: 0.78, label: "Right", shapeBias: 0.003 },
    ]);
    update(tracker, [
      { x: 0.39, label: "Left", shapeBias: -0.003 },
      { x: 0.61, label: "Right", shapeBias: 0.003 },
    ]);

    // The detector now reports spatial left-to-right order and no useful label.
    const crossed = update(tracker, [
      { x: 0.43, label: "Unknown", shapeBias: 0.003 },
      { x: 0.57, label: "Unknown", shapeBias: -0.003 },
    ]);
    expect(crossed.trackingIds).toEqual(first.trackingIds);
    expect(centerX(crossed.landmarks[0])).toBeCloseTo(0.57, 2);
    expect(centerX(crossed.landmarks[1])).toBeCloseTo(0.43, 2);

    const separated = update(tracker, [
      { x: 0.29, label: "Right", shapeBias: 0.003 },
      { x: 0.71, label: "Left", shapeBias: -0.003 },
    ]);
    expect(separated.trackingIds).toEqual(first.trackingIds);
    expect(centerX(separated.landmarks[0])).toBeCloseTo(0.71, 2);
    expect(centerX(separated.landmarks[1])).toBeCloseTo(0.29, 2);
  });

  it("rejects a detection that jumps onto the other hand", () => {
    const tracker = new HandIdentityTracker<DetectionSource>();
    const first = update(tracker, [
      { x: 0.24, label: "Left" },
      { x: 0.76, label: "Right" },
    ]);
    update(tracker, [
      { x: 0.26, label: "Left" },
      { x: 0.74, label: "Right" },
    ]);

    const outlierFrame = update(tracker, [
      { x: 0.7, label: "Left" },
      { x: 0.73, label: "Right" },
    ]);
    expect(outlierFrame.trackingIds).toEqual([first.trackingIds[1]]);
    expect(outlierFrame.handedness).toEqual(["Right"]);

    const recovered = update(tracker, [
      { x: 0.28, label: "Left" },
      { x: 0.71, label: "Right" },
    ]);
    expect(recovered.trackingIds).toEqual(first.trackingIds);
    expect(recovered.handedness).toEqual(["Left", "Right"]);
  });

  it("restores the same identity after a short single-hand disappearance", () => {
    const tracker = new HandIdentityTracker<DetectionSource>({
      maxMissedFrames: 4,
    });
    const first = update(tracker, [
      { x: 0.3, label: "Left" },
      { x: 0.7, label: "Right" },
    ]);

    for (const rightX of [0.69, 0.68, 0.67]) {
      const partial = update(tracker, [{ x: rightX, label: "Right" }]);
      expect(partial.trackingIds).toEqual([first.trackingIds[1]]);
    }

    const restored = update(tracker, [
      { x: 0.31, label: "Left", source: "glove-enhanced" },
      { x: 0.66, label: "Right" },
    ]);
    expect(restored.trackingIds).toEqual(first.trackingIds);
    expect(restored.handedness).toEqual(["Left", "Right"]);
    expect(restored.detectionSources).toEqual(["glove-enhanced", "standard"]);
  });

  it("drops invalid landmark sets and validates configuration", () => {
    expect(() => new HandIdentityTracker({ maxMissedFrames: -1 })).toThrow(
      RangeError
    );
    expect(() => new HandIdentityTracker({ maxMatchDistance: 0 })).toThrow(
      RangeError
    );

    const tracker = new HandIdentityTracker();
    const result = tracker.update({
      landmarks: [[{ x: 0.5, y: 0.5, z: 0 }]],
      handedness: ["Left"],
    });
    expect(result.trackingIds).toEqual([]);
    expect(tracker.activeTrackCount).toBe(0);
  });
});
