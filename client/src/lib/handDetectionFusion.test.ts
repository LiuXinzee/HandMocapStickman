import { describe, expect, it } from "vitest";
import {
  mapRoiLandmarksToFrame,
  mergeHandDetections,
  type DetectionLandmark,
  type HandDetectionCandidate,
} from "./handDetectionFusion";

function makeHand(
  centerX: number,
  centerY: number,
  handedness: string,
  source: HandDetectionCandidate["source"] = "standard",
  score = 0.9
): HandDetectionCandidate {
  const landmarks: DetectionLandmark[] = Array.from(
    { length: 21 },
    (_, index) => {
      const column = (index % 5) - 2;
      const row = Math.floor(index / 5);
      return {
        x: centerX + column * 0.012,
        y: centerY - row * 0.018,
        z: -row * 0.002,
      };
    }
  );

  return { landmarks, handedness, score, source };
}

describe("mapRoiLandmarksToFrame", () => {
  it("maps ROI-normalized x/y/z into full-frame coordinates", () => {
    const mapped = mapRoiLandmarksToFrame(
      [
        { x: 0, y: 0, z: -0.2 },
        { x: 0.5, y: 0.25, z: 0.4 },
        { x: 1, y: 1, z: 0 },
      ],
      { x: 160, y: 120, width: 320, height: 240 },
      640,
      480
    );

    expect(mapped).toEqual([
      { x: 0.25, y: 0.25, z: -0.1 },
      { x: 0.5, y: 0.375, z: 0.2 },
      { x: 0.75, y: 0.75, z: 0 },
    ]);
  });

  it("returns no landmarks for invalid dimensions, regions, or points", () => {
    const validLandmarks = [{ x: 0.5, y: 0.5, z: 0 }];

    expect(
      mapRoiLandmarksToFrame(
        validLandmarks,
        { x: 0, y: 0, width: 0, height: 20 },
        100,
        100
      )
    ).toEqual([]);
    expect(
      mapRoiLandmarksToFrame(
        validLandmarks,
        { x: 0, y: 0, width: 20, height: 20 },
        0,
        100
      )
    ).toEqual([]);
    expect(
      mapRoiLandmarksToFrame(
        [{ x: Number.NaN, y: 0.5, z: 0 }],
        { x: 0, y: 0, width: 20, height: 20 },
        100,
        100
      )
    ).toEqual([]);
  });
});

describe("mergeHandDetections", () => {
  it("keeps the primary detection when an ROI pass finds the same hand", () => {
    const primary = makeHand(0.3, 0.6, "Left");
    const duplicate = makeHand(0.31, 0.59, "Left", "glove-roi", 0.99);

    expect(mergeHandDetections([primary], [duplicate], 2)).toEqual([primary]);
  });

  it("keeps explicitly different hands even when their skeletons overlap", () => {
    const left = makeHand(0.5, 0.55, "Left");
    const right = makeHand(0.53, 0.555, "Right", "glove-roi");

    expect(mergeHandDetections([left], [right], 2)).toEqual([left, right]);
  });

  it("deduplicates the same skeleton when ROI handedness flips", () => {
    const primary = makeHand(0.5, 0.55, "Left");
    const flippedDuplicate = makeHand(0.505, 0.555, "Right", "glove-roi");

    expect(mergeHandDetections([primary], [flippedDuplicate], 2)).toEqual([
      primary,
    ]);
  });

  it("fills the second hand from an ROI fallback", () => {
    const primary = makeHand(0.25, 0.55, "Left");
    const secondRoiHand = {
      ...makeHand(0.75, 0.55, "Right", "glove-roi"),
      regionIndex: 1,
    };

    expect(mergeHandDetections([primary], [secondRoiHand], 2)).toEqual([
      primary,
      secondRoiHand,
    ]);
  });

  it("deduplicates same-label skeletons while respecting maxHands", () => {
    const left = makeHand(0.2, 0.6, "Left");
    const duplicateLeft = makeHand(0.205, 0.595, "left", "glove-roi");
    const right = makeHand(0.55, 0.6, "Right", "glove-roi");
    const extra = makeHand(0.85, 0.6, "Unknown", "glove-roi");

    expect(
      mergeHandDetections([left], [duplicateLeft, right, extra], 2)
    ).toEqual([left, right]);
  });

  it("ignores invalid candidates and invalid limits", () => {
    const valid = makeHand(0.5, 0.5, "Left");
    const invalid = {
      ...makeHand(0.8, 0.5, "Right", "glove-roi"),
      landmarks: [{ x: Number.NaN, y: 0.5, z: 0 }],
    };

    expect(mergeHandDetections([invalid, valid], null, 2)).toEqual([valid]);
    expect(mergeHandDetections([valid], [], Number.NaN)).toEqual([]);
    expect(mergeHandDetections([valid], [], 0)).toEqual([]);
    expect(mergeHandDetections([valid], [], 0.5)).toEqual([]);
  });
});
