import { describe, expect, it } from "vitest";
import {
  classifyGloveSurface,
  enhanceDarkGloveFrame,
  enhanceGloveFrame,
  enhanceLightGloveFrame,
  type GloveVisionRegion,
  type GloveVisionLandmark,
} from "./gloveVision";

const WIDTH = 128;
const HEIGHT = 96;

const POINTS: ReadonlyArray<readonly [number, number]> = [
  [64, 82],
  [53, 68],
  [43, 63],
  [34, 57],
  [25, 50],
  [51, 54],
  [49, 42],
  [48, 29],
  [47, 16],
  [61, 52],
  [61, 38],
  [61, 24],
  [61, 10],
  [71, 53],
  [73, 39],
  [74, 26],
  [75, 14],
  [81, 57],
  [86, 46],
  [89, 36],
  [92, 26],
];

const CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [0, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [0, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [0, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [5, 9],
  [9, 13],
  [13, 17],
];

const LANDMARKS: GloveVisionLandmark[] = POINTS.map(([x, y]) => ({
  x: x / (WIDTH - 1),
  y: y / (HEIGHT - 1),
  z: 0,
}));

type Rgb = readonly [number, number, number];

function makeFrame(color: Rgb): Uint8ClampedArray {
  const frame = new Uint8ClampedArray(WIDTH * HEIGHT * 4);
  for (let offset = 0; offset < frame.length; offset += 4) {
    frame[offset] = color[0];
    frame[offset + 1] = color[1];
    frame[offset + 2] = color[2];
    frame[offset + 3] = 255;
  }
  return frame;
}

function paintDisk(
  frame: Uint8ClampedArray,
  centerX: number,
  centerY: number,
  radius: number,
  color: Rgb
): void {
  const minX = Math.max(0, Math.floor(centerX - radius));
  const maxX = Math.min(WIDTH - 1, Math.ceil(centerX + radius));
  const minY = Math.max(0, Math.floor(centerY - radius));
  const maxY = Math.min(HEIGHT - 1, Math.ceil(centerY + radius));
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const dx = x - centerX;
      const dy = y - centerY;
      if (dx * dx + dy * dy > radius * radius) continue;
      const offset = (y * WIDTH + x) * 4;
      frame[offset] = color[0];
      frame[offset + 1] = color[1];
      frame[offset + 2] = color[2];
    }
  }
}

function paintSegment(
  frame: Uint8ClampedArray,
  from: readonly [number, number],
  to: readonly [number, number],
  radius: number,
  color: Rgb,
  start = 0,
  end = 1
): void {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const steps = Math.max(1, Math.ceil(Math.hypot(dx, dy)));
  for (
    let step = Math.ceil(steps * start);
    step <= Math.floor(steps * end);
    step++
  ) {
    const progress = step / steps;
    paintDisk(
      frame,
      from[0] + dx * progress,
      from[1] + dy * progress,
      radius,
      color
    );
  }
}

function paintRect(
  frame: Uint8ClampedArray,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  color: Rgb
): void {
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const offset = (y * WIDTH + x) * 4;
      frame[offset] = color[0];
      frame[offset + 1] = color[1];
      frame[offset + 2] = color[2];
    }
  }
}

function makeHandFrame(
  handColor: Rgb,
  silverPalm: boolean,
  silverColor: Rgb = [194, 202, 210]
): Uint8ClampedArray {
  const frame = makeFrame([224, 228, 232]);
  paintRect(frame, 48, 52, 83, 83, handColor);
  for (const [from, to] of CONNECTIONS) {
    paintSegment(frame, POINTS[from], POINTS[to], 6, handColor);
  }

  if (silverPalm) {
    for (const [from, to] of [
      [2, 4],
      [6, 8],
      [10, 12],
      [14, 16],
      [18, 20],
    ] as const) {
      paintSegment(frame, POINTS[from], POINTS[to], 2, silverColor, 0.18, 0.78);
    }
  }
  return frame;
}

function makeLightHandFrame(
  points: ReadonlyArray<readonly [number, number]> = POINTS,
  surface: "rough" | "uniform" = "uniform"
): { frame: Uint8ClampedArray; landmarks: GloveVisionLandmark[] } {
  const frame = makeFrame([188, 194, 200]);
  const handColor: Rgb = [244, 247, 250];
  for (const [from, to] of CONNECTIONS) {
    paintSegment(frame, points[from], points[to], 7, handColor);
  }
  if (surface === "rough") {
    const seamColor: Rgb = [210, 218, 224];
    for (const [from, to] of CONNECTIONS) {
      paintSegment(frame, points[from], points[to], 1, seamColor, 0.12, 0.88);
    }
    for (const index of [0, 5, 9, 13, 17]) {
      paintDisk(frame, points[index][0], points[index][1], 2, [226, 232, 237]);
    }
  }
  return {
    frame,
    landmarks: points.map(([x, y]) => ({
      x: x / (WIDTH - 1),
      y: y / (HEIGHT - 1),
      z: 0,
    })),
  };
}

function makeDarkFrame(): Uint8ClampedArray {
  return makeFrame([224, 228, 232]);
}

function paintCompactGlove(
  frame: Uint8ClampedArray,
  centerX: number,
  centerY: number,
  color: Rgb = [22, 26, 30]
): void {
  paintDisk(frame, centerX, centerY + 7, 12, color);
  paintRect(frame, centerX - 9, centerY + 5, centerX + 9, centerY + 20, color);
  for (const offset of [-8, 0, 8]) {
    paintSegment(
      frame,
      [centerX + offset, centerY + 2],
      [centerX + offset, centerY - 17],
      3,
      color
    );
  }
  paintSegment(
    frame,
    [centerX - 8, centerY + 7],
    [centerX - 17, centerY],
    3,
    color
  );
}

function expectValidRegion(region: GloveVisionRegion): void {
  expect(Number.isInteger(region.x)).toBe(true);
  expect(Number.isInteger(region.y)).toBe(true);
  expect(Number.isInteger(region.width)).toBe(true);
  expect(Number.isInteger(region.height)).toBe(true);
  expect(region.x).toBeGreaterThanOrEqual(0);
  expect(region.y).toBeGreaterThanOrEqual(0);
  expect(region.width).toBeGreaterThan(0);
  expect(region.height).toBeGreaterThan(0);
  expect(region.x + region.width).toBeLessThanOrEqual(WIDTH);
  expect(region.y + region.height).toBeLessThanOrEqual(HEIGHT);
  expect(Math.abs(region.width - region.height)).toBeLessThanOrEqual(1);
  expect(region.area).toBeGreaterThan(0);
}

function pixelAt(frame: Uint8ClampedArray, x: number, y: number): Rgb {
  const offset = (y * WIDTH + x) * 4;
  return [frame[offset], frame[offset + 1], frame[offset + 2]];
}

describe("classifyGloveSurface", () => {
  it("classifies a dark glove with silver finger inserts as palm", () => {
    const frame = makeHandFrame([23, 27, 31], true);
    const result = classifyGloveSurface(frame, WIDTH, HEIGHT, LANDMARKS);

    expect(result.isGlove).toBe(true);
    expect(result.surface).toBe("palm");
    expect(result.silverRatio).toBeGreaterThan(0.085);
    expect(result.darkRatio).toBeGreaterThan(0.2);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it("classifies a plain black glove as back", () => {
    const frame = makeHandFrame([21, 24, 29], false);
    const result = classifyGloveSurface(frame, WIDTH, HEIGHT, LANDMARKS);

    expect(result.isGlove).toBe(true);
    expect(result.surface).toBe("back");
    expect(result.darkRatio).toBeGreaterThan(0.8);
    expect(result.silverRatio).toBeLessThan(0.01);
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it("recognizes the silver palm under dim lighting", () => {
    const frame = makeHandFrame([18, 22, 27], true, [118, 126, 134]);
    const result = classifyGloveSurface(frame, WIDTH, HEIGHT, LANDMARKS);

    expect(result.isGlove).toBe(true);
    expect(result.surface).toBe("palm");
    expect(result.confidence).toBeGreaterThan(0.5);
  });

  it("keeps a warm-lit black glove distinct from skin", () => {
    const frame = makeHandFrame([35, 30, 27], false);
    const surface = classifyGloveSurface(frame, WIDTH, HEIGHT, LANDMARKS);
    const enhanced = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);

    expect(surface.isGlove).toBe(true);
    expect(surface.surface).toBe("back");
    expect(enhanced.found).toBe(true);
  });

  it("combines rough palm texture and uniform back texture with hand geometry", () => {
    const mirroredPoints = POINTS.map(([x, y]) => [WIDTH - 1 - x, y] as const);
    const rightPalmFrame = makeLightHandFrame(mirroredPoints, "rough");
    const rightBackFrame = makeLightHandFrame(POINTS, "uniform");
    const leftPalmFrame = makeLightHandFrame(POINTS, "rough");
    const leftBackFrame = makeLightHandFrame(mirroredPoints, "uniform");

    const classify = (
      sample: ReturnType<typeof makeLightHandFrame>,
      handedness: "Left" | "Right"
    ) =>
      classifyGloveSurface(
        sample.frame,
        WIDTH,
        HEIGHT,
        sample.landmarks,
        handedness
      );
    const rightPalm = classify(rightPalmFrame, "Right");
    const rightBack = classify(rightBackFrame, "Right");
    const leftPalm = classify(leftPalmFrame, "Left");
    const leftBack = classify(leftBackFrame, "Left");

    expect(rightPalm).toMatchObject({ isGlove: true, surface: "palm" });
    expect(rightPalm.lightRatio).toBeGreaterThan(0.75);
    expect(rightPalm.confidence).toBeGreaterThan(0.5);
    expect(rightBack.surface).toBe("back");
    expect(leftPalm.surface).toBe("palm");
    expect(leftBack.surface).toBe("back");
  });

  it("uses decisive glove texture when handedness metadata briefly flips", () => {
    const roughPalm = makeLightHandFrame(POINTS, "rough");
    const uniformBack = makeLightHandFrame(POINTS, "uniform");
    const palmWithWrongGeometry = classifyGloveSurface(
      roughPalm.frame,
      WIDTH,
      HEIGHT,
      roughPalm.landmarks,
      "Right"
    );
    const backWithWrongGeometry = classifyGloveSurface(
      uniformBack.frame,
      WIDTH,
      HEIGHT,
      uniformBack.landmarks,
      "Left"
    );

    expect(palmWithWrongGeometry.surface).toBe("palm");
    expect(backWithWrongGeometry.surface).toBe("back");
  });

  it("keeps a near-edge-on white hand orientation unknown", () => {
    const sidePoints = POINTS.map(point => [...point] as [number, number]);
    sidePoints[5] = [60, 70];
    sidePoints[17] = [58, 64];
    const side = makeLightHandFrame(sidePoints);

    const result = classifyGloveSurface(
      side.frame,
      WIDTH,
      HEIGHT,
      side.landmarks,
      "Right"
    );

    expect(result.isGlove).toBe(true);
    expect(result.surface).toBe("unknown");
    expect(result.confidence).toBe(0);
  });

  it("does not treat skin or an empty bright frame as a glove", () => {
    for (const skinColor of [
      [205, 146, 112],
      [92, 58, 43],
      [75, 50, 40],
      [42, 29, 23],
    ] as const) {
      const skin = makeHandFrame(skinColor, false);
      const skinResult = classifyGloveSurface(skin, WIDTH, HEIGHT, LANDMARKS);
      expect(skinResult).toMatchObject({
        surface: "unknown",
        isGlove: false,
      });
    }

    const empty = makeFrame([238, 238, 238]);
    const emptyResult = classifyGloveSurface(empty, WIDTH, HEIGHT, LANDMARKS);
    expect(emptyResult).toMatchObject({ surface: "unknown", isGlove: false });
  });

  it("rejects incomplete or non-finite landmark sets", () => {
    const frame = makeFrame([238, 238, 238]);
    expect(
      classifyGloveSurface(frame, WIDTH, HEIGHT, LANDMARKS.slice(0, 20))
    ).toMatchObject({ surface: "unknown", isGlove: false, confidence: 0 });

    const invalid = LANDMARKS.map(landmark => ({ ...landmark }));
    invalid[8].x = Number.NaN;
    expect(classifyGloveSurface(frame, WIDTH, HEIGHT, invalid)).toMatchObject({
      surface: "unknown",
      isGlove: false,
      confidence: 0,
    });
  });
});

describe("enhanceDarkGloveFrame", () => {
  it("finds and recolors a connected dark hand while preserving the input", () => {
    const frame = makeHandFrame([23, 27, 31], true);
    const snapshot = new Uint8ClampedArray(frame);
    const originalDark = pixelAt(frame, 64, 80);
    const originalSilver = pixelAt(frame, 61, 27);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);

    expect(result.found).toBe(true);
    expect(result.pixels).not.toBe(frame);
    expect(frame).toEqual(snapshot);
    expect(result.threshold).toBeGreaterThanOrEqual(32);
    expect(result.threshold).toBeLessThanOrEqual(112);
    expect(result.maskCoverage).toBeGreaterThan(0.04);
    expect(result.maskCoverage).toBeLessThan(0.5);
    expect(result.regions).toHaveLength(1);
    expectValidRegion(result.regions[0]);

    const enhancedDark = pixelAt(result.pixels, 64, 80);
    expect(originalDark[0]).toBeLessThan(40);
    expect(enhancedDark[0]).toBeGreaterThan(enhancedDark[1]);
    expect(enhancedDark[1]).toBeGreaterThan(enhancedDark[2]);
    expect(enhancedDark[0]).toBeGreaterThan(170);

    const enhancedSilver = pixelAt(result.pixels, 61, 27);
    expect(originalSilver[2] - originalSilver[0]).toBeLessThan(25);
    expect(enhancedSilver[0]).toBeGreaterThan(enhancedSilver[1]);
    expect(enhancedSilver[1]).toBeGreaterThan(enhancedSilver[2]);
  });

  it("keeps one ROI per fist when open silver bands cross both silhouettes", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    const silver: Rgb = [190, 198, 206];
    paintDisk(frame, 35, 48, 20, dark);
    paintDisk(frame, 96, 48, 14, dark);
    paintRect(frame, 15, 47, 55, 49, silver);
    paintRect(frame, 82, 47, 110, 49, silver);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(2);
    const regions = [...result.regions].sort((a, b) => a.x - b.x);
    expect(regions[0].x).toBeLessThan(35);
    expect(regions[0].x + regions[0].width).toBeGreaterThan(35);
    expect(regions[1].x).toBeLessThan(96);
    expect(regions[1].x + regions[1].width).toBeGreaterThan(96);

    const enhancedSilver = pixelAt(result.pixels, 35, 48);
    expect(enhancedSilver[0]).toBeGreaterThan(enhancedSilver[1]);
    expect(enhancedSilver[1]).toBeGreaterThan(enhancedSilver[2]);
  });

  it("does not bridge ordinary bright gaps between connected fingers", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintRect(frame, 48, 52, 79, 80, dark);
    paintRect(frame, 51, 28, 57, 55, dark);
    paintRect(frame, 61, 28, 67, 55, dark);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    const enhancedGap = pixelAt(result.pixels, 59, 40);
    expect(enhancedGap[0]).toBeLessThan(enhancedGap[1]);
    expect(enhancedGap[1]).toBeLessThan(enhancedGap[2]);
  });

  it("does not bridge the narrow bright gap between separate gloves", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintDisk(frame, 47, 48, 15, dark);
    paintDisk(frame, 80, 48, 15, dark);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(2);
    const regions = [...result.regions].sort((a, b) => a.x - b.x);
    expect(regions[0].x + regions[0].width).toBeGreaterThan(47);
    expect(regions[1].x).toBeLessThan(80);
    const enhancedGap = pixelAt(result.pixels, 63, 48);
    expect(enhancedGap[0]).toBeLessThan(enhancedGap[1]);
    expect(enhancedGap[1]).toBeLessThan(enhancedGap[2]);
  });

  it("returns an unchanged copy when no suitable dark component exists", () => {
    const frame = makeFrame([236, 238, 240]);
    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);

    expect(result).toMatchObject({ found: false, maskCoverage: 0 });
    expect(result.regions).toEqual([]);
    expect(result.pixels).not.toBe(frame);
    expect(result.pixels).toEqual(frame);
  });

  it("returns one padded square ROI for each independent glove component", () => {
    const frame = makeDarkFrame();
    paintCompactGlove(frame, 31, 51);
    paintCompactGlove(frame, 96, 51);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);
    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(2);
    result.regions.forEach(expectValidRegion);

    const regions = [...result.regions].sort((a, b) => a.x - b.x);
    expect(regions[0].x).toBeLessThan(31);
    expect(regions[0].x + regions[0].width).toBeGreaterThan(31);
    expect(regions[1].x).toBeLessThan(96);
    expect(regions[1].x + regions[1].width).toBeGreaterThan(96);
  });

  it("ranks compact gloves above a larger border-spanning dark background", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintRect(frame, 0, 0, 4, HEIGHT - 1, dark);
    paintRect(frame, 8, 0, 12, HEIGHT - 1, dark);
    paintRect(frame, 16, 0, 20, HEIGHT - 1, dark);
    paintRect(frame, 0, 43, 20, 51, dark);
    paintCompactGlove(frame, 55, 51);
    paintCompactGlove(frame, 103, 51);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);
    expect(result.regions).toHaveLength(2);
    const centers = result.regions
      .map(region => region.x + region.width * 0.5)
      .sort((a, b) => a - b);
    expect(centers[0]).toBeGreaterThan(38);
    expect(centers[0]).toBeLessThan(72);
    expect(centers[1]).toBeGreaterThan(84);
  });

  it("keeps a real glove that touches one frame edge", () => {
    const frame = makeDarkFrame();
    paintCompactGlove(frame, 17, 50);
    paintCompactGlove(frame, 93, 50);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);
    expect(result.regions).toHaveLength(2);
    const regions = [...result.regions].sort((a, b) => a.x - b.x);
    expect(regions[0].x).toBe(0);
    expect(regions[0].x + regions[0].width).toBeGreaterThan(17);
    expect(regions[1].x).toBeLessThan(93);
  });

  it("uses a historical ROI hint to retain a smaller side-on glove", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintDisk(frame, 35, 50, 18, dark);
    paintDisk(frame, 102, 50, 8, dark);

    const withoutHint = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT, 1);
    expect(
      withoutHint.regions[0].x + withoutHint.regions[0].width * 0.5
    ).toBeLessThan(60);

    const withHint = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT, 1, [
      { x: 89, y: 37, width: 26, height: 26, area: 0 },
    ]);
    expect(withHint.regions).toHaveLength(1);
    const hintedCenter =
      withHint.regions[0].x + withHint.regions[0].width * 0.5;
    expect(hintedCenter).toBeGreaterThan(85);
  });

  it("recolors disconnected dark glove fragments inside a historical ROI", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintDisk(frame, 32, 50, 18, dark);
    paintDisk(frame, 91, 50, 5, dark);
    paintDisk(frame, 108, 50, 5, dark);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT, 1, [
      { x: 80, y: 38, width: 40, height: 24, area: 0 },
    ]);
    for (const x of [91, 108]) {
      const enhanced = pixelAt(result.pixels, x, 50);
      expect(enhanced[0]).toBeGreaterThan(enhanced[1]);
      expect(enhanced[1]).toBeGreaterThan(enhanced[2]);
      expect(enhanced[0]).toBeGreaterThan(170);
    }
  });

  it("splits a conservatively large horizontal merged component into overlapping ROIs", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintDisk(frame, 34, 48, 15, dark);
    paintDisk(frame, 94, 48, 15, dark);
    paintRect(frame, 34, 45, 94, 51, dark);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);
    expect(result.regions).toHaveLength(3);
    result.regions.forEach(expectValidRegion);

    const [whole, ...splitCandidates] = result.regions;
    expect(whole.x).toBeLessThan(34);
    expect(whole.x + whole.width).toBeGreaterThan(94);
    const regions = splitCandidates.sort((a, b) => a.x - b.x);
    expect(regions[0].x + regions[0].width).toBeGreaterThan(regions[1].x);
    expect(regions[0].width).toBeLessThan(WIDTH);
    expect(regions[1].width).toBeLessThan(WIDTH);
  });

  it("splits a conservatively large vertical merged component into overlapping ROIs", () => {
    const frame = makeDarkFrame();
    const dark: Rgb = [22, 26, 30];
    paintDisk(frame, 64, 17, 13, dark);
    paintDisk(frame, 64, 79, 13, dark);
    paintRect(frame, 61, 17, 67, 79, dark);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);
    expect(result.regions).toHaveLength(3);
    result.regions.forEach(expectValidRegion);

    const [whole, ...splitCandidates] = result.regions;
    expect(whole.y).toBeLessThan(17);
    expect(whole.y + whole.height).toBeGreaterThan(79);
    const regions = splitCandidates.sort((a, b) => a.y - b.y);
    expect(regions[0].y + regions[0].height).toBeGreaterThan(regions[1].y);
    expect(regions[0].height).toBeLessThan(HEIGHT);
    expect(regions[1].height).toBeLessThan(HEIGHT);
  });

  it("does not split an ordinary single-hand component", () => {
    const frame = makeDarkFrame();
    paintCompactGlove(frame, 64, 48);

    const result = enhanceDarkGloveFrame(frame, WIDTH, HEIGHT);
    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    expectValidRegion(result.regions[0]);
  });

  it("validates RGBA array dimensions", () => {
    expect(() => enhanceDarkGloveFrame(new Uint8ClampedArray(7), 2, 1)).toThrow(
      RangeError
    );
    expect(() =>
      classifyGloveSurface(new Uint8ClampedArray(7), 2, 1, LANDMARKS)
    ).toThrow(RangeError);
  });
});

describe("enhanceLightGloveFrame", () => {
  it("finds a satin glove on a warm background without mutating input", () => {
    const frame = makeFrame([112, 75, 58]);
    paintCompactGlove(frame, 64, 48, [230, 224, 209]);
    const snapshot = new Uint8ClampedArray(frame);

    const result = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result.found).toBe(true);
    expect(frame).toEqual(snapshot);
    expect(result.threshold).toBeGreaterThanOrEqual(138);
    expect(result.threshold).toBeLessThanOrEqual(224);
    expect(result.maskCoverage).toBeGreaterThan(0.03);
    expect(result.maskCoverage).toBeLessThan(0.4);
    expect(result.regions).toHaveLength(1);
    expectValidRegion(result.regions[0]);
    const glove = pixelAt(result.pixels, 64, 55);
    expect(glove[0]).toBeGreaterThan(glove[1]);
    expect(glove[1]).toBeGreaterThan(glove[2]);
    expect(pixelAt(result.pixels, 5, 5)[0]).toBeLessThan(100);
  });

  it("preserves a sharpened fold without splitting the hand ROI", () => {
    const frame = makeFrame([105, 72, 55]);
    paintCompactGlove(frame, 64, 48, [224, 220, 208]);
    paintSegment(frame, [55, 49], [55, 31], 1, [252, 249, 238]);
    paintRect(frame, 64, 36, 64, 66, [180, 177, 169]);
    const result = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    const outputContrast =
      pixelAt(result.pixels, 63, 54)[0] - pixelAt(result.pixels, 64, 54)[0];
    const toneOnlyContrast =
      Math.round((224 - result.threshold) * 0.12) -
      Math.round((180 - result.threshold) * 0.12);
    expect(outputContrast).toBeGreaterThan(toneOnlyContrast + 20);
  });

  it("rejects a uniform white scene during cold start", () => {
    const frame = makeFrame([238, 238, 238]);
    const result = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result).toMatchObject({ found: false, maskCoverage: 0 });
    expect(result.regions).toEqual([]);
    expect(result.pixels).toEqual(frame);
  });

  it("uses a historical ROI when a white glove merges into a white scene", () => {
    const frame = makeFrame([232, 232, 232]);
    paintRect(frame, 64, 20, 64, 75, [207, 207, 207]);
    const hint = { x: 35, y: 12, width: 72, height: 72, area: 0 };

    const cold = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 1);
    const result = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 1, [hint]);

    expect(cold.found).toBe(false);
    expect(result).toMatchObject({ found: true, maskCoverage: 0 });
    expect(result.regions).toHaveLength(1);
    expectValidRegion(result.regions[0]);
    expect(pixelAt(result.pixels, 64, 45)).not.toEqual(pixelAt(frame, 64, 45));
  });

  it("returns separate ROIs for two light gloves", () => {
    const frame = makeFrame([108, 72, 54]);
    paintCompactGlove(frame, 31, 50, [228, 222, 207]);
    paintCompactGlove(frame, 96, 50, [235, 230, 215]);

    const result = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 2);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(2);
    result.regions.forEach(expectValidRegion);
  });

  it("keeps the dark-glove path in automatic mode", () => {
    const frame = makeDarkFrame();
    paintCompactGlove(frame, 64, 48);

    const result = enhanceGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    const glove = pixelAt(result.pixels, 64, 55);
    expect(glove[0]).toBeGreaterThan(glove[1]);
    expect(glove[1]).toBeGreaterThan(glove[2]);
  });

  it("ranks a light hand above a disconnected white device frame", () => {
    const frame = makeFrame([108, 72, 54]);
    paintCompactGlove(frame, 38, 54, [230, 224, 210]);
    const device: Rgb = [242, 242, 237];
    paintRect(frame, 75, 5, 122, 9, device);
    paintRect(frame, 75, 9, 79, 43, device);
    paintRect(frame, 118, 9, 122, 43, device);
    paintRect(frame, 75, 39, 122, 43, device);

    const result = enhanceLightGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    expect(result.regions[0].x + result.regions[0].width * 0.5).toBeLessThan(
      65
    );
  });

  it("keeps a dark hand ahead of a white device in automatic mode", () => {
    const frame = makeFrame([120, 78, 58]);
    paintCompactGlove(frame, 40, 54);
    const device: Rgb = [242, 242, 237];
    paintRect(frame, 75, 5, 122, 9, device);
    paintRect(frame, 75, 9, 79, 43, device);
    paintRect(frame, 118, 9, 122, 43, device);
    paintRect(frame, 75, 39, 122, 43, device);

    const result = enhanceGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    expect(result.regions[0].x + result.regions[0].width * 0.5).toBeLessThan(
      65
    );
    const glove = pixelAt(result.pixels, 40, 61);
    expect(glove[0]).toBeGreaterThan(glove[1]);
    expect(glove[1]).toBeGreaterThan(glove[2]);
  });

  it("keeps a light hand ahead of a dark device in automatic mode", () => {
    const frame = makeFrame([112, 75, 58]);
    paintCompactGlove(frame, 38, 54, [230, 224, 210]);
    const device: Rgb = [18, 21, 24];
    paintRect(frame, 75, 5, 122, 9, device);
    paintRect(frame, 75, 9, 79, 43, device);
    paintRect(frame, 118, 9, 122, 43, device);
    paintRect(frame, 75, 39, 122, 43, device);

    const result = enhanceGloveFrame(frame, WIDTH, HEIGHT, 1);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(1);
    expect(result.regions[0].x + result.regions[0].width * 0.5).toBeLessThan(
      65
    );
    const glove = pixelAt(result.pixels, 38, 61);
    expect(glove[0]).toBeGreaterThan(glove[1]);
    expect(glove[1]).toBeGreaterThan(glove[2]);
  });

  it("merges one light and one dark hand in automatic mode", () => {
    const frame = makeFrame([112, 75, 58]);
    paintCompactGlove(frame, 31, 52, [230, 224, 210]);
    paintCompactGlove(frame, 96, 52);

    const result = enhanceGloveFrame(frame, WIDTH, HEIGHT, 2);

    expect(result.found).toBe(true);
    expect(result.regions).toHaveLength(2);
    const centers = result.regions
      .map(region => region.x + region.width * 0.5)
      .sort((first, second) => first - second);
    expect(centers[0]).toBeLessThan(64);
    expect(centers[1]).toBeGreaterThan(64);
    for (const point of [
      [31, 59],
      [96, 59],
    ] as const) {
      const glove = pixelAt(result.pixels, point[0], point[1]);
      expect(glove[0]).toBeGreaterThan(glove[1]);
      expect(glove[1]).toBeGreaterThan(glove[2]);
    }
  });
});
