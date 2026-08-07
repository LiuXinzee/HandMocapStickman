export interface GloveVisionLandmark {
  x: number;
  y: number;
  z?: number;
}

export interface GloveVisionRegion {
  /** Top-left pixel coordinate in the source frame. */
  x: number;
  y: number;
  width: number;
  height: number;
  /** Pixel area of the connected component that produced this ROI. */
  area: number;
}

export interface EnhancedGloveFrame {
  /** Enhanced RGBA pixels. The input array is never modified. */
  pixels: Uint8ClampedArray;
  found: boolean;
  /** Fraction of the frame covered by selected glove components and their holes. */
  maskCoverage: number;
  /** Adaptive luminance threshold used by the selected dark/light mask. */
  threshold: number;
  /** Padded, near-square source-frame ROIs suitable for MediaPipe retries. */
  regions: GloveVisionRegion[];
  /** Internal ranking score for choosing between light and dark candidates. */
  candidateScore?: number;
  /** Per-region scores used when automatic mode merges light and dark hands. */
  candidateScores?: number[];
}

export type GloveSurface = "palm" | "back" | "unknown";

export interface GloveSurfaceClassification {
  surface: GloveSurface;
  isGlove: boolean;
  silverRatio: number;
  darkRatio: number;
  confidence: number;
}

const MAX_COMPONENTS = 2;
const SKIN_R = 205;
const SKIN_G = 148;
const SKIN_B = 113;

// Pairs use the MediaPipe Hands 21-landmark topology.
const SAMPLE_CONNECTIONS = new Uint8Array([
  0, 1, 1, 2, 2, 3, 3, 4, 0, 5, 5, 6, 6, 7, 7, 8, 0, 9, 9, 10, 10, 11, 11, 12,
  0, 13, 13, 14, 14, 15, 15, 16, 0, 17, 17, 18, 18, 19, 19, 20, 5, 9, 9, 13, 13,
  17,
]);

const BRIDGE_DIRECTIONS = new Int8Array([1, 0, 0, 1, 1, 1, 1, -1]);

const UNKNOWN_SURFACE: GloveSurfaceClassification = {
  surface: "unknown",
  isGlove: false,
  silverRatio: 0,
  darkRatio: 0,
  confidence: 0,
};

function validateFrame(
  data: Uint8ClampedArray,
  width: number,
  height: number
): number {
  if (!(data instanceof Uint8ClampedArray)) {
    throw new TypeError("data must be a Uint8ClampedArray");
  }
  if (!Number.isSafeInteger(width) || width <= 0) {
    throw new RangeError("width must be a positive integer");
  }
  if (!Number.isSafeInteger(height) || height <= 0) {
    throw new RangeError("height must be a positive integer");
  }

  const pixelCount = width * height;
  if (!Number.isSafeInteger(pixelCount) || pixelCount > 0x7fffffff) {
    throw new RangeError("frame dimensions are too large");
  }
  if (data.length !== pixelCount * 4) {
    throw new RangeError(
      `RGBA data length must be ${pixelCount * 4}, received ${data.length}`
    );
  }
  return pixelCount;
}

function luminance(r: number, g: number, b: number): number {
  return (77 * r + 150 * g + 29 * b) >> 8;
}

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function histogramPercentile(
  histogram: Uint32Array,
  total: number,
  ratio: number
): number {
  if (total <= 0) return 0;

  const target = Math.max(1, Math.ceil(total * ratio));
  let cumulative = 0;
  for (let i = 0; i < 256; i++) {
    cumulative += histogram[i];
    if (cumulative >= target) return i;
  }
  return 255;
}

function adaptiveDarkThreshold(histogram: Uint32Array, total: number): number {
  if (total <= 0) return 0;

  let weightedSum = 0;
  for (let i = 0; i < 256; i++) weightedSum += i * histogram[i];

  let backgroundWeight = 0;
  let backgroundSum = 0;
  let bestVariance = -1;
  let otsuThreshold = 0;

  for (let i = 0; i < 255; i++) {
    backgroundWeight += histogram[i];
    if (backgroundWeight === 0) continue;

    const foregroundWeight = total - backgroundWeight;
    if (foregroundWeight === 0) break;

    backgroundSum += i * histogram[i];
    const backgroundMean = backgroundSum / backgroundWeight;
    const foregroundMean = (weightedSum - backgroundSum) / foregroundWeight;
    const delta = backgroundMean - foregroundMean;
    const variance = backgroundWeight * foregroundWeight * delta * delta;
    if (variance > bestVariance) {
      bestVariance = variance;
      otsuThreshold = i;
    }
  }

  const percentile35 = histogramPercentile(histogram, total, 0.35);

  // Otsu isolates a dark mode; the percentile cap keeps uniform frames from
  // turning into a full-frame component.
  return Math.round(
    clamp(Math.min(otsuThreshold + 18, percentile35 + 8), 32, 112)
  );
}

function isLowChromaDark(
  r: number,
  g: number,
  b: number,
  threshold: number
): boolean {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const chroma = max - min;
  const light = luminance(r, g, b);
  const warmSkinTone =
    r - b >= 8 &&
    r > g * 1.12 &&
    g > b * 1.06 &&
    chroma / Math.max(max, 1) >= 0.3;
  return (
    light <= threshold &&
    chroma <= 48 &&
    chroma * 2 <= max + 20 &&
    !warmSkinTone
  );
}

function adaptiveLightThreshold(histogram: Uint32Array, total: number): number {
  if (total <= 0) return 255;

  const median = histogramPercentile(histogram, total, 0.5);
  const percentile72 = histogramPercentile(histogram, total, 0.72);
  return Math.round(clamp(Math.max(median + 18, percentile72 - 18), 138, 224));
}

function isLowChromaLight(
  r: number,
  g: number,
  b: number,
  threshold: number
): boolean {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const chroma = max - min;
  const light = luminance(r, g, b);
  const warmSkinTone =
    r - b >= 18 &&
    r > g * 1.07 &&
    g > b * 1.04 &&
    chroma / Math.max(max, 1) >= 0.13;
  return (
    light >= threshold &&
    min >= 105 &&
    chroma <= 54 &&
    chroma * 4 <= max + 36 &&
    !warmSkinTone
  );
}

function bridgeSinglePixelLightGaps(
  labels: Int32Array,
  width: number,
  height: number
): void {
  if (width < 3 || height < 3) return;

  const additions = new Uint8Array(labels.length);
  for (let y = 1; y + 1 < height; y++) {
    let pixel = y * width + 1;
    for (let x = 1; x + 1 < width; x++, pixel++) {
      if (labels[pixel] === -1) continue;
      const horizontal = labels[pixel - 1] === -1 && labels[pixel + 1] === -1;
      const vertical =
        labels[pixel - width] === -1 && labels[pixel + width] === -1;
      const diagonal =
        (labels[pixel - width - 1] === -1 &&
          labels[pixel + width + 1] === -1) ||
        (labels[pixel - width + 1] === -1 && labels[pixel + width - 1] === -1);
      if (horizontal || vertical || diagonal) additions[pixel] = 1;
    }
  }

  for (let pixel = 0; pixel < labels.length; pixel++) {
    if (additions[pixel] !== 0) labels[pixel] = -1;
  }
}

function bridgeNarrowNeutralGaps(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  labels: Int32Array,
  threshold: number,
  backgroundLuminance: number
): void {
  const maxGap = Math.round(clamp(Math.min(width, height) * 0.035, 2, 8));
  const maxInsertLuminance = backgroundLuminance - 8;
  if (maxInsertLuminance < Math.max(82, threshold + 18)) return;

  const additions = new Uint8Array(labels.length);
  const isNeutralInsert = (pixel: number) => {
    const offset = pixel * 4;
    if (data[offset + 3] < 16) return false;

    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const light = luminance(r, g, b);
    return (
      min >= 72 &&
      max - min <= 58 &&
      light >= threshold + 18 &&
      light <= maxInsertLuminance
    );
  };

  // Use only the original dark mask as endpoints. Applying all bridges after
  // the scan prevents a short insert from growing into a broad closing pass.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const start = y * width + x;
      if (labels[start] !== -1) continue;

      for (
        let direction = 0;
        direction < BRIDGE_DIRECTIONS.length;
        direction += 2
      ) {
        const dx = BRIDGE_DIRECTIONS[direction];
        const dy = BRIDGE_DIRECTIONS[direction + 1];
        let neutralCount = 0;
        let gapLength = 0;

        for (let step = 1; step <= maxGap + 1; step++) {
          const nextX = x + dx * step;
          const nextY = y + dy * step;
          if (nextX < 0 || nextX >= width || nextY < 0 || nextY >= height) {
            break;
          }

          const next = nextY * width + nextX;
          if (labels[next] === -1) {
            if (gapLength > 0 && neutralCount * 4 >= gapLength * 3) {
              for (let fillStep = 1; fillStep <= gapLength; fillStep++) {
                additions[(y + dy * fillStep) * width + x + dx * fillStep] = 1;
              }
            }
            break;
          }

          gapLength++;
          if (isNeutralInsert(next)) neutralCount++;
        }
      }
    }
  }

  for (let pixel = 0; pixel < labels.length; pixel++) {
    if (additions[pixel] !== 0) labels[pixel] = -1;
  }
}

function makePaddedSquareRegion(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  area: number,
  frameWidth: number,
  frameHeight: number
): GloveVisionRegion {
  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;
  const paddedSide = Math.ceil(Math.max(boxWidth, boxHeight) * 1.34);
  const side = Math.max(1, Math.min(paddedSide, frameWidth, frameHeight));
  const centerX = (minX + maxX + 1) * 0.5;
  const centerY = (minY + maxY + 1) * 0.5;
  const x = Math.round(clamp(centerX - side * 0.5, 0, frameWidth - side));
  const y = Math.round(clamp(centerY - side * 0.5, 0, frameHeight - side));
  return { x, y, width: side, height: side, area };
}

interface ClippedHintRegion {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  area: number;
}

function clipHintRegions(
  regions: ReadonlyArray<GloveVisionRegion>,
  frameWidth: number,
  frameHeight: number
): ClippedHintRegion[] {
  const clipped: ClippedHintRegion[] = [];
  for (let index = 0; index < regions.length && clipped.length < 2; index++) {
    const region = regions[index];
    if (
      !region ||
      !Number.isFinite(region.x) ||
      !Number.isFinite(region.y) ||
      !Number.isFinite(region.width) ||
      !Number.isFinite(region.height) ||
      region.width <= 0 ||
      region.height <= 0
    ) {
      continue;
    }

    const minX = Math.floor(clamp(region.x, 0, frameWidth));
    const minY = Math.floor(clamp(region.y, 0, frameHeight));
    const maxX = Math.ceil(clamp(region.x + region.width, 0, frameWidth));
    const maxY = Math.ceil(clamp(region.y + region.height, 0, frameHeight));
    if (maxX <= minX || maxY <= minY) continue;
    clipped.push({
      minX,
      minY,
      maxX,
      maxY,
      area: (maxX - minX) * (maxY - minY),
    });
  }
  return clipped;
}

function componentCandidateScore(
  area: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  fillRatio: number,
  minArea: number,
  frameWidth: number,
  frameHeight: number,
  pixelCount: number,
  hintOverlaps: Int32Array,
  hints: ReadonlyArray<ClippedHintRegion>
): number {
  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;
  const widthRatio = boxWidth / frameWidth;
  const heightRatio = boxHeight / frameHeight;
  const boxAreaRatio = (boxWidth * boxHeight) / pixelCount;
  // Area remains the safest cold-start signal: patterned clothing produces
  // many compact dark specks. Temporal hints can still override it once a
  // glove has been observed.
  const areaScore = Math.min(Math.sqrt(area / minArea) * 1.25, 5);
  const compactnessScore =
    clamp(1 - Math.abs(fillRatio - 0.43) / 0.43, 0, 1) * 1.5;
  const centerX = ((minX + maxX + 1) * 0.5) / frameWidth;
  const centerY = ((minY + maxY + 1) * 0.5) / frameHeight;
  const centerScore =
    clamp(1 - Math.hypot(centerX - 0.5, centerY - 0.52) / 0.72, 0, 1) * 0.7;

  const touchesLeft = minX === 0;
  const touchesRight = maxX === frameWidth - 1;
  const touchesTop = minY === 0;
  const touchesBottom = maxY === frameHeight - 1;
  const borderTouches =
    Number(touchesLeft) +
    Number(touchesRight) +
    Number(touchesTop) +
    Number(touchesBottom);
  const spansOppositeBorders =
    (touchesLeft && touchesRight) || (touchesTop && touchesBottom);
  const borderPenalty = borderTouches * 0.32 + (spansOppositeBorders ? 3.2 : 0);
  const extentPenalty =
    Math.max(0, widthRatio - 0.72) * 7 +
    Math.max(0, heightRatio - 0.82) * 7 +
    Math.max(0, boxAreaRatio - 0.48) * 6;

  let hintAffinity = 0;
  for (let index = 0; index < hints.length; index++) {
    const overlap = hintOverlaps[index];
    if (overlap === 0) continue;
    const componentShare = overlap / area;
    const expectedDarkHintArea = Math.max(1, hints[index].area * 0.3);
    const hintCoverage = Math.min(1, overlap / expectedDarkHintArea);
    hintAffinity = Math.max(
      hintAffinity,
      Math.sqrt(componentShare * hintCoverage)
    );
  }

  return (
    areaScore +
    compactnessScore +
    centerScore +
    hintAffinity * 18 -
    borderPenalty -
    extentPenalty
  );
}

function lightComponentCandidateScore(
  area: number,
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  fillRatio: number,
  coreFillRatio: number,
  minArea: number,
  frameWidth: number,
  frameHeight: number,
  pixelCount: number,
  hintOverlaps: Int32Array,
  hints: ReadonlyArray<ClippedHintRegion>
): number {
  const baseScore = componentCandidateScore(
    area,
    minX,
    minY,
    maxX,
    maxY,
    fillRatio,
    minArea,
    frameWidth,
    frameHeight,
    pixelCount,
    hintOverlaps,
    hints
  );
  const boxWidth = maxX - minX + 1;
  const boxHeight = maxY - minY + 1;
  const aspectRatio = Math.max(boxWidth / boxHeight, boxHeight / boxWidth);
  const aspectPenalty = Math.max(0, aspectRatio - 1.9) * 2.8;
  const handFillBonus =
    clamp(1 - Math.abs(fillRatio - 0.42) / 0.34, 0, 1) * 1.2;
  const palmCoreBonus = clamp((coreFillRatio - 0.08) / 0.52, 0, 1) * 3;
  const hollowCorePenalty = Math.max(0, 0.12 - coreFillRatio) * 20;

  return (
    baseScore +
    handFillBonus -
    aspectPenalty +
    palmCoreBonus -
    hollowCorePenalty
  );
}

function buildGloveRegions(
  selectedIds: Int32Array,
  selectedAreas: Int32Array,
  selectedMinX: Int32Array,
  selectedMinY: Int32Array,
  selectedMaxX: Int32Array,
  selectedMaxY: Int32Array,
  frameWidth: number,
  frameHeight: number,
  pixelCount: number,
  minArea: number
): GloveVisionRegion[] {
  const hasSecondComponent = selectedIds.length > 1 && selectedIds[1] !== 0;

  // A bridge between two gloves can turn them into one component. Keep the
  // complete ROI for side-on single hands, then add two overlapping split
  // candidates when the component is large and elongated.
  if (selectedIds.length > 1 && !hasSecondComponent) {
    const minX = selectedMinX[0];
    const minY = selectedMinY[0];
    const maxX = selectedMaxX[0];
    const maxY = selectedMaxY[0];
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    const shortSide = Math.max(1, Math.min(boxWidth, boxHeight));
    const longSide = Math.max(boxWidth, boxHeight);
    const mergedAreaThreshold = Math.max(
      Math.floor(pixelCount * 0.035),
      minArea * 6
    );

    if (
      longSide / shortSide >= 1.75 &&
      selectedAreas[0] >= mergedAreaThreshold
    ) {
      const wholeRegion = makePaddedSquareRegion(
        minX,
        minY,
        maxX,
        maxY,
        selectedAreas[0],
        frameWidth,
        frameHeight
      );
      if (boxWidth > boxHeight) {
        const firstMaxX = minX + Math.ceil(boxWidth * 0.62) - 1;
        const secondMinX = minX + Math.floor(boxWidth * 0.38);
        return [
          wholeRegion,
          makePaddedSquareRegion(
            minX,
            minY,
            firstMaxX,
            maxY,
            selectedAreas[0],
            frameWidth,
            frameHeight
          ),
          makePaddedSquareRegion(
            secondMinX,
            minY,
            maxX,
            maxY,
            selectedAreas[0],
            frameWidth,
            frameHeight
          ),
        ];
      }

      const firstMaxY = minY + Math.ceil(boxHeight * 0.62) - 1;
      const secondMinY = minY + Math.floor(boxHeight * 0.38);
      return [
        wholeRegion,
        makePaddedSquareRegion(
          minX,
          minY,
          maxX,
          firstMaxY,
          selectedAreas[0],
          frameWidth,
          frameHeight
        ),
        makePaddedSquareRegion(
          minX,
          secondMinY,
          maxX,
          maxY,
          selectedAreas[0],
          frameWidth,
          frameHeight
        ),
      ];
    }
  }

  const regions: GloveVisionRegion[] = [];
  for (let slot = 0; slot < selectedIds.length; slot++) {
    if (selectedIds[slot] === 0) break;
    regions.push(
      makePaddedSquareRegion(
        selectedMinX[slot],
        selectedMinY[slot],
        selectedMaxX[slot],
        selectedMaxY[slot],
        selectedAreas[slot],
        frameWidth,
        frameHeight
      )
    );
  }
  return regions;
}

function buildRegionScores(
  regions: ReadonlyArray<GloveVisionRegion>,
  selectedIds: Int32Array,
  selectedScores: Float64Array
): number[] {
  let componentCount = 0;
  while (
    componentCount < selectedIds.length &&
    selectedIds[componentCount] !== 0
  ) {
    componentCount++;
  }
  if (componentCount === 0) return [];
  if (regions.length > componentCount) {
    return regions.map(() => selectedScores[0]);
  }
  return regions.map((_, index) => selectedScores[index]);
}

/**
 * Converts large, low-chroma dark hand-shaped regions into a skin-like image.
 * This is intended as a MediaPipe retry input when the original glove frame
 * is not detected. It does not mutate `data`.
 */
export function enhanceDarkGloveFrame(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  maxHands = 2,
  preferredRegions: ReadonlyArray<GloveVisionRegion> = []
): EnhancedGloveFrame {
  const pixelCount = validateFrame(data, width, height);
  if (!Number.isFinite(maxHands)) {
    throw new RangeError("maxHands must be finite");
  }

  const output = new Uint8ClampedArray(data);
  const hints = clipHintRegions(preferredRegions, width, height);
  const requestedComponents = Math.min(
    MAX_COMPONENTS,
    Math.max(0, Math.trunc(maxHands))
  );

  const histogram = new Uint32Array(256);
  let opaquePixelCount = 0;
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel++, offset += 4) {
    if (data[offset + 3] < 16) continue;
    histogram[luminance(data[offset], data[offset + 1], data[offset + 2])]++;
    opaquePixelCount++;
  }

  const threshold = adaptiveDarkThreshold(histogram, opaquePixelCount);
  if (requestedComponents === 0 || opaquePixelCount === 0) {
    return {
      pixels: output,
      found: false,
      maskCoverage: 0,
      threshold,
      regions: [],
    };
  }

  // -1 is an unvisited dark pixel, 0 is background, and positive values are
  // connected-component identifiers.
  const labels = new Int32Array(pixelCount);
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel++, offset += 4) {
    if (
      data[offset + 3] >= 16 &&
      isLowChromaDark(
        data[offset],
        data[offset + 1],
        data[offset + 2],
        threshold
      )
    ) {
      labels[pixel] = -1;
    }
  }

  bridgeNarrowNeutralGaps(
    data,
    width,
    height,
    labels,
    threshold,
    histogramPercentile(histogram, opaquePixelCount, 0.75)
  );

  const queue = new Int32Array(pixelCount);
  const selectedIds = new Int32Array(requestedComponents);
  const selectedAreas = new Int32Array(requestedComponents);
  const selectedMinX = new Int32Array(requestedComponents);
  const selectedMinY = new Int32Array(requestedComponents);
  const selectedMaxX = new Int32Array(requestedComponents);
  const selectedMaxY = new Int32Array(requestedComponents);
  const selectedScores = new Float64Array(requestedComponents);
  selectedScores.fill(Number.NEGATIVE_INFINITY);
  const minArea = Math.max(24, Math.floor(pixelCount * 0.002));
  const maxArea = Math.floor(pixelCount * 0.58);
  const minExtent = Math.max(4, Math.floor(Math.min(width, height) * 0.025));
  const hintOverlaps = new Int32Array(hints.length);
  let componentId = 0;

  for (let start = 0; start < pixelCount; start++) {
    if (labels[start] !== -1) continue;

    componentId++;
    let head = 0;
    let tail = 0;
    let area = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    hintOverlaps.fill(0);

    labels[start] = componentId;
    queue[tail++] = start;

    while (head < tail) {
      const pixel = queue[head++];
      const x = pixel % width;
      const y = (pixel / width) | 0;
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let hintIndex = 0; hintIndex < hints.length; hintIndex++) {
        const hint = hints[hintIndex];
        if (
          x >= hint.minX &&
          x < hint.maxX &&
          y >= hint.minY &&
          y < hint.maxY
        ) {
          hintOverlaps[hintIndex]++;
        }
      }

      const fromY = y > 0 ? y - 1 : y;
      const toY = y + 1 < height ? y + 1 : y;
      const fromX = x > 0 ? x - 1 : x;
      const toX = x + 1 < width ? x + 1 : x;
      for (let neighborY = fromY; neighborY <= toY; neighborY++) {
        let neighbor = neighborY * width + fromX;
        for (let neighborX = fromX; neighborX <= toX; neighborX++, neighbor++) {
          if (neighborX === x && neighborY === y) continue;
          if (labels[neighbor] === -1) {
            labels[neighbor] = componentId;
            queue[tail++] = neighbor;
          }
        }
      }
    }

    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    const fillRatio = area / (boxWidth * boxHeight);
    if (
      area < minArea ||
      area > maxArea ||
      boxWidth < minExtent ||
      boxHeight < minExtent ||
      fillRatio < 0.12 ||
      fillRatio > 0.95
    ) {
      continue;
    }

    const candidateScore = componentCandidateScore(
      area,
      minX,
      minY,
      maxX,
      maxY,
      fillRatio,
      minArea,
      width,
      height,
      pixelCount,
      hintOverlaps,
      hints
    );

    // Rank compact hand-like components above broad border-spanning regions.
    // Historical crop overlap is a strong hint, but area remains a tie-breaker.
    for (let slot = 0; slot < requestedComponents; slot++) {
      if (
        candidateScore < selectedScores[slot] ||
        (candidateScore === selectedScores[slot] && area <= selectedAreas[slot])
      ) {
        continue;
      }
      for (let move = requestedComponents - 1; move > slot; move--) {
        selectedScores[move] = selectedScores[move - 1];
        selectedAreas[move] = selectedAreas[move - 1];
        selectedIds[move] = selectedIds[move - 1];
        selectedMinX[move] = selectedMinX[move - 1];
        selectedMinY[move] = selectedMinY[move - 1];
        selectedMaxX[move] = selectedMaxX[move - 1];
        selectedMaxY[move] = selectedMaxY[move - 1];
      }
      selectedScores[slot] = candidateScore;
      selectedAreas[slot] = area;
      selectedIds[slot] = componentId;
      selectedMinX[slot] = minX;
      selectedMinY[slot] = minY;
      selectedMaxX[slot] = maxX;
      selectedMaxY[slot] = maxY;
      break;
    }
  }

  if (selectedIds[0] === 0) {
    return {
      pixels: output,
      found: false,
      maskCoverage: 0,
      threshold,
      regions: [],
    };
  }

  const regions = buildGloveRegions(
    selectedIds,
    selectedAreas,
    selectedMinX,
    selectedMinY,
    selectedMaxX,
    selectedMaxY,
    width,
    height,
    pixelCount,
    minArea
  );

  const firstId = selectedIds[0];
  const secondId = requestedComponents > 1 ? selectedIds[1] : 0;
  const selectedMask = new Uint8Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const label = labels[pixel];
    if (label === firstId || (secondId !== 0 && label === secondId)) {
      selectedMask[pixel] = 1;
    }
  }
  // A silver palm can split one glove into several dark islands. Once a
  // temporal hand crop exists, retain every dark island inside that crop so
  // the ROI retry receives a complete silhouette.
  for (const hint of hints) {
    for (let y = hint.minY; y < hint.maxY; y++) {
      let pixel = y * width + hint.minX;
      for (let x = hint.minX; x < hint.maxX; x++, pixel++) {
        if (labels[pixel] > 0) selectedMask[pixel] = 1;
      }
    }
  }
  const isSelected = (pixel: number) => selectedMask[pixel] !== 0;

  // Fade the background so the recolored glove dominates the retry frame.
  for (let offset = 0; offset < data.length; offset += 4) {
    output[offset] = (data[offset] * 3 + 238 * 5) >> 3;
    output[offset + 1] = (data[offset + 1] * 3 + 238 * 5) >> 3;
    output[offset + 2] = (data[offset + 2] * 3 + 238 * 5) >> 3;
  }

  let selectedPixelCount = 0;
  const colorAsSkin = (pixel: number) => {
    const offset = pixel * 4;
    const light = luminance(data[offset], data[offset + 1], data[offset + 2]);
    const detail = Math.round(clamp((light - threshold * 0.35) * 0.18, -8, 24));
    output[offset] = SKIN_R + detail;
    output[offset + 1] = SKIN_G + Math.round(detail * 0.75);
    output[offset + 2] = SKIN_B + Math.round(detail * 0.55);
  };

  for (let pixel = 0; pixel < pixelCount; pixel++) {
    if (!isSelected(pixel)) continue;
    colorAsSkin(pixel);
    selectedPixelCount++;
  }

  // Flood-fill all non-glove pixels reachable from the image boundary. Any
  // remaining compact bright region is a closed hole such as a silver insert.
  const outside = new Uint8Array(pixelCount);
  let head = 0;
  let tail = 0;
  const seedOutside = (pixel: number) => {
    if (!isSelected(pixel) && outside[pixel] === 0) {
      outside[pixel] = 1;
      queue[tail++] = pixel;
    }
  };

  for (let x = 0; x < width; x++) {
    seedOutside(x);
    if (height > 1) seedOutside((height - 1) * width + x);
  }
  for (let y = 1; y + 1 < height; y++) {
    seedOutside(y * width);
    if (width > 1) seedOutside(y * width + width - 1);
  }

  while (head < tail) {
    const pixel = queue[head++];
    const x = pixel % width;
    if (x > 0) {
      const next = pixel - 1;
      if (!isSelected(next) && outside[next] === 0) {
        outside[next] = 1;
        queue[tail++] = next;
      }
    }
    if (x + 1 < width) {
      const next = pixel + 1;
      if (!isSelected(next) && outside[next] === 0) {
        outside[next] = 1;
        queue[tail++] = next;
      }
    }
    if (pixel >= width) {
      const next = pixel - width;
      if (!isSelected(next) && outside[next] === 0) {
        outside[next] = 1;
        queue[tail++] = next;
      }
    }
    if (pixel + width < pixelCount) {
      const next = pixel + width;
      if (!isSelected(next) && outside[next] === 0) {
        outside[next] = 1;
        queue[tail++] = next;
      }
    }
  }

  const totalSelectedArea =
    selectedAreas[0] + (requestedComponents > 1 ? selectedAreas[1] : 0);
  const maxHoleArea = Math.max(20, Math.floor(totalSelectedArea * 0.14));

  for (let start = 0; start < pixelCount; start++) {
    if (isSelected(start) || outside[start] !== 0) continue;

    head = 0;
    tail = 0;
    let brightNeutral = 0;
    outside[start] = 2;
    queue[tail++] = start;

    while (head < tail) {
      const pixel = queue[head++];
      const offset = pixel * 4;
      const r = data[offset];
      const g = data[offset + 1];
      const b = data[offset + 2];
      const chroma = Math.max(r, g, b) - Math.min(r, g, b);
      if (luminance(r, g, b) >= threshold + 18 && chroma <= 58) {
        brightNeutral++;
      }

      const x = pixel % width;
      if (x > 0) {
        const next = pixel - 1;
        if (!isSelected(next) && outside[next] === 0) {
          outside[next] = 2;
          queue[tail++] = next;
        }
      }
      if (x + 1 < width) {
        const next = pixel + 1;
        if (!isSelected(next) && outside[next] === 0) {
          outside[next] = 2;
          queue[tail++] = next;
        }
      }
      if (pixel >= width) {
        const next = pixel - width;
        if (!isSelected(next) && outside[next] === 0) {
          outside[next] = 2;
          queue[tail++] = next;
        }
      }
      if (pixel + width < pixelCount) {
        const next = pixel + width;
        if (!isSelected(next) && outside[next] === 0) {
          outside[next] = 2;
          queue[tail++] = next;
        }
      }
    }

    if (tail <= maxHoleArea && brightNeutral / tail >= 0.55) {
      for (let i = 0; i < tail; i++) colorAsSkin(queue[i]);
      selectedPixelCount += tail;
    }
  }

  return {
    pixels: output,
    found: true,
    maskCoverage: selectedPixelCount / pixelCount,
    threshold,
    regions,
    candidateScore: selectedScores[0],
    candidateScores: buildRegionScores(regions, selectedIds, selectedScores),
  };
}

function makeHintSquareRegion(
  hint: ClippedHintRegion,
  frameWidth: number,
  frameHeight: number
): GloveVisionRegion {
  const hintWidth = hint.maxX - hint.minX;
  const hintHeight = hint.maxY - hint.minY;
  const side = Math.max(
    1,
    Math.min(Math.max(hintWidth, hintHeight), frameWidth, frameHeight)
  );
  const centerX = (hint.minX + hint.maxX) * 0.5;
  const centerY = (hint.minY + hint.maxY) * 0.5;
  return {
    x: Math.round(clamp(centerX - side * 0.5, 0, frameWidth - side)),
    y: Math.round(clamp(centerY - side * 0.5, 0, frameHeight - side)),
    width: side,
    height: side,
    area: hint.area,
  };
}

function enhanceHintRegions(
  data: Uint8ClampedArray,
  output: Uint8ClampedArray,
  width: number,
  height: number,
  hints: ReadonlyArray<ClippedHintRegion>
): void {
  const lightAt = (pixel: number) => {
    const offset = pixel * 4;
    return luminance(data[offset], data[offset + 1], data[offset + 2]);
  };

  for (const hint of hints) {
    let lightSum = 0;
    let sampleCount = 0;
    for (let y = hint.minY; y < hint.maxY; y += 2) {
      let pixel = y * width + hint.minX;
      for (let x = hint.minX; x < hint.maxX; x += 2, pixel += 2) {
        lightSum += lightAt(pixel);
        sampleCount++;
      }
    }
    const meanLight = sampleCount > 0 ? lightSum / sampleCount : 128;

    const minY = Math.max(1, hint.minY);
    const maxY = Math.min(height - 1, hint.maxY);
    const minX = Math.max(1, hint.minX);
    const maxX = Math.min(width - 1, hint.maxX);
    for (let y = minY; y < maxY; y++) {
      let pixel = y * width + minX;
      for (let x = minX; x < maxX; x++, pixel++) {
        const center = lightAt(pixel);
        const laplacian =
          center * 4 -
          lightAt(pixel - 1) -
          lightAt(pixel + 1) -
          lightAt(pixel - width) -
          lightAt(pixel + width);
        const delta = Math.round(
          clamp(laplacian * 0.2, -20, 20) +
            clamp((center - meanLight) * 0.12, -8, 8)
        );
        const offset = pixel * 4;
        output[offset] = clamp(data[offset] + delta, 0, 255);
        output[offset + 1] = clamp(data[offset + 1] + delta, 0, 255);
        output[offset + 2] = clamp(data[offset + 2] + delta, 0, 255);
      }
    }
  }
}

/**
 * Builds a skin-like MediaPipe retry image for bright, low-chroma gloves.
 * Local sharpening keeps folds and contours without producing a binary edge
 * image.
 */
export function enhanceLightGloveFrame(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  maxHands = 2,
  preferredRegions: ReadonlyArray<GloveVisionRegion> = []
): EnhancedGloveFrame {
  const pixelCount = validateFrame(data, width, height);
  if (!Number.isFinite(maxHands)) {
    throw new RangeError("maxHands must be finite");
  }

  const output = new Uint8ClampedArray(data);
  const hints = clipHintRegions(preferredRegions, width, height);
  const requestedComponents = Math.min(
    MAX_COMPONENTS,
    Math.max(0, Math.trunc(maxHands))
  );
  const histogram = new Uint32Array(256);
  let opaquePixelCount = 0;
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel++, offset += 4) {
    if (data[offset + 3] < 16) continue;
    histogram[luminance(data[offset], data[offset + 1], data[offset + 2])]++;
    opaquePixelCount++;
  }

  const threshold = adaptiveLightThreshold(histogram, opaquePixelCount);
  if (requestedComponents === 0 || opaquePixelCount === 0) {
    return {
      pixels: output,
      found: false,
      maskCoverage: 0,
      threshold,
      regions: [],
    };
  }

  const labels = new Int32Array(pixelCount);
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel++, offset += 4) {
    if (
      data[offset + 3] >= 16 &&
      isLowChromaLight(
        data[offset],
        data[offset + 1],
        data[offset + 2],
        threshold
      )
    ) {
      labels[pixel] = -1;
    }
  }
  bridgeSinglePixelLightGaps(labels, width, height);

  const queue = new Int32Array(pixelCount);
  const selectedIds = new Int32Array(requestedComponents);
  const selectedAreas = new Int32Array(requestedComponents);
  const selectedMinX = new Int32Array(requestedComponents);
  const selectedMinY = new Int32Array(requestedComponents);
  const selectedMaxX = new Int32Array(requestedComponents);
  const selectedMaxY = new Int32Array(requestedComponents);
  const selectedScores = new Float64Array(requestedComponents);
  selectedScores.fill(Number.NEGATIVE_INFINITY);
  const minArea = Math.max(24, Math.floor(pixelCount * 0.0025));
  const maxArea = Math.floor(pixelCount * 0.56);
  const minExtent = Math.max(4, Math.floor(Math.min(width, height) * 0.025));
  const hintOverlaps = new Int32Array(hints.length);
  let componentId = 0;
  for (let start = 0; start < pixelCount; start++) {
    if (labels[start] !== -1) continue;

    componentId++;
    let head = 0;
    let tail = 0;
    let area = 0;
    let minX = width;
    let maxX = 0;
    let minY = height;
    let maxY = 0;
    hintOverlaps.fill(0);
    labels[start] = componentId;
    queue[tail++] = start;

    while (head < tail) {
      const pixel = queue[head++];
      const x = pixel % width;
      const y = (pixel / width) | 0;
      area++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
      for (let hintIndex = 0; hintIndex < hints.length; hintIndex++) {
        const hint = hints[hintIndex];
        if (
          x >= hint.minX &&
          x < hint.maxX &&
          y >= hint.minY &&
          y < hint.maxY
        ) {
          hintOverlaps[hintIndex]++;
        }
      }

      const fromY = y > 0 ? y - 1 : y;
      const toY = y + 1 < height ? y + 1 : y;
      const fromX = x > 0 ? x - 1 : x;
      const toX = x + 1 < width ? x + 1 : x;
      for (let neighborY = fromY; neighborY <= toY; neighborY++) {
        let neighbor = neighborY * width + fromX;
        for (let neighborX = fromX; neighborX <= toX; neighborX++, neighbor++) {
          if (neighborX === x && neighborY === y) continue;
          if (labels[neighbor] === -1) {
            labels[neighbor] = componentId;
            queue[tail++] = neighbor;
          }
        }
      }
    }
    const boxWidth = maxX - minX + 1;
    const boxHeight = maxY - minY + 1;
    const fillRatio = area / (boxWidth * boxHeight);
    const spansOppositeBorders =
      (minX === 0 && maxX === width - 1) || (minY === 0 && maxY === height - 1);
    if (
      area < minArea ||
      area > maxArea ||
      boxWidth < minExtent ||
      boxHeight < minExtent ||
      fillRatio < 0.1 ||
      fillRatio > 0.88 ||
      spansOppositeBorders
    ) {
      continue;
    }

    const coreMinX = minX + Math.floor(boxWidth * 0.3);
    const coreMaxX = maxX - Math.floor(boxWidth * 0.3);
    const coreMinY = minY + Math.floor(boxHeight * 0.3);
    const coreMaxY = maxY - Math.floor(boxHeight * 0.3);
    let corePixels = 0;
    let coreArea = 0;
    for (let coreY = coreMinY; coreY <= coreMaxY; coreY++) {
      let corePixel = coreY * width + coreMinX;
      for (let coreX = coreMinX; coreX <= coreMaxX; coreX++, corePixel++) {
        coreArea++;
        if (labels[corePixel] === componentId) corePixels++;
      }
    }
    const coreFillRatio = corePixels / Math.max(coreArea, 1);
    const candidateScore = lightComponentCandidateScore(
      area,
      minX,
      minY,
      maxX,
      maxY,
      fillRatio,
      coreFillRatio,
      minArea,
      width,
      height,
      pixelCount,
      hintOverlaps,
      hints
    );
    for (let slot = 0; slot < requestedComponents; slot++) {
      if (
        candidateScore < selectedScores[slot] ||
        (candidateScore === selectedScores[slot] && area <= selectedAreas[slot])
      ) {
        continue;
      }
      for (let move = requestedComponents - 1; move > slot; move--) {
        selectedScores[move] = selectedScores[move - 1];
        selectedAreas[move] = selectedAreas[move - 1];
        selectedIds[move] = selectedIds[move - 1];
        selectedMinX[move] = selectedMinX[move - 1];
        selectedMinY[move] = selectedMinY[move - 1];
        selectedMaxX[move] = selectedMaxX[move - 1];
        selectedMaxY[move] = selectedMaxY[move - 1];
      }
      selectedScores[slot] = candidateScore;
      selectedAreas[slot] = area;
      selectedIds[slot] = componentId;
      selectedMinX[slot] = minX;
      selectedMinY[slot] = minY;
      selectedMaxX[slot] = maxX;
      selectedMaxY[slot] = maxY;
      break;
    }
  }
  if (selectedIds[0] === 0) {
    const hintRegions = hints
      .slice(0, requestedComponents)
      .map(hint => makeHintSquareRegion(hint, width, height));
    if (hintRegions.length === 0) {
      return {
        pixels: output,
        found: false,
        maskCoverage: 0,
        threshold,
        regions: [],
      };
    }

    enhanceHintRegions(data, output, width, height, hints);
    return {
      pixels: output,
      found: true,
      maskCoverage: 0,
      threshold,
      regions: hintRegions,
    };
  }

  const regions = buildGloveRegions(
    selectedIds,
    selectedAreas,
    selectedMinX,
    selectedMinY,
    selectedMaxX,
    selectedMaxY,
    width,
    height,
    pixelCount,
    minArea
  );
  const firstId = selectedIds[0];
  const secondId = requestedComponents > 1 ? selectedIds[1] : 0;
  const selectedMask = new Uint8Array(pixelCount);
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    const label = labels[pixel];
    if (label === firstId || (secondId !== 0 && label === secondId)) {
      selectedMask[pixel] = 1;
    }
  }
  const isSelected = (pixel: number) => selectedMask[pixel] !== 0;
  const outside = new Uint8Array(pixelCount);
  let head = 0;
  let tail = 0;
  const seedOutside = (pixel: number) => {
    if (!isSelected(pixel) && outside[pixel] === 0) {
      outside[pixel] = 1;
      queue[tail++] = pixel;
    }
  };
  const enqueueNeighbors = (pixel: number, mark: number) => {
    const x = pixel % width;
    const enqueue = (next: number) => {
      if (!isSelected(next) && outside[next] === 0) {
        outside[next] = mark;
        queue[tail++] = next;
      }
    };
    if (x > 0) enqueue(pixel - 1);
    if (x + 1 < width) enqueue(pixel + 1);
    if (pixel >= width) enqueue(pixel - width);
    if (pixel + width < pixelCount) enqueue(pixel + width);
  };

  for (let x = 0; x < width; x++) {
    seedOutside(x);
    if (height > 1) seedOutside((height - 1) * width + x);
  }
  for (let y = 1; y + 1 < height; y++) {
    seedOutside(y * width);
    if (width > 1) seedOutside(y * width + width - 1);
  }
  while (head < tail) {
    enqueueNeighbors(queue[head++], 1);
  }
  const selectedArea =
    selectedAreas[0] + (requestedComponents > 1 ? selectedAreas[1] : 0);
  const maxHoleArea = Math.max(16, Math.floor(selectedArea * 0.18));
  for (let start = 0; start < pixelCount; start++) {
    if (isSelected(start) || outside[start] !== 0) continue;
    head = 0;
    tail = 0;
    outside[start] = 2;
    queue[tail++] = start;
    while (head < tail) {
      enqueueNeighbors(queue[head++], 2);
    }
    if (tail <= maxHoleArea) {
      for (let index = 0; index < tail; index++) {
        selectedMask[queue[index]] = 1;
      }
    }
  }

  for (let offset = 0; offset < data.length; offset += 4) {
    output[offset] = (data[offset] * 3 + 46 * 5) >> 3;
    output[offset + 1] = (data[offset + 1] * 3 + 50 * 5) >> 3;
    output[offset + 2] = (data[offset + 2] * 3 + 56 * 5) >> 3;
  }
  const lightAt = (pixel: number) => {
    const offset = pixel * 4;
    return luminance(data[offset], data[offset + 1], data[offset + 2]);
  };
  let selectedPixelCount = 0;
  for (let pixel = 0; pixel < pixelCount; pixel++) {
    if (!isSelected(pixel)) continue;
    const x = pixel % width;
    const center = lightAt(pixel);
    const left = x > 0 ? lightAt(pixel - 1) : center;
    const right = x + 1 < width ? lightAt(pixel + 1) : center;
    const top = pixel >= width ? lightAt(pixel - width) : center;
    const bottom = pixel + width < pixelCount ? lightAt(pixel + width) : center;
    const laplacian = center * 4 - left - right - top - bottom;
    const detail = Math.round(
      clamp((center - threshold) * 0.12, -12, 18) +
        clamp(laplacian * 0.2, -22, 22)
    );
    const offset = pixel * 4;
    output[offset] = clamp(SKIN_R + detail, 0, 255);
    output[offset + 1] = clamp(SKIN_G + Math.round(detail * 0.78), 0, 255);
    output[offset + 2] = clamp(SKIN_B + Math.round(detail * 0.58), 0, 255);
    selectedPixelCount++;
  }

  return {
    pixels: output,
    found: true,
    maskCoverage: selectedPixelCount / pixelCount,
    threshold,
    regions,
    candidateScore: selectedScores[0],
    candidateScores: buildRegionScores(regions, selectedIds, selectedScores),
  };
}

interface EnhancedRegionCandidate {
  frame: EnhancedGloveFrame;
  region: GloveVisionRegion;
  score: number;
  sourceOrder: number;
}

function regionsRepresentSameHand(
  first: GloveVisionRegion,
  second: GloveVisionRegion
): boolean {
  const overlapWidth = Math.max(
    0,
    Math.min(first.x + first.width, second.x + second.width) -
      Math.max(first.x, second.x)
  );
  const overlapHeight = Math.max(
    0,
    Math.min(first.y + first.height, second.y + second.height) -
      Math.max(first.y, second.y)
  );
  const overlapArea = overlapWidth * overlapHeight;
  const smallerArea = Math.min(
    first.width * first.height,
    second.width * second.height
  );
  return smallerArea > 0 && overlapArea / smallerArea >= 0.55;
}

function copyEnhancedRegion(
  target: Uint8ClampedArray,
  source: Uint8ClampedArray,
  width: number,
  height: number,
  region: GloveVisionRegion
): void {
  const minX = Math.floor(clamp(region.x, 0, width));
  const minY = Math.floor(clamp(region.y, 0, height));
  const maxX = Math.ceil(clamp(region.x + region.width, 0, width));
  const maxY = Math.ceil(clamp(region.y + region.height, 0, height));
  for (let y = minY; y < maxY; y++) {
    const from = (y * width + minX) * 4;
    const to = (y * width + maxX) * 4;
    target.set(source.subarray(from, to), from);
  }
}

export function enhanceGloveFrame(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  maxHands = 2,
  preferredRegions: ReadonlyArray<GloveVisionRegion> = []
): EnhancedGloveFrame {
  const light = enhanceLightGloveFrame(
    data,
    width,
    height,
    maxHands,
    preferredRegions
  );
  const dark = enhanceDarkGloveFrame(
    data,
    width,
    height,
    maxHands,
    preferredRegions
  );
  const lightHasMask = light.found && light.maskCoverage > 0;
  const darkHasMask = dark.found && dark.maskCoverage > 0;
  if (lightHasMask && darkHasMask) {
    const requestedComponents = Math.min(
      MAX_COMPONENTS,
      Math.max(0, Math.trunc(maxHands))
    );
    const lightScore = light.candidateScore ?? Number.NEGATIVE_INFINITY;
    const darkScore = dark.candidateScore ?? Number.NEGATIVE_INFINITY;

    // Three regions represent the whole contour plus two alternating crops for
    // one merged component. Keep that group intact instead of treating its
    // overlapping crops as independent hands.
    if (
      light.regions.length > requestedComponents ||
      dark.regions.length > requestedComponents
    ) {
      return lightScore >= darkScore ? light : dark;
    }

    const candidates: EnhancedRegionCandidate[] = [];
    const addCandidates = (frame: EnhancedGloveFrame, sourceOrder: number) => {
      frame.regions.forEach((region, index) => {
        candidates.push({
          frame,
          region,
          score:
            frame.candidateScores?.[index] ??
            frame.candidateScore ??
            Number.NEGATIVE_INFINITY,
          sourceOrder,
        });
      });
    };
    addCandidates(light, 0);
    addCandidates(dark, 1);
    candidates.sort(
      (first, second) =>
        second.score - first.score ||
        second.region.area - first.region.area ||
        first.sourceOrder - second.sourceOrder
    );

    const selected: EnhancedRegionCandidate[] = [];
    for (const candidate of candidates) {
      if (selected.length >= requestedComponents) break;
      if (
        selected.some(existing =>
          regionsRepresentSameHand(existing.region, candidate.region)
        )
      ) {
        continue;
      }
      selected.push(candidate);
    }
    if (selected.length === 0) {
      return lightScore >= darkScore ? light : dark;
    }

    const baseFrame = selected[0].frame;
    if (
      selected.length === baseFrame.regions.length &&
      selected.every(candidate => candidate.frame === baseFrame)
    ) {
      return baseFrame;
    }

    const pixels = new Uint8ClampedArray(baseFrame.pixels);
    for (let index = 1; index < selected.length; index++) {
      const candidate = selected[index];
      if (candidate.frame !== baseFrame) {
        copyEnhancedRegion(
          pixels,
          candidate.frame.pixels,
          width,
          height,
          candidate.region
        );
      }
    }
    const usesLight = selected.some(candidate => candidate.frame === light);
    const usesDark = selected.some(candidate => candidate.frame === dark);
    return {
      pixels,
      found: true,
      maskCoverage:
        usesLight && usesDark
          ? Math.min(1, light.maskCoverage + dark.maskCoverage)
          : baseFrame.maskCoverage,
      threshold: baseFrame.threshold,
      regions: selected.map(candidate => candidate.region),
      candidateScore: selected[0].score,
      candidateScores: selected.map(candidate => candidate.score),
    };
  }
  if (lightHasMask) return light;
  if (darkHasMask) return dark;
  return light.found ? light : dark;
}

function unknownSurface(): GloveSurfaceClassification {
  return { ...UNKNOWN_SURFACE };
}

/**
 * Classifies the visible glove face by sampling a narrow band around the
 * MediaPipe skeleton. Silver inserts indicate the palm; a predominantly dark
 * neutral surface indicates the back.
 */
export function classifyGloveSurface(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  landmarks: ReadonlyArray<GloveVisionLandmark>
): GloveSurfaceClassification {
  const pixelCount = validateFrame(data, width, height);
  if (!Array.isArray(landmarks) || landmarks.length !== 21) {
    return unknownSurface();
  }

  const pointsX = new Float32Array(21);
  const pointsY = new Float32Array(21);
  let minX = width;
  let maxX = -1;
  let minY = height;
  let maxY = -1;

  for (let i = 0; i < 21; i++) {
    const landmark = landmarks[i];
    if (
      !landmark ||
      !Number.isFinite(landmark.x) ||
      !Number.isFinite(landmark.y) ||
      landmark.x < -0.25 ||
      landmark.x > 1.25 ||
      landmark.y < -0.25 ||
      landmark.y > 1.25
    ) {
      return unknownSurface();
    }

    const x = clamp(landmark.x, 0, 1) * (width - 1);
    const y = clamp(landmark.y, 0, 1) * (height - 1);
    pointsX[i] = x;
    pointsY[i] = y;
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  const handSpan = Math.max(maxX - minX, maxY - minY);
  if (!Number.isFinite(handSpan) || handSpan < 3) return unknownSurface();

  const sampleMask = new Uint8Array(pixelCount);
  const radius = Math.max(1, Math.min(7, Math.round(handSpan * 0.018)));

  const markDisk = (centerX: number, centerY: number, diskRadius = radius) => {
    const x0 = Math.max(0, Math.floor(centerX - diskRadius));
    const x1 = Math.min(width - 1, Math.ceil(centerX + diskRadius));
    const y0 = Math.max(0, Math.floor(centerY - diskRadius));
    const y1 = Math.min(height - 1, Math.ceil(centerY + diskRadius));
    const diskRadiusSquared = diskRadius * diskRadius;
    for (let y = y0; y <= y1; y++) {
      const dy = y - centerY;
      let pixel = y * width + x0;
      for (let x = x0; x <= x1; x++, pixel++) {
        const dx = x - centerX;
        if (dx * dx + dy * dy <= diskRadiusSquared) sampleMask[pixel] = 1;
      }
    }
  };

  for (
    let connection = 0;
    connection < SAMPLE_CONNECTIONS.length;
    connection += 2
  ) {
    const from = SAMPLE_CONNECTIONS[connection];
    const to = SAMPLE_CONNECTIONS[connection + 1];
    const x0 = pointsX[from];
    const y0 = pointsY[from];
    const dx = pointsX[to] - x0;
    const dy = pointsY[to] - y0;
    const steps = Math.max(
      1,
      Math.ceil(Math.hypot(dx, dy) / Math.max(1, radius))
    );
    for (let step = 0; step <= steps; step++) {
      const progress = step / steps;
      markDisk(x0 + dx * progress, y0 + dy * progress);
    }
  }

  // Add a small central palm sample without expanding to the silhouette edge.
  let palmX = 0;
  let palmY = 0;
  const palmIndices = [0, 5, 9, 13, 17] as const;
  for (let i = 0; i < palmIndices.length; i++) {
    palmX += pointsX[palmIndices[i]];
    palmY += pointsY[palmIndices[i]];
  }
  markDisk(
    palmX / palmIndices.length,
    palmY / palmIndices.length,
    Math.max(radius, Math.round(radius * 2.2))
  );

  let sampleCount = 0;
  let darkCount = 0;
  let darkLightSum = 0;
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel++, offset += 4) {
    if (sampleMask[pixel] === 0 || data[offset + 3] < 16) continue;
    sampleCount++;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const light = luminance(r, g, b);

    if (isLowChromaDark(r, g, b, 105)) {
      darkCount++;
      darkLightSum += light;
    }
  }

  if (sampleCount < 12) return unknownSurface();

  const darkMean = darkCount > 0 ? darkLightSum / darkCount : 70;
  const silverThreshold = clamp(Math.round(darkMean + 60), 110, 145);
  let silverCount = 0;
  for (let pixel = 0, offset = 0; pixel < pixelCount; pixel++, offset += 4) {
    if (sampleMask[pixel] === 0 || data[offset + 3] < 16) continue;
    const r = data[offset];
    const g = data[offset + 1];
    const b = data[offset + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const chroma = max - min;
    const light = luminance(r, g, b);
    if (light >= silverThreshold && min >= 82 && chroma <= 58) silverCount++;
  }

  const silverRatio = silverCount / sampleCount;
  const darkRatio = darkCount / sampleCount;
  const materialRatio = silverRatio + darkRatio;
  const isGlove =
    darkRatio >= 0.42 ||
    (darkRatio >= 0.2 && silverRatio >= 0.09 && materialRatio >= 0.5);

  if (!isGlove) {
    return {
      surface: "unknown",
      isGlove: false,
      silverRatio,
      darkRatio,
      confidence: 0,
    };
  }

  const silverShare = silverRatio / Math.max(materialRatio, 1e-6);
  if (silverRatio >= 0.085 && silverShare >= 0.1) {
    const silverEvidence = clamp((silverRatio - 0.06) / 0.28, 0, 1);
    const materialEvidence = clamp((materialRatio - 0.4) / 0.5, 0, 1);
    return {
      surface: "palm",
      isGlove: true,
      silverRatio,
      darkRatio,
      confidence: clamp(
        0.52 + silverEvidence * 0.3 + materialEvidence * 0.16,
        0,
        0.98
      ),
    };
  }

  if (darkRatio >= 0.5 && silverRatio < 0.085) {
    const darknessEvidence = clamp((darkRatio - 0.4) / 0.55, 0, 1);
    const purityEvidence = clamp(1 - silverRatio / 0.085, 0, 1);
    return {
      surface: "back",
      isGlove: true,
      silverRatio,
      darkRatio,
      confidence: clamp(
        0.55 + darknessEvidence * 0.3 + purityEvidence * 0.12,
        0,
        0.98
      ),
    };
  }

  return {
    surface: "unknown",
    isGlove: true,
    silverRatio,
    darkRatio,
    confidence: 0,
  };
}
