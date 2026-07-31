export interface DetectionLandmark {
  x: number;
  y: number;
  z: number;
}

/** Pixel-space crop within the full camera frame. */
export interface DetectionRegion {
  x: number;
  y: number;
  width: number;
  height: number;
}

export type DetectionCandidateSource = "standard" | "glove-roi";

export interface HandDetectionCandidate {
  landmarks: DetectionLandmark[];
  handedness: string;
  score: number;
  source: DetectionCandidateSource;
  regionIndex?: number;
  /** Stable retry-region identity used to replace only the matching cache. */
  regionKey?: string;
}

const PALM_LANDMARK_INDICES = [0, 5, 9, 13, 17] as const;
const MIN_DUPLICATE_DISTANCE = 0.035;

function isFiniteNumber(value: number): boolean {
  return Number.isFinite(value);
}

function isValidLandmark(landmark: DetectionLandmark): boolean {
  return (
    landmark != null &&
    isFiniteNumber(landmark.x) &&
    isFiniteNumber(landmark.y) &&
    isFiniteNumber(landmark.z)
  );
}

function isValidRegion(
  region: DetectionRegion | null | undefined
): region is DetectionRegion {
  return (
    region != null &&
    isFiniteNumber(region.x) &&
    isFiniteNumber(region.y) &&
    isFiniteNumber(region.width) &&
    isFiniteNumber(region.height) &&
    region.width > 0 &&
    region.height > 0
  );
}

/**
 * Maps MediaPipe landmarks normalized to an ROI back into full-frame normalized
 * coordinates. Z follows MediaPipe's X-based scale and is therefore scaled by
 * the ROI width.
 */
export function mapRoiLandmarksToFrame(
  landmarks: ReadonlyArray<DetectionLandmark> | null | undefined,
  region: DetectionRegion | null | undefined,
  frameWidth: number,
  frameHeight: number
): DetectionLandmark[] {
  if (
    !landmarks ||
    landmarks.length === 0 ||
    !landmarks.every(isValidLandmark) ||
    !isValidRegion(region) ||
    !isFiniteNumber(frameWidth) ||
    !isFiniteNumber(frameHeight) ||
    frameWidth <= 0 ||
    frameHeight <= 0
  ) {
    return [];
  }

  const zScale = region.width / frameWidth;
  return landmarks.map(landmark => ({
    x: (region.x + landmark.x * region.width) / frameWidth,
    y: (region.y + landmark.y * region.height) / frameHeight,
    z: landmark.z * zScale,
  }));
}

interface Point2D {
  x: number;
  y: number;
}

function distance(a: Point2D, b: Point2D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function selectedPalmLandmarks(
  landmarks: ReadonlyArray<DetectionLandmark>
): ReadonlyArray<DetectionLandmark> {
  const palm = PALM_LANDMARK_INDICES.filter(
    index => index < landmarks.length
  ).map(index => landmarks[index]);
  return palm.length >= 2 ? palm : landmarks;
}

function centroid(landmarks: ReadonlyArray<DetectionLandmark>): Point2D {
  const points = selectedPalmLandmarks(landmarks);
  const sum = points.reduce(
    (total, landmark) => ({
      x: total.x + landmark.x,
      y: total.y + landmark.y,
    }),
    { x: 0, y: 0 }
  );
  return { x: sum.x / points.length, y: sum.y / points.length };
}

function handScale(landmarks: ReadonlyArray<DetectionLandmark>): number {
  const center = centroid(landmarks);
  let maxDistance = 0;
  for (const landmark of landmarks) {
    maxDistance = Math.max(maxDistance, distance(center, landmark));
  }
  return maxDistance;
}

function meanLandmarkDistance(
  first: ReadonlyArray<DetectionLandmark>,
  second: ReadonlyArray<DetectionLandmark>
): number {
  const count = Math.min(first.length, second.length);
  let total = 0;
  for (let index = 0; index < count; index++) {
    total += distance(first[index], second[index]);
  }
  return total / count;
}

function normalizedHandedness(handedness: string): string | null {
  const normalized = handedness.trim().toLowerCase();
  return normalized && normalized !== "unknown" ? normalized : null;
}

function isGeometricDuplicate(
  first: HandDetectionCandidate,
  second: HandDetectionCandidate
): boolean {
  const firstHandedness = normalizedHandedness(first.handedness);
  const secondHandedness = normalizedHandedness(second.handedness);
  const scale = Math.max(
    handScale(first.landmarks),
    handScale(second.landmarks)
  );
  const centerDistance = distance(
    centroid(first.landmarks),
    centroid(second.landmarks)
  );
  const landmarkDistance = meanLandmarkDistance(
    first.landmarks,
    second.landmarks
  );

  // Handedness occasionally flips between the full-frame and ROI passes. Only
  // collapse opposite labels when the complete skeleton is nearly identical;
  // partially overlapping left/right hands still remain separate.
  if (
    firstHandedness != null &&
    secondHandedness != null &&
    firstHandedness !== secondHandedness
  ) {
    return (
      centerDistance <= Math.max(0.018, scale * 0.18) &&
      landmarkDistance <= Math.max(0.025, scale * 0.3)
    );
  }

  const centerThreshold = Math.max(MIN_DUPLICATE_DISTANCE, scale * 0.45);
  const landmarkThreshold = Math.max(
    MIN_DUPLICATE_DISTANCE * 1.4,
    scale * 0.55
  );

  return (
    centerDistance <= centerThreshold && landmarkDistance <= landmarkThreshold
  );
}

function isValidCandidate(candidate: HandDetectionCandidate): boolean {
  return (
    candidate != null &&
    Array.isArray(candidate.landmarks) &&
    candidate.landmarks.length > 0 &&
    candidate.landmarks.every(isValidLandmark) &&
    typeof candidate.handedness === "string" &&
    isFiniteNumber(candidate.score) &&
    (candidate.source === "standard" || candidate.source === "glove-roi") &&
    (candidate.regionIndex === undefined ||
      (Number.isInteger(candidate.regionIndex) &&
        candidate.regionIndex >= 0)) &&
    (candidate.regionKey === undefined ||
      (typeof candidate.regionKey === "string" &&
        candidate.regionKey.length > 0))
  );
}

/**
 * Adds valid primary detections first, then fills missing hands from ROI
 * detections. Near-identical skeletons are removed without collapsing an
 * explicitly Left/Right pair during hand overlap.
 */
export function mergeHandDetections(
  primary: ReadonlyArray<HandDetectionCandidate> | null | undefined,
  fallback: ReadonlyArray<HandDetectionCandidate> | null | undefined,
  maxHands = 2
): HandDetectionCandidate[] {
  if (!Number.isFinite(maxHands)) return [];
  const limit = Math.floor(maxHands);
  if (limit <= 0) return [];

  const merged: HandDetectionCandidate[] = [];
  const candidates = [
    ...(Array.isArray(primary) ? primary : []),
    ...(Array.isArray(fallback) ? fallback : []),
  ];

  for (const candidate of candidates) {
    if (!isValidCandidate(candidate)) continue;
    if (merged.some(existing => isGeometricDuplicate(existing, candidate)))
      continue;
    merged.push(candidate);
    if (merged.length >= limit) break;
  }

  return merged;
}
