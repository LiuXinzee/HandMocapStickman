export interface HandIdentityLandmark {
  x: number;
  y: number;
  z?: number;
}

export interface HandIdentityFrameInput<TSource extends string = string> {
  landmarks: ReadonlyArray<ReadonlyArray<HandIdentityLandmark>>;
  handedness?: ReadonlyArray<string>;
  detectionSources?: ReadonlyArray<TSource>;
}

export interface StableHandIdentityFrame<TSource extends string = string> {
  /** Landmark arrays reordered into stable temporal track order. */
  landmarks: Array<ReadonlyArray<HandIdentityLandmark>>;
  /** Temporally stabilized Left/Right labels in the same order. */
  handedness: string[];
  detectionSources: Array<TSource | undefined>;
  trackingIds: string[];
  /** Original index of each returned detection before stable reordering. */
  inputIndices: number[];
}

export interface HandIdentityTrackerOptions {
  /** Number of missing frames for which an identity remains recoverable. */
  maxMissedFrames?: number;
  /** Absolute normalized-coordinate cap for a valid temporal match. */
  maxMatchDistance?: number;
}

interface DetectionFeature<TSource extends string> {
  landmarks: ReadonlyArray<HandIdentityLandmark>;
  handedness: string;
  source: TSource | undefined;
  inputIndex: number;
  centerX: number;
  centerY: number;
  scale: number;
}

interface HandTrack {
  id: string;
  order: number;
  centerX: number;
  centerY: number;
  velocityX: number;
  velocityY: number;
  scale: number;
  landmarks: ReadonlyArray<HandIdentityLandmark>;
  missedFrames: number;
  age: number;
  leftEvidence: number;
  rightEvidence: number;
  stableHandedness: string;
  pendingHandedness: string;
  pendingHandednessFrames: number;
}

interface MatchedDetection<TSource extends string> {
  track: HandTrack;
  detection: DetectionFeature<TSource>;
}

const PALM_INDICES = [0, 5, 9, 13, 17] as const;
const HANDEDNESS_SWITCH_FRAMES = 15;

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function normalizeHandedness(label: string | undefined): string {
  const normalized = label?.trim().toLowerCase();
  if (normalized === "left") return "Left";
  if (normalized === "right") return "Right";
  return "Unknown";
}

function handednessRank(label: string): number {
  if (label === "Left") return 0;
  if (label === "Right") return 1;
  return 2;
}

function makeDetectionFeature<TSource extends string>(
  landmarks: ReadonlyArray<HandIdentityLandmark>,
  handedness: string | undefined,
  source: TSource | undefined,
  inputIndex: number
): DetectionFeature<TSource> | null {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) return null;
  for (let index = 0; index < landmarks.length; index++) {
    const landmark = landmarks[index];
    if (
      !landmark ||
      !Number.isFinite(landmark.x) ||
      !Number.isFinite(landmark.y) ||
      (landmark.z !== undefined && !Number.isFinite(landmark.z))
    ) {
      return null;
    }
  }

  let centerX = 0;
  let centerY = 0;
  for (let index = 0; index < PALM_INDICES.length; index++) {
    const landmark = landmarks[PALM_INDICES[index]];
    centerX += landmark.x;
    centerY += landmark.y;
  }
  centerX /= PALM_INDICES.length;
  centerY /= PALM_INDICES.length;

  let scale = 0;
  for (let index = 0; index < landmarks.length; index++) {
    const dx = landmarks[index].x - centerX;
    const dy = landmarks[index].y - centerY;
    scale = Math.max(scale, Math.hypot(dx, dy));
  }
  if (!Number.isFinite(scale) || scale < 1e-4) return null;

  return {
    landmarks,
    handedness: normalizeHandedness(handedness),
    source,
    inputIndex,
    centerX,
    centerY,
    scale,
  };
}

function assignColdStartHandedness<TSource extends string>(
  detections: Array<DetectionFeature<TSource>>
): void {
  if (detections.length !== 2) return;
  const labels = new Set(detections.map(detection => detection.handedness));
  if (labels.has("Left") && labels.has("Right")) return;

  const spatialOrder = [...detections].sort(
    (first, second) => first.centerX - second.centerX
  );
  // The source frame is not mirrored: the user's right hand appears on the
  // left side of the camera image during a normal two-hand cold start.
  spatialOrder[0].handedness = "Right";
  spatialOrder[1].handedness = "Left";
}

function normalizedShapeDistance<TSource extends string>(
  track: HandTrack,
  detection: DetectionFeature<TSource>
): number {
  const trackScale = Math.max(track.scale, 1e-4);
  const detectionScale = Math.max(detection.scale, 1e-4);
  let total = 0;
  for (let index = 0; index < 21; index++) {
    const trackX = (track.landmarks[index].x - track.centerX) / trackScale;
    const trackY = (track.landmarks[index].y - track.centerY) / trackScale;
    const detectionX =
      (detection.landmarks[index].x - detection.centerX) / detectionScale;
    const detectionY =
      (detection.landmarks[index].y - detection.centerY) / detectionScale;
    total += Math.hypot(trackX - detectionX, trackY - detectionY);
  }
  return total / 21;
}

function matchCost<TSource extends string>(
  track: HandTrack,
  detection: DetectionFeature<TSource>,
  maxMatchDistance: number
): number {
  const predictionFrames = Math.min(4, track.missedFrames + 1);
  const predictedX = track.centerX + track.velocityX * predictionFrames;
  const predictedY = track.centerY + track.velocityY * predictionFrames;
  const predictedDistance = Math.hypot(
    detection.centerX - predictedX,
    detection.centerY - predictedY
  );
  const scale = Math.max(track.scale, detection.scale, 0.05);
  const gate = Math.min(
    maxMatchDistance,
    Math.max(0.14, scale * 2.8) + Math.min(track.missedFrames, 4) * 0.055
  );
  if (predictedDistance > gate) return Number.POSITIVE_INFINITY;

  let handednessPenalty = 0;
  if (
    track.stableHandedness !== "Unknown" &&
    detection.handedness !== "Unknown"
  ) {
    handednessPenalty =
      track.stableHandedness === detection.handedness ? 0 : 0.04;
  } else if (detection.handedness === "Unknown") {
    handednessPenalty = 0.04;
  }

  const speed = Math.hypot(track.velocityX, track.velocityY);
  let directionPenalty = 0;
  if (speed > 0.012) {
    const displacementX = detection.centerX - track.centerX;
    const displacementY = detection.centerY - track.centerY;
    const displacement = Math.hypot(displacementX, displacementY);
    if (displacement > 0.008) {
      const cosine =
        (displacementX * track.velocityX + displacementY * track.velocityY) /
        (displacement * speed);
      if (cosine < 0) directionPenalty = -cosine * 0.22;
    }
  }

  return (
    predictedDistance / gate +
    normalizedShapeDistance(track, detection) * 0.12 +
    handednessPenalty +
    directionPenalty
  );
}

function updateHandednessEvidence(
  track: HandTrack,
  observedHandedness: string,
  allowSwitch: boolean
): void {
  track.leftEvidence *= 0.9;
  track.rightEvidence *= 0.9;
  if (observedHandedness === "Left") track.leftEvidence += 1;
  if (observedHandedness === "Right") track.rightEvidence += 1;

  if (track.stableHandedness === "Unknown") {
    if (track.leftEvidence > track.rightEvidence + 0.5) {
      track.stableHandedness = "Left";
    } else if (track.rightEvidence > track.leftEvidence + 0.5) {
      track.stableHandedness = "Right";
    }
    return;
  }

  if (
    observedHandedness === "Unknown" ||
    observedHandedness === track.stableHandedness
  ) {
    track.pendingHandedness = "Unknown";
    track.pendingHandednessFrames = 0;
    return;
  }

  if (track.pendingHandedness === observedHandedness) {
    track.pendingHandednessFrames++;
  } else {
    track.pendingHandedness = observedHandedness;
    track.pendingHandednessFrames = 1;
  }

  if (
    allowSwitch &&
    track.pendingHandednessFrames >= HANDEDNESS_SWITCH_FRAMES
  ) {
    track.stableHandedness = observedHandedness;
    track.pendingHandedness = "Unknown";
    track.pendingHandednessFrames = 0;
    track.leftEvidence = observedHandedness === "Left" ? 2.5 : 0;
    track.rightEvidence = observedHandedness === "Right" ? 2.5 : 0;
  }
}

function setStableHandedness(track: HandTrack, handedness: string): void {
  track.stableHandedness = handedness;
  track.pendingHandedness = "Unknown";
  track.pendingHandednessFrames = 0;
  track.leftEvidence = handedness === "Left" ? 2.5 : 0;
  track.rightEvidence = handedness === "Right" ? 2.5 : 0;
}

function oppositeHandedness(handedness: string): string {
  return handedness === "Left" ? "Right" : "Left";
}

function ensureUniqueHandednessPair(tracks: HandTrack[]): void {
  if (tracks.length !== 2) return;
  const [first, second] = tracks;
  if (
    first.stableHandedness !== "Unknown" &&
    second.stableHandedness !== "Unknown" &&
    first.stableHandedness !== second.stableHandedness
  ) {
    return;
  }

  if (
    first.stableHandedness === "Unknown" &&
    second.stableHandedness !== "Unknown"
  ) {
    setStableHandedness(first, oppositeHandedness(second.stableHandedness));
    return;
  }
  if (
    second.stableHandedness === "Unknown" &&
    first.stableHandedness !== "Unknown"
  ) {
    setStableHandedness(second, oppositeHandedness(first.stableHandedness));
    return;
  }

  if (
    first.stableHandedness === second.stableHandedness &&
    first.age !== second.age
  ) {
    const established = first.age > second.age ? first : second;
    const newer = established === first ? second : first;
    const establishedSide =
      established.stableHandedness === "Unknown"
        ? established.centerX < newer.centerX
          ? "Right"
          : "Left"
        : established.stableHandedness;
    setStableHandedness(established, establishedSide);
    setStableHandedness(newer, oppositeHandedness(establishedSide));
    return;
  }

  const spatialOrder = [...tracks].sort(
    (leftmost, rightmost) => leftmost.centerX - rightmost.centerX
  );
  setStableHandedness(spatialOrder[0], "Right");
  setStableHandedness(spatialOrder[1], "Left");
}

/**
 * Stateful two-hand identity matcher. The class keeps only compact temporal
 * features and returns detections in stable track creation order.
 */
export class HandIdentityTracker<TSource extends string = string> {
  private readonly maxMissedFrames: number;
  private readonly maxMatchDistance: number;
  private tracks: HandTrack[] = [];
  private nextTrackNumber = 1;

  constructor(options: HandIdentityTrackerOptions = {}) {
    const maxMissedFrames = options.maxMissedFrames ?? 6;
    const maxMatchDistance = options.maxMatchDistance ?? 0.42;
    if (!Number.isInteger(maxMissedFrames) || maxMissedFrames < 0) {
      throw new RangeError("maxMissedFrames must be a non-negative integer");
    }
    if (!Number.isFinite(maxMatchDistance) || maxMatchDistance <= 0) {
      throw new RangeError("maxMatchDistance must be positive and finite");
    }
    this.maxMissedFrames = maxMissedFrames;
    this.maxMatchDistance = maxMatchDistance;
  }

  get activeTrackCount(): number {
    return this.tracks.length;
  }

  reset(): void {
    this.tracks = [];
    this.nextTrackNumber = 1;
  }

  update(
    input: HandIdentityFrameInput<TSource>
  ): StableHandIdentityFrame<TSource> {
    const features: Array<DetectionFeature<TSource>> = [];
    const landmarks = Array.isArray(input?.landmarks) ? input.landmarks : [];
    for (let inputIndex = 0; inputIndex < landmarks.length; inputIndex++) {
      const feature = makeDetectionFeature(
        landmarks[inputIndex],
        input.handedness?.[inputIndex],
        input.detectionSources?.[inputIndex],
        inputIndex
      );
      if (feature) features.push(feature);
    }

    const assignment = this.findBestAssignment(features);
    if (this.tracks.length === 0) {
      assignColdStartHandedness(features);
    }
    const matched: Array<MatchedDetection<TSource>> = [];
    const usedDetections = new Uint8Array(features.length);

    for (let trackIndex = 0; trackIndex < this.tracks.length; trackIndex++) {
      const track = this.tracks[trackIndex];
      const detectionIndex = assignment[trackIndex];
      if (detectionIndex < 0) {
        track.missedFrames++;
        track.velocityX *= 0.9;
        track.velocityY *= 0.9;
        continue;
      }

      const detection = features[detectionIndex];
      usedDetections[detectionIndex] = 1;
      this.updateTrack(track, detection, this.tracks.length < 2);
      matched.push({ track, detection });
    }

    this.tracks = this.tracks.filter(
      track => track.missedFrames <= this.maxMissedFrames
    );

    if (this.tracks.length < 2) {
      const unmatched = features
        .filter((_, index) => usedDetections[index] === 0)
        .sort((first, second) => {
          const rankDelta =
            handednessRank(first.handedness) -
            handednessRank(second.handedness);
          return rankDelta || first.centerX - second.centerX;
        });

      for (const detection of unmatched) {
        if (this.tracks.length >= 2) break;
        const track = this.createTrack(detection);
        this.tracks.push(track);
        matched.push({ track, detection });
      }
    }

    ensureUniqueHandednessPair(this.tracks);

    matched.sort((first, second) => first.track.order - second.track.order);
    return {
      landmarks: matched.map(item => item.detection.landmarks),
      handedness: matched.map(item => item.track.stableHandedness),
      detectionSources: matched.map(item => item.detection.source),
      trackingIds: matched.map(item => item.track.id),
      inputIndices: matched.map(item => item.detection.inputIndex),
    };
  }

  private findBestAssignment(
    detections: Array<DetectionFeature<TSource>>
  ): Int16Array {
    const trackCount = this.tracks.length;
    const current = new Int16Array(trackCount);
    const best = new Int16Array(trackCount);
    current.fill(-1);
    best.fill(-1);
    const used = new Uint8Array(detections.length);
    let bestMatches = -1;
    let bestCost = Number.POSITIVE_INFINITY;

    const search = (trackIndex: number, matches: number, totalCost: number) => {
      if (trackIndex >= trackCount) {
        if (
          matches > bestMatches ||
          (matches === bestMatches && totalCost < bestCost)
        ) {
          bestMatches = matches;
          bestCost = totalCost;
          best.set(current);
        }
        return;
      }

      current[trackIndex] = -1;
      search(trackIndex + 1, matches, totalCost);

      for (
        let detectionIndex = 0;
        detectionIndex < detections.length;
        detectionIndex++
      ) {
        if (used[detectionIndex] !== 0) continue;
        const cost = matchCost(
          this.tracks[trackIndex],
          detections[detectionIndex],
          this.maxMatchDistance
        );
        if (!Number.isFinite(cost)) continue;
        used[detectionIndex] = 1;
        current[trackIndex] = detectionIndex;
        search(trackIndex + 1, matches + 1, totalCost + cost);
        used[detectionIndex] = 0;
      }
      current[trackIndex] = -1;
    };

    search(0, 0, 0);
    return best;
  }

  private createTrack(detection: DetectionFeature<TSource>): HandTrack {
    const stableHandedness = detection.handedness;
    return {
      id: `hand-${this.nextTrackNumber}`,
      order: this.nextTrackNumber++,
      centerX: detection.centerX,
      centerY: detection.centerY,
      velocityX: 0,
      velocityY: 0,
      scale: detection.scale,
      landmarks: detection.landmarks,
      missedFrames: 0,
      age: 1,
      leftEvidence: stableHandedness === "Left" ? 2.5 : 0,
      rightEvidence: stableHandedness === "Right" ? 2.5 : 0,
      stableHandedness,
      pendingHandedness: "Unknown",
      pendingHandednessFrames: 0,
    };
  }

  private updateTrack(
    track: HandTrack,
    detection: DetectionFeature<TSource>,
    allowHandednessSwitch: boolean
  ): void {
    const elapsedFrames = track.missedFrames + 1;
    const observedVelocityX =
      (detection.centerX - track.centerX) / elapsedFrames;
    const observedVelocityY =
      (detection.centerY - track.centerY) / elapsedFrames;
    track.velocityX = track.velocityX * 0.2 + observedVelocityX * 0.8;
    track.velocityY = track.velocityY * 0.2 + observedVelocityY * 0.8;
    track.centerX = detection.centerX;
    track.centerY = detection.centerY;
    track.scale = track.scale * 0.35 + detection.scale * 0.65;
    track.landmarks = detection.landmarks;
    track.missedFrames = 0;
    track.age++;
    updateHandednessEvidence(
      track,
      detection.handedness,
      allowHandednessSwitch
    );
  }
}
