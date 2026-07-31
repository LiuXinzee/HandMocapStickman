import type { GloveVisionLandmark, GloveVisionRegion } from "./gloveVision";

export interface TrackedGloveRegion extends GloveVisionRegion {
  trackingId: string;
  ageMs: number;
}

export interface GloveRoiTrackerOptions {
  /** How long a missing hand remains available for a retry crop. */
  maxAgeMs?: number;
  /** Maximum time for which measured velocity is extrapolated. */
  predictionHorizonMs?: number;
  /** Padding around the last complete 21-point landmark box. */
  paddingScale?: number;
}

interface TrackedHandBox {
  trackingId: string;
  order: number;
  palmCenterX: number;
  palmCenterY: number;
  boxCenterX: number;
  boxCenterY: number;
  boxWidth: number;
  boxHeight: number;
  velocityX: number;
  velocityY: number;
  lastSeenAt: number;
}

interface HandBoxMeasurement {
  palmCenterX: number;
  palmCenterY: number;
  boxCenterX: number;
  boxCenterY: number;
  boxWidth: number;
  boxHeight: number;
}

interface LandmarkDetection {
  landmarks: ReadonlyArray<GloveVisionLandmark>;
}

const PALM_INDICES = [0, 5, 9, 13, 17] as const;

function clamp(value: number, min: number, max: number): number {
  return value < min ? min : value > max ? max : value;
}

function measureHandBox(
  landmarks: ReadonlyArray<GloveVisionLandmark>
): HandBoxMeasurement | null {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) return null;

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let palmCenterX = 0;
  let palmCenterY = 0;

  for (let index = 0; index < landmarks.length; index++) {
    const landmark = landmarks[index];
    if (
      !landmark ||
      !Number.isFinite(landmark.x) ||
      !Number.isFinite(landmark.y)
    ) {
      return null;
    }
    minX = Math.min(minX, landmark.x);
    minY = Math.min(minY, landmark.y);
    maxX = Math.max(maxX, landmark.x);
    maxY = Math.max(maxY, landmark.y);
  }

  for (const index of PALM_INDICES) {
    palmCenterX += landmarks[index].x;
    palmCenterY += landmarks[index].y;
  }
  palmCenterX /= PALM_INDICES.length;
  palmCenterY /= PALM_INDICES.length;

  const boxWidth = maxX - minX;
  const boxHeight = maxY - minY;
  if (boxWidth < 1e-4 || boxHeight < 1e-4) return null;

  return {
    palmCenterX,
    palmCenterY,
    boxCenterX: (minX + maxX) * 0.5,
    boxCenterY: (minY + maxY) * 0.5,
    boxWidth,
    boxHeight,
  };
}

function makeSquareRegion(
  centerX: number,
  centerY: number,
  side: number,
  frameWidth: number,
  frameHeight: number,
  area: number
): GloveVisionRegion {
  const clampedSide = Math.max(
    1,
    Math.min(Math.round(side), frameWidth, frameHeight)
  );
  const x = Math.round(
    clamp(centerX - clampedSide * 0.5, 0, frameWidth - clampedSide)
  );
  const y = Math.round(
    clamp(centerY - clampedSide * 0.5, 0, frameHeight - clampedSide)
  );
  return {
    x,
    y,
    width: clampedSide,
    height: clampedSide,
    area: Math.max(0, Math.round(area)),
  };
}

function palmCenter(
  detection: LandmarkDetection,
  frameWidth: number,
  frameHeight: number
): { x: number; y: number } | null {
  const measurement = measureHandBox(detection.landmarks);
  if (!measurement) return null;
  return {
    x: measurement.palmCenterX * frameWidth,
    y: measurement.palmCenterY * frameHeight,
  };
}

/**
 * Retains compact landmark boxes for temporarily missing hands. It never
 * fabricates landmarks; predictions are used only to choose a retry crop.
 */
export class GloveRoiTracker {
  private readonly maxAgeMs: number;
  private readonly predictionHorizonMs: number;
  private readonly paddingScale: number;
  private readonly tracks = new Map<string, TrackedHandBox>();
  private nextOrder = 0;

  constructor(options: GloveRoiTrackerOptions = {}) {
    this.maxAgeMs = options.maxAgeMs ?? 1_100;
    this.predictionHorizonMs = options.predictionHorizonMs ?? 280;
    this.paddingScale = options.paddingScale ?? 1.38;
    if (!Number.isFinite(this.maxAgeMs) || this.maxAgeMs <= 0) {
      throw new RangeError("maxAgeMs must be positive and finite");
    }
    if (
      !Number.isFinite(this.predictionHorizonMs) ||
      this.predictionHorizonMs < 0
    ) {
      throw new RangeError(
        "predictionHorizonMs must be non-negative and finite"
      );
    }
    if (!Number.isFinite(this.paddingScale) || this.paddingScale < 1) {
      throw new RangeError("paddingScale must be at least 1");
    }
  }

  reset(): void {
    this.tracks.clear();
    this.nextOrder = 0;
  }

  update(
    trackingIds: ReadonlyArray<string>,
    hands: ReadonlyArray<ReadonlyArray<GloveVisionLandmark>>,
    now: number
  ): void {
    if (!Number.isFinite(now)) return;

    this.prune(now);
    const count = Math.min(trackingIds.length, hands.length);
    const incomingIds = new Set<string>();
    for (let index = 0; index < count; index++) {
      const trackingId = trackingIds[index]?.trim();
      if (trackingId) incomingIds.add(trackingId);
    }
    const claimedTracks = new Set<TrackedHandBox>();

    for (let index = 0; index < count; index++) {
      const trackingId = trackingIds[index]?.trim();
      const measurement = measureHandBox(hands[index]);
      if (!trackingId || !measurement) continue;

      let previous = this.tracks.get(trackingId);
      if (previous && claimedTracks.has(previous)) continue;

      if (!previous) {
        const nearestTrack = Array.from(this.tracks.values())
          .filter(
            track =>
              !claimedTracks.has(track) && !incomingIds.has(track.trackingId)
          )
          .map(track => {
            const ageMs = Math.max(0, now - track.lastSeenAt);
            const predictionMs = Math.min(ageMs, this.predictionHorizonMs);
            const predictedX =
              track.palmCenterX + track.velocityX * predictionMs;
            const predictedY =
              track.palmCenterY + track.velocityY * predictionMs;
            const distance = Math.hypot(
              measurement.palmCenterX - predictedX,
              measurement.palmCenterY - predictedY
            );
            const matchGate =
              Math.max(
                0.14,
                Math.max(
                  track.boxWidth,
                  track.boxHeight,
                  measurement.boxWidth,
                  measurement.boxHeight
                ) * 1.45
              ) +
              Math.min(ageMs / this.maxAgeMs, 1) * 0.08;
            return { track, distance, matchGate };
          })
          .filter(candidate => candidate.distance <= candidate.matchGate)
          .sort((first, second) => first.distance - second.distance)[0]?.track;

        if (nearestTrack) {
          this.tracks.delete(nearestTrack.trackingId);
          nearestTrack.trackingId = trackingId;
          this.tracks.set(trackingId, nearestTrack);
          previous = nearestTrack;
        }
      }

      if (!previous) {
        if (this.tracks.size >= 2) {
          const oldestUnclaimed = Array.from(this.tracks.values())
            .filter(
              track =>
                !claimedTracks.has(track) && !incomingIds.has(track.trackingId)
            )
            .sort((first, second) => first.lastSeenAt - second.lastSeenAt)[0];
          if (oldestUnclaimed) this.tracks.delete(oldestUnclaimed.trackingId);
        }
        previous = {
          trackingId,
          order: this.nextOrder++,
          ...measurement,
          velocityX: 0,
          velocityY: 0,
          lastSeenAt: now,
        };
        this.tracks.set(trackingId, previous);
      } else {
        this.applyMeasurement(previous, measurement, now);
      }
      claimedTracks.add(previous);
    }

    this.prune(now);
  }

  predict(
    frameWidth: number,
    frameHeight: number,
    now: number
  ): TrackedGloveRegion[] {
    if (
      !Number.isSafeInteger(frameWidth) ||
      !Number.isSafeInteger(frameHeight) ||
      frameWidth <= 0 ||
      frameHeight <= 0 ||
      !Number.isFinite(now)
    ) {
      return [];
    }

    this.prune(now);
    const minFrameSide = Math.min(frameWidth, frameHeight);
    return Array.from(this.tracks.values())
      .sort((first, second) => first.order - second.order)
      .map(track => {
        const ageMs = Math.max(0, now - track.lastSeenAt);
        const predictionMs = Math.min(ageMs, this.predictionHorizonMs);
        const centerX =
          (track.boxCenterX + track.velocityX * predictionMs) * frameWidth;
        const centerY =
          (track.boxCenterY + track.velocityY * predictionMs) * frameHeight;
        const boxWidth = track.boxWidth * frameWidth;
        const boxHeight = track.boxHeight * frameHeight;
        const speedPixelsPerMs = Math.hypot(
          track.velocityX * frameWidth,
          track.velocityY * frameHeight
        );
        const ageInflation = 1 + Math.min(ageMs / this.maxAgeMs, 1) * 0.28;
        const motionMargin = speedPixelsPerMs * predictionMs * 1.4;
        const side = clamp(
          Math.max(boxWidth, boxHeight) * this.paddingScale * ageInflation +
            motionMargin,
          minFrameSide * 0.16,
          minFrameSide
        );
        return {
          ...makeSquareRegion(
            centerX,
            centerY,
            side,
            frameWidth,
            frameHeight,
            boxWidth * boxHeight
          ),
          trackingId: track.trackingId,
          ageMs,
        };
      });
  }

  private applyMeasurement(
    track: TrackedHandBox,
    measurement: HandBoxMeasurement,
    now: number
  ): void {
    const elapsedMs = now - track.lastSeenAt;
    if (elapsedMs > 0) {
      const observedVelocityX = clamp(
        (measurement.palmCenterX - track.palmCenterX) / elapsedMs,
        -0.002,
        0.002
      );
      const observedVelocityY = clamp(
        (measurement.palmCenterY - track.palmCenterY) / elapsedMs,
        -0.002,
        0.002
      );
      track.velocityX = track.velocityX * 0.45 + observedVelocityX * 0.55;
      track.velocityY = track.velocityY * 0.45 + observedVelocityY * 0.55;
    }

    track.palmCenterX = measurement.palmCenterX;
    track.palmCenterY = measurement.palmCenterY;
    track.boxCenterX = measurement.boxCenterX;
    track.boxCenterY = measurement.boxCenterY;
    track.boxWidth = track.boxWidth * 0.3 + measurement.boxWidth * 0.7;
    track.boxHeight = track.boxHeight * 0.3 + measurement.boxHeight * 0.7;
    track.lastSeenAt = now;
  }

  private prune(now: number): void {
    this.tracks.forEach((track, trackingId) => {
      if (now - track.lastSeenAt > this.maxAgeMs) {
        this.tracks.delete(trackingId);
      }
    });
  }
}

/**
 * Removes at most one predicted region per full-frame detection. One-to-one
 * matching matters when two predicted crops overlap during a hand crossing.
 */
export function excludeDetectedGloveRegions(
  regions: ReadonlyArray<TrackedGloveRegion>,
  detections: ReadonlyArray<LandmarkDetection>,
  frameWidth: number,
  frameHeight: number
): TrackedGloveRegion[] {
  if (regions.length === 0 || detections.length === 0) return [...regions];
  if (frameWidth <= 0 || frameHeight <= 0) return [...regions];

  const matches: Array<{
    detectionIndex: number;
    regionIndex: number;
    distance: number;
  }> = [];
  for (
    let detectionIndex = 0;
    detectionIndex < detections.length;
    detectionIndex++
  ) {
    const center = palmCenter(
      detections[detectionIndex],
      frameWidth,
      frameHeight
    );
    if (!center) continue;
    for (let regionIndex = 0; regionIndex < regions.length; regionIndex++) {
      const region = regions[regionIndex];
      const regionCenterX = region.x + region.width * 0.5;
      const regionCenterY = region.y + region.height * 0.5;
      const normalizedDistance =
        Math.hypot(center.x - regionCenterX, center.y - regionCenterY) /
        Math.max(region.width, region.height, 1);
      if (normalizedDistance <= 0.78) {
        matches.push({
          detectionIndex,
          regionIndex,
          distance: normalizedDistance,
        });
      }
    }
  }

  matches.sort((first, second) => first.distance - second.distance);
  const assignedDetections = new Set<number>();
  const assignedRegions = new Set<number>();
  for (const match of matches) {
    if (
      assignedDetections.has(match.detectionIndex) ||
      assignedRegions.has(match.regionIndex)
    ) {
      continue;
    }
    assignedDetections.add(match.detectionIndex);
    assignedRegions.add(match.regionIndex);
  }

  return regions.filter((_, index) => !assignedRegions.has(index));
}
