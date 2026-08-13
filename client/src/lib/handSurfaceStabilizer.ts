import type { GloveSurface } from "./gloveVision";

export interface HandSurfaceState {
  score: number;
  surface: GloveSurface;
}

function clamp01(value: number): number {
  return Math.min(1, Math.max(0, value));
}

export function stabilizeHandSurface(
  previous: HandSurfaceState | undefined,
  observedSurface: GloveSurface | undefined,
  observedConfidence: number
): HandSurfaceState {
  let score = previous?.score ?? 0;
  let surface = previous?.surface ?? "unknown";

  if (observedSurface && observedSurface !== "unknown") {
    const confidence = Number.isFinite(observedConfidence)
      ? clamp01(observedConfidence)
      : 0;
    const evidence = observedSurface === "palm" ? confidence : -confidence;
    const blend =
      !previous || surface === "unknown"
        ? 1
        : observedSurface === surface
          ? 0.28
          : 0.16;
    score = previous
      ? previous.score * (1 - blend) + evidence * blend
      : evidence;
  }

  if (surface === "unknown") {
    surface = score > 0.12 ? "palm" : score < -0.12 ? "back" : "unknown";
  } else if (surface === "palm" && score < -0.28) {
    surface = "back";
  } else if (surface === "back" && score > 0.28) {
    surface = "palm";
  }

  return { score, surface };
}
