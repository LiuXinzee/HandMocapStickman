export const GLB_FINGER_COUNT = 5;
export const GLB_SEGMENTS_PER_FINGER = 4;
export const GLB_MAX_BEND_DEGREES = 120;

export type GlbQuaternion = readonly [number, number, number, number];

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

/**
 * Accept either one bend angle per finger (5 values) or one per GLB finger
 * bone (20 values, ordered finger-major). Invalid and missing values become 0.
 */
export function expandGlbFingerBends(
  bends: readonly number[] | null | undefined
): number[][] {
  const values = bends ?? [];
  const hasPerBoneValues =
    values.length >= GLB_FINGER_COUNT * GLB_SEGMENTS_PER_FINGER;

  return Array.from({ length: GLB_FINGER_COUNT }, (_, fingerIndex) =>
    Array.from({ length: GLB_SEGMENTS_PER_FINGER }, (_, segmentIndex) => {
      const sourceIndex = hasPerBoneValues
        ? fingerIndex * GLB_SEGMENTS_PER_FINGER + segmentIndex
        : fingerIndex;
      const value = values[sourceIndex];
      return Number.isFinite(value) ? clamp(value, 0, GLB_MAX_BEND_DEGREES) : 0;
    })
  );
}

/** Convert the glove protocol's [w, x, y, z] tuple to a unit quaternion. */
export function normalizeGlbQuaternion(
  quaternion: GlbQuaternion | null | undefined
): [number, number, number, number] | null {
  if (!quaternion?.every(Number.isFinite)) return null;

  const magnitude = Math.hypot(...quaternion);
  if (magnitude < 1e-8) return null;

  return quaternion.map(value => value / magnitude) as [
    number,
    number,
    number,
    number,
  ];
}
