import { describe, expect, it } from "vitest";
import { closestProperRotation } from "./rotationFit";
const I = [
  [1, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
];
const transpose = (a: number[][]) => a[0].map((_, c) => a.map(r => r[c]));
const mul = (a: number[][], b: number[][]) =>
  a.map(row => b[0].map((_, c) => row.reduce((n, v, k) => n + v * b[k][c], 0)));
const det = ([[a, b, c], [d, e, f], [g, h, i]]: number[][]) =>
  a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
function close(a: number[][], b: number[][]) {
  a.forEach((r, i) => r.forEach((v, j) => expect(v).toBeCloseTo(b[i][j], 8)));
}
describe("proper rotation fitting", () => {
  it("recovers rotations including 180 degrees with nonuniform axis weights", () => {
    for (const angle of [0, 0.01, 37, 90, 177, 180]) {
      const t = (angle * Math.PI) / 180,
        c = Math.cos(t),
        s = Math.sin(t);
      const r = [
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1],
      ];
      close(
        closestProperRotation(
          mul(r, [
            [1, 0, 0],
            [0, 3, 0],
            [0, 0, 2],
          ])
        )!,
        r
      );
    }
  });
  it("handles negative determinant covariance with the best proper rotation, never a reflection", () => {
    const left = [
      [0, 0, 1],
      [1, 0, 0],
      [0, 1, 0],
    ];
    const right = [
      [0, -1, 0],
      [1, 0, 0],
      [0, 0, 1],
    ];
    const a = mul(
      mul(left, [
        [-0.3, 0, 0],
        [0, 1.1, 0],
        [0, 0, 2.4],
      ]),
      transpose(right)
    );
    expect(det(a)).toBeLessThan(0);
    const fit = closestProperRotation(a)!;
    close(fit, mul(left, transpose(right)));
    close(mul(fit, transpose(fit)), I);
    expect(det(fit)).toBeCloseTo(1, 9);
  });
  it("returns a rotation for the reported covariance while preserving orthonormality", () => {
    const fit = closestProperRotation([
      [-2.55, 3.69, 1.44],
      [-0.71, -0.54, 0.46],
      [0.19, -6.56, -1.74],
    ])!;
    expect(fit).not.toBeNull();
    close(mul(fit, transpose(fit)), I);
    expect(det(fit)).toBeCloseTo(1, 9);
  });
  it("rejects undefined orientation from one axis, zero information, or nonfinite data", () => {
    expect(
      closestProperRotation([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ])
    ).toBeNull();
    expect(
      closestProperRotation([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ])
    ).toBeNull();
    expect(
      closestProperRotation([
        [NaN, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ])
    ).toBeNull();
    expect(closestProperRotation([])).toBeNull();
    // A unique orientation is still determined by two independent observations.
    close(
      closestProperRotation([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
      ])!,
      I
    );
  });
});
