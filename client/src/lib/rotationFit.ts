/**
 * Closest proper rotation for A = sum(target * measured^T).
 * Davenport's q method maximizes q^T K q over unit quaternions.
 * Unlike polar iteration with a positive-determinant precondition, it can
 * return the best proper rotation even when observations disagree.
 * Reference: https://ntrs.nasa.gov/citations/19990104598
 */
export function closestProperRotation(a: number[][]): number[][] | null {
  if (
    a.length !== 3 ||
    a.some(row => row.length !== 3 || row.some(v => !Number.isFinite(v)))
  )
    return null;
  const scale = Math.max(...a.flat().map(Math.abs));
  if (scale < 1e-12) return null;
  const b = a.map(row => row.map(v => v / scale));
  const trace = b[0][0] + b[1][1] + b[2][2];
  const z = [b[2][1] - b[1][2], b[0][2] - b[2][0], b[1][0] - b[0][1]];
  const k: number[][] = Array.from({ length: 4 }, () => [0, 0, 0, 0]);
  k[0][0] = trace;
  for (let r = 0; r < 3; r++) {
    k[0][r + 1] = k[r + 1][0] = z[r];
    for (let c = 0; c < 3; c++)
      k[r + 1][c + 1] = b[r][c] + b[c][r] - (r === c ? trace : 0);
  }
  const vectors: number[][] = Array.from({ length: 4 }, (_, r) =>
    Array.from({ length: 4 }, (_, c) => (r === c ? 1 : 0))
  );
  // Symmetric Jacobi diagonalization: choose the largest algebraic eigenvalue,
  // not the largest magnitude (negative eigenvalues are common here).
  for (let iteration = 0; iteration < 100; iteration++) {
    let p = 0,
      q = 1;
    for (let r = 0; r < 4; r++)
      for (let c = r + 1; c < 4; c++)
        if (Math.abs(k[r][c]) > Math.abs(k[p][q])) {
          p = r;
          q = c;
        }
    if (Math.abs(k[p][q]) < 1e-13) break;
    const theta = 0.5 * Math.atan2(2 * k[p][q], k[q][q] - k[p][p]);
    const c = Math.cos(theta),
      s = Math.sin(theta);
    const pp = k[p][p],
      qq = k[q][q],
      pq = k[p][q];
    for (let i = 0; i < 4; i++) {
      if (i !== p && i !== q) {
        const ip = k[i][p],
          iq = k[i][q];
        k[i][p] = k[p][i] = c * ip - s * iq;
        k[i][q] = k[q][i] = s * ip + c * iq;
      }
      const vp = vectors[i][p],
        vq = vectors[i][q];
      vectors[i][p] = c * vp - s * vq;
      vectors[i][q] = s * vp + c * vq;
    }
    k[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq;
    k[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq;
    k[p][q] = k[q][p] = 0;
  }
  const order = [0, 1, 2, 3].sort((i, j) => k[j][j] - k[i][i]);
  // Identical best eigenvalues leave the orientation undetermined (e.g. one axis).
  if (k[order[0]][order[0]] - k[order[1]][order[1]] < 1e-9) return null;
  const raw = vectors.map(row => row[order[0]]);
  const norm = Math.hypot(...raw);
  const [w, x, y, zq] = raw.map(v => v / norm);
  return [
    [1 - 2 * (y * y + zq * zq), 2 * (x * y - w * zq), 2 * (x * zq + w * y)],
    [2 * (x * y + w * zq), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - w * x)],
    [2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)],
  ];
}
