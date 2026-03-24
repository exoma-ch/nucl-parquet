/**
 * Log-log interpolation on sorted x/y arrays.
 *
 * Both arrays must be the same length and sorted ascending by x.
 * Returns NaN if x <= 0 or arrays are empty.
 * Clamps to boundary values outside the tabulated range.
 * Falls back to linear interpolation when either bracketing value is <= 0.
 */
export function logLogInterp(xs: Float64Array | number[], ys: Float64Array | number[], x: number): number {
  const n = xs.length;
  if (n === 0 || x <= 0) return NaN;

  // Clamp to table range
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];

  // Binary search for the bracketing interval
  let lo = 0;
  let hi = n - 1;
  while (lo < hi - 1) {
    const mid = (lo + hi) >>> 1;
    if (xs[mid] <= x) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  // Exact match check
  if (xs[lo] === x) return ys[lo];
  if (xs[hi] === x) return ys[hi];

  const e0 = xs[lo];
  const e1 = xs[hi];
  const v0 = ys[lo];
  const v1 = ys[hi];

  // Handle zero or negative values (can't take log)
  if (v0 <= 0 || v1 <= 0) {
    const t = (x - e0) / (e1 - e0);
    return v0 + t * (v1 - v0);
  }

  // Log-log interpolation
  const logX = Math.log(x);
  const logE0 = Math.log(e0);
  const logE1 = Math.log(e1);
  const logV0 = Math.log(v0);
  const logV1 = Math.log(v1);

  const t = (logX - logE0) / (logE1 - logE0);
  return Math.exp(logV0 + t * (logV1 - logV0));
}
