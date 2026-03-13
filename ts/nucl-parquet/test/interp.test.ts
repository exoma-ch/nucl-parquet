import { describe, expect, it } from "vitest";
import { logLogInterp } from "../src/interp.js";

describe("logLogInterp", () => {
  it("returns exact match", () => {
    const e = [1.0, 10.0, 100.0];
    const v = [100.0, 10.0, 1.0];
    expect(logLogInterp(e, v, 10.0)).toBe(10.0);
  });

  it("interpolates power law exactly (midpoint)", () => {
    // For a perfect power law v = 1000/E, log-log interp is exact
    const e = [1.0, 1000.0];
    const v = [1000.0, 1.0];
    const result = logLogInterp(e, v, 10.0);
    // v(10) should be 100
    expect(Math.abs(result - 100.0)).toBeLessThan(1e-10);
  });

  it("clamps below range", () => {
    const e = [1.0, 10.0];
    const v = [100.0, 10.0];
    expect(logLogInterp(e, v, 0.5)).toBe(100.0);
  });

  it("clamps above range", () => {
    const e = [1.0, 10.0];
    const v = [100.0, 10.0];
    expect(logLogInterp(e, v, 20.0)).toBe(10.0);
  });

  it("returns NaN for empty arrays", () => {
    expect(logLogInterp([], [], 1.0)).toBeNaN();
  });

  it("returns NaN for zero energy", () => {
    const e = [1.0, 10.0];
    const v = [100.0, 10.0];
    expect(logLogInterp(e, v, 0.0)).toBeNaN();
  });

  it("returns NaN for negative energy", () => {
    const e = [1.0, 10.0];
    const v = [100.0, 10.0];
    expect(logLogInterp(e, v, -5.0)).toBeNaN();
  });

  it("falls back to linear interpolation for zero values", () => {
    const e = [1.0, 2.0, 3.0];
    const v = [0.0, 5.0, 10.0];
    // Between e[0]=1 and e[1]=2, v0=0 so linear fallback
    const result = logLogInterp(e, v, 1.5);
    // linear: 0 + 0.5 * (5 - 0) = 2.5
    expect(result).toBe(2.5);
  });

  it("handles single element array", () => {
    const e = [5.0];
    const v = [42.0];
    // Any positive energy clamps to the single value
    expect(logLogInterp(e, v, 1.0)).toBe(42.0);
    expect(logLogInterp(e, v, 5.0)).toBe(42.0);
    expect(logLogInterp(e, v, 100.0)).toBe(42.0);
  });

  it("works with Float64Array inputs", () => {
    const e = new Float64Array([1.0, 1000.0]);
    const v = new Float64Array([1000.0, 1.0]);
    const result = logLogInterp(e, v, 10.0);
    expect(Math.abs(result - 100.0)).toBeLessThan(1e-10);
  });
});
