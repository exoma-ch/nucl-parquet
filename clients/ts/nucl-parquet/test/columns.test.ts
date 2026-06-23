import { readFile } from "fs/promises";
import { join } from "path";
import { describe, expect, it } from "vitest";
import { catimaColumns, stoppingColumns, xsColumns } from "../src/columns.js";

const DATA_DIR = join(import.meta.dirname, "../../../../data");

describe("stoppingColumns", () => {
  it("returns typed arrays for a per-source stopping parquet", async () => {
    // The unprovenanced stopping.parquet aggregate was removed in #143 (#137).
    // Per-source files (PSTAR/ASTAR/ESTAR/dSTAR/tSTAR/catima_*) are the canonical
    // shape; full TS-client cleanup of docstrings + sample paths is #147.
    const buf = await readFile(join(DATA_DIR, "stopping/PSTAR.parquet"));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const cols = await stoppingColumns(ab);

    expect(cols.source).toBeInstanceOf(Array);
    expect(cols.targetZ).toBeInstanceOf(Int32Array);
    expect(cols.energyMeV).toBeInstanceOf(Float64Array);
    expect(cols.dedx).toBeInstanceOf(Float64Array);

    // All columns same length
    const n = cols.source.length;
    expect(n).toBeGreaterThan(0);
    expect(cols.targetZ.length).toBe(n);
    expect(cols.energyMeV.length).toBe(n);
    expect(cols.dedx.length).toBe(n);

    // Sanity: PSTAR file contains PSTAR-source rows
    expect(cols.source).toContain("PSTAR");

    // Sanity: all dedx > 0
    expect(cols.dedx.every((v) => v > 0)).toBe(true);
  });
});

describe("xsColumns", () => {
  it("returns typed arrays for a neutron XS file", async () => {
    const buf = await readFile(join(DATA_DIR, "endfb-8.1/xs/n_Cu.parquet"));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const cols = await xsColumns(ab);

    expect(cols.targetA).toBeInstanceOf(Int32Array);
    expect(cols.residualZ).toBeInstanceOf(Int32Array);
    expect(cols.residualA).toBeInstanceOf(Int32Array);
    expect(cols.state).toBeInstanceOf(Array);
    expect(cols.energyMeV).toBeInstanceOf(Float64Array);
    expect(cols.xsMb).toBeInstanceOf(Float64Array);

    const n = cols.targetA.length;
    expect(n).toBeGreaterThan(100);
    expect(cols.energyMeV.length).toBe(n);
    expect(cols.xsMb.length).toBe(n);

    // Cu-63 and Cu-65 should be present
    expect(Array.from(new Set(cols.targetA))).toEqual(expect.arrayContaining([63, 65]));
  });
});

describe("catimaColumns", () => {
  it("surfaces proj_A so isotopes are distinguishable (#248)", async () => {
    const buf = await readFile(join(DATA_DIR, "stopping/catima/catima.parquet"));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    const cols = await catimaColumns(ab);

    expect(cols.projZ).toBeInstanceOf(Int32Array);
    expect(cols.projA).toBeInstanceOf(Int32Array);
    expect(cols.targetZ).toBeInstanceOf(Int32Array);
    expect(cols.energyMeVu).toBeInstanceOf(Float64Array);
    expect(cols.dedx).toBeInstanceOf(Float64Array);
    expect(cols.straggling).toBeInstanceOf(Float64Array);

    // All columns share one row count.
    const n = cols.projZ.length;
    expect(n).toBeGreaterThan(0);
    expect(cols.projA.length).toBe(n);
    expect(cols.targetZ.length).toBe(n);
    expect(cols.energyMeVu.length).toBe(n);
    expect(cols.dedx.length).toBe(n);
    expect(cols.straggling.length).toBe(n);

    // Multiple isotopes of He (Z=2) — both A=3 and A=4 must be present, else a
    // (projZ, targetZ) lookup would silently merge them (the #246/#248 bug).
    const heMassNumbers = new Set<number>();
    for (let i = 0; i < n; i++) {
      if (cols.projZ[i] === 2) heMassNumbers.add(cols.projA[i]);
    }
    expect(Array.from(heMassNumbers)).toEqual(expect.arrayContaining([3, 4]));
  });
});
