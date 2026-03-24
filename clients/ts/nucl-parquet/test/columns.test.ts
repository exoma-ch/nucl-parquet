import { readFile } from "fs/promises";
import { join } from "path";
import { describe, expect, it } from "vitest";
import { stoppingColumns, xsColumns } from "../src/columns.js";

const DATA_DIR = join(import.meta.dirname, "../../../../data");

describe("stoppingColumns", () => {
  it("returns typed arrays for stopping.parquet", async () => {
    const buf = await readFile(join(DATA_DIR, "stopping/stopping.parquet"));
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

    // Sanity: sources include PSTAR
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
