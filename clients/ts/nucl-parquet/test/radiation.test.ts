import { join } from "path";
import { describe, expect, it } from "vitest";
import { GROUND, RadiationDb } from "../src/radiation.js";

const META_DIR = join(import.meta.dirname, "../../../../data/meta");

// Options go in the SECOND argument. The three-argument form
// `describe(name, fn, options)` was deprecated in Vitest 3 and removed in 4.
describe("RadiationDb (#175 acceptance)", { timeout: 60_000 }, () => {
  it("Ni-60 emissions include the 1173 + 1333 keV γ (Co-60 β⁻ daughter convention)", async () => {
    const db = await RadiationDb.open(META_DIR);
    const lines = await db.emissions(28, 60, GROUND);
    expect(lines.length).toBe(758);
    const has1173 = lines.some(
      (e) => e.radType === "gamma" && Math.abs(e.energyKeV - 1173.2) < 0.5,
    );
    const has1333 = lines.some(
      (e) => e.radType === "gamma" && Math.abs(e.energyKeV - 1332.5) < 0.5,
    );
    expect(has1173 && has1333).toBe(true);
  });

  it("identifyGamma(1173.2) returns Ni-60 as a candidate", async () => {
    const db = await RadiationDb.open(META_DIR);
    const candidates = await db.identifyGamma(1173.2, 1.0, 5.0);
    const foundNi60 = candidates.some((c) => c.z === 28 && c.a === 60);
    expect(
      foundNi60,
      `identifyGamma(1173.2) must include Ni-60 (got ${candidates.length} candidates)`,
    ).toBe(true);
  });
  it("the default state reaches rows on the shipped data", async () => {
    // The default was `""`, which #380 retired. Every rebuilt row carries
    // `g`/`m`/`m2`/`m3` or null, so `emissions(28, 60)` matched nothing and
    // returned an empty array — the same defect this repo fixed in
    // `nucl_parquet/loader.py`. Asserting a count, not `toBeDefined()`:
    // an empty array is exactly what the bug produced.
    const db = await RadiationDb.open(META_DIR);
    expect(await db.emissions(28, 60)).toHaveLength(758);
    expect(await db.emissionsFiltered(28, 60, undefined, "gamma")).not.toHaveLength(0);
    expect(await db.emissions(28, 60, "")).toHaveLength(0);
  });

  it("keeps a null state null rather than spelling it as the ground state", async () => {
    // 26 radiation rows state no isomeric state. Coercing them to `""` — or to
    // `g` — would assert something ENSDF does not, and `columns.ts` already
    // refuses that for the same column.
    const db = await RadiationDb.open(META_DIR);
    const lines = await db.emissions(28, 60, GROUND);
    expect(lines.every((e) => e.state === GROUND)).toBe(true);
    expect(await db.emissions(28, 60, null)).toHaveLength(0);
  });
}); // 60s: the eager γ index build dominates this suite
