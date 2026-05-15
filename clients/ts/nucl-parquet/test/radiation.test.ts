import { join } from "path";
import { describe, expect, it } from "vitest";
import { RadiationDb } from "../src/radiation.js";

const META_DIR = join(import.meta.dirname, "../../../../data/meta");

describe("RadiationDb (#175 acceptance)", () => {
  it("Ni-60 emissions include the 1173 + 1333 keV γ (Co-60 β⁻ daughter convention)", async () => {
    const db = await RadiationDb.open(META_DIR);
    const lines = await db.emissions(28, 60, "");
    expect(lines.length).toBeGreaterThan(0);
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
}, /* longer timeout for the eager γ index build */ { timeout: 60_000 });
