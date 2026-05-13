import { join } from "path";
import { describe, expect, it } from "vitest";
import { CoincidencesDb } from "../src/coincidences.js";

const META_DIR = join(import.meta.dirname, "../../../../data/meta");

describe("CoincidencesDb (#175 acceptance)", () => {
  it("Co-60 β⁻ ⊗ 1173 keV γ pair has pair_intensity ≈ 0.9986", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const pairs = await db.pairs(28, 60); // daughter Ni-60
    const hit = pairs.find(
      (e) =>
        e.emission1.radType === "beta" &&
        Math.abs(e.emission2.energyKeV - 1173.0) < 2.0 &&
        e.parentDecayMode === "beta-",
    );
    expect(hit, "Co-60 β⁻ ⊗ 1173 keV γ pair must exist").toBeDefined();
    expect(hit!.pairIntensity).toBeGreaterThan(0.9956);
    expect(hit!.pairIntensity).toBeLessThan(1.001);
  });

  it("Y-86 K X-ray ⊗ 1077 keV γ pair ships with emission1_shell='K'", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const pairs = await db.pairs(38, 86); // daughter Sr-86
    const hit = pairs.find(
      (e) =>
        e.emission1.radType === "xray" &&
        e.emission1.shell === "K" &&
        e.parentDecayMode === "KshellEC" &&
        Math.abs(e.emission2.energyKeV - 1077.0) < 2.0,
    );
    expect(hit, "Y-86 K X-ray ⊗ 1077 keV γ pair must exist").toBeDefined();
  });

  it("v0.11 Co-60 1173/1333 γ-γ cascade preserved", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const pairs = await db.pairs(28, 60);
    const canonical = pairs.some(
      (e) =>
        e.emission1.radType === "gamma" &&
        e.emission2.radType === "gamma" &&
        ((Math.abs(e.emission1.energyKeV - 1173.0) < 2.0 &&
          Math.abs(e.emission2.energyKeV - 1333.0) < 2.0) ||
          (Math.abs(e.emission1.energyKeV - 1333.0) < 2.0 &&
            Math.abs(e.emission2.energyKeV - 1173.0) < 2.0)),
    );
    expect(canonical, "Co-60 1173/1333 γ-γ cascade must be preserved").toBe(true);
  });

  it("Sr-90 → Y-90 pure β⁻ produces zero mixed pairs", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const mixed = await db.pairsFiltered(39, 90, {
      parentDecayMode: "beta-",
    });
    const nonGamma = mixed.filter((e) => e.emission1.radType !== "gamma");
    expect(
      nonGamma.length,
      `Sr-90 → Y-90 pure β⁻ must produce zero mixed pairs (got ${nonGamma.length})`,
    ).toBe(0);
  });
});
