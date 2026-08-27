/**
 * Cross-language golden-file parity test for the TS SDK.
 *
 * Reads the JSON fixtures at `tests/golden/fixtures/` (canonical Python
 * output) and asserts the TS SDK produces row-for-row equivalent results
 * after the documented normalization. Per #176 / docs/NULL_CONVENTIONS.md.
 */
import { readFile } from "fs/promises";
import { join } from "path";
import { describe, expect, it } from "vitest";
import {
  CoincidencesDb,
  GROUND,
  RadiationDb,
  type CoincidenceEntry,
  type EmissionEntry,
} from "../src/index.js";

const META_DIR = join(import.meta.dirname, "../../../../data/meta");
const FIXTURE_DIR = join(import.meta.dirname, "../../../../tests/golden/fixtures");

const FLOAT_PRECISION = 6;

function round6(v: number): number | string {
  if (!Number.isFinite(v)) {
    if (v === Infinity) return "Infinity";
    if (v === -Infinity) return "-Infinity";
    throw new Error(`NaN encountered`);
  }
  return Math.round(v * 1e6) / 1e6;
}

function optRound(v: number | null): number | string | null {
  return v == null ? null : round6(v);
}

async function loadFixture(name: string): Promise<unknown[]> {
  const path = join(FIXTURE_DIR, `${name}.json`);
  const text = await readFile(path, "utf8");
  return JSON.parse(text);
}

function coincToObj(e: CoincidenceEntry): Record<string, unknown> {
  return {
    Z: e.z,
    A: e.a,
    parent_state: e.parentState,
    parent_decay_mode: e.parentDecayMode,
    daughter_ex_keV: optRound(e.daughterExKeV),
    emission1_rad_type: e.emission1.radType,
    emission1_energy_keV: round6(e.emission1.energyKeV),
    emission1_intensity: round6(e.emission1.intensity),
    emission1_shell: e.emission1.shell,
    emission2_rad_type: e.emission2.radType,
    emission2_energy_keV: round6(e.emission2.energyKeV),
    emission2_intensity: round6(e.emission2.intensity),
    emission2_shell: e.emission2.shell,
    pair_intensity: round6(e.pairIntensity),
  };
}

type CoincRow = ReturnType<typeof coincToObj>;

function sortCoinc(rows: CoincRow[]): CoincRow[] {
  // Matches the Python normalize._COINC_SORT_KEY tuple.
  return rows.slice().sort((a, b) => {
    const cmp = (x: unknown, y: unknown): number => {
      if (x === y) return 0;
      if (x == null) return -1;
      if (y == null) return 1;
      if (typeof x === "string" && typeof y === "string") return x < y ? -1 : x > y ? 1 : 0;
      if (typeof x === "number" && typeof y === "number") return x - y;
      return String(x) < String(y) ? -1 : 1;
    };
    return (
      cmp(a.Z, b.Z) ||
      cmp(a.A, b.A) ||
      cmp(a.parent_state, b.parent_state) ||
      cmp(a.parent_decay_mode, b.parent_decay_mode) ||
      cmp(a.emission1_rad_type, b.emission1_rad_type) ||
      cmp(a.emission1_energy_keV, b.emission1_energy_keV) ||
      cmp(a.emission2_rad_type, b.emission2_rad_type) ||
      cmp(a.emission2_energy_keV, b.emission2_energy_keV) ||
      cmp(a.daughter_ex_keV, b.daughter_ex_keV) ||
      cmp(a.emission1_shell, b.emission1_shell) ||
      cmp(a.emission2_shell, b.emission2_shell) ||
      cmp(a.pair_intensity, b.pair_intensity)
    );
  });
}

describe("golden parity (#176)", () => {
  it("Co-60 β-γ matches Python fixture", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const rows = await db.pairsFiltered(28, 60, {
      parentState: "",
      parentDecayMode: "beta-",
      emission1RadType: "beta",
      emission2RadType: "gamma",
    });
    const actual = sortCoinc(rows.map(coincToObj));
    const golden = await loadFixture("co60_beta_gamma");
    expect(actual).toEqual(golden);
  });

  it("Y-86 K X-ray ⊗ γ matches Python fixture", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const rows = await db.pairsFiltered(38, 86, {
      parentState: "",
      parentDecayMode: "KshellEC",
      emission1RadType: "xray",
      emission2RadType: "gamma",
      minIntensity: 1e-5,
    });
    const actual = sortCoinc(rows.map(coincToObj));
    const golden = await loadFixture("y86_kshell_xray_gamma");
    expect(actual).toEqual(golden);
  });

  it("Co-60 γ-γ regression matches Python fixture", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const rows = await db.pairsFiltered(28, 60, {
      parentState: "",
      parentDecayMode: "beta-",
      emission1RadType: "gamma",
      emission2RadType: "gamma",
      minIntensity: 0.5,
    });
    const actual = sortCoinc(rows.map(coincToObj));
    const golden = await loadFixture("co60_gamma_gamma");
    expect(actual).toEqual(golden);
  });

  it("Sr-90 → Y-90 pure β⁻ matches the empty Python fixture", async () => {
    const db = await CoincidencesDb.open(META_DIR);
    const pairs = await db.pairsFiltered(39, 90, {
      parentDecayMode: "beta-",
    });
    const mixed = pairs.filter((e) => e.emission1.radType !== "gamma");
    const actual = sortCoinc(mixed.map(coincToObj));
    const golden = await loadFixture("sr90_y90_negative");
    expect(actual).toEqual(golden);
  });

  it("Ni-60 emissions match Python fixture (daughter-keyed)", async () => {
    const db = await RadiationDb.open(META_DIR);
    // `""` here matched nothing once #380 retired that spelling, so this
    // compared an empty array against a 619-row fixture. The Python side of
    // this parity pair defaults to `GROUND` too — that is the point of the
    // fixture, so both must ask the same question.
    const lines = (await db.emissions(28, 60, GROUND)).filter((e) => e.intensityPct >= 5.0);
    const actual = lines
      .map((e: EmissionEntry) => ({
        Z: e.z,
        A: e.a,
        state: e.state,
        rad_type: e.radType,
        energy_keV: round6(e.energyKeV),
        intensity_pct: round6(e.intensityPct),
        decay_mode: e.decayMode,
        rad_subtype: e.radSubtype,
        icc_total: optRound(e.iccTotal),
        vacancy_shell: e.vacancyShell,
      }))
      .sort((a, b) => {
        if (a.Z !== b.Z) return a.Z - b.Z;
        if (a.A !== b.A) return a.A - b.A;
        if (a.state !== b.state) return a.state < b.state ? -1 : 1;
        if (a.rad_type !== b.rad_type) return a.rad_type < b.rad_type ? -1 : 1;
        const ax = typeof a.energy_keV === "number" ? a.energy_keV : 0;
        const bx = typeof b.energy_keV === "number" ? b.energy_keV : 0;
        return ax - bx;
      });
    const golden = await loadFixture("ni60_emissions");
    expect(actual).toEqual(golden);
  });

  it("identifyGamma(1173.2) matches Python fixture", { timeout: 60_000 }, async () => {
    const db = await RadiationDb.open(META_DIR);
    const cands = await db.identifyGamma(1173.2, 1.0, 0.1);
    // Re-sort with the Python helper's normalization: rounded |delta|,
    // -intensity, then (Z, A) as final tiebreaker.
    cands.sort((a, b) => {
      const da = Math.round(Math.abs(a.deltaKeV) * 1e6) / 1e6;
      const db_ = Math.round(Math.abs(b.deltaKeV) * 1e6) / 1e6;
      if (da !== db_) return da - db_;
      if (a.intensityPct !== b.intensityPct) return b.intensityPct - a.intensityPct;
      if (a.z !== b.z) return a.z - b.z;
      return a.a - b.a;
    });
    const golden = (await loadFixture("identify_gamma_1173keV")) as unknown[];
    const actual = cands.slice(0, golden.length).map((c) => ({
      Z: c.z,
      A: c.a,
      state: c.state,
      energy_keV: round6(c.energyKeV),
      intensity_pct: round6(c.intensityPct),
      delta_keV: round6(c.deltaKeV),
    }));
    expect(actual).toEqual(golden);
  });
});
