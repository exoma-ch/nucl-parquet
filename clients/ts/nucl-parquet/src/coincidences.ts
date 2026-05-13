/**
 * Coincidence pairs — γ-γ + mixed-emission (β/EC X-ray/Auger/511 keV ⊗ γ)
 * cascade pairs from the augmented schema in #170.
 *
 * Mirrors the Rust `CoincidencesDb` (`clients/rs/nucl-parquet/src/meta.rs`)
 * and the Python `nucl_parquet.coincidences()` helper.
 *
 * **Daughter-keyed convention.** Each row is filed under the daughter
 * nucleus where the γ cascade lives, not the parent that decayed. The
 * Co-60 (parent) β⁻ → Ni-60 cascade pairs are under `(Z=28, A=60)`.
 *
 * **Lazy per-element loading.** `open()` does no file I/O; per-Symbol
 * files are loaded + cached on first `pairsForElement(z)` access. Memory
 * cost is bounded by elements queried, not the full 67 MB dataset.
 */
import { parquetRead } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { zToSymbol } from "./z_symbols.js";

/**
 * One side of a coincidence pair — a single emission line.
 *
 * `radType` values: `"gamma" | "beta" | "xray" | "auger" | "annihilation_511"`.
 * `shell` is populated only for X-ray and Auger emissions (K/L/M/N).
 * `iccTotal` is populated only for γ emissions; `null` otherwise.
 */
export interface Emission {
  radType: string;
  energyKeV: number;
  intensity: number;
  shell: string | null;
  iccTotal: number | null;
}

/**
 * A single coincidence pair — two emissions in one parent decay event.
 *
 * `parentDecayMode` is `null` for γ-γ rows where the cascade enters via
 * deeper cascading rather than direct parent feeding.
 *
 * The `parentLevelKeV` / `intermediateLevelKeV` / `finalLevelKeV` cascade
 * structure is populated for γ-γ rows; mixed-emission rows have
 * `finalLevelKeV = null`.
 */
export interface CoincidenceEntry {
  /** Daughter Z (where the cascade γ are emitted). */
  z: number;
  /** Daughter A. */
  a: number;
  /** Parent isomeric state: `""` (ground) | `"m"` | `"m2"`. */
  parentState: string;
  /** Parent decay channel: `"beta-" | "beta+" | "KshellEC" | ...`. */
  parentDecayMode: string | null;
  /** Daughter level fed by the parent decay (keV). */
  daughterExKeV: number | null;
  /** γ-γ cascade head level (keV); null for mixed-emission rows. */
  parentLevelKeV: number | null;
  /** γ-γ intermediate level (keV); null for mixed-emission rows. */
  intermediateLevelKeV: number | null;
  /** γ-γ cascade final level (keV); null for mixed-emission rows. */
  finalLevelKeV: number | null;
  emission1: Emission;
  emission2: Emission;
  /** Relative pair intensity per parent decay. */
  pairIntensity: number;
}

/** Filter for `CoincidencesDb.pairsFiltered`. */
export interface CoincidenceFilter {
  parentState?: string;
  parentDecayMode?: string;
  emission1RadType?: string;
  emission2RadType?: string;
  /** Drop pairs with `pairIntensity <= minIntensity`. */
  minIntensity?: number;
}

async function loadFileAsArrayBuffer(path: string): Promise<ArrayBuffer> {
  const { readFile } = await import("fs/promises");
  const buffer = await readFile(path);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

async function readParquetFile(path: string): Promise<Record<string, unknown>[]> {
  const arrayBuffer = await loadFileAsArrayBuffer(path);
  const flat: Record<string, unknown>[] = [];
  await parquetRead({
    file: {
      byteLength: arrayBuffer.byteLength,
      slice(start: number, end: number) {
        return arrayBuffer.slice(start, end);
      },
    },
    compressors,
    rowFormat: "object",
    onComplete(data: unknown) {
      // hyparquet returns rows as a flat array when rowFormat='object'.
      for (const row of data as Record<string, unknown>[]) flat.push(row);
    },
  });
  return flat;
}

async function dirExists(dir: string): Promise<boolean> {
  try {
    const { stat } = await import("fs/promises");
    const s = await stat(dir);
    return s.isDirectory();
  } catch {
    return false;
  }
}

function emissionFromRow(
  row: Record<string, unknown>,
  side: 1 | 2,
  iccFieldName: string,
): Emission {
  const radType = (row[`emission${side}_rad_type`] as string | null) ?? "";
  const energy = row[`emission${side}_energy_keV`];
  const intensity = row[`emission${side}_intensity`];
  const shell = row[`emission${side}_shell`];
  const icc = row[iccFieldName];
  return {
    radType,
    energyKeV: typeof energy === "number" ? energy : Number(energy ?? 0),
    intensity: typeof intensity === "number" ? intensity : Number(intensity ?? 0),
    shell: shell == null ? null : (shell as string),
    iccTotal: icc == null ? null : Number(icc),
  };
}

function toNumOrNull(v: unknown): number | null {
  if (v == null) return null;
  return typeof v === "number" ? v : Number(v);
}

function toNum(v: unknown): number {
  if (v == null) return 0;
  return typeof v === "number" ? v : Number(v);
}

/**
 * Coincidence database — lazy per-element loading.
 *
 * @example
 * ```ts
 * const db = await CoincidencesDb.open("path/to/data/meta");
 * const co60 = await db.pairs(28, 60);  // daughter Ni-60
 * const beta = co60.filter(e => e.emission1.radType === "beta");
 * ```
 */
export class CoincidencesDb {
  private coincDir: string;
  private cache: Map<number, CoincidenceEntry[]> = new Map();

  private constructor(coincDir: string) {
    this.coincDir = coincDir;
  }

  /** Open `data/meta/ensdf/coincidences/`. Does no file I/O beyond directory check. */
  static async open(metaDir: string): Promise<CoincidencesDb> {
    const { join } = await import("path");
    const dir = join(metaDir, "ensdf", "coincidences");
    if (!(await dirExists(dir))) {
      throw new Error(`coincidences directory not found: ${dir}`);
    }
    return new CoincidencesDb(dir);
  }

  /** All coincidence pairs for daughter element Z (cached after first read). */
  async pairsForElement(z: number): Promise<CoincidenceEntry[]> {
    const cached = this.cache.get(z);
    if (cached) return cached;
    const symbol = zToSymbol(z);
    if (!symbol) throw new Error(`unknown Z=${z}`);
    const { join } = await import("path");
    const path = join(this.coincDir, `${symbol}.parquet`);
    const rows = await readParquetFile(path);
    const entries: CoincidenceEntry[] = rows.map((row) => ({
      z: toNum(row["Z"]),
      a: toNum(row["A"]),
      parentState: (row["parent_state"] as string | null) ?? "",
      parentDecayMode: (row["parent_decay_mode"] as string | null) ?? null,
      daughterExKeV: toNumOrNull(row["daughter_ex_keV"]),
      parentLevelKeV: toNumOrNull(row["parent_level_keV"]),
      intermediateLevelKeV: toNumOrNull(row["intermediate_level_keV"]),
      finalLevelKeV: toNumOrNull(row["final_level_keV"]),
      emission1: emissionFromRow(row, 1, "gamma1_icc_total"),
      emission2: emissionFromRow(row, 2, "gamma2_icc_total"),
      pairIntensity: toNum(row["pair_intensity"]),
    }));
    this.cache.set(z, entries);
    return entries;
  }

  /** Pairs for daughter (Z, A). */
  async pairs(z: number, a: number): Promise<CoincidenceEntry[]> {
    const all = await this.pairsForElement(z);
    return all.filter((e) => e.a === a);
  }

  /** Pairs for daughter (Z, A) filtered by `f`. */
  async pairsFiltered(
    z: number,
    a: number,
    f: CoincidenceFilter,
  ): Promise<CoincidenceEntry[]> {
    const all = await this.pairsForElement(z);
    const min = f.minIntensity ?? 0;
    return all.filter((e) => {
      if (e.a !== a) return false;
      if (f.parentState !== undefined && e.parentState !== f.parentState) return false;
      if (f.parentDecayMode !== undefined && e.parentDecayMode !== f.parentDecayMode)
        return false;
      if (f.emission1RadType !== undefined && e.emission1.radType !== f.emission1RadType)
        return false;
      if (f.emission2RadType !== undefined && e.emission2.radType !== f.emission2RadType)
        return false;
      return e.pairIntensity > min;
    });
  }
}
