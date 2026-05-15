/**
 * Radiation lines — γ, X-ray, and Auger emissions per (Z, A, state) plus
 * the `identifyGamma` cross-isotope search.
 *
 * Mirrors the Rust `RadiationDb` (`clients/rs/nucl-parquet/src/meta.rs`)
 * and the Python `nucl_parquet.gamma_lines()` / `identify_gamma()` helpers.
 *
 * **Daughter-keyed convention.** Each emission row is filed under the
 * nucleus that emits it, not the parent that decayed. The Co-60 source's
 * iconic 1173 + 1333 keV γ live under `(Z=28, A=60)` (Ni-60), since they
 * are emitted by the de-exciting Ni-60 nucleus.
 *
 * **Lazy per-element loading + eager-built γ index.** `open()` does no
 * file I/O; per-isotope files are loaded on first access. The
 * `identifyGamma` index is built on first call by scanning every
 * per-element file and is reused thereafter (~15 MB resident).
 */
import { parquetRead } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { zToSymbol } from "./z_symbols.js";

export interface EmissionEntry {
  z: number;
  a: number;
  state: string;
  radType: string;
  energyKeV: number;
  intensityPct: number;
  decayMode: string | null;
  radSubtype: string | null;
  iccTotal: number | null;
  vacancyShell: string | null;
}

export interface GammaCandidate {
  z: number;
  a: number;
  state: string;
  energyKeV: number;
  intensityPct: number;
  deltaKeV: number;
}

interface GammaIndexEntry {
  energyKeV: number;
  intensityPct: number;
  z: number;
  a: number;
  state: string;
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
      for (const row of data as Record<string, unknown>[]) flat.push(row);
    },
  });
  return flat;
}

async function listParquetFiles(dir: string): Promise<string[]> {
  const { readdir } = await import("fs/promises");
  const { join } = await import("path");
  const entries = await readdir(dir);
  return entries.filter((e) => e.endsWith(".parquet")).map((e) => join(dir, e));
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

function toNumOrNull(v: unknown): number | null {
  if (v == null) return null;
  return typeof v === "number" ? v : Number(v);
}

function toNum(v: unknown): number {
  if (v == null) return 0;
  return typeof v === "number" ? v : Number(v);
}

function rowToEmission(row: Record<string, unknown>): EmissionEntry {
  return {
    z: toNum(row["Z"]),
    a: toNum(row["A"]),
    state: (row["state"] as string | null) ?? "",
    radType: (row["rad_type"] as string | null) ?? "",
    energyKeV: toNum(row["energy_keV"]),
    intensityPct: toNum(row["intensity_pct"]),
    decayMode: (row["decay_mode"] as string | null) ?? null,
    radSubtype: (row["rad_subtype"] as string | null) ?? null,
    iccTotal: toNumOrNull(row["icc_total"]),
    vacancyShell: (row["vacancy_shell"] as string | null) ?? null,
  };
}

/**
 * Radiation database — lazy per-element loading.
 *
 * @example
 * ```ts
 * const db = await RadiationDb.open("path/to/data/meta");
 * const ni60 = await db.emissions(28, 60, "");  // 1173 + 1333 keV γ
 * const cands = await db.identifyGamma(1173.2, 1.0, 5.0);
 * ```
 */
export class RadiationDb {
  private radDir: string;
  private cache: Map<number, EmissionEntry[]> = new Map();
  private gammaIndex: GammaIndexEntry[] | null = null;
  private gammaIndexPromise: Promise<GammaIndexEntry[]> | null = null;

  private constructor(radDir: string) {
    this.radDir = radDir;
  }

  /** Open `data/meta/ensdf/radiation/`. Does no file I/O beyond directory check. */
  static async open(metaDir: string): Promise<RadiationDb> {
    const { join } = await import("path");
    const dir = join(metaDir, "ensdf", "radiation");
    if (!(await dirExists(dir))) {
      throw new Error(`radiation directory not found: ${dir}`);
    }
    return new RadiationDb(dir);
  }

  /** All emissions for element Z (cached after first read). */
  async emissionsForElement(z: number): Promise<EmissionEntry[]> {
    const cached = this.cache.get(z);
    if (cached) return cached;
    const symbol = zToSymbol(z);
    if (!symbol) throw new Error(`unknown Z=${z}`);
    const { join } = await import("path");
    const path = join(this.radDir, `${symbol}.parquet`);
    const rows = await readParquetFile(path);
    const entries = rows.map(rowToEmission);
    this.cache.set(z, entries);
    return entries;
  }

  /** All emissions for nuclide (Z, A, state). `state=""` for ground. */
  async emissions(z: number, a: number, state = ""): Promise<EmissionEntry[]> {
    const all = await this.emissionsForElement(z);
    return all.filter((e) => e.a === a && e.state === state);
  }

  /** Emissions filtered by `radType` and minimum `intensityPct`. */
  async emissionsFiltered(
    z: number,
    a: number,
    state = "",
    radType?: string,
    minIntensityPct = 0,
  ): Promise<EmissionEntry[]> {
    const slice = await this.emissions(z, a, state);
    return slice.filter((e) => {
      if (radType !== undefined && e.radType !== radType) return false;
      return e.intensityPct >= minIntensityPct;
    });
  }

  /**
   * Candidate nuclides emitting a γ near `energyKeV` (within `toleranceKeV`).
   *
   * Mirrors `nucl_parquet.identify_gamma()`. Returns matches with
   * `intensityPct >= minIntensityPct`, sorted by `|deltaKeV|` then by
   * intensity descending.
   *
   * On first call, builds a sorted cross-isotope γ index (~15 MB) by
   * scanning every per-element file. Subsequent calls reuse the index.
   */
  async identifyGamma(
    energyKeV: number,
    toleranceKeV = 2.0,
    minIntensityPct = 0.1,
  ): Promise<GammaCandidate[]> {
    const index = await this.buildOrGetIndex();
    const lo = energyKeV - toleranceKeV;
    const hi = energyKeV + toleranceKeV;
    // Index is sorted by energyKeV; binary-search the lower bound then scan.
    const start = lowerBound(index, lo);
    const out: GammaCandidate[] = [];
    for (let i = start; i < index.length; i++) {
      const e = index[i];
      if (e.energyKeV > hi) break;
      if (e.intensityPct < minIntensityPct) continue;
      out.push({
        z: e.z,
        a: e.a,
        state: e.state,
        energyKeV: e.energyKeV,
        intensityPct: e.intensityPct,
        deltaKeV: e.energyKeV - energyKeV,
      });
    }
    out.sort((a, b) => {
      const da = Math.abs(a.deltaKeV) - Math.abs(b.deltaKeV);
      if (da !== 0) return da;
      return b.intensityPct - a.intensityPct;
    });
    return out;
  }

  private async buildOrGetIndex(): Promise<GammaIndexEntry[]> {
    if (this.gammaIndex) return this.gammaIndex;
    // Cache the in-flight promise so concurrent callers share one scan. On
    // rejection (transient I/O blip, corrupt file, OOM during the
    // full-corpus scan), clear the slot so the next call retries — caching
    // a rejected promise would permanently poison identifyGamma for the
    // lifetime of this RadiationDb instance.
    if (!this.gammaIndexPromise) {
      const promise = (async () => {
        const idx: GammaIndexEntry[] = [];
        const files = await listParquetFiles(this.radDir);
        for (const file of files) {
          const rows = await readParquetFile(file);
          for (const row of rows) {
            if (row["rad_type"] !== "gamma") continue;
            idx.push({
              energyKeV: toNum(row["energy_keV"]),
              intensityPct: toNum(row["intensity_pct"]),
              z: toNum(row["Z"]),
              a: toNum(row["A"]),
              state: (row["state"] as string | null) ?? "",
            });
          }
        }
        idx.sort((a, b) => a.energyKeV - b.energyKeV);
        this.gammaIndex = idx;
        return idx;
      })();
      promise.catch(() => {
        if (this.gammaIndexPromise === promise) {
          this.gammaIndexPromise = null;
        }
      });
      this.gammaIndexPromise = promise;
    }
    return this.gammaIndexPromise;
  }
}

/** First index `i` such that `arr[i].energyKeV >= target`, or `arr.length`. */
function lowerBound(arr: GammaIndexEntry[], target: number): number {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (arr[mid].energyKeV < target) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
