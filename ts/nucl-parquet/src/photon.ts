/**
 * EPDL97 photon interaction cross-sections.
 *
 * Provides per-element, per-process cross-section lookups with log-log
 * interpolation. All data is loaded at construction time into typed arrays.
 */

import { parquetRead } from "hyparquet";
import { logLogInterp } from "./interp.js";
import { Process } from "./types.js";

/** Sorted energy/cross-section table for a single (element, process) pair. */
interface XsTable {
  /** Energies in MeV, sorted ascending. */
  energy: Float64Array;
  /** Cross-sections in barns/atom, same length as energy. */
  xs: Float64Array;
}

/** Sorted x/y table for form factors or scattering functions. */
interface TabTable {
  x: Float64Array;
  y: Float64Array;
}

/** Make a key for the xs_tables map. */
function xsKey(z: number, process: Process): string {
  return `${z}:${process}`;
}

/**
 * Read a Parquet file and return rows as an array of column-name -> value objects.
 * Works in both Node.js (via fs) and browser (via fetch).
 */
async function readParquetFile(path: string): Promise<Record<string, unknown>[][]> {
  const arrayBuffer = await loadFileAsArrayBuffer(path);
  const rows: Record<string, unknown>[][] = [];
  await parquetRead({
    file: {
      byteLength: arrayBuffer.byteLength,
      slice(start: number, end: number) {
        return arrayBuffer.slice(start, end);
      },
    },
    onComplete(data: Record<string, unknown>[][]) {
      rows.push(...data);
    },
  });
  return rows;
}

/** Load file as ArrayBuffer — Node.js implementation. */
async function loadFileAsArrayBuffer(path: string): Promise<ArrayBuffer> {
  const { readFile } = await import("fs/promises");
  const buffer = await readFile(path);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

/** Sort parallel arrays by the first array (ascending). Returns new sorted arrays. */
function sortParallel(a: number[], b: number[]): [Float64Array, Float64Array] {
  const indices = Array.from({ length: a.length }, (_, i) => i);
  indices.sort((i, j) => a[i] - a[j]);
  const sortedA = new Float64Array(a.length);
  const sortedB = new Float64Array(b.length);
  for (let k = 0; k < indices.length; k++) {
    sortedA[k] = a[indices[k]];
    sortedB[k] = b[indices[k]];
  }
  return [sortedA, sortedB];
}

/** List .parquet files in a directory. */
async function listParquetFiles(dir: string): Promise<string[]> {
  const { readdir } = await import("fs/promises");
  const { join } = await import("path");
  const entries = await readdir(dir);
  return entries
    .filter((e) => e.endsWith(".parquet"))
    .map((e) => join(dir, e));
}

/** Check if a directory exists. */
async function dirExists(dir: string): Promise<boolean> {
  try {
    const { stat } = await import("fs/promises");
    const s = await stat(dir);
    return s.isDirectory();
  } catch {
    return false;
  }
}

/**
 * Photon interaction database loaded from EPDL97 Parquet files.
 *
 * Use the static `open()` factory method to create an instance.
 *
 * @example
 * ```ts
 * const db = await PhotonDb.open("path/to/nucl-parquet/meta");
 * const sigma = db.crossSection(29, 0.511, Process.Photoelectric); // barns/atom
 * const ff = db.formFactor(29, 0.5); // dimensionless
 * ```
 */
export class PhotonDb {
  private xsTables: Map<string, XsTable>;
  private formFactors: Map<number, TabTable>;
  private scatteringFns: Map<number, TabTable>;

  private constructor(
    xsTables: Map<string, XsTable>,
    formFactors: Map<number, TabTable>,
    scatteringFns: Map<number, TabTable>,
  ) {
    this.xsTables = xsTables;
    this.formFactors = formFactors;
    this.scatteringFns = scatteringFns;
  }

  /**
   * Load EPDL97 data from the nucl-parquet `meta/` directory.
   *
   * Reads:
   * - `meta/epdl97/photon_xs/*.parquet` — per-process cross-sections
   * - `meta/epdl97/form_factors/*.parquet` — Rayleigh form factors
   * - `meta/epdl97/scattering_fn/*.parquet` — Compton scattering functions
   */
  static async open(metaDir: string): Promise<PhotonDb> {
    const { join } = await import("path");
    const xsDir = join(metaDir, "epdl97", "photon_xs");
    const ffDir = join(metaDir, "epdl97", "form_factors");
    const sfDir = join(metaDir, "epdl97", "scattering_fn");

    const xsTables = await PhotonDb.loadXsTables(xsDir);
    const formFactors = await PhotonDb.loadTabTables(ffDir, "form_factor");
    const scatteringFns = await PhotonDb.loadTabTables(sfDir, "scattering_fn");

    return new PhotonDb(xsTables, formFactors, scatteringFns);
  }

  /**
   * Interpolate cross-section [barns/atom] for element Z at energy E [MeV].
   *
   * Returns 0.0 if the element or process is not in the database.
   * Uses log-log interpolation on the EPDL97 energy grid.
   */
  crossSection(z: number, energyMeV: number, process: Process): number {
    const table = this.xsTables.get(xsKey(z, process));
    if (!table) return 0.0;
    return logLogInterp(table.energy, table.xs, energyMeV);
  }

  /**
   * All process cross-sections for element Z at energy E [MeV].
   *
   * Returns [photoelectric, compton, rayleigh, pair_total] in barns/atom.
   */
  allCrossSections(z: number, energyMeV: number): [number, number, number, number] {
    return [
      this.crossSection(z, energyMeV, Process.Photoelectric),
      this.crossSection(z, energyMeV, Process.Incoherent),
      this.crossSection(z, energyMeV, Process.Coherent),
      this.crossSection(z, energyMeV, Process.PairTotal),
    ];
  }

  /** Total cross-section [barns/atom] for element Z at energy E [MeV]. */
  totalCrossSection(z: number, energyMeV: number): number {
    return this.crossSection(z, energyMeV, Process.Total);
  }

  /**
   * Rayleigh atomic form factor for element Z at momentum transfer q.
   *
   * q = sin(theta/2) / lambda in units matching the EPDL97 table (typically 1/cm).
   * Returns Z (atomic number) at q=0.
   */
  formFactor(z: number, q: number): number {
    const table = this.formFactors.get(z);
    if (!table) return 0.0;
    return logLogInterp(table.x, table.y, q);
  }

  /**
   * Compton incoherent scattering function S(q) for element Z.
   *
   * Ranges from 0 (forward) to Z (backward scattering).
   */
  scatteringFunction(z: number, q: number): number {
    const table = this.scatteringFns.get(z);
    if (!table) return 0.0;
    return logLogInterp(table.x, table.y, q);
  }

  /** Check if data is loaded for element Z. */
  hasElement(z: number): boolean {
    return this.xsTables.has(xsKey(z, Process.Total));
  }

  // --- Internal loading ---

  private static async loadXsTables(dir: string): Promise<Map<string, XsTable>> {
    const tables = new Map<string, XsTable>();

    if (!(await dirExists(dir))) {
      throw new Error(`Data directory not found: ${dir}`);
    }

    const files = await listParquetFiles(dir);
    for (const file of files) {
      const rows = await readParquetFile(file);

      // Accumulate per-process data
      const processData = new Map<Process, { energies: number[]; xs: number[] }>();
      let zVal: number | undefined;

      for (const rowGroup of rows) {
        for (const row of rowGroup) {
          const z = row["Z"] as number;
          if (zVal === undefined) zVal = z;

          const procStr = row["process"] as string;
          const process = procStr as Process;
          if (!Object.values(Process).includes(process)) continue;

          if (!processData.has(process)) {
            processData.set(process, { energies: [], xs: [] });
          }
          const entry = processData.get(process)!;
          entry.energies.push(row["energy_MeV"] as number);
          entry.xs.push(row["xs_barns"] as number);
        }
      }

      if (zVal !== undefined) {
        for (const [process, data] of processData) {
          const [sortedE, sortedXs] = sortParallel(data.energies, data.xs);
          tables.set(xsKey(zVal, process), { energy: sortedE, xs: sortedXs });
        }
      }
    }

    return tables;
  }

  private static async loadTabTables(
    dir: string,
    valueColName: string,
  ): Promise<Map<number, TabTable>> {
    const tables = new Map<number, TabTable>();

    if (!(await dirExists(dir))) {
      return tables; // optional data
    }

    const files = await listParquetFiles(dir);
    for (const file of files) {
      const rows = await readParquetFile(file);

      const xVals: number[] = [];
      const yVals: number[] = [];
      let zVal: number | undefined;

      for (const rowGroup of rows) {
        for (const row of rowGroup) {
          const z = row["Z"] as number;
          if (zVal === undefined) zVal = z;

          xVals.push(row["momentum_transfer"] as number);
          yVals.push(row[valueColName] as number);
        }
      }

      if (zVal !== undefined) {
        const [sortedX, sortedY] = sortParallel(xVals, yVals);
        tables.set(zVal, { x: sortedX, y: sortedY });
      }
    }

    return tables;
  }
}
