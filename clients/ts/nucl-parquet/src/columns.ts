/**
 * Zero-copy column extraction for WASM transfer.
 *
 * Uses hyparquet's onChunk callback to accumulate column data natively,
 * then wraps numeric columns in typed arrays — no JSON serialisation round-trip.
 *
 * Intended use:
 *   const cols = await stoppingColumns(arrayBuffer);
 *   wasm.load_stopping_arrays(cols.source, cols.targetZ, cols.energyMeV, cols.dedx);
 */

import { parquetRead } from "hyparquet";
import type { ColumnData } from "hyparquet";
import { compressors } from "hyparquet-compressors";

// ---------------------------------------------------------------------------
// Stopping power columns
// ---------------------------------------------------------------------------

/** Column-oriented stopping power data for direct WASM transfer. */
export interface StoppingColumns {
  /** Stopping source name (e.g. "PSTAR", "ASTAR", "catima"). */
  source: string[];
  /** Target element atomic number. */
  targetZ: Int32Array;
  /** Projectile kinetic energy [MeV]. */
  energyMeV: Float64Array;
  /** Mass stopping power [MeV cm²/g]. */
  dedx: Float64Array;
}

/** Column-oriented catima stopping data (energy in MeV/u). */
export interface CatimaColumns {
  projZ: Int32Array;
  targetZ: Int32Array;
  energyMeVu: Float64Array;
  dedx: Float64Array;
  /** Bohr energy straggling variance dOmega2/d(rho*x) [MeV^2 cm^2/g]. */
  straggling: Float64Array;
}

// ---------------------------------------------------------------------------
// Cross-section columns
// ---------------------------------------------------------------------------

/** Column-oriented cross-section data for direct WASM transfer. */
export interface XsColumns {
  targetA: Int32Array;
  residualZ: Int32Array;
  residualA: Int32Array;
  /** Isomeric state label (empty string for ground state). */
  state: string[];
  energyMeV: Float64Array;
  /** Cross-section [mb]. */
  xsMb: Float64Array;
}

// ---------------------------------------------------------------------------
// Core extraction
// ---------------------------------------------------------------------------

/**
 * Read all columns from a Parquet ArrayBuffer using hyparquet's chunk API.
 * Returns a map of column name → raw array (number[] or string[]).
 */
async function extractColumns(
  buffer: ArrayBuffer,
): Promise<Map<string, number[] | string[]>> {
  const cols = new Map<string, number[] | string[]>();

  await parquetRead({
    file: {
      byteLength: buffer.byteLength,
      slice: (start: number, end: number) => buffer.slice(start, end),
    },
    compressors,
    onChunk({ columnName, columnData }: ColumnData) {
      if (!cols.has(columnName)) {
        cols.set(columnName, []);
      }
      const arr = cols.get(columnName) as unknown[];
      for (let i = 0; i < columnData.length; i++) arr.push(columnData[i]);
    },
  });

  return cols;
}

function getF64(cols: Map<string, number[] | string[]>, name: string): Float64Array {
  return new Float64Array((cols.get(name) as number[]) ?? []);
}

function getI32(cols: Map<string, number[] | string[]>, name: string): Int32Array {
  return new Int32Array((cols.get(name) as number[]) ?? []);
}

function getStrings(cols: Map<string, number[] | string[]>, name: string): string[] {
  return (cols.get(name) as string[]) ?? [];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extract stopping power columns from a `stopping.parquet` ArrayBuffer.
 * Schema: source str, target_Z i32, energy_MeV f64, dedx f64
 */
export async function stoppingColumns(buffer: ArrayBuffer): Promise<StoppingColumns> {
  const cols = await extractColumns(buffer);
  return {
    source: getStrings(cols, "source"),
    targetZ: getI32(cols, "target_Z"),
    energyMeV: getF64(cols, "energy_MeV"),
    dedx: getF64(cols, "dedx"),
  };
}

/**
 * Extract catima stopping columns from a `catima.parquet` ArrayBuffer.
 * Schema: proj_Z i32, target_Z i32, energy_MeV_u f64, dedx f64, straggling f64
 */
export async function catimaColumns(buffer: ArrayBuffer): Promise<CatimaColumns> {
  const cols = await extractColumns(buffer);
  return {
    projZ: getI32(cols, "proj_Z"),
    targetZ: getI32(cols, "target_Z"),
    energyMeVu: getF64(cols, "energy_MeV_u"),
    dedx: getF64(cols, "dedx"),
    straggling: getF64(cols, "straggling"),
  };
}

/**
 * Extract cross-section columns from an XS parquet ArrayBuffer.
 * Schema: target_A i32, residual_Z i32, residual_A i32, state str, energy_MeV f64, xs_mb f64
 */
export async function xsColumns(buffer: ArrayBuffer): Promise<XsColumns> {
  const cols = await extractColumns(buffer);
  return {
    targetA: getI32(cols, "target_A"),
    residualZ: getI32(cols, "residual_Z"),
    residualA: getI32(cols, "residual_A"),
    state: getStrings(cols, "state"),
    energyMeV: getF64(cols, "energy_MeV"),
    xsMb: getF64(cols, "xs_mb"),
  };
}
