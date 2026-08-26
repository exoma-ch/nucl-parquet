/**
 * Zero-copy column extraction for WASM transfer.
 *
 * Uses hyparquet's onChunk callback to accumulate column data natively,
 * then wraps numeric columns in typed arrays — no JSON serialisation round-trip.
 *
 * Intended use:
 *   // Per-source parquet (PSTAR.parquet / ASTAR.parquet / ESTAR.parquet /
 *   // dSTAR.parquet / tSTAR.parquet) — schema: source, target_Z, energy_MeV, dedx
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
  /** Stopping source name (one of "PSTAR", "ASTAR", "ESTAR", "dSTAR", "tSTAR"). */
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
  /**
   * Projectile mass number. CatIMA stopping is reduced-mass dependent, so
   * isotopes of the same Z differ (up to ~15% below ~0.01 MeV/u). Lookups must
   * key on (projZ, projA, targetZ) — keying on (projZ, targetZ) alone merges
   * isotopes onto one energy axis (see #248).
   */
  projA: Int32Array;
  targetZ: Int32Array;
  energyMeVu: Float64Array;
  dedx: Float64Array;
  /** Bohr energy straggling variance dOmega2/d(rho*x) [MeV^2 cm^2/g]. */
  straggling: Float64Array;
}

// ---------------------------------------------------------------------------
// Cross-section columns
// ---------------------------------------------------------------------------

/**
 * Column-oriented cross-section data for direct WASM transfer.
 *
 * Nullable integer columns are carried as a values buffer plus a `…Valid`
 * mask, which is how Arrow itself represents nullability and therefore what
 * `Int32Array::is_null(i)` reads in the Rust client
 * (`clients/rs/nucl-parquet/src/xs.rs`). `valid[i] === 0` means the row has no
 * value there and `values[i]` is meaningless — it is NOT a zero.
 *
 * There is no in-band "missing" integer, deliberately. Collapsing a null
 * `residual_Z` onto 0 is the sentinel collision representation principle 3
 * exists to prevent: Z=0 is a real value, so (n,tot), (n,el) and (n,f) would
 * all key as (0, 0) and interleave with each other (#362).
 */
export interface XsColumns {
  /** Library slug, e.g. "jendl-5". */
  library: string[];
  /**
   * `"production"` or `"channel"`.
   *
   * A production row and a channel row are *different quantities over the same
   * evaluation*, so summing `xsMb` across both double-counts. This is not the
   * same distinction as residual nullity and cannot substitute for it: channel
   * rows are a mix of named-product and no-product (2.98M vs 8.64M rows in the
   * shipped tables), so a consumer needs both signals.
   */
  kind: string[];
  projectile: string[];
  projZ: Int32Array;
  projA: Int32Array;
  targetZ: Int32Array;
  targetA: Int32Array;
  /**
   * 1 where `targetA[i]` is a real mass number.
   *
   * Guarded for the same reason the Rust client guards it: `target_A = 0` is
   * the ENDF natural-element convention, so a null read as 0 would silently
   * masquerade as a natural-abundance row. Non-null throughout the shipped
   * data, so this is a guard rather than a routine branch.
   */
  targetAValid: Uint8Array;
  /**
   * ENDF MT reaction number — the canonical channel identity (principle 2).
   * Null where the source carries none; evaluated production tables are
   * entirely null today and gain MT with the #347 rebuild.
   */
  mt: Int32Array;
  /** 1 where `mt[i]` is a real MT number. */
  mtValid: Uint8Array;
  residualZ: Int32Array;
  residualA: Int32Array;
  /**
   * 1 where the row names a single product, 0 where it names none.
   *
   * Covers `residualZ` and `residualA` together — the pair is meaningless
   * unless both are present, so this is the AND of their validity, matching
   * `rz.is_null(i) || ra.is_null(i)` in the Rust client. A 0 here means the row
   * is a transport channel — (n,tot), (n,el), (n,f) — identified by `mt`, not
   * by a residual.
   */
  residualValid: Uint8Array;
  /**
   * Isomeric state, from the one vocabulary in
   * `nucl_parquet/state_vocabulary.py` (#380): `'g'` ground, `'m'`/`'m2'`/…
   * isomers, `'l'` an unresolved level, `'sum'` summed over all states, and
   * **`null` for "not stated"**.
   *
   * `null` is a real, distinct answer here — not a gap to paper over. `'g'`
   * asserts the ground state; `'l'` asserts an unresolved level; `null`
   * asserts nothing, and #380 retired `''` precisely because it had been
   * carrying all three meanings at once depending on the source table.
   *
   * Carried as an in-band `null` rather than a `…Valid` mask, unlike the
   * integer columns. The masks exist to keep those columns as typed arrays;
   * a string column is a boxed JS array either way, so a mask would be pure
   * overhead — and `null` in the element type makes handling it a compile
   * error under `strict`, which is a stronger guarantee than a mask can give.
   */
  state: (string | null)[];
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

/**
 * Fetch a column that the schema declares present, or throw.
 *
 * `?? []` used to stand in for a missing column, which silently produced a
 * zero-length array while its siblings kept full length — so every index-based
 * read after it referred to a different row than the caller believed.
 */
function required(
  cols: Map<string, number[] | string[]>,
  name: string,
): (number | null)[] | (string | null)[] {
  const col = cols.get(name);
  if (col === undefined) {
    throw new Error(
      `column '${name}' is not in this parquet file (found: ${[...cols.keys()].join(", ")}). ` +
        `Cross-section files carry the canonical 18-column schema; this looks like a different table.`,
    );
  }
  return col as (number | null)[] | (string | null)[];
}

/**
 * A non-nullable float column.
 *
 * Throws rather than coercing, because `new Float64Array([1, null])` is
 * `[1, 0]` — and 0 is a perfectly plausible cross-section, so the corruption
 * would be invisible. Nullable float columns (`xs_err_mb`, `energy_err_MeV`)
 * are not exposed yet; when they are, they need NaN-for-null, not this.
 */
function getF64(cols: Map<string, number[] | string[]>, name: string): Float64Array {
  const src = required(cols, name) as (number | null)[];
  const out = new Float64Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v === null || v === undefined) {
      throw new Error(
        `column '${name}' has a null at row ${i}, but the canonical schema declares it non-null. ` +
          `Refusing to coerce it to 0 — that would be indistinguishable from a real measurement.`,
      );
    }
    out[i] = v;
  }
  return out;
}

/**
 * A non-nullable int column.
 *
 * Same reasoning as `getF64`: `new Int32Array([1, null, 2])` is `[1, 0, 2]`,
 * and 0 is a real Z, a real A and a real MT. Use `getNullableI32` for columns
 * the schema allows to be null.
 */
function getI32(cols: Map<string, number[] | string[]>, name: string): Int32Array {
  const src = required(cols, name) as (number | null)[];
  const out = new Int32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v === null || v === undefined) {
      throw new Error(
        `column '${name}' has a null at row ${i}, but the canonical schema declares it non-null. ` +
          `Refusing to coerce it to 0 — 0 is a real value for this column (#362).`,
      );
    }
    out[i] = v;
  }
  return out;
}

/**
 * A nullable int column, as an Arrow-style (values, validity) pair.
 *
 * Null slots are left as 0 in `values` for determinism, and are meaningless:
 * `valid[i]` is the only thing that says whether `values[i]` means anything.
 */
function getNullableI32(
  cols: Map<string, number[] | string[]>,
  name: string,
): { values: Int32Array; valid: Uint8Array } {
  const src = required(cols, name) as (number | null)[];
  const values = new Int32Array(src.length);
  const valid = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v === null || v === undefined) continue;
    values[i] = v;
    valid[i] = 1;
  }
  return { values, valid };
}

/**
 * A non-nullable string column.
 *
 * Throws rather than substituting `""`, for the same reason the numeric
 * accessors throw: `library`, `kind` and `projectile` are declared non-null,
 * and a null silently becoming `""` would drop the row out of every
 * `kind === "production"` filter without anything saying so.
 */
function getStrings(cols: Map<string, number[] | string[]>, name: string): string[] {
  const src = required(cols, name) as (string | null)[];
  const out = new Array<string>(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (v === null || v === undefined) {
      throw new Error(
        `column '${name}' has a null at row ${i}, but the canonical schema declares it non-null. ` +
          `Refusing to substitute '' — that would silently drop the row out of value filters.`,
      );
    }
    out[i] = v;
  }
  return out;
}

/**
 * A nullable string column, keeping the null.
 *
 * `state` is the only one, and its null is load-bearing: #380 gave the column a
 * single vocabulary in which every value is a positive statement and absence is
 * `null`. Substituting `""` would resurrect the exact spelling that PR retired —
 * one token that meant "summed over states", "not stated" and "the ground
 * state" depending on which table you read — and would merge the 6.5M rows that
 * currently ship `null` into the 38M legacy `''` rows on sight.
 */
function getNullableStrings(
  cols: Map<string, number[] | string[]>,
  name: string,
): (string | null)[] {
  const src = required(cols, name) as (string | null)[];
  const out = new Array<string | null>(src.length);
  for (let i = 0; i < src.length; i++) out[i] = src[i] ?? null;
  return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extract stopping power columns from a per-source NIST parquet ArrayBuffer
 * (PSTAR.parquet, ASTAR.parquet, ESTAR.parquet, dSTAR.parquet, tSTAR.parquet).
 * Schema: source str, target_Z i32, energy_MeV f64, dedx f64.
 *
 * The previously-shipped `stopping.parquet` aggregate is no longer published
 * (deleted in #143); fetch the per-source file you need. For full Z×Z catima
 * coverage use `catimaColumns` against `stopping/catima/catima.parquet`.
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
 * Schema: proj_Z i32, proj_A i32, target_Z i32, energy_MeV_u f64, dedx f64, straggling f64
 */
export async function catimaColumns(buffer: ArrayBuffer): Promise<CatimaColumns> {
  const cols = await extractColumns(buffer);
  return {
    projZ: getI32(cols, "proj_Z"),
    projA: getI32(cols, "proj_A"),
    targetZ: getI32(cols, "target_Z"),
    energyMeVu: getF64(cols, "energy_MeV_u"),
    dedx: getF64(cols, "dedx"),
    straggling: getF64(cols, "straggling"),
  };
}

/**
 * Extract cross-section columns from an XS parquet ArrayBuffer.
 *
 * Canonical schema (`nucl_parquet._schemas.CANONICAL_XS_SCHEMA`):
 * library str, kind str, projectile str, proj_Z i32, proj_A i32, target_Z i32,
 * target_A i32, MT i32?, residual_Z i32?, residual_A i32?, state str?,
 * energy_MeV f64, xs_mb f64, energy_err_MeV f64?, xs_err_mb f64?,
 * source_entry str, author str, year i32.
 *
 * `MT`, `residual_Z` and `residual_A` are nullable and come back with a
 * `…Valid` mask; see {@link XsColumns}. The uncertainty and provenance columns
 * are not extracted yet — they are nullable floats/strings and want their own
 * treatment (a null `xs_err_mb` must not become a claimed ±0).
 *
 * To index rows by residual the way the Rust client does, use
 * {@link residualKeyedIndices} rather than reading `residualZ` directly.
 */
export async function xsColumns(buffer: ArrayBuffer): Promise<XsColumns> {
  const cols = await extractColumns(buffer);
  const mt = getNullableI32(cols, "MT");
  const rz = getNullableI32(cols, "residual_Z");
  const ra = getNullableI32(cols, "residual_A");
  const targetA = getNullableI32(cols, "target_A");

  // The residual pair is meaningless unless both halves are present, so the
  // mask is their AND — the same condition as `rz.is_null(i) || ra.is_null(i)`
  // in the Rust client. They are never half-null in the shipped data; this
  // keeps that an observation rather than an assumption.
  const residualValid = new Uint8Array(rz.valid.length);
  for (let i = 0; i < residualValid.length; i++) {
    residualValid[i] = rz.valid[i] & ra.valid[i];
  }

  return {
    library: getStrings(cols, "library"),
    kind: getStrings(cols, "kind"),
    projectile: getStrings(cols, "projectile"),
    projZ: getI32(cols, "proj_Z"),
    projA: getI32(cols, "proj_A"),
    targetZ: getI32(cols, "target_Z"),
    targetA: targetA.values,
    targetAValid: targetA.valid,
    mt: mt.values,
    mtValid: mt.valid,
    residualZ: rz.values,
    residualA: ra.values,
    residualValid,
    state: getNullableStrings(cols, "state"),
    energyMeV: getF64(cols, "energy_MeV"),
    xsMb: getF64(cols, "xs_mb"),
  };
}

/**
 * Indices of the rows that can be keyed by residual — i.e. those naming a
 * single product, with a usable target mass.
 *
 * This is the TypeScript spelling of what `CrossSectionDb::parse` does in
 * `clients/rs/nucl-parquet/src/xs.rs`, which skips a row when
 * `rz.is_null(i) || ra.is_null(i) || ta.is_null(i)`. Both clients answer the
 * same question the same way; only the shape of the answer differs, because a
 * column extractor cannot drop rows without making transport channels
 * unreachable — `residualValid[i] === 0` is how you ask for those instead.
 *
 * ```ts
 * const cols = await xsColumns(buf);
 * for (const i of residualKeyedIndices(cols)) {
 *   key(cols.targetA[i], cols.residualZ[i], cols.residualA[i], cols.state[i]);
 * }
 * ```
 */
export function residualKeyedIndices(cols: XsColumns): Uint32Array {
  const n = cols.residualValid.length;
  const scratch = new Uint32Array(n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    if (cols.residualValid[i] === 1 && cols.targetAValid[i] === 1) scratch[k++] = i;
  }
  // `slice`, not `subarray`: a subarray is a view onto the full-length scratch
  // buffer, so `.buffer`/`.byteLength` would describe n entries rather than k —
  // a quiet trap for exactly the WASM hand-off this module exists for, and it
  // would pin the whole allocation alive for the caller's lifetime.
  return scratch.slice(0, k);
}
