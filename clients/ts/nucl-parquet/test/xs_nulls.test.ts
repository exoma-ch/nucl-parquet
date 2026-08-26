/**
 * Null-residual handling in the TS cross-section reader (#362).
 *
 * `new Int32Array([1, null, 2])` is `Int32Array([1, 0, 2])`, because
 * `Number(null) === 0`. Reading `residual_Z` that way keys every transport
 * channel — (n,tot), (n,el), (n,f) — as (0, 0), so unrelated curves interleave
 * under one bogus product. Z=0 is a real value, which is exactly why
 * `CLAUDE.md` principle 3 says null and not a sentinel.
 *
 * The first three tests carry the names of their counterparts in
 * `clients/rs/nucl-parquet/src/xs.rs` on purpose: the two clients answer the
 * same question and should be greppable together. The Rust client skips such
 * rows when building its residual-keyed map; the TS extractor cannot drop rows
 * without making transport channels unreachable, so it marks them invalid and
 * `residualKeyedIndices` reproduces the skip.
 */

import { readFile } from "fs/promises";
import { join } from "path";
import { parquetWriteBuffer } from "hyparquet-writer";
import { describe, expect, it } from "vitest";
import { residualKeyedIndices, xsColumns } from "../src/columns.js";

const DATA_DIR = join(import.meta.dirname, "../../../../data");

/**
 * A canonical-shape xs table built in memory, so the null paths are covered
 * without depending on which rows the shipped tree happens to contain.
 *
 * Mirrors `synthetic_xs` in `clients/rs/nucl-parquet/src/xs.rs`. `null`
 * residuals mean "this channel names no product" — a null, not a zero, because
 * Z=0 is a real value (representation principle 3).
 */
interface SyntheticRow {
  targetA: number | null;
  residualZ: number | null;
  residualA: number | null;
  mt?: number | null;
  kind?: string;
  state?: string | null;
  energyMeV?: number;
  xsMb?: number;
}

function syntheticXs(rows: SyntheticRow[]): ArrayBuffer {
  const col = <T>(f: (r: SyntheticRow) => T) => rows.map(f);
  return parquetWriteBuffer({
    columnData: [
      { name: "library", data: col(() => "synthetic"), type: "STRING" },
      { name: "kind", data: col((r) => r.kind ?? "channel"), type: "STRING" },
      { name: "projectile", data: col(() => "n"), type: "STRING" },
      { name: "proj_Z", data: col(() => 0), type: "INT32" },
      { name: "proj_A", data: col(() => 1), type: "INT32" },
      { name: "target_Z", data: col(() => 29), type: "INT32" },
      { name: "target_A", data: col((r) => r.targetA), type: "INT32", nullable: true },
      { name: "MT", data: col((r) => r.mt ?? null), type: "INT32", nullable: true },
      { name: "residual_Z", data: col((r) => r.residualZ), type: "INT32", nullable: true },
      { name: "residual_A", data: col((r) => r.residualA), type: "INT32", nullable: true },
      { name: "state", data: col((r) => (r.state === undefined ? "g" : r.state)), type: "STRING", nullable: true },
      { name: "energy_MeV", data: col((r) => r.energyMeV ?? 1.0), type: "DOUBLE" },
      { name: "xs_mb", data: col((r) => r.xsMb ?? 100.0), type: "DOUBLE" },
    ],
  });
}

/** A channels file: every (n,tot)/(n,el)/(n,f) row has a null residual. */
const CHANNELS = join(DATA_DIR, "endfb-8.0/channels/n_U.parquet");
/** A production file: every row names a product. */
const PRODUCTION = join(DATA_DIR, "jendl-5/xs/n_Cu.parquet");
/**
 * Heavy-ion fragment production: every row has a **null `state`**, because the
 * Geant4 run names no isomeric state. The live case for #380's vocabulary — the
 * `state` column's null is reachable today, not only after the rebuild.
 */
const NULL_STATE = join(DATA_DIR, "hi-xs-prod/xs/ar40_Ac.parquet");

async function columnsOf(path: string) {
  const buf = await readFile(path);
  return xsColumns(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}

describe("null residuals", () => {
  it("null_residuals_are_skipped_not_keyed_as_zero", async () => {
    const cols = await columnsOf(CHANNELS);

    // Count first, assert once: `expect` inside a 300k-iteration loop is slow
    // enough to blow the test timeout, and says nothing extra.
    let nullResidualRows = 0;
    for (let i = 0; i < cols.residualValid.length; i++) {
      if (cols.residualValid[i] === 0) nullResidualRows++;
    }
    // n_U.parquet really does carry a large null-residual population; if this
    // ever became 0 the rest of the test would pass having found nothing.
    expect(nullResidualRows).toBeGreaterThan(100_000);

    const keyed = residualKeyedIndices(cols);
    let keyedButNull = 0;
    let zeroZeroKeyed = 0;
    for (const i of keyed) {
      if (cols.residualValid[i] === 0) keyedButNull++;
      // The observable symptom of the old coercion: rows claiming the (0, 0)
      // product, which is what every channel row decoded to.
      if (cols.residualZ[i] === 0 && cols.residualA[i] === 0) zeroZeroKeyed++;
    }
    expect(keyedButNull).toBe(0);
    expect(zeroZeroKeyed).toBe(0);
    expect(keyed.length).toBe(cols.residualValid.length - nullResidualRows);
  });

  it("null_target_a_is_skipped", async () => {
    // target_A = 0 is the ENDF natural-element convention, so a null read as 0
    // would silently masquerade as a natural-abundance row. Non-null across the
    // shipped data, so this pins the guard rather than exercising a live case.
    const cols = await columnsOf(CHANNELS);
    expect(cols.targetAValid.length).toBe(cols.energyMeV.length);

    const synthetic: typeof cols = {
      ...cols,
      targetAValid: Uint8Array.from(cols.targetAValid),
      residualValid: Uint8Array.from(cols.residualValid),
    };
    // Force one otherwise-keyable row to have a null target_A.
    const keyable = residualKeyedIndices(cols);
    expect(keyable.length).toBeGreaterThan(0);
    const victim = keyable[0];
    synthetic.targetAValid[victim] = 0;

    const after = new Set(residualKeyedIndices(synthetic));
    expect(after.has(victim)).toBe(false);
    expect(after.size).toBe(keyable.length - 1);
  });

  it("all_null_residuals_yields_an_empty_result_not_an_error", async () => {
    // A pure transport-channel table is valid input to a residual-indexed
    // consumer — it simply has no keyable rows. It must not throw, and must not
    // quietly produce one (0, 0) reaction.
    const cols = await columnsOf(CHANNELS);
    const allNull: typeof cols = { ...cols, residualValid: new Uint8Array(cols.residualValid.length) };
    expect(residualKeyedIndices(allNull).length).toBe(0);
  });
});

describe("production rows are unaffected", () => {
  it("keeps every row of a file that names a product on each one", async () => {
    const cols = await columnsOf(PRODUCTION);
    const n = cols.energyMeV.length;
    expect(n).toBeGreaterThan(100);
    expect(residualKeyedIndices(cols).length).toBe(n);
    // Every residual is marked valid, and none of them is the (0,0) key.
    expect(Array.from(cols.residualValid).every((v) => v === 1)).toBe(true);
  });
});

describe("null state (#380's vocabulary)", () => {
  it("null_state_is_not_coerced_to_the_empty_string", async () => {
    // The same bug class as the residual coercion, one column over, and live:
    // hi-xs-prod ships a null `state` on every row. Reading it as `''` would
    // merge those rows with the ~38M legacy `''` rows still in the tree — and
    // `''` is the spelling #380 retired precisely because it meant three
    // different things depending on which table you read.
    const cols = await columnsOf(NULL_STATE);
    const n = cols.energyMeV.length;
    expect(n).toBeGreaterThan(0);

    let nulls = 0;
    let empties = 0;
    for (const s of cols.state) {
      if (s === null) nulls++;
      else if (s === "") empties++;
    }
    expect(nulls).toBe(n);
    expect(empties).toBe(0);
  });

  it("keeps 'g' and null as different answers", async () => {
    // 'g' asserts the ground state. null asserts nothing. If the client
    // collapsed either onto the other, a consumer could not tell "this nuclide
    // is in its ground state" from "nobody recorded a state".
    const cols = await columnsOf(NULL_STATE);
    expect(cols.state.some((s) => s === null)).toBe(true);
    expect(cols.state.some((s) => s === "g")).toBe(false);
  });
});

describe("kind and MT", () => {
  it("exposes kind so production and channel rows can be told apart", async () => {
    // Summing xs_mb across both kinds double-counts: they are different
    // quantities over the same evaluation.
    const chan = await columnsOf(CHANNELS);
    expect(new Set(chan.kind)).toEqual(new Set(["channel"]));

    const prod = await columnsOf(PRODUCTION);
    expect(new Set(prod.kind)).toEqual(new Set(["production"]));
  });

  it("does not treat kind as a proxy for residual nullity", async () => {
    // 'channel' rows are a mix: some name a product, some do not. A consumer
    // that filtered on kind alone and then read residualZ would be back where
    // it started, so both signals have to be exposed independently.
    const cols = await columnsOf(CHANNELS);
    let channelWithResidual = 0;
    let channelWithout = 0;
    for (let i = 0; i < cols.kind.length; i++) {
      if (cols.kind[i] !== "channel") continue;
      if (cols.residualValid[i] === 1) channelWithResidual++;
      else channelWithout++;
    }
    expect(channelWithResidual).toBeGreaterThan(0);
    expect(channelWithout).toBeGreaterThan(0);
  });

  it("exposes MT with a validity mask, since evaluated tables have none yet", async () => {
    const chan = await columnsOf(CHANNELS);
    expect(chan.mtValid.length).toBe(chan.energyMeV.length);
    // Channel rows carry MT — it is the channel's identity (principle 2).
    expect(Array.from(chan.mtValid).every((v) => v === 1)).toBe(true);
    const mts = new Set<number>();
    for (let i = 0; i < chan.mt.length; i++) if (chan.mtValid[i] === 1) mts.add(chan.mt[i]);
    // total, elastic, non-elastic and fission all name no product.
    expect(mts).toContain(1);
    expect(mts).toContain(2);
    expect(mts).toContain(18);

    // MT=1/2/18 are precisely the rows with no residual — the collision #362
    // describes. Check they are distinguishable now.
    const byMt = new Map<number, number>();
    for (let i = 0; i < chan.mt.length; i++) {
      if (chan.residualValid[i] === 0) byMt.set(chan.mt[i], (byMt.get(chan.mt[i]) ?? 0) + 1);
    }
    expect(byMt.size).toBeGreaterThan(1);

    // The evaluated production table has no MT at all yet (it arrives with the
    // #347 rebuild), so the mask must be able to say "absent" rather than 0 —
    // MT=0 is not a valid ENDF reaction number, and would be a sentinel.
    const prod = await columnsOf(PRODUCTION);
    expect(Array.from(prod.mtValid).every((v) => v === 0)).toBe(true);
  });
});

describe("synthetic fixtures — the exact cases the Rust client pins", () => {
  it("null_residuals_are_skipped_not_keyed_as_zero (synthetic)", async () => {
    // Two transport-channel rows and one real (n,g) channel — the same fixture
    // as the Rust test of this name.
    const cols = await xsColumns(
      syntheticXs([
        { targetA: 63, residualZ: null, residualA: null, mt: 1, energyMeV: 1.0, xsMb: 500.0 },
        { targetA: 63, residualZ: null, residualA: null, mt: 2, energyMeV: 2.0, xsMb: 400.0 },
        { targetA: 63, residualZ: 30, residualA: 64, mt: 102, energyMeV: 1.0, xsMb: 100.0 },
      ]),
    );
    const keyed = residualKeyedIndices(cols);
    expect(Array.from(keyed)).toEqual([2]);
    expect(cols.residualZ[2]).toBe(30);
    expect(cols.residualA[2]).toBe(64);
    // The two channel rows must not read as the (0, 0) product.
    expect(cols.residualValid[0]).toBe(0);
    expect(cols.residualValid[1]).toBe(0);
    // They remain reachable and distinguishable — by MT, which is their identity.
    expect(cols.mt[0]).toBe(1);
    expect(cols.mt[1]).toBe(2);
  });

  it("null_target_a_is_skipped (synthetic)", async () => {
    // target_A = 0 is the ENDF natural-element convention, so a null read as 0
    // would silently masquerade as a natural-abundance row.
    const cols = await xsColumns(
      syntheticXs([
        { targetA: null, residualZ: 30, residualA: 64, xsMb: 100.0 },
        { targetA: 63, residualZ: 31, residualA: 65, xsMb: 200.0 },
      ]),
    );
    expect(Array.from(residualKeyedIndices(cols))).toEqual([1]);
    expect(cols.targetAValid[0]).toBe(0);
  });

  it("all_null_residuals_yields_an_empty_result_not_an_error (synthetic)", async () => {
    // A pure transport-channel file is valid input to a residual-indexed
    // consumer — it just has no keyable rows.
    const cols = await xsColumns(
      syntheticXs([{ targetA: 238, residualZ: null, residualA: null, mt: 18, xsMb: 500.0 }]),
    );
    expect(residualKeyedIndices(cols).length).toBe(0);
    expect(cols.energyMeV.length).toBe(1); // the row is still readable as a channel
  });

  it("a half-null residual pair is not keyable", async () => {
    // Never occurs in the shipped tables, which is exactly why it needs a
    // synthetic fixture: `residualValid` is the AND of the two halves, and an
    // OR would let this row through to be read as residualA = 0 — the sentinel
    // collision again, one column over. The Rust client checks both halves
    // independently for the same reason.
    const cols = await xsColumns(
      syntheticXs([
        { targetA: 63, residualZ: 30, residualA: null },
        { targetA: 63, residualZ: null, residualA: 64 },
        { targetA: 63, residualZ: 30, residualA: 64 },
      ]),
    );
    expect(Array.from(residualKeyedIndices(cols))).toEqual([2]);
  });

  it("MT=0 would be a sentinel, so absence is carried by the mask", async () => {
    const cols = await xsColumns(
      syntheticXs([
        { targetA: 63, residualZ: 30, residualA: 64, mt: null },
        { targetA: 63, residualZ: 30, residualA: 64, mt: 102 },
      ]),
    );
    expect(cols.mtValid[0]).toBe(0);
    expect(cols.mtValid[1]).toBe(1);
    expect(cols.mt[1]).toBe(102);
  });

  it("refuses a null in a non-null *string* column", async () => {
    // A null `library` silently becoming "" would drop the row out of every
    // value filter — the same silent-wrongness as the integer coercion, so it
    // gets the same answer. `state` is nullable and keeps its null; the other
    // string columns are declared non-null and must say so.
    const buf = parquetWriteBuffer({
      columnData: [
        { name: "library", data: [null], type: "STRING", nullable: true },
        { name: "kind", data: ["channel"], type: "STRING" },
        { name: "projectile", data: ["n"], type: "STRING" },
        { name: "proj_Z", data: [0], type: "INT32" },
        { name: "proj_A", data: [1], type: "INT32" },
        { name: "target_Z", data: [29], type: "INT32" },
        { name: "target_A", data: [63], type: "INT32", nullable: true },
        { name: "MT", data: [1], type: "INT32", nullable: true },
        { name: "residual_Z", data: [null], type: "INT32", nullable: true },
        { name: "residual_A", data: [null], type: "INT32", nullable: true },
        { name: "state", data: [null], type: "STRING", nullable: true },
        { name: "energy_MeV", data: [1.0], type: "DOUBLE" },
        { name: "xs_mb", data: [1.0], type: "DOUBLE" },
      ],
    });
    await expect(xsColumns(buf)).rejects.toThrow(/Refusing to substitute/);
  });

  it("keeps a null state null, rather than resurrecting the empty string", async () => {
    // #380 retired `''` because one token was carrying three meanings — "summed
    // over states" on an ENDF row, "not stated" on an EXFOR row, "the ground
    // state" in meta/ensdf. Substituting `''` for a null here would reintroduce
    // exactly that, and merge the rows that ship null into the legacy ones.
    const cols = await xsColumns(
      syntheticXs([
        { targetA: 63, residualZ: 30, residualA: 64, state: null },
        { targetA: 63, residualZ: 30, residualA: 64, state: "g" },
        { targetA: 63, residualZ: 30, residualA: 64, state: "sum" },
      ]),
    );
    expect(cols.state[0]).toBeNull();
    expect(cols.state[1]).toBe("g");
    expect(cols.state[2]).toBe("sum");
    // 'g' asserts the ground state; null asserts nothing. Distinct answers.
    expect(cols.state[0]).not.toBe(cols.state[1]);
  });

  it("refuses a null in a column the schema declares non-null", async () => {
    // xs_mb = 0 is a real cross-section, so coercing a null there would be
    // invisible. Fail loudly instead.
    const buf = parquetWriteBuffer({
      columnData: [
        { name: "library", data: ["x"], type: "STRING" },
        { name: "kind", data: ["channel"], type: "STRING" },
        { name: "projectile", data: ["n"], type: "STRING" },
        { name: "proj_Z", data: [0], type: "INT32" },
        { name: "proj_A", data: [1], type: "INT32" },
        { name: "target_Z", data: [29], type: "INT32" },
        { name: "target_A", data: [63], type: "INT32", nullable: true },
        { name: "MT", data: [1], type: "INT32", nullable: true },
        { name: "residual_Z", data: [null], type: "INT32", nullable: true },
        { name: "residual_A", data: [null], type: "INT32", nullable: true },
        { name: "state", data: [""], type: "STRING" },
        { name: "energy_MeV", data: [1.0], type: "DOUBLE" },
        { name: "xs_mb", data: [null], type: "DOUBLE", nullable: true },
      ],
    });
    await expect(xsColumns(buf)).rejects.toThrow(/declares it non-null/);
  });
});

describe("column accessors refuse to invent data", () => {
  it("throws rather than silently misaligning when a column is absent", async () => {
    // A zero-length column beside full-length siblings makes every subsequent
    // index-based read refer to a different row than the caller believes.
    const buf = await readFile(join(DATA_DIR, "stopping/PSTAR.parquet"));
    const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    await expect(xsColumns(ab)).rejects.toThrow(/not in this parquet file/);
  });
});
