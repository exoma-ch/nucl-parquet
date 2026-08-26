/**
 * nucl-parquet MCP server — wraps local data via DuckDB.
 *
 * SSoT refactor (epic #173, Sub-D #178): the MCP is a thin shell over
 * DuckDB-backed Parquet views, not a re-implementation of parquet I/O.
 * All data is read from the local data directory — no HTTP fetching.
 */

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { createRequire } from "node:module";
import { join, resolve } from "node:path";
import { homedir } from "node:os";

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import duckdb from "duckdb";
import { z } from "zod";

const require = createRequire(import.meta.url);
export const { version: PKG_VERSION } = require("../package.json") as { version: string };

// ---------------------------------------------------------------------------
// Data directory resolution
// ---------------------------------------------------------------------------

function resolveDataDir(): string {
  // 1. Explicit env var
  const envDir = process.env.NUCL_PARQUET_DATA;
  if (envDir && existsSync(envDir)) return resolve(envDir);

  // 2. Repo-local data/ (development)
  const repoDir = resolve(import.meta.dirname ?? ".", "..", "..", "..", "..", "data");
  if (existsSync(join(repoDir, "catalog.json"))) return repoDir;

  // 3. ~/.nucl-parquet/ (installed data)
  const homeData = join(homedir(), ".nucl-parquet");
  if (existsSync(join(homeData, "catalog.json"))) return homeData;

  throw new Error(
    "Cannot find nucl-parquet data directory. Set NUCL_PARQUET_DATA env var " +
    "or place data at ~/.nucl-parquet/. Data releases: " +
    "https://github.com/exoma-ch/nucl-parquet/releases",
  );
}

// ---------------------------------------------------------------------------
// Catalog (loaded from disk, not hardcoded)
// ---------------------------------------------------------------------------

interface Library {
  name: string;
  description: string;
  source_url?: string;
  projectiles?: string[];
  data_type?: string;
  version: string;
  path?: string;
}

interface ViewDef {
  path: string;
  type?: "file" | "glob";
  optional?: boolean;
  note?: string;
}

interface Catalog {
  /**
   * The data release this server is serving, e.g. `2026.8.3`.
   *
   * Distinct from `Library.version`, which is the *evaluation's* version
   * ("2023-iso"). They answer different questions, and conflating them would
   * make a cross-server check silently wrong rather than merely absent (#348).
   *
   * Read from the catalog on disk, never compiled in — a build-time constant
   * would be a second source of truth for the one fact whose whole job is to
   * identify the data actually being read.
   */
  data_version: string;
  libraries: Record<string, Library>;
  views?: Record<string, ViewDef>;
  [key: string]: unknown;
}

let _catalog: Catalog | undefined;

export function ensureCatalog(): Catalog {
  if (_catalog) return _catalog;
  const dataDir = resolveDataDir();
  _catalog = parseCatalog(readFileSync(join(dataDir, "catalog.json"), "utf-8"), dataDir);
  return _catalog;
}

/**
 * Parse and validate a catalog.
 *
 * Separate from {@link ensureCatalog} because that one memoises a module-level
 * singleton, which a test cannot re-enter — so the validation below would be
 * unreachable from a test and, in practice, unverified.
 */
export function parseCatalog(raw: string, dataDir: string): Catalog {
  const parsed = JSON.parse(raw) as Catalog;
  // `as Catalog` is a claim, not a check. `data_version` is required by
  // data/catalog.schema.json, and the whole point of reporting it is that it
  // identifies the tree actually being read — so an absent one must fail here
  // rather than surface as `undefined` in a referral an agent is trying to
  // verify (#348). The Rust server refuses the same catalog, for the same
  // reason.
  if (typeof parsed.data_version !== "string" || parsed.data_version === "") {
    throw new Error(
      `catalog.json at ${dataDir} has no 'data_version'. It is required by ` +
        `data/catalog.schema.json and identifies the data release this server serves; ` +
        `refusing to start rather than report an unknown release as if it were known.`,
    );
  }
  return parsed;
}

/**
 * The `list_libraries` payload: the data release, then the libraries.
 *
 * A named function rather than an inline object so a test can assert the shape
 * the tool actually returns. Asserting only that the catalog *has* a
 * data_version would pass while the tool still returned a bare array.
 */
export function libraryListPayload(cat: Catalog): {
  data_version: string;
  libraries: Array<Record<string, unknown>>;
} {
  return {
    // Beside the array rather than on each entry, so the release and a
    // library's evaluation `version` cannot be read as the same kind of thing.
    data_version: cat.data_version,
    libraries: Object.entries(cat.libraries)
      .filter(([, lib]) => lib.projectiles)
      .map(([id, lib]) => ({
        id,
        name: lib.name,
        description: lib.description,
        projectiles: lib.projectiles,
        version: lib.version,
        data_type: lib.data_type,
      })),
  };
}

// ---------------------------------------------------------------------------
// DuckDB connection (lazy init, thread-safe via singleton)
// ---------------------------------------------------------------------------

let _db: duckdb.Database | undefined;

function hasParquetFiles(dir: string): boolean {
  if (!existsSync(dir)) return false;
  try {
    return readdirSync(dir).some((f) => f.endsWith(".parquet"));
  } catch {
    return false;
  }
}

function registerParquet(db: duckdb.Database, filePath: string, viewName: string): void {
  if (existsSync(filePath)) {
    db.run(`CREATE VIEW ${viewName} AS SELECT * FROM read_parquet('${filePath}')`);
  }
}

function registerGlob(db: duckdb.Database, dir: string, viewName: string): void {
  if (hasParquetFiles(dir)) {
    const glob = join(dir, "*.parquet");
    db.run(`CREATE VIEW ${viewName} AS SELECT * FROM read_parquet('${glob}')`);
  }
}

function registerViews(db: duckdb.Database, dataDir: string): void {
  const catalog = ensureCatalog();

  // --- Cross-section libraries ---
  const libViews: string[] = [];
  for (const [libKey, lib] of Object.entries(catalog.libraries)) {
    if (!lib.path) continue;
    const libDir = join(dataDir, lib.path);
    if (!hasParquetFiles(libDir)) continue;

    const viewName = libKey.replace(/-/g, "_").replace(/\./g, "_");
    const glob = join(libDir, "*.parquet");
    db.run(
      `CREATE VIEW ${viewName} AS SELECT *, '${libKey}' AS library FROM read_parquet('${glob}', filename=true)`,
    );
    if (lib.data_type === "cross_sections") libViews.push(viewName);
  }
  if (libViews.length > 0) {
    const union = libViews.map((v) => `SELECT * FROM ${v}`).join(" UNION ALL ");
    db.run(`CREATE VIEW xs AS ${union}`);
  }

  // --- Catalog-driven view registration ---
  // All views declared in catalog.json::views — single source of truth.
  // New data tables become queryable by adding an entry to catalog.json,
  // no code changes needed in any client (Python, TypeScript, Rust).
  for (const [viewName, viewDef] of Object.entries(catalog.views ?? {})) {
    const viewPath = join(dataDir, viewDef.path);
    if (viewDef.type === "glob") {
      registerGlob(db, viewPath, viewName);
    } else {
      registerParquet(db, viewPath, viewName);
    }
  }

  // --- Special views that need logic beyond simple registration ---

  // ground_states: when nuclides.parquet exists, override with filtered view
  const nuclidesPath = join(dataDir, "meta", "ensdf", "nuclides.parquet");
  if (existsSync(nuclidesPath)) {
    db.run("CREATE OR REPLACE VIEW ground_states AS SELECT * FROM nuclides WHERE state = ''");
  }

  // EADL aliases: eadl_transitions (v0.11 compat) + fluorescence (radiative subset)
  const eadlDir = join(dataDir, "meta", "eadl");
  if (hasParquetFiles(eadlDir)) {
    db.run("CREATE VIEW eadl_transitions AS SELECT * FROM atomic_relaxation");
    db.run("CREATE VIEW fluorescence AS SELECT * FROM atomic_relaxation WHERE transition_type = 'radiative'");
  }
}

export function getDb(): duckdb.Database {
  if (_db) return _db;
  const dataDir = resolveDataDir();
  const db = new duckdb.Database(":memory:");
  registerViews(db, dataDir);
  _db = db;
  return db;
}

// ---------------------------------------------------------------------------
// DuckDB async helpers
// ---------------------------------------------------------------------------

function dbAll(db: duckdb.Database, sql: string, params?: unknown[]): Promise<Record<string, unknown>[]> {
  return new Promise((resolve, reject) => {
    const cb = (err: Error | null, rows: Record<string, unknown>[]) => {
      if (err) reject(err);
      else resolve(rows);
    };
    if (params && params.length > 0) {
      db.all(sql, ...params, cb);
    } else {
      db.all(sql, cb);
    }
  });
}

function dbRun(db: duckdb.Database, sql: string): Promise<void> {
  return new Promise((resolve, reject) => {
    db.run(sql, (err: Error | null) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

/** Serialize rows to JSON, converting BigInt values to Number. */
function safeStringify(obj: unknown, indent?: number): string {
  return JSON.stringify(
    obj,
    (_key, value) => (typeof value === "bigint" ? Number(value) : value),
    indent,
  );
}

// ---------------------------------------------------------------------------
// Query helper
// ---------------------------------------------------------------------------

async function query(
  sql: string,
  params?: unknown[],
  maxRows: number = 500,
): Promise<{ total: number; truncated: boolean; rows: Record<string, unknown>[] }> {
  const db = getDb();
  // Get total count via subquery
  const countSql = `SELECT CAST(COUNT(*) AS INTEGER) AS n FROM (${sql})`;
  const countRows = await dbAll(db, countSql, params);
  const total = (countRows[0]?.n as number) ?? 0;

  // Get limited rows
  const limitSql = `${sql} LIMIT ${maxRows}`;
  const rows = await dbAll(db, limitSql, params);
  return { total, truncated: total > maxRows, rows };
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

/**
 * Server instructions, naming the data release in the client's context.
 *
 * The referral case #348 is about does not go through `list_libraries`:
 * hyrr-mcp sends an agent straight to `get_cross_sections` for full σ(E)
 * curves, having stated the release *it* computed against. Reporting the
 * release only from a tool the agent has no reason to call would leave that
 * claim as unverifiable as it was — so it is stated here too, where it reaches
 * the model without a round trip, and in `list_libraries` where it can be read
 * programmatically. Both read the same loaded catalog, so they cannot disagree.
 *
 * Kept word-for-word in step with the Rust server's `instructions()`; a test
 * pins that the two do not drift.
 */
export function instructions(dataVersion: string): string {
  return (
    `Serving nucl-parquet data release ${dataVersion}. This identifies the *data*, ` +
    `not this server's version — report it whenever you compare results with another ` +
    `server or tool. If another source states a different release, say so: agreement ` +
    `or disagreement computed across two releases is an artefact of the mismatch, not ` +
    `a physics result. Call list_libraries to read the same value programmatically.`
  );
}

const server = new McpServer(
  {
    name: "nucl-parquet",
    // The package version — the software, not the data. Both are reported
    // because they answer different questions, and only one of them changes
    // when the data does.
    version: PKG_VERSION,
  },
  { instructions: instructions(ensureCatalog().data_version) },
);

// ---------------------------------------------------------------------------
// Library / cross-section tools
// ---------------------------------------------------------------------------

server.tool(
  "list_libraries",
  "List all available nuclear data libraries with projectiles and descriptions",
  {},
  async () => {
    // An envelope, not a bare array: `data_version` identifies the data
    // *release* these libraries came out of, a different fact from any one
    // library's evaluation `version`, and it has nowhere else to live (#348).
    return {
      content: [
        { type: "text" as const, text: safeStringify(libraryListPayload(ensureCatalog()), 2) },
      ],
    };
  },
);

server.tool(
  "list_isotopes",
  "List available target elements for a given library and projectile. Returns element symbols.",
  {
    library: z.string().describe("Library ID, e.g. 'tendl-2025', 'endfb-8.1'"),
    projectile: z.string().describe("Projectile: n, p, d, t, h, a, g"),
  },
  async ({ library, projectile }) => {
    const cat = ensureCatalog();
    const lib = cat.libraries[library];
    if (!lib) throw new Error(`Unknown library: ${library}. Use list_libraries to see available libraries.`);
    if (!lib.projectiles?.includes(projectile)) {
      throw new Error(`Projectile '${projectile}' not available for ${library}. Available: ${lib.projectiles?.join(", ") ?? "none"}`);
    }

    const dataDir = resolveDataDir();
    const libPath = lib.path;
    if (!libPath?.endsWith("xs/")) {
      throw new Error(`Library ${library} path does not end with 'xs/' — cannot derive manifest path`);
    }
    const manifestPath = join(dataDir, libPath.replace("xs/", "manifest.json"));
    if (!existsSync(manifestPath)) {
      throw new Error(`Manifest not found at ${manifestPath}`);
    }
    const manifest = JSON.parse(readFileSync(manifestPath, "utf-8")) as { elements?: string[] };
    const elements = manifest.elements ?? [];

    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ library, projectile, elements, count: elements.length }, 2),
      }],
    };
  },
);

server.tool(
  "get_cross_sections",
  "Get nuclear reaction cross-section data for a specific target element. Returns energy (MeV) and cross-section (mb) arrays with reaction product info.",
  {
    library: z.string().describe("Library ID, e.g. 'tendl-2025'"),
    projectile: z.string().describe("Projectile: n, p, d, t, h, a, g"),
    element: z.string().describe("Target element symbol, e.g. 'Cu', 'Fe', 'Au'"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ library, projectile, element, max_rows }) => {
    if (!/^[npdthag]$/.test(projectile)) {
      throw new Error(`Invalid projectile: '${projectile}'. Must be one of: n, p, d, t, h, a, g`);
    }
    if (!/^[A-Z][a-z]?$/.test(element)) {
      throw new Error(`Invalid element symbol: '${element}'. Must be 1-2 letters (e.g. 'Cu', 'Fe')`);
    }

    const cat = ensureCatalog();
    const lib = cat.libraries[library];
    if (!lib) throw new Error(`Unknown library: ${library}`);

    const dataDir = resolveDataDir();
    const parquetPath = join(dataDir, `${lib.path}${projectile}_${element}.parquet`);
    if (!existsSync(parquetPath)) {
      throw new Error(`No data for ${projectile}_${element} in ${library}`);
    }

    const db = getDb();
    const result = await query(
      `SELECT * FROM read_parquet('${parquetPath}')`,
      [],
      max_rows ?? 500,
    );

    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ library, projectile, element, ...result }, 2),
      }],
    };
  },
);

// ---------------------------------------------------------------------------
// Nuclear structure tools (DuckDB views)
// ---------------------------------------------------------------------------

server.tool(
  "get_decay_data",
  "Get radioactive decay data (half-lives, decay modes, daughters) for a nuclide or element. Filter by Z and/or A.",
  {
    z: z.number().optional().describe("Atomic number (e.g. 92 for U)"),
    a: z.number().optional().describe("Mass number (e.g. 238)"),
  },
  async ({ z: zNum, a: aNum }) => {
    if (zNum === undefined && aNum === undefined) {
      throw new Error("Provide at least z or a to filter decay data.");
    }
    const conditions: string[] = [];
    const params: unknown[] = [];
    if (zNum !== undefined) { conditions.push("Z = ?"); params.push(zNum); }
    if (aNum !== undefined) { conditions.push("A = ?"); params.push(aNum); }

    const result = await query(
      `SELECT * FROM decay WHERE ${conditions.join(" AND ")}`,
      params,
    );
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ z: zNum, a: aNum, count: result.total, rows: result.rows }, 2),
      }],
    };
  },
);

server.tool(
  "get_abundances",
  "Get natural isotope abundances and atomic masses for an element.",
  {
    z: z.number().describe("Atomic number (e.g. 29 for Cu)"),
  },
  async ({ z: zNum }) => {
    const result = await query("SELECT * FROM abundances WHERE Z = ?", [zNum]);
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ z: zNum, count: result.total, isotopes: result.rows }, 2),
      }],
    };
  },
);

server.tool(
  "get_stopping_power",
  "Get mass stopping power (dE/dx) data for a projectile in a target element.",
  {
    source: z.string().describe(
      "Data source: PSTAR (protons), ASTAR (α, NIST ICRU-49), ESTAR (electrons), dSTAR, tSTAR (velocity-scaled from PSTAR), or catima (full Z×Z table)",
    ),
    target_z: z.number().describe("Target element atomic number"),
  },
  async ({ source, target_z }) => {
    const viewMap: Record<string, string> = {
      PSTAR: "stopping", ASTAR: "stopping", ESTAR: "stopping",
      dSTAR: "stopping", tSTAR: "stopping", catima: "catima_stopping",
    };
    if (!(source in viewMap)) {
      throw new Error(`Unknown source '${source}'. Valid: PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, catima`);
    }
    const view = viewMap[source];
    const sql = source === "catima"
      ? `SELECT * FROM ${view} WHERE target_Z = ?`
      : `SELECT * FROM ${view} WHERE source = ? AND target_Z = ?`;
    const params = source === "catima" ? [target_z] : [source, target_z];

    const result = await query(sql, params);
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ source, target_z, count: result.total, rows: result.rows }, 2),
      }],
    };
  },
);

// ---------------------------------------------------------------------------
// Radiation / coincidence / spectra tools
// ---------------------------------------------------------------------------

server.tool(
  "get_radiation",
  "Get radiation emissions (gammas, X-rays, Auger electrons, conversion electrons) for a nuclide.",
  {
    z: z.number().describe("Atomic number of the parent nuclide"),
    a: z.number().optional().describe("Mass number (omit for all isotopes of element Z)"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ z: zNum, a: aNum, max_rows }) => {
    const conditions = ["Z = ?"];
    const params: unknown[] = [zNum];
    if (aNum !== undefined) { conditions.push("A = ?"); params.push(aNum); }

    const result = await query(
      `SELECT * FROM radiation WHERE ${conditions.join(" AND ")}`,
      params,
      max_rows ?? 500,
    );
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ z: zNum, a: aNum, ...result }, 2),
      }],
    };
  },
);

server.tool(
  "get_coincidences",
  "Get gamma-gamma and mixed-emission coincidence pairs for a nuclide. Includes beta/EC/X-ray/Auger/511 keV annihilation paired with gammas.",
  {
    z: z.number().describe("Atomic number of the parent nuclide"),
    a: z.number().optional().describe("Mass number (omit for all isotopes of element Z)"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ z: zNum, a: aNum, max_rows }) => {
    const conditions = ["Z = ?"];
    const params: unknown[] = [zNum];
    if (aNum !== undefined) { conditions.push("A = ?"); params.push(aNum); }

    const result = await query(
      `SELECT * FROM coincidences WHERE ${conditions.join(" AND ")}`,
      params,
      max_rows ?? 500,
    );
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ z: zNum, a: aNum, ...result }, 2),
      }],
    };
  },
);

server.tool(
  "get_summing_partners",
  "Get ICC-corrected summing partners for HPGe true-coincidence-summing (TCS) corrections. Returns emission pairs with pre-computed icc_correction_factor and pure_emission_joint_intensity.",
  {
    z: z.number().describe("Atomic number of the daughter nuclide (filing convention)"),
    a: z.number().describe("Mass number"),
    primary_energy_keV: z.number().optional().describe("Filter to pairs matching this energy on either side"),
    tolerance_keV: z.number().optional().describe("Energy match tolerance in keV (default 0.5)"),
    emission1_rad_type: z.string().optional().describe("Filter side 1: 'gamma', 'xray', 'auger'"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ z: zNum, a: aNum, primary_energy_keV, tolerance_keV, emission1_rad_type, max_rows }) => {
    const conditions = ["Z = ?", "A = ?"];
    const params: unknown[] = [zNum, aNum];
    const tol = tolerance_keV ?? 0.5;

    if (primary_energy_keV !== undefined) {
      conditions.push("(ABS(emission1_energy_keV - ?) < ? OR ABS(emission2_energy_keV - ?) < ?)");
      params.push(primary_energy_keV, tol, primary_energy_keV, tol);
    }
    if (emission1_rad_type !== undefined) {
      conditions.push("emission1_rad_type = ?");
      params.push(emission1_rad_type);
    }

    const result = await query(
      `SELECT * FROM summing_partners WHERE ${conditions.join(" AND ")} ORDER BY pure_emission_joint_intensity DESC`,
      params,
      max_rows ?? 500,
    );
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ z: zNum, a: aNum, primary_energy_keV, ...result }, 2),
      }],
    };
  },
);

server.tool(
  "get_emissions",
  "Get absolute per-decay photon emission intensities (NuDat-equivalent). Returns gamma emissions for a parent nuclide with absolute intensities (photon emission probability per decay, 0-100%). Filed by parent, not daughter.",
  {
    parent_z: z.number().describe("Atomic number of the decaying parent (e.g. 27 for Co-60)"),
    parent_a: z.number().describe("Mass number of the parent (e.g. 60 for Co-60)"),
    parent_state: z.string().optional().describe("Nuclear state: '' (ground), 'm', 'm2'"),
    decay_mode: z.string().optional().describe("Filter by decay mode: 'beta-', 'KshellEC', 'IT', etc."),
    energy_keV: z.number().optional().describe("Filter to gammas near this energy"),
    tolerance_keV: z.number().optional().describe("Energy tolerance (default 0.5 keV)"),
    min_intensity_pct: z.number().optional().describe("Minimum absolute intensity (%) to include"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ parent_z, parent_a, parent_state, decay_mode, energy_keV, tolerance_keV, min_intensity_pct, max_rows }) => {
    const conditions = ["parent_Z = ?", "parent_A = ?", "parent_state = ?"];
    const params: unknown[] = [parent_z, parent_a, parent_state ?? ""];
    const tol = tolerance_keV ?? 0.5;

    if (decay_mode !== undefined) {
      conditions.push("decay_mode = ?");
      params.push(decay_mode);
    }
    if (energy_keV !== undefined) {
      conditions.push("ABS(energy_keV - ?) < ?");
      params.push(energy_keV, tol);
    }
    if (min_intensity_pct !== undefined && min_intensity_pct > 0) {
      conditions.push("intensity_pct >= ?");
      params.push(min_intensity_pct);
    }

    const result = await query(
      `SELECT * FROM emissions WHERE ${conditions.join(" AND ")} ORDER BY intensity_pct DESC`,
      params,
      max_rows ?? 500,
    );
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ parent_z, parent_a, parent_state: parent_state ?? "", ...result }, 2),
      }],
    };
  },
);

server.tool(
  "get_beta_spectrum",
  "Get the continuous beta-decay kinetic-energy spectrum for a nuclide. Returns pre-tabulated Fermi-function spectra (dN/dE normalized to 1).",
  {
    z: z.number().describe("Atomic number of the parent nuclide"),
    a: z.number().describe("Mass number of the parent nuclide"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ z: zNum, a: aNum, max_rows }) => {
    const result = await query(
      "SELECT * FROM beta_spectra WHERE Z = ? AND A = ?",
      [zNum, aNum],
      max_rows ?? 500,
    );
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ z: zNum, a: aNum, ...result }, 2),
      }],
    };
  },
);

server.tool(
  "get_compound_compositions",
  "Get elemental compositions (weight fractions) for NIST XCOM standard materials. Useful for Bragg-additive cross-section calculations.",
  {
    material: z.string().optional().describe("Material name (e.g. 'Water, Liquid'). Omit to list all materials."),
  },
  async ({ material }) => {
    const db = getDb();
    if (material === undefined) {
      const rows = await dbAll(db, "SELECT DISTINCT material FROM compound_compositions ORDER BY material");
      const materials = rows.map((r) => r.material as string);
      return {
        content: [{
          type: "text" as const,
          text: safeStringify({ count: materials.length, materials }, 2),
        }],
      };
    }
    const result = await query(
      "SELECT * FROM compound_compositions WHERE material = ?",
      [material],
    );
    if (result.total === 0) {
      const allRows = await dbAll(db, "SELECT DISTINCT material FROM compound_compositions ORDER BY material");
      const all = allRows.map((r) => r.material as string);
      throw new Error(`Unknown material: '${material}'. Available: ${all.join(", ")}`);
    }
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ material, count: result.total, composition: result.rows }, 2),
      }],
    };
  },
);

server.tool(
  "get_electron_stopping",
  "Get electron stopping power with collision/radiative split. Richer than ESTAR — includes ~183 compounds plus all elements Z=1..98.",
  {
    target: z.string().optional().describe("Compound name (e.g. 'G4_WATER'). For elements use target_z instead."),
    target_z: z.number().optional().describe("Atomic number for elemental targets"),
    max_rows: z.number().optional().describe("Max rows to return (default 500)"),
  },
  async ({ target, target_z, max_rows }) => {
    if (target === undefined && target_z === undefined) {
      throw new Error("Provide target (compound name) or target_z (atomic number).");
    }
    let result;
    if (target_z !== undefined) {
      result = await query(
        "SELECT * FROM electron_stopping WHERE target_Z = ?",
        [target_z],
        max_rows ?? 500,
      );
    } else {
      result = await query(
        "SELECT * FROM electron_stopping WHERE name = ? OR g4_name = ?",
        [target, target],
        max_rows ?? 500,
      );
    }
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ target, target_z, ...result }, 2),
      }],
    };
  },
);

// ---------------------------------------------------------------------------
// SQL escape hatch (Sub-E #179)
// ---------------------------------------------------------------------------

// Patterns that indicate file-access functions — blocked in user SQL to prevent
// read_parquet('/etc/passwd') style attacks. The pre-registered DuckDB views
// already expose all nuclear data; there's no legitimate need for raw file access.
const BLOCKED_FUNCTIONS = /\b(read_parquet|parquet_scan|parquet_metadata|parquet_schema|read_csv|read_csv_auto|read_json|read_json_auto|read_text|read_blob|glob|copy|export|attach|load|install|create|drop|alter|insert|update|delete|truncate|query_table|pragma)\b/i;

const ALLOWED_FIRST_WORDS = new Set(["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"]);

server.tool(
  "sql_query",
  "Execute read-only SQL against all 70+ nuclear data tables. Supports JOINs, aggregations, window functions. Use describe_schema() to discover available tables and columns.",
  {
    sql: z.string().describe("Read-only SQL query. DDL/DML will be rejected."),
    max_rows: z.number().optional().describe("Max rows to return (default 10000)"),
  },
  async ({ sql: userSql, max_rows }) => {
    const stripped = userSql.trim();
    if (!stripped) throw new Error("Empty SQL query");

    const firstWord = stripped.split(/\s/)[0].toUpperCase();
    if (!ALLOWED_FIRST_WORDS.has(firstWord)) {
      throw new Error(`Only read queries are allowed (SELECT, WITH, EXPLAIN, DESCRIBE). Got: ${firstWord}`);
    }
    if (BLOCKED_FUNCTIONS.test(stripped)) {
      throw new Error("Only read queries are allowed (SELECT, WITH, EXPLAIN, DESCRIBE). File-access and DDL functions are blocked.");
    }

    const db = getDb();
    const limit = max_rows ?? 10000;
    let total: number;
    let rows: Record<string, unknown>[];
    try {
      // Get total count first, then fetch limited rows
      const countRows = await dbAll(db, `SELECT CAST(COUNT(*) AS INTEGER) AS n FROM (${stripped})`);
      total = (countRows[0]?.n as number) ?? 0;
      rows = await dbAll(db, `SELECT * FROM (${stripped}) LIMIT ${limit}`);
    } catch (e) {
      throw new Error(`SQL error: ${(e as Error).message}`);
    }

    const truncated = total > limit;
    const display = rows;

    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ total, truncated, rows: display }, 2),
      }],
    };
  },
);

server.tool(
  "describe_schema",
  "List all available tables/views with their column names and types. Use this to discover what data is available before writing SQL queries.",
  {},
  async () => {
    const db = getDb();
    const tables = await dbAll(db, "SHOW TABLES");
    const tableNames = tables.map((r) => r.name as string).sort();

    const schema: Record<string, { name: string; type: string }[]> = {};
    for (const tbl of tableNames) {
      try {
        const cols = await dbAll(db, `DESCRIBE ${tbl}`);
        schema[tbl] = cols.map((c) => ({ name: c.column_name as string, type: c.column_type as string }));
      } catch {
        schema[tbl] = [];
      }
    }
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ tables: tableNames.length, schema }, 2),
      }],
    };
  },
);

server.tool(
  "list_tables",
  "List all available table/view names (short form of describe_schema).",
  {},
  async () => {
    const db = getDb();
    const tables = await dbAll(db, "SHOW TABLES");
    const tableNames = tables.map((r) => r.name as string).sort();
    return {
      content: [{
        type: "text" as const,
        text: safeStringify({ count: tableNames.length, tables: tableNames }, 2),
      }],
    };
  },
);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const transport = new StdioServerTransport();
await server.connect(transport);
