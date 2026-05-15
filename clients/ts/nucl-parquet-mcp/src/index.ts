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

interface Catalog {
  libraries: Record<string, Library>;
  [key: string]: unknown;
}

let _catalog: Catalog | undefined;

export function ensureCatalog(): Catalog {
  if (_catalog) return _catalog;
  const dataDir = resolveDataDir();
  const raw = readFileSync(join(dataDir, "catalog.json"), "utf-8");
  _catalog = JSON.parse(raw) as Catalog;
  return _catalog;
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

  // --- Shared meta ---
  registerParquet(db, join(dataDir, "meta", "abundances.parquet"), "abundances");
  registerParquet(db, join(dataDir, "meta", "decay.parquet"), "decay");
  registerParquet(db, join(dataDir, "meta", "elements.parquet"), "elements");

  // --- Stopping powers ---
  registerGlob(db, join(dataDir, "stopping"), "stopping");
  registerParquet(db, join(dataDir, "stopping", "catima", "catima.parquet"), "catima_stopping");
  registerParquet(db, join(dataDir, "stopping", "compounds", "PSTAR_compounds.parquet"), "pstar_compounds");
  registerParquet(db, join(dataDir, "stopping", "compounds", "ASTAR_compounds.parquet"), "astar_compounds");
  registerParquet(db, join(dataDir, "stopping", "em", "electron_stopping.parquet"), "electron_stopping");
  registerParquet(db, join(dataDir, "stopping", "em", "density_effect_params.parquet"), "density_effect_params");

  // --- ENSDF data ---
  const nuclidesPath = join(dataDir, "meta", "ensdf", "nuclides.parquet");
  if (existsSync(nuclidesPath)) {
    registerParquet(db, nuclidesPath, "nuclides");
    db.run("CREATE VIEW ground_states AS SELECT * FROM nuclides WHERE state = ''");
  } else {
    registerParquet(db, join(dataDir, "meta", "ensdf", "ground_states.parquet"), "ground_states");
  }
  registerGlob(db, join(dataDir, "meta", "ensdf", "gammas"), "ensdf_gammas");
  registerGlob(db, join(dataDir, "meta", "ensdf", "levels"), "ensdf_levels");
  registerGlob(db, join(dataDir, "meta", "ensdf", "radiation"), "radiation");
  registerGlob(db, join(dataDir, "meta", "ensdf", "coincidences"), "coincidences");
  registerGlob(db, join(dataDir, "meta", "ensdf", "beta_spectra"), "beta_spectra");

  // --- NUDEX data ---
  registerParquet(db, join(dataDir, "meta", "capture_gammas.parquet"), "capture_gammas");
  registerParquet(db, join(dataDir, "meta", "capture_gammas_summary.parquet"), "capture_gammas_summary");
  registerParquet(db, join(dataDir, "meta", "icc_factors.parquet"), "icc_factors");
  registerParquet(db, join(dataDir, "meta", "level_density_bfm.parquet"), "level_density_bfm");
  registerParquet(db, join(dataDir, "meta", "level_density_ctm.parquet"), "level_density_ctm");
  registerParquet(db, join(dataDir, "meta", "level_density_params.parquet"), "level_density_params");
  registerParquet(db, join(dataDir, "meta", "psf_e1.parquet"), "psf_e1");
  registerParquet(db, join(dataDir, "meta", "psf_gdr_lor.parquet"), "psf_gdr_lor");
  registerParquet(db, join(dataDir, "meta", "psf_gdr_mlo.parquet"), "psf_gdr_mlo");
  registerParquet(db, join(dataDir, "meta", "psf_gdr_slo.parquet"), "psf_gdr_slo");
  registerParquet(db, join(dataDir, "meta", "psf_gdr_theor.parquet"), "psf_gdr_theor");
  registerParquet(db, join(dataDir, "meta", "psf_photonuclear.parquet"), "psf_photonuclear");
  registerParquet(db, join(dataDir, "meta", "nudex_shellcor.parquet"), "nudex_shellcor");
  registerParquet(db, join(dataDir, "meta", "nudex_special_inputs.parquet"), "nudex_special_inputs");
  registerParquet(db, join(dataDir, "meta", "nudex_general_stat.parquet"), "nudex_general_stat");
  registerParquet(db, join(dataDir, "meta", "nudex_levels.parquet"), "nudex_levels");
  registerParquet(db, join(dataDir, "meta", "nudex_level_gammas.parquet"), "nudex_level_gammas");
  registerParquet(db, join(dataDir, "meta", "nudex_isotopes.parquet"), "nudex_isotopes");
  registerParquet(db, join(dataDir, "meta", "spectrum_xs.parquet"), "spectrum_xs");

  // --- Kerma, neutron, dose ---
  registerGlob(db, join(dataDir, "meta", "kerma"), "kerma");
  registerGlob(db, join(dataDir, "meta", "neutron_total"), "neutron_total");
  registerParquet(db, join(dataDir, "meta", "dose_constants.parquet"), "dose_constants");

  // --- XCOM / compound ---
  registerParquet(db, join(dataDir, "meta", "xcom_elements.parquet"), "xcom_elements");
  registerParquet(db, join(dataDir, "meta", "xcom_compounds.parquet"), "xcom_compounds");
  registerParquet(db, join(dataDir, "meta", "compound_compositions.parquet"), "compound_compositions");

  // --- EPDL97 ---
  registerGlob(db, join(dataDir, "meta", "epdl97", "photon_xs"), "epdl_photon_xs");
  registerGlob(db, join(dataDir, "meta", "epdl97", "form_factors"), "epdl_form_factors");
  registerGlob(db, join(dataDir, "meta", "epdl97", "scattering_fn"), "epdl_scattering_fn");
  registerGlob(db, join(dataDir, "meta", "epdl97", "anomalous"), "epdl_anomalous");
  registerGlob(db, join(dataDir, "meta", "epdl97", "subshell_pe"), "epdl_subshell_pe");

  // --- EADL ---
  const eadlDir = join(dataDir, "meta", "eadl");
  if (hasParquetFiles(eadlDir)) {
    registerGlob(db, eadlDir, "atomic_relaxation");
    db.run("CREATE VIEW eadl_transitions AS SELECT * FROM atomic_relaxation");
    db.run("CREATE VIEW fluorescence AS SELECT * FROM atomic_relaxation WHERE transition_type = 'radiative'");
  }

  // --- EEDL ---
  registerGlob(db, join(dataDir, "meta", "eedl"), "eedl_electron_xs");

  // --- G4EMLOW photon/electron ---
  registerParquet(db, join(dataDir, "em", "photon_pair.parquet"), "photon_pair");
  registerParquet(db, join(dataDir, "em", "photon_rayleigh_cdf.parquet"), "photon_rayleigh_cdf");
  registerParquet(db, join(dataDir, "em", "xray_form_factor.parquet"), "xray_form_factor");
  registerParquet(db, join(dataDir, "em", "photon_compton.parquet"), "photon_compton");
  registerParquet(db, join(dataDir, "em", "compton_scattering_function.parquet"), "compton_scattering_function");
  registerParquet(db, join(dataDir, "em", "compton_doppler_profiles.parquet"), "compton_doppler_profiles");
  registerParquet(db, join(dataDir, "em", "photon_pe.parquet"), "photon_pe");
  registerParquet(db, join(dataDir, "em", "photon_pe_high_z_params.parquet"), "photon_pe_high_z_params");
  registerParquet(db, join(dataDir, "em", "photon_pe_angular.parquet"), "photon_pe_angular");
  registerParquet(db, join(dataDir, "em", "photon_pe_total.parquet"), "photon_pe_total");
  registerParquet(db, join(dataDir, "em", "electron_brem.parquet"), "electron_brem");
  registerParquet(db, join(dataDir, "em", "electron_brem_sb_dcs.parquet"), "electron_brem_sb_dcs");
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
// Z → element symbol
// ---------------------------------------------------------------------------

const Z_TO_SYMBOL = [
  "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
  "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
  "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
  "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
] as const;

function zToSymbol(zNum: number): string {
  if (zNum < 1 || zNum >= Z_TO_SYMBOL.length) {
    throw new Error(`Z=${zNum} out of range (1-${Z_TO_SYMBOL.length - 1})`);
  }
  return Z_TO_SYMBOL[zNum];
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "nucl-parquet",
  version: PKG_VERSION,
});

// ---------------------------------------------------------------------------
// Library / cross-section tools
// ---------------------------------------------------------------------------

server.tool(
  "list_libraries",
  "List all available nuclear data libraries with projectiles and descriptions",
  {},
  async () => {
    const cat = ensureCatalog();
    const libs = Object.entries(cat.libraries)
      .filter(([, lib]) => lib.projectiles)
      .map(([id, lib]) => ({
        id,
        name: lib.name,
        description: lib.description,
        projectiles: lib.projectiles,
        version: lib.version,
        data_type: lib.data_type,
      }));
    return { content: [{ type: "text" as const, text: safeStringify(libs, 2) }] };
  },
);

server.tool(
  "list_isotopes",
  "List available target elements for a given library and projectile. Returns element symbols.",
  {
    library: z.string().describe("Library ID, e.g. 'tendl-2024', 'endfb-8.1'"),
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
    library: z.string().describe("Library ID, e.g. 'tendl-2024'"),
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
const BLOCKED_FUNCTIONS = /\b(read_parquet|read_csv|read_csv_auto|read_json|read_json_auto|read_text|glob|copy|export|attach|load|install|create|drop|alter|insert|update|delete|truncate)\b/i;

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
    let rows: Record<string, unknown>[];
    try {
      rows = await dbAll(db, stripped);
    } catch (e) {
      throw new Error(`SQL error: ${(e as Error).message}`);
    }

    const total = rows.length;
    const limit = max_rows ?? 10000;
    const truncated = total > limit;
    const display = truncated ? rows.slice(0, limit) : rows;

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
