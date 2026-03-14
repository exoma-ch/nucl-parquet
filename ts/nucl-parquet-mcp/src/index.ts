import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { parquetRead } from "hyparquet";
import { compressors } from "hyparquet-compressors";
import { z } from "zod";

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL =
  process.env.NUCL_PARQUET_BASE_URL ??
  "https://raw.githubusercontent.com/exoma-ch/nucl-parquet/main/";

// ---------------------------------------------------------------------------
// Embedded catalog (from catalog.json)
// ---------------------------------------------------------------------------

interface Library {
  name: string;
  description: string;
  source_url: string;
  projectiles: string[];
  data_type: string;
  version: string;
  path: string;
}

interface Catalog {
  libraries: Record<string, Library>;
  shared: {
    meta: { path: string; files: Record<string, string> };
    stopping: { path: string; files: Record<string, string>; sources: string[] };
  };
}

const CATALOG: Catalog = {
  libraries: {
    "tendl-2024": { name: "TENDL-2024", description: "TALYS Evaluated Nuclear Data Library 2024 (IAEA/PSI)", source_url: "https://tendl.web.psi.ch/tendl_2024/tendl2024.html", projectiles: ["p", "d", "t", "h", "a"], data_type: "cross_sections", version: "2024", path: "tendl-2024/xs/" },
    "endfb-8.1": { name: "ENDF/B-VIII.1", description: "US Evaluated Nuclear Data File (NNDC/BNL)", source_url: "https://www.nndc.bnl.gov/endf-b8.1/", projectiles: ["n", "p", "d", "t", "h", "a"], data_type: "cross_sections", version: "VIII.1", path: "endfb-8.1/xs/" },
    "jeff-4.0": { name: "JEFF-4.0", description: "Joint Evaluated Fission and Fusion File (NEA)", source_url: "https://www.oecd-nea.org/dbdata/jeff/", projectiles: ["n", "p"], data_type: "cross_sections", version: "4.0", path: "jeff-4.0/xs/" },
    "jendl-5": { name: "JENDL-5", description: "Japanese Evaluated Nuclear Data Library (JAEA)", source_url: "https://wwwndc.jaea.go.jp/jendl/j5/j5.html", projectiles: ["n", "p", "d", "a"], data_type: "cross_sections", version: "5", path: "jendl-5/xs/" },
    "tendl-2025": { name: "TENDL-2025", description: "TALYS Evaluated Nuclear Data Library 2025 (PSI)", source_url: "https://tendl.web.psi.ch/", projectiles: ["n", "p", "d", "t", "h", "a"], data_type: "cross_sections", version: "2025", path: "tendl-2025/xs/" },
    "cendl-3.2": { name: "CENDL-3.2", description: "Chinese Evaluated Nuclear Data Library (CIAE)", source_url: "http://www.nuclear.csdb.cn/", projectiles: ["n"], data_type: "cross_sections", version: "3.2", path: "cendl-3.2/xs/" },
    "brond-3.1": { name: "BROND-3.1", description: "Russian Evaluated Nuclear Data Library (IPPE)", source_url: "https://vant.ippe.ru/", projectiles: ["n"], data_type: "cross_sections", version: "3.1", path: "brond-3.1/xs/" },
    "fendl-3.2": { name: "FENDL-3.2", description: "Fusion Evaluated Nuclear Data Library (IAEA)", source_url: "https://www-nds.iaea.org/fendl/", projectiles: ["n"], data_type: "cross_sections", version: "3.2", path: "fendl-3.2/xs/" },
    "eaf-2010": { name: "EAF-2010", description: "European Activation File (CCFE)", source_url: "https://fispact.ukaea.uk/", projectiles: ["n"], data_type: "cross_sections", version: "2010", path: "eaf-2010/xs/" },
    "irdff-2": { name: "IRDFF-II", description: "International Reactor Dosimetry and Fusion File (IAEA)", source_url: "https://www-nds.iaea.org/IRDFF/", projectiles: ["n"], data_type: "cross_sections", version: "II", path: "irdff-2/xs/" },
    "iaea-medical": { name: "IAEA-Medical", description: "Medical isotope production cross-sections (IAEA)", source_url: "https://www-nds.iaea.org/medical/", projectiles: ["p", "d", "h", "a"], data_type: "cross_sections", version: "latest", path: "iaea-medical/xs/" },
    "jendl-ad-2017": { name: "JENDL/AD-2017", description: "Activation/Dosimetry Library (JAEA)", source_url: "https://wwwndc.jaea.go.jp/jendl/jad/jad.html", projectiles: ["n", "p"], data_type: "cross_sections", version: "2017", path: "jendl-ad-2017/xs/" },
    "jendl-deu-2020": { name: "JENDL-DEU-2020", description: "Dedicated deuteron-induced reaction library (JAEA)", source_url: "https://wwwndc.jaea.go.jp/jendl/deu/deu.html", projectiles: ["d"], data_type: "cross_sections", version: "2020", path: "jendl-deu-2020/xs/" },
    "iaea-pd-2019": { name: "IAEA-PD-2019", description: "Photonuclear Data Library (IAEA)", source_url: "https://www-nds.iaea.org/photonuclear/", projectiles: ["g"], data_type: "cross_sections", version: "2019", path: "iaea-pd-2019/xs/" },
    "exfor": { name: "EXFOR", description: "Experimental nuclear reaction data (IAEA NDS)", source_url: "https://www-nds.iaea.org/exfor/", projectiles: ["n", "p", "d", "t", "h", "a"], data_type: "experimental_cross_sections", version: "latest", path: "exfor/" },
  },
  shared: {
    meta: { path: "meta/", files: { abundances: "abundances.parquet", decay: "decay.parquet", elements: "elements.parquet" } },
    stopping: { path: "stopping/", files: { stopping: "stopping.parquet" }, sources: ["PSTAR", "ASTAR", "ICRU73", "MSTAR"] },
  },
};

// ---------------------------------------------------------------------------
// Parquet fetch + cache
// ---------------------------------------------------------------------------

const parquetCache = new Map<string, Record<string, unknown>[]>();

export async function fetchParquetRows(
  relativePath: string,
  baseUrl: string = BASE_URL,
): Promise<Record<string, unknown>[]> {
  if (parquetCache.has(relativePath)) return parquetCache.get(relativePath)!;

  const url = new URL(relativePath, baseUrl).href;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching ${url}`);
  const buffer = await resp.arrayBuffer();

  const rows: Record<string, unknown>[] = [];
  await parquetRead({
    file: buffer,
    compressors,
    rowFormat: "object",
    onComplete: (data: Record<string, unknown>[]) => {
      rows.push(...data);
    },
  });

  parquetCache.set(relativePath, rows);
  return rows;
}

async function fetchManifest(libraryId: string): Promise<Record<string, unknown>> {
  const lib = CATALOG.libraries[libraryId];
  if (!lib) throw new Error(`Unknown library: ${libraryId}`);
  const manifestPath = lib.path.replace(/xs\/$/, "manifest.json");
  const url = new URL(manifestPath, BASE_URL).href;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`HTTP ${resp.status} fetching manifest for ${libraryId}`);
  return (await resp.json()) as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function getCatalog(): Catalog {
  return CATALOG;
}

function truncateRows(rows: Record<string, unknown>[], limit: number): { rows: Record<string, unknown>[]; truncated: boolean; total: number } {
  if (rows.length <= limit) return { rows, truncated: false, total: rows.length };
  return { rows: rows.slice(0, limit), truncated: true, total: rows.length };
}

// ---------------------------------------------------------------------------
// MCP Server
// ---------------------------------------------------------------------------

const server = new McpServer({
  name: "nucl-parquet",
  version: "0.3.6",
});

server.tool(
  "list_libraries",
  "List all available nuclear data libraries with projectiles and descriptions",
  {},
  async () => {
    const libs = Object.entries(CATALOG.libraries).map(([id, lib]) => ({
      id,
      name: lib.name,
      description: lib.description,
      projectiles: lib.projectiles,
      version: lib.version,
      data_type: lib.data_type,
    }));
    return { content: [{ type: "text" as const, text: JSON.stringify(libs, null, 2) }] };
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
    const lib = CATALOG.libraries[library];
    if (!lib) throw new Error(`Unknown library: ${library}. Use list_libraries to see available libraries.`);
    if (!lib.projectiles.includes(projectile)) {
      throw new Error(`Projectile '${projectile}' not available for ${library}. Available: ${lib.projectiles.join(", ")}`);
    }

    const manifest = await fetchManifest(library);
    const elements = (manifest.elements as string[]) ?? [];

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({ library, projectile, elements, count: elements.length }, null, 2),
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
    const lib = CATALOG.libraries[library];
    if (!lib) throw new Error(`Unknown library: ${library}`);

    const parquetPath = `${lib.path}${projectile}_${element}.parquet`;
    const rows = await fetchParquetRows(parquetPath);
    const result = truncateRows(rows, max_rows ?? 500);

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          library,
          projectile,
          element,
          ...result,
        }, null, 2),
      }],
    };
  },
);

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

    const metaPath = CATALOG.shared.meta.path + CATALOG.shared.meta.files.decay;
    const rows = await fetchParquetRows(metaPath);

    const filtered = rows.filter((row) => {
      if (zNum !== undefined && row["Z"] !== zNum) return false;
      if (aNum !== undefined && row["A"] !== aNum) return false;
      return true;
    });

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({ z: zNum, a: aNum, count: filtered.length, rows: filtered }, null, 2),
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
    const metaPath = CATALOG.shared.meta.path + CATALOG.shared.meta.files.abundances;
    const rows = await fetchParquetRows(metaPath);

    const filtered = rows.filter((row) => row["Z"] === zNum);

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({ z: zNum, count: filtered.length, isotopes: filtered }, null, 2),
      }],
    };
  },
);

server.tool(
  "get_stopping_power",
  "Get mass stopping power (dE/dx) data for a projectile in a target element.",
  {
    source: z.string().describe("Data source: PSTAR, ASTAR, ICRU73, or MSTAR"),
    target_z: z.number().describe("Target element atomic number"),
  },
  async ({ source, target_z }) => {
    const stoppingPath = CATALOG.shared.stopping.path + CATALOG.shared.stopping.files.stopping;
    const rows = await fetchParquetRows(stoppingPath);

    const filtered = rows.filter(
      (row) => row["source"] === source && row["target_Z"] === target_z,
    );

    return {
      content: [{
        type: "text" as const,
        text: JSON.stringify({
          source,
          target_z,
          count: filtered.length,
          rows: filtered,
        }, null, 2),
      }],
    };
  },
);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const transport = new StdioServerTransport();
await server.connect(transport);
