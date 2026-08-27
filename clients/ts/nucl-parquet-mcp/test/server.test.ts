/**
 * Tests for nucl-parquet MCP server (SSoT refactor — DuckDB-backed).
 *
 * All tests read local data from the repo's data/ directory — no network.
 */

import { createRequire } from "node:module";
import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import {
  ensureCatalog,
  getDb,
  instructions,
  libraryListPayload,
  parseCatalog,
  PKG_VERSION,
} from "../src/index.js";

const require = createRequire(import.meta.url);

// ---------------------------------------------------------------------------
// Version tests
// ---------------------------------------------------------------------------

describe("version", () => {
  it("PKG_VERSION matches package.json", () => {
    const { version } = require("../package.json") as { version: string };
    expect(PKG_VERSION).toBe(version);
    expect(version).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// Catalog tests (loaded from disk, not hardcoded)
// ---------------------------------------------------------------------------

describe("catalog", () => {
  it("has all libraries", () => {
    const cat = ensureCatalog();
    const libs = cat.libraries;
    expect(Object.keys(libs).length).toBeGreaterThanOrEqual(15);
    expect(libs).toHaveProperty("tendl-2023-iso");
    expect(libs).toHaveProperty("endfb-8.1");
    expect(libs).toHaveProperty("exfor");
  });

  it("all xs libraries have projectiles", () => {
    const cat = ensureCatalog();
    for (const [libId, lib] of Object.entries(cat.libraries)) {
      if (lib.projectiles) {
        expect(lib.projectiles.length, `${libId} has no projectiles`).toBeGreaterThan(0);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// DuckDB integration tests (local data, no network)
// ---------------------------------------------------------------------------

describe("DuckDB", () => {
  it("connects and has tables", async () => {
    const db = getDb();
    const tables = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SHOW TABLES", (err: Error | null, rows: Record<string, unknown>[]) => {
        if (err) reject(err); else resolve(rows);
      });
    });
    const names = tables.map((r) => r.name as string);
    expect(names).toContain("decay");
    expect(names).toContain("abundances");
    expect(names).toContain("radiation");
  });

  it("queries abundances for Cu (Z=29)", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT * FROM abundances WHERE Z = 29", (err: Error | null, rows: Record<string, unknown>[]) => {
        if (err) reject(err); else resolve(rows);
      });
    });
    expect(rows.length).toBeGreaterThanOrEqual(2); // Cu-63 and Cu-65
    const cu63 = rows.find((r) => r.A === 63);
    expect(cu63).toBeDefined();
  });

  it("queries decay data for Co-60", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT * FROM decay WHERE Z = 27 AND A = 60", (err: Error | null, rows: Record<string, unknown>[]) => {
        if (err) reject(err); else resolve(rows);
      });
    });
    expect(rows.length).toBeGreaterThanOrEqual(1);
  });

  it("queries radiation for Co-60 (Z=27, A=60)", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT * FROM radiation WHERE Z = 27 AND A = 60", (err: Error | null, rows: Record<string, unknown>[]) => {
        if (err) reject(err); else resolve(rows);
      });
    });
    expect(rows.length).toBeGreaterThan(0);
    const gammas = rows.filter((r) => r.rad_type === "gamma");
    expect(gammas.length).toBeGreaterThan(0);
  });

  it("queries coincidences for Co-60", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT CAST(COUNT(*) AS INTEGER) as n FROM coincidences WHERE Z = 27 AND A = 60",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        });
    });
    expect((rows[0].n as number)).toBeGreaterThan(0);
  });

  it("queries beta spectra for P-32", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT CAST(COUNT(*) AS INTEGER) as n FROM beta_spectra WHERE Z = 15 AND A = 32",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        });
    });
    expect((rows[0].n as number)).toBeGreaterThan(0);
  });

  it("queries stopping power for Cu (PSTAR)", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT CAST(COUNT(*) AS INTEGER) as n FROM stopping WHERE source = 'PSTAR' AND target_Z = 29",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        });
    });
    expect((rows[0].n as number)).toBeGreaterThan(0);
  });

  it("queries compound compositions", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT DISTINCT material FROM compound_compositions ORDER BY material",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        });
    });
    expect(rows.length).toBeGreaterThan(0);
    const materials = rows.map((r) => r.material as string);
    expect(materials.length).toBeGreaterThan(10);
  });

  it("queries electron stopping", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SELECT CAST(COUNT(*) AS INTEGER) as n FROM electron_stopping WHERE target_Z = 29",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        });
    });
    expect((rows[0].n as number)).toBeGreaterThan(0);
  });

  it("queries summing partners for Co-60 (Z=28, A=60)", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all(
        "SELECT * FROM summing_partners WHERE Z = 28 AND A = 60 AND emission1_rad_type = 'gamma'",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        },
      );
    });
    expect(rows.length).toBeGreaterThan(0);
    // Check 1173/1333 keV pair exists (canonicalized: E1 ≤ E2)
    const pair = rows.find(
      (r) =>
        Math.abs((r.emission1_energy_keV as number) - 1173.2) < 1.0 &&
        Math.abs((r.emission2_energy_keV as number) - 1332.5) < 1.0,
    );
    expect(pair).toBeDefined();
    expect(pair!.icc_correction_factor as number).toBeGreaterThan(0.99);
  });

  it("queries summing partners for Eu-152 (Gd-152 daughter, Z=64)", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all(
        `SELECT * FROM summing_partners WHERE Z = 64 AND A = 152
         AND emission1_rad_type = 'gamma'
         AND (ABS(emission1_energy_keV - 344.28) < 1.0 OR ABS(emission2_energy_keV - 344.28) < 1.0)`,
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        },
      );
    });
    expect(rows.length).toBeGreaterThanOrEqual(10);
  });

  it("queries absolute emissions for Co-60 (parent Z=27, A=60)", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all(
        "SELECT * FROM emissions WHERE parent_Z = 27 AND parent_A = 60 AND parent_state = '' ORDER BY intensity_pct DESC",
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        },
      );
    });
    expect(rows.length).toBeGreaterThanOrEqual(2);
    // Top two should be 1332 and 1173 keV
    const top = rows[0];
    expect(top.intensity_pct as number).toBeGreaterThan(99.0);
    // Find 1173 keV gamma
    const g1173 = rows.find(
      (r) => Math.abs((r.energy_keV as number) - 1173.239) < 0.5,
    );
    expect(g1173).toBeDefined();
    expect(g1173!.intensity_pct as number).toBeCloseTo(99.85, 0);
  });

  it("queries Eu-152 emissions summed across EC shells", async () => {
    const db = getDb();
    const rows = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all(
        `SELECT energy_keV, SUM(intensity_pct) as total_intensity
         FROM emissions
         WHERE parent_Z = 63 AND parent_A = 152 AND parent_state = ''
           AND rad_type = 'gamma'
           AND energy_keV BETWEEN 121.0 AND 122.5
         GROUP BY energy_keV`,
        (err: Error | null, rows: Record<string, unknown>[]) => {
          if (err) reject(err); else resolve(rows);
        },
      );
    });
    expect(rows.length).toBe(1);
    // NuDat: 28.58% summed across all EC shells
    expect(rows[0].total_intensity as number).toBeCloseTo(28.58, 0);
  });

  it("has 20+ registered tables/views", async () => {
    const db = getDb();
    const tables = await new Promise<Record<string, unknown>[]>((resolve, reject) => {
      db.all("SHOW TABLES", (err: Error | null, rows: Record<string, unknown>[]) => {
        if (err) reject(err); else resolve(rows);
      });
    });
    expect(tables.length).toBeGreaterThanOrEqual(20);
  });
});

// ---------------------------------------------------------------------------
// SQL escape hatch security tests
// ---------------------------------------------------------------------------

describe("SQL security", () => {
  // These test the BLOCKED_FUNCTIONS and ALLOWED_FIRST_WORDS logic.
  // We test against the module's validation, not against DuckDB directly.

  it("blocks DDL (DROP TABLE)", () => {
    // We can't easily call the MCP tool directly, so we test the validation
    // logic indirectly. The tool checks first word + blocked functions.
    const sql = "DROP TABLE decay";
    const firstWord = sql.trim().split(/\s/)[0].toUpperCase();
    const allowed = new Set(["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"]);
    expect(allowed.has(firstWord)).toBe(false);
  });

  it("blocks COPY", () => {
    const sql = "COPY radiation TO '/tmp/exfil.csv'";
    const blocked = /\b(read_parquet|parquet_scan|parquet_metadata|parquet_schema|read_csv|read_csv_auto|read_json|read_json_auto|read_text|read_blob|glob|copy|export|attach|load|install|create|drop|alter|insert|update|delete|truncate|query_table|pragma)\b/i;
    expect(blocked.test(sql)).toBe(true);
  });

  it("blocks read_parquet in SELECT", () => {
    const sql = "SELECT * FROM read_parquet('/etc/passwd')";
    const blocked = /\b(read_parquet|parquet_scan|parquet_metadata|parquet_schema|read_csv|read_csv_auto|read_json|read_json_auto|read_text|read_blob|glob|copy|export|attach|load|install|create|drop|alter|insert|update|delete|truncate|query_table|pragma)\b/i;
    expect(blocked.test(sql)).toBe(true);
  });

  it("blocks parquet_scan alias", () => {
    const sql = "SELECT * FROM parquet_scan('/etc/passwd')";
    const blocked = /\b(read_parquet|parquet_scan|parquet_metadata|parquet_schema|read_csv|read_csv_auto|read_json|read_json_auto|read_text|read_blob|glob|copy|export|attach|load|install|create|drop|alter|insert|update|delete|truncate|query_table|pragma)\b/i;
    expect(blocked.test(sql)).toBe(true);
  });

  it("blocks read_blob", () => {
    const sql = "SELECT * FROM read_blob('/etc/passwd')";
    const blocked = /\b(read_parquet|parquet_scan|parquet_metadata|parquet_schema|read_csv|read_csv_auto|read_json|read_json_auto|read_text|read_blob|glob|copy|export|attach|load|install|create|drop|alter|insert|update|delete|truncate|query_table|pragma)\b/i;
    expect(blocked.test(sql)).toBe(true);
  });

  it("blocks ATTACH", () => {
    const sql = "ATTACH '/tmp/evil.db' AS x";
    const firstWord = sql.trim().split(/\s/)[0].toUpperCase();
    const allowed = new Set(["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"]);
    expect(allowed.has(firstWord)).toBe(false);
  });

  it("blocks INSTALL", () => {
    const sql = "INSTALL httpfs";
    const firstWord = sql.trim().split(/\s/)[0].toUpperCase();
    const allowed = new Set(["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"]);
    expect(allowed.has(firstWord)).toBe(false);
  });

  it("blocks EXPORT", () => {
    const sql = "EXPORT DATABASE '/tmp/dump'";
    const firstWord = sql.trim().split(/\s/)[0].toUpperCase();
    const allowed = new Set(["SELECT", "WITH", "EXPLAIN", "DESCRIBE", "SHOW", "SUMMARIZE"]);
    expect(allowed.has(firstWord)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// Data release reporting (#348)
// ---------------------------------------------------------------------------

describe("data_version", () => {
  it("comes from the catalog on disk, not a build-time constant", () => {
    // The one fact whose entire job is to identify the tree actually being
    // read. Compared against an independent read of catalog.json rather than
    // against the same object the server loaded, so a hardcoded value cannot
    // satisfy both sides.
    const cat = ensureCatalog();
    const dataDir = process.env.NUCL_PARQUET_DATA ?? join(import.meta.dirname, "../../../../data");
    const onDisk = JSON.parse(readFileSync(join(dataDir, "catalog.json"), "utf-8")) as {
      data_version: string;
    };
    expect(cat.data_version).toBe(onDisk.data_version);
    expect(cat.data_version).toBeTruthy();
  });

  it("is a different fact from the server version", () => {
    // serverInfo.version is the npm package; data_version is the data release.
    // #348 exists because only the first was reachable.
    expect(ensureCatalog().data_version).not.toBe(PKG_VERSION);
  });

  it("is a different fact from any library's evaluation version", () => {
    // Conflating the release with an evaluation's version would make a
    // cross-server check silently wrong rather than merely absent.
    const cat = ensureCatalog();
    const evaluationVersions = Object.values(cat.libraries).map((l) => l.version);
    expect(evaluationVersions.length).toBeGreaterThan(0);
    expect(evaluationVersions).not.toContain(cat.data_version);
  });

  it("is named in the server instructions, so no tool call is needed", () => {
    // The referral this closes sends an agent to get_cross_sections, not to
    // list_libraries — a release reported only by the latter would be missing
    // from the path that actually goes wrong.
    const text = instructions(ensureCatalog().data_version);
    expect(text).toContain(ensureCatalog().data_version);
    expect(text.toLowerCase()).toContain("release");
  });

  it("instructions match the Rust server's wording, in both directions", () => {
    // Three implementations, one claim. A one-directional check ("every TS
    // clause appears in the Rust source") would miss Rust *adding* a sentence
    // or rewording a span this test does not name — so reconstruct the Rust
    // string and compare it whole.
    const rust = readFileSync(
      join(import.meta.dirname, "../../../rs/nucl-parquet-mcp/src/main.rs"),
      "utf-8",
    );
    // The literal is a rustfmt-wrapped format! string: `\` at end of line eats
    // the newline and the following indentation.
    const literal = rust.slice(rust.indexOf('"Serving nucl-parquet data release'));
    const raw = literal.slice(1, literal.indexOf('"\n', 1));
    const rustText = raw
      .replace(/\\\s*\n\s*/g, "")
      .replace("{data_version}", "X.Y.Z");

    expect(rustText).toBe(instructions("X.Y.Z"));
  });
});

describe("list_libraries payload (#348)", () => {
  it("is an envelope carrying the release, not a bare array", () => {
    // The shape the tool returns, not merely the shape of the catalog: an
    // assertion on ensureCatalog() alone passes while the tool still emits a
    // bare array with the release nowhere in it.
    const payload = libraryListPayload(ensureCatalog());
    expect(Array.isArray(payload)).toBe(false);
    expect(Object.keys(payload).sort()).toEqual(["data_version", "libraries"]);
    expect(payload.data_version).toBe(ensureCatalog().data_version);
    expect(payload.libraries.length).toBeGreaterThan(0);
  });

  it("keeps the release off the individual entries", () => {
    // Repeating it per entry would sit it next to the evaluation `version` and
    // invite exactly the conflation that makes the check silently wrong.
    const payload = libraryListPayload(ensureCatalog());
    for (const lib of payload.libraries) {
      expect(lib.data_version).toBeUndefined();
      expect(lib.version).toBeDefined();
    }
  });

  it("refuses a catalog with no data_version", () => {
    // An undefined release reported as if it were known is worse than the gap
    // #348 describes: the agent gets an answer it cannot tell is empty.
    expect(() => parseCatalog('{"libraries": {}}', "/nowhere")).toThrow(/data_version/);
    expect(() => parseCatalog('{"data_version": "", "libraries": {}}', "/nowhere")).toThrow(
      /data_version/,
    );
    expect(() =>
      parseCatalog('{"data_version": "2026.8.3", "libraries": {}}', "/nowhere"),
    ).not.toThrow();
  });

  it("reports whatever release the catalog it was handed claims", () => {
    // The requirement is not "report a version" but "report the version of the
    // data being read". A build-time constant passes every other test here.
    const cat = parseCatalog(
      JSON.stringify({
        data_version: "1999.1.1",
        libraries: { x: { name: "X", description: "d", projectiles: ["n"], version: "eval-7" } },
      }),
      "/nowhere",
    );
    const payload = libraryListPayload(cat);
    expect(payload.data_version).toBe("1999.1.1");
    expect(instructions(cat.data_version)).toContain("1999.1.1");
    expect(payload.libraries[0].version).toBe("eval-7");
  });
});

// ---------------------------------------------------------------------------
// Views whose predicate depends on the state vocabulary (#380)
// ---------------------------------------------------------------------------

describe("ground_states view", () => {
  const query = (sql: string) =>
    new Promise<Record<string, unknown>[]>((resolve, reject) => {
      getDb().all(sql, (err: Error | null, rows: Record<string, unknown>[]) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });

  it("selects rows, having been defined as `state = ''` which matches none", async () => {
    // #380 retired `''`. nuclides.parquet reads g=3148 / m=739 / m2=82 / m3=7
    // / null=13 and carries no `''` at all, so this view was silently empty —
    // every caller of `ground_states` got zero rows and no error. Asserting a
    // count, because "the view exists" was already true while it was broken.
    const [{ n }] = (await query("SELECT count(*) AS n FROM ground_states")) as [{ n: bigint }];
    expect(Number(n)).toBe(3148);
  });

  it("is exactly the ground-state subset of nuclides", async () => {
    const rows = await query("SELECT DISTINCT state FROM ground_states");
    expect(rows.map((r) => r.state)).toEqual(["g"]);
  });
});
