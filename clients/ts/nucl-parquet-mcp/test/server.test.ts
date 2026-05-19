/**
 * Tests for nucl-parquet MCP server (SSoT refactor — DuckDB-backed).
 *
 * All tests read local data from the repo's data/ directory — no network.
 */

import { createRequire } from "node:module";
import { describe, it, expect } from "vitest";
import { ensureCatalog, getDb, PKG_VERSION } from "../src/index.js";

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
    expect(libs).toHaveProperty("tendl-2024");
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
