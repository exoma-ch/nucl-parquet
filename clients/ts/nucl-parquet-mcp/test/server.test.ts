import { describe, it, expect } from "vitest";
import { getCatalog } from "../src/index.js";

describe("catalog", () => {
  const catalog = getCatalog();

  it("has all expected libraries", () => {
    const ids = Object.keys(catalog.libraries);
    expect(ids.length).toBeGreaterThanOrEqual(15);
    expect(ids).toContain("tendl-2024");
    expect(ids).toContain("endfb-8.1");
    expect(ids).toContain("exfor");
  });

  it("all libraries have non-empty projectiles", () => {
    for (const [id, lib] of Object.entries(catalog.libraries)) {
      expect(lib.projectiles.length, `${id} has no projectiles`).toBeGreaterThan(0);
    }
  });

  it("all libraries have a path ending with /", () => {
    for (const [id, lib] of Object.entries(catalog.libraries)) {
      expect(lib.path, `${id} path should end with /`).toMatch(/\/$/);
    }
  });

  it("shared meta has required files", () => {
    expect(catalog.shared.meta.files).toHaveProperty("abundances");
    expect(catalog.shared.meta.files).toHaveProperty("decay");
    expect(catalog.shared.meta.files).toHaveProperty("elements");
  });

  it("shared stopping has sources", () => {
    expect(catalog.shared.stopping.sources).toContain("PSTAR");
    expect(catalog.shared.stopping.sources).toContain("ASTAR");
  });
});

describe("URL construction", () => {
  const catalog = getCatalog();

  it("builds correct cross-section path", () => {
    const lib = catalog.libraries["tendl-2024"];
    const path = `${lib.path}p_Cu.parquet`;
    expect(path).toBe("tendl-2024/xs/p_Cu.parquet");
  });

  it("builds correct meta path", () => {
    const path = catalog.shared.meta.path + catalog.shared.meta.files.decay;
    expect(path).toBe("meta/decay.parquet");
  });

  it("builds correct stopping path", () => {
    const path = catalog.shared.stopping.path + catalog.shared.stopping.files.stopping;
    expect(path).toBe("stopping/stopping.parquet");
  });

  it("builds correct manifest path from library path", () => {
    const lib = catalog.libraries["tendl-2024"];
    const manifestPath = lib.path.replace(/xs\/$/, "manifest.json");
    expect(manifestPath).toBe("tendl-2024/manifest.json");
  });
});

describe("fetchParquetRows (integration)", () => {
  // These tests require network access and hit raw.githubusercontent.com.
  // Run with: npm test -- --reporter=verbose

  it("fetches and parses abundances.parquet", async () => {
    const { fetchParquetRows } = await import("../src/index.js");
    const rows = await fetchParquetRows("meta/abundances.parquet");
    expect(rows.length).toBeGreaterThan(0);

    // Cu-63 should exist
    const cu63 = rows.find((r) => r["Z"] === 29 && r["A"] === 63);
    expect(cu63).toBeDefined();
    const abundance = cu63!["abundance"] as number;
    expect(abundance).toBeGreaterThan(0.5);
    expect(abundance).toBeLessThan(0.8);
  }, 30_000);

  it("fetches cross-section data", async () => {
    const { fetchParquetRows } = await import("../src/index.js");
    const rows = await fetchParquetRows("fendl-3.2/xs/n_Cu.parquet");
    expect(rows.length).toBeGreaterThan(0);
    expect(rows[0]).toHaveProperty("energy_MeV");
    expect(rows[0]).toHaveProperty("xs_mb");
  }, 30_000);

  it("caches results on second call", async () => {
    const { fetchParquetRows } = await import("../src/index.js");
    const t0 = Date.now();
    await fetchParquetRows("meta/elements.parquet");
    const t1 = Date.now();
    await fetchParquetRows("meta/elements.parquet");
    const t2 = Date.now();

    // Second call should be near-instant (cached)
    expect(t2 - t1).toBeLessThan(t1 - t0 + 10);
  }, 30_000);
});
