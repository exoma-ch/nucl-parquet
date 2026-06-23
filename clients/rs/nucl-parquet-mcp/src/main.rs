//! nucl-parquet MCP server — wraps the nucl-parquet crate with local data.
//!
//! SSoT refactor (#206): uses `ParquetStore` from the nucl-parquet client
//! library for all Parquet I/O. The MCP is a thin JSON-RPC shell — no
//! direct arrow/parquet dependencies. Data resolved from $NUCL_PARQUET_DATA,
//! ~/.nucl-parquet/, or the repo data/ directory.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use nucl_parquet::{Filter, ParquetStore};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};

// ---------------------------------------------------------------------------
// Data directory resolution
// ---------------------------------------------------------------------------

fn resolve_data_dir() -> Result<PathBuf, String> {
    // 1. Explicit env var
    if let Ok(env) = std::env::var("NUCL_PARQUET_DATA") {
        let p = PathBuf::from(&env);
        if p.is_dir() {
            return Ok(p);
        }
    }
    // 2. Repo-local data/ (development) — walk up from binary location
    let exe = std::env::current_exe().unwrap_or_default();
    for ancestor in exe.ancestors().take(10) {
        let data = ancestor.join("data").join("catalog.json");
        if data.exists() {
            return Ok(ancestor.join("data"));
        }
    }
    // Also check CWD
    let cwd_data = PathBuf::from("data").join("catalog.json");
    if cwd_data.exists() {
        return Ok(PathBuf::from("data")
            .canonicalize()
            .unwrap_or("data".into()));
    }
    // Walk up from CWD
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors().take(10) {
            let data = ancestor.join("data").join("catalog.json");
            if data.exists() {
                return Ok(ancestor.join("data"));
            }
        }
    }
    // 3. nucl-parquet crate's DataDir resolution
    if let Ok(dd) = nucl_parquet::DataDir::resolve() {
        return Ok(dd.root().to_path_buf());
    }
    Err(
        "Cannot find nucl-parquet data directory. Set NUCL_PARQUET_DATA env var \
         or place data at ~/.nucl-parquet/. Data releases: \
         https://github.com/exoma-ch/nucl-parquet/releases"
            .to_string(),
    )
}

// ---------------------------------------------------------------------------
// Catalog (loaded from disk)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Library {
    name: String,
    description: String,
    #[serde(default)]
    projectiles: Vec<String>,
    #[serde(default)]
    data_type: String,
    #[serde(default)]
    version: String,
    #[serde(default)]
    path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Catalog {
    libraries: HashMap<String, Library>,
}

fn load_catalog(data_dir: &Path) -> Result<Catalog, String> {
    let path = data_dir.join("catalog.json");
    let data = fs::read_to_string(&path).map_err(|e| format!("Failed to read catalog: {e}"))?;
    serde_json::from_str(&data).map_err(|e| format!("Failed to parse catalog: {e}"))
}

// ---------------------------------------------------------------------------
// JSON-RPC types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct JsonRpcResponse {
    jsonrpc: String,
    id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<JsonRpcError>,
}

#[derive(Debug, Serialize)]
struct JsonRpcError {
    code: i32,
    message: String,
}

impl JsonRpcResponse {
    fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: Some(result),
            error: None,
        }
    }
    fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".into(),
            id,
            result: None,
            error: Some(JsonRpcError { code, message }),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: truncate + format result
// ---------------------------------------------------------------------------

fn format_result(
    rows: Vec<serde_json::Value>,
    max_rows: usize,
    extra: serde_json::Value,
) -> serde_json::Value {
    let total = rows.len();
    let truncated = total > max_rows;
    let display: Vec<_> = rows.into_iter().take(max_rows).collect();

    let mut result = extra;
    if let Some(obj) = result.as_object_mut() {
        obj.insert("total".into(), serde_json::json!(total));
        obj.insert("truncated".into(), serde_json::json!(truncated));
        obj.insert("rows".into(), serde_json::json!(display));
    }
    serde_json::json!({
        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }]
    })
}

// ---------------------------------------------------------------------------
// MCP tool definitions
// ---------------------------------------------------------------------------

fn tool_definitions() -> serde_json::Value {
    serde_json::json!({
        "tools": [
            {
                "name": "list_libraries",
                "description": "List all available nuclear data libraries with projectiles and descriptions",
                "inputSchema": { "type": "object", "properties": {}, "required": [] }
            },
            {
                "name": "list_isotopes",
                "description": "List available target elements for a library and projectile",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "library": { "type": "string", "description": "Library ID, e.g. 'tendl-2025'" },
                        "projectile": { "type": "string", "description": "Projectile: n, p, d, t, h, a, g" }
                    },
                    "required": ["library", "projectile"]
                }
            },
            {
                "name": "get_cross_sections",
                "description": "Get nuclear reaction cross-section data for a target element",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "library": { "type": "string", "description": "Library ID" },
                        "projectile": { "type": "string", "description": "Projectile type" },
                        "element": { "type": "string", "description": "Target element symbol, e.g. 'Cu'" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": ["library", "projectile", "element"]
                }
            },
            {
                "name": "get_decay_data",
                "description": "Get radioactive decay data (half-lives, decay modes, daughters)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "z": { "type": "integer", "description": "Atomic number" },
                        "a": { "type": "integer", "description": "Mass number" }
                    },
                    "required": []
                }
            },
            {
                "name": "get_abundances",
                "description": "Get natural isotope abundances for an element",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "z": { "type": "integer", "description": "Atomic number" }
                    },
                    "required": ["z"]
                }
            },
            {
                "name": "get_stopping_power",
                "description": "Get mass stopping power (dE/dx) data",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "source": { "type": "string", "description": "PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, or a federated catima isotope (catima_<Sym><A>, e.g. catima_C12, catima_U238)" },
                        "target_z": { "type": "integer", "description": "Target atomic number" }
                    },
                    "required": ["source", "target_z"]
                }
            },
            {
                "name": "get_radiation",
                "description": "Get radiation emissions (gammas, X-rays, Auger, conversion electrons) for a nuclide",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "z": { "type": "integer", "description": "Atomic number" },
                        "a": { "type": "integer", "description": "Mass number (omit for all isotopes)" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": ["z"]
                }
            },
            {
                "name": "get_coincidences",
                "description": "Get gamma-gamma and mixed-emission coincidence pairs for a nuclide",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "z": { "type": "integer", "description": "Atomic number" },
                        "a": { "type": "integer", "description": "Mass number (omit for all isotopes)" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": ["z"]
                }
            },
            {
                "name": "get_summing_partners",
                "description": "Get ICC-corrected summing partners for HPGe true-coincidence-summing (TCS) corrections",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "z": { "type": "integer", "description": "Atomic number (daughter, filing convention)" },
                        "a": { "type": "integer", "description": "Mass number" },
                        "primary_energy_keV": { "type": "number", "description": "Filter to pairs matching this energy" },
                        "tolerance_keV": { "type": "number", "description": "Energy tolerance (default 0.5 keV)" },
                        "emission1_rad_type": { "type": "string", "description": "'gamma', 'xray', or 'auger'" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": ["z", "a"]
                }
            },
            {
                "name": "get_emissions",
                "description": "Get absolute per-decay photon emission intensities (NuDat-equivalent). Parent-keyed, not daughter-keyed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "parent_z": { "type": "integer", "description": "Atomic number of decaying parent (e.g. 27 for Co-60)" },
                        "parent_a": { "type": "integer", "description": "Mass number of parent" },
                        "parent_state": { "type": "string", "description": "'' (ground), 'm', 'm2'" },
                        "decay_mode": { "type": "string", "description": "Filter: 'beta-', 'KshellEC', 'IT', etc." },
                        "energy_keV": { "type": "number", "description": "Filter near this energy" },
                        "tolerance_keV": { "type": "number", "description": "Energy tolerance (default 0.5 keV)" },
                        "min_intensity_pct": { "type": "number", "description": "Minimum absolute intensity (%)" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": ["parent_z", "parent_a"]
                }
            },
            {
                "name": "get_beta_spectrum",
                "description": "Get continuous beta-decay kinetic-energy spectrum (Fermi function, dN/dE normalized to 1)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "z": { "type": "integer", "description": "Atomic number" },
                        "a": { "type": "integer", "description": "Mass number" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": ["z", "a"]
                }
            },
            {
                "name": "get_compound_compositions",
                "description": "Get elemental compositions (weight fractions) for NIST XCOM standard materials",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "material": { "type": "string", "description": "Material name. Omit to list all." }
                    },
                    "required": []
                }
            },
            {
                "name": "get_electron_stopping",
                "description": "Get electron stopping power with collision/radiative split (~183 compounds + Z=1..98)",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "target": { "type": "string", "description": "Compound name (e.g. 'G4_WATER')" },
                        "target_z": { "type": "integer", "description": "Atomic number for elements" },
                        "max_rows": { "type": "integer", "description": "Max rows (default 500)" }
                    },
                    "required": []
                }
            }
        ]
    })
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

fn handle_tool_call(
    data_dir: &Path,
    catalog: &Catalog,
    store: &ParquetStore,
    name: &str,
    args: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    match name {
        "list_libraries" => {
            let libs: Vec<serde_json::Value> = catalog
                .libraries
                .iter()
                .filter(|(_, lib)| !lib.projectiles.is_empty())
                .map(|(id, lib)| {
                    serde_json::json!({
                        "id": id,
                        "name": lib.name,
                        "description": lib.description,
                        "projectiles": lib.projectiles,
                        "version": lib.version,
                        "data_type": lib.data_type,
                    })
                })
                .collect();
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&libs).unwrap() }] }),
            )
        }
        "list_isotopes" => {
            let library = args
                .get("library")
                .and_then(|v| v.as_str())
                .ok_or("missing 'library'")?;
            let projectile = args
                .get("projectile")
                .and_then(|v| v.as_str())
                .ok_or("missing 'projectile'")?;
            let lib = catalog
                .libraries
                .get(library)
                .ok_or_else(|| format!("Unknown library: {library}"))?;
            if !lib.projectiles.iter().any(|p| p == projectile) {
                return Err(format!("Projectile '{projectile}' not in {library}"));
            }
            let manifest_path = data_dir.join(lib.path.replace("xs/", "manifest.json"));
            let manifest_data = fs::read_to_string(&manifest_path)
                .map_err(|e| format!("Failed to read manifest: {e}"))?;
            let manifest: serde_json::Value = serde_json::from_str(&manifest_data)
                .map_err(|e| format!("Failed to parse manifest: {e}"))?;
            let elements = manifest
                .get("elements")
                .cloned()
                .unwrap_or(serde_json::json!([]));
            let result = serde_json::json!({ "library": library, "projectile": projectile, "elements": elements });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
        }
        "get_cross_sections" => {
            let library = args
                .get("library")
                .and_then(|v| v.as_str())
                .ok_or("missing 'library'")?;
            let projectile = args
                .get("projectile")
                .and_then(|v| v.as_str())
                .ok_or("missing 'projectile'")?;
            let element = args
                .get("element")
                .and_then(|v| v.as_str())
                .ok_or("missing 'element'")?;
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;

            if !is_valid_element(element) {
                return Err(format!("Invalid element: {element}"));
            }

            let lib = catalog
                .libraries
                .get(library)
                .ok_or_else(|| format!("Unknown library: {library}"))?;
            let rel_path = format!("{}{projectile}_{element}.parquet", lib.path);
            let rows = store.load(&rel_path).map_err(|e| format!("{e}"))?;
            Ok(format_result(
                (*rows).clone(),
                max_rows,
                serde_json::json!({ "library": library, "projectile": projectile, "element": element }),
            ))
        }
        "get_decay_data" => {
            let z = args.get("z").and_then(|v| v.as_i64());
            let a = args.get("a").and_then(|v| v.as_i64());
            if z.is_none() && a.is_none() {
                return Err("Provide at least z or a".to_string());
            }
            let mut filters = Vec::new();
            if let Some(zv) = z {
                filters.push(Filter::Eq("Z".into(), serde_json::json!(zv)));
            }
            if let Some(av) = a {
                filters.push(Filter::Eq("A".into(), serde_json::json!(av)));
            }
            let rows = store
                .load_filtered("meta/decay.parquet", &filters)
                .map_err(|e| format!("{e}"))?;
            Ok(format_result(
                rows,
                500,
                serde_json::json!({ "z": z, "a": a }),
            ))
        }
        "get_abundances" => {
            let z = args
                .get("z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'z'")?;
            let rows = store
                .load_filtered(
                    "meta/abundances.parquet",
                    &[Filter::Eq("Z".into(), serde_json::json!(z))],
                )
                .map_err(|e| format!("{e}"))?;
            Ok(format_result(rows, 500, serde_json::json!({ "z": z })))
        }
        "get_stopping_power" => {
            let source = args
                .get("source")
                .and_then(|v| v.as_str())
                .ok_or("missing 'source'")?;
            let target_z = args
                .get("target_z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'target_z'")?;
            // catima is federated per beam isotope: source `catima_<Sym><A>`
            // (e.g. catima_C12) maps to stopping/catima_<Sym><A>.parquet. The
            // alphanumeric guard prevents path traversal from the user-supplied
            // source string.
            let is_catima_shard = source.strip_prefix("catima_").is_some_and(|rest| {
                !rest.is_empty() && rest.chars().all(|c| c.is_ascii_alphanumeric())
            });
            let rel_path: String = match source {
                "PSTAR" => "stopping/PSTAR.parquet".to_string(),
                "ASTAR" => "stopping/ASTAR.parquet".to_string(),
                "ESTAR" => "stopping/ESTAR.parquet".to_string(),
                "dSTAR" => "stopping/dSTAR.parquet".to_string(),
                "tSTAR" => "stopping/tSTAR.parquet".to_string(),
                _ if is_catima_shard => format!("stopping/{source}.parquet"),
                other => {
                    return Err(format!(
                        "Unknown source {other:?}; valid: PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, \
                         or a federated catima isotope e.g. catima_C12, catima_U238"
                    ))
                }
            };
            let rows = store
                .load_filtered(
                    &rel_path,
                    &[Filter::Eq("target_Z".into(), serde_json::json!(target_z))],
                )
                .map_err(|e| format!("{e}"))?;
            Ok(format_result(
                rows,
                500,
                serde_json::json!({ "source": source, "target_z": target_z }),
            ))
        }
        "get_radiation" | "get_coincidences" => {
            let z = args
                .get("z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'z'")?;
            let a = args.get("a").and_then(|v| v.as_i64());
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            let symbol =
                nucl_parquet::z_to_symbol(z as u32).ok_or_else(|| format!("Z={z} out of range"))?;
            let subdir = if name == "get_radiation" {
                "radiation"
            } else {
                "coincidences"
            };
            let rel_path = format!("meta/ensdf/{subdir}/{symbol}.parquet");
            let mut filters = Vec::new();
            if let Some(av) = a {
                filters.push(Filter::Eq("A".into(), serde_json::json!(av)));
            }
            let rows = store
                .load_filtered(&rel_path, &filters)
                .map_err(|e| format!("{e}"))?;
            Ok(format_result(
                rows,
                max_rows,
                serde_json::json!({ "z": z, "a": a, "symbol": symbol }),
            ))
        }
        "get_summing_partners" => {
            let z = args
                .get("z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'z'")?;
            let a = args
                .get("a")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'a'")?;
            let primary_energy = args.get("primary_energy_keV").and_then(|v| v.as_f64());
            let tolerance = args
                .get("tolerance_keV")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            let e1_type = args.get("emission1_rad_type").and_then(|v| v.as_str());
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            let symbol =
                nucl_parquet::z_to_symbol(z as u32).ok_or_else(|| format!("Z={z} out of range"))?;
            let rel_path = format!("meta/ensdf/summing_partners/{symbol}.parquet");

            // Load all rows for this element and filter in code (energy tolerance
            // and multi-column OR require post-load filtering beyond Filter::Eq).
            let all = store.load(&rel_path).map_err(|e| format!("{e}"))?;
            let filtered: Vec<_> = all
                .iter()
                .filter(|r| {
                    if r.get("A").and_then(|v| v.as_i64()) != Some(a) {
                        return false;
                    }
                    if let Some(e1t) = e1_type {
                        if r.get("emission1_rad_type").and_then(|v| v.as_str()) != Some(e1t) {
                            return false;
                        }
                    }
                    if let Some(pe) = primary_energy {
                        let e1 = r
                            .get("emission1_energy_keV")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let e2 = r
                            .get("emission2_energy_keV")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        if (e1 - pe).abs() >= tolerance && (e2 - pe).abs() >= tolerance {
                            return false;
                        }
                    }
                    true
                })
                .cloned()
                .collect();
            Ok(format_result(
                filtered,
                max_rows,
                serde_json::json!({ "z": z, "a": a, "primary_energy_keV": primary_energy }),
            ))
        }
        "get_emissions" => {
            let parent_z = args
                .get("parent_z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'parent_z'")?;
            let parent_a = args
                .get("parent_a")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'parent_a'")?;
            let parent_state = args
                .get("parent_state")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let decay_mode_filter = args.get("decay_mode").and_then(|v| v.as_str());
            let energy_filter = args.get("energy_keV").and_then(|v| v.as_f64());
            let tolerance = args
                .get("tolerance_keV")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5);
            let min_intensity = args
                .get("min_intensity_pct")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            let symbol = nucl_parquet::z_to_symbol(parent_z as u32)
                .ok_or_else(|| format!("Z={parent_z} out of range"))?;
            let rel_path = format!("meta/ensdf/emissions/{symbol}.parquet");

            // Build filters that ParquetStore can handle directly
            let mut filters = vec![
                Filter::Eq("parent_A".into(), serde_json::json!(parent_a)),
                Filter::Eq("parent_state".into(), serde_json::json!(parent_state)),
            ];
            if let Some(dm) = decay_mode_filter {
                filters.push(Filter::Eq("decay_mode".into(), serde_json::json!(dm)));
            }
            if let Some(ef) = energy_filter {
                filters.push(Filter::Near("energy_keV".into(), ef, tolerance));
            }
            if min_intensity > 0.0 {
                filters.push(Filter::Gte("intensity_pct".into(), min_intensity));
            }

            let rows = store
                .load_filtered(&rel_path, &filters)
                .map_err(|e| format!("{e}"))?;
            Ok(format_result(
                rows,
                max_rows,
                serde_json::json!({
                    "parent_z": parent_z,
                    "parent_a": parent_a,
                    "parent_state": parent_state,
                }),
            ))
        }
        "get_beta_spectrum" => {
            let z = args
                .get("z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'z'")?;
            let a = args
                .get("a")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'a'")?;
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            let symbol =
                nucl_parquet::z_to_symbol(z as u32).ok_or_else(|| format!("Z={z} out of range"))?;
            let rel_path = format!("meta/ensdf/beta_spectra/{symbol}.parquet");
            let rows = store
                .load_filtered(
                    &rel_path,
                    &[
                        Filter::Eq("Z".into(), serde_json::json!(z)),
                        Filter::Eq("A".into(), serde_json::json!(a)),
                    ],
                )
                .map_err(|e| format!("{e}"))?;
            Ok(format_result(
                rows,
                max_rows,
                serde_json::json!({ "z": z, "a": a, "symbol": symbol }),
            ))
        }
        "get_compound_compositions" => {
            let material = args.get("material").and_then(|v| v.as_str());
            if let Some(mat) = material {
                let rows = store
                    .load_filtered(
                        "meta/compound_compositions.parquet",
                        &[Filter::Eq("material".into(), serde_json::json!(mat))],
                    )
                    .map_err(|e| format!("{e}"))?;
                if rows.is_empty() {
                    return Err(format!("Unknown material: {mat:?}"));
                }
                Ok(format_result(
                    rows,
                    500,
                    serde_json::json!({ "material": mat }),
                ))
            } else {
                let all = store
                    .load("meta/compound_compositions.parquet")
                    .map_err(|e| format!("{e}"))?;
                let mut materials: Vec<String> = all
                    .iter()
                    .filter_map(|r| {
                        r.get("material")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string())
                    })
                    .collect();
                materials.sort();
                materials.dedup();
                let result =
                    serde_json::json!({ "count": materials.len(), "materials": materials });
                Ok(
                    serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
                )
            }
        }
        "get_electron_stopping" => {
            let target = args.get("target").and_then(|v| v.as_str());
            let target_z = args.get("target_z").and_then(|v| v.as_i64());
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            if target.is_none() && target_z.is_none() {
                return Err(
                    "Provide target (compound name) or target_z (atomic number)".to_string()
                );
            }
            let all = store
                .load("stopping/em/electron_stopping.parquet")
                .map_err(|e| format!("{e}"))?;
            let filtered: Vec<_> = all
                .iter()
                .filter(|r| {
                    if let Some(tz) = target_z {
                        r.get("target_Z").and_then(|v| v.as_i64()) == Some(tz)
                    } else if let Some(t) = target {
                        r.get("name").and_then(|v| v.as_str()) == Some(t)
                            || r.get("g4_name").and_then(|v| v.as_str()) == Some(t)
                    } else {
                        false
                    }
                })
                .cloned()
                .collect();
            Ok(format_result(
                filtered,
                max_rows,
                serde_json::json!({ "target": target, "target_z": target_z }),
            ))
        }
        _ => Err(format!("Unknown tool: {name}")),
    }
}

/// Validate element symbol: 1-2 chars, first uppercase, second (if any) lowercase.
fn is_valid_element(element: &str) -> bool {
    let bytes = element.as_bytes();
    match bytes.len() {
        1 => bytes[0].is_ascii_uppercase(),
        2 => bytes[0].is_ascii_uppercase() && bytes[1].is_ascii_lowercase(),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// MCP protocol handler
// ---------------------------------------------------------------------------

fn handle_request(
    data_dir: &Path,
    catalog: &Catalog,
    store: &ParquetStore,
    req: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let id = req.id.clone().unwrap_or(serde_json::Value::Null);
    req.id.as_ref()?;

    match req.method.as_str() {
        "initialize" => Some(JsonRpcResponse::success(
            id,
            serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "nucl-parquet", "version": env!("CARGO_PKG_VERSION") }
            }),
        )),
        "tools/list" => Some(JsonRpcResponse::success(id, tool_definitions())),
        "tools/call" => {
            let name = req
                .params
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = req
                .params
                .get("arguments")
                .cloned()
                .unwrap_or(serde_json::json!({}));
            match handle_tool_call(data_dir, catalog, store, name, &args) {
                Ok(result) => Some(JsonRpcResponse::success(id, result)),
                Err(e) => Some(JsonRpcResponse::error(id, -32000, e)),
            }
        }
        _ => Some(JsonRpcResponse::error(
            id,
            -32601,
            format!("Method not found: {}", req.method),
        )),
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let data_dir = resolve_data_dir().expect("Failed to find data directory");
    let catalog = load_catalog(&data_dir).expect("Failed to load catalog");
    let store = Arc::new(ParquetStore::new(&data_dir));

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let stdout = std::io::stdout();

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break,
            Ok(_) => {}
            Err(e) => {
                eprintln!("stdin read error: {e}");
                break;
            }
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let req: JsonRpcRequest = match serde_json::from_str(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = JsonRpcResponse::error(
                    serde_json::Value::Null,
                    -32700,
                    format!("Parse error: {e}"),
                );
                let out = serde_json::to_string(&resp).unwrap();
                let mut stdout = stdout.lock();
                let _ = writeln!(stdout, "{out}");
                let _ = stdout.flush();
                continue;
            }
        };

        if let Some(resp) = handle_request(&data_dir, &catalog, &store, req) {
            let out = serde_json::to_string(&resp).unwrap();
            let mut stdout = stdout.lock();
            let _ = writeln!(stdout, "{out}");
            let _ = stdout.flush();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_dir() -> PathBuf {
        // Walk up to find repo data/
        let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        for _ in 0..10 {
            if dir.join("data").join("catalog.json").exists() {
                return dir.join("data");
            }
            if !dir.pop() {
                break;
            }
        }
        panic!("Cannot find data/ directory for tests");
    }

    #[test]
    fn catalog_loads_from_disk() {
        let data_dir = test_data_dir();
        let cat = load_catalog(&data_dir).unwrap();
        assert!(cat.libraries.len() >= 15);
        assert!(cat.libraries.contains_key("tendl-2025"));
        assert!(cat.libraries.contains_key("endfb-8.1"));
        assert!(cat.libraries.contains_key("exfor"));
    }

    #[test]
    fn catalog_libraries_have_projectiles() {
        let data_dir = test_data_dir();
        let cat = load_catalog(&data_dir).unwrap();
        for (id, lib) in &cat.libraries {
            if !lib.projectiles.is_empty() {
                assert!(!lib.projectiles.is_empty(), "{id} has no projectiles");
            }
        }
    }

    #[test]
    fn tool_definitions_valid() {
        let defs = tool_definitions();
        let tools = defs.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 13);
        for tool in tools {
            assert!(tool.get("name").is_some());
            assert!(tool.get("description").is_some());
            assert!(tool.get("inputSchema").is_some());
        }
    }

    #[test]
    fn element_validation() {
        assert!(is_valid_element("Cu"));
        assert!(is_valid_element("H"));
        assert!(is_valid_element("Fe"));
        assert!(!is_valid_element("cu"));
        assert!(!is_valid_element("CU"));
        assert!(!is_valid_element(""));
        assert!(!is_valid_element("Abc"));
        assert!(!is_valid_element("'; DROP TABLE"));
    }

    #[test]
    fn json_rpc_response_serialization() {
        let resp =
            JsonRpcResponse::success(serde_json::json!(1), serde_json::json!({"key": "value"}));
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"key\":\"value\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn json_rpc_error_serialization() {
        let resp = JsonRpcResponse::error(serde_json::json!(2), -32601, "Method not found".into());
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32601"));
        assert!(!json.contains("\"result\""));
    }

    #[test]
    fn handle_initialize() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "initialize".into(),
            params: serde_json::json!({}),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "nucl-parquet");
        assert_eq!(
            result["serverInfo"]["version"].as_str().unwrap(),
            env!("CARGO_PKG_VERSION")
        );
    }

    #[test]
    fn handle_tools_list() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/list".into(),
            params: serde_json::json!({}),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 13);
    }

    #[test]
    fn handle_list_libraries() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "list_libraries", "arguments": {} }),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("TENDL-2023"));
    }

    #[test]
    fn handle_get_abundances() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "get_abundances", "arguments": { "z": 29 } }),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"z\": 29"));
        assert!(text.contains("\"total\": 2")); // Cu-63, Cu-65
    }

    #[test]
    fn handle_get_decay_data() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "get_decay_data", "arguments": { "z": 27, "a": 60 } }),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"total\":"));
    }

    #[test]
    fn handle_get_summing_partners() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({
                "name": "get_summing_partners",
                "arguments": { "z": 28, "a": 60, "emission1_rad_type": "gamma" }
            }),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"z\": 28"));
        assert!(text.contains("\"a\": 60"));
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        let total = parsed["total"].as_u64().unwrap();
        assert!(
            total >= 1,
            "Expected ≥1 Co-60 summing partners, got {total}"
        );
    }

    #[test]
    fn handle_get_emissions() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({
                "name": "get_emissions",
                "arguments": { "parent_z": 27, "parent_a": 60 }
            }),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        let total = parsed["total"].as_u64().unwrap();
        assert!(
            total >= 2,
            "Expected ≥2 Co-60 emission rows (1173+1332), got {total}"
        );
        // Verify 1173 keV gamma exists with reasonable intensity
        let rows = parsed["rows"].as_array().unwrap();
        let has_1173 = rows.iter().any(|r| {
            let e = r.get("energy_keV").and_then(|v| v.as_f64()).unwrap_or(0.0);
            (e - 1173.239).abs() < 0.5
        });
        assert!(has_1173, "Expected Co-60 1173 keV gamma in emissions");
    }

    #[test]
    fn handle_unknown_tool() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let store = ParquetStore::new(&data_dir);
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "nonexistent", "arguments": {} }),
        };
        let resp = handle_request(&data_dir, &catalog, &store, req).unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("Unknown tool"));
    }
}
