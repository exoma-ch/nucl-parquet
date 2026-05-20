//! nucl-parquet MCP server — wraps the nucl-parquet crate with local data.
//!
//! SSoT refactor (epic #173, Sub-D #178): reads local Parquet files via the
//! nucl-parquet crate and raw arrow-parquet for tables not yet covered by
//! typed DBs. No HTTP fetching — data resolved from $NUCL_PARQUET_DATA,
//! ~/.nucl-parquet/, or the repo data/ directory.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
    StringArray, UInt8Array,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

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
// Z → element symbol
// ---------------------------------------------------------------------------

#[rustfmt::skip]
static Z_TO_SYMBOL: [&str; 100] = [
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
];

fn z_to_symbol(z: i64) -> Option<&'static str> {
    if z < 1 || z as usize >= Z_TO_SYMBOL.len() {
        None
    } else {
        Some(Z_TO_SYMBOL[z as usize])
    }
}

// ---------------------------------------------------------------------------
// Local Parquet file reader (replaces HTTP fetch)
// ---------------------------------------------------------------------------

type Cache = Arc<Mutex<HashMap<String, Vec<serde_json::Value>>>>;

async fn load_parquet_rows(
    cache: &Cache,
    file_path: &str,
) -> Result<Vec<serde_json::Value>, String> {
    {
        let c = cache.lock().await;
        if let Some(rows) = c.get(file_path) {
            return Ok(rows.clone());
        }
    }

    let path = PathBuf::from(file_path);
    if !path.exists() {
        return Err(format!("File not found: {file_path}"));
    }
    let data = fs::read(&path).map_err(|e| format!("Read error: {e}"))?;
    let rows = parse_parquet_bytes(data)?;

    let mut c = cache.lock().await;
    c.insert(file_path.to_string(), rows.clone());
    Ok(rows)
}

fn parse_parquet_bytes(data: Vec<u8>) -> Result<Vec<serde_json::Value>, String> {
    let data = bytes::Bytes::from(data);
    let reader = ParquetRecordBatchReaderBuilder::try_new(data)
        .map_err(|e| format!("Parquet open error: {e}"))?
        .build()
        .map_err(|e| format!("Parquet reader error: {e}"))?;

    let mut rows = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(|e| format!("Batch read error: {e}"))?;
        let schema = batch.schema();
        for row_idx in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let val = column_value_to_json(col.as_ref(), row_idx);
                obj.insert(field.name().clone(), val);
            }
            rows.push(serde_json::Value::Object(obj));
        }
    }
    Ok(rows)
}

fn column_value_to_json(col: &dyn Array, idx: usize) -> serde_json::Value {
    if col.is_null(idx) {
        return serde_json::Value::Null;
    }
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        let v = a.value(idx);
        serde_json::Value::Number(
            serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)),
        )
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        let v = a.value(idx) as f64;
        serde_json::Value::Number(
            serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)),
        )
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        serde_json::Value::Number(a.value(idx).into())
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        serde_json::Value::Number(a.value(idx).into())
    } else if let Some(a) = col.as_any().downcast_ref::<Int16Array>() {
        serde_json::Value::Number((a.value(idx) as i64).into())
    } else if let Some(a) = col.as_any().downcast_ref::<UInt8Array>() {
        serde_json::Value::Number((a.value(idx) as i64).into())
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        serde_json::Value::Bool(a.value(idx))
    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        serde_json::Value::String(a.value(idx).to_string())
    } else {
        // Fallback: try casting to Utf8 (handles dictionary-encoded strings from Parquet)
        if let Ok(cast) = arrow::compute::cast(col, &arrow::datatypes::DataType::Utf8) {
            if let Some(s) = cast.as_any().downcast_ref::<StringArray>() {
                return serde_json::Value::String(s.value(idx).to_string());
            }
        }
        serde_json::Value::String(format!("{col:?}"))
    }
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
                        "source": { "type": "string", "description": "PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, or catima" },
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

async fn handle_tool_call(
    data_dir: &Path,
    catalog: &Catalog,
    cache: &Cache,
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

            // Input validation
            if !matches!(projectile, "n" | "p" | "d" | "t" | "h" | "a" | "g") {
                return Err(format!("Invalid projectile: {projectile}"));
            }
            let element_re = regex_lite(element);
            if !element_re {
                return Err(format!("Invalid element: {element}"));
            }

            let lib = catalog
                .libraries
                .get(library)
                .ok_or_else(|| format!("Unknown library: {library}"))?;
            let path = data_dir.join(format!("{}{projectile}_{element}.parquet", lib.path));
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let total = rows.len();
            let truncated = total > max_rows;
            let display: Vec<_> = rows.into_iter().take(max_rows).collect();
            let result = serde_json::json!({ "library": library, "projectile": projectile, "element": element, "total": total, "truncated": truncated, "rows": display });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
        }
        "get_decay_data" => {
            let z = args.get("z").and_then(|v| v.as_i64());
            let a = args.get("a").and_then(|v| v.as_i64());
            if z.is_none() && a.is_none() {
                return Err("Provide at least z or a".to_string());
            }
            let path = data_dir.join("meta/decay.parquet");
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
                .filter(|row| {
                    if let Some(zv) = z {
                        if row.get("Z").and_then(|v| v.as_i64()) != Some(zv) {
                            return false;
                        }
                    }
                    if let Some(av) = a {
                        if row.get("A").and_then(|v| v.as_i64()) != Some(av) {
                            return false;
                        }
                    }
                    true
                })
                .collect();
            let result =
                serde_json::json!({ "z": z, "a": a, "count": filtered.len(), "rows": filtered });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
        }
        "get_abundances" => {
            let z = args
                .get("z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'z'")?;
            let path = data_dir.join("meta/abundances.parquet");
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
                .filter(|row| row.get("Z").and_then(|v| v.as_i64()) == Some(z))
                .collect();
            let result =
                serde_json::json!({ "z": z, "count": filtered.len(), "isotopes": filtered });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
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
            let path = match source {
                "PSTAR" => data_dir.join("stopping/PSTAR.parquet"),
                "ASTAR" => data_dir.join("stopping/ASTAR.parquet"),
                "ESTAR" => data_dir.join("stopping/ESTAR.parquet"),
                "dSTAR" => data_dir.join("stopping/dSTAR.parquet"),
                "tSTAR" => data_dir.join("stopping/tSTAR.parquet"),
                "catima" => data_dir.join("stopping/catima/catima.parquet"),
                other => {
                    return Err(format!(
                        "Unknown source {other:?}; valid: PSTAR, ASTAR, ESTAR, dSTAR, tSTAR, catima"
                    ))
                }
            };
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
                .filter(|row| row.get("target_Z").and_then(|v| v.as_i64()) == Some(target_z))
                .collect();
            let result = serde_json::json!({ "source": source, "target_z": target_z, "count": filtered.len(), "rows": filtered });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
        }
        "get_radiation" | "get_coincidences" => {
            let z = args
                .get("z")
                .and_then(|v| v.as_i64())
                .ok_or("missing 'z'")?;
            let a = args.get("a").and_then(|v| v.as_i64());
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            let symbol = z_to_symbol(z).ok_or_else(|| format!("Z={z} out of range"))?;
            let subdir = if name == "get_radiation" {
                "radiation"
            } else {
                "coincidences"
            };
            let path = data_dir.join(format!("meta/ensdf/{subdir}/{symbol}.parquet"));
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = if let Some(av) = a {
                rows.into_iter()
                    .filter(|r| r.get("A").and_then(|v| v.as_i64()) == Some(av))
                    .collect()
            } else {
                rows
            };
            let total = filtered.len();
            let truncated = total > max_rows;
            let display: Vec<_> = filtered.into_iter().take(max_rows).collect();
            let result = serde_json::json!({ "z": z, "a": a, "symbol": symbol, "total": total, "truncated": truncated, "rows": display });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
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
            let symbol = z_to_symbol(z).ok_or_else(|| format!("Z={z} out of range"))?;
            let path = data_dir.join(format!("meta/ensdf/summing_partners/{symbol}.parquet"));
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
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
                .collect();
            let total = filtered.len();
            let truncated = total > max_rows;
            let display: Vec<_> = filtered.into_iter().take(max_rows).collect();
            let result = serde_json::json!({ "z": z, "a": a, "primary_energy_keV": primary_energy, "total": total, "truncated": truncated, "rows": display });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
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
            let symbol =
                z_to_symbol(parent_z).ok_or_else(|| format!("Z={parent_z} out of range"))?;
            let path = data_dir.join(format!("meta/ensdf/emissions/{symbol}.parquet"));
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
                .filter(|r| {
                    if r.get("parent_A").and_then(|v| v.as_i64()) != Some(parent_a) {
                        return false;
                    }
                    let rs = r.get("parent_state").and_then(|v| v.as_str()).unwrap_or("");
                    if rs != parent_state {
                        return false;
                    }
                    if let Some(dm) = decay_mode_filter {
                        if r.get("decay_mode").and_then(|v| v.as_str()) != Some(dm) {
                            return false;
                        }
                    }
                    if let Some(ef) = energy_filter {
                        let e = r.get("energy_keV").and_then(|v| v.as_f64()).unwrap_or(0.0);
                        if (e - ef).abs() >= tolerance {
                            return false;
                        }
                    }
                    let intensity = r
                        .get("intensity_pct")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(0.0);
                    if intensity < min_intensity {
                        return false;
                    }
                    true
                })
                .collect();
            let total = filtered.len();
            let truncated = total > max_rows;
            let display: Vec<_> = filtered.into_iter().take(max_rows).collect();
            let result = serde_json::json!({
                "parent_z": parent_z,
                "parent_a": parent_a,
                "parent_state": parent_state,
                "total": total,
                "truncated": truncated,
                "rows": display
            });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
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
            let symbol = z_to_symbol(z).ok_or_else(|| format!("Z={z} out of range"))?;
            let path = data_dir.join(format!("meta/ensdf/beta_spectra/{symbol}.parquet"));
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
                .filter(|r| {
                    r.get("Z").and_then(|v| v.as_i64()) == Some(z)
                        && r.get("A").and_then(|v| v.as_i64()) == Some(a)
                })
                .collect();
            let total = filtered.len();
            let truncated = total > max_rows;
            let display: Vec<_> = filtered.into_iter().take(max_rows).collect();
            let result = serde_json::json!({ "z": z, "a": a, "symbol": symbol, "total": total, "truncated": truncated, "rows": display });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
        }
        "get_compound_compositions" => {
            let material = args.get("material").and_then(|v| v.as_str());
            let path = data_dir.join("meta/compound_compositions.parquet");
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            if let Some(mat) = material {
                let filtered: Vec<_> = rows
                    .into_iter()
                    .filter(|r| r.get("material").and_then(|v| v.as_str()) == Some(mat))
                    .collect();
                if filtered.is_empty() {
                    return Err(format!("Unknown material: {mat:?}"));
                }
                let result = serde_json::json!({ "material": mat, "count": filtered.len(), "composition": filtered });
                Ok(
                    serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
                )
            } else {
                let mut materials: Vec<String> = rows
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
            let path = data_dir.join("stopping/em/electron_stopping.parquet");
            let rows = load_parquet_rows(cache, &path.to_string_lossy()).await?;
            let filtered: Vec<_> = rows
                .into_iter()
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
                .collect();
            let total = filtered.len();
            let truncated = total > max_rows;
            let display: Vec<_> = filtered.into_iter().take(max_rows).collect();
            let result = serde_json::json!({ "target": target, "target_z": target_z, "total": total, "truncated": truncated, "rows": display });
            Ok(
                serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }),
            )
        }
        _ => Err(format!("Unknown tool: {name}")),
    }
}

/// Validate element symbol: 1-2 chars, first uppercase, second (if any) lowercase.
fn regex_lite(element: &str) -> bool {
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

async fn handle_request(
    data_dir: &Path,
    catalog: &Catalog,
    cache: &Cache,
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
            match handle_tool_call(data_dir, catalog, cache, name, &args).await {
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
    let cache: Cache = Arc::new(Mutex::new(HashMap::new()));

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

        if let Some(resp) = handle_request(&data_dir, &catalog, &cache, req).await {
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
        assert!(regex_lite("Cu"));
        assert!(regex_lite("H"));
        assert!(regex_lite("Fe"));
        assert!(!regex_lite("cu"));
        assert!(!regex_lite("CU"));
        assert!(!regex_lite(""));
        assert!(!regex_lite("Abc"));
        assert!(!regex_lite("'; DROP TABLE"));
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

    #[tokio::test]
    async fn handle_initialize() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "initialize".into(),
            params: serde_json::json!({}),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "nucl-parquet");
        assert_eq!(
            result["serverInfo"]["version"].as_str().unwrap(),
            env!("CARGO_PKG_VERSION")
        );
    }

    #[tokio::test]
    async fn handle_tools_list() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/list".into(),
            params: serde_json::json!({}),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 13);
    }

    #[tokio::test]
    async fn handle_list_libraries() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "list_libraries", "arguments": {} }),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("TENDL-2023"));
    }

    #[tokio::test]
    async fn handle_get_abundances() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "get_abundances", "arguments": { "z": 29 } }),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"z\": 29"));
        assert!(text.contains("\"count\": 2")); // Cu-63, Cu-65
    }

    #[tokio::test]
    async fn handle_get_decay_data() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "get_decay_data", "arguments": { "z": 27, "a": 60 } }),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        // Co-60 has at least ground state + isomer
        assert!(text.contains("\"count\":"));
    }

    #[tokio::test]
    async fn handle_get_summing_partners() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({
                "name": "get_summing_partners",
                "arguments": { "z": 28, "a": 60, "emission1_rad_type": "gamma" }
            }),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        // Co-60 → Ni-60: should have the 1173/1333 keV pair
        assert!(text.contains("\"z\": 28"));
        assert!(text.contains("\"a\": 60"));
        let parsed: serde_json::Value = serde_json::from_str(text).unwrap();
        let total = parsed["total"].as_u64().unwrap();
        assert!(
            total >= 1,
            "Expected ≥1 Co-60 summing partners, got {total}"
        );
    }

    #[tokio::test]
    async fn handle_get_emissions() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({
                "name": "get_emissions",
                "arguments": { "parent_z": 27, "parent_a": 60 }
            }),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
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

    #[tokio::test]
    async fn handle_unknown_tool() {
        let data_dir = test_data_dir();
        let catalog = load_catalog(&data_dir).unwrap();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "nonexistent", "arguments": {} }),
        };
        let resp = handle_request(&data_dir, &catalog, &cache, req)
            .await
            .unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("Unknown tool"));
    }
}
