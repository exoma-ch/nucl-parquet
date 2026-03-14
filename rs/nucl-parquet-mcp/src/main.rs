//! nucl-parquet MCP server — lazy-loads Parquet files from GitHub.
//!
//! Implements the MCP (Model Context Protocol) over stdio using plain JSON-RPC 2.0.
//! Fetches individual Parquet files on demand from raw.githubusercontent.com,
//! parses them with the `parquet` crate, and returns results as JSON.

use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, StringArray};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::Mutex;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const BASE_URL: &str = "https://raw.githubusercontent.com/exoma-ch/nucl-parquet/main/";

// ---------------------------------------------------------------------------
// Embedded catalog
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Library {
    name: String,
    description: String,
    projectiles: Vec<String>,
    data_type: String,
    version: String,
    path: String,
}

fn catalog() -> HashMap<String, Library> {
    let mut m = HashMap::new();
    let libs: &[(&str, &str, &str, &[&str], &str, &str, &str)] = &[
        ("tendl-2024", "TENDL-2024", "TALYS Evaluated Nuclear Data Library 2024 (IAEA/PSI)", &["p","d","t","h","a"], "cross_sections", "2024", "tendl-2024/xs/"),
        ("endfb-8.1", "ENDF/B-VIII.1", "US Evaluated Nuclear Data File (NNDC/BNL)", &["n","p","d","t","h","a"], "cross_sections", "VIII.1", "endfb-8.1/xs/"),
        ("jeff-4.0", "JEFF-4.0", "Joint Evaluated Fission and Fusion File (NEA)", &["n","p"], "cross_sections", "4.0", "jeff-4.0/xs/"),
        ("jendl-5", "JENDL-5", "Japanese Evaluated Nuclear Data Library (JAEA)", &["n","p","d","a"], "cross_sections", "5", "jendl-5/xs/"),
        ("tendl-2025", "TENDL-2025", "TALYS Evaluated Nuclear Data Library 2025 (PSI)", &["n","p","d","t","h","a"], "cross_sections", "2025", "tendl-2025/xs/"),
        ("cendl-3.2", "CENDL-3.2", "Chinese Evaluated Nuclear Data Library (CIAE)", &["n"], "cross_sections", "3.2", "cendl-3.2/xs/"),
        ("brond-3.1", "BROND-3.1", "Russian Evaluated Nuclear Data Library (IPPE)", &["n"], "cross_sections", "3.1", "brond-3.1/xs/"),
        ("fendl-3.2", "FENDL-3.2", "Fusion Evaluated Nuclear Data Library (IAEA)", &["n"], "cross_sections", "3.2", "fendl-3.2/xs/"),
        ("eaf-2010", "EAF-2010", "European Activation File (CCFE)", &["n"], "cross_sections", "2010", "eaf-2010/xs/"),
        ("irdff-2", "IRDFF-II", "International Reactor Dosimetry and Fusion File (IAEA)", &["n"], "cross_sections", "II", "irdff-2/xs/"),
        ("iaea-medical", "IAEA-Medical", "Medical isotope production cross-sections (IAEA)", &["p","d","h","a"], "cross_sections", "latest", "iaea-medical/xs/"),
        ("jendl-ad-2017", "JENDL/AD-2017", "Activation/Dosimetry Library (JAEA)", &["n","p"], "cross_sections", "2017", "jendl-ad-2017/xs/"),
        ("jendl-deu-2020", "JENDL-DEU-2020", "Dedicated deuteron-induced reaction library (JAEA)", &["d"], "cross_sections", "2020", "jendl-deu-2020/xs/"),
        ("iaea-pd-2019", "IAEA-PD-2019", "Photonuclear Data Library (IAEA)", &["g"], "cross_sections", "2019", "iaea-pd-2019/xs/"),
        ("exfor", "EXFOR", "Experimental nuclear reaction data (IAEA NDS)", &["n","p","d","t","h","a"], "experimental_cross_sections", "latest", "exfor/"),
    ];
    for &(id, name, desc, projs, dt, ver, path) in libs {
        m.insert(id.to_string(), Library {
            name: name.to_string(),
            description: desc.to_string(),
            projectiles: projs.iter().map(|s| s.to_string()).collect(),
            data_type: dt.to_string(),
            version: ver.to_string(),
            path: path.to_string(),
        });
    }
    m
}

// ---------------------------------------------------------------------------
// Parquet fetch + cache
// ---------------------------------------------------------------------------

type Cache = Arc<Mutex<HashMap<String, Vec<serde_json::Value>>>>;

async fn fetch_parquet_rows(
    client: &reqwest::Client,
    cache: &Cache,
    relative_path: &str,
) -> Result<Vec<serde_json::Value>, String> {
    {
        let c = cache.lock().await;
        if let Some(rows) = c.get(relative_path) {
            return Ok(rows.clone());
        }
    }

    let base = std::env::var("NUCL_PARQUET_BASE_URL").unwrap_or_else(|_| BASE_URL.to_string());
    let url = format!("{}{}", base.trim_end_matches('/'), if relative_path.starts_with('/') { relative_path.to_string() } else { format!("/{relative_path}") });

    let resp = client.get(&url).send().await.map_err(|e| format!("HTTP error: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("HTTP {} fetching {url}", resp.status()));
    }
    let data = resp.bytes().await.map_err(|e| format!("Read error: {e}"))?;
    let rows = parse_parquet_bytes(data)?;

    let mut c = cache.lock().await;
    c.insert(relative_path.to_string(), rows.clone());
    Ok(rows)
}

fn parse_parquet_bytes(data: Bytes) -> Result<Vec<serde_json::Value>, String> {
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
        serde_json::Value::Number(serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)))
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        serde_json::Value::Number(a.value(idx).into())
    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        serde_json::Value::String(a.value(idx).to_string())
    } else {
        // Fallback: try as string via debug
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
        Self { jsonrpc: "2.0".into(), id, result: Some(result), error: None }
    }
    fn error(id: serde_json::Value, code: i32, message: String) -> Self {
        Self { jsonrpc: "2.0".into(), id, result: None, error: Some(JsonRpcError { code, message }) }
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
                        "library": { "type": "string", "description": "Library ID, e.g. 'tendl-2024'" },
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
                        "source": { "type": "string", "description": "PSTAR, ASTAR, ICRU73, or MSTAR" },
                        "target_z": { "type": "integer", "description": "Target atomic number" }
                    },
                    "required": ["source", "target_z"]
                }
            }
        ]
    })
}

// ---------------------------------------------------------------------------
// Tool dispatch
// ---------------------------------------------------------------------------

async fn handle_tool_call(
    client: &reqwest::Client,
    cache: &Cache,
    name: &str,
    args: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let cat = catalog();
    match name {
        "list_libraries" => {
            let libs: Vec<serde_json::Value> = cat.iter().map(|(id, lib)| {
                serde_json::json!({
                    "id": id,
                    "name": lib.name,
                    "description": lib.description,
                    "projectiles": lib.projectiles,
                    "version": lib.version,
                    "data_type": lib.data_type,
                })
            }).collect();
            Ok(serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&libs).unwrap() }] }))
        }
        "list_isotopes" => {
            let library = args.get("library").and_then(|v| v.as_str()).ok_or("missing 'library'")?;
            let projectile = args.get("projectile").and_then(|v| v.as_str()).ok_or("missing 'projectile'")?;
            let lib = cat.get(library).ok_or_else(|| format!("Unknown library: {library}"))?;
            if !lib.projectiles.iter().any(|p| p == projectile) {
                return Err(format!("Projectile '{projectile}' not in {library}"));
            }
            let manifest_path = lib.path.replace("xs/", "manifest.json");
            let url = format!("{}{}", std::env::var("NUCL_PARQUET_BASE_URL").unwrap_or_else(|_| BASE_URL.to_string()).trim_end_matches('/'), format!("/{manifest_path}"));
            let resp = client.get(&url).send().await.map_err(|e| e.to_string())?;
            let manifest: serde_json::Value = resp.json::<serde_json::Value>().await.map_err(|e| e.to_string())?;
            let elements = manifest.get("elements").cloned().unwrap_or(serde_json::json!([]));
            let result = serde_json::json!({ "library": library, "projectile": projectile, "elements": elements });
            Ok(serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }))
        }
        "get_cross_sections" => {
            let library = args.get("library").and_then(|v| v.as_str()).ok_or("missing 'library'")?;
            let projectile = args.get("projectile").and_then(|v| v.as_str()).ok_or("missing 'projectile'")?;
            let element = args.get("element").and_then(|v| v.as_str()).ok_or("missing 'element'")?;
            let max_rows = args.get("max_rows").and_then(|v| v.as_u64()).unwrap_or(500) as usize;
            let lib = cat.get(library).ok_or_else(|| format!("Unknown library: {library}"))?;
            let path = format!("{}{projectile}_{element}.parquet", lib.path);
            let rows = fetch_parquet_rows(client, cache, &path).await?;
            let total = rows.len();
            let truncated = total > max_rows;
            let display_rows: Vec<_> = rows.into_iter().take(max_rows).collect();
            let result = serde_json::json!({ "library": library, "projectile": projectile, "element": element, "total": total, "truncated": truncated, "rows": display_rows });
            Ok(serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }))
        }
        "get_decay_data" => {
            let z = args.get("z").and_then(|v| v.as_i64());
            let a = args.get("a").and_then(|v| v.as_i64());
            if z.is_none() && a.is_none() {
                return Err("Provide at least z or a".to_string());
            }
            let rows = fetch_parquet_rows(client, cache, "meta/decay.parquet").await?;
            let filtered: Vec<_> = rows.into_iter().filter(|row| {
                if let Some(zv) = z { if row.get("Z").and_then(|v| v.as_i64()) != Some(zv) { return false; } }
                if let Some(av) = a { if row.get("A").and_then(|v| v.as_i64()) != Some(av) { return false; } }
                true
            }).collect();
            let result = serde_json::json!({ "z": z, "a": a, "count": filtered.len(), "rows": filtered });
            Ok(serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }))
        }
        "get_abundances" => {
            let z = args.get("z").and_then(|v| v.as_i64()).ok_or("missing 'z'")?;
            let rows = fetch_parquet_rows(client, cache, "meta/abundances.parquet").await?;
            let filtered: Vec<_> = rows.into_iter().filter(|row| row.get("Z").and_then(|v| v.as_i64()) == Some(z)).collect();
            let result = serde_json::json!({ "z": z, "count": filtered.len(), "isotopes": filtered });
            Ok(serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }))
        }
        "get_stopping_power" => {
            let source = args.get("source").and_then(|v| v.as_str()).ok_or("missing 'source'")?;
            let target_z = args.get("target_z").and_then(|v| v.as_i64()).ok_or("missing 'target_z'")?;
            let rows = fetch_parquet_rows(client, cache, "stopping/stopping.parquet").await?;
            let filtered: Vec<_> = rows.into_iter().filter(|row| {
                row.get("source").and_then(|v| v.as_str()) == Some(source) &&
                row.get("target_Z").and_then(|v| v.as_i64()) == Some(target_z)
            }).collect();
            let result = serde_json::json!({ "source": source, "target_z": target_z, "count": filtered.len(), "rows": filtered });
            Ok(serde_json::json!({ "content": [{ "type": "text", "text": serde_json::to_string_pretty(&result).unwrap() }] }))
        }
        _ => Err(format!("Unknown tool: {name}")),
    }
}

// ---------------------------------------------------------------------------
// MCP protocol handler
// ---------------------------------------------------------------------------

async fn handle_request(
    client: &reqwest::Client,
    cache: &Cache,
    req: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    let id = req.id.clone().unwrap_or(serde_json::Value::Null);

    // Notifications (no id) don't get responses
    if req.id.is_none() {
        return None;
    }

    match req.method.as_str() {
        "initialize" => {
            Some(JsonRpcResponse::success(id, serde_json::json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "nucl-parquet", "version": "0.3.7" }
            })))
        }
        "tools/list" => {
            Some(JsonRpcResponse::success(id, tool_definitions()))
        }
        "tools/call" => {
            let name = req.params.get("name").and_then(|v| v.as_str()).unwrap_or("");
            let args = req.params.get("arguments").cloned().unwrap_or(serde_json::json!({}));
            match handle_tool_call(client, cache, name, &args).await {
                Ok(result) => Some(JsonRpcResponse::success(id, result)),
                Err(e) => Some(JsonRpcResponse::error(id, -32000, e)),
            }
        }
        _ => {
            Some(JsonRpcResponse::error(id, -32601, format!("Method not found: {}", req.method)))
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    let client = reqwest::Client::builder()
        .user_agent("nucl-parquet-mcp/0.3.7")
        .build()
        .expect("Failed to build HTTP client");

    let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let stdout = std::io::stdout();

    let mut line = String::new();
    loop {
        line.clear();
        match reader.read_line(&mut line).await {
            Ok(0) => break, // EOF
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

        if let Some(resp) = handle_request(&client, &cache, req).await {
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

    #[test]
    fn catalog_has_all_libraries() {
        let cat = catalog();
        assert!(cat.len() >= 15);
        assert!(cat.contains_key("tendl-2024"));
        assert!(cat.contains_key("endfb-8.1"));
        assert!(cat.contains_key("exfor"));
    }

    #[test]
    fn catalog_projectiles_not_empty() {
        let cat = catalog();
        for (id, lib) in &cat {
            assert!(!lib.projectiles.is_empty(), "{id} has no projectiles");
        }
    }

    #[test]
    fn tool_definitions_valid() {
        let defs = tool_definitions();
        let tools = defs.get("tools").unwrap().as_array().unwrap();
        assert_eq!(tools.len(), 6);
        for tool in tools {
            assert!(tool.get("name").is_some());
            assert!(tool.get("description").is_some());
            assert!(tool.get("inputSchema").is_some());
        }
    }

    #[test]
    fn json_rpc_response_serialization() {
        let resp = JsonRpcResponse::success(
            serde_json::json!(1),
            serde_json::json!({"key": "value"}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"jsonrpc\":\"2.0\""));
        assert!(json.contains("\"key\":\"value\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn json_rpc_error_serialization() {
        let resp = JsonRpcResponse::error(
            serde_json::json!(2),
            -32601,
            "Method not found".into(),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32601"));
        assert!(!json.contains("\"result\""));
    }

    #[test]
    fn parse_request() {
        let json = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "tools/list");
        assert_eq!(req.id, Some(serde_json::json!(1)));
    }

    #[tokio::test]
    async fn handle_initialize() {
        let client = reqwest::Client::new();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "initialize".into(),
            params: serde_json::json!({}),
        };
        let resp = handle_request(&client, &cache, req).await.unwrap();
        let result = resp.result.unwrap();
        assert_eq!(result["serverInfo"]["name"], "nucl-parquet");
    }

    #[tokio::test]
    async fn handle_tools_list() {
        let client = reqwest::Client::new();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/list".into(),
            params: serde_json::json!({}),
        };
        let resp = handle_request(&client, &cache, req).await.unwrap();
        let result = resp.result.unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 6);
    }

    #[tokio::test]
    async fn handle_list_libraries() {
        let client = reqwest::Client::new();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "list_libraries", "arguments": {} }),
        };
        let resp = handle_request(&client, &cache, req).await.unwrap();
        assert!(resp.error.is_none());
        let content = resp.result.unwrap();
        let text = content["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("TENDL-2024"));
    }

    #[tokio::test]
    async fn handle_unknown_tool() {
        let client = reqwest::Client::new();
        let cache: Cache = Arc::new(Mutex::new(HashMap::new()));
        let req = JsonRpcRequest {
            jsonrpc: "2.0".into(),
            id: Some(serde_json::json!(1)),
            method: "tools/call".into(),
            params: serde_json::json!({ "name": "nonexistent", "arguments": {} }),
        };
        let resp = handle_request(&client, &cache, req).await.unwrap();
        assert!(resp.error.is_some());
        assert!(resp.error.unwrap().message.contains("Unknown tool"));
    }
}
