//! Generic, cached Parquet-to-JSON reader.
//!
//! Provides schema-agnostic access to any Parquet file in the data directory.
//! The primary consumer is the MCP server, which needs to return JSON rows for
//! arbitrary tables without hand-coding per-table structs.
//!
//! Typed DBs (`StoppingDb`, `DoseDb`, etc.) remain the preferred API for
//! programmatic Rust consumers — they add interpolation, type safety, and
//! domain-specific query methods on top.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use arrow::array::*;
use arrow::datatypes::DataType;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

use crate::error::Error;

/// A filter predicate for [`ParquetStore::load_filtered`].
#[derive(Debug, Clone)]
pub enum Filter {
    /// Column value equals the given JSON value (exact match).
    Eq(String, Value),
    /// `|column - target| < tolerance` for numeric columns.
    Near(String, f64, f64),
    /// `column >= threshold` for numeric columns.
    Gte(String, f64),
}

/// Schema column info: (name, arrow type as string).
pub type ColumnInfo = (String, String);

/// Generic, cached Parquet-to-JSON reader.
///
/// Thread-safe (`Send + Sync`). Share via `Arc<ParquetStore>`.
/// Caches loaded files so repeated queries on the same file are free.
pub struct ParquetStore {
    data_dir: PathBuf,
    cache: RwLock<HashMap<String, Arc<Vec<Value>>>>,
}

impl ParquetStore {
    /// Create a new store rooted at `data_dir`. No I/O until first query.
    pub fn new(data_dir: impl AsRef<Path>) -> Self {
        Self {
            data_dir: data_dir.as_ref().to_path_buf(),
            cache: RwLock::new(HashMap::new()),
        }
    }

    /// Load all rows from a Parquet file (relative to data_dir), cached.
    ///
    /// Returns an `Arc` — cheap to clone, shared across concurrent readers.
    pub fn load(&self, rel_path: &str) -> crate::Result<Arc<Vec<Value>>> {
        // Fast path: check read lock.
        {
            let c = self.cache.read().unwrap_or_else(|e| e.into_inner());
            if let Some(rows) = c.get(rel_path) {
                return Ok(Arc::clone(rows));
            }
        }

        // Slow path: read file, cache, return.
        let path = self.data_dir.join(rel_path);
        if !path.exists() {
            // Try lazy HTTP fetch if configured
            #[cfg(feature = "fetch")]
            {
                let marker = self.data_dir.join(".lazy_base_url");
                if marker.exists() {
                    crate::DataDir::from_root(&self.data_dir).fetch_file(rel_path)?;
                }
            }
            if !path.exists() {
                return Err(Error::DataDirNotFound(path));
            }
        }
        let rows = Arc::new(parse_parquet_file(&path)?);

        let mut c = self.cache.write().unwrap_or_else(|e| e.into_inner());
        c.entry(rel_path.to_string())
            .or_insert_with(|| Arc::clone(&rows));
        Ok(rows)
    }

    /// Load rows matching all filters (AND semantics).
    ///
    /// Uses the cached full file; filters are applied post-load.
    pub fn load_filtered(&self, rel_path: &str, filters: &[Filter]) -> crate::Result<Vec<Value>> {
        let all = self.load(rel_path)?;
        if filters.is_empty() {
            return Ok((*all).clone());
        }
        Ok(all
            .iter()
            .filter(|row| filters.iter().all(|f| matches_filter(row, f)))
            .cloned()
            .collect())
    }

    /// Column names and Arrow type names for a Parquet file.
    pub fn schema(&self, rel_path: &str) -> crate::Result<Vec<ColumnInfo>> {
        let path = self.data_dir.join(rel_path);
        let file = fs::File::open(&path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let schema = builder.schema();
        Ok(schema
            .fields()
            .iter()
            .map(|f| (f.name().clone(), format!("{}", f.data_type())))
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Filter matching
// ---------------------------------------------------------------------------

fn matches_filter(row: &Value, filter: &Filter) -> bool {
    match filter {
        Filter::Eq(col, expected) => row.get(col).is_some_and(|v| v == expected),
        Filter::Near(col, target, tolerance) => row
            .get(col)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| (v - target).abs() < *tolerance),
        Filter::Gte(col, threshold) => row
            .get(col)
            .and_then(|v| v.as_f64())
            .is_some_and(|v| v >= *threshold),
    }
}

// ---------------------------------------------------------------------------
// Parquet → JSON conversion
// ---------------------------------------------------------------------------

/// Read a Parquet file into a Vec of JSON objects.
pub fn parse_parquet_file(path: &Path) -> crate::Result<Vec<Value>> {
    let data = fs::read(path)?;
    let data = bytes::Bytes::from(data);
    let reader = ParquetRecordBatchReaderBuilder::try_new(data)?.build()?;

    let mut rows = Vec::new();
    for batch_result in reader {
        let batch = batch_result?;
        let schema = batch.schema();
        for row_idx in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let col = batch.column(col_idx);
                let val = column_value_to_json(col.as_ref(), row_idx);
                obj.insert(field.name().clone(), val);
            }
            rows.push(Value::Object(obj));
        }
    }
    Ok(rows)
}

/// Convert a single cell in an Arrow array to a JSON value.
///
/// Handles: Float64, Float32, Int64, Int32, Int16, UInt8, Boolean, String,
/// and dictionary-encoded strings (via cast fallback).
pub fn column_value_to_json(col: &dyn Array, idx: usize) -> Value {
    if col.is_null(idx) {
        return Value::Null;
    }
    if let Some(a) = col.as_any().downcast_ref::<Float64Array>() {
        let v = a.value(idx);
        Value::Number(
            serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)),
        )
    } else if let Some(a) = col.as_any().downcast_ref::<Float32Array>() {
        let v = a.value(idx) as f64;
        Value::Number(
            serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0)),
        )
    } else if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Value::Number(a.value(idx).into())
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        Value::Number(a.value(idx).into())
    } else if let Some(a) = col.as_any().downcast_ref::<Int16Array>() {
        Value::Number((a.value(idx) as i64).into())
    } else if let Some(a) = col.as_any().downcast_ref::<UInt8Array>() {
        Value::Number((a.value(idx) as i64).into())
    } else if let Some(a) = col.as_any().downcast_ref::<BooleanArray>() {
        Value::Bool(a.value(idx))
    } else if let Some(a) = col.as_any().downcast_ref::<StringArray>() {
        Value::String(a.value(idx).to_string())
    } else {
        // Fallback: try casting to Utf8 (handles dictionary-encoded strings)
        if let Ok(cast) = arrow::compute::cast(col, &DataType::Utf8) {
            if let Some(s) = cast.as_any().downcast_ref::<StringArray>() {
                return Value::String(s.value(idx).to_string());
            }
        }
        Value::String(format!("{col:?}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send<T: Send>() {}
    fn _assert_sync<T: Sync>() {}

    #[test]
    fn parquet_store_is_send_sync() {
        _assert_send::<ParquetStore>();
        _assert_sync::<ParquetStore>();
    }

    fn data_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn load_decay_parquet() {
        let store = ParquetStore::new(data_dir());
        let rows = store.load("meta/decay.parquet").unwrap();
        assert!(!rows.is_empty(), "decay.parquet should have rows");
        // Check first row has expected columns
        let first = &rows[0];
        assert!(first.get("Z").is_some());
        assert!(first.get("A").is_some());
        assert!(first.get("decay_mode").is_some());
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn filter_by_z() {
        let store = ParquetStore::new(data_dir());
        let co = store
            .load_filtered(
                "meta/decay.parquet",
                &[Filter::Eq("Z".into(), serde_json::json!(27))],
            )
            .unwrap();
        assert!(!co.is_empty(), "Co (Z=27) should have decay entries");
        assert!(co
            .iter()
            .all(|r| r.get("Z").and_then(|v| v.as_i64()) == Some(27)));
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn schema_introspection() {
        let store = ParquetStore::new(data_dir());
        let schema = store.schema("meta/decay.parquet").unwrap();
        let names: Vec<_> = schema.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"Z"));
        assert!(names.contains(&"decay_mode"));
        assert!(names.contains(&"branching"));
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn per_element_file() {
        let store = ParquetStore::new(data_dir());
        let rows = store
            .load_filtered(
                "meta/ensdf/radiation/Ni.parquet",
                &[Filter::Eq("A".into(), serde_json::json!(60))],
            )
            .unwrap();
        assert!(!rows.is_empty(), "Ni-60 radiation should have rows");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cache_hit() {
        let store = ParquetStore::new(data_dir());
        let _ = store.load("meta/decay.parquet").unwrap();
        // Second load should hit cache (Arc pointer equality)
        let a = store.load("meta/decay.parquet").unwrap();
        let b = store.load("meta/decay.parquet").unwrap();
        assert!(Arc::ptr_eq(&a, &b), "cached loads should return same Arc");
    }
}
