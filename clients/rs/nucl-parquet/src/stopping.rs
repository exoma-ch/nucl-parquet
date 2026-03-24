//! Stopping power databases (NIST PSTAR/ASTAR/ESTAR and CatIMA).
//!
//! Provides mass stopping power lookups via log-log interpolation.
//! `StoppingDb` is `Send + Sync` — load once, share via `Arc`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::Error;
use crate::interp::{log_log_interp, sort_paired_vecs};

/// Mass stopping power database for NIST tabulated sources (PSTAR, ASTAR, ESTAR, …)
/// and CatIMA heavy-ion calculations.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<StoppingDb>`.
#[derive(Clone)]
pub struct StoppingDb {
    /// (source, target_Z) -> (energy_MeV sorted, dedx sorted)
    nist: HashMap<(String, u32), (Vec<f64>, Vec<f64>)>,
    /// (proj_Z, target_Z) -> (energy_MeV_u sorted, dedx sorted)
    catima: HashMap<(u32, u32), (Vec<f64>, Vec<f64>)>,
}

// Safety: all data is immutable after construction.
unsafe impl Send for StoppingDb {}
unsafe impl Sync for StoppingDb {}

impl StoppingDb {
    /// Load stopping power data from the nucl-parquet `stopping/` directory.
    ///
    /// Reads all `*.parquet` files (PSTAR, ASTAR, ESTAR, …) and
    /// `catima/catima.parquet`.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = data_dir.as_ref();

        if !dir.exists() {
            return Err(Error::DataDirNotFound(dir.to_path_buf()));
        }

        let nist = Self::load_nist(dir)?;
        let catima = Self::load_catima(dir)?;

        Ok(Self { nist, catima })
    }

    /// Mass stopping power [MeV cm²/g] for a NIST source (e.g. "PSTAR", "ASTAR", "ESTAR").
    ///
    /// Returns `f64::NAN` if the (source, target_Z) combination is not loaded.
    #[inline]
    pub fn dedx(&self, source: &str, target_z: u32, energy_mev: f64) -> f64 {
        match self.nist.get(&(source.to_string(), target_z)) {
            Some((e, s)) => log_log_interp(e, s, energy_mev),
            None => f64::NAN,
        }
    }

    /// CatIMA mass stopping power [MeV cm²/g].
    ///
    /// `energy_mev_u` is the projectile kinetic energy per nucleon.
    /// Returns `f64::NAN` if the (proj_Z, target_Z) pair is not loaded.
    #[inline]
    pub fn catima_dedx(&self, proj_z: u32, target_z: u32, energy_mev_u: f64) -> f64 {
        match self.catima.get(&(proj_z, target_z)) {
            Some((e, s)) => log_log_interp(e, s, energy_mev_u),
            None => f64::NAN,
        }
    }

    // --- Internal loaders ---

    fn load_nist(dir: &Path) -> crate::Result<HashMap<(String, u32), (Vec<f64>, Vec<f64>)>> {
        let mut map: HashMap<(String, u32), (Vec<f64>, Vec<f64>)> = HashMap::new();

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

            for batch in reader {
                let batch = batch?;

                let source_col = batch
                    .column_by_name("source")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let z_col = batch
                    .column_by_name("target_Z")
                    .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
                let e_col = batch
                    .column_by_name("energy_MeV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let s_col = batch
                    .column_by_name("dedx")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                if let (Some(src), Some(z), Some(e), Some(s)) = (source_col, z_col, e_col, s_col) {
                    for i in 0..batch.num_rows() {
                        let key = (src.value(i).to_string(), z.value(i) as u32);
                        let entry = map.entry(key).or_default();
                        entry.0.push(e.value(i));
                        entry.1.push(s.value(i));
                    }
                }
            }
        }

        for (e_vec, s_vec) in map.values_mut() {
            sort_paired_vecs(e_vec, s_vec);
        }

        Ok(map)
    }

    fn load_catima(dir: &Path) -> crate::Result<HashMap<(u32, u32), (Vec<f64>, Vec<f64>)>> {
        let catima_path = dir.join("catima").join("catima.parquet");
        let mut map: HashMap<(u32, u32), (Vec<f64>, Vec<f64>)> = HashMap::new();

        if !catima_path.exists() {
            return Ok(map);
        }

        let file = fs::File::open(&catima_path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let pz_col = batch
                .column_by_name("proj_Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let tz_col = batch
                .column_by_name("target_Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let e_col = batch
                .column_by_name("energy_MeV_u")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let s_col = batch
                .column_by_name("dedx")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(pz), Some(tz), Some(e), Some(s)) = (pz_col, tz_col, e_col, s_col) {
                for i in 0..batch.num_rows() {
                    let key = (pz.value(i) as u32, tz.value(i) as u32);
                    let entry = map.entry(key).or_default();
                    entry.0.push(e.value(i));
                    entry.1.push(s.value(i));
                }
            }
        }

        for (e_vec, s_vec) in map.values_mut() {
            sort_paired_vecs(e_vec, s_vec);
        }

        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_dir() -> std::path::PathBuf {
        // Repo root is three levels up from src/
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("stopping")
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn open_succeeds() {
        let db = StoppingDb::open(data_dir()).unwrap();
        // PSTAR proton stopping in Cu (Z=29) at 10 MeV should be a positive finite value
        let s = db.dedx("PSTAR", 29, 10.0);
        assert!(s.is_finite() && s > 0.0, "PSTAR Cu 10 MeV: {s}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn catima_open_succeeds() {
        let db = StoppingDb::open(data_dir()).unwrap();
        // Proton (Z=1) in Cu (Z=29) at 100 MeV/u
        let s = db.catima_dedx(1, 29, 100.0);
        assert!(s.is_finite() && s > 0.0, "CatIMA p in Cu 100 MeV/u: {s}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn miss_returns_nan() {
        let db = StoppingDb::open(data_dir()).unwrap();
        let s = db.dedx("NONEXISTENT", 999, 10.0);
        assert!(s.is_nan());
    }
}
