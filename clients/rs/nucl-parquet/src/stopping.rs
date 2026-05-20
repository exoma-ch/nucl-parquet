//! Stopping power databases (NIST PSTAR/ASTAR/ESTAR + dSTAR/tSTAR and CatIMA).
//!
//! Provides mass stopping power lookups via log-log interpolation.
//! `StoppingDb` is `Send + Sync` — load once, share via `Arc`.
//!
//! ## Source routing (post-#137)
//!
//! - p, d, t → NIST PSTAR via [`dedx`]; d/t velocity-scaled at the caller
//!   (E_p = E / A) before lookup (dSTAR/tSTAR pre-built).
//! - α → NIST ASTAR via [`dedx`] (ICRU-49 reference; reproducible via
//!   `nucl_parquet.build_stopping`).
//! - ³He → [`catima_dedx`] with `proj_z = 2` (no NIST ³He table exists).
//! - e → NIST ESTAR via [`dedx`].
//! - heavy ions → [`catima_dedx`] (catima's full 92×92 master).
//!
//! The previously-shipped He3STAR.parquet and the broken ASTAR.parquet
//! (Z²-scaled from PSTAR at the wrong energy axis) were removed in #143.
//!
//! [`dedx`]: StoppingDb::dedx
//! [`catima_dedx`]: StoppingDb::catima_dedx

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::Error;
use crate::interp::{log_log_interp, sort_paired_vecs, XYTable};

/// Mass stopping power database for NIST tabulated sources (PSTAR, ASTAR, ESTAR,
/// dSTAR, tSTAR) and CatIMA heavy-ion calculations.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<StoppingDb>`.
#[derive(Clone)]
pub struct StoppingDb {
    /// (source, target_Z) -> (energy_MeV sorted, dedx sorted)
    nist: HashMap<(String, u32), XYTable>,
    /// (proj_Z, target_Z) -> (energy_MeV_u sorted, dedx sorted)
    catima: HashMap<(u32, u32), XYTable>,
    /// (proj_Z, target_Z) -> Bohr straggling dΩ²/d(ρx) [MeV² cm²/g] (constant per pair)
    catima_strag: HashMap<(u32, u32), f64>,
}

// Safety: all data is immutable after construction.
unsafe impl Send for StoppingDb {}
unsafe impl Sync for StoppingDb {}

impl StoppingDb {
    /// Load stopping power data from the nucl-parquet `stopping/` directory.
    ///
    /// Reads all `*.parquet` files at the top level (PSTAR, ASTAR, ESTAR,
    /// dSTAR, tSTAR, catima_*) plus the full master at `catima/catima.parquet`.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = data_dir.as_ref();

        if !dir.exists() {
            return Err(Error::DataDirNotFound(dir.to_path_buf()));
        }

        let nist = Self::load_nist(dir)?;
        let (catima, catima_strag) = Self::load_catima(dir)?;

        Ok(Self {
            nist,
            catima,
            catima_strag,
        })
    }

    /// Mass stopping power [MeV cm²/g] for a NIST source.
    ///
    /// Valid `source` values: `"PSTAR"`, `"ASTAR"`, `"ESTAR"`, `"dSTAR"`, `"tSTAR"`.
    /// Returns `f64::NAN` if the (source, target_Z) combination is not loaded.
    ///
    /// `energy_mev` is the projectile's *total* kinetic energy. ASTAR is keyed
    /// on total α KE; PSTAR/dSTAR/tSTAR are projectile-specific tables; ESTAR
    /// is electron KE. For ³He route via [`catima_dedx`] (`proj_z = 2`) — no
    /// NIST ³He table exists.
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

    /// Bohr energy straggling variance dΩ²/d(ρx) [MeV² cm²/g].
    ///
    /// Energy-independent (high-energy Bohr limit).
    /// Returns `f64::NAN` if the (proj_Z, target_Z) pair is not loaded.
    #[inline]
    pub fn catima_straggling(&self, proj_z: u32, target_z: u32) -> f64 {
        self.catima_strag
            .get(&(proj_z, target_z))
            .copied()
            .unwrap_or(f64::NAN)
    }

    // --- Raw table access (for consumers that do their own interpolation) ---

    /// Raw (energy_MeV, dedx) table for a NIST source + target_Z.
    ///
    /// Returns `None` if the (source, target_Z) combination is not loaded.
    /// Energy array is sorted ascending.
    pub fn nist_table(&self, source: &str, target_z: u32) -> Option<&XYTable> {
        self.nist.get(&(source.to_string(), target_z))
    }

    /// Iterate all loaded (source, target_Z) pairs in the NIST tables.
    pub fn nist_keys(&self) -> impl Iterator<Item = (&str, u32)> {
        self.nist.keys().map(|(s, z)| (s.as_str(), *z))
    }

    /// Raw (energy_MeV_u, dedx) table for a CatIMA (proj_Z, target_Z) pair.
    ///
    /// Returns `None` if the pair is not loaded.
    pub fn catima_table(&self, proj_z: u32, target_z: u32) -> Option<&XYTable> {
        self.catima.get(&(proj_z, target_z))
    }

    // --- Internal loaders ---

    fn load_nist(dir: &Path) -> crate::Result<HashMap<(String, u32), XYTable>> {
        let mut map: HashMap<(String, u32), XYTable> = HashMap::new();

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

                let src_col_ref = batch.column_by_name("source");
                let src_values = src_col_ref.and_then(|c| crate::interp::as_string_array(c));
                let z_col = batch
                    .column_by_name("target_Z")
                    .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
                let e_col = batch
                    .column_by_name("energy_MeV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let s_col = batch
                    .column_by_name("dedx")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                if let (Some(src), Some(z), Some(e), Some(s)) = (src_values, z_col, e_col, s_col) {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..batch.num_rows() {
                        let key = (src[i].unwrap_or("").to_string(), z.value(i) as u32);
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

    #[allow(clippy::type_complexity)] // two parallel maps keyed by (Z, A); splitting into a struct adds noise.
    fn load_catima(
        dir: &Path,
    ) -> crate::Result<(HashMap<(u32, u32), XYTable>, HashMap<(u32, u32), f64>)> {
        let catima_path = dir.join("catima").join("catima.parquet");
        let mut map: HashMap<(u32, u32), XYTable> = HashMap::new();
        let mut strag_map: HashMap<(u32, u32), f64> = HashMap::new();

        if !catima_path.exists() {
            return Ok((map, strag_map));
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
            let strag_col = batch
                .column_by_name("straggling")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(pz), Some(tz), Some(e), Some(s)) = (pz_col, tz_col, e_col, s_col) {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    let key = (pz.value(i) as u32, tz.value(i) as u32);
                    let entry = map.entry(key).or_default();
                    entry.0.push(e.value(i));
                    entry.1.push(s.value(i));
                    if let Some(strag) = strag_col {
                        strag_map.entry(key).or_insert(strag.value(i));
                    }
                }
            }
        }

        for (e_vec, s_vec) in map.values_mut() {
            sort_paired_vecs(e_vec, s_vec);
        }

        Ok((map, strag_map))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn data_dir() -> std::path::PathBuf {
        // Repo root is three levels up from this crate's manifest dir; data
        // moved under `data/` in the repo layout refactor.
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
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

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn alpha_on_cu_matches_nist_icru49_anchors() {
        // Post-#137 ASTAR.parquet is reproducible from NIST. These anchors
        // mirror tests/test_stopping_anchors.py — agreement to <1% guards
        // against fetcher / data regressions and the Z²-at-wrong-axis bug
        // class.
        let db = StoppingDb::open(data_dir()).unwrap();
        // (energy_MeV total, expected ICRU-49 dedx)
        let anchors: &[(f64, f64)] = &[
            (4.0, 483.8),   // 1 MeV/u
            (20.0, 177.4),  // 5 MeV/u
            (80.0, 64.54),  // 20 MeV/u
            (400.0, 19.32), // 100 MeV/u
        ];
        for &(e, expected) in anchors {
            let got = db.dedx("ASTAR", 29, e);
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 0.01,
                "α Cu at {e} MeV: got {got}, NIST {expected}, rel err {rel}",
            );
        }
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn legacy_he3star_source_missing() {
        // He3STAR.parquet was deleted in #143 — callers must use catima_dedx
        // with proj_z=2 instead. A `dedx("He3STAR", ...)` lookup returns NaN.
        let db = StoppingDb::open(data_dir()).unwrap();
        assert!(db.dedx("He3STAR", 29, 10.0).is_nan());
    }
}
