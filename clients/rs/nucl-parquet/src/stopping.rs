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
//! - ³He → [`catima_dedx`] with `proj_z = 2, proj_a = 3` (no NIST ³He table exists).
//! - e → NIST ESTAR via [`dedx`].
//! - heavy ions → [`catima_dedx`] (catima's full per-isotope master).
//!
//! CatIMA lookups are keyed per-isotope `(proj_Z, proj_A, target_Z)`: stopping
//! is reduced-mass dependent, so isotopes of the same Z must not be collapsed
//! together (see #246).
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
    /// (source, compound_name) -> (energy_MeV sorted, dedx sorted)
    compounds: HashMap<(String, String), XYTable>,
    /// (proj_Z, proj_A, target_Z) -> (energy_MeV_u sorted, dedx sorted)
    ///
    /// Keyed per-isotope: CatIMA stopping is reduced-mass dependent, so isotopes
    /// of the same Z differ (up to ~15% below ~0.01 MeV/u where nuclear stopping
    /// dominates). Collapsing them onto (proj_Z, target_Z) interleaves their
    /// energy axes and corrupts the interpolation — see #246.
    catima: HashMap<(u32, u32, u32), XYTable>,
    /// (proj_Z, proj_A, target_Z) -> Bohr straggling dΩ²/d(ρx) [MeV² cm²/g] (constant per pair)
    catima_strag: HashMap<(u32, u32, u32), f64>,
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
        let compounds = Self::load_compounds(dir)?;
        let (catima, catima_strag) = Self::load_catima(dir)?;

        Ok(Self {
            nist,
            compounds,
            catima,
            catima_strag,
        })
    }

    /// Construct from in-memory NIST Parquet bytes (a single `*.parquet` file
    /// containing `source`, `target_Z`, `energy_MeV`, `dedx` columns).
    ///
    /// Compounds and CatIMA data will be empty. Suitable for loading a single
    /// PSTAR/ASTAR/ESTAR/dSTAR/tSTAR file from bytes.
    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        let bytes = bytes::Bytes::from(data.to_vec());
        let mut nist: HashMap<(String, u32), XYTable> = HashMap::new();
        Self::parse_nist_into(bytes, &mut nist)?;

        for (e_vec, s_vec) in nist.values_mut() {
            sort_paired_vecs(e_vec, s_vec);
        }

        Ok(Self {
            nist,
            compounds: HashMap::new(),
            catima: HashMap::new(),
            catima_strag: HashMap::new(),
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

    /// Mass stopping power [MeV cm²/g] for a NIST compound material.
    ///
    /// `source`: `"PSTAR_compound"` or `"ASTAR_compound"`.
    /// `compound`: NIST compound name in UPPER_SNAKE_CASE (e.g., `"WATER_LIQUID"`).
    /// Returns `f64::NAN` if the (source, compound) combination is not loaded.
    #[inline]
    pub fn compound_dedx(&self, source: &str, compound: &str, energy_mev: f64) -> f64 {
        match self
            .compounds
            .get(&(source.to_string(), compound.to_string()))
        {
            Some((e, s)) => log_log_interp(e, s, energy_mev),
            None => f64::NAN,
        }
    }

    /// CatIMA mass stopping power [MeV cm²/g] for a specific projectile isotope.
    ///
    /// `proj_a` is the projectile mass number: CatIMA tables are tabulated
    /// per-isotope and differ between isotopes of the same Z at low energy
    /// (up to ~15% below ~0.01 MeV/u). `energy_mev_u` is the projectile kinetic
    /// energy per nucleon.
    ///
    /// Returns `f64::NAN` if the (proj_Z, proj_A, target_Z) triple is not loaded.
    #[inline]
    pub fn catima_dedx(&self, proj_z: u32, proj_a: u32, target_z: u32, energy_mev_u: f64) -> f64 {
        match self.catima.get(&(proj_z, proj_a, target_z)) {
            Some((e, s)) => log_log_interp(e, s, energy_mev_u),
            None => f64::NAN,
        }
    }

    /// Bohr energy straggling variance dΩ²/d(ρx) [MeV² cm²/g].
    ///
    /// Energy-independent (high-energy Bohr limit).
    /// Returns `f64::NAN` if the (proj_Z, proj_A, target_Z) triple is not loaded.
    #[inline]
    pub fn catima_straggling(&self, proj_z: u32, proj_a: u32, target_z: u32) -> f64 {
        self.catima_strag
            .get(&(proj_z, proj_a, target_z))
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// Raw NIST stopping power table for a (source, target_Z) pair.
    ///
    /// Returns `(energy_MeV[], dedx[])` sorted by energy, or `None` if not loaded.
    #[inline]
    pub fn nist_table(&self, source: &str, target_z: u32) -> Option<&XYTable> {
        self.nist.get(&(source.to_string(), target_z))
    }

    /// Iterate all loaded NIST (source, target_Z) keys.
    pub fn nist_keys(&self) -> impl Iterator<Item = (&str, u32)> + '_ {
        self.nist.keys().map(|(s, z)| (s.as_str(), *z))
    }

    /// Raw compound stopping power table for a (source, compound) pair.
    ///
    /// Returns `(energy_MeV[], dedx[])` sorted by energy, or `None` if not loaded.
    /// Use this when you need the full table (e.g. for plotting or custom
    /// interpolation) rather than a single-point lookup via `compound_dedx()`.
    #[inline]
    pub fn compound_table(&self, source: &str, compound: &str) -> Option<&XYTable> {
        self.compounds
            .get(&(source.to_string(), compound.to_string()))
    }

    /// Iterate all loaded compound (source, name) keys.
    pub fn compound_keys(&self) -> impl Iterator<Item = (&str, &str)> + '_ {
        self.compounds.keys().map(|(s, c)| (s.as_str(), c.as_str()))
    }

    /// Raw CatIMA stopping power table for a (proj_Z, proj_A, target_Z) triple.
    ///
    /// Returns `(energy_MeV_u[], dedx[])` sorted by energy, or `None` if not loaded.
    #[inline]
    pub fn catima_table(&self, proj_z: u32, proj_a: u32, target_z: u32) -> Option<&XYTable> {
        self.catima.get(&(proj_z, proj_a, target_z))
    }

    /// Iterate all loaded CatIMA (proj_Z, proj_A, target_Z) keys.
    pub fn catima_keys(&self) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        self.catima.keys().copied()
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
            Self::parse_nist_into(file, &mut map)?;
        }

        for (e_vec, s_vec) in map.values_mut() {
            sort_paired_vecs(e_vec, s_vec);
        }

        Ok(map)
    }

    fn parse_nist_into(
        reader_source: impl parquet::file::reader::ChunkReader + 'static,
        map: &mut HashMap<(String, u32), XYTable>,
    ) -> crate::Result<()> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(reader_source)?.build()?;

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

        Ok(())
    }

    fn load_compounds(dir: &Path) -> crate::Result<HashMap<(String, String), XYTable>> {
        let mut map: HashMap<(String, String), XYTable> = HashMap::new();
        let compounds_dir = dir.join("compounds");
        if !compounds_dir.exists() {
            return Ok(map);
        }

        for entry in fs::read_dir(&compounds_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

            for batch in reader {
                let batch = batch?;

                let src_col_ref = batch.column_by_name("source");
                let src_values = src_col_ref.and_then(|c| crate::interp::as_string_array(c));
                let cmp_col_ref = batch.column_by_name("compound");
                let cmp_values = cmp_col_ref.and_then(|c| crate::interp::as_string_array(c));
                let e_col = batch
                    .column_by_name("energy_MeV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let s_col = batch
                    .column_by_name("dedx")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                if let (Some(src), Some(cmp), Some(e), Some(s)) =
                    (src_values, cmp_values, e_col, s_col)
                {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..batch.num_rows() {
                        let key = (
                            src[i].unwrap_or("").to_string(),
                            cmp[i].unwrap_or("").to_string(),
                        );
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

    #[allow(clippy::type_complexity)] // two parallel maps keyed by (Z, A, Z); splitting into a struct adds noise.
    fn load_catima(
        dir: &Path,
    ) -> crate::Result<(HashMap<(u32, u32, u32), XYTable>, HashMap<(u32, u32, u32), f64>)> {
        let catima_path = dir.join("catima").join("catima.parquet");
        let mut map: HashMap<(u32, u32, u32), XYTable> = HashMap::new();
        let mut strag_map: HashMap<(u32, u32, u32), f64> = HashMap::new();

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
            let pa_col = batch
                .column_by_name("proj_A")
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

            if let (Some(pz), Some(pa), Some(tz), Some(e), Some(s)) =
                (pz_col, pa_col, tz_col, e_col, s_col)
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    let key = (pz.value(i) as u32, pa.value(i) as u32, tz.value(i) as u32);
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
        // Proton (Z=1, A=1) in Cu (Z=29) at 100 MeV/u
        let s = db.catima_dedx(1, 1, 29, 100.0);
        assert!(s.is_finite() && s > 0.0, "CatIMA p in Cu 100 MeV/u: {s}");

        // Isotope resolution: He-3 and He-4 in Fe must differ at low energy
        // (reduced-mass-dependent nuclear stopping) — the #246 regression guard.
        let he3 = db.catima_dedx(2, 3, 26, 0.001);
        let he4 = db.catima_dedx(2, 4, 26, 0.001);
        assert!(he3.is_finite() && he4.is_finite(), "He3 {he3}, He4 {he4}");
        let rel = (he3 - he4).abs() / he4.max(he3);
        assert!(
            rel > 0.05,
            "He3 vs He4 in Fe at 0.001 MeV/u should differ >5%, got {rel} (he3={he3}, he4={he4})"
        );
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
    fn compound_water_positive() {
        let db = StoppingDb::open(data_dir()).unwrap();
        let s = db.compound_dedx("PSTAR_compound", "WATER_LIQUID", 10.0);
        assert!(s.is_finite() && s > 0.0, "PSTAR water 10 MeV: {s}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn compound_miss_returns_nan() {
        let db = StoppingDb::open(data_dir()).unwrap();
        assert!(db
            .compound_dedx("PSTAR_compound", "NONEXISTENT", 10.0)
            .is_nan());
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn from_bytes_matches_open() {
        let db_file = StoppingDb::open(data_dir()).unwrap();
        // Read the first NIST parquet file
        let first_file = std::fs::read_dir(data_dir())
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| {
                let p = e.path();
                !p.is_dir() && p.extension().and_then(|x| x.to_str()) == Some("parquet")
            })
            .expect("at least one stopping file");
        let data = std::fs::read(first_file.path()).unwrap();
        let db_bytes = StoppingDb::from_bytes(&data).unwrap();
        // Find a loaded (source, Z) and compare
        for (source, z) in db_bytes.nist_keys().take(3) {
            let val_file = db_file.dedx(source, z, 10.0);
            let val_bytes = db_bytes.dedx(source, z, 10.0);
            if val_file.is_finite() && val_bytes.is_finite() {
                assert!(
                    (val_file - val_bytes).abs() / val_file < 1e-10,
                    "{source} Z={z} dedx mismatch: {val_file} vs {val_bytes}"
                );
                return;
            }
        }
        panic!("no matching NIST key found");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn compound_table_returns_data() {
        let db = StoppingDb::open(data_dir()).unwrap();
        let table = db
            .compound_table("PSTAR_compound", "WATER_LIQUID")
            .expect("NIST water compound table should be loaded");
        assert!(table.0.len() > 10, "table should have >10 energy points");
        assert_eq!(
            table.0.len(),
            table.1.len(),
            "energy and dedx lengths must match"
        );
        // Energies should be sorted ascending
        for w in table.0.windows(2) {
            assert!(w[0] < w[1], "energies not sorted: {} >= {}", w[0], w[1]);
        }
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn compound_keys_contains_water() {
        let db = StoppingDb::open(data_dir()).unwrap();
        let keys: Vec<_> = db.compound_keys().collect();
        assert!(
            keys.iter()
                .any(|(s, c)| *s == "PSTAR_compound" && *c == "WATER_LIQUID"),
            "compound_keys() should contain PSTAR_compound/WATER_LIQUID, got {keys:?}"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn compound_table_anchor_water_10mev() {
        // NIST PSTAR compound water at 10 MeV: ~45.4 MeV cm²/g (ICRU-49).
        // This value differs from Bragg-mixed H+O because NIST uses the
        // compound I-value (75 eV for water vs 19.2/95.0 for H/O).
        let db = StoppingDb::open(data_dir()).unwrap();
        let val = db.compound_dedx("PSTAR_compound", "WATER_LIQUID", 10.0);
        assert!(val.is_finite());
        let rel = (val - 45.4).abs() / 45.4;
        assert!(
            rel < 0.02,
            "NIST water 10 MeV: got {val}, expected ~45.4, rel {rel}"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn nist_vs_bragg_water_divergence() {
        // NIST tabulated compound water (route 1) uses the measured compound
        // mean-excitation energy (I ≈ 75 eV); Bragg additivity over elemental
        // H + O (route 2) implicitly mixes the elemental I-values (H 19.2, O
        // 95 eV). Bragg therefore *overestimates* stopping, and the gap grows
        // toward low energy where the Bethe ln(2mₑc²β²/I) term dominates.
        //
        // Ground-truthed against the data (Python reference): ~18% at 0.1 MeV,
        // ~8% at 0.3 MeV, ~2.7% at 1 MeV, falling below 1.5% past 10 MeV.
        let db = StoppingDb::open(data_dir()).unwrap();

        // Bragg: water = 11.19% H + 88.81% O by mass.
        let divergence = |e: f64| {
            let nist = db.compound_dedx("PSTAR_compound", "WATER_LIQUID", e);
            let bragg = 0.1119 * db.dedx("PSTAR", 1, e) + 0.8881 * db.dedx("PSTAR", 8, e);
            assert!(nist.is_finite() && bragg.is_finite(), "non-finite at {e} MeV");
            (bragg - nist) / nist
        };

        let d_low = divergence(0.1);
        let d_mid = divergence(1.0);
        let d_high = divergence(10.0);

        // Bragg overestimates at every energy, substantially so at low energy.
        assert!(
            d_low > 0.10,
            "Bragg vs NIST at 0.1 MeV: {:.1}% (expected >10%)",
            d_low * 100.0
        );
        assert!(
            (0.01..0.05).contains(&d_mid),
            "Bragg vs NIST at 1 MeV: {:.1}% (expected ~2.7%)",
            d_mid * 100.0
        );
        // Monotone decrease in the I-value correction with energy.
        assert!(
            d_low > d_mid && d_mid > d_high && d_high > 0.0,
            "divergence should shrink monotonically with energy: \
             0.1 MeV {:.1}%, 1 MeV {:.1}%, 10 MeV {:.1}%",
            d_low * 100.0,
            d_mid * 100.0,
            d_high * 100.0,
        );
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
