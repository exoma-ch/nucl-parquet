//! Nuclide metadata: isotopic abundances, radioactive decay, and dose constants.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int32Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;


// ---------------------------------------------------------------------------
// AbundancesDb
// ---------------------------------------------------------------------------

/// Natural isotopic abundance entry.
#[derive(Debug, Clone)]
pub struct AbundanceEntry {
    pub z: u32,
    pub a: u32,
    pub symbol: String,
    /// Natural isotopic abundance (fraction, 0–1). 0.0 for purely synthetic isotopes.
    pub abundance: f64,
    /// Atomic mass in unified atomic mass units (u).
    pub atomic_mass: f64,
}

/// Natural isotopic abundance database.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<AbundancesDb>`.
#[derive(Clone)]
pub struct AbundancesDb {
    data: HashMap<u32, Vec<AbundanceEntry>>,
}

unsafe impl Send for AbundancesDb {}
unsafe impl Sync for AbundancesDb {}

impl AbundancesDb {
    /// Load isotopic abundance data from `meta/abundances.parquet`.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let path = data_dir.as_ref().join("abundances.parquet");
        let mut data: HashMap<u32, Vec<AbundanceEntry>> = HashMap::new();

        let file = fs::File::open(&path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let a_col = batch
                .column_by_name("A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let sym_col = batch
                .column_by_name("symbol")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let ab_col = batch
                .column_by_name("abundance")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let mass_col = batch
                .column_by_name("atomic_mass")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(z), Some(a), Some(sym), Some(ab), Some(mass)) =
                (z_col, a_col, sym_col, ab_col, mass_col)
            {
                for i in 0..batch.num_rows() {
                    let entry = AbundanceEntry {
                        z: z.value(i) as u32,
                        a: a.value(i) as u32,
                        symbol: sym.value(i).to_string(),
                        abundance: ab.value(i),
                        atomic_mass: mass.value(i),
                    };
                    data.entry(entry.z).or_default().push(entry);
                }
            }
        }

        Ok(Self { data })
    }

    /// All isotopes of element Z, sorted by mass number.
    pub fn isotopes(&self, z: u32) -> &[AbundanceEntry] {
        self.data.get(&z).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Natural abundance (0–1) of isotope (Z, A). Returns 0.0 if not found.
    pub fn abundance(&self, z: u32, a: u32) -> f64 {
        self.isotopes(z)
            .iter()
            .find(|e| e.a == a)
            .map(|e| e.abundance)
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// DecayDb
// ---------------------------------------------------------------------------

/// A single radioactive decay branch.
#[derive(Debug, Clone)]
pub struct DecayEntry {
    pub z: u32,
    pub a: u32,
    /// Nuclear isomeric state (empty string for ground state).
    pub state: String,
    /// Half-life in seconds. `None` for stable nuclides or unknown.
    pub half_life_s: Option<f64>,
    /// Decay mode string (e.g. "beta-", "beta+", "alpha", "EC", "stable").
    pub decay_mode: String,
    /// Proton number of daughter nucleus. `None` for stable nuclides.
    pub daughter_z: Option<u32>,
    /// Mass number of daughter nucleus. `None` for stable nuclides.
    pub daughter_a: Option<u32>,
    /// Isomeric state of daughter nucleus.
    pub daughter_state: String,
    /// Branching fraction (0–1).
    pub branching: f64,
}

/// Radioactive decay database.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<DecayDb>`.
#[derive(Clone)]
pub struct DecayDb {
    /// (Z, A) -> decay entries
    data: HashMap<(u32, u32), Vec<DecayEntry>>,
}

unsafe impl Send for DecayDb {}
unsafe impl Sync for DecayDb {}

impl DecayDb {
    /// Load decay data from `meta/decay.parquet`.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let path = data_dir.as_ref().join("decay.parquet");
        let mut data: HashMap<(u32, u32), Vec<DecayEntry>> = HashMap::new();

        let file = fs::File::open(&path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let a_col = batch
                .column_by_name("A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let state_col = batch
                .column_by_name("state")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let hl_col = batch
                .column_by_name("half_life_s")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let mode_col = batch
                .column_by_name("decay_mode")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let dz_col = batch
                .column_by_name("daughter_Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let da_col = batch
                .column_by_name("daughter_A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let ds_col = batch
                .column_by_name("daughter_state")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let br_col = batch
                .column_by_name("branching")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (
                Some(z),
                Some(a),
                Some(state),
                Some(hl),
                Some(mode),
                Some(dz),
                Some(da),
                Some(ds),
                Some(br),
            ) = (
                z_col, a_col, state_col, hl_col, mode_col, dz_col, da_col, ds_col, br_col,
            ) {
                for i in 0..batch.num_rows() {
                    let z_val = z.value(i) as u32;
                    let a_val = a.value(i) as u32;
                    let entry = DecayEntry {
                        z: z_val,
                        a: a_val,
                        state: state.value(i).to_string(),
                        half_life_s: if hl.is_null(i) {
                            None
                        } else {
                            Some(hl.value(i))
                        },
                        decay_mode: mode.value(i).to_string(),
                        daughter_z: if dz.is_null(i) {
                            None
                        } else {
                            Some(dz.value(i) as u32)
                        },
                        daughter_a: if da.is_null(i) {
                            None
                        } else {
                            Some(da.value(i) as u32)
                        },
                        daughter_state: ds.value(i).to_string(),
                        branching: br.value(i),
                    };
                    data.entry((z_val, a_val)).or_default().push(entry);
                }
            }
        }

        Ok(Self { data })
    }

    /// All decay branches for a given nuclide (Z, A, state).
    pub fn modes(&self, z: u32, a: u32, state: &str) -> Vec<&DecayEntry> {
        self.data
            .get(&(z, a))
            .map(|entries| {
                entries
                    .iter()
                    .filter(|e| e.state == state)
                    .collect()
            })
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// DoseDb
// ---------------------------------------------------------------------------

/// Dose rate constant database (µSv·m²·MBq⁻¹·h⁻¹).
///
/// Thread-safe: `Send + Sync`. Share via `Arc<DoseDb>`.
#[derive(Clone)]
pub struct DoseDb {
    /// (Z, A, state) -> dose constant in µSv·m²/(MBq·h)
    data: HashMap<(u32, u32, String), f64>,
}

unsafe impl Send for DoseDb {}
unsafe impl Sync for DoseDb {}

impl DoseDb {
    /// Load dose constant data from `meta/dose_constants.parquet`.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let path = data_dir.as_ref().join("dose_constants.parquet");
        let mut data: HashMap<(u32, u32, String), f64> = HashMap::new();

        let file = fs::File::open(&path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let a_col = batch
                .column_by_name("A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let state_col = batch
                .column_by_name("state")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let k_col = batch
                .column_by_name("k_uSv_m2_MBq_h")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(z), Some(a), Some(state), Some(k)) = (z_col, a_col, state_col, k_col) {
                for i in 0..batch.num_rows() {
                    data.insert(
                        (z.value(i) as u32, a.value(i) as u32, state.value(i).to_string()),
                        k.value(i),
                    );
                }
            }
        }

        Ok(Self { data })
    }

    /// Dose rate constant [µSv·m²/(MBq·h)] for nuclide (Z, A, state).
    ///
    /// Returns `None` if the nuclide is not in the database.
    /// Returns `Some(0.0)` for stable nuclides or those with no gamma emission.
    pub fn dose_constant(&self, z: u32, a: u32, state: &str) -> Option<f64> {
        self.data.get(&(z, a, state.to_string())).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("meta")
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn abundances_cu_isotopes() {
        let db = AbundancesDb::open(meta_dir()).unwrap();
        let isotopes = db.isotopes(29); // Cu
        // Cu has 2 stable isotopes: Cu-63 and Cu-65
        let stable: Vec<_> = isotopes.iter().filter(|e| e.abundance > 0.0).collect();
        assert_eq!(stable.len(), 2, "Cu stable isotopes: {}", stable.len());
        let total: f64 = stable.iter().map(|e| e.abundance).sum();
        assert!((total - 1.0).abs() < 1e-3, "Cu abundance sum: {total}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn decay_cu64_beta() {
        let db = DecayDb::open(meta_dir()).unwrap();
        let modes = db.modes(29, 64, ""); // Cu-64 ground state
        assert!(!modes.is_empty(), "Cu-64 should have decay modes");
        let has_beta = modes
            .iter()
            .any(|m| m.decay_mode.starts_with("beta"));
        assert!(has_beta, "Cu-64 should have beta decay");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn dose_i131_positive() {
        let db = DoseDb::open(meta_dir()).unwrap();
        let k = db.dose_constant(53, 131, ""); // I-131
        assert!(k.is_some(), "I-131 should have a dose constant");
        assert!(k.unwrap() > 0.0, "I-131 dose constant should be positive");
    }
}
