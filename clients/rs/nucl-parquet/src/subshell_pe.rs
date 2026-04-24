//! EPDL97 subshell photoelectric cross-sections.
//!
//! Per-element, per-subshell photoelectric cross-sections and binding energies.
//! Data loaded from `meta/epdl97/subshell_pe/*.parquet`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::Error;
use crate::interp::{log_log_interp, sort_paired_vecs};

/// Subshell photoelectric cross-section database (EPDL97).
///
/// Thread-safe: `Send + Sync`. Share via `Arc<SubshellPeDb>`.
#[derive(Clone)]
pub struct SubshellPeDb {
    /// (Z, shell_index) -> (energies_MeV sorted, xs_barns sorted)
    xs_tables: HashMap<(u8, u8), (Vec<f64>, Vec<f64>)>,
    /// (Z, shell_index) -> binding edge energy [MeV]
    binding_energies: HashMap<(u8, u8), f64>,
    /// Z -> list of shell names (index matches shell_index)
    shell_names: HashMap<u8, Vec<String>>,
}

unsafe impl Send for SubshellPeDb {}
unsafe impl Sync for SubshellPeDb {}

impl SubshellPeDb {
    /// Load subshell photoelectric data from the nucl-parquet `meta/` directory.
    ///
    /// Reads `meta/epdl97/subshell_pe/*.parquet`.
    pub fn open(meta_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = meta_dir.as_ref().join("epdl97").join("subshell_pe");

        if !dir.exists() {
            return Err(Error::DataDirNotFound(dir.to_path_buf()));
        }

        let mut xs_tables = HashMap::new();
        let mut binding_energies = HashMap::new();
        let mut shell_names: HashMap<u8, Vec<String>> = HashMap::new();

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

            let mut shell_data: HashMap<String, (Vec<f64>, Vec<f64>, f64)> = HashMap::new();
            let mut z_val: Option<u8> = None;

            for batch in reader {
                let batch = batch?;

                let z_col = batch
                    .column_by_name("Z")
                    .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
                let e_col = batch
                    .column_by_name("energy_MeV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let shell_col_ref = batch.column_by_name("subshell");
                let shell_values = shell_col_ref.and_then(|c| crate::interp::as_string_array(c));
                let xs_col = batch
                    .column_by_name("xs_barns")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let edge_col = batch
                    .column_by_name("edge_MeV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                if let (Some(z), Some(e), Some(shell), Some(xs), Some(edge)) =
                    (z_col, e_col, shell_values, xs_col, edge_col)
                {
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..batch.num_rows() {
                        if z_val.is_none() {
                            z_val = Some(z.value(i) as u8);
                        }
                        let shell_name = shell[i].unwrap_or("").to_string();
                        let entry = shell_data
                            .entry(shell_name)
                            .or_insert_with(|| (Vec::new(), Vec::new(), edge.value(i)));
                        entry.0.push(e.value(i));
                        entry.1.push(xs.value(i));
                    }
                }
            }

            if let Some(z) = z_val {
                let mut names: Vec<String> = shell_data.keys().cloned().collect();
                names.sort();

                for (idx, name) in names.iter().enumerate() {
                    let (mut energies, mut xs, edge) = shell_data.remove(name).expect("key exists");
                    sort_paired_vecs(&mut energies, &mut xs);
                    let shell_idx = idx as u8;
                    xs_tables.insert((z, shell_idx), (energies, xs));
                    binding_energies.insert((z, shell_idx), edge);
                }
                shell_names.insert(z, names);
            }
        }

        Ok(Self {
            xs_tables,
            binding_energies,
            shell_names,
        })
    }

    /// Photoelectric cross-section [barns/atom] for a specific subshell.
    ///
    /// `shell` is the 0-based shell index (ordered alphabetically by shell name).
    /// Returns `f64::NAN` if the element or shell is not loaded.
    #[inline]
    pub fn cross_section(&self, z: u8, shell: u8, energy_mev: f64) -> f64 {
        match self.xs_tables.get(&(z, shell)) {
            Some((e, xs)) => log_log_interp(e, xs, energy_mev),
            None => f64::NAN,
        }
    }

    /// Binding energy [MeV] for a specific subshell.
    ///
    /// Returns `f64::NAN` if the element or shell is not loaded.
    #[inline]
    pub fn binding_energy(&self, z: u8, shell: u8) -> f64 {
        self.binding_energies
            .get(&(z, shell))
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// Number of subshells loaded for element Z.
    pub fn num_shells(&self, z: u8) -> usize {
        self.shell_names.get(&z).map(|v| v.len()).unwrap_or(0)
    }

    /// Shell name at the given index (e.g. "K", "L1", "M3").
    pub fn shell_name(&self, z: u8, shell: u8) -> Option<&str> {
        self.shell_names
            .get(&z)
            .and_then(|v| v.get(shell as usize))
            .map(|s| s.as_str())
    }

    /// Check if data is loaded for element Z.
    pub fn has_element(&self, z: u8) -> bool {
        self.shell_names.contains_key(&z)
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
    fn open_succeeds() {
        let db = SubshellPeDb::open(meta_dir()).unwrap();
        assert!(db.has_element(29)); // Cu
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_has_multiple_shells() {
        let db = SubshellPeDb::open(meta_dir()).unwrap();
        let n = db.num_shells(29);
        assert!(n > 1, "Cu should have multiple subshells, got {n}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_xs_positive() {
        let db = SubshellPeDb::open(meta_dir()).unwrap();
        // Shell 0 (alphabetically first, e.g. "K") at 0.1 MeV
        let xs = db.cross_section(29, 0, 0.1);
        assert!(xs.is_finite() && xs > 0.0, "Cu shell 0 XS at 0.1 MeV: {xs}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_binding_energy_positive() {
        let db = SubshellPeDb::open(meta_dir()).unwrap();
        let be = db.binding_energy(29, 0);
        assert!(
            be.is_finite() && be > 0.0,
            "Cu shell 0 binding energy: {be}"
        );
    }
}
