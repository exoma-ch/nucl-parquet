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
            Self::parse_one_into(
                file,
                &mut xs_tables,
                &mut binding_energies,
                &mut shell_names,
            )?;
        }

        Ok(Self {
            xs_tables,
            binding_energies,
            shell_names,
        })
    }

    /// Construct from a single element's subshell PE Parquet bytes.
    ///
    /// The element Z is determined from the `Z` column in the data.
    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        let bytes = bytes::Bytes::from(data.to_vec());
        let mut xs_tables = HashMap::new();
        let mut binding_energies = HashMap::new();
        let mut shell_names: HashMap<u8, Vec<String>> = HashMap::new();
        Self::parse_one_into(
            bytes,
            &mut xs_tables,
            &mut binding_energies,
            &mut shell_names,
        )?;
        Ok(Self {
            xs_tables,
            binding_energies,
            shell_names,
        })
    }

    #[allow(clippy::type_complexity)]
    fn parse_one_into(
        reader_source: impl parquet::file::reader::ChunkReader + 'static,
        xs_tables: &mut HashMap<(u8, u8), (Vec<f64>, Vec<f64>)>,
        binding_energies: &mut HashMap<(u8, u8), f64>,
        shell_names: &mut HashMap<u8, Vec<String>>,
    ) -> crate::Result<()> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(reader_source)?.build()?;

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

        Ok(())
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

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn open_succeeds() {
        let db = SubshellPeDb::open(data_meta_dir()).unwrap();
        assert!(db.has_element(29)); // Cu
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_has_multiple_shells() {
        let db = SubshellPeDb::open(data_meta_dir()).unwrap();
        let n = db.num_shells(29);
        assert!(n > 1, "Cu should have multiple subshells, got {n}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_xs_positive() {
        let db = SubshellPeDb::open(data_meta_dir()).unwrap();
        // Shell 0 (alphabetically first, e.g. "K") at 0.1 MeV
        let xs = db.cross_section(29, 0, 0.1);
        assert!(xs.is_finite() && xs > 0.0, "Cu shell 0 XS at 0.1 MeV: {xs}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_binding_energy_positive() {
        let db = SubshellPeDb::open(data_meta_dir()).unwrap();
        let be = db.binding_energy(29, 0);
        assert!(
            be.is_finite() && be > 0.0,
            "Cu shell 0 binding energy: {be}"
        );
    }

    fn data_meta_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
            .join("meta")
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn from_bytes_matches_open() {
        let db_file = SubshellPeDb::open(data_meta_dir()).unwrap();
        let pe_dir = data_meta_dir().join("epdl97").join("subshell_pe");
        let first_file = std::fs::read_dir(&pe_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.path().extension().and_then(|x| x.to_str()) == Some("parquet"))
            .expect("at least one subshell_pe file");
        let data = std::fs::read(first_file.path()).unwrap();
        let db_bytes = SubshellPeDb::from_bytes(&data).unwrap();
        for z in 1..=100u8 {
            if db_bytes.has_element(z) {
                assert_eq!(
                    db_bytes.num_shells(z),
                    db_file.num_shells(z),
                    "Z={z} shell count mismatch"
                );
                let xs_file = db_file.cross_section(z, 0, 0.1);
                let xs_bytes = db_bytes.cross_section(z, 0, 0.1);
                assert!(
                    (xs_file - xs_bytes).abs() < 1e-12,
                    "Z={z} shell 0 XS mismatch: {xs_file} vs {xs_bytes}"
                );
                break;
            }
        }
    }
}
