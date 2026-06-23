//! EEDL electron interaction cross-sections.
//!
//! Per-element electron cross-sections for elastic scattering, bremsstrahlung,
//! excitation, and ionization processes. Data loaded from `meta/eedl/*.parquet`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::Error;
use crate::interp::{log_log_interp, sort_paired_vecs};

/// Electron interaction process types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElectronProcess {
    /// Elastic scattering
    Elastic,
    /// Bremsstrahlung radiation
    Bremsstrahlung,
    /// Excitation
    Excitation,
    /// Ionization (total — sum of all subshell ionization)
    Ionization,
}

/// EEDL electron interaction database.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<ElectronDb>`.
#[derive(Clone)]
pub struct ElectronDb {
    /// (Z, ElectronProcess) -> (energies_MeV sorted, xs_barns sorted)
    xs_tables: HashMap<(u8, ElectronProcess), (Vec<f64>, Vec<f64>)>,
}

unsafe impl Send for ElectronDb {}
unsafe impl Sync for ElectronDb {}

impl ElectronDb {
    /// Load EEDL data from the nucl-parquet `meta/` directory.
    ///
    /// Reads `meta/eedl/*.parquet`.
    pub fn open(meta_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = meta_dir.as_ref().join("eedl");

        if !dir.exists() {
            return Err(Error::DataDirNotFound(dir.to_path_buf()));
        }

        let mut xs_tables = HashMap::new();

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            Self::parse_one_into(file, &mut xs_tables)?;
        }

        Ok(Self { xs_tables })
    }

    /// Construct from a single element's EEDL Parquet bytes.
    ///
    /// The element Z is determined from the `Z` column in the data.
    pub fn from_bytes(data: &[u8]) -> crate::Result<Self> {
        let bytes = bytes::Bytes::from(data.to_vec());
        let mut xs_tables = HashMap::new();
        Self::parse_one_into(bytes, &mut xs_tables)?;
        Ok(Self { xs_tables })
    }

    #[allow(clippy::type_complexity)]
    fn parse_one_into(
        reader_source: impl parquet::file::reader::ChunkReader + 'static,
        xs_tables: &mut HashMap<(u8, ElectronProcess), (Vec<f64>, Vec<f64>)>,
    ) -> crate::Result<()> {
        let reader = ParquetRecordBatchReaderBuilder::try_new(reader_source)?.build()?;

        let mut process_data: HashMap<ElectronProcess, (Vec<f64>, Vec<f64>)> = HashMap::new();
        let mut z_val: Option<u8> = None;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let e_col = batch
                .column_by_name("energy_MeV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let proc_col_ref = batch.column_by_name("process");
            let proc_values = proc_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let xs_col = batch
                .column_by_name("xs_barns")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(z), Some(e), Some(proc), Some(xs)) = (z_col, e_col, proc_values, xs_col) {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    if z_val.is_none() {
                        z_val = Some(z.value(i) as u8);
                    }
                    let proc_str = match proc[i] {
                        Some(s) => s,
                        None => continue,
                    };
                    let process = match proc_str {
                        "elastic" => Some(ElectronProcess::Elastic),
                        "bremsstrahlung" => Some(ElectronProcess::Bremsstrahlung),
                        "excitation" => Some(ElectronProcess::Excitation),
                        _ if proc_str.starts_with("ionization") => {
                            Some(ElectronProcess::Ionization)
                        }
                        _ => None,
                    };
                    if let Some(p) = process {
                        let entry = process_data.entry(p).or_default();
                        entry.0.push(e.value(i));
                        entry.1.push(xs.value(i));
                    }
                }
            }
        }

        if let Some(z) = z_val {
            // Subshell ionization rows share the same energy grid; sum them.
            if let Some((energies, xs_vals)) = process_data.remove(&ElectronProcess::Ionization) {
                let summed = sum_by_energy(&energies, &xs_vals);
                process_data.insert(ElectronProcess::Ionization, summed);
            }

            for (process, (mut energies, mut xs)) in process_data {
                sort_paired_vecs(&mut energies, &mut xs);
                xs_tables.insert((z, process), (energies, xs));
            }
        }

        Ok(())
    }

    /// Cross-section [barns/atom] for element Z at energy E [MeV] for a given process.
    ///
    /// Returns `f64::NAN` if the element or process is not loaded.
    #[inline]
    pub fn cross_section(&self, z: u8, energy_mev: f64, process: ElectronProcess) -> f64 {
        match self.xs_tables.get(&(z, process)) {
            Some((e, xs)) => log_log_interp(e, xs, energy_mev),
            None => f64::NAN,
        }
    }

    /// Total cross-section [barns/atom] (elastic + bremsstrahlung + excitation + ionization).
    ///
    /// Returns `f64::NAN` if no data is loaded for the element.
    #[inline]
    pub fn total_cross_section(&self, z: u8, energy_mev: f64) -> f64 {
        let processes = [
            ElectronProcess::Elastic,
            ElectronProcess::Bremsstrahlung,
            ElectronProcess::Excitation,
            ElectronProcess::Ionization,
        ];
        let mut total = 0.0;
        let mut found = false;
        for p in &processes {
            let xs = self.cross_section(z, energy_mev, *p);
            if xs.is_finite() {
                total += xs;
                found = true;
            }
        }
        if found {
            total
        } else {
            f64::NAN
        }
    }

    /// Check if data is loaded for element Z.
    pub fn has_element(&self, z: u8) -> bool {
        self.xs_tables.contains_key(&(z, ElectronProcess::Elastic))
    }

    /// Number of elements loaded.
    pub fn num_elements(&self) -> usize {
        self.xs_tables
            .keys()
            .filter(|(_, p)| *p == ElectronProcess::Elastic)
            .count()
    }
}

/// Sum cross-section values that share the same energy.
///
/// EEDL ionization data has separate rows per subshell at the same energy grid.
/// This function groups by energy and sums the cross-sections.
fn sum_by_energy(energies: &[f64], values: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut map: HashMap<u64, f64> = HashMap::new();
    let mut order: Vec<u64> = Vec::new();

    for (&e, &v) in energies.iter().zip(values.iter()) {
        let key = e.to_bits();
        let entry = map.entry(key).or_insert_with(|| {
            order.push(key);
            0.0
        });
        *entry += v;
    }

    let mut result_e: Vec<f64> = Vec::with_capacity(order.len());
    let mut result_v: Vec<f64> = Vec::with_capacity(order.len());
    for key in &order {
        result_e.push(f64::from_bits(*key));
        result_v.push(map[key]);
    }

    (result_e, result_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn open_succeeds() {
        let db = ElectronDb::open(data_meta_dir()).unwrap();
        assert!(db.has_element(29)); // Cu
        assert!(db.num_elements() > 0);
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_elastic_xs_positive() {
        let db = ElectronDb::open(data_meta_dir()).unwrap();
        let xs = db.cross_section(29, 1.0, ElectronProcess::Elastic);
        assert!(xs.is_finite() && xs > 0.0, "Cu elastic XS at 1 MeV: {xs}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_total_xs_positive() {
        let db = ElectronDb::open(data_meta_dir()).unwrap();
        let total = db.total_cross_section(29, 1.0);
        assert!(
            total.is_finite() && total > 0.0,
            "Cu total electron XS at 1 MeV: {total}"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_bremsstrahlung_positive() {
        let db = ElectronDb::open(data_meta_dir()).unwrap();
        let xs = db.cross_section(29, 1.0, ElectronProcess::Bremsstrahlung);
        assert!(
            xs.is_finite() && xs > 0.0,
            "Cu bremsstrahlung XS at 1 MeV: {xs}"
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
        let db_file = ElectronDb::open(data_meta_dir()).unwrap();
        let eedl_dir = data_meta_dir().join("eedl");
        let first_file = std::fs::read_dir(&eedl_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| e.path().extension().and_then(|x| x.to_str()) == Some("parquet"))
            .expect("at least one EEDL file");
        let data = std::fs::read(first_file.path()).unwrap();
        let db_bytes = ElectronDb::from_bytes(&data).unwrap();
        // Find the Z loaded and compare
        for z in 1..=100u8 {
            if db_bytes.has_element(z) {
                let xs_file = db_file.cross_section(z, 1.0, ElectronProcess::Elastic);
                let xs_bytes = db_bytes.cross_section(z, 1.0, ElectronProcess::Elastic);
                assert!(
                    (xs_file - xs_bytes).abs() < 1e-12,
                    "Z={z} elastic XS mismatch: {xs_file} vs {xs_bytes}"
                );
                break;
            }
        }
    }
}
