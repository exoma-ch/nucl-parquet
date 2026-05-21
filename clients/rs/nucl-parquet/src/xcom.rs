//! XCOM mass attenuation and energy-absorption coefficients.
//!
//! Provides µ/ρ and µ_en/ρ lookups for elements (by Z) and compounds (by name)
//! with log-log interpolation. Data loaded from `meta/xcom_elements.parquet`
//! and `meta/xcom_compounds.parquet`.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::interp::{log_log_interp, sort_paired_vecs, XYTable};

/// XCOM mass attenuation coefficient database.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<XcomDb>`.
#[derive(Clone)]
pub struct XcomDb {
    /// Z -> (energy_MeV sorted, mu_rho_cm2_g sorted)
    elem_mu_rho: HashMap<u8, XYTable>,
    /// Z -> (energy_MeV sorted, mu_en_rho_cm2_g sorted)
    elem_mu_en_rho: HashMap<u8, XYTable>,
    /// material name -> (energy_MeV sorted, mu_rho_cm2_g sorted)
    comp_mu_rho: HashMap<String, XYTable>,
    /// material name -> (energy_MeV sorted, mu_en_rho_cm2_g sorted)
    comp_mu_en_rho: HashMap<String, XYTable>,
}

unsafe impl Send for XcomDb {}
unsafe impl Sync for XcomDb {}

impl XcomDb {
    /// Load XCOM data from the nucl-parquet `meta/` directory.
    ///
    /// Reads `meta/xcom_elements.parquet` and `meta/xcom_compounds.parquet`.
    pub fn open(meta_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let meta = meta_dir.as_ref();

        let (elem_mu_rho, elem_mu_en_rho) =
            Self::load_elements(&meta.join("xcom_elements.parquet"))?;
        let (comp_mu_rho, comp_mu_en_rho) =
            Self::load_compounds(&meta.join("xcom_compounds.parquet"))?;

        Ok(Self {
            elem_mu_rho,
            elem_mu_en_rho,
            comp_mu_rho,
            comp_mu_en_rho,
        })
    }

    /// Construct from in-memory Parquet bytes.
    ///
    /// `elements_data` is the contents of `xcom_elements.parquet`.
    /// `compounds_data` is the contents of `xcom_compounds.parquet` (pass an
    /// empty slice to omit compounds).
    pub fn from_bytes(elements_data: &[u8], compounds_data: &[u8]) -> crate::Result<Self> {
        let (elem_mu_rho, elem_mu_en_rho) =
            Self::parse_elements(bytes::Bytes::from(elements_data.to_vec()))?;
        let (comp_mu_rho, comp_mu_en_rho) = if compounds_data.is_empty() {
            (HashMap::new(), HashMap::new())
        } else {
            Self::parse_compounds(bytes::Bytes::from(compounds_data.to_vec()))?
        };

        Ok(Self {
            elem_mu_rho,
            elem_mu_en_rho,
            comp_mu_rho,
            comp_mu_en_rho,
        })
    }

    /// Mass attenuation coefficient µ/ρ [cm²/g] for element Z.
    ///
    /// Returns `f64::NAN` if the element is not loaded.
    #[inline]
    pub fn mu_rho(&self, z: u8, energy_mev: f64) -> f64 {
        match self.elem_mu_rho.get(&z) {
            Some((e, v)) => log_log_interp(e, v, energy_mev),
            None => f64::NAN,
        }
    }

    /// Mass energy-absorption coefficient µ_en/ρ [cm²/g] for element Z.
    ///
    /// Returns `f64::NAN` if the element is not loaded.
    #[inline]
    pub fn mu_en_rho(&self, z: u8, energy_mev: f64) -> f64 {
        match self.elem_mu_en_rho.get(&z) {
            Some((e, v)) => log_log_interp(e, v, energy_mev),
            None => f64::NAN,
        }
    }

    /// Compound mass attenuation coefficient µ/ρ [cm²/g] by material name.
    ///
    /// Returns `f64::NAN` if the compound is not loaded.
    #[inline]
    pub fn compound_mu_rho(&self, compound: &str, energy_mev: f64) -> f64 {
        match self.comp_mu_rho.get(compound) {
            Some((e, v)) => log_log_interp(e, v, energy_mev),
            None => f64::NAN,
        }
    }

    /// Compound mass energy-absorption coefficient µ_en/ρ [cm²/g] by material name.
    ///
    /// Returns `f64::NAN` if the compound is not loaded.
    #[inline]
    pub fn compound_mu_en_rho(&self, compound: &str, energy_mev: f64) -> f64 {
        match self.comp_mu_en_rho.get(compound) {
            Some((e, v)) => log_log_interp(e, v, energy_mev),
            None => f64::NAN,
        }
    }

    /// Check if data is loaded for element Z.
    pub fn has_element(&self, z: u8) -> bool {
        self.elem_mu_rho.contains_key(&z)
    }

    /// List of loaded compound material names.
    pub fn compound_names(&self) -> Vec<&str> {
        self.comp_mu_rho.keys().map(|s| s.as_str()).collect()
    }

    // --- Internal loaders ---

    fn load_elements(path: &Path) -> crate::Result<(HashMap<u8, XYTable>, HashMap<u8, XYTable>)> {
        let file = fs::File::open(path)?;
        Self::parse_elements(file)
    }

    fn parse_elements(
        reader_source: impl parquet::file::reader::ChunkReader + 'static,
    ) -> crate::Result<(HashMap<u8, XYTable>, HashMap<u8, XYTable>)> {
        let mut mu_rho: HashMap<u8, XYTable> = HashMap::new();
        let mut mu_en: HashMap<u8, XYTable> = HashMap::new();

        let reader = ParquetRecordBatchReaderBuilder::try_new(reader_source)?.build()?;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let e_col = batch
                .column_by_name("energy_MeV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let mr_col = batch
                .column_by_name("mu_rho_cm2_g")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let men_col = batch
                .column_by_name("mu_en_rho_cm2_g")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(z), Some(e), Some(mr), Some(men)) = (z_col, e_col, mr_col, men_col) {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    let zv = z.value(i) as u8;
                    let ev = e.value(i);
                    let entry_mr = mu_rho.entry(zv).or_default();
                    entry_mr.0.push(ev);
                    entry_mr.1.push(mr.value(i));
                    let entry_men = mu_en.entry(zv).or_default();
                    entry_men.0.push(ev);
                    entry_men.1.push(men.value(i));
                }
            }
        }

        for (e_vec, v_vec) in mu_rho.values_mut() {
            sort_paired_vecs(e_vec, v_vec);
        }
        for (e_vec, v_vec) in mu_en.values_mut() {
            sort_paired_vecs(e_vec, v_vec);
        }

        Ok((mu_rho, mu_en))
    }

    fn load_compounds(
        path: &Path,
    ) -> crate::Result<(HashMap<String, XYTable>, HashMap<String, XYTable>)> {
        let file = fs::File::open(path)?;
        Self::parse_compounds(file)
    }

    fn parse_compounds(
        reader_source: impl parquet::file::reader::ChunkReader + 'static,
    ) -> crate::Result<(HashMap<String, XYTable>, HashMap<String, XYTable>)> {
        let mut mu_rho: HashMap<String, XYTable> = HashMap::new();
        let mut mu_en: HashMap<String, XYTable> = HashMap::new();

        let reader = ParquetRecordBatchReaderBuilder::try_new(reader_source)?.build()?;

        for batch in reader {
            let batch = batch?;

            let mat_col_ref = batch.column_by_name("material");
            let mat_values = mat_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let e_col = batch
                .column_by_name("energy_MeV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let mr_col = batch
                .column_by_name("mu_rho_cm2_g")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let men_col = batch
                .column_by_name("mu_en_rho_cm2_g")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(mat), Some(e), Some(mr), Some(men)) = (mat_values, e_col, mr_col, men_col)
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    let name = mat[i].unwrap_or("").to_string();
                    let ev = e.value(i);
                    let entry_mr = mu_rho.entry(name.clone()).or_default();
                    entry_mr.0.push(ev);
                    entry_mr.1.push(mr.value(i));
                    let entry_men = mu_en.entry(name).or_default();
                    entry_men.0.push(ev);
                    entry_men.1.push(men.value(i));
                }
            }
        }

        for (e_vec, v_vec) in mu_rho.values_mut() {
            sort_paired_vecs(e_vec, v_vec);
        }
        for (e_vec, v_vec) in mu_en.values_mut() {
            sort_paired_vecs(e_vec, v_vec);
        }

        Ok((mu_rho, mu_en))
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
        let db = XcomDb::open(meta_dir()).unwrap();
        assert!(db.has_element(29)); // Cu
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_mu_rho_positive() {
        let db = XcomDb::open(meta_dir()).unwrap();
        let mu = db.mu_rho(29, 0.1);
        assert!(mu.is_finite() && mu > 0.0, "Cu mu/rho at 0.1 MeV: {mu}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cu_mu_en_rho_positive() {
        let db = XcomDb::open(meta_dir()).unwrap();
        let mu = db.mu_en_rho(29, 0.1);
        assert!(mu.is_finite() && mu > 0.0, "Cu mu_en/rho at 0.1 MeV: {mu}");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn water_compound_exists() {
        let db = XcomDb::open(meta_dir()).unwrap();
        let names = db.compound_names();
        assert!(
            names.contains(&"water"),
            "water should be in compound list: {names:?}"
        );
        let mu = db.compound_mu_rho("water", 0.1);
        assert!(mu.is_finite() && mu > 0.0, "water mu/rho at 0.1 MeV: {mu}");
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
        let db_file = XcomDb::open(data_meta_dir()).unwrap();
        let elem_data = std::fs::read(data_meta_dir().join("xcom_elements.parquet")).unwrap();
        let comp_data = std::fs::read(data_meta_dir().join("xcom_compounds.parquet")).unwrap();
        let db_bytes = XcomDb::from_bytes(&elem_data, &comp_data).unwrap();
        let mu_file = db_file.mu_rho(29, 0.1);
        let mu_bytes = db_bytes.mu_rho(29, 0.1);
        assert!(
            (mu_file - mu_bytes).abs() < 1e-12,
            "Cu mu/rho mismatch: {mu_file} vs {mu_bytes}"
        );
        let mu_water_file = db_file.compound_mu_rho("water", 0.1);
        let mu_water_bytes = db_bytes.compound_mu_rho("water", 0.1);
        assert!(
            (mu_water_file - mu_water_bytes).abs() < 1e-12,
            "water mu/rho mismatch: {mu_water_file} vs {mu_water_bytes}"
        );
    }
}
