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

use crate::interp::{log_log_interp, sort_paired_vecs};

/// XCOM mass attenuation coefficient database.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<XcomDb>`.
#[derive(Clone)]
pub struct XcomDb {
    /// Z -> (energy_MeV sorted, mu_rho_cm2_g sorted)
    elem_mu_rho: HashMap<u8, (Vec<f64>, Vec<f64>)>,
    /// Z -> (energy_MeV sorted, mu_en_rho_cm2_g sorted)
    elem_mu_en_rho: HashMap<u8, (Vec<f64>, Vec<f64>)>,
    /// material name -> (energy_MeV sorted, mu_rho_cm2_g sorted)
    comp_mu_rho: HashMap<String, (Vec<f64>, Vec<f64>)>,
    /// material name -> (energy_MeV sorted, mu_en_rho_cm2_g sorted)
    comp_mu_en_rho: HashMap<String, (Vec<f64>, Vec<f64>)>,
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

    fn load_elements(
        path: &Path,
    ) -> crate::Result<(
        HashMap<u8, (Vec<f64>, Vec<f64>)>,
        HashMap<u8, (Vec<f64>, Vec<f64>)>,
    )> {
        let mut mu_rho: HashMap<u8, (Vec<f64>, Vec<f64>)> = HashMap::new();
        let mut mu_en: HashMap<u8, (Vec<f64>, Vec<f64>)> = HashMap::new();

        let file = fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

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
    ) -> crate::Result<(
        HashMap<String, (Vec<f64>, Vec<f64>)>,
        HashMap<String, (Vec<f64>, Vec<f64>)>,
    )> {
        let mut mu_rho: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();
        let mut mu_en: HashMap<String, (Vec<f64>, Vec<f64>)> = HashMap::new();

        let file = fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

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
}
