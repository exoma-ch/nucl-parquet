//! EADL atomic relaxation data (fluorescence X-rays and Auger electrons).
//!
//! After a photoelectric interaction creates a vacancy in an inner shell,
//! the atom relaxes via radiative (X-ray) or non-radiative (Auger) transitions.
//! This module provides the transition probabilities and energies needed to
//! sample the resulting secondary particles.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// Type of atomic relaxation transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionType {
    /// Radiative transition — emits a characteristic X-ray photon.
    Radiative,
    /// Auger transition — emits an electron.
    Auger,
}

/// A single atomic relaxation transition.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Shell where the vacancy was created (e.g., "K", "L1").
    pub vacancy_shell: String,
    /// Shell that fills the vacancy (e.g., "L3").
    pub filling_shell: String,
    /// Radiative (X-ray) or Auger (electron).
    pub transition_type: TransitionType,
    /// Transition energy in keV.
    pub energy_kev: f64,
    /// Transition probability (fractional, sums to ~1 per shell).
    pub probability: f64,
    /// Binding energy of the vacancy shell in keV.
    pub edge_kev: f64,
}

/// EADL atomic relaxation database.
///
/// Thread-safe: `Send + Sync`.
pub struct RelaxationDb {
    /// Z -> list of transitions (sorted by vacancy_shell, then probability desc)
    transitions: HashMap<u8, Vec<Transition>>,
}

impl RelaxationDb {
    /// Load EADL data from the nucl-parquet `meta/` directory.
    ///
    /// Reads `meta/eadl/*.parquet`.
    pub fn open(meta_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let dir = meta_dir.as_ref().join("eadl");
        let mut transitions: HashMap<u8, Vec<Transition>> = HashMap::new();

        if !dir.exists() {
            return Ok(Self { transitions });
        }

        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

            let mut z_val: Option<u8> = None;
            let mut trans_list = Vec::new();

            for batch in reader {
                let batch = batch?;

                let z_col = batch
                    .column_by_name("Z")
                    .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
                let vacancy_col = batch
                    .column_by_name("vacancy_shell")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let filling_col = batch
                    .column_by_name("filling_shell")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let type_col = batch
                    .column_by_name("transition_type")
                    .and_then(|c| c.as_any().downcast_ref::<StringArray>());
                let energy_col = batch
                    .column_by_name("energy_keV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let prob_col = batch
                    .column_by_name("probability")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
                let edge_col = batch
                    .column_by_name("edge_keV")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                if let (Some(z), Some(vac), Some(fill), Some(tt), Some(e), Some(p), Some(edge)) = (
                    z_col,
                    vacancy_col,
                    filling_col,
                    type_col,
                    energy_col,
                    prob_col,
                    edge_col,
                ) {
                    for i in 0..batch.num_rows() {
                        if z_val.is_none() {
                            z_val = Some(z.value(i) as u8);
                        }
                        trans_list.push(Transition {
                            vacancy_shell: vac.value(i).to_string(),
                            filling_shell: fill.value(i).to_string(),
                            transition_type: match tt.value(i) {
                                "radiative" => TransitionType::Radiative,
                                _ => TransitionType::Auger,
                            },
                            energy_kev: e.value(i),
                            probability: p.value(i),
                            edge_kev: edge.value(i),
                        });
                    }
                }
            }

            if let Some(z) = z_val {
                // Sort by shell, then probability descending
                trans_list.sort_by(|a, b| {
                    a.vacancy_shell
                        .cmp(&b.vacancy_shell)
                        .then(b.probability.partial_cmp(&a.probability).unwrap())
                });
                transitions.insert(z, trans_list);
            }
        }

        Ok(Self { transitions })
    }

    /// Get all transitions for element Z.
    pub fn transitions(&self, z: u8) -> &[Transition] {
        self.transitions
            .get(&z)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get transitions for a specific vacancy shell (e.g., "K").
    pub fn shell_transitions(&self, z: u8, shell: &str) -> Vec<&Transition> {
        self.transitions(z)
            .iter()
            .filter(|t| t.vacancy_shell == shell)
            .collect()
    }

    /// Get only radiative (X-ray) transitions for element Z and shell.
    pub fn radiative_transitions(&self, z: u8, shell: &str) -> Vec<&Transition> {
        self.transitions(z)
            .iter()
            .filter(|t| t.vacancy_shell == shell && t.transition_type == TransitionType::Radiative)
            .collect()
    }

    /// Fluorescence yield for a shell = sum of radiative transition probabilities.
    pub fn fluorescence_yield(&self, z: u8, shell: &str) -> f64 {
        self.transitions(z)
            .iter()
            .filter(|t| t.vacancy_shell == shell && t.transition_type == TransitionType::Radiative)
            .map(|t| t.probability)
            .sum()
    }

    /// Check if data is loaded for element Z.
    pub fn has_element(&self, z: u8) -> bool {
        self.transitions.contains_key(&z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn load_and_query_cu() {
        let db = RelaxationDb::open("../../meta").unwrap();
        assert!(db.has_element(29));

        let k_trans = db.shell_transitions(29, "K");
        assert!(!k_trans.is_empty());

        // K fluorescence yield for Cu should be ~0.44
        let fy = db.fluorescence_yield(29, "K");
        assert!(fy > 0.3 && fy < 0.6, "Cu K fluorescence yield: {fy}");

        // K-alpha line should be ~8 keV
        let radiative = db.radiative_transitions(29, "K");
        let strongest = radiative
            .iter()
            .max_by(|a, b| a.probability.partial_cmp(&b.probability).unwrap());
        assert!(strongest.is_some());
        let ka = strongest.unwrap();
        assert!(
            ka.energy_kev > 7.0 && ka.energy_kev < 9.0,
            "Cu K-alpha: {} keV",
            ka.energy_kev
        );
    }
}
