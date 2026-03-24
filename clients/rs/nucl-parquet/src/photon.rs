//! EPDL97 photon interaction cross-sections.
//!
//! Provides per-element, per-process cross-section lookups with log-log
//! interpolation. All data is loaded at construction time into contiguous
//! arrays. The resulting `PhotonDb` is `Send + Sync` and can be shared
//! across threads via `Arc` with zero synchronization overhead.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Float64Array, Int32Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::Error;
use crate::interp::log_log_interp;

/// Photon interaction process types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Process {
    /// Total cross-section (sum of all processes)
    Total,
    /// Coherent (Rayleigh) scattering
    Coherent,
    /// Incoherent (Compton) scattering
    Incoherent,
    /// Photoelectric absorption (total)
    Photoelectric,
    /// Pair production — nuclear field
    PairNuclear,
    /// Pair production — electron field
    PairElectron,
    /// Pair production — total (nuclear + electron)
    PairTotal,
}

impl Process {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "total" => Some(Self::Total),
            "coherent" => Some(Self::Coherent),
            "incoherent" => Some(Self::Incoherent),
            "photoelectric" => Some(Self::Photoelectric),
            "pair_nuclear" => Some(Self::PairNuclear),
            "pair_electron" => Some(Self::PairElectron),
            "pair_total" => Some(Self::PairTotal),
            _ => None,
        }
    }
}

/// Sorted energy/cross-section table for a single (element, process) pair.
#[derive(Debug, Clone)]
struct XsTable {
    /// Energies in MeV, sorted ascending.
    energy: Vec<f64>,
    /// Cross-sections in barns/atom, same length as `energy`.
    xs: Vec<f64>,
}

/// Sorted momentum-transfer / value table for form factors or scattering functions.
#[derive(Debug, Clone)]
struct TabTable {
    x: Vec<f64>,
    y: Vec<f64>,
}

/// Photon interaction database loaded from EPDL97 Parquet files.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<PhotonDb>` across rayon threads.
///
/// # Example
///
/// ```no_run
/// use nucl_parquet::{PhotonDb, Process};
///
/// let db = PhotonDb::open("path/to/nucl-parquet/meta")?;
/// let sigma = db.cross_section(29, 0.511, Process::Photoelectric); // barns/atom
/// let ff = db.form_factor(29, 0.5); // dimensionless
/// # Ok::<(), nucl_parquet::Error>(())
/// ```
pub struct PhotonDb {
    /// (Z, Process) -> XsTable
    xs_tables: HashMap<(u8, Process), XsTable>,
    /// Z -> form factor table (momentum_transfer -> FF)
    form_factors: HashMap<u8, TabTable>,
    /// Z -> incoherent scattering function table
    scattering_fns: HashMap<u8, TabTable>,
}

// Safety: all data is immutable after construction.
unsafe impl Send for PhotonDb {}
unsafe impl Sync for PhotonDb {}

impl PhotonDb {
    /// Load EPDL97 data from the nucl-parquet `meta/` directory.
    ///
    /// Reads:
    /// - `meta/epdl97/photon_xs/*.parquet` — per-process cross-sections
    /// - `meta/epdl97/form_factors/*.parquet` — Rayleigh form factors
    /// - `meta/epdl97/scattering_fn/*.parquet` — Compton scattering functions
    pub fn open(meta_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let meta = meta_dir.as_ref();

        let xs_dir = meta.join("epdl97").join("photon_xs");
        let ff_dir = meta.join("epdl97").join("form_factors");
        let sf_dir = meta.join("epdl97").join("scattering_fn");

        let xs_tables = Self::load_xs_tables(&xs_dir)?;
        let form_factors = Self::load_tab_tables(&ff_dir, "form_factor")?;
        let scattering_fns = Self::load_tab_tables(&sf_dir, "scattering_fn")?;

        Ok(Self {
            xs_tables,
            form_factors,
            scattering_fns,
        })
    }

    /// Interpolate cross-section [barns/atom] for element Z at energy E [MeV].
    ///
    /// Returns 0.0 if the element or process is not in the database.
    /// Uses log-log interpolation on the EPDL97 energy grid.
    #[inline]
    pub fn cross_section(&self, z: u8, energy_mev: f64, process: Process) -> f64 {
        match self.xs_tables.get(&(z, process)) {
            Some(table) => log_log_interp(&table.energy, &table.xs, energy_mev),
            None => 0.0,
        }
    }

    /// All process cross-sections for element Z at energy E [MeV].
    ///
    /// Returns (photoelectric, compton, rayleigh, pair_total) in barns/atom.
    #[inline]
    pub fn all_cross_sections(&self, z: u8, energy_mev: f64) -> [f64; 4] {
        [
            self.cross_section(z, energy_mev, Process::Photoelectric),
            self.cross_section(z, energy_mev, Process::Incoherent),
            self.cross_section(z, energy_mev, Process::Coherent),
            self.cross_section(z, energy_mev, Process::PairTotal),
        ]
    }

    /// Total cross-section [barns/atom] for element Z at energy E [MeV].
    #[inline]
    pub fn total_cross_section(&self, z: u8, energy_mev: f64) -> f64 {
        self.cross_section(z, energy_mev, Process::Total)
    }

    /// Rayleigh atomic form factor for element Z at momentum transfer q.
    ///
    /// q = sin(θ/2) / λ in units matching the EPDL97 table (typically 1/cm).
    /// Returns Z (atomic number) at q=0.
    #[inline]
    pub fn form_factor(&self, z: u8, q: f64) -> f64 {
        match self.form_factors.get(&z) {
            Some(table) => log_log_interp(&table.x, &table.y, q),
            None => 0.0,
        }
    }

    /// Compton incoherent scattering function S(q) for element Z.
    ///
    /// Ranges from 0 (forward) to Z (backward scattering).
    #[inline]
    pub fn scattering_function(&self, z: u8, q: f64) -> f64 {
        match self.scattering_fns.get(&z) {
            Some(table) => log_log_interp(&table.x, &table.y, q),
            None => 0.0,
        }
    }

    /// Check if data is loaded for element Z.
    pub fn has_element(&self, z: u8) -> bool {
        self.xs_tables.contains_key(&(z, Process::Total))
    }

    /// Number of elements loaded.
    pub fn num_elements(&self) -> usize {
        self.xs_tables
            .keys()
            .filter(|(_, p)| *p == Process::Total)
            .count()
    }

    // --- Internal loading ---

    fn load_xs_tables(dir: &Path) -> crate::Result<HashMap<(u8, Process), XsTable>> {
        let mut tables = HashMap::new();

        if !dir.exists() {
            return Err(Error::DataDirNotFound(dir.to_path_buf()));
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

            // Accumulate rows per process
            let mut process_data: HashMap<Process, (Vec<f64>, Vec<f64>)> = HashMap::new();
            let mut z_val: Option<u8> = None;

            for batch in reader {
                let batch = batch?;

                let z_col = batch
                    .column_by_name("Z")
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "Z".into(),
                    })?
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "Z (wrong type)".into(),
                    })?;

                let e_col = batch
                    .column_by_name("energy_MeV")
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "energy_MeV".into(),
                    })?
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "energy_MeV (wrong type)".into(),
                    })?;

                let proc_col = batch
                    .column_by_name("process")
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "process".into(),
                    })?
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "process (wrong type)".into(),
                    })?;

                let xs_col = batch
                    .column_by_name("xs_barns")
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "xs_barns".into(),
                    })?
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| Error::MissingColumn {
                        file: path.clone(),
                        column: "xs_barns (wrong type)".into(),
                    })?;

                for i in 0..batch.num_rows() {
                    if z_val.is_none() {
                        z_val = Some(z_col.value(i) as u8);
                    }

                    let proc_str = proc_col.value(i);
                    if let Some(process) = Process::from_str(proc_str) {
                        let entry = process_data.entry(process).or_default();
                        entry.0.push(e_col.value(i));
                        entry.1.push(xs_col.value(i));
                    }
                }
            }

            if let Some(z) = z_val {
                for (process, (mut energy, mut xs)) in process_data {
                    // Sort by energy (should already be sorted, but ensure)
                    let mut indices: Vec<usize> = (0..energy.len()).collect();
                    indices.sort_by(|&a, &b| energy[a].partial_cmp(&energy[b]).unwrap());
                    let sorted_e: Vec<f64> = indices.iter().map(|&i| energy[i]).collect();
                    let sorted_xs: Vec<f64> = indices.iter().map(|&i| xs[i]).collect();
                    energy = sorted_e;
                    xs = sorted_xs;

                    tables.insert((z, process), XsTable { energy, xs });
                }
            }
        }

        Ok(tables)
    }

    fn load_tab_tables(dir: &Path, value_col_name: &str) -> crate::Result<HashMap<u8, TabTable>> {
        let mut tables = HashMap::new();

        if !dir.exists() {
            return Ok(tables); // optional data
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("parquet") {
                continue;
            }

            let file = fs::File::open(&path)?;
            let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

            let mut x_vals = Vec::new();
            let mut y_vals = Vec::new();
            let mut z_val: Option<u8> = None;

            for batch in reader {
                let batch = batch?;

                let z_col = batch
                    .column_by_name("Z")
                    .and_then(|c| c.as_any().downcast_ref::<Int32Array>());

                let x_col = batch
                    .column_by_name("momentum_transfer")
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                let y_col = batch
                    .column_by_name(value_col_name)
                    .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

                if let (Some(z), Some(x), Some(y)) = (z_col, x_col, y_col) {
                    for i in 0..batch.num_rows() {
                        if z_val.is_none() {
                            z_val = Some(z.value(i) as u8);
                        }
                        x_vals.push(x.value(i));
                        y_vals.push(y.value(i));
                    }
                }
            }

            if let Some(z) = z_val {
                // Sort by x
                let mut indices: Vec<usize> = (0..x_vals.len()).collect();
                indices.sort_by(|&a, &b| x_vals[a].partial_cmp(&x_vals[b]).unwrap());
                let sorted_x: Vec<f64> = indices.iter().map(|&i| x_vals[i]).collect();
                let sorted_y: Vec<f64> = indices.iter().map(|&i| y_vals[i]).collect();

                tables.insert(
                    z,
                    TabTable {
                        x: sorted_x,
                        y: sorted_y,
                    },
                );
            }
        }

        Ok(tables)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Integration tests require actual data files.
    // Run with: cargo test -- --ignored
    // from the nucl-parquet repo root.

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn load_and_query() {
        let db = PhotonDb::open("../../meta").unwrap();
        assert!(db.num_elements() >= 90);
        assert!(db.has_element(29)); // Cu

        // Cu photoelectric at 511 keV — should be ~0.27 barns
        let pe = db.cross_section(29, 0.511, Process::Photoelectric);
        assert!(pe > 0.1 && pe < 1.0, "Cu PE at 511 keV: {pe}");

        // Cu Compton at 511 keV — should be ~8.3 barns
        let compton = db.cross_section(29, 0.511, Process::Incoherent);
        assert!(
            compton > 5.0 && compton < 15.0,
            "Cu Compton at 511 keV: {compton}"
        );

        // Form factor at q=0 should equal Z
        let ff0 = db.form_factor(29, 0.0001); // near zero
        assert!(ff0 > 28.0 && ff0 < 30.0, "Cu FF(0): {ff0}");

        // All cross-sections
        let [pe, comp, coh, pair] = db.all_cross_sections(29, 0.511);
        assert!(pe > 0.0);
        assert!(comp > 0.0);
        assert!(coh > 0.0);
        assert!(pair == 0.0, "no pair production at 511 keV");
    }
}
