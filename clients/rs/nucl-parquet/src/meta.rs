//! Nuclide metadata: isotopic abundances, radioactive decay, and dose constants.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock, RwLock};

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

/// IUPAC element symbols indexed by Z (1..=118). Empty string for Z=0
/// (never used). Matches the file-naming convention of
/// `meta/ensdf/{coincidences,radiation,levels}/{Symbol}.parquet`.
static Z_TO_SYMBOL: [&str; 119] = [
    "", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
    "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
];

fn z_to_symbol(z: u32) -> Option<&'static str> {
    if z == 0 || z as usize >= Z_TO_SYMBOL.len() {
        None
    } else {
        Some(Z_TO_SYMBOL[z as usize])
    }
}

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
            let sym_col_ref = batch.column_by_name("symbol");
            let sym_values = sym_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let ab_col = batch
                .column_by_name("abundance")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let mass_col = batch
                .column_by_name("atomic_mass")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(z), Some(a), Some(sym), Some(ab), Some(mass)) =
                (z_col, a_col, sym_values, ab_col, mass_col)
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    let entry = AbundanceEntry {
                        z: z.value(i) as u32,
                        a: a.value(i) as u32,
                        symbol: sym[i].unwrap_or("").to_string(),
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
            let state_col_ref = batch.column_by_name("state");
            let state_values = state_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let hl_col = batch
                .column_by_name("half_life_s")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let mode_col_ref = batch.column_by_name("decay_mode");
            let mode_values = mode_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let dz_col = batch
                .column_by_name("daughter_Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let da_col = batch
                .column_by_name("daughter_A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let ds_col_ref = batch.column_by_name("daughter_state");
            let ds_values = ds_col_ref.and_then(|c| crate::interp::as_string_array(c));
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
                z_col,
                a_col,
                state_values,
                hl_col,
                mode_values,
                dz_col,
                da_col,
                ds_values,
                br_col,
            ) {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    let z_val = z.value(i) as u32;
                    let a_val = a.value(i) as u32;
                    let entry = DecayEntry {
                        z: z_val,
                        a: a_val,
                        state: state[i].unwrap_or("").to_string(),
                        half_life_s: if hl.is_null(i) {
                            None
                        } else {
                            Some(hl.value(i))
                        },
                        decay_mode: mode[i].unwrap_or("").to_string(),
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
                        daughter_state: ds[i].unwrap_or("").to_string(),
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
            .map(|entries| entries.iter().filter(|e| e.state == state).collect())
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
            let state_col_ref = batch.column_by_name("state");
            let state_values = state_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let k_col = batch
                .column_by_name("k_uSv_m2_MBq_h")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(z), Some(a), Some(state), Some(k)) = (z_col, a_col, state_values, k_col) {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    data.insert(
                        (
                            z.value(i) as u32,
                            a.value(i) as u32,
                            state[i].unwrap_or("").to_string(),
                        ),
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

// ---------------------------------------------------------------------------
// CoincidencesDb (issues #170, #175)
// ---------------------------------------------------------------------------

/// One side of a coincidence pair — a single emission line.
///
/// `rad_type` values: `"gamma"`, `"beta"`, `"xray"`, `"auger"`, `"annihilation_511"`.
/// `shell` is populated only for X-ray and Auger emissions (K/L/M/N).
/// `icc_total` is populated only for γ emissions; `None` otherwise.
#[derive(Debug, Clone)]
pub struct Emission {
    pub rad_type: String,
    pub energy_kev: f64,
    pub intensity: f32,
    pub shell: Option<String>,
    pub icc_total: Option<f32>,
}

/// A single coincidence pair — two emissions observed in one parent decay.
///
/// Schema mirrors `data/meta/ensdf/coincidences/{Symbol}.parquet` post-#170
/// (mixed-emission build). γ-γ rows from v0.11 keep the cascade-level columns
/// (`parent_level_kev`, `intermediate_level_kev`, `final_level_kev`)
/// populated; mixed-emission rows have NULL `final_level_kev`.
///
/// **Daughter-keyed**: `z` and `a` are the *daughter* nucleus where the γ
/// cascade lives. For Co-60 β⁻ → Ni-60 cascade, rows are filed under
/// `(z=28, a=60)`.
#[derive(Debug, Clone)]
pub struct CoincidenceEntry {
    /// Daughter Z (where the cascade γ are emitted).
    pub z: u32,
    /// Daughter A.
    pub a: u32,
    /// Parent isomeric state: `""` (ground) | `"m"` | `"m2"`.
    pub parent_state: String,
    /// Parent decay channel that fed the cascade.
    ///
    /// `None` for γ-γ rows where the parent_level couldn't be matched to a
    /// directly fed daughter level (the cascade enters that level via deeper
    /// cascade rather than direct feeding). Values include `"beta-"`,
    /// `"beta+"`, `"KshellEC"`, `"LshellEC"`, `"MshellEC"`, `"NshellEC"`,
    /// `"alpha"`, `"IT"`.
    pub parent_decay_mode: Option<String>,
    /// Daughter level excitation energy that the parent decay fed (keV).
    pub daughter_ex_kev: Option<f64>,
    /// γ₁ parent-level energy (keV) — γ-γ rows only.
    pub parent_level_kev: Option<f64>,
    /// γ₁ daughter / γ₂ parent level energy (keV) — γ-γ rows only.
    pub intermediate_level_kev: Option<f64>,
    /// γ₂ daughter level energy (keV) — γ-γ rows only.
    pub final_level_kev: Option<f64>,
    pub emission1: Emission,
    pub emission2: Emission,
    /// Relative pair intensity per parent decay (see schema docs for normalization).
    pub pair_intensity: f32,
}

/// Filter for [`CoincidencesDb::pairs_filtered`].
///
/// All fields default to `None` (no filter). Filters compose via AND.
#[derive(Debug, Clone, Default)]
pub struct CoincidenceFilter {
    pub parent_state: Option<String>,
    pub parent_decay_mode: Option<String>,
    pub emission1_rad_type: Option<String>,
    pub emission2_rad_type: Option<String>,
    /// Drop pairs with `pair_intensity ≤ min_intensity`.
    pub min_intensity: f32,
}

/// Coincidence database — γ-γ + mixed-emission pairs per (Z, A).
///
/// **Lazy per-element loading.** `open()` only verifies the data directory;
/// individual `{Symbol}.parquet` files are read and cached on first access via
/// [`CoincidencesDb::pairs_for_element`]. Memory cost is bounded by the
/// elements actually queried (typically ≤ 10 MB per element); the full 67 MB
/// dataset is never resident unless the caller iterates all 104 elements.
///
/// Eager loading was tried first and produced a 6 GB working-set blow-up
/// from String allocations across ~5M rows × ~10 string columns (see
/// pitfalls section of issue #175).
///
/// Thread-safe: `Send + Sync`. Share via `Arc<CoincidencesDb>`.
pub struct CoincidencesDb {
    coinc_dir: PathBuf,
    cache: RwLock<HashMap<u32, Arc<Vec<CoincidenceEntry>>>>,
}

unsafe impl Send for CoincidencesDb {}
unsafe impl Sync for CoincidencesDb {}

impl CoincidencesDb {
    /// Open the coincidences directory under `data_dir/meta/ensdf/`.
    ///
    /// Does no file I/O beyond verifying the directory exists.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let coinc_dir = data_dir.as_ref().join("ensdf").join("coincidences");
        if !coinc_dir.is_dir() {
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("coincidences directory not found: {}", coinc_dir.display()),
            )));
        }
        Ok(Self {
            coinc_dir,
            cache: RwLock::new(HashMap::new()),
        })
    }

    /// All coincidence pairs for daughter element Z (loaded + cached).
    ///
    /// Reads `coincidences/{Symbol}.parquet` once and caches the result as
    /// an `Arc<Vec<…>>`. Subsequent calls return a cheap clone of the Arc.
    pub fn pairs_for_element(&self, z: u32) -> crate::Result<Arc<Vec<CoincidenceEntry>>> {
        if let Some(cached) = self.cache.read().unwrap().get(&z) {
            return Ok(Arc::clone(cached));
        }
        let symbol = z_to_symbol(z).ok_or_else(|| {
            crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown Z={z}"),
            ))
        })?;
        let path = self.coinc_dir.join(format!("{symbol}.parquet"));
        let entries = Self::load_file(&path)?;
        let arc = Arc::new(entries);
        self.cache.write().unwrap().insert(z, Arc::clone(&arc));
        Ok(arc)
    }

    fn load_file(path: &Path) -> crate::Result<Vec<CoincidenceEntry>> {
        let mut out: Vec<CoincidenceEntry> = Vec::new();
        let file = fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let a_col = batch
                .column_by_name("A")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let parent_state_ref = batch.column_by_name("parent_state");
            let parent_state = parent_state_ref.and_then(|c| crate::interp::as_string_array(c));
            let parent_decay_mode_ref = batch.column_by_name("parent_decay_mode");
            let parent_decay_mode =
                parent_decay_mode_ref.and_then(|c| crate::interp::as_string_array(c));
            let daughter_ex_col = batch
                .column_by_name("daughter_ex_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let parent_lvl_col = batch
                .column_by_name("parent_level_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let inter_lvl_col = batch
                .column_by_name("intermediate_level_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let final_lvl_col = batch
                .column_by_name("final_level_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let pair_intensity_col = batch
                .column_by_name("pair_intensity")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            let e1_rad_type_ref = batch.column_by_name("emission1_rad_type");
            let e1_rad_type = e1_rad_type_ref.and_then(|c| crate::interp::as_string_array(c));
            let e1_energy_col = batch
                .column_by_name("emission1_energy_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let e1_intensity_col = batch
                .column_by_name("emission1_intensity")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            let e1_shell_ref = batch.column_by_name("emission1_shell");
            let e1_shell = e1_shell_ref.and_then(|c| crate::interp::as_string_array(c));

            let e2_rad_type_ref = batch.column_by_name("emission2_rad_type");
            let e2_rad_type = e2_rad_type_ref.and_then(|c| crate::interp::as_string_array(c));
            let e2_energy_col = batch
                .column_by_name("emission2_energy_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let e2_intensity_col = batch
                .column_by_name("emission2_intensity")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            let e2_shell_ref = batch.column_by_name("emission2_shell");
            let e2_shell = e2_shell_ref.and_then(|c| crate::interp::as_string_array(c));

            // ICC totals are γ-γ-legacy columns (one per side, both nullable).
            let g1_icc_col = batch
                .column_by_name("gamma1_icc_total")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            let g2_icc_col = batch
                .column_by_name("gamma2_icc_total")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            // Required columns must all be present; skip the file if not.
            let (
                Some(z),
                Some(a),
                Some(ps),
                Some(pi),
                Some(e1rt),
                Some(e1e),
                Some(e1i),
                Some(e2rt),
                Some(e2e),
                Some(e2i),
            ) = (
                z_col,
                a_col,
                parent_state.as_ref(),
                pair_intensity_col,
                e1_rad_type.as_ref(),
                e1_energy_col,
                e1_intensity_col,
                e2_rad_type.as_ref(),
                e2_energy_col,
                e2_intensity_col,
            )
            else {
                continue;
            };

            for i in 0..batch.num_rows() {
                let z_val = z.value(i) as u32;
                let a_val = a.value(i) as u32;
                let entry = CoincidenceEntry {
                    z: z_val,
                    a: a_val,
                    parent_state: ps[i].unwrap_or("").to_string(),
                    parent_decay_mode: parent_decay_mode
                        .as_ref()
                        .and_then(|c| c[i].map(|s| s.to_string())),
                    daughter_ex_kev: daughter_ex_col.and_then(|c| {
                        if c.is_null(i) {
                            None
                        } else {
                            Some(c.value(i))
                        }
                    }),
                    parent_level_kev: parent_lvl_col.and_then(|c| {
                        if c.is_null(i) {
                            None
                        } else {
                            Some(c.value(i))
                        }
                    }),
                    intermediate_level_kev: inter_lvl_col.and_then(|c| {
                        if c.is_null(i) {
                            None
                        } else {
                            Some(c.value(i))
                        }
                    }),
                    final_level_kev: final_lvl_col.and_then(|c| {
                        if c.is_null(i) {
                            None
                        } else {
                            Some(c.value(i))
                        }
                    }),
                    emission1: Emission {
                        rad_type: e1rt[i].unwrap_or("").to_string(),
                        energy_kev: e1e.value(i),
                        intensity: e1i.value(i),
                        shell: e1_shell.as_ref().and_then(|c| c[i].map(|s| s.to_string())),
                        icc_total: g1_icc_col.and_then(|c| {
                            if c.is_null(i) {
                                None
                            } else {
                                Some(c.value(i))
                            }
                        }),
                    },
                    emission2: Emission {
                        rad_type: e2rt[i].unwrap_or("").to_string(),
                        energy_kev: e2e.value(i),
                        intensity: e2i.value(i),
                        shell: e2_shell.as_ref().and_then(|c| c[i].map(|s| s.to_string())),
                        icc_total: g2_icc_col.and_then(|c| {
                            if c.is_null(i) {
                                None
                            } else {
                                Some(c.value(i))
                            }
                        }),
                    },
                    pair_intensity: pi.value(i),
                };
                out.push(entry);
            }
        }
        Ok(out)
    }

    /// All coincidence pairs for daughter nucleus (Z, A).
    ///
    /// Remember: rows are filed under the *daughter* nucleus where γ are
    /// emitted. For Co-60 (parent) β⁻ → Ni-60 cascade pairs, call
    /// `pairs(28, 60)`, not `pairs(27, 60)`.
    pub fn pairs(&self, z: u32, a: u32) -> crate::Result<Vec<CoincidenceEntry>> {
        Ok(self
            .pairs_for_element(z)?
            .iter()
            .filter(|e| e.a == a)
            .cloned()
            .collect())
    }
}

impl CoincidencesDb {
    /// Coincidence pairs for daughter (Z, A) filtered by `f`.
    pub fn pairs_filtered(
        &self,
        z: u32,
        a: u32,
        f: &CoincidenceFilter,
    ) -> crate::Result<Vec<CoincidenceEntry>> {
        Ok(self
            .pairs_for_element(z)?
            .iter()
            .filter(|e| e.a == a)
            .filter(|e| f.matches(e))
            .cloned()
            .collect())
    }
}

impl CoincidenceFilter {
    fn matches(&self, e: &CoincidenceEntry) -> bool {
        if let Some(s) = &self.parent_state {
            if &e.parent_state != s {
                return false;
            }
        }
        if let Some(m) = &self.parent_decay_mode {
            if e.parent_decay_mode.as_deref() != Some(m.as_str()) {
                return false;
            }
        }
        if let Some(r) = &self.emission1_rad_type {
            if &e.emission1.rad_type != r {
                return false;
            }
        }
        if let Some(r) = &self.emission2_rad_type {
            if &e.emission2.rad_type != r {
                return false;
            }
        }
        e.pair_intensity > self.min_intensity
    }
}

// ---------------------------------------------------------------------------
// RadiationDb (issue #175) — emissions per (Z, A, state) + identify_gamma()
// ---------------------------------------------------------------------------

/// A single radiation emission line (γ, X-ray, or Auger).
///
/// Schema mirrors `data/meta/ensdf/radiation/{Symbol}.parquet`. The `state`
/// field follows the v0.10.x empty-string convention for ground state.
#[derive(Debug, Clone)]
pub struct EmissionEntry {
    pub z: u32,
    pub a: u32,
    pub state: String,
    pub rad_type: String,
    pub energy_kev: f64,
    pub intensity_pct: f64,
    pub decay_mode: Option<String>,
    pub rad_subtype: Option<String>,
    pub icc_total: Option<f32>,
    pub vacancy_shell: Option<String>,
}

/// One match returned by [`RadiationDb::identify_gamma`].
#[derive(Debug, Clone)]
pub struct GammaCandidate {
    pub z: u32,
    pub a: u32,
    pub state: String,
    pub energy_kev: f64,
    pub intensity_pct: f64,
    pub delta_kev: f64,
}

/// Radiation database — γ, X-ray, and Auger emission lines per (Z, A, state).
///
/// Loaded eagerly from `meta/ensdf/radiation/{Symbol}.parquet`. Supports both
/// "emissions of isotope X" lookup (`emissions`) and "what emits γ near
/// energy E" search (`identify_gamma`).
///
/// Lean γ-only row stored in the cross-isotope index used by
/// [`RadiationDb::identify_gamma`]. Sized to ~24 bytes/row to keep the index
/// resident at ~15 MB across the full dataset.
#[derive(Debug, Clone, Copy)]
struct GammaIndexEntry {
    energy_kev: f64,
    intensity_pct: f64,
    z: u16,
    a: u16,
    /// 0=ground, 1=m, 2=m2, 255=other (rare).
    state_idx: u8,
}

/// **Daughter-filed convention.** Each emission row is filed under the
/// nucleus *that emits the γ / X-ray / Auger electron*, not the parent that
/// decayed to produce it. The 1173 / 1333 keV γ from a Co-60 calibration
/// source live under `(Z=28, A=60)` (daughter Ni-60), since they are emitted
/// by the de-exciting Ni-60 nucleus. The historical convention matches the
/// coincidence build's daughter-keying. To collect everything a parent
/// source emits, walk parent → daughter via `DecayDb::modes` and call
/// `emissions(daughter_z, daughter_a, state)` for each.
///
/// **Lazy per-element loading + eager-built γ index.**
///
/// `open()` does no file I/O for per-isotope emissions; per-element files are
/// loaded on first access via [`RadiationDb::emissions_for_element`]. The
/// `identify_gamma` index is built on first call (one full scan of every
/// `radiation/{Symbol}.parquet`) and reused thereafter — ~15 MB resident.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<RadiationDb>`.
pub struct RadiationDb {
    rad_dir: PathBuf,
    cache: RwLock<HashMap<u32, Arc<Vec<EmissionEntry>>>>,
    gamma_index: OnceLock<Vec<GammaIndexEntry>>,
}

unsafe impl Send for RadiationDb {}
unsafe impl Sync for RadiationDb {}

impl RadiationDb {
    /// Open the radiation directory. Does no file I/O beyond verifying it exists.
    pub fn open(data_dir: impl AsRef<Path>) -> crate::Result<Self> {
        let rad_dir = data_dir.as_ref().join("ensdf").join("radiation");
        if !rad_dir.is_dir() {
            return Err(crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("radiation directory not found: {}", rad_dir.display()),
            )));
        }
        Ok(Self {
            rad_dir,
            cache: RwLock::new(HashMap::new()),
            gamma_index: OnceLock::new(),
        })
    }

    /// All emissions for element Z (loaded + cached).
    pub fn emissions_for_element(&self, z: u32) -> crate::Result<Arc<Vec<EmissionEntry>>> {
        if let Some(cached) = self.cache.read().unwrap().get(&z) {
            return Ok(Arc::clone(cached));
        }
        let symbol = z_to_symbol(z).ok_or_else(|| {
            crate::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("unknown Z={z}"),
            ))
        })?;
        let path = self.rad_dir.join(format!("{symbol}.parquet"));
        let entries = Self::load_file(&path)?;
        let arc = Arc::new(entries);
        self.cache.write().unwrap().insert(z, Arc::clone(&arc));
        Ok(arc)
    }

    fn load_file(path: &Path) -> crate::Result<Vec<EmissionEntry>> {
        let mut out: Vec<EmissionEntry> = Vec::new();
        let file = fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;

        for batch in reader {
            let batch = batch?;

            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let a_col = batch
                .column_by_name("A")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let state_ref = batch.column_by_name("state");
            let state = state_ref.and_then(|c| crate::interp::as_string_array(c));
            let rad_type_ref = batch.column_by_name("rad_type");
            let rad_type = rad_type_ref.and_then(|c| crate::interp::as_string_array(c));
            let energy_col = batch
                .column_by_name("energy_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let intensity_col = batch
                .column_by_name("intensity_pct")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let decay_mode_ref = batch.column_by_name("decay_mode");
            let decay_mode = decay_mode_ref.and_then(|c| crate::interp::as_string_array(c));
            let rad_subtype_ref = batch.column_by_name("rad_subtype");
            let rad_subtype = rad_subtype_ref.and_then(|c| crate::interp::as_string_array(c));
            let icc_col = batch
                .column_by_name("icc_total")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());
            let vacancy_ref = batch.column_by_name("vacancy_shell");
            let vacancy = vacancy_ref.and_then(|c| crate::interp::as_string_array(c));

            let (Some(z), Some(a), Some(st), Some(rt), Some(en), Some(it)) = (
                z_col,
                a_col,
                state.as_ref(),
                rad_type.as_ref(),
                energy_col,
                intensity_col,
            ) else {
                continue;
            };

            for i in 0..batch.num_rows() {
                let z_val = z.value(i) as u32;
                let a_val = a.value(i) as u32;
                let state_val = st[i].unwrap_or("").to_string();
                let rt_val = rt[i].unwrap_or("").to_string();
                out.push(EmissionEntry {
                    z: z_val,
                    a: a_val,
                    state: state_val,
                    rad_type: rt_val,
                    energy_kev: en.value(i),
                    intensity_pct: it.value(i),
                    decay_mode: decay_mode
                        .as_ref()
                        .and_then(|c| c[i].map(|s| s.to_string())),
                    rad_subtype: rad_subtype
                        .as_ref()
                        .and_then(|c| c[i].map(|s| s.to_string())),
                    icc_total: icc_col
                        .and_then(|c| if c.is_null(i) { None } else { Some(c.value(i)) }),
                    vacancy_shell: vacancy.as_ref().and_then(|c| c[i].map(|s| s.to_string())),
                });
            }
        }
        Ok(out)
    }

    /// All emissions for nuclide (Z, A, state). `state=""` for ground.
    pub fn emissions(&self, z: u32, a: u32, state: &str) -> crate::Result<Vec<EmissionEntry>> {
        Ok(self
            .emissions_for_element(z)?
            .iter()
            .filter(|e| e.a == a && e.state == state)
            .cloned()
            .collect())
    }

    /// Emissions filtered by `rad_type` and minimum intensity_pct.
    pub fn emissions_filtered(
        &self,
        z: u32,
        a: u32,
        state: &str,
        rad_type: Option<&str>,
        min_intensity_pct: f64,
    ) -> crate::Result<Vec<EmissionEntry>> {
        Ok(self
            .emissions_for_element(z)?
            .iter()
            .filter(|e| e.a == a && e.state == state)
            .filter(|e| {
                if let Some(rt) = rad_type {
                    if e.rad_type != rt {
                        return false;
                    }
                }
                e.intensity_pct >= min_intensity_pct
            })
            .cloned()
            .collect())
    }

    /// Candidate nuclides emitting a γ near `energy_kev` (within `tolerance_kev`).
    ///
    /// Equivalent to the Python loader's `identify_gamma()`. Returns matches
    /// with `intensity_pct >= min_intensity_pct`, sorted by `delta_kev` (closest
    /// match first), then by intensity descending.
    ///
    /// On first call, builds a sorted cross-isotope γ index (~15 MB) by
    /// scanning every per-element radiation file. Subsequent calls reuse the
    /// index via `OnceLock`.
    pub fn identify_gamma(
        &self,
        energy_kev: f64,
        tolerance_kev: f64,
        min_intensity_pct: f64,
    ) -> crate::Result<Vec<GammaCandidate>> {
        let index = self.gamma_index_or_build()?;
        let lo = energy_kev - tolerance_kev;
        let hi = energy_kev + tolerance_kev;
        let start = index.partition_point(|e| e.energy_kev < lo);
        let end = index.partition_point(|e| e.energy_kev <= hi);

        let mut out: Vec<GammaCandidate> = index[start..end]
            .iter()
            .filter(|e| e.intensity_pct >= min_intensity_pct)
            .map(|e| GammaCandidate {
                z: e.z as u32,
                a: e.a as u32,
                state: match e.state_idx {
                    0 => String::new(),
                    1 => "m".to_string(),
                    2 => "m2".to_string(),
                    _ => "?".to_string(),
                },
                energy_kev: e.energy_kev,
                intensity_pct: e.intensity_pct,
                delta_kev: e.energy_kev - energy_kev,
            })
            .collect();
        out.sort_by(|a, b| {
            a.delta_kev
                .abs()
                .partial_cmp(&b.delta_kev.abs())
                .unwrap()
                .then(b.intensity_pct.partial_cmp(&a.intensity_pct).unwrap())
        });
        Ok(out)
    }

    fn gamma_index_or_build(&self) -> crate::Result<&[GammaIndexEntry]> {
        if let Some(idx) = self.gamma_index.get() {
            return Ok(idx);
        }
        // Race-safe: if two threads call simultaneously, both build, and
        // OnceLock keeps the first. The duplicate work is rare and the cost
        // is bounded.
        let built = self.build_gamma_index()?;
        let _ = self.gamma_index.set(built);
        Ok(self.gamma_index.get().expect("gamma_index just set"))
    }

    fn build_gamma_index(&self) -> crate::Result<Vec<GammaIndexEntry>> {
        let mut idx: Vec<GammaIndexEntry> = Vec::new();
        for entry in fs::read_dir(&self.rad_dir)? {
            let path = entry?.path();
            if path.extension().is_none_or(|e| e != "parquet") {
                continue;
            }
            Self::extend_gamma_index_from_file(&path, &mut idx)?;
        }
        idx.sort_by(|a, b| a.energy_kev.partial_cmp(&b.energy_kev).unwrap());
        Ok(idx)
    }

    fn extend_gamma_index_from_file(
        path: &Path,
        idx: &mut Vec<GammaIndexEntry>,
    ) -> crate::Result<()> {
        let file = fs::File::open(path)?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
        for batch in reader {
            let batch = batch?;
            let z_col = batch
                .column_by_name("Z")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let a_col = batch
                .column_by_name("A")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            let state_ref = batch.column_by_name("state");
            let state = state_ref.and_then(|c| crate::interp::as_string_array(c));
            let rad_type_ref = batch.column_by_name("rad_type");
            let rad_type = rad_type_ref.and_then(|c| crate::interp::as_string_array(c));
            let energy_col = batch
                .column_by_name("energy_keV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let intensity_col = batch
                .column_by_name("intensity_pct")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let (Some(z), Some(a), Some(st), Some(rt), Some(en), Some(it)) = (
                z_col,
                a_col,
                state.as_ref(),
                rad_type.as_ref(),
                energy_col,
                intensity_col,
            ) else {
                continue;
            };
            for i in 0..batch.num_rows() {
                if rt[i] != Some("gamma") {
                    continue;
                }
                let state_idx = match st[i].unwrap_or("") {
                    "" => 0u8,
                    "m" => 1,
                    "m2" => 2,
                    _ => 255,
                };
                idx.push(GammaIndexEntry {
                    energy_kev: en.value(i),
                    intensity_pct: it.value(i),
                    z: z.value(i) as u16,
                    a: a.value(i) as u16,
                    state_idx,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta_dir() -> std::path::PathBuf {
        // CARGO_MANIFEST_DIR = <repo>/clients/rs/nucl-parquet → <repo>/data/meta.
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
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
        let has_beta = modes.iter().any(|m| m.decay_mode.starts_with("beta"));
        assert!(has_beta, "Cu-64 should have beta decay");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn coincidences_co60_beta_gamma_pair_intensity() {
        // Co-60 β⁻ → Ni-60. Cascade pair filed under daughter (Z=28, A=60).
        // Acceptance per #170 / sub-A: β⁻ ⊗ 1173 keV γ pair_intensity ≈ 0.9986.
        let db = CoincidencesDb::open(meta_dir()).unwrap();
        let pairs = db.pairs(28, 60).unwrap();
        let hit = pairs.iter().find(|e| {
            e.emission1.rad_type == "beta"
                && (e.emission2.energy_kev - 1173.0).abs() < 2.0
                && e.parent_decay_mode.as_deref() == Some("beta-")
        });
        let hit = hit.expect("Co-60 β⁻ ⊗ 1173 keV γ pair must exist");
        assert!(
            (hit.pair_intensity - 0.9986).abs() < 0.003,
            "Co-60 β-γ pair_intensity: got {}, want ≈ 0.9986",
            hit.pair_intensity
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn coincidences_y86_kshell_xray_gamma() {
        // Y-86 EC → Sr-86, K X-ray ⊗ 1077 keV γ (prompt-γ PET).
        let db = CoincidencesDb::open(meta_dir()).unwrap();
        let pairs = db.pairs(38, 86).unwrap();
        let k_xray_gamma = pairs.iter().any(|e| {
            e.emission1.rad_type == "xray"
                && e.emission1.shell.as_deref() == Some("K")
                && e.parent_decay_mode.as_deref() == Some("KshellEC")
                && (e.emission2.energy_kev - 1077.0).abs() < 2.0
        });
        assert!(k_xray_gamma, "Y-86 K X-ray ⊗ 1077 keV γ pair must exist");
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn coincidences_co60_gamma_gamma_preserved() {
        // The v0.11 γ-γ Co-60 1173/1333 cascade pair must survive the
        // augmented schema (regression test).
        let db = CoincidencesDb::open(meta_dir()).unwrap();
        let pairs = db.pairs(28, 60).unwrap();
        let canonical = pairs.iter().any(|e| {
            e.emission1.rad_type == "gamma"
                && e.emission2.rad_type == "gamma"
                && (((e.emission1.energy_kev - 1173.0).abs() < 2.0
                    && (e.emission2.energy_kev - 1333.0).abs() < 2.0)
                    || ((e.emission1.energy_kev - 1333.0).abs() < 2.0
                        && (e.emission2.energy_kev - 1173.0).abs() < 2.0))
        });
        assert!(
            canonical,
            "Co-60 1173/1333 γ-γ cascade pair must be preserved"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn coincidences_sr90_y90_pure_beta_no_mixed() {
        // Sr-90 (parent) β⁻ → Y-90 ground state. No daughter γ cascade → no
        // mixed pairs flagged with parent_decay_mode='beta-'.
        let db = CoincidencesDb::open(meta_dir()).unwrap();
        let filter = CoincidenceFilter {
            parent_decay_mode: Some("beta-".to_string()),
            ..Default::default()
        };
        let mixed: Vec<_> = db
            .pairs_filtered(39, 90, &filter)
            .unwrap()
            .into_iter()
            .filter(|e| e.emission1.rad_type != "gamma")
            .collect();
        assert!(
            mixed.is_empty(),
            "Sr-90 → Y-90 pure β⁻ must produce zero mixed pairs (got {})",
            mixed.len()
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn radiation_emissions_ni60_has_co60_decay_gammas() {
        // The 1173 + 1333 keV γ are emitted by the daughter Ni-60 nucleus
        // (de-exciting after Co-60 β⁻ decay), so they're filed under
        // (Z=28, A=60) — same daughter-keyed convention as coincidences.
        let db = RadiationDb::open(meta_dir()).unwrap();
        let lines = db.emissions(28, 60, "").unwrap();
        assert!(!lines.is_empty(), "Ni-60 radiation lines must exist");
        let has_1173 = lines
            .iter()
            .any(|e| e.rad_type == "gamma" && (e.energy_kev - 1173.2).abs() < 0.5);
        let has_1333 = lines
            .iter()
            .any(|e| e.rad_type == "gamma" && (e.energy_kev - 1332.5).abs() < 0.5);
        assert!(
            has_1173 && has_1333,
            "Co-60 β⁻ → Ni-60 cascade must emit 1173 + 1333 keV γ"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn radiation_identify_gamma_1173() {
        // "Peak at 1173 keV from my Co-60 source, what is it?" → the 1173.239
        // keV γ is emitted by daughter Ni-60 (Z=28, A=60) de-exciting after
        // Co-60 β⁻ decay. Must appear in identify_gamma results.
        let db = RadiationDb::open(meta_dir()).unwrap();
        let candidates = db.identify_gamma(1173.2, 1.0, 5.0).unwrap(); // >5% intensity
        let found_ni60 = candidates.iter().any(|c| c.z == 28 && c.a == 60);
        assert!(
            found_ni60,
            "identify_gamma(1173.2) must include Ni-60 (the daughter of Co-60 β⁻) (got {} candidates)",
            candidates.len()
        );
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
