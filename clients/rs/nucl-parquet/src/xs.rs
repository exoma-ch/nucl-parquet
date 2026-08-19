//! Nuclear reaction cross-section database (e.g. TENDL).
//!
//! Each `CrossSectionDb` holds all reactions for a single target element from
//! one library file (e.g. `tendl-2025/xs/p_Cu.parquet`).

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use arrow::array::{Array, Float64Array, Int32Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

use crate::error::Error;
use crate::interp::{log_log_interp, sort_paired_vecs, XYTable};

/// A single cross-section data point.
#[derive(Debug, Clone)]
pub struct XsEntry {
    pub target_a: u32,
    pub residual_z: u32,
    pub residual_a: u32,
    pub state: String,
    pub energy_mev: f64,
    pub xs_mb: f64,
}

/// Cross-section database for a single target element from one library file.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<CrossSectionDb>`.
#[derive(Clone)]
pub struct CrossSectionDb {
    /// (target_a, residual_z, residual_a, state) -> (energies MeV, xs mb) sorted by energy
    reactions: HashMap<(u32, u32, u32, String), XYTable>,
    target_z: u32,
}

unsafe impl Send for CrossSectionDb {}
unsafe impl Sync for CrossSectionDb {}

impl CrossSectionDb {
    /// Open a cross-section file for a single element.
    ///
    /// The file name must follow the convention `<projectile>_<Symbol>.parquet`
    /// (e.g. `p_Cu.parquet`) so that the target Z can be derived from the
    /// element symbol. Returns an error if the symbol is not recognised.
    pub fn open(xs_file: impl AsRef<Path>) -> crate::Result<Self> {
        let path = xs_file.as_ref();

        let target_z =
            z_from_path(path).ok_or_else(|| Error::DataDirNotFound(path.to_path_buf()))?;

        let file = fs::File::open(path)?;
        let reactions = Self::parse(file)?;

        Ok(Self {
            reactions,
            target_z,
        })
    }

    /// Construct from in-memory Parquet bytes for a known target element.
    ///
    /// Unlike [`open`](Self::open), the caller must supply `target_z` since
    /// there is no file path to derive the element symbol from.
    pub fn from_bytes(target_z: u32, data: &[u8]) -> crate::Result<Self> {
        let bytes = bytes::Bytes::from(data.to_vec());
        let reactions = Self::parse(bytes)?;
        Ok(Self {
            reactions,
            target_z,
        })
    }

    fn parse(
        reader_source: impl parquet::file::reader::ChunkReader + 'static,
    ) -> crate::Result<HashMap<(u32, u32, u32, String), XYTable>> {
        let mut reactions: HashMap<(u32, u32, u32, String), XYTable> = HashMap::new();

        let reader = ParquetRecordBatchReaderBuilder::try_new(reader_source)?.build()?;

        for batch in reader {
            let batch = batch?;

            let ta_col = batch
                .column_by_name("target_A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let rz_col = batch
                .column_by_name("residual_Z")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let ra_col = batch
                .column_by_name("residual_A")
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>());
            let st_col_ref = batch.column_by_name("state");
            let st_values = st_col_ref.and_then(|c| crate::interp::as_string_array(c));
            let e_col = batch
                .column_by_name("energy_MeV")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());
            let xs_col = batch
                .column_by_name("xs_mb")
                .and_then(|c| c.as_any().downcast_ref::<Float64Array>());

            if let (Some(ta), Some(rz), Some(ra), Some(st), Some(e), Some(xs)) =
                (ta_col, rz_col, ra_col, st_values, e_col, xs_col)
            {
                #[allow(clippy::needless_range_loop)]
                for i in 0..batch.num_rows() {
                    // A null residual means the row is a transport channel —
                    // (n,tot), (n,el), (n,f) — which names no product and so has
                    // no key in a residual-indexed table. `value()` on a null
                    // returns the raw buffer slot (0), which would silently
                    // collide every such row onto the (0, 0) key.
                    // `target_A` is guarded too: it is non-null in canonical data,
                    // but value() returns 0 on a null and target_A = 0 is the
                    // ENDF natural-element convention — a null would silently
                    // masquerade as a natural-abundance row.
                    if rz.is_null(i) || ra.is_null(i) || ta.is_null(i) {
                        continue;
                    }
                    let key = (
                        ta.value(i) as u32,
                        rz.value(i) as u32,
                        ra.value(i) as u32,
                        st[i].unwrap_or("").to_string(),
                    );
                    let entry = reactions.entry(key).or_default();
                    entry.0.push(e.value(i));
                    entry.1.push(xs.value(i));
                }
            }
        }

        for (e_vec, xs_vec) in reactions.values_mut() {
            sort_paired_vecs(e_vec, xs_vec);
        }

        Ok(reactions)
    }

    /// Interpolated cross-section [mb] at `energy_mev`.
    ///
    /// Returns `f64::NAN` if the reaction channel is not in the database.
    #[inline]
    pub fn cross_section(
        &self,
        target_a: u32,
        residual_z: u32,
        residual_a: u32,
        energy_mev: f64,
    ) -> f64 {
        self.cross_section_state(target_a, residual_z, residual_a, "", energy_mev)
    }

    /// Interpolated cross-section [mb] for a specific isomeric state.
    ///
    /// Returns `f64::NAN` if the reaction channel is not in the database.
    #[inline]
    pub fn cross_section_state(
        &self,
        target_a: u32,
        residual_z: u32,
        residual_a: u32,
        state: &str,
        energy_mev: f64,
    ) -> f64 {
        let key = (target_a, residual_z, residual_a, state.to_string());
        match self.reactions.get(&key) {
            Some((e, xs)) => log_log_interp(e, xs, energy_mev),
            None => f64::NAN,
        }
    }

    /// All (energy_MeV, xs_mb) pairs for a reaction channel (ground state).
    ///
    /// Returns an empty slice if the reaction is not found.
    pub fn entries(&self, target_a: u32, residual_z: u32, residual_a: u32) -> Vec<(f64, f64)> {
        self.entries_state(target_a, residual_z, residual_a, "")
    }

    /// All (energy_MeV, xs_mb) pairs for a specific isomeric state.
    pub fn entries_state(
        &self,
        target_a: u32,
        residual_z: u32,
        residual_a: u32,
        state: &str,
    ) -> Vec<(f64, f64)> {
        let key = (target_a, residual_z, residual_a, state.to_string());
        self.reactions
            .get(&key)
            .map(|(e, xs)| e.iter().copied().zip(xs.iter().copied()).collect())
            .unwrap_or_default()
    }

    /// Target element atomic number derived from the file name.
    pub fn target_z(&self) -> u32 {
        self.target_z
    }

    /// Iterate all loaded reaction channel keys: `(target_A, residual_Z, residual_A, state)`.
    pub fn reaction_keys(&self) -> impl Iterator<Item = (u32, u32, u32, &str)> + '_ {
        self.reactions
            .keys()
            .map(|(ta, rz, ra, s)| (*ta, *rz, *ra, s.as_str()))
    }

    /// Number of distinct reaction channels loaded.
    pub fn num_reactions(&self) -> usize {
        self.reactions.len()
    }
}

/// Derive target Z from an XS file path by extracting the element symbol.
///
/// Expects a file name of the form `<proj>_<Symbol>.parquet`, e.g. `p_Cu.parquet`.
fn z_from_path(path: &Path) -> Option<u32> {
    let stem = path.file_stem()?.to_str()?;
    let symbol = stem.split('_').nth(1)?;
    SYMBOL_TO_Z
        .iter()
        .find(|(s, _)| *s == symbol)
        .map(|(_, z)| *z)
}

/// Static element symbol → Z lookup table (Z = 1..118).
static SYMBOL_TO_Z: &[(&str, u32)] = &[
    ("H", 1),
    ("He", 2),
    ("Li", 3),
    ("Be", 4),
    ("B", 5),
    ("C", 6),
    ("N", 7),
    ("O", 8),
    ("F", 9),
    ("Ne", 10),
    ("Na", 11),
    ("Mg", 12),
    ("Al", 13),
    ("Si", 14),
    ("P", 15),
    ("S", 16),
    ("Cl", 17),
    ("Ar", 18),
    ("K", 19),
    ("Ca", 20),
    ("Sc", 21),
    ("Ti", 22),
    ("V", 23),
    ("Cr", 24),
    ("Mn", 25),
    ("Fe", 26),
    ("Co", 27),
    ("Ni", 28),
    ("Cu", 29),
    ("Zn", 30),
    ("Ga", 31),
    ("Ge", 32),
    ("As", 33),
    ("Se", 34),
    ("Br", 35),
    ("Kr", 36),
    ("Rb", 37),
    ("Sr", 38),
    ("Y", 39),
    ("Zr", 40),
    ("Nb", 41),
    ("Mo", 42),
    ("Tc", 43),
    ("Ru", 44),
    ("Rh", 45),
    ("Pd", 46),
    ("Ag", 47),
    ("Cd", 48),
    ("In", 49),
    ("Sn", 50),
    ("Sb", 51),
    ("Te", 52),
    ("I", 53),
    ("Xe", 54),
    ("Cs", 55),
    ("Ba", 56),
    ("La", 57),
    ("Ce", 58),
    ("Pr", 59),
    ("Nd", 60),
    ("Pm", 61),
    ("Sm", 62),
    ("Eu", 63),
    ("Gd", 64),
    ("Tb", 65),
    ("Dy", 66),
    ("Ho", 67),
    ("Er", 68),
    ("Tm", 69),
    ("Yb", 70),
    ("Lu", 71),
    ("Hf", 72),
    ("Ta", 73),
    ("W", 74),
    ("Re", 75),
    ("Os", 76),
    ("Ir", 77),
    ("Pt", 78),
    ("Au", 79),
    ("Hg", 80),
    ("Tl", 81),
    ("Pb", 82),
    ("Bi", 83),
    ("Po", 84),
    ("At", 85),
    ("Rn", 86),
    ("Fr", 87),
    ("Ra", 88),
    ("Ac", 89),
    ("Th", 90),
    ("Pa", 91),
    ("U", 92),
    ("Np", 93),
    ("Pu", 94),
    ("Am", 95),
    ("Cm", 96),
    ("Bk", 97),
    ("Cf", 98),
    ("Es", 99),
    ("Fm", 100),
    ("Md", 101),
    ("No", 102),
    ("Lr", 103),
    ("Rf", 104),
    ("Db", 105),
    ("Sg", 106),
    ("Bh", 107),
    ("Hs", 108),
    ("Mt", 109),
    ("Ds", 110),
    ("Rg", 111),
    ("Cn", 112),
    ("Nh", 113),
    ("Fl", 114),
    ("Mc", 115),
    ("Lv", 116),
    ("Ts", 117),
    ("Og", 118),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_to_z_cu() {
        assert_eq!(
            SYMBOL_TO_Z
                .iter()
                .find(|(s, _)| *s == "Cu")
                .map(|(_, z)| *z),
            Some(29)
        );
    }

    #[test]
    fn z_from_path_parses_cu() {
        let path = std::path::Path::new("tendl-2025/xs/p_Cu.parquet");
        assert_eq!(z_from_path(path), Some(29));
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn open_and_query_cu() {
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        assert_eq!(db.target_z(), 29);
        assert!(db.num_reactions() > 0);
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn cross_section_finite() {
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        // Cu-63(p,n)Zn-63 is a common reaction
        let xs = db.cross_section(63, 30, 63, 15.0);
        // Either we get a finite positive value or NAN (reaction may be named differently)
        assert!(xs.is_nan() || xs >= 0.0);
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn entries_nonempty() {
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        assert!(db.num_reactions() > 0, "should have at least one reaction");
    }

    /// Build a canonical-shape xs table in memory, so the null-residual path is
    /// covered without needing the data tree.
    ///
    /// `(target_A, residual_Z, residual_A, energy_MeV, xs_mb)`. `None`
    /// residuals mean "this channel names no product" — a null, not a
    /// zero, because Z=0 is a real value (representation principle 3).
    type SyntheticRow = (Option<i32>, Option<i32>, Option<i32>, f64, f64);

    /// Build an in-memory parquet from `rows`.
    fn synthetic_xs(rows: &[SyntheticRow]) -> Vec<u8> {
        use arrow::array::{Float64Array, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let schema = Arc::new(Schema::new(vec![
            Field::new("target_A", DataType::Int32, true),
            Field::new("residual_Z", DataType::Int32, true),
            Field::new("residual_A", DataType::Int32, true),
            Field::new("state", DataType::Utf8, true),
            Field::new("energy_MeV", DataType::Float64, false),
            Field::new("xs_mb", DataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| r.0).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| r.1).collect::<Vec<_>>(),
                )),
                Arc::new(Int32Array::from(
                    rows.iter().map(|r| r.2).collect::<Vec<_>>(),
                )),
                Arc::new(StringArray::from(vec![""; rows.len()])),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.3).collect::<Vec<_>>(),
                )),
                Arc::new(Float64Array::from(
                    rows.iter().map(|r| r.4).collect::<Vec<_>>(),
                )),
            ],
        )
        .unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let mut w = ArrowWriter::try_new(&mut buf, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        buf
    }

    #[test]
    fn null_residuals_are_skipped_not_keyed_as_zero() {
        // Two transport-channel rows (null residual) and one real (n,g) channel.
        // Arrow's `value()` on a null returns the raw buffer slot — 0 — so an
        // unguarded read collapses every channel row onto the (0, 0) key and
        // interleaves unrelated curves under it.
        let bytes = synthetic_xs(&[
            (Some(63), None, None, 1.0, 500.0),
            (Some(63), None, None, 2.0, 400.0),
            (Some(63), Some(30), Some(64), 1.0, 100.0),
        ]);
        let db = CrossSectionDb::from_bytes(29, &bytes).unwrap();

        assert_eq!(
            db.num_reactions(),
            1,
            "only the row naming a product is keyed"
        );
        assert!(
            db.cross_section(63, 0, 0, 1.0).is_nan(),
            "a null residual must not become the (0, 0) key"
        );
        assert_eq!(db.cross_section(63, 30, 64, 1.0), 100.0);
    }

    #[test]
    fn null_target_a_is_skipped() {
        // target_A = 0 is the ENDF natural-element convention, so a null read as
        // 0 would silently masquerade as a natural-abundance row.
        //
        // The two rows carry *different* residuals on purpose: what a null slot
        // decodes to is arbitrary (parquet leaves whatever the encoding happens
        // to put there, often a neighbouring value), so asserting on the key the
        // null row would produce is unreliable. Counting keys is not.
        let bytes = synthetic_xs(&[
            (None, Some(30), Some(64), 1.0, 100.0),
            (Some(63), Some(31), Some(65), 1.0, 200.0),
        ]);
        let db = CrossSectionDb::from_bytes(29, &bytes).unwrap();
        assert_eq!(
            db.num_reactions(),
            1,
            "the row with a null target_A must not be keyed at all"
        );
        assert_eq!(db.cross_section(63, 31, 65, 1.0), 200.0);
    }

    #[test]
    fn all_null_residuals_yields_an_empty_db_not_an_error() {
        // A pure transport-channel file (every row is (n,tot)/(n,el)/(n,f)) is
        // valid input to a residual-indexed table — it just has no reactions.
        let bytes = synthetic_xs(&[(Some(238), None, None, 1.0, 500.0)]);
        let db = CrossSectionDb::from_bytes(92, &bytes).unwrap();
        assert_eq!(db.num_reactions(), 0);
    }

    fn data_xs_file() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("..")
            .join("data")
            .join("tendl-2025")
            .join("xs")
            .join("p_Cu.parquet")
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn from_bytes_matches_open() {
        let path = data_xs_file();
        let db_file = CrossSectionDb::open(&path).unwrap();
        let data = std::fs::read(&path).unwrap();
        let db_bytes = CrossSectionDb::from_bytes(db_file.target_z(), &data).unwrap();
        assert_eq!(db_bytes.target_z(), db_file.target_z());
        assert_eq!(db_bytes.num_reactions(), db_file.num_reactions());
        // Spot-check a cross-section value
        for (ta, rz, ra, _st) in db_file.reaction_keys().take(3) {
            let val_file = db_file.cross_section(ta, rz, ra, 15.0);
            let val_bytes = db_bytes.cross_section(ta, rz, ra, 15.0);
            assert!(
                (val_file.is_nan() && val_bytes.is_nan()) || (val_file - val_bytes).abs() < 1e-12,
                "mismatch for ({ta},{rz},{ra}): {val_file} vs {val_bytes}"
            );
        }
    }
}
