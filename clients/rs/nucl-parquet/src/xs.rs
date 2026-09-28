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

/// Residual state meaning "summed over every isomeric state of this product".
///
/// This is what a residual-keyed production row carries when it is *the* total
/// for that product. It is not a peer of `g`/`m`/`m2` but a sum over them, and
/// it is the default for the convenience lookups below because "how much Cu-64
/// is produced" is a question about the sum.
pub const SUM: &str = "sum";

/// The ground state, for both the target and the residual.
pub const GROUND: &str = "g";

/// A single cross-section data point.
#[derive(Debug, Clone)]
pub struct XsEntry {
    pub target_a: u32,
    /// Isomeric state of the *target*. `None` for a natural element — see
    /// [`ReactionKey`].
    pub target_state: Option<String>,
    pub residual_z: u32,
    pub residual_a: u32,
    pub state: String,
    pub energy_mev: f64,
    pub xs_mb: f64,
}

/// What identifies a reaction channel within one element's file.
///
/// `(target_A, target_state, residual_Z, residual_A, state)`.
///
/// `target_A` alone does not identify a target: Br-80 and Br-80m share it, and
/// they are different nuclei with different cross-sections (#353). Before
/// `target_state` joined this key the two merged into one `XYTable` and their
/// curves interleaved — 37,316 distinct keys in the shipped data collide that
/// way, 35,606 of them in tendl-2025 alone.
///
/// `target_state` is `None` exactly when `target_A == 0`, the ENDF
/// natural-element convention: a natural element is a mixture, so it has no one
/// isomeric state and a null is the honest answer rather than a stand-in
/// ground state. That equivalence holds across every shipped shard carrying the
/// column, and `natural_targets_key_a_null_state` pins it.
pub type ReactionKey = (u32, Option<String>, u32, u32, String);

/// Cross-section database for a single target element from one library file.
///
/// Thread-safe: `Send + Sync`. Share via `Arc<CrossSectionDb>`.
#[derive(Clone)]
pub struct CrossSectionDb {
    /// [`ReactionKey`] -> (energies MeV, xs mb) sorted by energy
    reactions: HashMap<ReactionKey, XYTable>,
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
    ) -> crate::Result<HashMap<ReactionKey, XYTable>> {
        let mut reactions: HashMap<ReactionKey, XYTable> = HashMap::new();

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
            // Absent in the shards no longer rebuilt by `fetch_endf_libs.py`
            // (endfb-8.0, the EXFOR tables, the older IAEA sets). Absent is not
            // the same as null: null says "this target has no one state", while
            // absent says "this shard cannot tell you". Both key as `None`; the
            // convenience lookups below fall back so neither goes unreachable.
            let ts_col_ref = batch.column_by_name("target_state");
            let ts_values = ts_col_ref.and_then(|c| crate::interp::as_string_array(c));
            // #347 added `kind='channel'` rows: the per-MT partials that a
            // production row sums. They are identified by MT, and this map is
            // keyed by residual, so they have no key here — several MTs feed one
            // residual and all of them would land on it. Cu-63 -> Ni-61 is fed by
            // MT 44, 106 and 115, so at 30 MeV this table held 192.35, 7.03, 9.64
            // and the correct 209.02 under one key, sorted into a single curve by
            // energy. Keeping only production rows is what this map meant before
            // the channels existed; exposing them properly needs MT in the key.
            let kind_col_ref = batch.column_by_name("kind");
            let kind_values = kind_col_ref.and_then(|c| crate::interp::as_string_array(c));
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
                    // Shards predating #347 have no `kind` column and are all
                    // production, so an absent column keeps every row.
                    if kind_values
                        .as_ref()
                        .and_then(|k| k[i])
                        .is_some_and(|k| k != "production")
                    {
                        continue;
                    }
                    let key = (
                        ta.value(i) as u32,
                        ts_values
                            .as_ref()
                            .and_then(|ts| ts[i])
                            .map(|s| s.to_string()),
                        rz.value(i) as u32,
                        ra.value(i) as u32,
                        st[i].unwrap_or(SUM).to_string(),
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

    /// Look a channel up, resolving an unspecified target state.
    ///
    /// The caller who does not name a target state means "the ordinary target",
    /// which is spelled two different ways across the corpus: `Some("g")` in the
    /// shards `fetch_endf_libs.py` rebuilds, and `None` in those that never
    /// carried the column at all. Trying ground and then not-stated reaches both
    /// without a shard-name special case. A caller who *does* name one gets an
    /// exact lookup and no fallback — the point of naming it is to be told when
    /// it is missing rather than quietly handed the other one.
    fn lookup(
        &self,
        target_a: u32,
        target_state: Option<Option<&str>>,
        residual_z: u32,
        residual_a: u32,
        state: &str,
    ) -> Option<&XYTable> {
        let probe = |ts: Option<&str>| {
            self.reactions.get(&(
                target_a,
                ts.map(str::to_string),
                residual_z,
                residual_a,
                state.to_string(),
            ))
        };
        match target_state {
            Some(ts) => probe(ts),
            // A natural element is keyed null by construction, so ground is not
            // a candidate for it and probing ground first would be dead work.
            None if target_a == 0 => probe(None),
            None => probe(Some(GROUND)).or_else(|| probe(None)),
        }
    }

    /// Interpolated cross-section [mb] at `energy_mev`, summed over the
    /// residual's isomeric states and for a target in its ordinary state.
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
        self.cross_section_state(target_a, residual_z, residual_a, SUM, energy_mev)
    }

    /// Interpolated cross-section [mb] for a specific isomeric state of the
    /// *residual*.
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
        self.cross_section_target(target_a, None, residual_z, residual_a, state, energy_mev)
    }

    /// Interpolated cross-section [mb] naming the isomeric state of *both* the
    /// target and the residual.
    ///
    /// `target_state` is `Some(None)` for a natural element, `Some(Some("m"))`
    /// for an isomeric target, and `None` to let the lookup resolve it. Reach
    /// for this when the target has an isomer — Br-80 and Br-80m are different
    /// nuclei and the other methods will hand you the ground one (#353).
    #[inline]
    pub fn cross_section_target(
        &self,
        target_a: u32,
        target_state: Option<Option<&str>>,
        residual_z: u32,
        residual_a: u32,
        state: &str,
        energy_mev: f64,
    ) -> f64 {
        match self.lookup(target_a, target_state, residual_z, residual_a, state) {
            Some((e, xs)) => log_log_interp(e, xs, energy_mev),
            None => f64::NAN,
        }
    }

    /// All (energy_MeV, xs_mb) pairs for a reaction channel, summed over the
    /// residual's isomeric states.
    ///
    /// Returns an empty slice if the reaction is not found.
    pub fn entries(&self, target_a: u32, residual_z: u32, residual_a: u32) -> Vec<(f64, f64)> {
        self.entries_state(target_a, residual_z, residual_a, SUM)
    }

    /// All (energy_MeV, xs_mb) pairs for a specific isomeric state of the residual.
    pub fn entries_state(
        &self,
        target_a: u32,
        residual_z: u32,
        residual_a: u32,
        state: &str,
    ) -> Vec<(f64, f64)> {
        self.entries_target(target_a, None, residual_z, residual_a, state)
    }

    /// All (energy_MeV, xs_mb) pairs, naming the isomeric state of both the
    /// target and the residual. See [`cross_section_target`](Self::cross_section_target).
    pub fn entries_target(
        &self,
        target_a: u32,
        target_state: Option<Option<&str>>,
        residual_z: u32,
        residual_a: u32,
        state: &str,
    ) -> Vec<(f64, f64)> {
        self.lookup(target_a, target_state, residual_z, residual_a, state)
            .map(|(e, xs)| e.iter().copied().zip(xs.iter().copied()).collect())
            .unwrap_or_default()
    }

    /// Target element atomic number derived from the file name.
    pub fn target_z(&self) -> u32 {
        self.target_z
    }

    /// Iterate all loaded reaction channel keys.
    ///
    /// See [`ReactionKey`] for the tuple. `target_state` joined it in 0.17 —
    /// without it two isomers of one target were indistinguishable here.
    pub fn reaction_keys(&self) -> impl Iterator<Item = (u32, Option<&str>, u32, u32, &str)> + '_ {
        self.reactions
            .keys()
            .map(|(ta, ts, rz, ra, s)| (*ta, ts.as_deref(), *rz, *ra, s.as_str()))
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
        // This test used to accept `xs.is_nan() || xs >= 0.0`, which is every
        // possible outcome, so it stayed green while `cross_section()` returned
        // NAN for every reaction in the corpus (#380 retired the `state = ''`
        // this defaulted to). A value, not a disjunction.
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        // Cu-63(p,n)Zn-63 at 15 MeV is an exact grid point in tendl-2025.
        let xs = db.cross_section(63, 30, 63, 15.0);
        assert!(
            (xs - 221.241).abs() < 1e-3,
            "Cu-63(p,n)Zn-63 at 15 MeV should be 221.241 mb, got {xs}"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn mt_partials_do_not_contaminate_the_production_curve() {
        // #347 added `kind='channel'` rows. tendl-2025 feeds Cu-63 -> Ni-61 from
        // MT 44, 106 and 115, and unfiltered all three partials plus the
        // production sum keyed on the same residual: 48 points where the curve
        // has 15, and four different values at 30 MeV. Which one came back
        // depended on where the energy sort put it.
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        let pts = db.entries(63, 28, 61);
        assert_eq!(
            pts.len(),
            15,
            "the production curve alone, not the partials"
        );
        let xs = db.cross_section(63, 28, 61, 30.0);
        assert!(
            (xs - 209.0245).abs() < 1e-3,
            "expected the production sum 209.0245 mb at 30 MeV, got {xs} \
             (192.3509 = MT 44, 7.0334 = MT 106, 9.6402 = MT 115)"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn shipped_target_isomers_are_distinguishable() {
        // #353 against the real data rather than a fixture. tendl-2025 evaluates
        // Cu-70, Cu-70m and Cu-70m2 as separate targets, and at 15 MeV their
        // Zn-70 production differs by 50%. Keyed on target_A alone all three
        // land in one XYTable, so two of these three numbers are unreachable and
        // the one you get back depends on the energy sort.
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        let at = |ts: &str| db.cross_section_target(70, Some(Some(ts)), 30, 70, SUM, 15.0);
        assert!(
            (at(GROUND) - 73.0705).abs() < 1e-3,
            "Cu-70 -> {}",
            at(GROUND)
        );
        assert!((at("m") - 49.1469).abs() < 1e-3, "Cu-70m -> {}", at("m"));
        assert!((at("m2") - 48.6608).abs() < 1e-3, "Cu-70m2 -> {}", at("m2"));
        assert!(
            (db.cross_section(70, 30, 70, 15.0) - 73.0705).abs() < 1e-3,
            "the unqualified lookup must resolve to the ground-state target"
        );
    }

    #[test]
    #[ignore = "requires nucl-parquet data files"]
    fn entries_nonempty() {
        // Named for `entries`, which it never called — so the empty vec that
        // `entries()` returned for every reaction went unnoticed.
        let db = CrossSectionDb::open(data_xs_file()).unwrap();
        assert!(db.num_reactions() > 0, "should have at least one reaction");
        let pts = db.entries(63, 30, 63);
        assert_eq!(pts.len(), 21, "Cu-63 -> Zn-63 production curve");
        assert!(
            pts.iter()
                .any(|&(e, xs)| e == 15.0 && (xs - 221.241).abs() < 1e-3),
            "the 15 MeV point must be in the curve"
        );
        assert!(
            pts.windows(2).all(|w| w[0].0 < w[1].0),
            "entries must come back sorted by energy"
        );
    }

    /// Build a canonical-shape xs table in memory, so the null-residual path is
    /// covered without needing the data tree.
    ///
    /// `(target_A, target_state, residual_Z, residual_A, state, energy_MeV, xs_mb)`.
    /// `None` residuals mean "this channel names no product" — a null, not a
    /// zero, because Z=0 is a real value (representation principle 3).
    ///
    /// The fixture used to hardcode `state = ""` for every row, which is why no
    /// test here noticed that `cross_section()` had stopped matching anything:
    /// the fixture agreed with the bug. #380 retired that spelling, so the
    /// vocabulary is now spelled out per row and `state` is a parameter.
    type SyntheticRow = (
        Option<i32>,
        Option<&'static str>,
        Option<i32>,
        Option<i32>,
        &'static str,
        f64,
        f64,
    );

    /// Build an in-memory parquet from `rows`.
    ///
    /// `with_target_state` writes the shard layout that `fetch_endf_libs.py`
    /// produces; without it the column is absent, which is the shape endfb-8.0
    /// and the EXFOR tables still have.
    fn synthetic_xs_opt(rows: &[SyntheticRow], with_target_state: bool) -> Vec<u8> {
        use arrow::array::{ArrayRef, Float64Array, Int32Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let mut fields = vec![Field::new("target_A", DataType::Int32, true)];
        let mut cols: Vec<ArrayRef> = vec![Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        ))];
        if with_target_state {
            fields.push(Field::new("target_state", DataType::Utf8, true));
            cols.push(Arc::new(StringArray::from(
                rows.iter().map(|r| r.1).collect::<Vec<_>>(),
            )));
        }
        fields.extend([
            Field::new("residual_Z", DataType::Int32, true),
            Field::new("residual_A", DataType::Int32, true),
            Field::new("state", DataType::Utf8, true),
            Field::new("energy_MeV", DataType::Float64, false),
            Field::new("xs_mb", DataType::Float64, false),
        ]);
        cols.extend::<Vec<ArrayRef>>(vec![
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.2).collect::<Vec<_>>(),
            )),
            Arc::new(Int32Array::from(
                rows.iter().map(|r| r.3).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                rows.iter().map(|r| r.4).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.5).collect::<Vec<_>>(),
            )),
            Arc::new(Float64Array::from(
                rows.iter().map(|r| r.6).collect::<Vec<_>>(),
            )),
        ]);

        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema.clone(), cols).unwrap();

        let mut buf: Vec<u8> = Vec::new();
        let mut w = ArrowWriter::try_new(&mut buf, schema, None).unwrap();
        w.write(&batch).unwrap();
        w.close().unwrap();
        buf
    }

    fn synthetic_xs(rows: &[SyntheticRow]) -> Vec<u8> {
        synthetic_xs_opt(rows, true)
    }

    #[test]
    fn null_residuals_are_skipped_not_keyed_as_zero() {
        // Two transport-channel rows (null residual) and one real (n,g) channel.
        // Arrow's `value()` on a null returns the raw buffer slot — 0 — so an
        // unguarded read collapses every channel row onto the (0, 0) key and
        // interleaves unrelated curves under it.
        let bytes = synthetic_xs(&[
            (Some(63), Some(GROUND), None, None, SUM, 1.0, 500.0),
            (Some(63), Some(GROUND), None, None, SUM, 2.0, 400.0),
            (Some(63), Some(GROUND), Some(30), Some(64), SUM, 1.0, 100.0),
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
    fn two_target_isomers_do_not_share_a_channel() {
        // #353. Br-80 and Br-80m are different nuclei that share target_A = 80.
        // Before `target_state` joined the key these two rows landed in one
        // XYTable and `sort_paired_vecs` interleaved them by energy, so a lookup
        // at 1.0 MeV returned whichever the sort happened to put first and the
        // other evaluation was unreachable. 37,316 keys in the shipped data
        // collide this way.
        let bytes = synthetic_xs(&[
            (Some(80), Some(GROUND), Some(35), Some(79), SUM, 1.0, 100.0),
            (Some(80), Some("m"), Some(35), Some(79), SUM, 1.0, 250.0),
        ]);
        let db = CrossSectionDb::from_bytes(35, &bytes).unwrap();

        assert_eq!(db.num_reactions(), 2, "the isomers must key separately");
        assert_eq!(
            db.cross_section_target(80, Some(Some(GROUND)), 35, 79, SUM, 1.0),
            100.0
        );
        assert_eq!(
            db.cross_section_target(80, Some(Some("m")), 35, 79, SUM, 1.0),
            250.0,
            "the isomeric target must be reachable, not shadowed by the ground state"
        );
        // The unqualified call resolves to ground rather than to an arbitrary one.
        assert_eq!(db.cross_section(80, 35, 79, 1.0), 100.0);
    }

    #[test]
    fn the_default_state_is_the_live_vocabulary_not_the_retired_one() {
        // #380 retired `state = ''`. `cross_section()` and `entries()` kept
        // defaulting to it, so against every rebuilt shard they returned NAN and
        // an empty vec for *every* reaction — zero rows in the corpus match `''`.
        // Production rows carry `sum`, the state summed over the residual's
        // isomers, which is what an unqualified "how much of this product" means.
        let bytes = synthetic_xs(&[
            (Some(63), Some(GROUND), Some(30), Some(64), SUM, 1.0, 100.0),
            (Some(63), Some(GROUND), Some(30), Some(64), "m", 1.0, 40.0),
        ]);
        let db = CrossSectionDb::from_bytes(29, &bytes).unwrap();

        assert_eq!(
            db.cross_section(63, 30, 64, 1.0),
            100.0,
            "the unqualified lookup must reach the summed production row"
        );
        assert_eq!(db.entries(63, 30, 64), vec![(1.0, 100.0)]);
        assert_eq!(db.cross_section_state(63, 30, 64, "m", 1.0), 40.0);
        assert!(
            db.cross_section_state(63, 30, 64, "", 1.0).is_nan(),
            "the retired spelling must not resolve to anything"
        );
    }

    #[test]
    fn natural_targets_key_a_null_state() {
        // `target_state IS NULL` <=> `target_A = 0` holds across every shipped
        // shard that carries the column: a natural element is a mixture and has
        // no one isomeric state, so a null is the honest value rather than a
        // stand-in ground state (representation principle 3).
        let bytes = synthetic_xs(&[(Some(0), None, Some(30), Some(64), SUM, 1.0, 77.0)]);
        let db = CrossSectionDb::from_bytes(29, &bytes).unwrap();

        assert_eq!(
            db.reaction_keys().next().unwrap().1,
            None,
            "a natural element must not be keyed as ground"
        );
        assert_eq!(db.cross_section(0, 30, 64, 1.0), 77.0);
        assert_eq!(
            db.cross_section_target(0, Some(None), 30, 64, SUM, 1.0),
            77.0
        );
        assert!(
            db.cross_section_target(0, Some(Some(GROUND)), 30, 64, SUM, 1.0)
                .is_nan(),
            "naming a state explicitly must not fall back to the null key"
        );
    }

    #[test]
    fn a_shard_without_the_column_is_still_reachable() {
        // endfb-8.0 and the EXFOR tables have no `target_state` column at all.
        // Every row keys as `None`, and an unqualified lookup has to reach them
        // or the fix for #353 would strand three libraries.
        let bytes = synthetic_xs_opt(
            &[(Some(63), None, Some(30), Some(64), SUM, 1.0, 100.0)],
            false,
        );
        let db = CrossSectionDb::from_bytes(29, &bytes).unwrap();

        assert_eq!(
            db.cross_section(63, 30, 64, 1.0),
            100.0,
            "a legacy shard must not become unreachable"
        );
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
            (None, Some(GROUND), Some(30), Some(64), SUM, 1.0, 100.0),
            (Some(63), Some(GROUND), Some(31), Some(65), SUM, 1.0, 200.0),
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
        let bytes = synthetic_xs(&[(Some(238), Some(GROUND), None, None, SUM, 1.0, 500.0)]);
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
        for (ta, ts, rz, ra, st) in db_file.reaction_keys().take(3) {
            let val_file = db_file.cross_section_target(ta, Some(ts), rz, ra, st, 15.0);
            let val_bytes = db_bytes.cross_section_target(ta, Some(ts), rz, ra, st, 15.0);
            assert!(
                (val_file.is_nan() && val_bytes.is_nan()) || (val_file - val_bytes).abs() < 1e-12,
                "mismatch for ({ta},{rz},{ra}): {val_file} vs {val_bytes}"
            );
        }
    }
}
