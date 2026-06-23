//! Cross-language golden-file parity test for the Rust crate.
//!
//! Reads the JSON fixtures at `tests/golden/fixtures/` (canonical Python
//! output) and asserts the Rust crate produces row-for-row equivalent
//! results after the documented normalization (sort + float rounding +
//! NULL convention).
//!
//! Per #176 / `docs/NULL_CONVENTIONS.md`. Re-run
//! `uv run python tests/golden/generate.py` to refresh fixtures.

use std::path::PathBuf;

use nucl_parquet::{
    CoincidenceFilter, CoincidencesDb, Emission, EmissionEntry, GammaCandidate, RadiationDb,
    StoppingDb,
};
use serde::Serialize;
use serde_json::{json, Value};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("data")
}

fn meta_dir() -> PathBuf {
    data_dir().join("meta")
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("..")
        .join("tests")
        .join("golden")
        .join("fixtures")
        .join(format!("{name}.json"))
}

/// Round to 6 decimal places, matching the Python normalize.py rule.
fn round6(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    (v * 1e6).round() / 1e6
}

/// Convert f32 → JSON value, applying the float-precision rule. `+inf`
/// becomes the JSON string "Infinity" per docs/NULL_CONVENTIONS.md.
fn f32_to_json(v: f32) -> Value {
    let v = v as f64;
    if v.is_infinite() {
        return Value::String(if v > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        });
    }
    json!(round6(v))
}

fn f64_to_json(v: f64) -> Value {
    if v.is_infinite() {
        return Value::String(if v > 0.0 {
            "Infinity".into()
        } else {
            "-Infinity".into()
        });
    }
    json!(round6(v))
}

fn opt_f64_to_json(v: Option<f64>) -> Value {
    match v {
        Some(x) => f64_to_json(x),
        None => Value::Null,
    }
}

fn opt_f32_to_json(v: Option<f32>) -> Value {
    match v {
        Some(x) => f32_to_json(x),
        None => Value::Null,
    }
}

fn opt_string_to_json(s: Option<&str>) -> Value {
    match s {
        Some(x) => Value::String(x.to_string()),
        None => Value::Null,
    }
}

#[derive(Serialize)]
struct CoincRow {
    #[serde(rename = "Z")]
    z: u32,
    #[serde(rename = "A")]
    a: u32,
    parent_state: String,
    parent_decay_mode: Value,
    #[serde(rename = "daughter_ex_keV")]
    daughter_ex_kev: Value,
    emission1_rad_type: String,
    #[serde(rename = "emission1_energy_keV")]
    emission1_energy_kev: Value,
    emission1_intensity: Value,
    emission1_shell: Value,
    emission2_rad_type: String,
    #[serde(rename = "emission2_energy_keV")]
    emission2_energy_kev: Value,
    emission2_intensity: Value,
    emission2_shell: Value,
    pair_intensity: Value,
}

fn coinc_to_json(
    z: u32,
    a: u32,
    parent_state: &str,
    parent_decay_mode: Option<&str>,
    daughter_ex_kev: Option<f64>,
    emission1: &Emission,
    emission2: &Emission,
    pair_intensity: f32,
) -> Value {
    serde_json::to_value(CoincRow {
        z,
        a,
        parent_state: parent_state.to_string(),
        parent_decay_mode: opt_string_to_json(parent_decay_mode),
        daughter_ex_kev: opt_f64_to_json(daughter_ex_kev),
        emission1_rad_type: emission1.rad_type.clone(),
        emission1_energy_kev: f64_to_json(emission1.energy_kev),
        emission1_intensity: f32_to_json(emission1.intensity),
        emission1_shell: opt_string_to_json(emission1.shell.as_deref()),
        emission2_rad_type: emission2.rad_type.clone(),
        emission2_energy_kev: f64_to_json(emission2.energy_kev),
        emission2_intensity: f32_to_json(emission2.intensity),
        emission2_shell: opt_string_to_json(emission2.shell.as_deref()),
        pair_intensity: f32_to_json(pair_intensity),
    })
    .expect("serialize CoincRow")
}

fn sort_coinc(rows: &mut [Value]) {
    // Matches the Python normalize.py sort key. Ties on the (Z, A, channel,
    // emission_energies) prefix are broken on daughter_ex_keV / shells /
    // pair_intensity — multiple cascade rows can share the same channel +
    // emission energies but differ on which level fed them.
    rows.sort_by(|a, b| {
        let key = |v: &Value| {
            (
                v["Z"].as_u64().unwrap_or(0),
                v["A"].as_u64().unwrap_or(0),
                v["parent_state"].as_str().unwrap_or("").to_string(),
                v["parent_decay_mode"].as_str().unwrap_or("").to_string(),
                v["emission1_rad_type"].as_str().unwrap_or("").to_string(),
                v["emission1_energy_keV"].as_f64().unwrap_or(0.0).to_bits(),
                v["emission2_rad_type"].as_str().unwrap_or("").to_string(),
                v["emission2_energy_keV"].as_f64().unwrap_or(0.0).to_bits(),
                v["daughter_ex_keV"].as_f64().unwrap_or(0.0).to_bits(),
                v["emission1_shell"].as_str().unwrap_or("").to_string(),
                v["emission2_shell"].as_str().unwrap_or("").to_string(),
                v["pair_intensity"].as_f64().unwrap_or(0.0).to_bits(),
            )
        };
        key(a).cmp(&key(b))
    });
}

fn load_fixture(name: &str) -> Value {
    let path = fixture_path(name);
    let s = std::fs::read_to_string(&path)
        .unwrap_or_else(|_| panic!("fixture not found: {}", path.display()));
    serde_json::from_str(&s).expect("parse fixture JSON")
}

fn coincidences_filtered_rows(
    db: &CoincidencesDb,
    z: u32,
    a: u32,
    filter: CoincidenceFilter,
) -> Vec<Value> {
    let pairs = db.pairs_filtered(z, a, &filter).expect("pairs_filtered");
    let mut rows: Vec<Value> = pairs
        .iter()
        .filter(|e| e.pair_intensity > filter.min_intensity)
        .map(|e| {
            coinc_to_json(
                e.z,
                e.a,
                &e.parent_state,
                e.parent_decay_mode.as_deref(),
                e.daughter_ex_kev,
                &e.emission1,
                &e.emission2,
                e.pair_intensity,
            )
        })
        .collect();
    sort_coinc(&mut rows);
    rows
}

#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_co60_beta_gamma() {
    let db = CoincidencesDb::open(meta_dir()).unwrap();
    let actual = coincidences_filtered_rows(
        &db,
        28,
        60,
        CoincidenceFilter {
            parent_state: Some(String::new()),
            parent_decay_mode: Some("beta-".into()),
            emission1_rad_type: Some("beta".into()),
            emission2_rad_type: Some("gamma".into()),
            ..Default::default()
        },
    );
    let golden = load_fixture("co60_beta_gamma");
    assert_eq!(Value::Array(actual), golden, "Co-60 β-γ goldens diverged");
}

#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_y86_kshell_xray_gamma() {
    let db = CoincidencesDb::open(meta_dir()).unwrap();
    let actual = coincidences_filtered_rows(
        &db,
        38,
        86,
        CoincidenceFilter {
            parent_state: Some(String::new()),
            parent_decay_mode: Some("KshellEC".into()),
            emission1_rad_type: Some("xray".into()),
            emission2_rad_type: Some("gamma".into()),
            min_intensity: 1e-5,
            ..Default::default()
        },
    );
    let golden = load_fixture("y86_kshell_xray_gamma");
    assert_eq!(
        Value::Array(actual),
        golden,
        "Y-86 K X-ray ⊗ γ goldens diverged"
    );
}

#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_co60_gamma_gamma() {
    let db = CoincidencesDb::open(meta_dir()).unwrap();
    let actual = coincidences_filtered_rows(
        &db,
        28,
        60,
        CoincidenceFilter {
            parent_state: Some(String::new()),
            parent_decay_mode: Some("beta-".into()),
            emission1_rad_type: Some("gamma".into()),
            emission2_rad_type: Some("gamma".into()),
            min_intensity: 0.5,
            ..Default::default()
        },
    );
    let golden = load_fixture("co60_gamma_gamma");
    assert_eq!(Value::Array(actual), golden, "Co-60 γ-γ goldens diverged");
}

#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_sr90_y90_negative() {
    let db = CoincidencesDb::open(meta_dir()).unwrap();
    let pairs = db
        .pairs_filtered(
            39,
            90,
            &CoincidenceFilter {
                parent_decay_mode: Some("beta-".into()),
                ..Default::default()
            },
        )
        .unwrap();
    let mixed: Vec<_> = pairs
        .iter()
        .filter(|e| e.emission1.rad_type != "gamma")
        .collect();
    let actual: Vec<Value> = mixed
        .into_iter()
        .map(|e| {
            coinc_to_json(
                e.z,
                e.a,
                &e.parent_state,
                e.parent_decay_mode.as_deref(),
                e.daughter_ex_kev,
                &e.emission1,
                &e.emission2,
                e.pair_intensity,
            )
        })
        .collect();
    let golden = load_fixture("sr90_y90_negative");
    assert_eq!(
        Value::Array(actual),
        golden,
        "Sr-90 negative-case goldens diverged"
    );
}

fn emission_to_json(e: &EmissionEntry) -> Value {
    json!({
        "Z": e.z,
        "A": e.a,
        "state": e.state,
        "rad_type": e.rad_type,
        "energy_keV": f64_to_json(e.energy_kev),
        "intensity_pct": f64_to_json(e.intensity_pct),
        "decay_mode": opt_string_to_json(e.decay_mode.as_deref()),
        "rad_subtype": opt_string_to_json(e.rad_subtype.as_deref()),
        "icc_total": opt_f32_to_json(e.icc_total),
        "vacancy_shell": opt_string_to_json(e.vacancy_shell.as_deref()),
    })
}

#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_ni60_emissions() {
    let db = RadiationDb::open(meta_dir()).unwrap();
    let lines = db.emissions(28, 60, "").unwrap();
    let mut rows: Vec<Value> = lines
        .iter()
        .filter(|e| e.intensity_pct >= 5.0)
        .map(emission_to_json)
        .collect();
    rows.sort_by(|a, b| {
        let key = |v: &Value| {
            (
                v["Z"].as_u64().unwrap_or(0),
                v["A"].as_u64().unwrap_or(0),
                v["state"].as_str().unwrap_or("").to_string(),
                v["rad_type"].as_str().unwrap_or("").to_string(),
                v["energy_keV"].as_f64().unwrap_or(0.0).to_bits(),
            )
        };
        key(a).cmp(&key(b))
    });
    let golden = load_fixture("ni60_emissions");
    assert_eq!(
        Value::Array(rows),
        golden,
        "Ni-60 emissions goldens diverged"
    );
}

fn gamma_to_json(c: &GammaCandidate) -> Value {
    json!({
        "Z": c.z,
        "A": c.a,
        "state": c.state,
        "energy_keV": f64_to_json(c.energy_kev),
        "intensity_pct": f64_to_json(c.intensity_pct),
        "delta_keV": f64_to_json(c.delta_kev),
    })
}

#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_identify_gamma_1173() {
    let db = RadiationDb::open(meta_dir()).unwrap();
    let mut candidates = db.identify_gamma(1173.2, 1.0, 0.1).expect("identify_gamma");
    // Re-sort to match the Python helper's normalization. Compare
    // `|delta|` after rounding to 6 decimal places so float-precision drift
    // (e.g. `1173.1 - 1173.2 = -0.09999999…` vs the rounded `0.1` in the
    // fixture) doesn't break the sort. Final (Z, A) tiebreak is deterministic.
    let round6 = |v: f64| (v.abs() * 1e6).round() / 1e6;
    candidates.sort_by(|a, b| {
        round6(a.delta_kev)
            .partial_cmp(&round6(b.delta_kev))
            .unwrap()
            .then(b.intensity_pct.partial_cmp(&a.intensity_pct).unwrap())
            .then(a.z.cmp(&b.z))
            .then(a.a.cmp(&b.a))
    });
    let mut rows: Vec<Value> = candidates.iter().map(gamma_to_json).collect();
    // Python COINCIDENCE_SQL via IDENTIFY_GAMMA_SQL returns 20-row limited result.
    // The Rust implementation is unbounded; trim to match the fixture's cardinality.
    let golden_arr = load_fixture("identify_gamma_1173keV");
    let n = golden_arr.as_array().unwrap().len();
    rows.truncate(n);
    assert_eq!(
        Value::Array(rows),
        golden_arr,
        "identify_gamma(1173.2) goldens diverged"
    );
}

/// CatIMA per-isotope stopping-power parity (#246).
///
/// Asserts `StoppingDb::catima_dedx(proj_z, proj_a, target_z, …)` reproduces the
/// Python loader's log-log-interpolated catima value for each fixture row. The
/// fixture deliberately includes same-Z isotope pairs (He-3/He-4, C-12/C-13) at
/// low energy — the case the old `(proj_Z, target_Z)`-keyed lookup corrupted by
/// merging isotopes onto one energy axis.
#[test]
#[ignore = "requires nucl-parquet data files and committed golden fixtures"]
fn golden_catima_stopping() {
    let db = StoppingDb::open(data_dir().join("stopping")).unwrap();
    let golden = load_fixture("catima_stopping");
    let rows = golden.as_array().expect("catima fixture is an array");
    assert!(!rows.is_empty(), "catima fixture is empty");

    for row in rows {
        let pz = row["proj_Z"].as_u64().unwrap() as u32;
        let pa = row["proj_A"].as_u64().unwrap() as u32;
        let tz = row["target_Z"].as_u64().unwrap() as u32;
        let eu = row["energy_MeV_u"].as_f64().unwrap();
        let expected = row["dedx"].as_f64().unwrap();

        let got = db.catima_dedx(pz, pa, tz, eu);
        assert!(
            got.is_finite(),
            "catima_dedx({pz},{pa},{tz},{eu}) = {got} (expected ~{expected})"
        );
        // Compare at the fixture's 6-decimal precision (the harness convention);
        // the underlying agreement is ~1e-15 relative. The isotope-merge bug this
        // guards (#246) produces percent-level errors, so it can't slip past.
        assert!(
            (round6(got) - expected).abs() < 1e-9,
            "catima parity drift at (Z={pz}, A={pa}, target={tz}, {eu} MeV/u): \
             Rust {got}, Python {expected}"
        );
    }
}
