use arrow::array::{Array, ArrayRef, LargeStringArray, StringArray};

/// Parallel x/y f64 vectors — ubiquitous table shape for interpolation (energy, value).
pub type XYTable = (Vec<f64>, Vec<f64>);

/// Try to downcast an Arrow array column to StringArray or LargeStringArray.
///
/// Polars writes Utf8 as LargeUtf8 in some cases, so we handle both.
pub fn as_string_array(col: &ArrayRef) -> Option<Vec<Option<&str>>> {
    if let Some(arr) = col.as_any().downcast_ref::<StringArray>() {
        Some(
            (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect(),
        )
    } else {
        col.as_any().downcast_ref::<LargeStringArray>().map(|arr| {
            (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(arr.value(i))
                    }
                })
                .collect()
        })
    }
}

/// Sort two parallel `Vec<f64>` arrays together by the first (key) array.
///
/// Used when loading tabular data to ensure the key array is in ascending
/// order before binary-search interpolation.
pub fn sort_paired_vecs(keys: &mut Vec<f64>, values: &mut Vec<f64>) {
    debug_assert_eq!(keys.len(), values.len());
    let mut indices: Vec<usize> = (0..keys.len()).collect();
    indices.sort_by(|&a, &b| keys[a].partial_cmp(&keys[b]).unwrap());
    let sorted_keys: Vec<f64> = indices.iter().map(|&i| keys[i]).collect();
    let sorted_vals: Vec<f64> = indices.iter().map(|&i| values[i]).collect();
    *keys = sorted_keys;
    *values = sorted_vals;
}

/// Log-log interpolation on sorted energy/value arrays.
///
/// Both arrays must be the same length and sorted by energy (ascending).
/// Returns NaN if energy is outside the tabulated range.
#[inline]
pub fn log_log_interp(energies: &[f64], values: &[f64], energy: f64) -> f64 {
    debug_assert_eq!(energies.len(), values.len());

    let n = energies.len();
    if n == 0 || energy < 0.0 {
        return f64::NAN;
    }

    // Clamp to table range
    if energy <= energies[0] {
        return values[0];
    }
    if energy >= energies[n - 1] {
        return values[n - 1];
    }

    // Binary search for the bracketing interval
    let idx = match energies.binary_search_by(|e| e.partial_cmp(&energy).unwrap()) {
        Ok(i) => return values[i], // exact match
        Err(i) => i - 1,           // energy is between [i-1, i]
    };

    let e0 = energies[idx];
    let e1 = energies[idx + 1];
    let v0 = values[idx];
    let v1 = values[idx + 1];

    // Handle zero or negative values/energies (can't take log)
    if v0 <= 0.0 || v1 <= 0.0 || e0 <= 0.0 || e1 <= 0.0 {
        // Fall back to linear interpolation
        let denom = e1 - e0;
        if denom.abs() < f64::MIN_POSITIVE {
            return v0;
        }
        let t = (energy - e0) / denom;
        return v0 + t * (v1 - v0);
    }

    // Log-log interpolation: log(v) = log(v0) + (log(v1)-log(v0)) * (log(E)-log(E0)) / (log(E1)-log(E0))
    let log_e = energy.ln();
    let log_e0 = e0.ln();
    let log_e1 = e1.ln();
    let log_v0 = v0.ln();
    let log_v1 = v1.ln();

    let t = (log_e - log_e0) / (log_e1 - log_e0);
    (log_v0 + t * (log_v1 - log_v0)).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match() {
        let e = [1.0, 10.0, 100.0];
        let v = [100.0, 10.0, 1.0];
        assert_eq!(log_log_interp(&e, &v, 10.0), 10.0);
    }

    #[test]
    fn midpoint_power_law() {
        // For a perfect power law v = 1000/E, log-log interp is exact
        let e = [1.0, 1000.0];
        let v = [1000.0, 1.0];
        let result = log_log_interp(&e, &v, 10.0);
        // v(10) should be 100
        assert!((result - 100.0).abs() < 1e-10);
    }

    #[test]
    fn clamp_below() {
        let e = [1.0, 10.0];
        let v = [100.0, 10.0];
        assert_eq!(log_log_interp(&e, &v, 0.5), 100.0);
    }

    #[test]
    fn clamp_above() {
        let e = [1.0, 10.0];
        let v = [100.0, 10.0];
        assert_eq!(log_log_interp(&e, &v, 20.0), 10.0);
    }

    #[test]
    fn q_zero_form_factor() {
        // q=0 is physically valid (forward scattering); must not be rejected.
        let e = [1.0, 10.0];
        let v = [100.0, 10.0];
        assert_eq!(log_log_interp(&e, &v, 0.0), 100.0);
    }

    #[test]
    fn zero_first_energy_falls_back_to_linear() {
        // Table starting at 0 would produce ln(0) = -inf → NaN under pure log-log.
        // Interval [0, 10] with query at 5 must fall back to linear: 0 + 0.5 * (10 - 0) = 5.
        let e = [0.0, 10.0];
        let v = [0.0, 10.0];
        let result = log_log_interp(&e, &v, 5.0);
        assert!(result.is_finite());
        assert!((result - 5.0).abs() < 1e-10);
    }

    #[test]
    fn negative_energy_returns_nan() {
        let e = [1.0, 10.0];
        let v = [100.0, 10.0];
        assert!(log_log_interp(&e, &v, -1.0).is_nan());
    }
}
