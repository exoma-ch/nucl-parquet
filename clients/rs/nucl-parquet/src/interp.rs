/// Log-log interpolation on sorted energy/value arrays.
///
/// Both arrays must be the same length and sorted by energy (ascending).
/// Returns NaN if energy is outside the tabulated range.
#[inline]
pub fn log_log_interp(energies: &[f64], values: &[f64], energy: f64) -> f64 {
    debug_assert_eq!(energies.len(), values.len());

    let n = energies.len();
    if n == 0 || energy <= 0.0 {
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

    // Handle zero or negative values (can't take log)
    if v0 <= 0.0 || v1 <= 0.0 {
        // Fall back to linear interpolation
        let t = (energy - e0) / (e1 - e0);
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
}
