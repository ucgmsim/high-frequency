//! Significant duration.
//!
//! These are the measures that catch envelope and timing errors, which pSA is
//! nearly blind to. The one-sample shift from the `stdd(0)` out-of-bounds read is
//! the motivating example: it barely moves a response spectrum but it does move
//! the energy buildup.

use crate::arias::cumulative_arias;

/// Time between two fractions of the total Arias intensity, in seconds.
///
/// Linear interpolation between samples rather than nearest-sample, so the result
/// is not quantised to `dt` — quantisation would hide exactly the sub-sample
/// timing differences these measures exist to detect.
pub fn significant_duration(acc: &[f64], dt: f64, lo: f64, hi: f64) -> f64 {
    let c = cumulative_arias(acc, dt);
    let total = *c.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return 0.0;
    }
    let t_lo = crossing_time(&c, total * lo, dt);
    let t_hi = crossing_time(&c, total * hi, dt);
    (t_hi - t_lo).max(0.0)
}

/// Time at which a nondecreasing curve first reaches `target`, interpolated.
fn crossing_time(c: &[f64], target: f64, dt: f64) -> f64 {
    match c.iter().position(|&v| v >= target) {
        None => (c.len().saturating_sub(1)) as f64 * dt,
        Some(0) => 0.0,
        Some(i) => {
            let (a, b) = (c[i - 1], c[i]);
            let frac = if b > a { (target - a) / (b - a) } else { 0.0 };
            (i as f64 - 1.0 + frac) * dt
        }
    }
}

/// `(Ds575, Ds595)` — the two standard significant durations.
pub fn significant_durations(acc: &[f64], dt: f64) -> (f64, f64) {
    (
        significant_duration(acc, dt, 0.05, 0.75),
        significant_duration(acc, dt, 0.05, 0.95),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boxcar_duration_is_the_expected_fraction() {
        // Constant amplitude over T: the Arias buildup is linear, so Ds5-95 is
        // 90% of T and Ds5-75 is 70%.
        let dt = 0.001;
        let n = 10_001; // T = 10 s
        let a = vec![5.0f64; n];
        let (d575, d595) = significant_durations(&a, dt);
        assert!((d575 - 7.0).abs() < 0.01, "Ds575 {d575}");
        assert!((d595 - 9.0).abs() < 0.01, "Ds595 {d595}");
    }

    #[test]
    fn silence_has_zero_duration() {
        let a = vec![0.0f64; 100];
        assert_eq!(significant_durations(&a, 0.01), (0.0, 0.0));
    }

    #[test]
    fn interpolation_beats_dt_quantisation() {
        // Two boxcars differing by half a sample in start time must give durations
        // that differ, which nearest-sample crossing would not.
        let dt = 0.01;
        let mut a = vec![0.0f64; 1000];
        let mut b = vec![0.0f64; 1000];
        for i in 100..900 { a[i] = 1.0; }
        for i in 100..900 { b[i] = 1.0; }
        b[99] = 0.5; // half a sample of extra energy at the front
        let da = significant_duration(&a, dt, 0.05, 0.95);
        let db = significant_duration(&b, dt, 0.05, 0.95);
        assert_ne!(da, db, "sub-sample energy change must move the duration");
    }
}
