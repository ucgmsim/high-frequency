//! Arias intensity.

use crate::peak::{cumtrapz, trapz};
use crate::G_CM;

/// Arias intensity, `(pi / 2g) * integral(a^2 dt)`, in cm/s.
///
/// `g` is in cm/s² because `hb_high` writes cm/s². Using the m/s² value would
/// rescale the result by 100 without any other symptom.
pub fn arias_intensity(acc: &[f64], dt: f64) -> f64 {
    let sq: Vec<f64> = acc.iter().map(|x| x * x).collect();
    (std::f64::consts::PI / (2.0 * G_CM)) * trapz(&sq, dt)
}

/// The Arias buildup curve, same units, used by the duration measures.
pub fn cumulative_arias(acc: &[f64], dt: f64) -> Vec<f64> {
    let sq: Vec<f64> = acc.iter().map(|x| x * x).collect();
    let k = std::f64::consts::PI / (2.0 * G_CM);
    cumtrapz(&sq, dt).into_iter().map(|v| k * v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cumulative_ends_at_the_total() {
        let dt = 0.005;
        let a: Vec<f64> = (0..2000).map(|i| (i as f64 * 0.05).sin() * 50.0).collect();
        let total = arias_intensity(&a, dt);
        let c = cumulative_arias(&a, dt);
        assert!((c[c.len() - 1] / total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn monotonically_nondecreasing() {
        let dt = 0.005;
        let a: Vec<f64> = (0..500).map(|i| (i as f64).sin() * 10.0).collect();
        let c = cumulative_arias(&a, dt);
        assert!(c.windows(2).all(|w| w[1] >= w[0]), "Arias buildup must not decrease");
    }
}
