//! Cumulative absolute velocity.

use crate::peak::trapz;

/// CAV, `integral |a| dt`, in cm/s for cm/s² input.
///
/// The unstandardised form: no 0.025 g threshold is applied. For comparing two
/// simulators a threshold would only discard information and add a parameter.
pub fn cav(acc: &[f64], dt: f64) -> f64 {
    let abs: Vec<f64> = acc.iter().map(|x| x.abs()).collect();
    trapz(&abs, dt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cav_of_a_constant_is_height_times_duration() {
        let y = vec![-3.0f64; 101];
        assert!((cav(&y, 0.1) - 30.0).abs() < 1e-12);
    }
}
