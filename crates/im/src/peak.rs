//! Peak amplitudes.

/// Trapezoidal cumulative integral of `y` with step `dt`, starting at zero.
///
/// Shared by PGV, Arias intensity and CAV so the integration rule is identical
/// across them — a mismatch there would show up as a spurious difference between
/// duration and energy measures.
pub fn cumtrapz(y: &[f64], dt: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(y.len());
    let mut acc = 0.0f64;
    out.push(acc);
    for w in y.windows(2) {
        acc += 0.5 * (w[0] + w[1]) * dt;
        out.push(acc);
    }
    out
}

/// Trapezoidal definite integral.
pub fn trapz(y: &[f64], dt: f64) -> f64 {
    if y.len() < 2 {
        return 0.0;
    }
    let interior: f64 = y[1..y.len() - 1].iter().sum();
    (0.5 * (y[0] + y[y.len() - 1]) + interior) * dt
}

/// Peak ground acceleration: `max |a|`, in the input's units.
pub fn pga(acc: &[f64]) -> f64 {
    acc.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

/// Peak ground velocity, from the integrated record.
///
/// No baseline correction is applied. `hb_high` already runs `FLZERO` on its
/// random sequence, and for a *comparison between two simulators* any residual
/// drift is common to both; adding a correction here would introduce a free
/// parameter into the measuring instrument.
pub fn pgv(acc: &[f64], dt: f64) -> f64 {
    let v = cumtrapz(acc, dt);
    v.iter().fold(0.0f64, |m, &x| m.max(x.abs()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trapz_integrates_a_constant() {
        let y = vec![2.0; 101];
        // 100 intervals of 0.1 -> length 10, times height 2.
        assert!((trapz(&y, 0.1) - 20.0).abs() < 1e-12);
    }

    #[test]
    fn cumtrapz_of_a_ramp_is_quadratic() {
        let dt = 0.01;
        let n = 1001;
        let y: Vec<f64> = (0..n).map(|i| i as f64 * dt).collect();
        let c = cumtrapz(&y, dt);
        let t = (n - 1) as f64 * dt;
        // integral of t dt = t^2/2; trapezoid is exact for a linear integrand.
        assert!((c[n - 1] - t * t / 2.0).abs() < 1e-10);
        assert_eq!(c[0], 0.0);
    }

    #[test]
    fn pga_takes_the_absolute_peak() {
        assert_eq!(pga(&[1.0, -5.0, 3.0]), 5.0);
    }

    #[test]
    fn pgv_of_a_sine_matches_the_analytic_amplitude() {
        // a = A sin(wt) integrates to v = (A/w)(1 - cos(wt)), whose peak is 2A/w.
        let dt = 0.0005;
        let w = 2.0 * std::f64::consts::PI * 2.0;
        let a_amp = 100.0;
        let n = (3.0 / dt) as usize;
        let a: Vec<f64> = (0..n).map(|i| a_amp * (w * i as f64 * dt).sin()).collect();
        let got = pgv(&a, dt);
        let want = 2.0 * a_amp / w;
        assert!((got / want - 1.0).abs() < 1e-3, "pgv {got} vs {want}");
    }
}
