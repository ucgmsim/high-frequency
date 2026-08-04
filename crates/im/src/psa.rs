//! Pseudo-spectral acceleration via Newmark-β.
//!
//! Follows the formulation in `ucgmsim/IM_calculation`'s Rust branch. Two choices
//! there are worth restating because they are not the textbook ones:
//!
//! * **Solve for displacement, then differentiate.** The usual presentation
//!   solves for acceleration and integrates twice, which requires the input
//!   difference `a[i+1] - a[i]` — a noisy operation whose noise is then amplified
//!   by double integration. Stochastic high-frequency records are exactly the
//!   input where that matters. Solving implicitly for `u` and differentiating
//!   gives the smoothed response directly.
//! * **β chosen from a stability criterion.** The linear-acceleration method
//!   (β = 1/6) is more accurate but only conditionally stable, up to
//!   `dt/T ≈ 0.551`. Above a conservative 80% of that it falls back to the
//!   unconditionally stable constant-acceleration method (β = 1/4). Without this,
//!   short-period pSA with a coarse timestep diverges.

/// Newmark-β integration constants for the given timestep and frequency.
///
/// Returns `(gamma, beta)`. `gamma` is always 1/2 (second-order accurate, no
/// artificial damping).
fn choose_gamma_beta(dt: f64, w: f64) -> (f64, f64) {
    let gamma = 0.5;
    // Theoretical stability limit of the linear-acceleration method in dt/T.
    let stability_constant = 0.551328895421792;
    // Back off to 80% of it: this leaves a little accuracy on the table for very
    // short periods with large timesteps, in exchange for not sitting on the
    // stability boundary.
    let effective = 0.8 * stability_constant;
    let beta = if dt < effective * (2.0 * std::f64::consts::PI) / w {
        1.0 / 6.0 // linear acceleration
    } else {
        1.0 / 4.0 // constant acceleration, unconditionally stable
    };
    (gamma, beta)
}

/// Peak absolute relative displacement of a unit-mass SDOF oscillator driven by
/// ground acceleration `acc`.
///
/// `w` is the undamped angular frequency, `xi` the damping ratio. Returns
/// `max |u|`, from which pseudo-acceleration is `w² max|u|`.
fn peak_displacement(acc: &[f64], dt: f64, w: f64, xi: f64) -> f64 {
    let nt = acc.len();
    if nt == 0 {
        return 0.0;
    }
    let (gamma, beta) = choose_gamma_beta(dt, w);

    let one_over_beta_dt_sq = 1.0 / (beta * dt * dt);
    let one_over_beta_dt = 1.0 / (beta * dt);
    let k = w * w;
    let c = 2.0 * xi * w;
    let c_gamma_over_beta_dt = (gamma * c) / (beta * dt);
    let one_over_two_beta = 1.0 / (2.0 * beta);

    // Effective stiffness and the coefficients of the effective load, for u_{n+1}.
    let kbar = one_over_beta_dt_sq + k + c_gamma_over_beta_dt;
    let a1 = c_gamma_over_beta_dt + one_over_beta_dt_sq; // u_n
    let b1 = one_over_beta_dt + c * (gamma / beta - 1.0); // udot_n
    let c1 = c * dt * (gamma / (2.0 * beta) - 1.0) + one_over_two_beta - 1.0; // uddot_n

    // Recovering uddot_{n+1}.
    let a2 = one_over_beta_dt_sq; // (u_{n+1} - u_n)
    let b2 = -one_over_beta_dt; // udot_n
    let c2 = -c1; // uddot_n
    // Recovering udot_{n+1}.
    let a3 = 1.0 - gamma; // uddot_n
    let b3 = gamma; // uddot_{n+1}

    let mut u = 0.0f64;
    let mut udot = 0.0f64;
    // Ground motion enters as -a, so the initial acceleration is likewise negated.
    let mut uddot = -acc[0] - (c * udot + k * u);
    let mut umax = u.abs();

    for i in 0..nt - 1 {
        let f_next = -acc[i + 1];
        let pbar = f_next + a1 * u + b1 * udot + c1 * uddot;
        let u_next = pbar / kbar;
        let uddot_next = a2 * (u_next - u) + b2 * udot + c2 * uddot;
        let udot_next = udot + dt * (a3 * uddot + b3 * uddot_next);
        u = u_next;
        udot = udot_next;
        uddot = uddot_next;
        let m = u.abs();
        if m > umax {
            umax = m;
        }
    }
    umax
}

/// 5%-damped pseudo-spectral acceleration at one period, in the input's units.
pub fn psa(acc: &[f64], dt: f64, period: f64, damping: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI / period;
    w * w * peak_displacement(acc, dt, w, damping)
}

/// pSA at each of `periods`.
pub fn psa_set(acc: &[f64], dt: f64, periods: &[f64], damping: f64) -> Vec<f64> {
    periods.iter().map(|&t| psa(acc, dt, t, damping)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn zero_input_gives_zero_response() {
        let a = vec![0.0f64; 1000];
        assert_eq!(psa(&a, 0.005, 1.0, 0.05), 0.0);
    }

    #[test]
    fn beta_switches_on_the_stability_criterion() {
        // T = 1 s at dt = 0.01 is far inside the stable region.
        assert_eq!(choose_gamma_beta(0.01, 2.0 * PI), (0.5, 1.0 / 6.0));
        // T = 0.01 s at dt = 0.01 is dt/T = 1, well past it.
        assert_eq!(choose_gamma_beta(0.01, 100.0 * 2.0 * PI), (0.5, 0.25));
    }

    #[test]
    fn resonant_sine_amplifies_by_one_over_two_xi() {
        // A long sine at the oscillator's own frequency reaches steady-state
        // amplification 1/(2*xi) in pseudo-acceleration. This is a property of the
        // oscillator, not of any particular implementation, so it is a genuine
        // check that the solver is wired up correctly.
        let dt = 0.001;
        let t_n = 0.5;
        let w = 2.0 * PI / t_n;
        let xi = 0.05;
        let n = (60.0 / dt) as usize; // long enough to reach steady state
        let amp = 10.0;
        let a: Vec<f64> = (0..n).map(|i| amp * (w * i as f64 * dt).sin()).collect();

        let got = psa(&a, dt, t_n, xi);
        let want = amp / (2.0 * xi);
        assert!(
            (got / want - 1.0).abs() < 0.02,
            "resonant amplification {got} vs expected {want}"
        );
    }

    #[test]
    fn very_short_period_stays_bounded() {
        // The case the stability fallback exists for: dt/T = 1.
        let dt = 0.01;
        let a: Vec<f64> = (0..5000).map(|i| ((i as f64) * 0.3).sin() * 100.0).collect();
        let v = psa(&a, dt, 0.01, 0.05);
        assert!(v.is_finite() && v < 1.0e6, "short-period pSA diverged: {v}");
    }
}
