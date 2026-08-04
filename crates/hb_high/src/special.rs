//! `DGAMM` — the gamma function.

/// Chebyshev-like coefficients from the `DATA C` block at `hb_high_ref.f:2760`.
///
/// Transcribed with the embedded blanks removed — fixed-form Fortran ignores
/// blanks inside numeric literals, so `-0.42278 43350 98467 1 D0` is
/// `-0.4227843350984671`. The array is declared `C(0:19)` in the Fortran, i.e.
/// already 0-based, so this indexes identically.
const C: [f64; 20] = [
    1.0,
    -0.4227843350984671,
    -0.2330937364217867,
    0.1910911013876915,
    -0.2455249000540002e-1,
    -0.1764524455014432e-1,
    0.8023273022267347e-2,
    -0.8043297756042470e-3,
    -0.3608378162548e-3,
    0.1455961421399e-3,
    -0.175458597517e-4,
    -0.25889950224e-5,
    0.13385015466e-5,
    -0.2054743152e-6,
    -0.1595268e-9,
    0.62756218e-8,
    -0.12736143e-8,
    0.923397e-10,
    0.120028e-10,
    -0.42202e-11,
];

/// Highest index of `C` used, matching `DATA IN / 19 /`. The source notes a
/// single-precision variant would use 10; we are always double.
const IN: usize = 19;

/// The sentinel the Fortran returns on both error paths.
pub const GAMMA_ERROR: f64 = 1.0e75;

/// `FUNCTION DGAMM(X)` — `hb_high_ref.f:2745`. Gamma function, `real*8`.
///
/// Argument reduction to `[-0.5, 0.5]` followed by Horner evaluation.
///
/// Two error paths in the original write a diagnostic to unit 6 (stdout) and
/// return `1.0D75`: `x > 57`, and a non-positive integer argument where gamma
/// has a pole. Both are reproduced, including the return value, because
/// `stochastic_spectrum` does not check for it — so the sentinel propagates into the
/// spectrum and any change here would change output. The message goes to
/// stderr rather than stdout: unit 6 would corrupt nothing today (the waveform
/// goes to unit 22 and the distance to unit 0), but stdout is not a channel
/// this program otherwise uses, and `hf_sim.py` captures stderr.
pub fn gamma(x: f64) -> f64 {
    if x > 57.0 {
        eprintln!(" (FUNC.DGAMM) X(={x:.16E}) MUST BE SMALLER THAN 57.0");
        return GAMMA_ERROR;
    }

    let mut xx = x;
    let a: f64;
    let fctr: f64;

    if xx <= 1.5 {
        if xx >= 0.5 {
            a = xx - 1.0;
            fctr = 1.0;
        } else {
            // INT truncates toward zero, so for negative xx this rounds up.
            let m = xx.trunc() as i32;
            let mut aa = xx - m as f64;
            if aa == 0.0 {
                // Pole: gamma is undefined at non-positive integers.
                eprintln!(" (FUNC.DGAMM) INVALID ARGUMENT X ={x:.16E}");
                return GAMMA_ERROR;
            }
            let mg = if aa >= -0.5 {
                m.abs() + 1
            } else {
                aa += 1.0;
                m.abs() + 2
            };
            let mut z = 1.0f64;
            for _ in 1..=mg {
                z *= xx;
                xx += 1.0;
            }
            a = aa;
            fctr = 1.0 / z;
        }
    } else {
        let m = xx.trunc() as i32;
        let mut aa = xx - m as f64;
        let mg = if aa <= 0.5 {
            m - 1
        } else {
            aa -= 1.0;
            m
        };
        let mut z = 1.0f64;
        for _ in 1..=mg {
            z *= xx - 1.0;
            xx -= 1.0;
        }
        a = aa;
        fctr = z;
    }

    let mut y = C[IN];
    for i in (0..=IN - 1).rev() {
        y = C[i] + a * y;
    }

    fctr / ((1.0 + a) * y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_of_small_integers() {
        // Gamma(n) = (n-1)!
        for (n, want) in [
            (1.0, 1.0),
            (2.0, 1.0),
            (3.0, 2.0),
            (4.0, 6.0),
            (5.0, 24.0),
            (6.0, 120.0),
            (10.0, 362880.0),
        ] {
            let got = gamma(n);
            assert!(
                (got / want - 1.0).abs() < 1e-12,
                "gamma({n}) = {got}, want {want}"
            );
        }
    }

    #[test]
    fn gamma_of_half() {
        // Gamma(1/2) = sqrt(pi)
        let got = gamma(0.5);
        let want = std::f64::consts::PI.sqrt();
        assert!(
            (got / want - 1.0).abs() < 1e-12,
            "gamma(0.5) = {got}, want {want}"
        );
    }

    #[test]
    fn error_paths_return_sentinel() {
        assert_eq!(gamma(58.0), GAMMA_ERROR);
        assert_eq!(gamma(0.0), GAMMA_ERROR);
        assert_eq!(gamma(-3.0), GAMMA_ERROR);
    }
}
