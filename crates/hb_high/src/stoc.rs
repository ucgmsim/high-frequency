//! `stochastic_spectrum` — the stochastic source spectrum for one subfault.

use crate::fft::{fast, remove_quadratic_trend};
use crate::fort::{Array1, Complex32, Complex64};
use crate::rng::{fill_normal_deviates, Pcg32};
use crate::special::gamma;

/// `subroutine stochastic_spectrum(...)` — `hb_high_ref.f:1670`.
///
/// Builds the complex Fourier spectrum of one subfault's stochastic S-wave
/// motion: a Brune omega-squared source, a kappa/fmax high-cut, path Q, and the
/// Frankel two-corner operator, multiplied by a unit-power random phase
/// spectrum and mirrored to Hermitian symmetry. Writes `np2` values into `spectrum`.
///
/// `dlm` is declared and never used; kept in the signature for call-site parity.
///
/// # Precision layout
///
/// `a1`, `a2`, `a3`, `gsa`, `gm` and the `as` array are `real*8`; everything
/// else is `real*4`. So each `as(i)` term is *computed* in single precision and
/// then widened, and only the final product `a1*a2*a3*frank` accumulates in
/// double. The `as f64` casts below are exactly those widening points.
///
/// Two subtleties verified against gfortran 16.1.1 rather than assumed:
///
/// * `complex*8 * real*8` promotes the **complex** operand to `complex*16` and
///   multiplies in double, narrowing only on assignment. Doing the whole product
///   in `f32` differs in the last bit, so `spectrum(i) = ac(i)*as(i)*amp` is built
///   through [`Complex64`] here.
/// * Constant exponents need care, and the two cases here differ. gfortran folds
///   `x**(-1.0)` into a reciprocal — verified identical to `1.0/x` over 200,000
///   values — but Rust's `powf(-1.0)` is a libm call that disagrees with `1.0/x`
///   in about 1 case in 1,600. So `**(-1.0)` is written as an explicit division.
///   `x**0.5`, by contrast, gfortran does *not* fold: it calls `powf`. Rust
///   cannot express that portably — LLVM folds `powf(x, 0.5)` to `sqrt(x)` at
///   `-O2` but not at `-O0`, so the result would depend on optimisation level.
///   The only `x**0.5` in this routine feeds a dead store and is simply not
///   computed. See `PORTING_RULES.md` §4b.
///
/// # The random sequence must be unit-RMS
///
/// `amp = 1/(dt*sqrt(fsa/fold_count))` normalises so the average *power* spectrum is
/// unity, per Boore (1983) — a 2009-03-18 change from normalising the amplitude
/// spectrum, which reduced motions about 10% and was offset by raising the
/// default corner frequency 5%. This calibration assumes
/// [`normal_deviates`]'s renormalisation, so substituting a plain N(0,1)
/// generator would silently change the output level.
#[allow(clippy::too_many_arguments)]
pub fn stochastic_spectrum(
    rng: &mut Pcg32,
    np2: usize,
    distance_km: f32,
    window_s: f32,
    window_eps: f32,
    window_eta: f32,
    shear_velocity_km_s: f32,
    density_g_cm3: f32,
    dt: f32,
    subevent_moment: f32,
    _avg_subfault_km: f32,
    corner_frequency_hz: f32,
    fmax_hz: f32,
    kappa_s: f32,
    spectrum: &mut Array1<Complex32>,
    frequency_hz: &Array1<f32>,
    qbar: f32,
    q_exponent: f32,
    moment_scale: f32,
) {
    let pai = 3.1415926f32;
    let rp = 0.63f32;

    let fc2 = corner_frequency_hz * corner_frequency_hz;

    // fs is the free-surface factor; prtitn the vector partition factor for two
    // orthogonal components, nominally 1/sqrt(2) but written as two digits.
    let fs = 2.0f32;
    let prtitn = 0.71f32;

    let fold_count = np2 / 2 + 1;
    let distance_cm = distance_km * 100000.0;

    // Saragoni-Hart style envelope: b and c from the (window_eps, window_eta) window shape.
    let b = -window_eps * window_eta.ln() / (1.0 + window_eps * (window_eps.ln() - 1.0));
    let c = b / window_eps / window_s;
    // Computed in real*4, then widened -- gsa is real*8 but 2*b+1.0 is not.
    let gsa = (2.0 * b + 1.0) as f64;
    let gm = gamma(gsa);
    // The power is real*4; the division by gm and the sqrt are real*8; the
    // result narrows back to real*4.
    let aa = (((2.0 * c).powf(2.0 * b + 1.0) as f64) / gm).sqrt() as f32;

    let mut w = Array1::<f32>::new(np2);
    for i in 1..=np2 {
        let t = (i - 1) as f32 * dt;
        w[i] = aa * t.powf(b) * (-c * t).exp();
    }

    let beta = shear_velocity_km_s * 100000.0;
    let cc = rp * fs * prtitn / (4.0 * pai * density_g_cm3 * (beta * beta * beta));
    let omgc = 2.0 * pai * corner_frequency_hz;
    let omgm = 2.0 * pai * fmax_hz;

    let mut as_ = Array1::<f64>::new(np2);
    as_[1] = 0.0;
    for i in 2..=fold_count {
        let fr = frequency_hz[i];
        let fr2 = fr * fr;

        // The Q model qv = 150.0*fr**0.5 is computed by the Fortran but feeds
        // only the first, dead, a3 form below, so it is not computed here.
        // Earlier variants in the source: 100+10*fr**1.70 and 270*fr**0.5
        // ("Beresnev Northridge").
        //
        // Dropping it also removes the port's last CONSTANT-exponent powf.
        // That matters: LLVM folds powf(x, 0.5) into sqrt(x) at -O2 but not at
        // -O0, and gfortran's x**0.5 is a real powf call, so keeping it would
        // have made the port's output depend on optimisation level. See
        // PORTING_RULES.md §4b.

        let omg = 2.0 * pai * fr;
        let a1 = (cc * subevent_moment * (omg * omg / (1.0 + (omg / omgc) * (omg / omgc)))) as f64;

        let a2 = if kappa_s <= 0.0 {
            // (1.0 + (omg/omgm)**1)**(-1.0). The **1 is the identity, and
            // gfortran folds **(-1.0) to a reciprocal -- so this must be a
            // division, NOT powf(-1.0), which differs. See PORTING_RULES.md §4b.
            (1.0 / (1.0 + (omg / omgm))) as f64
        } else {
            (-pai * fr * kappa_s).exp() as f64
        };

        // The Fortran assigns a3 three times; only the last survives. The two
        // dead stores are recorded here rather than executed -- they are pure
        // and immediately overwritten, and each costs an exp() per frequency
        // bin on a path called three times per subfault:
        //     a3 = exp(-omg*distance_cm/(2.0*qv*beta))/distance_cm        ! fixed Q model
        //     a3 = exp(-0.5*omg*qbar*fr**(-0.5))/distance_cm        ! qbar, fixed exponent
        let a3 = ((-0.5 * omg * qbar * fr.powf(-q_exponent)).exp() / distance_cm) as f64;

        // Frankel two-corner convolution operator.
        let frank = moment_scale * (fc2 + fr2) / (fc2 + moment_scale * fr2);

        as_[i] = a1 * a2 * a3 * frank as f64;
    }

    let mut a = Array1::<f32>::new(np2);
    fill_normal_deviates(rng, np2, &mut a);
    remove_quadratic_trend(np2, dt, &mut a);

    let mut ac = Array1::<Complex32>::filled(np2, Complex32::ZERO);
    for i in 1..=np2 {
        ac[i] = Complex32::new(a[i] * w[i], 0.0);
    }

    fast(np2, &mut ac, -1);

    // Average POWER spectrum to unity (2009-03-18), not amplitude.
    let mut fsa = 0.0f32;
    for i in 1..=fold_count {
        fsa += ac[i].abs() * ac[i].abs();
    }
    let amp = 1.0 / (dt * (fsa / fold_count as f32).sqrt());

    // complex*8 * real*8 goes through complex*16; see the note above.
    let scale = |z: Complex32, s: f64, m: f32| -> Complex32 {
        let d = Complex64::new(z.re as f64, z.im as f64) * s * m as f64;
        Complex32::new(d.re as f32, d.im as f32)
    };

    let np = np2 / 2;
    for i in 1..=np {
        spectrum[i] = scale(ac[i], as_[i], amp);
        // conjg() is applied to the complex*16 product, before narrowing.
        let d = Complex64::new(ac[i + 1].re as f64, ac[i + 1].im as f64) * as_[i + 1];
        let d = d.conj() * amp as f64;
        spectrum[np2 - i + 1] = Complex32::new(d.re as f32, d.im as f32);
    }
    spectrum[fold_count] = scale(ac[fold_count], as_[fold_count], amp);
}
