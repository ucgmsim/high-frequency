//! `stoc_f` — the stochastic source spectrum for one subfault.

use crate::fft::{fast, flzero};
use crate::fort::{Array1, Complex32, Complex64};
use crate::rng::{normal_random_number, Pcg32};
use crate::special::dgamm;

/// `subroutine stoc_f(...)` — `hb_high_ref.f:1670`.
///
/// Builds the complex Fourier spectrum of one subfault's stochastic S-wave
/// motion: a Brune omega-squared source, a kappa/fmax high-cut, path Q, and the
/// Frankel two-corner operator, multiplied by a unit-power random phase
/// spectrum and mirrored to Hermitian symmetry. Writes `np2` values into `cw`.
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
///   in `f32` differs in the last bit, so `cw(i) = ac(i)*as(i)*amp` is built
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
/// `amp = 1/(dt*sqrt(fsa/nf))` normalises so the average *power* spectrum is
/// unity, per Boore (1983) — a 2009-03-18 change from normalising the amplitude
/// spectrum, which reduced motions about 10% and was offset by raising the
/// default corner frequency 5%. This calibration assumes
/// [`normal_random_number`]'s renormalisation, so substituting a plain N(0,1)
/// generator would silently change the output level.
#[allow(clippy::too_many_arguments)]
pub fn stoc_f(
    rng: &mut Pcg32,
    np2: usize,
    r: f32,
    tw: f32,
    eps: f32,
    eta: f32,
    betvs: f32,
    row: f32,
    dt: f32,
    smt: f32,
    _dlm: f32,
    fc: f32,
    fmx: f32,
    akapp: f32,
    cw: &mut Array1<Complex32>,
    dfr: &Array1<f32>,
    qb: f32,
    qfe: f32,
    bigc: f32,
) {
    let pai = 3.1415926f32;
    let rp = 0.63f32;

    let fc2 = fc * fc;

    // fs is the free-surface factor; prtitn the vector partition factor for two
    // orthogonal components, nominally 1/sqrt(2) but written as two digits.
    let fs = 2.0f32;
    let prtitn = 0.71f32;

    let nf = np2 / 2 + 1;
    let rxx = r * 100000.0;

    // Saragoni-Hart style envelope: b and c from the (eps, eta) window shape.
    let b = -eps * eta.ln() / (1.0 + eps * (eps.ln() - 1.0));
    let c = b / eps / tw;
    // Computed in real*4, then widened -- gsa is real*8 but 2*b+1.0 is not.
    let gsa = (2.0 * b + 1.0) as f64;
    let gm = dgamm(gsa);
    // The power is real*4; the division by gm and the sqrt are real*8; the
    // result narrows back to real*4.
    let aa = (((2.0 * c).powf(2.0 * b + 1.0) as f64) / gm).sqrt() as f32;

    let mut w = Array1::<f32>::new(np2);
    for i in 1..=np2 {
        let t = (i - 1) as f32 * dt;
        w[i] = aa * t.powf(b) * (-c * t).exp();
    }

    let beta = betvs * 100000.0;
    let cc = rp * fs * prtitn / (4.0 * pai * row * (beta * beta * beta));
    let omgc = 2.0 * pai * fc;
    let omgm = 2.0 * pai * fmx;

    let mut as_ = Array1::<f64>::new(np2);
    as_[1] = 0.0;
    for i in 2..=nf {
        let fr = dfr[i];
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
        let a1 = (cc * smt * (omg * omg / (1.0 + (omg / omgc) * (omg / omgc)))) as f64;

        let a2 = if akapp <= 0.0 {
            // (1.0 + (omg/omgm)**1)**(-1.0). The **1 is the identity, and
            // gfortran folds **(-1.0) to a reciprocal -- so this must be a
            // division, NOT powf(-1.0), which differs. See PORTING_RULES.md §4b.
            (1.0 / (1.0 + (omg / omgm))) as f64
        } else {
            (-pai * fr * akapp).exp() as f64
        };

        // The Fortran assigns a3 three times; only the last survives. The two
        // dead stores are recorded here rather than executed -- they are pure
        // and immediately overwritten, and each costs an exp() per frequency
        // bin on a path called three times per subfault:
        //     a3 = exp(-omg*rxx/(2.0*qv*beta))/rxx        ! fixed Q model
        //     a3 = exp(-0.5*omg*qb*fr**(-0.5))/rxx        ! qbar, fixed exponent
        let a3 = ((-0.5 * omg * qb * fr.powf(-qfe)).exp() / rxx) as f64;

        // Frankel two-corner convolution operator.
        let frank = bigc * (fc2 + fr2) / (fc2 + bigc * fr2);

        as_[i] = a1 * a2 * a3 * frank as f64;
    }

    let mut a = Array1::<f32>::new(np2);
    normal_random_number(rng, np2, &mut a);
    flzero(np2, dt, &mut a);

    let mut ac = Array1::<Complex32>::filled(np2, Complex32::ZERO);
    for i in 1..=np2 {
        ac[i] = Complex32::new(a[i] * w[i], 0.0);
    }

    fast(np2, &mut ac, -1);

    // Average POWER spectrum to unity (2009-03-18), not amplitude.
    let mut fsa = 0.0f32;
    for i in 1..=nf {
        fsa += ac[i].abs() * ac[i].abs();
    }
    let amp = 1.0 / (dt * (fsa / nf as f32).sqrt());

    // complex*8 * real*8 goes through complex*16; see the note above.
    let scale = |z: Complex32, s: f64, m: f32| -> Complex32 {
        let d = Complex64::new(z.re as f64, z.im as f64) * s * m as f64;
        Complex32::new(d.re as f32, d.im as f32)
    };

    let np = np2 / 2;
    for i in 1..=np {
        cw[i] = scale(ac[i], as_[i], amp);
        // conjg() is applied to the complex*16 product, before narrowing.
        let d = Complex64::new(ac[i + 1].re as f64, ac[i + 1].im as f64) * as_[i + 1];
        let d = d.conj() * amp as f64;
        cw[np2 - i + 1] = Complex32::new(d.re as f32, d.im as f32);
    }
    cw[nf] = scale(ac[nf], as_[nf], amp);
}
