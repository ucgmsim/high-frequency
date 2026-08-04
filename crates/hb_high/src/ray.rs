//! Ray theory. Tier 0 holds `cr`; `cagcon`, `dtdp`, `pnot`, `trav`, `ttime`,
//! `geom_terms` and `gf_amp_tt` follow in tiers 1-4.

use crate::fort::Complex64;

/// `function cr(p,v)` — `hb_high_ref.f:3349`. Complex vertical slowness
/// `eta = sqrt(1/v^2 - p^2)`, with an explicit branch-cut choice.
///
/// This is the numerically delicate heart of the ray code. It evaluates the
/// square root in polar form rather than algebraically so the branch can be
/// selected deliberately, and the selection at labels 12/13 must be
/// transliterated literally — see `PORTING_RULES.md` §2.
///
/// Two traps in the original worth naming:
///
/// * `pr = p` assigns a `complex*16` to a `real*8`, which silently takes the
///   real part. It is not a typo for `dreal(p)`.
/// * the local named `pi` is `dimag(p)`, the **imaginary part of p**, not
///   3.14159. The actual pi appears separately as the truncated 10-digit
///   literal `3.141592654d0`, which is copied verbatim.
pub fn cr(p: Complex64, v: f64) -> Complex64 {
    let t1 = 1.0e-08f64;
    let rsq = 1.0f64 / (v * v);
    let pr = p.re;
    // `pi` here is Im(p), matching the Fortran's variable name.
    let pi = p.im;
    let mut a = rsq - pr * pr + pi * pi;
    let mut b = -2.0f64 * pi * pr;
    let d = (a * a + b * b).sqrt().sqrt();

    // Near the real axis the phase is forced to 0 or pi rather than taken from
    // atan2, which would be ill-conditioned there.
    let phi = if pi.abs() < t1 {
        if a < 0.0 { 3.141592654f64 } else { 0.0 }
    } else {
        b.atan2(a)
    };

    let mut e = (phi / 2.0f64).cos();
    let mut f = (phi / 2.0f64).sin();

    // Labels 13/12: negate unless (f <= t1 and e > 0).
    //   IF(F.GT.T1) GO TO 13      -> f > t1 negates
    //   IF(E.GT.0.0D0) GO TO 12   -> otherwise e > 0 skips the negation
    //   13: e = -e; f = -f
    if f > t1 || e <= 0.0 {
        e = -e;
        f = -f;
    }

    a = d * e;
    b = d * f;
    Complex64::new(a, b)
}
