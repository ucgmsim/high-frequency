//! Radiation-pattern filtering and the inverse transform back to the time
//! domain.

use crate::fft::fast;
use crate::fort::{Array1, Complex32};

/// `subroutine highcor_f(nf,mf,np2,cw1,stdd,rdna)` — `hb_high_ref.f:2234`.
///
/// Multiplies the spectrum in `cw1` by the signed radiation pattern `rdna`,
/// mirrors it over the negative-frequency half, inverse-transforms, scales by
/// `1/(rp*prtitn*np2)`, and applies a raised-cosine taper over the final
/// `np2/10` samples. `stdd` receives the resulting real time series.
///
/// `rp = 0.63` is the radiation-pattern normalisation and `prtitn = 0.71` the
/// vector partition factor for two orthogonal components (nominally
/// `1/sqrt(2)`, written as two digits).
///
/// # The taper constant is a typo, and it is reproduced
///
/// The taper uses `dd = 3.14159625/n0` (`:2266`). That is **not** pi — the last
/// digits of `3.14159265` are transposed. Every other occurrence in the file is
/// some truncation of the correct value (`3.1415926`, `3.14159265`,
/// `3.141592654`), so this one is a genuine slip.
///
/// It is copied verbatim regardless. The error is about 1.1e-6 relative, which
/// leaves the taper very slightly short of a half cosine, so the final sample is
/// not exactly zero. Fixing it would change every waveform's tail. If it is ever
/// worth correcting, that is a Phase 3 re-baseline with a written justification,
/// not a quiet cleanup. See `PORTING_RULES.md` §1.
pub fn highcor_f(
    nf: usize,
    mf: usize,
    np2: usize,
    cw1: &mut Array1<Complex32>,
    stdd: &mut Array1<f32>,
    rdna: &Array1<f32>,
) {
    let rp = 0.63f32;
    let prtitn = 0.71f32;

    // Positive frequencies, signed radiation pattern (sign preserved since
    // 2004-12-21; the older code took abs()).
    for i in 1..=nf {
        cw1[i] = cw1[i] * rdna[i];
    }

    // Negative-frequency half, mirrored about nf.
    for i in nf + 1..=nf + mf {
        let mm = 2 * nf - i;
        cw1[i] = cw1[i] * rdna[mm];
    }

    fast(np2, cw1, 1);

    let fac = 1.0 / (rp * prtitn * np2 as f32);
    for i in 1..=np2 {
        stdd[i] = fac * cw1[i].re;
    }

    let n0 = np2 / 10;
    let dd = 3.14159625 / (n0 as f32);
    for i in 1..=n0 {
        let arg = 0.5 * (1.0 + (i as f32 * dd).cos());
        stdd[np2 - n0 + i] = stdd[np2 - n0 + i] * arg;
    }
}
