//! Site amplification.

use crate::fort::{Array1, Complex32};
use crate::state::Vmod;

/// `subroutine get_sitefacs(j0,nfreq,fn,an)` — `hb_high_ref.f:3020`.
///
/// Boore quarter-wavelength site amplification. For each log-frequency in `fn`,
/// finds the depth whose one-way S travel time is a quarter period, then returns
/// `an = 0.5 * ln(rho_src * V_src / (Vbar * rho_bar))` — half the log of the
/// impedance ratio between the source layer and the depth-averaged column.
///
/// `j0` is the source (bottom) layer index. Reads `thic`, `vsh` and `rho` from
/// `/vmod/`; `depth`, `vp`, `qp` and `qs` are untouched.
///
/// # Precision
///
/// Every local here is implicit `real*4` while the `/vmod/` fields read are
/// `real*8`, so each accumulation promotes to double, computes, then narrows on
/// assignment. `vdsrc = vs(j0)*dn(j0)` multiplies in double and stores single.
/// The `as f32` casts below are the narrowing points and are load-bearing.
///
/// The layer search reads at most `thic[j0]`: the index is tested against `j0`
/// at the top of the loop and only ever increments by one, so it cannot step
/// past. (An earlier analysis of mine claimed it could reach `j0+1`; that was
/// wrong.)
pub fn get_sitefacs(
    vmod: &Vmod,
    j0: usize,
    nfreq: usize,
    fn_: &Array1<f32>,
    an: &mut Array1<f32>,
) {
    let vdsrc = (vmod.vsh[j0] * vmod.rho[j0]) as f32;

    for kf in 1..=nfreq {
        let stt = 0.25 / fn_[kf].exp();

        let mut i = 2usize;
        let mut zdep = 0.0f32;
        let mut pz = 0.0f32;
        let mut tt = 0.0f32;
        let mut ttp = (vmod.thic[2] / vmod.vsh[2]) as f32;

        // Label 6145: walk down until a quarter-period of travel time has
        // accumulated, or the source layer is reached.
        while !(ttp >= stt || i == j0) {
            zdep = (zdep as f64 + vmod.thic[i]) as f32;
            pz = (pz as f64 + vmod.rho[i] * vmod.thic[i] / vmod.vsh[i]) as f32;
            tt = ttp;
            i += 1;
            ttp = (vmod.thic[i] / vmod.vsh[i] + tt as f64) as f32;
        }

        // Label 6146: interpolate the partial layer, then form the impedance
        // ratio. `(stt - tt)` is computed in single precision before being
        // promoted, matching the Fortran's mixed-type expressions.
        let bz = ((zdep as f64 + (stt - tt) as f64 * vmod.vsh[i]) / stt as f64) as f32;
        let pz = ((pz as f64 + vmod.rho[i] * (stt - tt) as f64) / stt as f64) as f32;

        an[kf] = 0.5 * (vdsrc / (bz * pz)).ln();
    }
}

/// `subroutine siteamp(np2,cw,dfr,nn,fn,an)` — `hb_high_ref.f:3120`.
///
/// Applies quarter-wavelength site amplification to a spectrum in place:
/// piecewise-linear interpolation of `an` against `ln(freq)` from the table
/// `fn`, exponentiated and multiplied onto each positive-frequency bin, then
/// Hermitian symmetry re-imposed.
///
/// `fn` must be sorted ascending. The interpolation pointer `kn` only ever
/// advances, so an unsorted table silently produces wrong factors rather than
/// an error. `get_sitefacs` is the only producer and does emit ascending
/// frequencies.
///
/// `fn` and `an` are natural logs of frequency and amplification respectively,
/// which is why the interpolation is linear in `freq = alog(dfr(i))` and the
/// result is exponentiated.
pub fn siteamp(
    np2: usize,
    cw: &mut Array1<Complex32>,
    dfr: &Array1<f32>,
    nn: usize,
    fn_: &Array1<f32>,
    an: &Array1<f32>,
) {
    let np = np2 / 2;
    let nf = np + 1;

    let mut kn = 1usize;
    let mut fm = 0.0f32;
    let mut am = an[kn];
    let mut fp = fn_[kn];
    let mut ap = an[kn];

    cw[1] = cw[1] * an[1];

    for i in 2..=np {
        let freq = dfr[i].ln();

        // Label 9123: advance the interpolation bracket. Written as an
        // if-then-with-backward-goto in the source, which is a do-while.
        if freq > fp && kn <= nn {
            loop {
                fm = fp;
                am = ap;
                kn += 1;
                if kn > nn {
                    // Past the table: pin the upper edge far away so the
                    // interpolation flattens to the last value.
                    fp = 1.0e+15;
                    ap = an[nn];
                } else {
                    fp = fn_[kn];
                    ap = an[kn];
                }
                if !(freq > fp && kn <= nn) {
                    break;
                }
            }
        }

        let fac = (am + (freq - fm) * (ap - am) / (fp - fm)).exp();
        cw[i] = cw[i] * fac;
    }

    // Re-impose Hermitian symmetry over the negative-frequency half.
    for i in 1..=np - 1 {
        cw[np2 - i + 1] = cw[i + 1].conj();
    }

    // Nyquist bin takes the top of the table.
    cw[nf] = cw[nf] * an[nn];
}
