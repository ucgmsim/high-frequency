//! Site amplification. Tier 0 holds `siteamp`; `get_sitefacs` follows in tier 1.

use crate::fort::{Array1, Complex32};

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
