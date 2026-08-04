//! Site amplification.

use crate::fort::{Array1, Complex32};
use crate::state::VelocityModel;

/// `subroutine site_amplification_factors(layer_count,frequency_count,fn,factors)` — `hb_high_ref.f:3020`.
///
/// Boore quarter-wavelength site amplification. For each log-frequency in `fn`,
/// finds the depth whose one-way S travel time is a quarter period, then returns
/// `factors = 0.5 * ln(rho_src * V_src / (Vbar * rho_bar))` — half the log of the
/// impedance ratio between the source layer and the depth-averaged column.
///
/// `layer_count` is the source (bottom) layer index. Reads `thickness_km`, `vsh_km_s` and `density_g_cm3` from
/// `/vmod/`; `depth`, `vp_km_s`, `attenuation_p` and `attenuation_s` are untouched.
///
/// # Precision
///
/// Every local here is implicit `real*4` while the `/vmod/` fields read are
/// `real*8`, so each accumulation promotes to double, computes, then narrows on
/// assignment. `vdsrc = vs(layer_count)*dn(layer_count)` multiplies in double and stores single.
/// The `as f32` casts below are the narrowing points and are load-bearing.
///
/// The layer search reads at most `thickness_km[layer_count]`: the index is tested against `layer_count`
/// at the top of the loop and only ever increments by one, so it cannot step
/// past. (An earlier analysis of mine claimed it could reach `layer_count+1`; that was
/// wrong.)
pub fn site_amplification_factors(
    vmod: &VelocityModel,
    layer_count: usize,
    frequency_count: usize,
    log_frequency: &Array1<f32>,
    factors: &mut Array1<f32>,
) {
    let vdsrc = (vmod.vsh_km_s[layer_count] * vmod.density_g_cm3[layer_count]) as f32;

    for kf in 1..=frequency_count {
        let stt = 0.25 / log_frequency[kf].exp();

        let mut i = 2usize;
        let mut zdep = 0.0f32;
        let mut pz = 0.0f32;
        let mut tt = 0.0f32;
        let mut ttp = (vmod.thickness_km[2] / vmod.vsh_km_s[2]) as f32;

        // Label 6145: walk down until a quarter-period of travel time has
        // accumulated, or the source layer is reached.
        while !(ttp >= stt || i == layer_count) {
            zdep = (zdep as f64 + vmod.thickness_km[i]) as f32;
            pz = (pz as f64 + vmod.density_g_cm3[i] * vmod.thickness_km[i] / vmod.vsh_km_s[i]) as f32;
            tt = ttp;
            i += 1;
            ttp = (vmod.thickness_km[i] / vmod.vsh_km_s[i] + tt as f64) as f32;
        }

        // Label 6146: interpolate the partial layer, then form the impedance
        // ratio. `(stt - tt)` is computed in single precision before being
        // promoted, matching the Fortran's mixed-type expressions.
        let bz = ((zdep as f64 + (stt - tt) as f64 * vmod.vsh_km_s[i]) / stt as f64) as f32;
        let pz = ((pz as f64 + vmod.density_g_cm3[i] * (stt - tt) as f64) / stt as f64) as f32;

        factors[kf] = 0.5 * (vdsrc / (bz * pz)).ln();
    }
}

/// `subroutine apply_site_amplification(np2,spectrum,frequency_hz,table_count,fn,factors)` — `hb_high_ref.f:3120`.
///
/// Applies quarter-wavelength site amplification to a spectrum in place:
/// piecewise-linear interpolation of `factors` against `ln(freq)` from the table
/// `fn`, exponentiated and multiplied onto each positive-frequency bin, then
/// Hermitian symmetry re-imposed.
///
/// `fn` must be sorted ascending. The interpolation pointer `kn` only ever
/// advances, so factors unsorted table silently produces wrong factors rather than
/// factors error. `site_amplification_factors` is the only producer and does emit ascending
/// frequencies.
///
/// `fn` and `factors` are natural logs of frequency and amplification respectively,
/// which is why the interpolation is linear in `freq = alog(frequency_hz(i))` and the
/// result is exponentiated.
pub fn apply_site_amplification(
    np2: usize,
    spectrum: &mut Array1<Complex32>,
    frequency_hz: &Array1<f32>,
    table_count: usize,
    log_frequency: &Array1<f32>,
    factors: &Array1<f32>,
) {
    let np = np2 / 2;
    let nf = np + 1;

    let mut kn = 1usize;
    let mut fm = 0.0f32;
    let mut am = factors[kn];
    let mut fp = log_frequency[kn];
    let mut ap = factors[kn];

    spectrum[1] = spectrum[1] * factors[1];

    for i in 2..=np {
        let freq = frequency_hz[i].ln();

        // Label 9123: advance the interpolation bracket. Written as factors
        // if-then-with-backward-goto in the source, which is a do-while.
        if freq > fp && kn <= table_count {
            loop {
                fm = fp;
                am = ap;
                kn += 1;
                if kn > table_count {
                    // Past the table: pin the upper edge far away so the
                    // interpolation flattens to the last value.
                    fp = 1.0e+15;
                    ap = factors[table_count];
                } else {
                    fp = log_frequency[kn];
                    ap = factors[kn];
                }
                if !(freq > fp && kn <= table_count) {
                    break;
                }
            }
        }

        let fac = (am + (freq - fm) * (ap - am) / (fp - fm)).exp();
        spectrum[i] = spectrum[i] * fac;
    }

    // Re-impose Hermitian symmetry over the negative-frequency half.
    for i in 1..=np - 1 {
        spectrum[np2 - i + 1] = spectrum[i + 1].conj();
    }

    // Nyquist bin takes the top of the table.
    spectrum[nf] = spectrum[nf] * factors[table_count];
}
