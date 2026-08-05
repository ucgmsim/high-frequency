//! Site amplification.

use crate::fort::Complex32;
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
/// The layer search reads at most `thickness_km[source_layer]`: the index is tested
/// against `source_layer` at the top of the loop and only ever increments by one, so it
/// cannot step past. (An earlier analysis of mine claimed it could reach `+1`; that was
/// wrong.)
///
/// The second argument was called `layer_count` here, which was a misreading: every caller
/// passes `ksrc`, the SOURCE layer, and the routine walks down towards it rather than to
/// the bottom of the model. Renamed with §2.3's index flip, since the two are easy to
/// confuse once both are 0-based.
pub fn site_amplification_factors(
    vmod: &VelocityModel,
    source_layer: usize,
    frequency_count: usize,
    log_frequency: &[f32],
    factors: &mut [f32],
) {
    let vdsrc = (vmod[source_layer].vsh_km_s * vmod[source_layer].density_g_cm3) as f32;

    // Both the frequency table and the velocity model are 0-based since §2.3. The two
    // tables are walked in lockstep, one factor per frequency.
    for (factor, &log_freq) in factors[..frequency_count].iter_mut().zip(log_frequency) {
        let stt = 0.25 / log_freq.exp();

        // Starts at the layer below the air layer: the Fortran's layer 2.
        let mut i = 1usize;
        let mut zdep = 0.0f32;
        let mut pz = 0.0f32;
        let mut tt = 0.0f32;
        let mut ttp = (vmod[1].thickness_km / vmod[1].vsh_km_s) as f32;

        // Label 6145: walk down until a quarter-period of travel time has accumulated, or
        // the source layer is reached. Genuinely sequential -- each step's `ttp` depends
        // on the previous one's -- so this stays a loop.
        while !(ttp >= stt || i == source_layer) {
            zdep = (zdep as f64 + vmod[i].thickness_km) as f32;
            pz = (pz as f64 + vmod[i].density_g_cm3 * vmod[i].thickness_km / vmod[i].vsh_km_s) as f32;
            tt = ttp;
            i += 1;
            ttp = (vmod[i].thickness_km / vmod[i].vsh_km_s + tt as f64) as f32;
        }

        // Label 6146: interpolate the partial layer, then form the impedance
        // ratio. `(stt - tt)` is computed in single precision before being
        // promoted, matching the Fortran's mixed-type expressions.
        let bz = ((zdep as f64 + (stt - tt) as f64 * vmod[i].vsh_km_s) / stt as f64) as f32;
        let pz = ((pz as f64 + vmod[i].density_g_cm3 * (stt - tt) as f64) / stt as f64) as f32;

        *factor = 0.5 * (vdsrc / (bz * pz)).ln();
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
    spectrum: &mut [Complex32],
    // `ln(frequency_hz[i])`, precomputed per segment. Index 0 is never read.
    log_frequency_hz: &[f32],
    table_count: usize,
    log_frequency: &[f32],
    factors: &[f32],
) {
    // `np2` was a separate argument and is the spectrum's length at every call site.
    let np2 = spectrum.len();
    let np = np2 / 2;

    // 0-based since §2.3. `kn` is the table cursor: the Fortran's `kn <= table_count`
    // becomes `kn < table_count` and its `kn > table_count` becomes
    // `kn >= table_count`, with the clamp reading the last entry as
    // `factors[table_count - 1]`. Traced against the 1-based original for
    // table_count = 6: it walks 1..7 there and 0..6 here, clamping on the same step.
    let mut kn = 0usize;
    let mut fm = 0.0f32;
    let mut am = factors[kn];
    let mut fp = log_frequency[kn];
    let mut ap = factors[kn];

    // DC. The factors are LOG amplitudes, so this exponentiates like every interior
    // bin does -- §2.6 defect 2. There is no interpolation to do at zero frequency:
    // `ln(0)` is undefined, so the bottom table entry is used, as the original did.
    spectrum[0] *= factors[0].exp();

    for i in 1..np {
        let freq = log_frequency_hz[i];

        // Label 9123: advance the interpolation bracket. Written as an
        // if-then-with-backward-goto in the source, which is a do-while.
        if freq > fp && kn < table_count {
            loop {
                fm = fp;
                am = ap;
                kn += 1;
                if kn >= table_count {
                    // Past the table: pin the upper edge far away so the
                    // interpolation flattens to the last value.
                    fp = 1.0e+15;
                    ap = factors[table_count - 1];
                } else {
                    fp = log_frequency[kn];
                    ap = factors[kn];
                }
                if !(freq > fp && kn < table_count) {
                    break;
                }
            }
        }

        let fac = (am + (freq - fm) * (ap - am) / (fp - fm)).exp();
        spectrum[i] *= fac;
    }

    // Re-impose Hermitian symmetry over the negative-frequency half. The Fortran
    // writes spectrum(np2 - i + 1) = conjg(spectrum(i + 1)) from a 1-based i; with
    // j = i - 1 the destination is np2 - j - 1 and the source is j + 1. Checked on
    // np2 = 16: Fortran i = 1 writes spectrum(16) from spectrum(2), storage 15 from
    // storage 1; j = 0 gives 16 - 0 - 1 = 15 from 0 + 1 = 1.
    for j in 0..np - 1 {
        spectrum[np2 - j - 1] = spectrum[j + 1].conj();
    }

    // Nyquist takes the top of the table, exponentiated for the same reason as DC.
    // This is the half of §2.6 defect 2 that was actually live: unlike DC -- which
    // `stochastic_spectrum` sets to zero, making the wrong gain unobservable -- this
    // bin carries a value.
    spectrum[np] *= factors[table_count - 1].exp();
}
