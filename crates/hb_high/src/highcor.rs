//! Radiation-pattern filtering and the inverse transform back to the time
//! domain.

use crate::fft::fast;
use crate::fort::Complex32;

/// `subroutine apply_radiation_and_invert(fold_count,mirror_count,np2,spectrum,time_series,radiation)` — `hb_high_ref.f:2234`.
///
/// Multiplies the spectrum in `spectrum` by the signed radiation pattern `radiation`,
/// mirrors it over the negative-frequency half, inverse-transforms, scales by
/// `1/(radiation_norm*partition_factor*np2)`, and applies a raised-cosine taper over the final
/// `np2/10` samples. `time_series` receives the resulting real time series.
///
/// `radiation_norm = 0.63` is the radiation-pattern normalisation and `partition_factor = 0.71` the
/// vector partition factor for two orthogonal components (nominally
/// `1/sqrt(2)`, written as two digits).
///
/// # The taper constant was a typo, and it is now fixed
///
/// The taper used `dd = 3.14159625/n0` (`:2266`). That is **not** pi — the last digits
/// of `3.14159265` are transposed. Every other occurrence in the file is some
/// truncation of the correct value (`3.1415926`, `3.14159265`, `3.141592654`), so this
/// one was a genuine slip rather than a deliberate approximation.
///
/// It was reproduced verbatim for as long as bit-identity was the contract, with a note
/// saying that correcting it needed a written justification rather than a quiet cleanup.
/// This is that justification. The error is about 1.1e-6 relative — roughly thirty times
/// the worst of the file's honest truncations — and it left the taper fractionally short
/// of a half cosine, so the final sample was not exactly zero. It is now
/// `std::f32::consts::PI`, and the taper closes properly.
///
/// This changes every waveform's tail, in the last `np2/10` samples only, by a factor
/// bounded by the taper's own 1.1e-6 error. See `PORTING_RULES.md` §1.
pub fn apply_radiation_and_invert(
    fold_count: usize,
    mirror_count: usize,
    spectrum: &mut [Complex32],
    time_series: &mut [f32],
    radiation: &[f32],
) {
    let radiation_norm = 0.63f32;
    let partition_factor = 0.71f32;
    // `np2` was a separate argument and is the spectrum's length at every call site.
    let np2 = spectrum.len();

    // 0-based since §2.3. Every index below is the Fortran's minus one; the loop bounds
    // moved with them rather than a `- 1` being sprinkled at each access, since a
    // half-converted expression is the thing that hides an off-by-one.

    // Positive frequencies, signed radiation pattern (sign preserved since
    // 2004-12-21; the older code took abs()).
    for i in 0..fold_count {
        spectrum[i] *= radiation[i];
    }

    // Negative-frequency half, mirrored about `fold_count`. The Fortran computes
    // `mm = 2*fold_count - i` from its 1-based `i`; with `j = i - 1` that is
    // `2*fold_count - j - 2` 0-based. Checked against np2 = 16, fold_count = 9:
    // Fortran i = 10 takes radiation(8), storage element 7; here j = 9 gives
    // 18 - 9 - 2 = 7.
    for j in fold_count..fold_count + mirror_count {
        spectrum[j] *= radiation[2 * fold_count - j - 2];
    }

    fast(spectrum, 1);

    let fac = 1.0 / (radiation_norm * partition_factor * np2 as f32);
    for (sample, bin) in time_series[..np2].iter_mut().zip(spectrum.iter()) {
        *sample = fac * bin.re;
    }

    // Raised-cosine taper over the final np2/10 samples. The original's `dd` used
    // 3.14159625, a transposition of pi's digits; see the note above.
    let n0 = np2 / 10;
    let dd = std::f32::consts::PI / (n0 as f32);
    for i in 1..=n0 {
        let arg = 0.5 * (1.0 + (i as f32 * dd).cos());
        // Fortran writes time_series(np2 - n0 + i) for i = 1..=n0, i.e. the last n0
        // samples; 0-based that is index np2 - n0 + i - 1.
        time_series[np2 - n0 + i - 1] *= arg;
    }
}
