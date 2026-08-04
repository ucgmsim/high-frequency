//! Radiation-pattern filtering and the inverse transform back to the time
//! domain.

use crate::fft::inverse;
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

    // Positive frequencies, signed radiation pattern (sign preserved since 2004-12-21;
    // the older code took abs()).
    for (bin, &gain) in spectrum[..fold_count].iter_mut().zip(radiation) {
        *bin *= gain;
    }

    // Negative-frequency half, mirrored about `fold_count`.
    //
    // The Fortran indexes this as `radiation(2*fold_count - i)`, which 0-based is
    // `radiation[2*fold_count - j - 2]` -- an expression that needed a worked example on
    // np2 = 16 to be believable. It is just the radiation array walked BACKWARDS: as `j`
    // runs `fold_count ..< fold_count + mirror_count`, the index runs `fold_count - 2`
    // down to `fold_count - mirror_count - 1`, which for the fixed
    // `mirror_count = fold_count - 2` is `fold_count - 2` down to `1`. A reversed zip
    // says that, and cannot be off by one.
    let mirror = &radiation[1..fold_count - 1];
    debug_assert_eq!(mirror.len(), mirror_count, "mirror_count is fold_count - 2");
    for (bin, &gain) in spectrum[fold_count..][..mirror_count].iter_mut().zip(mirror.iter().rev())
    {
        *bin *= gain;
    }

    inverse(spectrum);

    let fac = 1.0 / (radiation_norm * partition_factor * np2 as f32);
    for (sample, bin) in time_series[..np2].iter_mut().zip(spectrum.iter()) {
        *sample = fac * bin.re;
    }

    // Raised-cosine taper over the final np2/10 samples. The original's `dd` used
    // 3.14159625, a transposition of pi's digits; see the note above.
    //
    // The taper runs over the LAST `n0` samples, so the slice says which samples and the
    // enumeration says how far into the taper each one is. `i + 1` keeps the Fortran's
    // 1-based step number, which is what makes the final sample land on `cos(pi)`.
    let n0 = np2 / 10;
    let dd = std::f32::consts::PI / (n0 as f32);
    for (i, sample) in time_series[np2 - n0..np2].iter_mut().enumerate() {
        *sample *= 0.5 * (1.0 + ((i + 1) as f32 * dd).cos());
    }
}
