//! Radiation-pattern filtering and the inverse transform back to the time
//! domain.

use crate::fft::fast;
use crate::fort::{Array1, Complex32};

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
pub fn apply_radiation_and_invert(
    fold_count: usize,
    mirror_count: usize,
    np2: usize,
    spectrum: &mut Array1<Complex32>,
    time_series: &mut Array1<f32>,
    radiation: &Array1<f32>,
) {
    let radiation_norm = 0.63f32;
    let partition_factor = 0.71f32;

    // Positive frequencies, signed radiation pattern (sign preserved since
    // 2004-12-21; the older code took abs()).
    for i in 1..=fold_count {
        spectrum[i] = spectrum[i] * radiation[i];
    }

    // Negative-frequency half, mirrored about fold_count.
    for i in fold_count + 1..=fold_count + mirror_count {
        let mm = 2 * fold_count - i;
        spectrum[i] = spectrum[i] * radiation[mm];
    }

    fast(np2, spectrum, 1);

    let fac = 1.0 / (radiation_norm * partition_factor * np2 as f32);
    for i in 1..=np2 {
        time_series[i] = fac * spectrum[i].re;
    }

    let n0 = np2 / 10;
    let dd = 3.14159625 / (n0 as f32);
    for i in 1..=n0 {
        let arg = 0.5 * (1.0 + (i as f32 * dd).cos());
        time_series[np2 - n0 + i] = time_series[np2 - n0 + i] * arg;
    }
}
