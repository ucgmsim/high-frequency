//! Quarter-wavelength site amplification.
//!
//! See `PHYSICS.md` §4 for the physics; the short version is that a wave of frequency `f` is
//! most sensitive to the material within about a quarter wavelength of the surface, so the
//! amplification is the impedance contrast between the source region and the column averaged
//! down to that depth.

use ndarray::{ArrayView1, ArrayViewMut1, Axis, azip, s};

use crate::fft::Complex32;
use crate::state::VelocityModel;

/// Quarter-wavelength amplification factors, one per frequency in the site table.
///
/// Boore & Joyner (1997), which is what Graves & Pitarka (2010) cite for the "gross impedance
/// effects calculated using quarter wavelength theory". See `PHYSICS.md` §4.
///
/// For each log-frequency, walks down the velocity model until the accumulated one-way S-wave
/// travel time reaches a quarter period, averages velocity and density over that depth, and
/// returns
///
/// ```text
/// factor = 0.5 * ln( rho_src * beta_src / (rho_bar * beta_bar) )
/// ```
///
/// — a **log** amplitude, which is why [`apply_site_amplification`] exponentiates. Returning the
/// log lets the interpolation there be linear.
///
/// `source_layer` is the source layer index, and the walk descends *towards* it rather than to
/// the bottom of the model. It reads at most `thickness_km[source_layer]`: the index is tested
/// against `source_layer` at the top of the loop and only ever increments by one, so it cannot
/// step past.
///
/// # The `as f32` casts are load-bearing
///
/// The velocity-model fields are `f64` and the accumulators are `f32`, so every accumulation
/// widens, computes, then narrows on assignment, matching production output. Removing a cast,
/// or hoisting the arithmetic into `f64` throughout, changes the result. `(stt - tt)` in
/// particular is formed in `f32` before being widened.
pub fn site_amplification_factors(
    vmod: &VelocityModel,
    source_layer: usize,
    table_log_frequency: ArrayView1<f32>,
    factors: ArrayViewMut1<f32>,
) {
    let vdsrc = (vmod[source_layer].vsh_km_s * vmod[source_layer].density_g_cm3) as f32;

    // One factor per frequency; `azip!` asserts the two tables are the same length rather than
    // letting `zip` silently stop at the shorter.
    azip!((
        factor in factors,
        &log_freq in table_log_frequency,
    ) {
        // A quarter period of travel time is the target depth criterion.
        let stt = 0.25 / log_freq.exp();

        // Index 1, not 0: layer 0 is the inserted air layer.
        let mut i = 1usize;
        let mut zdep = 0.0f32;
        let mut pz = 0.0f32;
        let mut tt = 0.0f32;
        let mut ttp = (vmod[1].thickness_km / vmod[1].vsh_km_s) as f32;

        // Walk down until a quarter period has accumulated, or the source layer is reached.
        // Sequential: each step's `ttp` depends on the previous one's.
        while !(ttp >= stt || i == source_layer) {
            zdep = (zdep as f64 + vmod[i].thickness_km) as f32;
            pz = (pz as f64 + vmod[i].density_g_cm3 * vmod[i].thickness_km / vmod[i].vsh_km_s) as f32;
            tt = ttp;
            i += 1;
            ttp = (vmod[i].thickness_km / vmod[i].vsh_km_s + tt as f64) as f32;
        }

        // The walk overshoots by part of a layer, so interpolate within it, then form the
        // impedance ratio. `bz` is the travel-time averaged velocity, `pz` the averaged
        // density-over-velocity.
        let bz = ((zdep as f64 + (stt - tt) as f64 * vmod[i].vsh_km_s) / stt as f64) as f32;
        let pz = ((pz as f64 + vmod[i].density_g_cm3 * (stt - tt) as f64) / stt as f64) as f32;

        *factor = 0.5 * (vdsrc / (bz * pz)).ln();
    });
}

/// Resample the amplification table onto a transform's frequency axis, as a **linear gain per
/// bin**.
///
/// Piecewise-linear interpolation of `factors` against `ln(frequency)`, then exponentiated.
/// Both inputs are natural logs — of frequency and of amplification respectively — which is why
/// the interpolation is linear in `ln f` and the result is exponentiated at the end.
///
/// # Why this is separate from applying it
///
/// The curve depends only on the source layer and the transform length, not on the
/// component, the ray, or the spectrum it multiplies, so it is built once and applied to all
/// three components.
///
/// `gain` is `np2/2 + 1` long: the positive half of the spectrum plus Nyquist. The negative
/// half is not a free choice — [`apply_site_amplification`] mirrors it.
///
/// # `log_frequency_hz` must be sorted ascending
///
/// The interpolation cursor only ever advances, so an unsorted axis silently produces wrong
/// factors rather than an error. [`crate::stoc::SpectrumPlan`] is the only producer and does
/// emit ascending frequencies.
pub fn site_gain_curve(
    // `ln(frequency_hz[i])`, from the transform's plan. Index 0 is never read: `ln(0)` has no
    // meaning as a frequency.
    log_frequency_hz: ArrayView1<f32>,
    table_log_frequency: ArrayView1<f32>,
    factors: ArrayView1<f32>,
    mut gain: ArrayViewMut1<f32>,
) {
    let np = gain.len() - 1;
    let table_count = factors.len();

    // `kn` is the table cursor, and the clamp past the end reads `factors[table_count - 1]`.
    let mut kn = 0usize;
    let mut fm = 0.0f32;
    let mut am = factors[kn];
    let mut fp = table_log_frequency[kn];
    let mut ap = factors[kn];

    // DC. The factors are log amplitudes, so this exponentiates like every interior bin does.
    // There is nothing to interpolate at zero frequency, so the bottom table entry is used.
    gain[0] = factors[0].exp();
    // Nyquist takes the top of the table, exponentiated for the same reason as DC. Unlike DC —
    // which `stochastic_spectrum` leaves at zero, making any gain unobservable there — this
    // bin carries a value, so the gain applied to it is observable in the output.
    gain[np] = factors[table_count - 1].exp();

    for i in 1..np {
        let freq = log_frequency_hz[i];

        // Advance the interpolation bracket. A do-while: the first comparison has already been
        // made by the `if`.
        if freq > fp && kn < table_count {
            loop {
                fm = fp;
                am = ap;
                kn += 1;
                if kn >= table_count {
                    // Past the table: pin the upper edge far away so the interpolation
                    // flattens to the last value rather than extrapolating.
                    fp = 1.0e+15;
                    ap = factors[table_count - 1];
                } else {
                    fp = table_log_frequency[kn];
                    ap = factors[kn];
                }
                if !(freq > fp && kn < table_count) {
                    break;
                }
            }
        }

        gain[i] = (am + (freq - fm) * (ap - am) / (fp - fm)).exp();
    }
}

/// Multiply a spectrum by a gain curve and re-impose Hermitian symmetry, in place.
///
/// `gain` comes from [`site_gain_curve`] and covers bins `0..=np2/2`; the negative half is
/// mirrored rather than multiplied, so passing a curve of any other length is a bug the length
/// assertion catches.
pub fn apply_site_amplification(spectrum: &mut [Complex32], gain: ArrayView1<f32>) {
    let np2 = spectrum.len();
    let np = np2 / 2;
    assert_eq!(
        gain.len(),
        np + 1,
        "a gain curve covers the positive half plus Nyquist, {} bins for a {np2}-point transform",
        np + 1
    );

    // Bins `0..=np`, which is DC, the interior, and Nyquist in one pass. Nyquist is multiplied
    // before the mirror rather than after it, and that is not a reordering: the mirror reads
    // bins `1..np` and never bin `np`, so the two do not interact.
    azip!((
        bin in &mut ArrayViewMut1::from(&mut spectrum[..=np]),
        &g in gain,
    ) *bin *= g);

    // Re-impose Hermitian symmetry: bin `np2 - k` takes `conj(bin k)` for k in `1..np`, as a
    // reversed view of the head assigned into the tail, the same as the mirror in
    // `stoc::stochastic_spectrum`. For np2 = 16: dest 9 takes src 7 and dest 15 takes src 1.
    let mut view = ArrayViewMut1::from(spectrum);
    let (positive, mut negative) = view.view_mut().split_at(Axis(0), np + 1);
    azip!((dest in &mut negative, &src in positive.slice(s![1..np; -1])) *dest = src.conj());
}
