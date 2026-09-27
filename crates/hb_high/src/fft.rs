//! The complex transform, and the baseline correction that shares its callers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{ArrayViewMut1, azip};
use rustfft::{Fft, FftDirection, FftPlanner};

pub use rustfft::num_complex::Complex;

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

/// A planned transform of one length and direction. `rustfft` hands these out behind an
/// `Arc` because a plan is shareable and immutable once built.
type FftPlan = Arc<dyn Fft<f32>>;

thread_local! {
    /// Plans are cached per `(length, direction)`: building one is where `rustfft`
    /// does its twiddle precomputation, so planning per call would be expensive.
    ///
    /// Thread-local rather than a global mutex so stations simulated on separate
    /// threads never contend for a lock on the hottest path in the program.
    static PLANS: RefCell<HashMap<(usize, bool), FftPlan>> = RefCell::new(HashMap::new());
}

/// The odd multipliers of the length ladder — four steps per octave.
///
/// All 7-smooth, so `rustfft` has dedicated butterflies for the resulting mixed-radix
/// factorisations, and the ratios between successive rungs are 1.25, 1.20, 1.17 and 1.14.
const LENGTH_MULTIPLIERS: [usize; 4] = [1, 3, 5, 7];

/// The transform length to use for a signal of at least `at_least` samples.
///
/// Returns the smallest even number of the form `m · 2^a` with `m` drawn from
/// `LENGTH_MULTIPLIERS` — a ladder of four rungs per octave. Evenness is required by the
/// Hermitian mirrors that `spectrum::stochastic_spectrum` and `site::apply_site_amplification`
/// re-impose about the Nyquist bin.
///
/// # Why not `next_power_of_two`
///
/// A power-of-two ladder has one rung per octave, so it overshoots by up to 2x: a signal
/// of 80,601 samples rounds up to 131,072 where 81,920 = 5·2¹⁴ will do. Most of a
/// subfault's work (the normal draws, the envelope, the spectral shape and the Hermitian
/// mirror) is `O(np2)` elementwise passes, so a shorter length wins even where a
/// mixed-radix plan is less efficient per point than a radix-2 one.
///
/// # Why four rungs and not every 7-smooth number
///
/// Because the ladder sets how many [`crate::spectrum::SpectrumPlan`]s a station builds, and a
/// plan carries `O(np2)` precomputed transcendentals. Measured over 1,854 subfault windows
/// spread across 5–209 s, which is an Alpine Fault station's range:
///
/// | ladder | distinct lengths | mean overshoot | plan tables |
/// | --- | --- | --- | --- |
/// | powers of two | 7 | 1.470 | 2.5 MB |
/// | **four per octave** | **23** | **1.096** | **5.7 MB** |
/// | every 7-smooth number | 364 | 1.005 | 97.8 MB |
///
/// The exact ladder is only 9% tighter than this one and costs 17x the memory — per thread,
/// and production runs 32 of them. Four rungs takes 87% of the available saving for 3 MB.
#[must_use]
pub fn good_length(at_least: usize) -> usize {
    // At least 2, so a degenerate window still gives a transform the callers can index and
    // mirror.
    let at_least = at_least.max(2);
    LENGTH_MULTIPLIERS
        .iter()
        .map(|&multiplier| {
            let mut candidate = multiplier;
            // Doubling is what guarantees evenness, so an odd rung is never itself a
            // candidate -- `1` and the other multipliers all start odd.
            while candidate < at_least || !candidate.is_multiple_of(2) {
                candidate = candidate
                    .checked_mul(2)
                    .expect("a transform length that overflows usize is not a real request");
            }
            candidate
        })
        .min()
        .expect("the ladder is never empty")
}

/// Analysis transform.
///
/// Unnormalised; callers divide by the length where they need to.
pub fn forward(data: &mut [Complex32]) {
    transform(data, true)
}

/// Synthesis transform.
///
/// Unnormalised; [`crate::spectrum::radiate_and_invert`] divides by `np2`.
pub fn inverse(data: &mut [Complex32]) {
    transform(data, false)
}

/// In-place unnormalised complex FFT.
fn transform(data: &mut [Complex32], forward: bool) {
    let len = data.len();
    // Even, not necessarily a power of two: `spectrum::stochastic_spectrum` and
    // `site::apply_site_amplification` both re-impose Hermitian symmetry by splitting the
    // spectrum at `len / 2`, and an odd length has no Nyquist bin to mirror about.
    assert!(
        len.is_multiple_of(2),
        "the transform requires an even length, got {len}"
    );

    let plan = PLANS.with(|plans| {
        Arc::clone(plans.borrow_mut().entry((len, forward)).or_insert_with(|| {
            let direction = if forward {
                FftDirection::Forward
            } else {
                FftDirection::Inverse
            };
            FftPlanner::new().plan_fft(len, direction)
        }))
    });

    plan.process(data);
}

pub fn remove_quadratic_trend(dt: f32, acceleration: &mut [f32]) {
    let mut ve = 0.0f32;
    let mut de = 0.0f32;
    let a1 = dt / 2.0;
    let a2 = a1 * dt / 3.0;
    let count = acceleration.len();
    let nstps = count - 1;

    // Trapezoidal double integration.
    let mut previous = acceleration[0];
    for &next in &acceleration[1..] {
        de = de + ve * dt + a2 * (2.0 * previous + next);
        ve += a1 * (previous + next);
        previous = next;
    }

    let rnstp = nstps as f32;
    let t = rnstp * dt;
    let c1 = 2.0 / t * (ve - 3.0 / t * de);
    let c2 = 6.0 / t * (2.0 / t * de - ve) / t;

    azip!((index k, sample in ArrayViewMut1::from(&mut acceleration[2..])) {
        let a3 = (k + 2) as f32;
        *sample = *sample + c1 + c2 * a3 * dt;
    });
}

#[cfg(test)]
mod tests {
    use super::good_length;

    /// The odd part of `value` after dividing out every factor of 2, 3, 5 and 7.
    fn rough_part(mut value: usize) -> usize {
        for factor in [2, 3, 5, 7] {
            while value.is_multiple_of(factor) {
                value /= factor;
            }
        }
        value
    }

    /// The three things a caller relies on, over every length this program can form.
    ///
    /// `at_least` runs to well past the largest transform a production deck reaches
    /// (131,072 for an Alpine Fault subfault at 600 km), stepping so that both small and
    /// large regimes are covered.
    #[test]
    fn a_good_length_is_even_smooth_and_long_enough() {
        let candidates = (0..2000).chain((2000..300_000).step_by(97));
        for at_least in candidates {
            let length = good_length(at_least);
            assert!(
                length >= at_least.max(2),
                "good_length({at_least}) = {length} is shorter than asked for"
            );
            assert!(
                length.is_multiple_of(2),
                "good_length({at_least}) = {length} is odd"
            );
            assert_eq!(
                rough_part(length),
                1,
                "good_length({at_least}) = {length} is not 7-smooth"
            );
        }
    }

    /// The ladder has four rungs per octave, which is what bounds how many
    /// `SpectrumPlan`s a station builds. If a rung is added or removed, the memory
    /// table in the docs above is wrong.
    #[test]
    fn the_ladder_has_four_rungs_per_octave() {
        let rungs: std::collections::BTreeSet<usize> = (1024..2048).map(good_length).collect();
        assert_eq!(
            rungs.into_iter().collect::<Vec<_>>(),
            vec![1024, 1280, 1536, 1792, 2048],
            "the rungs between 1024 and 2048 moved"
        );
    }

    /// The overshoot is bounded by the widest gap in the ladder, 1 -> 5/4.
    #[test]
    fn the_overshoot_never_exceeds_a_quarter() {
        for at_least in (2..300_000).step_by(13) {
            let length = good_length(at_least);
            assert!(
                length as f64 <= at_least as f64 * 1.25,
                "good_length({at_least}) = {length} overshoots by more than 25%"
            );
        }
    }

    #[test]
    fn a_good_length_never_exceeds_the_power_of_two() {
        for at_least in (2..200_000).step_by(31) {
            let length = good_length(at_least);
            let power_of_two = at_least.next_power_of_two();
            assert!(
                length <= power_of_two,
                "good_length({at_least}) = {length} exceeds {power_of_two}"
            );
        }
    }

    /// It is the smallest rung of *this ladder*, not merely one of them.
    #[test]
    fn a_good_length_is_the_smallest_rung() {
        for at_least in 2..20_000usize {
            let length = good_length(at_least);
            let smaller = (at_least..length).find(|candidate| {
                candidate.is_multiple_of(2)
                    && super::LENGTH_MULTIPLIERS
                        .contains(&(candidate >> candidate.trailing_zeros()))
            });
            assert_eq!(
                smaller, None,
                "good_length({at_least}) = {length} but {smaller:?} also qualifies"
            );
        }
    }
}
