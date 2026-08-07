//! The Fortran's generator, reproduced exactly.
//!
//! A hand-rolled PCG32 seeded by folding eight consecutive integers into the state, with
//! normal deviates formed by Box-Muller and the whole block then rescaled to unit mean
//! square.
//!
//! Neither choice is one you would make today, and neither is defended here. They are
//! reproduced because the *stream* is the compatibility contract: a record matches the
//! original only if every consumer draws the same quantity of randomness in the same
//! order. Use [`super::Pcg`] for new work.
//!
//! (orig. `hb_high_ref.f:4033`)

use ndarray::ArrayViewMut1;

use super::{Draws, unit_interval_from};

const PCG_MULT: u64 = 6364136223846793005;
const PCG_INC_DEFAULT: u64 = 1442695040888963407;

/// Number of words the original fed to `random_seed(put=)`.
///
/// gfortran 16.1.1 reports 8 from `random_seed(size=n)` on x86-64. Must equal `seed_words`
/// in `pcg32.f`, because it sets how many words are folded into the state below.
const SEED_WORDS: i32 = 8;

/// PCG32 (O'Neill 2014), `pcg32_random_r` variant, with Box-Muller normals.
///
/// Fortran's `ISHFT` is a *logical* shift, which is why the state is `u64` here rather than
/// `i64`: `>>` on an unsigned type is the exact equivalent. The wrapping multiply matches
/// `-fwrapv` on the Fortran side.
#[derive(Clone, Debug)]
pub struct LegacyPcg {
    state: u64,
    inc: u64,
}

impl LegacyPcg {
    /// Equivalent of `init_random_seed(irand)`.
    pub fn seed(irand: i32) -> Self {
        let mut state: u64 = 0;
        let mut irand = irand;
        for _ in 0..SEED_WORDS {
            // int(irand,8) sign-extends, so route through i64.
            state = state
                .wrapping_mul(PCG_MULT)
                .wrapping_add(irand as i64 as u64);
            irand += 1;
        }
        let mut generator = Self {
            state,
            inc: PCG_INC_DEFAULT,
        };
        // Two discarded draws, matching pcg32.f.
        generator.next_u32();
        generator.next_u32();
        generator
    }

    /// Equivalent of `pcg32_next()`.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULT).wrapping_add(self.inc);

        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// A uniform deviate that is not zero, which is what `ln` downstream requires.
    #[inline]
    fn nonzero_uniform(&mut self) -> f32 {
        let mut value = self.uniform();
        while value == 0.0 {
            value = self.uniform();
        }
        value
    }
}

impl Draws for LegacyPcg {
    /// Equivalent of `next_f32(0)`. Returns `f32` in `[0, 1 - 2^-24]`.
    #[inline]
    fn uniform(&mut self) -> f32 {
        unit_interval_from(self.next_u32())
    }

    /// Box-Muller, taking the cosine component and discarding the sine partner — two
    /// uniform draws per call.
    ///
    /// # Why this is not [`Self::fill_normal`] with a length-1 slice
    ///
    /// That routine renormalises its whole output so the mean square equals one, which for
    /// a single value would force it to exactly ±1 and destroy the distribution. The
    /// renormalisation only makes sense over a block.
    fn normal(&mut self) -> f32 {
        let first = self.nonzero_uniform();
        let second = self.nonzero_uniform();
        let radius = (-first.ln() * 2.0).sqrt();
        radius * (std::f32::consts::TAU * second).cos()
    }

    /// Box-Muller pairs, then the whole vector rescaled so `sum(out^2) == out.len()`.
    ///
    /// # The rescale is not load-bearing for the spectrum
    ///
    /// It looks as though [`crate::stoc::stochastic_spectrum`]'s amplitude calibration must
    /// want a unit-RMS sequence. **It does not.** Trace a scale factor `s` through that
    /// routine: `a` is proportional to `s`; `remove_quadratic_trend` is linear and
    /// homogeneous of degree 1, so its output is too; `ac = a * w` and the forward
    /// transform are linear, so `ac ∝ s`; therefore `fsa = sum|ac|^2 ∝ s^2` and
    /// `amp = 1/(dt*sqrt(fsa/fold_count)) ∝ 1/s`. The product `ac * as_ * amp` is
    /// **proportional to `s^0`**, so the two cancel exactly.
    ///
    /// It is kept here anyway, because this implementation's whole job is to reproduce a
    /// stream rather than to be defensible, and the rescale is part of what the Fortran
    /// did. [`super::Pcg`] does not carry it.
    ///
    /// Draw accounting, which the shared stream depends on: `2*ceil(len/2)` draws plus one
    /// extra per rejected zero. When the length is odd the sine partner of the final pair
    /// is generated and discarded.
    fn fill_normal(&mut self, out: &mut [f32]) {
        // `radius` and `angle` persist across iterations: the odd-numbered draw computes
        // the pair and returns the cosine component, the even-numbered one returns the
        // sine component from the *same* pair. In the Fortran they are ordinary locals
        // whose values survive between iterations of the DO 8 loop.
        let mut radius = 0.0f32;
        let mut angle = 0.0f32;
        let mut cosine_next = true;

        for slot in out.iter_mut() {
            *slot = if cosine_next {
                radius = self.nonzero_uniform();
                angle = self.nonzero_uniform();
                // `angle *= TAU` rather than the Fortran's `TAU * angle`: IEEE 754
                // multiplication commutes bit-for-bit, so the operand order was never
                // load-bearing here -- unlike the ADDITION order in the reductions.
                angle *= std::f32::consts::TAU;
                radius = -radius.ln();
                radius = (radius + radius).sqrt();
                cosine_next = false;
                radius * angle.cos()
            } else {
                cosine_next = true;
                radius * angle.sin()
            };
        }

        if out.is_empty() {
            return;
        }
        // THE SUM STAYS A LEFT-TO-RIGHT f32 FOLD, and `Sum for f32` is one -- matching the
        // Fortran's `s = s + ..`. This is a REDUCTION, not an elementwise operation, so it
        // is not order-independent: any reassociating form (chunked, pairwise, parallel,
        // `ndarray`'s `.sum()`) would move every waveform this source produces.
        let sum_of_squares: f32 = out.iter().map(|&v| v * v).sum();
        // The count is promoted to real*4 for the division, and the sqrt is single
        // precision. Do not compute this in f64.
        let scale = (out.len() as f32 / sum_of_squares).sqrt();
        // The rescale IS elementwise, so it is the ndarray operator. Bound to a name
        // because `*=` needs a place expression, not a temporary.
        let mut scaled = ArrayViewMut1::from(out);
        scaled *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block form pins the mean square to exactly one, which is the property the
    /// Fortran's rescale exists to provide.
    #[test]
    fn the_block_form_has_unit_mean_square() {
        let mut generator = LegacyPcg::seed(42);
        for count in [1usize, 2, 3, 15, 16, 1000] {
            let mut out = vec![0.0f32; count];
            generator.fill_normal(&mut out);
            let sum_of_squares: f32 = out.iter().map(|v| v * v).sum();
            assert!(
                (sum_of_squares / count as f32 - 1.0).abs() < 1e-4,
                "count={count}: mean square {} != 1",
                sum_of_squares / count as f32
            );
        }
    }

    /// `2*ceil(len/2)` draws, absent zero rejections.
    ///
    /// This is the accounting the tier-4 golden depends on, so it is asserted rather than
    /// assumed: a fresh generator advanced by hand must land where the routine leaves one.
    #[test]
    fn the_block_form_draw_count_is_fixed() {
        for count in [1usize, 2, 3, 4, 7, 8] {
            let mut driven = LegacyPcg::seed(7);
            let mut out = vec![0.0f32; count];
            driven.fill_normal(&mut out);

            let mut by_hand = LegacyPcg::seed(7);
            for _ in 0..2 * count.div_ceil(2) {
                by_hand.uniform();
            }
            assert_eq!(
                driven.next_u32(),
                by_hand.next_u32(),
                "count={count}: stream position diverged"
            );
        }
    }
}
