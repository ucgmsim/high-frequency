//! The original Fortran generator, reproduced exactly.
//!
//! A hand-rolled PCG32 seeded by folding eight consecutive integers into the state, with
//! normal deviates formed by Box-Muller and the whole block then rescaled to unit mean
//! square. The stream is the compatibility contract: a record matches the original only if
//! every consumer draws the same quantity of randomness in the same order. Use
//! [`super::Pcg`] for new work.

use ndarray::ArrayViewMut1;

use super::{Draws, unit_interval_from};

const PCG_MULT: u64 = 6364136223846793005;
const PCG_INC_DEFAULT: u64 = 1442695040888963407;

/// Number of consecutive seed words folded into the state, as the original seeding did.
const SEED_WORDS: i32 = 8;

/// PCG32 (O'Neill 2014), `pcg32_random_r` variant, with Box-Muller normals.
///
/// The state is `u64` so that `>>` is a logical shift, and the multiply wraps, both as in
/// the original.
#[derive(Clone, Debug)]
pub struct LegacyPcg {
    state: u64,
    inc: u64,
}

impl LegacyPcg {
    /// Seed by folding `irand, irand+1, ..., irand+7` into the state, then discard two draws.
    pub fn seed(irand: i32) -> Self {
        let mut state: u64 = 0;
        for word in irand..irand + SEED_WORDS {
            // Sign-extend through i64.
            state = state
                .wrapping_mul(PCG_MULT)
                .wrapping_add(word as i64 as u64);
        }
        let mut generator = Self {
            state,
            inc: PCG_INC_DEFAULT,
        };
        // Two discarded draws.
        generator.next_u32();
        generator.next_u32();
        generator
    }

    /// One PCG32 output word.
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
    /// Narrowed to `i32` because that is the width this generator's seeding was defined at.
    fn respawn(&self, seed: u64) -> Self {
        Self::seed(seed as i32)
    }

    /// Returns `f32` in `[0, 1 - 2^-24]`.
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
    /// The rescale does not affect the spectrum: [`crate::stoc::stochastic_spectrum`]
    /// normalises by the realised power, which cancels any scale factor. It is kept only to
    /// reproduce the original stream; [`super::Pcg`] does not carry it.
    ///
    /// Draw accounting, which the shared stream depends on: `2*ceil(len/2)` draws plus one
    /// extra per rejected zero. When the length is odd the sine partner of the final pair
    /// is generated and discarded.
    fn fill_normal(&mut self, out: &mut [f32]) {
        // `radius` and `angle` persist across iterations: the odd-numbered draw computes
        // the pair and returns the cosine component, the even-numbered one returns the
        // sine component from the same pair.
        let mut radius = 0.0f32;
        let mut angle = 0.0f32;
        let mut cosine_next = true;

        for slot in out.iter_mut() {
            *slot = if cosine_next {
                radius = self.nonzero_uniform();
                angle = self.nonzero_uniform();
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
        // The sum must stay a left-to-right f32 fold, as in the original; any reassociating
        // form (chunked, pairwise, `ndarray`'s `.sum()`) would move every waveform this
        // source produces.
        let sum_of_squares: f32 = out.iter().map(|&v| v * v).sum();
        // Single precision throughout, as in the original. Do not compute this in f64.
        let scale = (out.len() as f32 / sum_of_squares).sqrt();
        // Bound to a name because `*=` needs a place expression, not a temporary.
        let mut scaled = ArrayViewMut1::from(out);
        scaled *= scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The block form pins the mean square to exactly one.
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
    /// The tier-4 golden depends on this accounting: a fresh generator advanced by hand must
    /// land where the routine leaves one.
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
