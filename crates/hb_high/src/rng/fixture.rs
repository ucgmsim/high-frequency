//! A draw source for validation only, whose uniforms are frozen.
//!
//! `tests/snapshot.rs` pins the whole pipeline against numbers checked into
//! `harness/golden/snapshot.txt`. Driving it from a source independent of the production
//! generator (which is allowed to change) means a difference is attributable to the code,
//! because the uniform sequence did not move.
//!
//! Do not change the recurrence or the `[0, 1)` conversion below. Its only job is to produce
//! the same sequence indefinitely; nothing scientific is computed from it, and it needs only
//! a decent spread over `[0, 1)`. It is SplitMix64 (Steele et al. 2014): short, and no state
//! beyond a counter.
//!
//! The normals are deliberately not frozen: they come from the ziggurat, the same
//! distribution code production runs, so the snapshot sees a change in how a normal deviate
//! is formed. A `rand_distr` upgrade can therefore move the snapshot, which is why its
//! version is pinned in `Cargo.toml`.

use rand_core::{Infallible, TryRng, utils};

use super::{Draws, unit_interval_from};

/// SplitMix64's increment and its two finalising multipliers.
const GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;

/// The validation draw source. See the module docs.
#[derive(Clone, Debug)]
pub struct FixtureDraws {
    state: u64,
}

impl FixtureDraws {
    /// The deck's seed is folded in so different seeds still give different runs — the
    /// gate compares two builds at matched seeds, not one build against a constant.
    pub fn seed(irand: i32) -> Self {
        Self {
            state: (irand as i64 as u64) ^ GAMMA,
        }
    }

    /// One SplitMix64 output word. Frozen — see the module docs.
    fn next_word(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
        z ^ (z >> 31)
    }
}

impl Draws for FixtureDraws {
    /// Narrowed to `i32` because that is what [`FixtureDraws::seed`] takes; the seeds only
    /// have to be distinct, not well spread.
    fn respawn(&self, seed: u64) -> Self {
        Self::seed(seed as i32)
    }

    /// The frozen `[0, 1)` sequence.
    ///
    /// Bits 40..64 of the output word over `2^24`. `the_uniform_sequence_is_frozen` pins it.
    #[inline]
    fn uniform(&mut self) -> f32 {
        unit_interval_from((self.next_word() >> 32) as u32)
    }

    /// The ziggurat, via the trait's default. Not frozen — see the module docs.
    fn normal(&mut self) -> f32 {
        use rand::RngExt as _;
        self.sample(rand_distr::StandardNormal)
    }
}

/// So the ziggurat can draw raw words from this source.
///
/// One word per call, from the same recurrence [`FixtureDraws::uniform`] uses, so mixing
/// uniform and normal draws stays deterministic — each consumes exactly one advance.
impl TryRng for FixtureDraws {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok((self.next_word() >> 32) as u32)
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(self.next_word())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        utils::fill_bytes_via_next_word(dst, || self.try_next_u64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The frozen sequence, pinned by value.
    ///
    /// If this fails, the recurrence or the `[0, 1)` conversion moved, and every snapshot
    /// recorded before it is void.
    ///
    /// Recorded as `f32` bit patterns, computed from the recurrence independently in Python
    /// rather than copied from this implementation's output.
    #[test]
    fn the_uniform_sequence_is_frozen() {
        let mut fixture = FixtureDraws::seed(20260807);
        let drawn: Vec<u32> = (0..6).map(|_| fixture.uniform().to_bits()).collect();
        assert_eq!(
            drawn,
            vec![
                1047616700, 1063987193, 1031711536, 1052079132, 1064334156, 1029383216
            ],
            "the frozen uniform sequence moved"
        );
    }

    /// A uniform draw and a raw word draw must cost the same one advance, or interleaving
    /// them would depend on which happened to be called.
    #[test]
    fn a_uniform_and_a_word_cost_the_same_advance() {
        let mut through_uniform = FixtureDraws::seed(5);
        let mut through_word = FixtureDraws::seed(5);
        through_uniform.uniform();
        through_word.try_next_u32().unwrap();
        assert_eq!(
            through_uniform.next_word(),
            through_word.next_word(),
            "the two paths advanced the state differently"
        );
    }
}
