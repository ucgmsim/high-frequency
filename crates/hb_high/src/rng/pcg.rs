//! The production draw source: PCG32 uniforms, ziggurat normals.
//!
//! This is what production uses. Both halves are chosen, not inherited:
//!
//! * **PCG32** for the uniforms, seeded through `rand_core`'s `seed_from_u64` expansion.
//!   That runs the seed through an avalanche mix and fills **both** the state and the
//!   increment from it, so different station seeds get different *streams* rather than
//!   different offsets in one — which is what PCG's stream parameter is for and what the
//!   Fortran's affine seed fold never used. See [`super::DrawSource::for_station`].
//!
//! * **The ziggurat algorithm** for the normals, via `rand_distr`. It accepts on the first
//!   try roughly 99% of the time, so a normal deviate costs about one uniform and no
//!   transcendental, against Box-Muller's two uniforms plus a `ln`, a `sqrt` and a `cos`.
//!   Measured on the Alpine deck, the normal-draw loop was 31% of a station's runtime.
//!
//! It is also an *exact* normal. Box-Muller is too, so that is not the argument here — the
//! argument is cost. What the ziggurat additionally removes is the zero-rejection branch
//! Box-Muller needs, which existed only because `ln(0)` is not a number.

use rand::{RngExt as _, SeedableRng};
use rand_core::Rng as _;
use rand_distr::StandardNormal;

use super::{Draws, unit_interval_from};

/// The production draw source.
#[derive(Clone, Debug)]
pub struct Pcg {
    generator: rand_pcg::Pcg32,
}

impl Pcg {
    /// Start a stream at `seed`.
    #[must_use]
    pub fn seed(seed: u64) -> Self {
        Self {
            generator: rand_pcg::Pcg32::seed_from_u64(seed),
        }
    }
}

impl Draws for Pcg {
    #[inline]
    fn uniform(&mut self) -> f32 {
        unit_interval_from(self.generator.next_u32())
    }

    #[inline]
    fn normal(&mut self) -> f32 {
        self.generator.sample(StandardNormal)
    }
}
