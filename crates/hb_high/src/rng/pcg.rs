//! The production draw source: PCG32 uniforms, ziggurat normals.
//!
//! * PCG32 for the uniforms, seeded through `rand_core`'s `seed_from_u64` expansion. That
//!   runs the seed through an avalanche mix and fills both the state and the increment
//!   from it, so different station seeds get different streams rather than different
//!   offsets in one.
//!
//! * The ziggurat algorithm for the normals, via `rand_distr`. It accepts on the first try
//!   roughly 99% of the time, so an exact normal deviate costs about one uniform and no
//!   transcendental, against Box-Muller's two uniforms plus a `ln`, a `sqrt` and a `cos`.

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
    /// `seed_from_u64` is a SplitMix64 expansion, so nearby sub-stream seeds land on
    /// unrelated states rather than at nearby offsets of one.
    fn respawn(&self, seed: u64) -> Self {
        Self::seed(seed)
    }

    #[inline]
    fn uniform(&mut self) -> f32 {
        unit_interval_from(self.generator.next_u32())
    }

    #[inline]
    fn normal(&mut self) -> f32 {
        self.generator.sample(StandardNormal)
    }
}
