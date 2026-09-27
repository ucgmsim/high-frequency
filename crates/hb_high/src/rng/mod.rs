//! Where the randomness in a seismogram comes from.
//!
//! Every stochastic quantity the program produces — the windowed noise each subfault's
//! spectrum is built from, the conical averaging of the radiation pattern, the
//! rupture-velocity perturbation — is drawn from one generator per station. The physics
//! does not care which generator, but the answer must be reproducible from a seed. So the
//! draw source is a trait with three implementations:
//!
//! * [`Pcg`] is what production uses: PCG32 for the uniforms and the ziggurat algorithm
//!   for the normals.
//!
//! * [`LegacyPcg`] reproduces the original Fortran generator exactly — a hand-rolled PCG32,
//!   with normals formed by Box-Muller and the whole block rescaled to unit mean square. It
//!   drives the tier-0 and tier-4 goldens, which compare against binaries dumped by
//!   `hb_high_v6.0.3`.
//!
//! * [`FixtureDraws`] is the snapshot test's source: frozen uniforms, production normals.
//!
//! # The implementations are not interchangeable mid-run
//!
//! Swapping the source changes every waveform. It also changes the number of uniforms a
//! normal costs — two for Box-Muller, usually one for the ziggurat — so it is a different
//! consumption pattern, not merely a different sequence. Choose one per run.
//!
//! What must hold for all of them, and what the tests assert, is the contract on
//! [`Draws`]: the same seed gives the same record, and different stations are independent.

mod compat;
mod fixture;
mod pcg;

pub use compat::LegacyPcg;
pub use fixture::FixtureDraws;
pub use pcg::Pcg;

/// A source of the random deviates a seismogram is built from.
///
/// Implementations are free to use any algorithm. What they may not do is vary their
/// output for a given construction — a source is a deterministic function of its seed, or
/// nothing downstream is reproducible.
///
/// Consumers are generic over this rather than taking an enum, so each is monomorphised
/// and the normal-draw loop, the largest consumer of RNG traffic, keeps a direct call.
pub trait Draws {
    /// A source of this kind, started at `seed`.
    ///
    /// A station's stream and each `(subfault, ray)`'s sub-stream are all built this way, so
    /// a sub-stream's draws are a function of its own identity alone and a subfault that
    /// contributes nothing can be skipped without moving the others. See `substream_seed`
    /// for how the identity is formed.
    fn from_seed(seed: u64) -> Self
    where
        Self: Sized;

    /// A uniform deviate on `[0, 1)`, advancing the stream.
    ///
    /// The half-open range is a contract: Box-Muller rejects zeros by re-drawing and would
    /// break on `-ln(x)` at the other end, so a source must never return exactly `1.0`. See
    /// `unit_interval_from`.
    fn uniform(&mut self) -> f32;

    /// One standard normal deviate, advancing the stream.
    fn normal(&mut self) -> f32;

    /// Fill `out` with standard normal deviates.
    ///
    /// Overridable because the block is the unit some sources normalise over — see
    /// [`LegacyPcg::fill_normal`], which rescales to unit mean square and so cannot be
    /// expressed as a loop over [`Draws::normal`].
    fn fill_normal(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.normal();
        }
    }

    /// Fill `out` with uniform deviates.
    ///
    /// Sequential: each slot takes the next draw.
    fn fill_uniform(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.uniform();
        }
    }
}

/// SplitMix64's increment: the golden-ratio odd constant (Steele et al. 2014).
pub(crate) const SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// SplitMix64's finalising mix (Steele et al. 2014).
///
/// A bijection on `u64`, which is the property the seeding below needs: distinct inputs
/// stay distinct, so two subfaults cannot be handed the same stream by an unlucky collision.
#[inline]
pub(crate) fn splitmix_finalise(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The seed for one subfault's own stream.
///
/// A subfault's identity is `(station, segment, grid position)` and nothing about the *order*
/// it was walked in; see [`Draws::from_seed`] for what that decouples. The gamma multiply
/// keeps small indices from mapping to small perturbations of the seed.
pub(crate) fn substream_seed(station_seed: u64, segment: usize, subfault: usize) -> u64 {
    splitmix_finalise(
        splitmix_finalise(station_seed ^ (segment as u64).wrapping_mul(SPLITMIX_GAMMA))
            ^ subfault as u64,
    )
}

/// The seed for one ray path's stream within a subfault.
///
/// Derived from the subfault's seed rather than from the station's, so a ray path is
/// identified relative to the subfault it leaves — and so that adding a ray type to the
/// rayset cannot renumber another subfault's streams.
pub(crate) fn ray_stream_seed(subfault_seed: u64, ray: usize) -> u64 {
    splitmix_finalise(subfault_seed ^ (ray as u64).wrapping_add(1).wrapping_mul(SPLITMIX_GAMMA))
}

/// The `[0, 1)` conversion every source in this module uses.
///
/// Takes the top 24 bits and divides by `2^24`: the integer-to-`f32` conversion is
/// lossless and the divisor is a power of two, so the result carries no rounding.
/// Dividing a full 32-bit value by `2^32` would round, and values near 1 would round up
/// to exactly 1.0, breaking the `[0, 1)` contract that Box-Muller's zero-rejection loops
/// assume.
#[inline]
pub(crate) fn unit_interval_from(word: u32) -> f32 {
    (word >> 8) as f32 / 16_777_216.0
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `[0, 1)` contract, for every implementation.
    #[test]
    fn every_source_stays_in_the_unit_interval() {
        let mut legacy = LegacyPcg::seed(123_456_789);
        let mut modern = Pcg::seed(123_456_789);
        let mut fixture = FixtureDraws::seed(123_456_789);
        for _ in 0..100_000 {
            for value in [legacy.uniform(), modern.uniform(), fixture.uniform()] {
                assert!((0.0..1.0).contains(&value), "uniform returned {value}");
            }
        }
    }

    /// Reproducibility, for every implementation. Same seed, same sequence.
    #[test]
    fn every_source_replays_from_its_seed() {
        fn replays<D: Draws>(build: impl Fn() -> D) {
            let (mut first, mut second) = (build(), build());
            for _ in 0..1_000 {
                assert_eq!(first.uniform().to_bits(), second.uniform().to_bits());
                assert_eq!(first.normal().to_bits(), second.normal().to_bits());
            }
        }
        replays(|| LegacyPcg::seed(4242));
        replays(|| Pcg::seed(4242));
        replays(|| FixtureDraws::seed(4242));
    }

    /// The normals are standard, for every implementation.
    ///
    /// A tolerance test, not a bit test: the whole point of the trait is that the
    /// algorithms differ. The band is `5/sqrt(n)`, comfortably outside sampling noise at
    /// this `n` and tight enough to catch a wrong scale or a missing mean shift.
    #[test]
    fn every_source_draws_standard_normals() {
        const COUNT: usize = 200_000;
        fn standard<D: Draws>(mut rng: D, name: &str) {
            let mut out = vec![0.0f32; COUNT];
            rng.fill_normal(&mut out);
            let mean = out.iter().map(|v| f64::from(*v)).sum::<f64>() / COUNT as f64;
            let variance = out
                .iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                / COUNT as f64;
            let band = 5.0 / (COUNT as f64).sqrt();
            assert!(mean.abs() < band, "{name}: mean {mean} outside +/-{band}");
            assert!(
                (variance - 1.0).abs() < band,
                "{name}: variance {variance} outside 1+/-{band}"
            );
        }
        standard(LegacyPcg::seed(11), "LegacyPcg");
        standard(Pcg::seed(11), "Pcg");
        standard(FixtureDraws::seed(11), "FixtureDraws");
    }

    /// The production normals reach into the tails.
    ///
    /// A ziggurat sample is an exact normal, so beyond 4 sigma it must produce roughly the
    /// 6.3 in 100,000 the distribution calls for rather than nothing at all. This is the
    /// property a bounded construction would fail.
    #[test]
    fn the_modern_source_reaches_into_the_tails() {
        const COUNT: usize = 400_000;
        let mut out = vec![0.0f32; COUNT];
        Pcg::seed(7).fill_normal(&mut out);
        let beyond = out.iter().filter(|v| v.abs() > 4.0).count();
        // Expectation is ~25 at this count; the band is wide because this is a Poisson
        // count, and the failure being guarded against is zero.
        assert!(
            (5..80).contains(&beyond),
            "{beyond} draws beyond 4 sigma, expected about 25"
        );
    }
}
