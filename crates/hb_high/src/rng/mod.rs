//! Where the randomness in a seismogram comes from.
//!
//! Every stochastic quantity the program produces — the windowed noise each subfault's
//! spectrum is built from, the conical averaging of the radiation pattern, the
//! rupture-velocity perturbation — is drawn from one generator per station. The physics
//! does not care *which* generator, but it cares enormously that the answer is
//! reproducible from a seed. So the draw source is a trait with three implementations,
//! and the choice is a deployment decision rather than a property of the model:
//!
//! * [`Pcg`] is what production uses: PCG32 for the uniforms and the ziggurat algorithm
//!   for the normals.
//!
//! * [`LegacyPcg`] reproduces the Fortran exactly — the hand-rolled PCG32, with normals
//!   formed by Box-Muller and the whole block rescaled to unit mean square. It exists to
//!   drive the tier-0 and tier-4 goldens, which compare against binaries dumped by
//!   `hb_high_v6.0.3`, and to regenerate historical results. It is slower and it is not
//!   meant to be otherwise.
//!
//! * [`FixtureDraws`] is the cheap gate's source: frozen uniforms, production normals.
//!   See its own docs for why it is frozen and what that does and does not buy.
//!
//! # The implementations are not interchangeable mid-run, and that is the point
//!
//! Swapping the source changes every waveform, because every consumer shares one stream.
//! It also changes the *number of draws* a normal costs — two uniforms plus three
//! transcendentals for Box-Muller against a ziggurat sample that usually costs one
//! uniform and no transcendental at all — so it is not merely a different sequence, it is
//! a different consumption pattern. Choose one per run and record which.
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
/// and the production path keeps a direct call. The normal-draw loop is the largest single
/// consumer of RNG traffic in the program; it should not pay a branch to be testable.
pub trait Draws {
    /// A uniform deviate on `[0, 1)`, advancing the stream.
    ///
    /// **The half-open range is a contract, not a convention.** Box-Muller rejects zeros
    /// by re-drawing and would break on `-ln(x)` at the other end, so a source that could
    /// return exactly `1.0` is not a valid implementation. Every implementation here takes
    /// the top 24 bits of a word over `2^24`, which is exact; the obvious `u32 / 2^32` rounds
    /// values near 1 *up* to exactly 1.0 and would break the contract.
    fn uniform(&mut self) -> f32;

    /// One standard normal deviate, advancing the stream.
    fn normal(&mut self) -> f32;

    /// Fill `out` with standard normal deviates.
    ///
    /// Overridable because the *block* is the unit some sources normalise over — see
    /// [`LegacyPcg::fill_normal`], which rescales to unit mean square and so cannot be
    /// expressed as a loop over [`Draws::normal`].
    fn fill_normal(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.normal();
        }
    }

    /// Fill `out` with uniform deviates.
    ///
    /// Sequential by necessity — each slot takes the next draw, and the order *is* the
    /// stream.
    fn fill_uniform(&mut self, out: &mut [f32]) {
        for slot in out.iter_mut() {
            *slot = self.uniform();
        }
    }
}

/// Which draw source a run uses.
///
/// Chosen once per run from the environment, because the deck format is a downstream
/// interface contract and cannot grow a field.
pub enum DrawSource {
    /// **The default.** See [`Pcg`].
    Modern(Pcg),
    /// Validation only — see [`FixtureDraws`]. Opt-in via `HB_FIXTURE_RNG`.
    Fixture(FixtureDraws),
}

impl DrawSource {
    /// Build the run's draw source.
    ///
    /// # What was wrong with the old seeding
    ///
    /// `init_random_seed` folded `irand, irand+1, ..., irand+7` into the state — eight
    /// nearly-identical values — and left `inc` at its default for every run. That
    /// reduces exactly to
    ///
    /// ```text
    /// state = C * seed + D        (mod 2^64), C and D fixed
    /// ```
    ///
    /// so it is an **affine map, not entropy mixing**, and every seed lands on the *same*
    /// LCG orbit at a different offset. PCG's stream parameter — the thing that exists to
    /// give genuinely independent sequences — was never used.
    ///
    /// Measured honestly: this does **not** show up as correlation between the draw
    /// streams of nearby seeds (max |r| 0.081 against a 0.067 noise floor over 60 seeds
    /// x 2000 draws, indistinguishable from properly-seeded). `C` is large enough that
    /// adjacent seeds land far apart on the orbit. It is replaced because it is
    /// indefensible on its own terms, not because a specific defect was traced to it.
    ///
    /// # Why the seed is a `u64`
    ///
    /// A station's seed is its identity, and identities should not be a scarce resource.
    /// The deck could only express `i32`, which is what forced `hf_sim.py` to derive station
    /// seeds as `int32(root) ^ stable_hash(name)`, landing about half of them negative.
    /// What the wider type buys is the other 2⁶⁴ − 2³² seeds, which is the space
    /// `numpy.random.SeedSequence` draws station seeds from on the Python side.
    pub fn for_station(seed: u64) -> Self {
        if std::env::var_os("HB_FIXTURE_RNG").is_some() {
            Self::Fixture(FixtureDraws::seed(seed as i32))
        } else {
            Self::Modern(Pcg::seed(seed))
        }
    }
}

impl Draws for DrawSource {
    #[inline]
    fn uniform(&mut self) -> f32 {
        match self {
            Self::Modern(g) => g.uniform(),
            Self::Fixture(g) => g.uniform(),
        }
    }

    #[inline]
    fn normal(&mut self) -> f32 {
        match self {
            Self::Modern(g) => g.normal(),
            Self::Fixture(g) => g.normal(),
        }
    }

    #[inline]
    fn fill_normal(&mut self, out: &mut [f32]) {
        match self {
            Self::Modern(g) => g.fill_normal(out),
            Self::Fixture(g) => g.fill_normal(out),
        }
    }

    #[inline]
    fn fill_uniform(&mut self, out: &mut [f32]) {
        match self {
            Self::Modern(g) => g.fill_uniform(out),
            Self::Fixture(g) => g.fill_uniform(out),
        }
    }
}

/// The `[0, 1)` conversion every source in this module uses.
///
/// Takes the top 24 bits and divides by `2^24`: the integer-to-`f32` conversion is
/// lossless and the divisor is a power of two, so the result carries no rounding.
/// Dividing a full 32-bit value by `2^32` would round, and values near 1 would round *up*
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

    /// The tails are the part Box-Muller's replacement had to get right.
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
