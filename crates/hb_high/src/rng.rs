//! Random number generation.
//!
//! `Pcg32` mirrors `reference/pcg32.f` line for line; the two are meant to be
//! diffed side by side. `normal_deviates` and `uniform_deviates` are transliterations
//! of the Fortran routines at `hb_high_ref.f:4033` and `:2428`, unchanged from
//! the original — they are algorithm, not generator.
//!
//! The entire output of the program depends on the draw sequence, so the
//! *number* of draws each routine consumes matters as much as their values.
//! See `PORTING_RULES.md` §5 on iteration order.
//!
//! # This module is deliberately left un-tidied
//!
//! §2.8 converted the rest of the crate's index loops to iterators. The four in here
//! were skipped on purpose, and the `#[allow]` below is the record of that decision
//! rather than an oversight: `REFACTOR.md` §2.7 replaces this whole module with
//! `rand_pcg`, gated on a draw-for-draw equality test, and "What not to do" is explicit
//! that "small" is not a reason to disturb a generator whose stream is baked into every
//! golden. Cleaning up code that is queued for deletion buys nothing and spends the one
//! thing this file has — a diff against `reference/pcg32.f` that a reader can follow.

// `needless_range_loop`: the loops write `out[..count]` sequentially and are correct as
// iterators. `assign_op_pattern`: the Box-Muller step is written in the Fortran's order.
// Both are staying until §2.7 deletes the module; see the note above.
#![allow(clippy::needless_range_loop, clippy::assign_op_pattern)]

/// A source of uniform deviates in `[0, 1)`.
///
/// This exists so that *validation* can drive the program with a draw source that is not
/// the production generator. Replacing the RNG is the one change that moves every number
/// at once, so a gate that compares two builds needs a source both can share — otherwise
/// there is no way to tell a refactoring mistake from the engine change carrying it.
///
/// Consumers are generic over this rather than taking an enum, so each is monomorphised
/// and the production path keeps a direct call. The draw loop is 80% of the program's
/// RNG traffic; it should not pay a branch to be testable.
///
/// **The `[0, 1)` half-open range is a contract, not a convention.**
/// `fill_normal_deviates` rejects zeros by re-drawing, and a source that could return
/// exactly 1.0 would break `-ln(x)` at the other end. See [`Pcg32::next_f32`] for why the
/// obvious `u32 / 2^32` does not satisfy it.
pub trait Draws {
    fn next_f32(&mut self) -> f32;
}

const PCG_MULT: u64 = 6364136223846793005;
const PCG_INC_DEFAULT: u64 = 1442695040888963407;

/// Number of words the original fed to `random_seed(put=)`.
///
/// gfortran 16.1.1 reports 8 from `random_seed(size=n)` on x86-64. This is not
/// a free choice: the original `init_random_seed` incremented its own argument
/// once per word, and `hb_high` reads that mutated value at line 1366 to gate
/// rupture-time jitter. Pinning it here keeps the branch behaving as production
/// does, independent of the toolchain. Must equal `seed_words` in `pcg32.f`.
pub const SEED_WORDS: i32 = 8;

/// PCG32 (O'Neill 2014), `pcg32_random_r` variant.
///
/// Fortran's `ISHFT` is a *logical* shift, which is why the state is `u64` here
/// rather than `i64`: `>>` on an unsigned type is the exact equivalent. The
/// wrapping multiply matches `-fwrapv` on the Fortran side.
#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Equivalent of `init_random_seed(irand)`.
    ///
    /// Returns the mutated seed alongside the generator, because the Fortran
    /// mutates its argument in place and the caller reads it afterwards. Making
    /// that a return value rather than a hidden side effect is the one place
    /// this file departs from a literal transliteration — the coupling is too
    /// important to leave implicit.
    pub fn seed(irand: i32) -> (Self, i32) {
        let mut state: u64 = 0;
        let mut irand = irand;
        for _ in 0..SEED_WORDS {
            // int(irand,8) sign-extends, so route through i64.
            state = state
                .wrapping_mul(PCG_MULT)
                .wrapping_add(irand as i64 as u64);
            irand += 1;
        }
        let mut g = Self { state, inc: PCG_INC_DEFAULT };
        // Two discarded draws, matching pcg32.f.
        g.next_u32();
        g.next_u32();
        (g, irand)
    }

    /// Equivalent of `pcg32_next()`.
    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(PCG_MULT).wrapping_add(self.inc);

        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Equivalent of `next_f32(0)`. Returns `f32` in `[0, 1 - 2^-24]`.
    ///
    /// Takes the top 24 bits and divides by `2^24`: the integer-to-`f32`
    /// conversion is lossless and the divisor is a power of two, so the result
    /// carries no rounding. Dividing a full 32-bit value by `2^32` would round,
    /// and values near 1 would round *up* to exactly 1.0, breaking the `[0,1)`
    /// contract that `normal_deviates`'s zero-rejection loops assume.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / 16777216.0
    }
}

impl Draws for Pcg32 {
    #[inline]
    fn next_f32(&mut self) -> f32 {
        Pcg32::next_f32(self)
    }
}

/// A draw source for **validation only**, and frozen forever.
///
/// # Why this exists
///
/// The cheap per-commit gate compares two builds of this program over the deck ladder.
/// That only means anything if both builds see the *same* draws — and the whole point of
/// Stage 3 is that the production generator is going to change. Driving both sides from a
/// source that is not the production generator makes the comparison independent of it: a
/// difference is then attributable to the code, because the draws provably did not move.
///
/// # Frozen means frozen
///
/// **Do not change this algorithm, ever.** Not to improve it, not to match a new
/// production engine, not to make it faster. Its only job is to produce the same sequence
/// today and in five years, so that a baseline recorded now is still comparable then. It
/// has no statistical burden to carry: nothing scientific is computed from it, and the
/// only property it needs is a decent spread over `[0, 1)` so the code paths exercised
/// are representative.
///
/// SplitMix64 (Steele et al. 2014), chosen because it is short enough to be obviously
/// correct and has no state beyond a counter. The `[0, 1)` conversion is the same top-24-
/// bits form [`Pcg32::next_f32`] uses, and for the same reason.
#[derive(Clone, Debug)]
pub struct FixtureDraws {
    state: u64,
}

impl FixtureDraws {
    /// The deck's seed is folded in so different seeds still give different runs — the
    /// gate compares two binaries at matched seeds, not one binary against a constant.
    pub fn seed(irand: i32) -> Self {
        Self { state: (irand as i64 as u64) ^ 0x9E37_79B9_7F4A_7C15 }
    }
}

impl Draws for FixtureDraws {
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Top 24 bits over 2^24, exactly as `Pcg32::next_f32` does, so the `[0, 1)`
        // contract holds identically.
        ((z >> 40) as u32) as f32 / 16777216.0
    }
}

/// Which draw source a run uses.
///
/// Chosen once per run from the environment, because the deck format is a downstream
/// interface contract and cannot grow a field.
pub enum DrawSource {
    /// **The default.** `rand_pcg`'s PCG32, seeded through `rand_core`'s `seed_from_u64`
    /// expansion — see [`DrawSource::for_run`] for why that matters.
    Modern(rand_pcg::Pcg32),
    /// The Fortran's `init_random_seed`. Opt-in via `HB_LEGACY_SEEDING`, and retained
    /// only to regenerate results produced before §3.1.
    Legacy(Pcg32),
    /// Validation only — see [`FixtureDraws`]. Opt-in via `HB_FIXTURE_RNG`.
    Fixture(FixtureDraws),
}

impl DrawSource {
    /// Build the run's draw source, and say whether the rupture-time jitter applies.
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
    /// # What replaces it
    ///
    /// `seed_from_u64` runs the seed through an avalanche expansion and fills **both**
    /// the state and the increment from it, so different seeds get different *streams*
    /// rather than different offsets in one. That is what `rand` provides and what the
    /// Fortran never had.
    ///
    /// # The jitter gate
    ///
    /// The Fortran gates rupture-time jitter on the *mutated* seed, `irand + 8 > 0`,
    /// which is an artifact of `init_random_seed` mutating its argument in place. It is a
    /// live footgun: `hf_sim.py` derives per-station seeds as `int32(root) ^ hash(name)`,
    /// so about **half of them are negative** and would silently lose jitter based on the
    /// sign of a name hash. (Dormant in production only because `rupv` defaults to -1, so
    /// the branch is unreachable.) Under modern seeding the jitter is simply on; legacy
    /// reproduces the sign test exactly.
    pub fn for_run(irand: i32) -> (Self, bool) {
        if std::env::var_os("HB_FIXTURE_RNG").is_some() {
            (Self::Fixture(FixtureDraws::seed(irand)), true)
        } else if std::env::var_os("HB_LEGACY_SEEDING").is_some() {
            let (rng, seeded) = Pcg32::seed(irand);
            (Self::Legacy(rng), seeded > 0)
        } else {
            use rand_core::SeedableRng;
            // Sign-extended through i64 so a negative deck seed maps to a distinct u64
            // rather than colliding with its magnitude.
            let g = rand_pcg::Pcg32::seed_from_u64(irand as i64 as u64);
            (Self::Modern(g), true)
        }
    }
}

impl Draws for DrawSource {
    #[inline]
    fn next_f32(&mut self) -> f32 {
        match self {
            // The 24-bit conversion is NOT negotiable and is not fidelity: dividing a
            // full u32 by 2^32 rounds values near 1 up to exactly 1.0, and the
            // zero-rejection loops in `fill_normal_deviates` need `[0, 1)`.
            Self::Modern(g) => {
                use rand_core::Rng;
                (g.next_u32() >> 8) as f32 / 16777216.0
            }
            Self::Legacy(g) => g.next_f32(),
            Self::Fixture(g) => g.next_f32(),
        }
    }
}

/// `subroutine fill_normal_deviates(count,out)` — `hb_high_ref.f:4033`.
///
/// Box-Muller pairs, then the whole vector is rescaled so that `sum(out**2) == count`
/// exactly.
///
/// # What the renormalisation actually buys — the old claim here was wrong
///
/// This comment used to say the rescale was load-bearing because
/// `stochastic_spectrum`'s amplitude calibration is tuned against a unit-RMS sequence.
/// **It is not.** Trace the scale factor `s` through that routine: `a` is proportional to
/// `s`; `remove_quadratic_trend` is linear and homogeneous of degree 1, so its output is
/// too; `ac = a * w` and the forward transform are linear, so `ac ∝ s`; therefore
/// `fsa = sum|ac|^2 ∝ s^2` and `amp = 1/(dt*sqrt(fsa/fold_count)) ∝ 1/s`. The product
/// `ac * as_ * amp` is **proportional to `s^0`**. `amp` is a self-normalisation against
/// the power of the very sequence that was rescaled, so the two cancel exactly.
///
/// The one live consumer is the rupture-velocity perturbation in `sim`, which uses a
/// deviate directly as a standard normal with sigma = `rv_sig1`. There the rescale pins
/// the RMS to exactly 1 where an un-normalised generator would land within
/// `1/sqrt(2N)` ~ 0.14% of it.
///
/// So this is two full passes over the buffer buying a 0.14% correction on the handful of
/// deviates that are read directly. Kept for now because it is also what the draw
/// accounting below describes, but it is not the calibration barrier it was documented as.
///
/// Draw accounting, which the shared stream depends on: `2*ceil(count/2)` draws
/// plus one extra per rejected zero. When `count` is odd the sine partner of the
/// final pair is generated and discarded.
pub fn fill_normal_deviates<R: Draws>(rng: &mut R, count: usize, out: &mut [f32]) {
    // `count` is the DRAW count and stays an explicit argument, deliberately not
    // inferred from `out.len()`. The number of deviates drawn is part of the RNG
    // stream -- every later draw depends on where the generator ended up -- so it is a
    // numerical decision, not a buffer property. Inferring it would mean a future
    // resize of the buffer silently moved every waveform, which is the §2.6b trap in a
    // new costume. Asserted rather than assumed:
    assert!(count <= out.len(), "draw count {count} exceeds buffer {}", out.len());
    // x1 and x2 persist across iterations: the odd-numbered draw computes the
    // pair and returns the cosine component, the even-numbered one returns the
    // sine component from the *same* pair. In the Fortran they are ordinary
    // locals whose values survive between iterations of the DO 8 loop.
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    // The original's computed `goto (1,2),j`.
    let mut j = 1;

    for n in 0..count {
        let w = if j == 1 {
            x1 = rng.next_f32();
            while x1 == 0.0 {
                x1 = rng.next_f32();
            }
            x2 = rng.next_f32();
            while x2 == 0.0 {
                x2 = rng.next_f32();
            }
            x2 = std::f32::consts::TAU * x2;
            x1 = -x1.ln();
            x1 = (x1 + x1).sqrt();
            j = 2;
            x1 * x2.cos()
        } else {
            j = 1;
            x1 * x2.sin()
        };
        out[n] = w;
    }

    let mut s = 0.0f32;
    for i in 0..count {
        s += out[i] * out[i];
    }
    // count is promoted to real*4 for the division, and the sqrt is single
    // precision. Do not compute this in f64.
    s = (count as f32 / s).sqrt();
    for i in 0..count {
        out[i] *= s;
    }
}

/// `subroutine RANU2(NRR,RN)` — `hb_high_ref.f:2428`. Uniform deviates.
pub fn fill_uniform_deviates<R: Draws>(rng: &mut R, count: usize, out: &mut [f32]) {
    // Explicit `count` for the same reason as `fill_normal_deviates`.
    assert!(count <= out.len(), "draw count {count} exceeds buffer {}", out.len());
    for i in 0..count {
        out[i] = rng.next_f32();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_mutates_irand_by_seed_words() {
        // The line-1366 coupling: irand comes back incremented by exactly 8.
        let (_, irand) = Pcg32::seed(0);
        assert_eq!(irand, SEED_WORDS);
        let (_, irand) = Pcg32::seed(-3);
        assert_eq!(irand, 5, "a small negative seed becomes positive, \
                              which flips the line-1366 jitter branch");
    }

    #[test]
    fn rand_numb_stays_in_unit_interval() {
        let (mut g, _) = Pcg32::seed(123456789);
        for _ in 0..100_000 {
            let u = g.next_f32();
            assert!((0.0..1.0).contains(&u), "next_f32 returned {u}");
        }
    }

    #[test]
    fn normal_random_number_has_unit_rms() {
        let (mut g, _) = Pcg32::seed(42);
        for nr in [1usize, 2, 3, 15, 16, 1000] {
            let mut a = vec![0.0f32; nr];
            fill_normal_deviates(&mut g, nr, &mut a);
            let ss: f32 = a.iter().map(|v| v * v).sum();
            // Renormalised so sum of squares == nr, to f32 rounding.
            assert!((ss / nr as f32 - 1.0).abs() < 1e-4,
                    "nr={nr}: sum of squares {ss} != {nr}");
        }
    }

    #[test]
    fn normal_random_number_draw_count() {
        // 2*ceil(nr/2) draws, absent zero rejections. Verified by comparing a
        // fresh generator advanced by hand against one driven through the
        // routine: the stream position must match.
        for nr in [1usize, 2, 3, 4, 7, 8] {
            let (mut a, _) = Pcg32::seed(7);
            let mut acc = vec![0.0f32; nr];
            fill_normal_deviates(&mut a, nr, &mut acc);

            let (mut b, _) = Pcg32::seed(7);
            for _ in 0..2 * nr.div_ceil(2) {
                b.next_f32();
            }
            assert_eq!(a.next_u32(), b.next_u32(),
                       "nr={nr}: stream position diverged");
        }
    }
}
