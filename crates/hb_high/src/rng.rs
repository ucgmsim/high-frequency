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

/// `subroutine fill_normal_deviates(count,out)` — `hb_high_ref.f:4033`.
///
/// Box-Muller pairs, then the whole vector is rescaled so that
/// `sum(out**2) == count` exactly. That renormalisation is **not** cosmetic:
/// `stochastic_spectrum`'s amplitude calibration is tuned against a unit-RMS sequence, so
/// substituting a plain N(0,1) generator changes the output level.
///
/// Draw accounting, which the shared stream depends on: `2*ceil(count/2)` draws
/// plus one extra per rejected zero. When `count` is odd the sine partner of the
/// final pair is generated and discarded.
pub fn fill_normal_deviates(rng: &mut Pcg32, count: usize, out: &mut [f32]) {
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
            x2 = 6.2831853 * x2;
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
pub fn fill_uniform_deviates(rng: &mut Pcg32, count: usize, out: &mut [f32]) {
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
