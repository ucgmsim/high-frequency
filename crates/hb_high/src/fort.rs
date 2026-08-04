//! Fortran compatibility layer: complex arithmetic, and shims for the intrinsics whose
//! semantics differ from the obvious Rust equivalent.
//!
//! Everything here exists to make transliteration mechanical. See `PORTING_RULES.md`
//! §3 and §4.
//!
//! **This module is nearly gone, which was the point of §2.3.** It began as ~230 lines
//! of 1-based arrays, hand-written complex arithmetic and rounding shims. `Complex` is
//! now a re-export rather than an implementation (§2.2); `Array2` and `Array1` are
//! deleted outright, the whole crate having moved to 0-based storage. What remains is
//! the two rounding intrinsics, which are genuine semantic differences from Rust rather
//! than transliteration scaffolding: Fortran's `NINT` rounds half away from zero where
//! `f32::round_ties_even` does not, and its `INT` truncates toward zero where a bare
//! `as` cast is only incidentally the same.

// ---------------------------------------------------------------------------
// Complex arithmetic
// ---------------------------------------------------------------------------

// `complex*8` / `complex*16` are `num_complex::Complex`, re-exported so call sites
// read the same as before.
//
// This was hand-written for as long as bit-identity was the goal: a dependency is
// free to implement `abs` or `exp` differently from gfortran, and nothing produces a
// compile error when it does. Checked rather than assumed before swapping, against
// the same gfortran 16.1.1 vectors the old unit tests pinned:
//
//   norm (was abs)   hypot both sides            IDENTICAL bit for bit
//   exp              exp(re)*(cos im, sin im)    IDENTICAL
//   mul              textbook four-multiply      IDENTICAL
//   div              1-2 ulps different          num-complex does not use
//                                                gfortran's Smith-with-range-reduction
//                                                branch
//
// So only division moved, at two call sites in `ray.rs`. See `REFACTOR.md` §2.2.
pub use rustfft::num_complex::Complex;

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

// ---------------------------------------------------------------------------
// Intrinsic shims
// ---------------------------------------------------------------------------

/// `round_half_away_from_zero(x)` — round half **away from zero**.
///
/// Not `round_ties_even` (which rounds half to even) and not a bare `as i32`
/// (which truncates).
pub fn round_half_away_from_zero(x: f32) -> i32 {
    x.round() as i32
}

/// `int(x)` — truncate **toward zero**, so `int(-1.7) == -1`.
///
/// `k2` at line 1371 of the original depends on this and can come out negative;
/// see `PORTING_RULES.md` §7.
pub fn truncate_toward_zero(x: f32) -> i32 {
    x.trunc() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nint_rounds_half_away_from_zero() {
        assert_eq!(round_half_away_from_zero(0.5), 1);
        assert_eq!(round_half_away_from_zero(1.5), 2);
        assert_eq!(round_half_away_from_zero(2.5), 3); // round_ties_even would give 2
        assert_eq!(round_half_away_from_zero(-0.5), -1);
        assert_eq!(round_half_away_from_zero(-2.5), -3);
    }

    #[test]
    fn int_truncates_toward_zero() {
        assert_eq!(truncate_toward_zero(1.7), 1);
        assert_eq!(truncate_toward_zero(-1.7), -1); // toward zero, not floor
        assert_eq!(truncate_toward_zero(-0.2), 0);
    }

    
    }
