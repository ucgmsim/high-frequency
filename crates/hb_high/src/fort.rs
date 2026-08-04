//! Fortran compatibility layer: the 1-based array, complex arithmetic, and shims for
//! the intrinsics whose semantics differ from the obvious Rust equivalent.
//!
//! Everything here exists to make transliteration mechanical. See `PORTING_RULES.md`
//! §3 and §4.
//!
//! **Shrinking as Stage 2 proceeds.** `Complex` is now a re-export rather than an
//! implementation (§2.2), the column-major `Array2` is gone entirely (§2.3), and the
//! remaining `Array1` survives only in `state.rs`'s velocity model, whose conversion
//! needs sign-off because it touches every kernel that reads a layer.

use std::ops::{Index, IndexMut};

// ---------------------------------------------------------------------------
// The 1-based array
// ---------------------------------------------------------------------------

/// A 1-based one-dimensional array, indexed exactly as the Fortran indexes it.
///
/// Index arithmetic must not be rewritten during transliteration, so this type
/// takes the Fortran index directly. Out-of-range access panics with the
/// offending index, which is how the known out-of-bounds reads in the original
/// (see `PORTING_RULES.md` §7) surface instead of silently reading adjacent
/// storage.
#[derive(Clone, Debug, PartialEq)]
pub struct Array1<T> {
    data: Vec<T>,
}

impl<T: Copy + Default> Array1<T> {
    /// `dimension x(n)` — valid indices are `1..=n`.
    pub fn new(n: usize) -> Self {
        Self { data: vec![T::default(); n] }
    }

    pub fn filled(n: usize, value: T) -> Self {
        Self { data: vec![value; n] }
    }

    /// Underlying storage, 0-based. For I/O and tests only — never for indexing
    /// inside a transliterated routine.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable underlying storage, 0-based.
    ///
    /// Exists for `fft::fast`, which passes its buffer straight to `rustfft` now that
    /// the element type is `num_complex::Complex` on both sides. Not for indexing
    /// inside a transliterated routine.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Index<usize> for Array1<T> {
    type Output = T;
    #[track_caller]
    fn index(&self, i: usize) -> &T {
        assert!(i >= 1, "Fortran index {i} is below 1");
        &self.data[i - 1]
    }
}

impl<T> IndexMut<usize> for Array1<T> {
    #[track_caller]
    fn index_mut(&mut self, i: usize) -> &mut T {
        assert!(i >= 1, "Fortran index {i} is below 1");
        &mut self.data[i - 1]
    }
}

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
    fn array1_is_one_based() {
        let mut a = Array1::<f32>::new(3);
        a[1] = 1.0;
        a[3] = 3.0;
        assert_eq!(a[1], 1.0);
        assert_eq!(a[2], 0.0);
        assert_eq!(a[3], 3.0);
        assert_eq!(a.as_slice(), &[1.0, 0.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "below 1")]
    fn array1_rejects_index_zero() {
        let a = Array1::<f32>::new(3);
        let _ = a[0];
    }

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
