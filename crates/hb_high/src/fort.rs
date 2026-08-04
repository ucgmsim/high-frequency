//! Fortran compatibility layer: 1-based arrays, complex arithmetic, and shims
//! for the intrinsics whose semantics differ from the obvious Rust equivalent.
//!
//! Everything here exists to make transliteration mechanical. See
//! `PORTING_RULES.md` §3 and §4.

use std::ops::{Add, Div, Index, IndexMut, Mul, Neg, Sub};

// ---------------------------------------------------------------------------
// 1-based arrays
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

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Underlying storage, 0-based. For I/O and tests only — never for indexing
    /// inside a transliterated routine.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
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

/// A 1-based two-dimensional array stored **column-major**, as Fortran does.
///
/// The layout is not an implementation detail: the original reads `stdd(0,l)`
/// (`PORTING_RULES.md` §7), and which element that aliases depends on
/// column-major ordering with the declared leading dimension.
#[derive(Clone, Debug, PartialEq)]
pub struct Array2<T> {
    data: Vec<T>,
    rows: usize,
}

impl<T: Copy + Default> Array2<T> {
    /// `dimension x(rows, cols)` — valid indices are `1..=rows`, `1..=cols`.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { data: vec![T::default(); rows * cols], rows }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.data.len() / self.rows
    }

    /// Column-major backing store. For I/O and tests only.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    pub fn fill(&mut self, value: T) {
        self.data.fill(value);
    }
}

impl<T> Index<(usize, usize)> for Array2<T> {
    type Output = T;
    #[track_caller]
    fn index(&self, (i, j): (usize, usize)) -> &T {
        assert!(i >= 1 && j >= 1, "Fortran index ({i},{j}) is below 1");
        assert!(i <= self.rows, "row {i} exceeds leading dimension {}", self.rows);
        &self.data[(j - 1) * self.rows + (i - 1)]
    }
}

impl<T> IndexMut<(usize, usize)> for Array2<T> {
    #[track_caller]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut T {
        assert!(i >= 1 && j >= 1, "Fortran index ({i},{j}) is below 1");
        assert!(i <= self.rows, "row {i} exceeds leading dimension {}", self.rows);
        let rows = self.rows;
        &mut self.data[(j - 1) * rows + (i - 1)]
    }
}

// ---------------------------------------------------------------------------
// Complex arithmetic
// ---------------------------------------------------------------------------

/// `complex*8` / `complex*16`, hand-written rather than taken from a crate.
///
/// A dependency would be free to implement `abs` or `exp` differently from
/// gfortran, and there would be no compile error when it did — just a few
/// wrong bits. See `Cargo.toml`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

impl<T> Complex<T> {
    pub const fn new(re: T, im: T) -> Self {
        Self { re, im }
    }
}

macro_rules! impl_complex {
    ($t:ty) => {
        impl Complex<$t> {
            pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

            /// `conjg(z)`
            pub fn conj(self) -> Self {
                Self { re: self.re, im: -self.im }
            }

            /// `cabs(z)` — gfortran computes this as a hypot, which is *not*
            /// the same as `(re*re + im*im).sqrt()` in the last bits.
            pub fn abs(self) -> $t {
                self.re.hypot(self.im)
            }

            /// `cexp(z)` = `exp(re) * (cos(im) + i sin(im))`
            pub fn exp(self) -> Self {
                let r = self.re.exp();
                Self { re: r * self.im.cos(), im: r * self.im.sin() }
            }
        }

        impl Add for Complex<$t> {
            type Output = Self;
            fn add(self, o: Self) -> Self {
                Self { re: self.re + o.re, im: self.im + o.im }
            }
        }

        impl Sub for Complex<$t> {
            type Output = Self;
            fn sub(self, o: Self) -> Self {
                Self { re: self.re - o.re, im: self.im - o.im }
            }
        }

        impl Neg for Complex<$t> {
            type Output = Self;
            fn neg(self) -> Self {
                Self { re: -self.re, im: -self.im }
            }
        }

        impl Mul for Complex<$t> {
            type Output = Self;
            /// Textbook four-multiply form, matching what gfortran emits for
            /// `complex` multiply at `-O0` without `-fcx-limited-range`.
            fn mul(self, o: Self) -> Self {
                Self {
                    re: self.re * o.re - self.im * o.im,
                    im: self.re * o.im + self.im * o.re,
                }
            }
        }

        impl Mul<$t> for Complex<$t> {
            type Output = Self;
            fn mul(self, s: $t) -> Self {
                Self { re: self.re * s, im: self.im * s }
            }
        }

        impl Div<$t> for Complex<$t> {
            type Output = Self;
            fn div(self, s: $t) -> Self {
                Self { re: self.re / s, im: self.im / s }
            }
        }
    };
}

impl_complex!(f32);
impl_complex!(f64);

// ---------------------------------------------------------------------------
// Intrinsic shims
// ---------------------------------------------------------------------------

/// `nint(x)` — round half **away from zero**.
///
/// Not `round_ties_even` (which rounds half to even) and not a bare `as i32`
/// (which truncates).
pub fn nint(x: f32) -> i32 {
    x.round() as i32
}

pub fn nint64(x: f64) -> i32 {
    x.round() as i32
}

/// `int(x)` — truncate **toward zero**, so `int(-1.7) == -1`.
///
/// `k2` at line 1371 of the original depends on this and can come out negative;
/// see `PORTING_RULES.md` §7.
pub fn int_trunc(x: f32) -> i32 {
    x.trunc() as i32
}

pub fn int_trunc64(x: f64) -> i32 {
    x.trunc() as i32
}

/// `sign(a,b)` — magnitude of `a` with the sign of `b`. Not `signum`.
pub fn sign(a: f32, b: f32) -> f32 {
    if b >= 0.0 { a.abs() } else { -a.abs() }
}

/// `mod(a,b)` for integers — takes the sign of `a`, same as Rust `%`.
/// Present so call sites read like the Fortran rather than needing a comment.
pub fn imod(a: i32, b: i32) -> i32 {
    a % b
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
    fn array2_is_column_major() {
        // dimension x(2,3): x(1,1) x(2,1) x(1,2) ... in memory
        let mut a = Array2::<f32>::new(2, 3);
        a[(1, 1)] = 11.0;
        a[(2, 1)] = 21.0;
        a[(1, 2)] = 12.0;
        assert_eq!(a.as_slice()[0], 11.0);
        assert_eq!(a.as_slice()[1], 21.0);
        assert_eq!(a.as_slice()[2], 12.0);
        assert_eq!(a.cols(), 3);
    }

    #[test]
    fn nint_rounds_half_away_from_zero() {
        assert_eq!(nint(0.5), 1);
        assert_eq!(nint(1.5), 2);
        assert_eq!(nint(2.5), 3); // round_ties_even would give 2
        assert_eq!(nint(-0.5), -1);
        assert_eq!(nint(-2.5), -3);
    }

    #[test]
    fn int_truncates_toward_zero() {
        assert_eq!(int_trunc(1.7), 1);
        assert_eq!(int_trunc(-1.7), -1); // toward zero, not floor
        assert_eq!(int_trunc(-0.2), 0);
    }

    #[test]
    fn sign_takes_magnitude_of_a_and_sign_of_b() {
        assert_eq!(sign(3.0, -1.0), -3.0);
        assert_eq!(sign(-3.0, 1.0), 3.0);
        assert_eq!(sign(-3.0, 0.0), 3.0); // +0 counts as positive
    }

    #[test]
    fn complex_conj_and_abs() {
        let z = Complex32::new(3.0, -4.0);
        assert_eq!(z.conj(), Complex32::new(3.0, 4.0));
        assert_eq!(z.abs(), 5.0);
    }
}
