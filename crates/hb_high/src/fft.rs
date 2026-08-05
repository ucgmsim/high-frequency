//! The complex transform, and the baseline correction that shares its callers.
//!
//! # The transform is `rustfft`, not the vendored radix-2 kernel
//!
//! `REFACTOR.md` §2.1. The transliterated radix-2 `FAST` accounted for 48.5% of
//! runtime in self time, with a further 27.1% in `libm` computing its twiddles — 75.6%
//! of the program between them. It recomputed `CEXP(THETA)` once per `(stage, k)` pair
//! on every call, three `libm` calls each, about 295k per subfault at `np2 = 16384`.
//!
//! `rustfft` caches twiddles in its plan and dispatches to SIMD kernels, so this
//! replaces the algorithm, the twiddle problem and the large-`np2` cache penalty in one
//! change. See `PROFILE.md`, whose items 1, 2 and 5 this supersedes.
//!
//! # Convention
//!
//! Verified empirically rather than taken from the old comment, since a sign error here
//! is silent: feeding `x[k] = exp(+2*pi*i*m*k/n)` to the vendored kernel with `ind = -1`
//! produced its peak at bin `m`, which is the standard forward DFT
//! `X[j] = sum_k x[k] exp(-2*pi*i*j*k/n)`. So `ind = -1` maps to `FftDirection::Forward`
//! and `ind = +1` to `Inverse`, with no conjugation. Neither direction is scaled, which
//! matches both the original and `rustfft`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{azip, ArrayViewMut1};
use rustfft::{Fft, FftDirection, FftPlanner};

// `complex*8` / `complex*16` are `num_complex::Complex`, re-exported here because this is
// the module that owns the transform and hands them to everyone else. They lived in
// `fort.rs` until §5.3, which deleted it.
//
// This was hand-written for as long as bit-identity was the goal: a dependency is free to
// implement `abs` or `exp` differently from gfortran, and nothing produces a compile error
// when it does. Checked rather than assumed before swapping, against the same gfortran
// 16.1.1 vectors the old unit tests pinned:
//
//   norm (was abs)   hypot both sides            IDENTICAL bit for bit
//   exp              exp(re)*(cos im, sin im)    IDENTICAL
//   mul              textbook four-multiply      IDENTICAL
//   div              1-2 ulps different          num-complex does not use gfortran's
//                                                Smith-with-range-reduction branch
//
// So only division moved, at two call sites in `ray.rs`. See `REFACTOR.md` §2.2.
pub use rustfft::num_complex::Complex;

pub type Complex32 = Complex<f32>;
pub type Complex64 = Complex<f64>;

/// A planned transform of one length and direction. `rustfft` hands these out behind an
/// `Arc` because a plan is shareable and immutable once built.
type FftPlan = Arc<dyn Fft<f32>>;

thread_local! {
    /// Plans are cached per `(length, direction)`: building one is where `rustfft`
    /// does its twiddle precomputation, so planning per call would reintroduce
    /// exactly the cost this change removes.
    ///
    /// Thread-local rather than a global mutex because `simulate` is called once per
    /// station and a future caller will want stations on separate threads; a shared
    /// lock would serialise them on the hottest path in the program.
    static PLANS: RefCell<HashMap<(usize, bool), FftPlan>> = RefCell::new(HashMap::new());
}

/// Analysis transform, `e^{-i w t}` — the Fortran's `FAST(NNN, ACE, -1)`.
///
/// Unnormalised; callers divide by the length where they need to.
pub fn forward(data: &mut [Complex32]) {
    transform(data, true)
}

/// Synthesis transform — the Fortran's `FAST(NNN, ACE, +1)`.
///
/// Unnormalised; [`crate::stoc::radiate_and_invert`] divides by `np2`.
pub fn inverse(data: &mut [Complex32]) {
    transform(data, false)
}

/// `SUBROUTINE FAST(NNN,ACE,IND)` — in-place unnormalised complex radix-2 FFT.
///
/// The `ind` argument is gone. It only ever took `-1` or `+1`, the body's sole use of it
/// was `let forward = ind == -1`, and it cost a runtime assertion to enforce a two-valued
/// domain the type system can express for free. The two call sites each have a fixed
/// direction, so this is two named functions rather than an enum — there is no dispatch
/// to preserve. `REFACTOR.md` §1.4b deferred this until §2.1 replaced the kernel, which
/// it has.
///
/// The inverted mapping — `-1` meaning *forward* — was the trap, and it needed the
/// module header's four paragraphs to defend. `forward()` and `inverse()` need none.
///
/// `len` must be a power of two. Every call site satisfies this (`np2` is built by
/// doubling from 2), but the assertion makes a future violation loud rather than
/// silently wrong.
///
/// The vendored kernel's twiddle argument used `3.141593`, a 7-digit truncation of pi
/// about 2 `f32` ulps off, and that value was load-bearing while it computed its own
/// twiddles. §2.1 handed the transform to `rustfft`, which builds correctly rounded
/// twiddles in its plan, so the constant is gone along with the kernel.
fn transform(data: &mut [Complex32], forward: bool) {
    // The length is the slice's, not a separate argument. Every call site passed
    // exactly `data.len()`, so the parameter could only ever have disagreed with
    // reality -- §2.3.
    let len = data.len();
    assert!(len.is_power_of_two(), "FAST requires a power-of-two length, got {len}");

    let plan = PLANS.with(|plans| {
        Arc::clone(plans.borrow_mut().entry((len, forward)).or_insert_with(|| {
            let direction =
                if forward { FftDirection::Forward } else { FftDirection::Inverse };
            FftPlanner::new().plan_fft(len, direction)
        }))
    });

    // Since §2.2 `Complex32` *is* `rustfft`'s element type, so the buffer goes
    // straight in with no conversion and no copy.
    plan.process(data);
}

/// `SUBROUTINE FLZERO(N,DT,A)` — remove the quadratic acceleration trend that
/// leaves final velocity and displacement at zero.
///
/// Note it modifies only `acceleration[2..]` in 0-based terms — the Fortran's
/// `A(3..=N)`. `A(1)` and `A(2)` are left untouched because the correction loop starts
/// at `I=3`. That asymmetry is preserved.
///
/// 0-based since §2.3. The length is the slice's; every caller passed `len()`.
pub fn remove_quadratic_trend(dt: f32, acceleration: &mut [f32]) {
    let mut ve = 0.0f32;
    let mut de = 0.0f32;
    let a1 = dt / 2.0;
    let a2 = a1 * dt / 3.0;
    let count = acceleration.len();
    let nstps = count - 1;

    // Trapezoidal double integration. Inherently serial -- `de` depends on the `ve`
    // from before this iteration's update, and both statements are inside `DO 1`, so
    // their order is load-bearing. What can be improved is the memory traffic: the
    // original loads `acceleration[i]` and `acceleration[i+1]` every iteration, and
    // consecutive iterations overlap by one element. Carrying the previous sample in a
    // register halves the loads.
    //
    // The arithmetic and its order are untouched, so this is bit-exact.
    let mut previous = acceleration[0];
    for &next in &acceleration[1..] {
        de = de + ve * dt + a2 * (2.0 * previous + next);
        ve += a1 * (previous + next);
        previous = next;
    }

    let rnstp = nstps as f32;
    let t = rnstp * dt;
    let c1 = 2.0 / t * (ve - 3.0 / t * de);
    let c2 = 6.0 / t * (2.0 / t * de - ve) / t;

    // The correction is affine in the sample index, so this loop is trivially
    // vectorisable -- but only if the compiler can see a contiguous slice rather than a
    // sequence of bounds-checked 1-based index expressions. Written as a slice iterator
    // for that reason; `a3` keeps the same value it had (`i - 1` for Fortran index `i`,
    // which is `k + 2` here) and the multiply order is unchanged, so this too is
    // bit-exact.
    azip!((index k, sample in ArrayViewMut1::from(&mut acceleration[2..])) {
        let a3 = (k + 2) as f32;
        *sample = *sample + c1 + c2 * a3 * dt;
    });
}
