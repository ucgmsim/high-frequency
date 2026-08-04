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

use rustfft::{Fft, FftDirection, FftPlanner};

use crate::fort::{Array1, Complex32};

thread_local! {
    /// Plans are cached per `(length, direction)`: building one is where `rustfft`
    /// does its twiddle precomputation, so planning per call would reintroduce
    /// exactly the cost this change removes.
    ///
    /// Thread-local rather than a global mutex because `simulate` is called once per
    /// station and a future caller will want stations on separate threads; a shared
    /// lock would serialise them on the hottest path in the program.
    static PLANS: RefCell<HashMap<(usize, bool), Arc<dyn Fft<f32>>>> =
        RefCell::new(HashMap::new());
}

/// `SUBROUTINE FAST(NNN,ACE,IND)` — in-place unnormalised complex radix-2 FFT.
///
/// `ind = -1` is the analysis transform (`e^{-i w t}`), `ind = +1` synthesis.
/// Neither direction is scaled; callers divide by `NNN` where they need to.
///
/// `nnn` must be a power of two. Every call site satisfies this — `np2` is built
/// by doubling from 2 at `:1089-1092` — but the assertion makes a future
/// violation loud instead of silently producing garbage, since the bit-reversal
/// permutation below is only a permutation for powers of two.
///
/// The `3.141593` in the twiddle argument is a 7-digit truncation of pi, about
/// 2 `f32` ulps off. It is load-bearing: substituting a more accurate value
/// changes the last bits of every transform. See `PORTING_RULES.md` §1.
pub fn fast(nnn: usize, ace: &mut Array1<Complex32>, ind: i32) {
    assert!(nnn.is_power_of_two(), "FAST requires a power-of-two length, got {nnn}");
    assert!(ind == 1 || ind == -1, "FAST direction must be +/-1, got {ind}");
    let forward = ind == -1;

    let plan = PLANS.with(|plans| {
        Arc::clone(plans.borrow_mut().entry((nnn, forward)).or_insert_with(|| {
            let direction =
                if forward { FftDirection::Forward } else { FftDirection::Inverse };
            FftPlanner::new().plan_fft(nnn, direction)
        }))
    });

    // No conversion: since §2.2, `Complex32` *is* `rustfft`'s element type, so the
    // buffer goes straight in.
    plan.process(&mut ace.as_mut_slice()[..nnn]);
}

/// `SUBROUTINE FLZERO(N,DT,A)` — remove the quadratic acceleration trend that
/// leaves final velocity and displacement at zero.
///
/// Note it modifies only `acceleration(3..=count)`: `acceleration(1)` and `acceleration(2)` are left untouched
/// because the correction loop starts at `I=3`. That asymmetry is preserved.
pub fn remove_quadratic_trend(count: usize, dt: f32, acceleration: &mut Array1<f32>) {
    let mut ve = 0.0f32;
    let mut de = 0.0f32;
    let a1 = dt / 2.0;
    let a2 = a1 * dt / 3.0;
    let nstps = count - 1;

    for i in 1..=nstps {
        // DE uses the value of VE from *before* this iteration's update; the
        // two statements are both inside DO 1 and their order matters.
        de = de + ve * dt + a2 * (2.0 * acceleration[i] + acceleration[i + 1]);
        ve = ve + a1 * (acceleration[i] + acceleration[i + 1]);
    }

    let rnstp = nstps as f32;
    let t = rnstp * dt;
    let c1 = 2.0 / t * (ve - 3.0 / t * de);
    let c2 = 6.0 / t * (2.0 / t * de - ve) / t;

    for i in 3..=count {
        let a3 = (i - 1) as f32;
        acceleration[i] = acceleration[i] + c1 + c2 * a3 * dt;
    }
}
