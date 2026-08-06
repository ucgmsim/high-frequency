//! The complex transform, and the baseline correction that shares its callers.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use ndarray::{ArrayViewMut1, azip};
use rustfft::{Fft, FftDirection, FftPlanner};

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

/// Analysis transform.
///
/// Unnormalised; callers divide by the length where they need to.
pub fn forward(data: &mut [Complex32]) {
    transform(data, true)
}

/// Synthesis transform.
///
/// Unnormalised; [`crate::stoc::radiate_and_invert`] divides by `np2`.
pub fn inverse(data: &mut [Complex32]) {
    transform(data, false)
}

/// in-place unnormalised complex radix-2 FFT.
fn transform(data: &mut [Complex32], forward: bool) {
    // The length is the slice's, not a separate argument. Every call site passed
    // exactly `data.len()`, so the parameter could only ever have disagreed with
    // reality -- §2.3.
    let len = data.len();
    assert!(
        len.is_power_of_two(),
        "FAST requires a power-of-two length, got {len}"
    );

    let plan = PLANS.with(|plans| {
        Arc::clone(plans.borrow_mut().entry((len, forward)).or_insert_with(|| {
            let direction = if forward {
                FftDirection::Forward
            } else {
                FftDirection::Inverse
            };
            FftPlanner::new().plan_fft(len, direction)
        }))
    });

    plan.process(data);
}

pub fn remove_quadratic_trend(dt: f32, acceleration: &mut [f32]) {
    let mut ve = 0.0f32;
    let mut de = 0.0f32;
    let a1 = dt / 2.0;
    let a2 = a1 * dt / 3.0;
    let count = acceleration.len();
    let nstps = count - 1;

    // Trapezoidal double integration.
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

    azip!((index k, sample in ArrayViewMut1::from(&mut acceleration[2..])) {
        let a3 = (k + 2) as f32;
        *sample = *sample + c1 + c2 * a3 * dt;
    });
}
