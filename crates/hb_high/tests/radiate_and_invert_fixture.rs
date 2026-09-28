//! Regression test for issue #8: a `radiate_and_invert` caller that hands over a
//! full-length (`np2`) radiation vector, rather than one sized to the folded axis
//! (`np2/2 + 1`), makes `fold_count` too large and the mirrored slice runs off the
//! end of `spectrum`.

use hb_high::fft::Complex32;
use hb_high::rng::{Draws, LegacyPcg};
use hb_high::stoc::radiate_and_invert;
use ndarray::Array1;

/// Complex spectrum of plausible magnitude, deterministic so runs are comparable.
fn spectrum(np2: usize) -> Vec<Complex32> {
    let mut g = LegacyPcg::seed(20260804);
    (0..np2)
        .map(|_| Complex32::new(g.uniform() - 0.5, g.uniform() - 0.5))
        .collect()
}

/// The radiation pattern covers only the folded frequencies (`np2/2 + 1`): `fold_count`
/// is derived from its length, so a properly folded vector must not panic.
#[test]
fn radiation_sized_to_the_fold_does_not_run_off_the_spectrum() {
    let np2 = 4096usize;
    let nf = np2 / 2 + 1;
    let radiation = Array1::from(vec![0.7f32; nf]);
    let mut spec = Array1::from(spectrum(np2));
    let mut time_series: Array1<f32> = Array1::zeros(np2);
    radiate_and_invert(spec.view_mut(), radiation.view(), time_series.view_mut());
    assert!(time_series.iter().all(|v| v.is_finite()));
}
