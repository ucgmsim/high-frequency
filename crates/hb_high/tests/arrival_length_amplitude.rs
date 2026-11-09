//! Regression test for issue #5: a transform shortened by
//! `SpectrumPlan::length_for_arrival` cuts the Saragoni–Hart envelope off early, and the
//! power normalisation in `stochastic_spectrum` used to put the subfault's full spectral
//! energy into the samples that were computed. The landed samples came out inflated by
//! `1/√S`, where `S` is the share of the envelope's energy inside the transform.

use hb_high::fft::Complex32;
use hb_high::rng::{Draws, LegacyPcg};
use hb_high::stoc::{
    RayPath, SourceModel, SpectrumPlan, SpectrumShape, radiate_and_invert, stochastic_spectrum,
};
use ndarray::Array1;

const DT: f32 = 0.005;
const WINDOW_S: f32 = 20.0;
const LANDED: usize = 200;
const SEEDS: i32 = 100;

/// The mean square of the first `LANDED` output samples, averaged over `SEEDS` seeds.
fn landed_mean_square(np2: usize) -> f64 {
    let model = SourceModel {
        dt: DT,
        window_eps: 0.2,
        window_eta: 0.05,
        kappa_s: 0.045,
    };
    let path = RayPath {
        distance_km: 50.0,
        window_s: WINDOW_S,
        shear_velocity_km_s: 3.5,
        density_g_cm3: 2.7,
        corner_frequency_hz: 1.0,
        subevent_moment: 1e22,
        moment_scale: 1.0,
        fmax_hz: 10.0,
        qbar: 0.01,
    };
    let plan = SpectrumPlan::new(np2, DT, 0.6, model.window_eps, model.window_eta);
    let mut shape = SpectrumShape::with_capacity(np2);
    shape.refresh(&plan, &model, &path);
    // As the production path does for a transform `length_for_arrival` shortened.
    if np2 < SpectrumPlan::length_for(WINDOW_S, DT) {
        shape.land_only_the_transforms_share(&plan);
    }
    let radiation = Array1::from(vec![1.0f32; plan.fold_count]);

    let mut total = 0.0f64;
    for seed in 0..SEEDS {
        let mut rng = LegacyPcg::seed(-(1000 + seed));
        let _ = rng.uniform();
        let mut spectrum: Array1<Complex32> = Array1::zeros(np2);
        let mut series: Array1<f32> = Array1::zeros(np2);
        stochastic_spectrum(&mut rng, &plan, &mut shape, spectrum.view_mut());
        radiate_and_invert(spectrum.view_mut(), radiation.view(), series.view_mut());
        total += series
            .iter()
            .take(LANDED)
            .map(|&v| (v as f64) * (v as f64))
            .sum::<f64>();
    }
    total / (SEEDS as usize * LANDED) as f64
}

/// The landed samples' RMS must not depend on whether the transform was shortened.
#[test]
fn a_shortened_transform_does_not_inflate_the_landed_samples() {
    let np2_full = SpectrumPlan::length_for(WINDOW_S, DT);
    // A contribution that starts `LANDED` samples before the record end.
    let np2_short = SpectrumPlan::length_for_arrival(WINDOW_S, DT, 1, LANDED);
    assert!(np2_short < np2_full, "{np2_short} vs {np2_full}");

    let full = landed_mean_square(np2_full).sqrt();
    let short = landed_mean_square(np2_short).sqrt();
    let ratio = short / full;
    // Before the fix this ratio is about 1/√S ≈ 4. The coarser frequency axis of the short
    // transform samples the amplitude spectrum differently, so allow some slack.
    assert!(
        (0.8..1.25).contains(&ratio),
        "landed RMS: shortened {short:e}, full {full:e}, ratio {ratio}"
    );
}
