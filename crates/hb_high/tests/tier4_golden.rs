//! Golden tests for `stochastic_spectrum` and `green_function`, against fixtures produced
//! by the original Fortran. Fixture filenames are the Fortran routine names (`stoc_f.bin`,
//! `gf_amp_tt.bin`).

use hb_high::fft::Complex32;
use hb_high::ray::{RayShape, RayState, WaveMode, green_function};
use hb_high::rng::{Draws, LegacyPcg};
use hb_high::spectrum::{
    SourceModel, SpectrumInputs, SpectrumPlan, SpectrumShape, WindowShape, stochastic_spectrum,
};
use hb_high::velocity::VelocityModel;
use ndarray::Array1;

mod common;
use common::*;

#[test]
fn stoc_f_matches_fortran() {
    let mut r = Golden::open("tier4", "stoc_f.bin");
    let mut cases = 0;
    let mut saw_negative_kappa = false;

    while !r.done() {
        let np2 = r.usize();
        let seed = r.i32();
        let nf = np2 / 2 + 1;

        let (rr, tw, eps, eta, betvs) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (row, dt, smt, dlm, fc) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (fmx, akapp, qb, qfe, bigc) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        if akapp <= 0.0 {
            saw_negative_kappa = true;
        }

        let dfr: Vec<f32> = (0..nf).map(|_| r.f32()).collect();
        let want: Vec<Complex32> = (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want_after: Vec<f32> = (0..8).map(|_| r.f32()).collect();

        let mut rng = LegacyPcg::seed(seed);
        // Built by struct literal rather than `SpectrumPlan::new`: the golden records `np2`
        // directly where `new` derives it from a window length, and this keeps the test
        // about the arithmetic rather than the caching.
        let window = WindowShape {
            peak_fraction: eps,
            end_fraction: eta,
        };
        let b = window.exponent();
        let plan = SpectrumPlan {
            np2,
            fold_count: nf,
            log_frequency_hz: dfr.iter().map(|f| f.ln()).collect(),
            path_exponent: dfr.iter().map(|f| f.powf(1.0 - qfe)).collect(),
            envelope_power: (0..np2).map(|i| (i as f32 * dt).powf(b)).collect(),
            frequency_hz: dfr.clone().into(),
            // Only `radiate_and_invert` reads it, and this test stops before that.
            taper: Array1::zeros(0),
        };
        // Recorded in the golden but unused by the computation.
        let _ = dlm;
        let mut cw: Array1<Complex32> = Array1::zeros(np2);
        // `SpectrumShape::refresh` computes the deterministic shape; `stochastic_spectrum`
        // applies the random draws to it.
        let mut shape = SpectrumShape::with_capacity(np2);
        shape.refresh(
            &plan,
            &SourceModel {
                dt,
                window,
                subevent_moment: smt,
                kappa_s: akapp,
                moment_scale: bigc,
            },
            &SpectrumInputs {
                distance_km: rr,
                window_s: tw,
                shear_velocity_km_s: betvs,
                density_g_cm3: row,
                corner_frequency_hz: fc,
                fmax_hz: fmx,
                qbar: qb,
            },
        );
        stochastic_spectrum(&mut rng, &plan, &mut shape, cw.view_mut());

        let tag = format!("stochastic_spectrum case {cases} (np2={np2} akapp={akapp})");
        // Scale from the golden, so the tolerance does not float with our own output.
        let scale = want
            .iter()
            .fold(0.0f32, |a, c| a.max(c.re.abs()).max(c.im.abs()));
        for i in 0..np2 {
            let at = i + 1;
            near32(&format!("{tag} cw[{at}].re"), cw[i].re, want[i].re, scale);
            near32(&format!("{tag} cw[{at}].im"), cw[i].im, want[i].im, scale);
        }
        // Generator position: stochastic_spectrum consumes np2 deviates via
        // normal_deviates, and the shared stream must stay in step.
        for (k, w) in want_after.iter().enumerate() {
            eq32(
                &format!("{tag} post-call draw {k} (generator position)"),
                rng.uniform(),
                *w,
            );
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
    assert!(
        saw_negative_kappa,
        "no case had akapp <= 0, so the alternative high-cut branch \
             ((1+omg/omgm)**(-1.0) rather than exp(-pi*f*kappa)) is untested"
    );
}

#[test]
fn gf_amp_tt_matches_fortran() {
    let mut r = Golden::open("tier4", "gf_amp_tt.bin");
    let mut cases = 0;
    // Which ray topologies the corpus actually reached.
    let mut itypes = std::collections::BTreeSet::new();
    let mut mds = std::collections::BTreeSet::new();

    while !r.done() {
        let j0 = r.usize();
        let itype = r.i32();
        let md = r.i32();
        itypes.insert(itype);
        mds.insert(md);
        let src_depth = r.f32();
        let range = r.f32();

        let mut vmod: VelocityModel = vec![hb_high::velocity::Layer::default(); j0];
        for layer in vmod.iter_mut() {
            layer.thickness_km = r.f64();
        }
        for layer in vmod.iter_mut() {
            layer.vp_km_s = r.f64();
        }
        for layer in vmod.iter_mut() {
            layer.vsh_km_s = r.f64();
        }
        for layer in vmod.iter_mut() {
            layer.attenuation_s = r.f32();
        }

        let want_nd = r.usize();
        let want_nh: Vec<i32> = (0..want_nd).map(|_| r.i32()).collect();
        let want_nm: Vec<i32> = (0..want_nd).map(|_| r.i32()).collect();
        let want_ndeep = r.i32();
        let want_love = r.i32();
        let (w_rp0, w_stime, w_rpath, w_qbar) = (r.f32(), r.f32(), r.f32(), r.f32());

        let mut st = RayState::default();
        let g = green_function(
            &mut st,
            &vmod,
            src_depth,
            range,
            RayShape::from_code(itype).expect("the corpus holds only traced ray types"),
            wave_mode_from_code(md),
        );

        let tag = format!(
            "green_function case {cases} (itype={itype} md={md} \
                           depth={src_depth} range={range})"
        );
        // The ray description itself, before the derived quantities: a wrong
        // segment list would otherwise only show up as a wrong travel time.
        assert_eq!(st.rays.segment_count(), want_nd, "{tag} nd");
        for k in 0..want_nd {
            // 0-based layer index against the golden's 1-based layer number.
            assert_eq!(
                st.rays.layer_indices[k] as i32 + 1,
                want_nh[k],
                "{tag} nh[{k}]"
            );
            assert_eq!(
                wave_mode_code(st.rays.wave_modes[k]),
                want_nm[k],
                "{tag} nm[{k}]"
            );
        }
        assert_eq!(
            st.travel.deepest_layer as i32 + 1,
            want_ndeep,
            "{tag} ndeep"
        );
        // The Fortran's `love`: 2 when the ray starts as SH, 1 for P-SV.
        let love = if st.rays.wave_modes[0] == WaveMode::Sh {
            2
        } else {
            1
        };
        assert_eq!(love, want_love, "{tag} love");

        eq32(&format!("{tag} rp0"), g.ray_parameter_s_per_km, w_rp0);
        eq32(&format!("{tag} stime"), g.travel_time_s, w_stime);
        eq32(&format!("{tag} rpath"), g.path_length_km, w_rpath);
        eq32(&format!("{tag} qbar"), g.qbar, w_qbar);
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 48);
    // Production only ever passes itype=1 and md=4; the corpus deliberately
    // covers the down-going and Moho-multiple topologies too.
    assert!(
        itypes.contains(&1) && itypes.contains(&2),
        "need both upgoing and down-going ray types, saw {itypes:?}"
    );
    assert!(
        itypes.iter().any(|&t| t > 2),
        "no itype > 2, so the Moho-multiple loops are untested; saw {itypes:?}"
    );
    assert!(mds.len() >= 2, "only one wave mode exercised: {mds:?}");
}
