//! Bit-identity gate for the tier-1 kernels.
//!
//! These routines communicate through common blocks, so each golden record
//! carries the input state, the arguments, **and** the emitted state — the
//! golden is a specification of the whole seam, not just of return values.
//!
//! Regenerate with `harness/kernels/gen_tier1_golden.sh`.
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.

use hb_high::geom::subfault_geometry;
use hb_high::ray::{geometric_spreading, build_ray_path, Takeoff};
use hb_high::site::site_amplification_factors;
use hb_high::state::{RayState, VelocityModel};

mod common;
use common::*;
use hb_high::state::WaveMode;

/// `dump_vmod`: `thickness_km`, `vsh_km_s`, `density_g_cm3` as `f64` for `j0` layers.
///
/// Local to tier 1 rather than shared: each driver dumps a different field set in a
/// different order, and that order is a property of the Fortran `write` statement, not
/// something a caller should be selecting. See `common`'s note on why there is no
/// `vmod(count, fields)`.
fn read_vmod(r: &mut Golden, j0: usize) -> VelocityModel {
    let mut v = VelocityModel::new();
    // 0-based since §2.3; the golden's dump order is the Fortran's layer 1..j0.
    for k in 0..j0 { v[k].thickness_km = r.f64(); }
    for k in 0..j0 { v[k].vsh_km_s = r.f64(); }
    for k in 0..j0 { v[k].density_g_cm3 = r.f64(); }
    v
}


#[test]
fn get_sitefacs_matches_fortran() {
    let mut r = Golden::open("tier1", "get_sitefacs.bin");
    let mut cases = 0;
    while !r.done() {
        let j0 = r.usize();
        let nfreq = r.usize();
        let vmod = read_vmod(&mut r, j0);
        let fn_: Vec<f32> = (0..nfreq).map(|_| r.f32()).collect();
        let want: Vec<f32> = (0..nfreq).map(|_| r.f32()).collect();

        let mut an = vec![0.0f32; nfreq];
        // The golden's `j0` is the Fortran's SOURCE layer number, 1-based; the argument is
        // now a 0-based layer index. Everything the routine reads from `vmod` is keyed off
        // it, so an unshifted value here reads the layer below and the amplification is
        // silently wrong rather than out of range -- which is how this test caught the
        // conversion.
        site_amplification_factors(&vmod, j0 - 1, nfreq, &fn_, &mut an);
        for k in 0..nfreq {
            eq32(&format!("site_amplification_factors j0={j0} an[{k}]"), an[k], want[k]);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}

#[test]
fn trav_matches_fortran() {
    let mut r = Golden::open("tier1", "trav.bin");

    // ONE state across every case, mirroring the Fortran's persistent common
    // block. This is deliberate: `build_ray_path` zeroes only alp(1:100) of 500, so
    // whether higher indices carry values from a previous ray is part of the
    // behaviour under test. A fresh state per case would not exercise it.
    let mut st = RayState::default();

    let mut cases = 0;
    while !r.done() {
        let j0 = r.usize();
        let n = r.usize();
        // The driver still writes `ir` into every record and always writes 1. The port's
        // signatures no longer take it (§2.8, per PORTING_RULES §6), but the field has
        // to be consumed to keep the reader aligned with the fixture.
        let ir = r.usize();
        assert_eq!(ir, 1, "the golden's degenerate ray index should always be 1");
        let ndeg = r.i32();
        let hs = r.f64();
        let hr = r.f64();
        let vmod = read_vmod(&mut r, j0);

        // `nh` holds LAYER indices, and the golden's are the Fortran's 1-based layer
        // numbers, so they shift as well as the segment index they are stored under.
        for k in 0..n {
            st.rays.nh[k] = r.i32() - 1;
        }
        for k in 0..n {
            st.rays.nm[k] = WaveMode::from_fortran(r.i32());
        }
        st.rays.nd = n as i32;
        st.rays.ndeg = ndeg;

        let (w_love, w_nup, w_ndeep) = (r.i32(), r.i32(), r.i32());
        let w_it: Vec<i32> = (0..n).map(|_| r.i32()).collect();
        let w_nup1: Vec<i32> = (0..n).map(|_| r.i32()).collect();
        let w_alp: Vec<f32> = (0..j0).map(|_| r.f32()).collect();
        let w_als: Vec<f32> = (0..j0).map(|_| r.f32()).collect();

        build_ray_path(&mut st, &vmod, hs, hr);

        let tag = format!("build_ray_path case {cases} (j0={j0} n={n} ndeg={ndeg})");
        assert_eq!(st.love, w_love, "{tag} love");
        assert_eq!(st.travel.nup.as_fortran(), w_nup, "{tag} nup");
        // A 0-based layer index here against a 1-based layer number in the golden.
        assert_eq!(st.travel.ndeep + 1, w_ndeep, "{tag} ndeep");
        for k in 0..n {
            assert_eq!(st.coff.it[k].as_fortran(), w_it[k], "{tag} it[{k}]");
            assert_eq!(st.coff.nup1[k].as_fortran(), w_nup1[k], "{tag} nup1[{k}]");
        }
        for k in 0..j0 {
            eq32(&format!("{tag} alp[{k}]"), st.travel.alp[k], w_alp[k]);
            eq32(&format!("{tag} als[{k}]"), st.travel.als[k], w_als[k]);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 8);
}

#[test]
fn geom_terms_matches_fortran() {
    let mut r = Golden::open("tier1", "geom_terms.bin");
    let mut cases = 0;
    while !r.done() {
        let j0 = r.usize();
        let n = r.usize();
        let itype = r.i32();
        let hs = r.f64();
        let p0 = r.f64();

        let mut vmod = VelocityModel::new();
        for k in 0..j0 {
            vmod[k].thickness_km = r.f64();
        }
        for k in 0..j0 {
            vmod[k].vsh_km_s = r.f64();
        }
        for k in 0..j0 {
            vmod[k].attenuation_s = r.f32();
        }

        let mut st = RayState::default();
        for k in 0..n {
            st.rays.nh[k] = r.i32() - 1;
        }
        st.rays.nd = n as i32;

        let w_rp = r.f64();
        let w_qb = r.f32();

        let (rp, qb) = geometric_spreading(&st, &vmod, hs, p0, Takeoff::from_ray_type(itype));
        let tag = format!("geometric_spreading case {cases} (itype={itype} p0={p0})");
        eq64(&format!("{tag} rp"), rp, w_rp);
        // The single-precision accumulation of qb is exactly what this pins.
        eq32(&format!("{tag} qb"), qb, w_qb);
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn subfault_geometry_stays_close_to_fortran() {
    // Bit-exactness ended with §2.5: this routine's only source of position is
    // `distance_azimuth`, which is now a WGS84 geodesic rather than DELAZ5. Kept as a
    // bounded check for the same reason as the delaz5 fixture -- it still catches a gross
    // error, and it quantifies the deliberate change on the real fault geometries the
    // fixture holds, which is the evidence Tier C needs to be interpretable.
    //
    // `depth_km` is untouched: it comes from the dip and the down-dip index, not from the
    // geodesy, so it is still asserted BIT-EXACT. That asymmetry is the point -- if depth
    // moved, something other than §2.5 changed.
    let mut r = Golden::open("tier1", "even_dist2.bin");
    let mut cases = 0;
    let mut worst_horiz = 0.0f64;
    let mut worst_slant = 0.0f64;
    let mut worst_takeoff_deg = 0.0f64;
    let mut worst_azimuth_deg = 0.0f64;
    while !r.done() {
        let nx = r.usize();
        let nw = r.usize();
        let (xlonq, ylatq, slon, slat) = (r.f32(), r.f32(), r.f32(), r.f32());
        let (azmq, dipangq, zm, astop) = (r.f32(), r.f32(), r.f32(), r.f32());
        let (dx, dy) = (r.f32(), r.f32());

        let g = subfault_geometry(
            xlonq, ylatq, slon, slat, azmq, dipangq, zm, astop, dx, dy, nx, nw,
        );

        // Driver dump order: ((dst,rl,th,ph,zet), j=1,nw), i=1,nx)
        for i in 1..=nx {
            for j in 1..=nw {
                let tag = format!("subfault_geometry case {cases} ({i},{j})");
                let ray = g.at(i, j);
                let (want_horiz, want_slant, want_takeoff, want_azimuth, want_depth) =
                    (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());

                let rel = |got: f32, want: f32| {
                    if want.abs() > 1.0 { ((got - want) / want).abs() as f64 } else { 0.0 }
                };
                worst_horiz = worst_horiz.max(rel(ray.horiz_km, want_horiz));
                worst_slant = worst_slant.max(rel(ray.slant_km, want_slant));
                worst_takeoff_deg = worst_takeoff_deg
                    .max((ray.takeoff_rad - want_takeoff).abs().to_degrees() as f64);
                worst_azimuth_deg = worst_azimuth_deg.max({
                    let d = (ray.azimuth_rad - want_azimuth).abs().to_degrees() as f64;
                    d.min(360.0 - d)
                });

                // Depth does not pass through the geodesy at all.
                eq32(&format!("{tag} zet"), ray.depth_km, want_depth);
            }
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 5);

    println!(
        "even_dist2 fixture: worst horiz {:.4}%, slant {:.4}%, takeoff {worst_takeoff_deg:.4} deg, \
         azimuth {worst_azimuth_deg:.4} deg",
        worst_horiz * 100.0,
        worst_slant * 100.0
    );
    assert!(worst_horiz < 0.01, "horizontal distance moved {:.4}%", worst_horiz * 100.0);
    assert!(worst_slant < 0.01, "slant distance moved {:.4}%", worst_slant * 100.0);
    assert!(worst_takeoff_deg < 0.5, "take-off angle moved {worst_takeoff_deg:.4} deg");
    assert!(worst_azimuth_deg < 0.5, "azimuth moved {worst_azimuth_deg:.4} deg");
}
