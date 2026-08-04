//! Bit-identity gate for the tier-2 kernels.
//!
//! Regenerate with `harness/kernels/gen_tier2_golden.sh`.
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.

use hb_high::fort::{Complex32, Complex64};
use hb_high::highcor::apply_radiation_and_invert;
use hb_high::radiation::{horizontal_radiation_spectrum, vertical_radiation_spectrum};
use hb_high::ray::{cagniard_time, cagniard_time_derivative};
use hb_high::rng::Pcg32;
use hb_high::state::{RayState, VelocityModel};

mod common;
use common::*;

/// Relative tolerance for the two `f64` goldens in this tier.
///
/// Two independent causes, both understood and both bounded:
///
/// * **Complex division.** §2.2 replaced the hand-written complex arithmetic with
///   `num-complex`, whose division does not use gfortran's Smith-with-range-reduction
///   branch and differs by 1-2 ulps. Only `cagniard_time_derivative` reaches one.
/// * **Pi.** §2.8 gave `vertical_slowness` the correctly rounded branch-cut phase, which
///   moves the real part of `eta` by up to 2.05e-10 of its magnitude — see
///   `tier0_golden::cr_stays_close_to_fortran`. Both routines here sum `eta` over layers,
///   so both inherit it.
///
/// The pi term dominates (~1e-16 for the division ulps). Measured worsts:
///
/// ```text
/// cagcon  1.082e-10   cagniard_time, a plain sum of eta over layers
/// dtdp    7.012e-10   cagniard_time_derivative, which divides BY eta
/// ```
///
/// `cagcon` stays under `eta`'s own 2.05e-10, as a sum should. `dtdp` amplifies it ~3.4x,
/// which is expected rather than alarming: it divides by `eta`, and near the cut `eta` is
/// small, so a fixed upstream perturbation is magnified by how close the case sits to the
/// cut. That amplification is data-dependent, so this needs real headroom over the
/// observed worst rather than a snug fit — hence ~7x, still four orders tighter than
/// anything that could hide a wrong branch, a swapped operand or a lost term.
const PI_DIVERGENCE_TOL: f64 = 5e-9;

/// Shared record layout for the `cagniard_time`/`cagniard_time_derivative` seam. The
/// header is local; the state block after it is identical to tier 3's.
fn read_ray_seam(r: &mut Golden) -> (RayState, VelocityModel, Complex64, f64, usize) {
    let ndp = r.usize();
    let p = Complex64::new(r.f64(), r.f64());
    let rr = r.f64();
    let (st, vmod) = r.ray_seam_state(ndp);
    (st, vmod, p, rr, ndp)
}


// `eq64` is gone: both `f64` goldens in this tier now go through `near64`, since the pi
// correction in `vertical_slowness` reaches every value they compare. The `f32` goldens
// below are untouched by it and are still exact.

/// Was exact; now bounded. `cagniard_time` sums `vertical_slowness` over layers, so it
/// inherits the pi correction described on [`near64`].
#[test]
fn cagcon_stays_close_to_fortran() {
    let mut r = Golden::open("tier2", "cagcon.bin");
    let mut n = 0;
    let mut div = Divergence::default();
    while !r.done() {
        let (st, vmod, p, rr, ndp) = read_ray_seam(&mut r);
        let want = Complex64::new(r.f64(), r.f64());
        let got = cagniard_time(&st, &vmod, p, rr);
        let re = format!("cagniard_time case {n} (ndeep={ndp}) re");
        let im = format!("cagniard_time case {n} (ndeep={ndp}) im");
        div.note(&re, got.re, want.re);
        div.note(&im, got.im, want.im);
        near64(&re, got.re, want.re, PI_DIVERGENCE_TOL);
        near64(&im, got.im, want.im, PI_DIVERGENCE_TOL);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 200);
    div.report("cagcon");
}

#[test]
fn dtdp_stays_close_to_fortran() {
    let mut r = Golden::open("tier2", "dtdp.bin");
    let mut n = 0;
    let mut div = Divergence::default();
    while !r.done() {
        let (st, vmod, p, rr, ndp) = read_ray_seam(&mut r);
        let want = Complex64::new(r.f64(), r.f64());
        // Exercises complex division: Smith's algorithm, not (ac+bd)/(c^2+d^2).
        let got = cagniard_time_derivative(&st, &vmod, p, rr);
        let re = format!("cagniard_time_derivative case {n} (ndeep={ndp}) re");
        let im = format!("cagniard_time_derivative case {n} (ndeep={ndp}) im");
        div.note(&re, got.re, want.re);
        div.note(&im, got.im, want.im);
        near64(&re, got.re, want.re, PI_DIVERGENCE_TOL);
        near64(&im, got.im, want.im, PI_DIVERGENCE_TOL);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 200);
    div.report("dtdp");
}

#[test]
fn highcor_f_matches_fortran() {
    let mut r = Golden::open("tier2", "highcor_f.bin");
    let mut cases = 0;
    while !r.done() {
        let nf = r.usize();
        let mf = r.usize();
        let np2 = r.usize();

        // 0-based since §2.3; labels keep the Fortran's 1-based index.
        let rdna: Vec<f32> = (0..nf).map(|_| r.f32()).collect();
        let mut cw1: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want_cw: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want_stdd: Vec<f32> = (0..np2).map(|_| r.f32()).collect();

        let mut stdd = vec![0.0; np2];
        apply_radiation_and_invert(nf, mf, cw1.as_mut_slice(), stdd.as_mut_slice(), rdna.as_slice());

        let cw_scale = want_cw.iter().fold(0.0f32, |a, c| a.max(c.re.abs()).max(c.im.abs()));
        let stdd_scale = want_stdd.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for i in 0..np2 {
            let at = i + 1;
            near32(&format!("apply_radiation_and_invert np2={np2} cw1[{at}].re"), cw1[i].re, want_cw[i].re, cw_scale);
            near32(&format!("apply_radiation_and_invert np2={np2} cw1[{at}].im"), cw1[i].im, want_cw[i].im, cw_scale);
            near32(&format!("apply_radiation_and_invert np2={np2} stdd[{at}]"), stdd[i], want_stdd[i], stdd_scale);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}

#[test]
fn radfrq_lin_matches_fortran() {
    let mut r = Golden::open("tier2", "radfrq_lin.bin");
    let mut cases = 0;
    while !r.done() {
        let (stra, dipa, raka) = (r.f32(), r.f32(), r.f32());
        let (pa, thaa, cmp) = (r.f32(), r.f32(), r.f32());
        let nfold = r.usize();
        let nr = r.usize();
        let seed = r.i32();

        let dfr: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();
        let want_fr1 = r.f32();
        let want_rdna: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();
        let want_after: Vec<f32> = (0..8).map(|_| r.f32()).collect();

        let (mut rng, _) = Pcg32::seed(seed);
        let mut rdna = vec![0.0; nfold];
        let fr1 = horizontal_radiation_spectrum(&mut rng, stra, dipa, raka, pa, thaa, dfr.as_slice(), nfold, cmp, nr, rdna.as_mut_slice());

        let tag = format!("horizontal_radiation_spectrum case {cases} (cmp={cmp})");
        eq32(&format!("{tag} fr1 (clobbered)"), fr1, want_fr1);
        for i in 0..nfold {
            eq32(&format!("{tag} rdna[{}]", i + 1), rdna[i], want_rdna[i]);
        }
        // The stream position after the call. This is what catches a version
        // that gets the pattern right while drawing the wrong number of
        // deviates -- 5 per iteration, in the order th, fa, strX, dipX, rakX.
        for (k, w) in want_after.iter().enumerate() {
            eq32(&format!("{tag} post-call draw {k} (generator position)"),
                 rng.next_f32(), *w);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn radv_lin_matches_fortran() {
    let mut r = Golden::open("tier2", "radv_lin.bin");
    let mut cases = 0;
    while !r.done() {
        let (stra, dipa, raka, pa, thaa) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let nfold = r.usize();
        let nr = r.usize();

        let dfr: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();
        let rna: Vec<f32> = (0..nr).map(|_| r.f32()).collect();
        let rnb: Vec<f32> = (0..nr).map(|_| r.f32()).collect();
        let want_fr1 = r.f32();
        let want_rdna: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();

        let mut rdna = vec![0.0; nfold];
        let fr1 = vertical_radiation_spectrum(stra, dipa, raka, pa, thaa, dfr.as_slice(), nfold, rna.as_slice(), rnb.as_slice(), nr, rdna.as_mut_slice());

        let tag = format!("vertical_radiation_spectrum case {cases}");
        eq32(&format!("{tag} fr1 (clobbered to 0.001)"), fr1, want_fr1);
        for i in 0..nfold {
            eq32(&format!("{tag} rdna[{}]", i + 1), rdna[i], want_rdna[i]);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}
