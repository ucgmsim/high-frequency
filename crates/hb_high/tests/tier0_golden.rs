//! Golden tests for the leaf kernels, against fixtures produced by the original Fortran.
//!
//! Each record holds the inputs as well as the outputs, and the tests read the inputs from
//! the file, so a test cannot pass by comparing two different input sets. Comparisons are
//! exact unless a test documents why its kernel deliberately differs.
//!
//! Fixture filenames are the Fortran routine names, not this crate's: `cr.bin` holds the
//! golden for `ray::vertical_slowness`, `delaz5.bin` for `geom::distance_azimuth`, and so
//! on.

use hb_high::fft::remove_quadratic_trend;
use hb_high::fft::{Complex32, Complex64};
use hb_high::geom::{GeoPoint, distance_azimuth};
use hb_high::radiation::{RadiationAngles, radiation_pattern};
use hb_high::ray::vertical_slowness;
use hb_high::site::{apply_site_amplification, site_gain_curve};
use ndarray::{ArrayView1, ArrayViewMut1};

mod common;
use common::*;

#[test]
fn rdatn_matches_fortran() {
    let mut r = Golden::open("tier0", "rdatn.bin");
    let mut n = 0;
    while !r.done() {
        let (str_, dip, rak, az, th) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (w_sh, w_sv) = (r.f32(), r.f32());
        let coefficients = radiation_pattern(RadiationAngles {
            strike_rad: str_,
            dip_rad: dip,
            rake_rad: rak,
            azimuth_rad: az,
            takeoff_rad: th,
        });
        let (sh, sv) = (coefficients.sh, coefficients.sv);
        eq32(&format!("radiation_pattern case {n} rdsh"), sh, w_sh);
        eq32(&format!("radiation_pattern case {n} rdsv"), sv, w_sv);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 2000);
}

#[test]
fn distance_azimuth_stays_close_to_fortran() {
    // `distance_azimuth` uses a WGS84 geodesic where the Fortran used a spherical
    // approximation, so this is a bounded comparison, not an exact one. It still catches
    // gross errors: a degrees/radians mix-up, swapped forward/back azimuths, or the wrong
    // slot of geographiclib's return tuple. Only distance and forward azimuth are compared.
    //
    // The azimuth bound is stratified by separation:
    //
    //   * under 500 km -- every source-to-station distance the program sees -- azimuth
    //     agrees to 0.045 degrees, and is asserted.
    //   * near-antipodal (the fixture reaches 18,729 km) it disagrees by up to 1.1
    //     degrees, because geodesic azimuth is ill-conditioned there. That figure is
    //     printed, not asserted.
    let mut r = Golden::open("tier0", "delaz5.bin");
    let mut n = 0;
    let mut worst_km_rel = 0.0f64;
    let mut worst_km_at = String::new();
    let mut worst_az_global = 0.0f64;
    let mut worst_az_global_at = String::new();
    let mut worst_az_near = 0.0f64;
    let mut worst_az_near_at = String::new();
    while !r.done() {
        let (thei, alei, thsi, alsi) = (r.f32(), r.f32(), r.f32(), r.f32());
        let iflag = r.i32();
        let [_delt, _deltdg, want_km, _azes, want_azdg, _azse, _azsedg] = [
            r.f32(),
            r.f32(),
            r.f32(),
            r.f32(),
            r.f32(),
            r.f32(),
            r.f32(),
        ];
        // `distance_azimuth` has no geocentric-radians mode; the fixture must not use it.
        assert!(iflag <= 0, "case {n} used the dead coord_mode > 0 path");

        let g = distance_azimuth(
            GeoPoint {
                lat_deg: thei,
                lon_deg: alei,
            },
            GeoPoint {
                lat_deg: thsi,
                lon_deg: alsi,
            },
        );

        // At zero separation the azimuth is arbitrary in both formulations.
        if want_km > 1.0 {
            let where_ = || format!("case {n} ({thei},{alei})->({thsi},{alsi}) {want_km} km");
            let rel = ((g.deltkm - want_km) / want_km).abs() as f64;
            if rel > worst_km_rel {
                worst_km_rel = rel;
                worst_km_at = where_();
            }
            let gap = {
                let d = (g.azesdg - want_azdg).abs() as f64;
                d.min(360.0 - d)
            };
            if gap > worst_az_global {
                worst_az_global = gap;
                worst_az_global_at = where_();
            }
            if want_km < 500.0 && gap > worst_az_near {
                worst_az_near = gap;
                worst_az_near_at = where_();
            }
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 2000);

    println!(
        "delaz5 fixture, 2000 cases:\n  \
         worst distance      {:.4}% at {worst_km_at}\n  \
         worst azimuth <500km {worst_az_near:.4} deg at {worst_az_near_at}\n  \
         worst azimuth global {worst_az_global:.4} deg at {worst_az_global_at}",
        worst_km_rel * 100.0
    );

    // The Fortran's distances are systematically short (a 6371 km sphere with a tangent
    // correction for the ellipsoid). The largest relative error is at the short end --
    // 0.32% at 1.04 km, about 3 m -- where it switches to a half-chord formula.
    assert!(
        worst_km_rel < 0.005,
        "distance moved by {:.4}% at {worst_km_at}, more than the 0.5% allowed",
        worst_km_rel * 100.0
    );
    assert!(
        worst_az_near < 0.1,
        "azimuth moved by {worst_az_near:.4} degrees at {worst_az_near_at}; under 500 km \
         the two formulations should agree to 0.1 degrees"
    );
}

/// `vertical_slowness` against the Fortran, exact except on the branch cut.
///
/// On the branch cut (`|Im(p)| < 1e-8` with `a < 0`) the phase is forced to pi rather
/// than taken from `atan2`. The Fortran used a truncated `3.141592654d0`, 4.1e-10 short
/// of pi, so its `cos(phi/2)` -- which should be exactly zero -- came out at ~5.7e-11;
/// with `std::f64::consts::PI` it is ~1e-17. The real part there is therefore only
/// required to be closer to zero than the golden, within a bound derived from the
/// Fortran's pi error. The imaginary part, which carries the magnitude of `eta` on this
/// branch, is compared exactly on every case.
#[test]
fn cr_stays_close_to_fortran() {
    let mut r = Golden::open("tier0", "cr.bin");
    let mut n = 0;
    // Worst divergence, relative to the magnitude of eta for that case.
    let mut worst_rel = 0.0f64;
    let mut worst_at = String::new();
    // Cases where the port and the oracle still agree bit for bit.
    let mut exact = 0;

    while !r.done() {
        let p = Complex64::new(r.f64(), r.f64());
        let v = r.f64();
        let want = Complex64::new(r.f64(), r.f64());
        let got = vertical_slowness(p, v);

        // The magnitude is untouched by the constant and is the natural scale for the
        // real part's departure from zero.
        eq64(&format!("vertical_slowness({p:?},{v}) im"), got.im, want.im);

        if got.re.to_bits() == want.re.to_bits() {
            exact += 1;
        } else {
            let scale = want.norm().max(f64::MIN_POSITIVE);
            let rel = (got.re - want.re).abs() / scale;
            if rel > worst_rel {
                worst_rel = rel;
                worst_at = format!("p={p:?} v={v}");
            }
            // Every divergent case must be one where the golden's value was numerical
            // noise around zero and ours is smaller noise.
            assert!(
                got.re.abs() < want.re.abs(),
                "vertical_slowness({p:?},{v}) re: rust {got:?} is not closer to zero \
                 than fortran {want:?}; the pi fix should only ever shrink this term"
            );
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 1500);

    println!(
        "cr fixture, {n} cases: {exact} still bit-exact, \
         worst real-part divergence {worst_rel:.3e} of |eta| at {worst_at}"
    );

    // The Fortran's pi error (4.1e-10) reaches `cos(phi/2)` halved, bounding the
    // departure at ~2.05e-10 of |eta|. Measured worst: 2.051e-10, on the 450 of 1500
    // cases on the branch cut; the other 1050 are bit-exact. The bound is 5x headroom.
    assert!(
        worst_rel < 1.0e-9,
        "vertical_slowness moved by {worst_rel:.3e} of |eta| at {worst_at}, more than \
         the Fortran's own pi error can account for"
    );
}

#[test]
fn flzero_matches_fortran() {
    let mut r = Golden::open("tier0", "flzero.bin");
    let mut cases = 0;
    while !r.done() {
        let n = r.i32() as usize;
        let dt = r.f32();
        // Labels report the Fortran's 1-based sample number, to match the fixture.
        let mut a = vec![0.0; n];
        for slot in a.iter_mut() {
            *slot = r.f32();
        }
        let a_in = a.clone();
        let want: Vec<f32> = (0..n).map(|_| r.f32()).collect();

        remove_quadratic_trend(dt, a.as_mut_slice());
        for i in 0..n {
            eq32(
                &format!("remove_quadratic_trend n={n} dt={dt} a[{}]", i + 1),
                a[i],
                want[i],
            );
        }
        // The correction starts at the third sample, so the first two must come back
        // untouched. Pinned explicitly because it is easy to "fix".
        eq32(
            &format!("remove_quadratic_trend n={n} a[1] must be untouched"),
            a[0],
            a_in[0],
        );
        eq32(
            &format!("remove_quadratic_trend n={n} a[2] must be untouched"),
            a[1],
            a_in[1],
        );
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn siteamp_matches_fortran() {
    let mut r = Golden::open("tier0", "siteamp.bin");
    let mut cases = 0;
    while !r.done() {
        let np2 = r.i32() as usize;
        let nn = r.i32() as usize;
        let np = np2 / 2;

        let dfr: Vec<f32> = (0..=np).map(|_| r.f32()).collect();
        let fn_: Vec<f32> = (0..nn).map(|_| r.f32()).collect();
        let an: Vec<f32> = (0..nn).map(|_| r.f32()).collect();
        let mut cw: Vec<Complex32> = (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want: Vec<Complex32> = (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();

        // `site_gain_curve` takes log frequencies (precomputed per segment by
        // `SpectrumPlan` in the program) and interpolates the gain;
        // `apply_site_amplification` applies it.
        let log_dfr: Vec<f32> = dfr.iter().map(|f| f.ln()).collect();
        let mut gain = vec![0.0f32; np + 1];
        site_gain_curve(
            ArrayView1::from(log_dfr.as_slice()),
            ArrayView1::from(&fn_[..nn]),
            ArrayView1::from(&an[..nn]),
            ArrayViewMut1::from(gain.as_mut_slice()),
        );
        apply_site_amplification(cw.as_mut_slice(), ArrayView1::from(gain.as_slice()));

        // The Fortran scaled the DC and Nyquist bins by the raw table factor while
        // exponentiating every other bin; this crate exponentiates all of them. Those two
        // bins (indices 0 and np2/2; Nyquist is its own Hermitian partner) are checked
        // separately below, and every other bin exactly.
        let nyquist = np2 / 2;
        for i in 0..np2 {
            if i == 0 || i == nyquist {
                continue;
            }
            eq32(
                &format!("apply_site_amplification np2={np2} [{}].re", i + 1),
                cw[i].re,
                want[i].re,
            );
            eq32(
                &format!("apply_site_amplification np2={np2} [{}].im", i + 1),
                cw[i].im,
                want[i].im,
            );
        }

        // The two excluded bins must differ from the golden by exactly exp(factor)/factor.
        // DC takes the first table entry and Nyquist the clamped last one.
        for (i, factor) in [(0usize, an[0]), (nyquist, an[nn - 1])] {
            let fortran_gain = factor;
            let fixed_gain = factor.exp();
            if want[i].re.abs() > 1e-20 && (fortran_gain - fixed_gain).abs() > 1e-6 {
                let ratio = cw[i].re / want[i].re;
                let expected = fixed_gain / fortran_gain;
                assert!(
                    (ratio / expected - 1.0).abs() < 1e-3,
                    "bin {i}: gain ratio {ratio} vs expected exp({factor})/{factor} = {expected}"
                );
            }
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}
