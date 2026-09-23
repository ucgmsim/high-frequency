//! Property tests: the contracts a caller depends on, independent of how the code
//! computes them.
//!
//! Unlike the golden tests, nothing here asserts a computed value, a bit pattern, or
//! anything about the internals -- only relationships that must hold for any correct
//! implementation, so an implementation can be replaced without touching this file.
//!
//! # Deliberately not asserted
//!
//! * **Transform scaling conventions.** The round trip is asserted *proportional* to the
//!   input with one constant, never what the constant is.
//! * **Idempotence of `remove_quadratic_trend`.** It does not hold: a second pass moves
//!   the signal a further ~6% at n=64.
//!
//! Tolerances are set from measured worst cases with two or more orders of magnitude of
//! headroom, so ordinary rounding differences pass and a genuine break fails.

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::fft::{Complex32, Complex64};
use hb_high::fft::{forward, inverse, remove_quadratic_trend};
use hb_high::geom::{FaultPlane, GeoPoint, distance_azimuth, subfault_geometry};
use hb_high::input::{Segment, Slip, Station, StochModel, Subfault, build_velocity_model};
use hb_high::radiation::{RadiationAngles, radiation_pattern};
use hb_high::ray::vertical_slowness;
use hb_high::rng::{Draws, LegacyPcg};
use hb_high::site::{apply_site_amplification, site_gain_curve};
use libm::tgamma as gamma;
use ndarray::{ArrayView1, ArrayViewMut1};
use proptest::prelude::*;

/// Angular difference in degrees, folded into `[0, 180]`.
fn angle_gap_deg(a: f32, b: f32) -> f32 {
    let d = (a - b).abs() % 360.0;
    if d > 180.0 { 360.0 - d } else { d }
}

/// A complex spectrum of `n` bins, filled deterministically from `seed`.
fn spectrum(n: usize, seed: i32) -> Vec<Complex32> {
    let mut rng = LegacyPcg::seed(seed);
    (0..n)
        .map(|_| Complex32::new(rng.uniform() - 0.5, rng.uniform() - 0.5))
        .collect()
}

// ---------------------------------------------------------------------------
// Geodesy — distance_azimuth
// ---------------------------------------------------------------------------

proptest! {
    /// A point is zero distance from itself. Catches catastrophic cancellation in the
    /// near-coincident regime.
    #[test]
    fn self_distance_is_zero(lat in -85.0f32..85.0, lon in -180.0f32..180.0) {
        let g = distance_azimuth(GeoPoint { lat_deg: lat, lon_deg: lon }, GeoPoint { lat_deg: lat, lon_deg: lon });
        prop_assert_eq!(g.deltkm, 0.0);
    }

    /// Distance does not depend on which point you call the source.
    #[test]
    fn distance_is_symmetric(
        lat_a in -85.0f32..85.0, lon_a in -180.0f32..180.0,
        lat_b in -85.0f32..85.0, lon_b in -180.0f32..180.0,
    ) {
        let there = distance_azimuth(GeoPoint { lat_deg: lat_a, lon_deg: lon_a }, GeoPoint { lat_deg: lat_b, lon_deg: lon_b });
        let back = distance_azimuth(GeoPoint { lat_deg: lat_b, lon_deg: lon_b }, GeoPoint { lat_deg: lat_a, lon_deg: lon_a });
        prop_assume!(there.deltkm > 10.0);
        let rel = (there.deltkm - back.deltkm).abs() / there.deltkm;
        prop_assert!(rel < 1e-5, "{} vs {}", there.deltkm, back.deltkm);
    }

    /// For a nearby station the azimuth matches the flat-Earth bearing to the offset.
    ///
    /// This pins **our use of** `geographiclib_rs` rather than the library's accuracy:
    /// the right slot out of a return tuple whose element meanings change with its width,
    /// degrees rather than radians, the wrap into `[0, 360)`, and the `f64` -> `f32`
    /// narrowing. Any of those going wrong moves the answer by tens or hundreds of
    /// degrees, so a half-degree bound is ample.
    ///
    /// Forward/back azimuth reciprocity (a 180-degree difference) is not tested: that is a
    /// sphere's property, and on an ellipsoid a long geodesic's azimuth changes along its
    /// length.
    #[test]
    fn azimuth_matches_the_flat_earth_bearing_nearby(
        lat in -70.0f32..70.0,
        lon in -180.0f32..180.0,
        dlat in -0.2f32..0.2,
        dlon in -0.2f32..0.2,
    ) {
        // Reject offsets too small for the bearing itself to be well conditioned.
        prop_assume!(dlat.hypot(dlon) > 0.01);
        let g = distance_azimuth(GeoPoint { lat_deg: lat, lon_deg: lon }, GeoPoint { lat_deg: lat + dlat, lon_deg: lon + dlon });

        // Bearing from north, with the longitude offset foreshortened by the latitude.
        let east = dlon as f64 * (lat as f64).to_radians().cos();
        let north = dlat as f64;
        let want = east.atan2(north).to_degrees().rem_euclid(360.0);

        prop_assert!(
            angle_gap_deg(g.azesdg, want as f32) < 0.5,
            "at ({lat},{lon}) + ({dlat},{dlon}): azimuth {}, flat-Earth bearing {want}",
            g.azesdg
        );
    }

    /// The azimuth is reported in `[0, 360)`, as the doc comment promises.
    #[test]
    fn azimuths_are_in_range(
        lat_a in -85.0f32..85.0, lon_a in -180.0f32..180.0,
        lat_b in -85.0f32..85.0, lon_b in -180.0f32..180.0,
    ) {
        let g = distance_azimuth(GeoPoint { lat_deg: lat_a, lon_deg: lon_a }, GeoPoint { lat_deg: lat_b, lon_deg: lon_b });
        prop_assert!((0.0..360.0).contains(&g.azesdg), "azimuth {} out of range", g.azesdg);
        prop_assert!(
            (0.0..std::f32::consts::TAU).contains(&g.azes),
            "azimuth {} rad", g.azes
        );
    }
}

/// Cardinal directions from the equator, and the scale of a degree.
///
/// Parametrised rather than generated: these are the four cases a reader checks by
/// hand. At the equator a degree of latitude is shorter than one of longitude because
/// the Earth is flattened.
#[test]
fn cardinal_azimuths_and_degree_scale() {
    for (dlat, dlon, want_az, what) in [
        (1.0f32, 0.0f32, 0.0f32, "north"),
        (0.0, 1.0, 90.0, "east"),
        (-1.0, 0.0, 180.0, "south"),
        (0.0, -1.0, 270.0, "west"),
    ] {
        let g = distance_azimuth(
            GeoPoint {
                lat_deg: 0.0,
                lon_deg: 0.0,
            },
            GeoPoint {
                lat_deg: dlat,
                lon_deg: dlon,
            },
        );
        assert!(
            angle_gap_deg(g.azesdg, want_az) < 0.01,
            "due {what}: azimuth {}, want {want_az}",
            g.azesdg
        );
        assert!(
            (100.0..125.0).contains(&g.deltkm),
            "one degree {what} is {} km, not a plausible degree",
            g.deltkm
        );
    }
    // Flattening: a degree of latitude is the shorter of the two.
    let lat_km = distance_azimuth(
        GeoPoint {
            lat_deg: 0.0,
            lon_deg: 0.0,
        },
        GeoPoint {
            lat_deg: 1.0,
            lon_deg: 0.0,
        },
    )
    .deltkm;
    let lon_km = distance_azimuth(
        GeoPoint {
            lat_deg: 0.0,
            lon_deg: 0.0,
        },
        GeoPoint {
            lat_deg: 0.0,
            lon_deg: 1.0,
        },
    )
    .deltkm;
    assert!(
        lat_km < lon_km,
        "lat {lat_km} should be < lon {lon_km} at the equator"
    );
}

// ---------------------------------------------------------------------------
// Geometry — subfault_geometry
// ---------------------------------------------------------------------------

proptest! {
    /// The slant distance is the hypotenuse of the horizontal distance and the
    /// subfault depth. This is the invariant that keeps the two from being confused
    /// at a call site — they differ only by whether depth is included, and the
    /// path-duration table and `d10` both want the slant one.
    #[test]
    fn slant_is_the_hypotenuse_of_horizontal_and_depth(
        dip_deg in 5.0f32..85.0,
        top_depth_km in 0.5f32..25.0,
        subfault_km in 0.5f32..4.0,
        station_offset_deg in 0.05f32..2.0,
    ) {
        let (along, down) = (4usize, 3usize);
        let g = subfault_geometry(
            &FaultPlane {
                origin: GeoPoint { lat_deg: -43.0, lon_deg: 173.0 },
                strike_deg: 220.0,
                dip_deg,
                top_depth_km,
                along_strike_offset_km: 0.5 * along as f32 * subfault_km,
                subfault_length_km: subfault_km,
                subfault_width_km: subfault_km,
                along_strike_count: along,
                down_dip_count: down,
            },
            GeoPoint { lat_deg: -43.0, lon_deg: 173.0 + station_offset_deg },
        );
        for i in 1..=along {
            for j in 1..=down {
                let ray = g.at(i, j);
                let (slant, horiz, depth) = (ray.slant_km, ray.horiz_km, ray.depth_km);
                prop_assert!(slant.is_finite() && horiz.is_finite() && depth.is_finite());
                let hyp = (horiz * horiz + depth * depth).sqrt();
                prop_assert!(
                    (hyp - slant).abs() / slant < 1e-5,
                    "({i},{j}) slant {slant} vs hypot {hyp}"
                );
                // Corollary a caller relies on when choosing between the two.
                prop_assert!(slant >= horiz, "({i},{j}) slant {slant} < horiz {horiz}");
            }
        }
    }

    /// Subfaults get deeper as you go down dip, for any dip that is not horizontal.
    #[test]
    fn depth_increases_down_dip(dip_deg in 5.0f32..85.0, top_depth_km in 0.5f32..25.0) {
        let (along, down) = (3usize, 5usize);
        let g = subfault_geometry(
            &FaultPlane {
                origin: GeoPoint { lat_deg: -43.0, lon_deg: 173.0 },
                strike_deg: 220.0,
                dip_deg,
                top_depth_km,
                along_strike_offset_km: 0.5 * along as f32 * 1.5,
                subfault_length_km: 1.5,
                subfault_width_km: 1.5,
                along_strike_count: along,
                down_dip_count: down,
            },
            GeoPoint { lat_deg: -43.0, lon_deg: 173.5 },
        );
        for i in 1..=along {
            for j in 2..=down {
                prop_assert!(
                    g.at(i, j).depth_km > g.at(i, j - 1).depth_km,
                    "({i},{j}) depth {} not below {}",
                    g.at(i, j).depth_km,
                    g.at(i, j - 1).depth_km
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Ray theory — vertical_slowness
// ---------------------------------------------------------------------------

proptest! {
    /// `eta^2 + p^2 == 1/v^2`. This *is* the definition of vertical slowness, and it
    /// holds whichever branch of the square root the implementation selects, so it pins
    /// the contract without constraining the branch-cut choice.
    #[test]
    fn slowness_satisfies_its_defining_identity(
        velocity_km_s in 0.5f64..9.0,
        p_scaled in -2.0f64..2.0,
        p_imag in -0.5f64..0.5,
    ) {
        let ray_parameter = Complex64::new(p_scaled / velocity_km_s, p_imag);
        let eta = vertical_slowness(ray_parameter, velocity_km_s);
        let lhs = eta * eta + ray_parameter * ray_parameter;
        let rhs = 1.0 / (velocity_km_s * velocity_km_s);
        prop_assert!((lhs.re - rhs).abs() / rhs < 1e-10, "re {} vs {rhs}", lhs.re);
        prop_assert!(lhs.im.abs() / rhs < 1e-10, "im {} should vanish", lhs.im);
    }

    /// Below the critical ray parameter the wave propagates vertically, so the
    /// slowness is real; above it the wave is evanescent and the slowness is
    /// imaginary. Callers branch on exactly this.
    #[test]
    fn propagating_and_evanescent_regimes_are_distinguished(
        velocity_km_s in 0.5f64..9.0,
        fraction in 0.05f64..0.9,
    ) {
        let critical = 1.0 / velocity_km_s;
        let propagating = vertical_slowness(Complex64::from(fraction * critical), velocity_km_s);
        prop_assert!(
            propagating.im.abs() < 1e-12 * critical.max(1.0),
            "propagating slowness should be real, got {propagating:?}"
        );
        prop_assert!(propagating.re.abs() > 0.0);

        let evanescent =
            vertical_slowness(Complex64::from(critical / fraction), velocity_km_s);
        prop_assert!(
            evanescent.re.abs() <= evanescent.im.abs(),
            "evanescent slowness should be dominated by its imaginary part, got {evanescent:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Gamma
// ---------------------------------------------------------------------------

proptest! {
    /// `Gamma(x+1) == x * Gamma(x)`, the functional equation any correct gamma satisfies.
    #[test]
    fn gamma_satisfies_its_recurrence(x in 0.1f64..18.0) {
        let ratio = gamma(x + 1.0) / (x * gamma(x));
        prop_assert!((ratio - 1.0).abs() < 1e-10, "Gamma({}) recurrence gave {ratio}", x);
    }

    /// Gamma is positive and increasing above its minimum near 1.4616.
    #[test]
    fn gamma_is_positive_and_increasing_above_its_minimum(x in 1.5f64..17.0) {
        prop_assert!(gamma(x) > 0.0);
        prop_assert!(gamma(x + 0.25) > gamma(x), "not increasing at {x}");
    }
}

/// The closed-form values, which fix the normalisation the recurrence alone cannot.
#[test]
fn gamma_matches_factorials_and_the_half_integer_case() {
    let mut factorial = 1.0f64;
    for n in 1..=10u32 {
        // Gamma(n) == (n-1)!
        let got = gamma(n as f64);
        assert!(
            (got / factorial - 1.0).abs() < 1e-12,
            "Gamma({n}) = {got}, want {factorial}"
        );
        factorial *= n as f64;
    }
    // Gamma(1/2) == sqrt(pi), and the recurrence then gives the rest of the ladder.
    let root_pi = std::f64::consts::PI.sqrt();
    for (x, want) in [
        (0.5, root_pi),
        (1.5, 0.5 * root_pi),
        (2.5, 0.75 * root_pi),
        (3.5, 1.875 * root_pi),
    ] {
        let got = gamma(x);
        assert!(
            (got / want - 1.0).abs() < 1e-12,
            "Gamma({x}) = {got}, want {want}"
        );
    }
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------

proptest! {
    /// Forward then inverse recovers the input up to **one** constant factor.
    ///
    /// Deliberately does not say what the factor is: a transform may be unnormalised
    /// (giving `n`) or normalise on either leg. What a caller relies on is that the pair
    /// is invertible and the scaling is uniform across bins; a per-bin scaling error is a
    /// real bug and this catches it.
    #[test]
    fn transform_round_trip_is_proportional_to_the_input(exponent in 3u32..9) {
        let n = 1usize << exponent;
        let original = spectrum(n, 7);
        let mut work = original.clone();
        forward(work.as_mut_slice());
        inverse(work.as_mut_slice());

        // Take the scale from the largest input bin, where it is best conditioned.
        let pivot = (0..n)
            .max_by(|&a, &b| original[a].norm().partial_cmp(&original[b].norm()).unwrap())
            .unwrap();
        let scale = work[pivot].re / original[pivot].re;
        prop_assert!(scale.is_finite() && scale.abs() > 0.0);

        let peak = original.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        for i in 0..n {
            for (got, want) in [
                (work[i].re, scale * original[i].re),
                (work[i].im, scale * original[i].im),
            ] {
                prop_assert!(
                    (got - want).abs() <= 1e-4 * scale.abs() * peak,
                    "bin {i}: {got} vs {want} (scale {scale})"
                );
            }
        }
    }

    /// The transform is linear. Cheap, and it is the property that a botched
    /// in-place butterfly or a mishandled scratch buffer breaks.
    #[test]
    fn transform_is_linear(exponent in 3u32..8, alpha in -3.0f32..3.0) {
        let n = 1usize << exponent;
        let (lhs, rhs) = (spectrum(n, 13), spectrum(n, 29));

        let combined: Vec<Complex32> =
            lhs.iter().zip(&rhs).map(|(l, r)| *l * alpha + *r).collect();

        let mut t_lhs = lhs.clone();
        let mut t_rhs = rhs.clone();
        let mut t_combined = combined.clone();
        for arr in [&mut t_lhs, &mut t_rhs, &mut t_combined] {
            forward(arr.as_mut_slice());
        }

        let peak = t_combined.iter().map(|c| c.norm()).fold(0.0f32, f32::max);
        for i in 0..n {
            let want = t_lhs[i] * alpha + t_rhs[i];
            prop_assert!(
                (t_combined[i].re - want.re).abs() <= 1e-4 * peak.max(1.0),
                "bin {i} re"
            );
            prop_assert!(
                (t_combined[i].im - want.im).abs() <= 1e-4 * peak.max(1.0),
                "bin {i} im"
            );
        }
    }

    /// A real signal has a real DC bin, because the zero-frequency term is a plain
    /// sum of real values. Convention-independent: normalisation scales it but
    /// cannot give it an imaginary part.
    #[test]
    fn real_input_has_a_real_dc_bin(exponent in 3u32..9) {
        let n = 1usize << exponent;
        let mut rng = LegacyPcg::seed(97);
        let mut sum = 0.0f32;
        let mut work = vec![Complex32::ZERO; n];
        for slot in work.iter_mut() {
            let v = rng.uniform() - 0.5;
            *slot = Complex32::new(v, 0.0);
            sum += v;
        }
        forward(work.as_mut_slice());
        prop_assert!(
            work[0].im.abs() <= 1e-4 * sum.abs().max(1.0),
            "DC bin {:?} should be real",
            work[0]
        );
    }
}

// ---------------------------------------------------------------------------
// Detrend
// ---------------------------------------------------------------------------

proptest! {
    /// The correction subtracted is a **quadratic** in the sample index, which is
    /// the whole content of "remove the quadratic trend". Checked by third
    /// difference, which annihilates any quadratic exactly.
    ///
    /// The integrated velocity and displacement are not tested for zero: the routine
    /// leaves the first two samples alone, so they do not reach it.
    #[test]
    fn correction_is_quadratic_in_sample_index(
        exponent in 5u32..11,
        offset in -1.0f32..1.0,
        dt in 0.001f32..0.05,
    ) {
        let n = 1usize << exponent;
        let mut rng = LegacyPcg::seed(11);
        let mut acceleration: Vec<f32> =
            (0..n).map(|_| rng.uniform() - 0.5 + offset).collect();
        let before = acceleration.clone();
        remove_quadratic_trend(dt, &mut acceleration);

        let correction: Vec<f64> = acceleration
            .iter()
            .zip(&before)
            .map(|(after, before)| (after - before) as f64)
            .collect();
        let span = correction.iter().fold(0.0f64, |acc, c| acc.max(c.abs()));
        prop_assume!(span > 1e-6);

        // Indices 0 and 1 are untouched by construction; the quadratic starts at 2.
        let mut worst = 0.0f64;
        for k in 2..n - 3 {
            let third = correction[k + 3] - 3.0 * correction[k + 2] + 3.0 * correction[k + 1]
                - correction[k];
            worst = worst.max(third.abs());
        }
        // The correction is recovered by differencing two `f32` samples of magnitude
        // ~|acceleration|, so it carries about half an ulp of THAT magnitude however small
        // the correction itself is. The third difference sums four such values with
        // coefficients 1, 3, 3, 1, so its noise floor is ~8 half-ulps of the acceleration.
        // A bound relative to `span` alone would flake when the fitted coefficients nearly
        // cancel and the span shrinks while the floor does not.
        let scale = before.iter().fold(0.0f64, |acc, a| acc.max(a.abs() as f64));
        let noise_floor = 8.0 * f32::EPSILON as f64 * scale;
        prop_assert!(
            worst < 1e-4 * span + noise_floor,
            "third difference {worst} exceeds {:e} (1e-4 of span {span} plus an f32 \
             recovery floor of {noise_floor:e}): correction is not quadratic",
            1e-4 * span + noise_floor
        );
    }
}

// ---------------------------------------------------------------------------
// Radiation pattern
// ---------------------------------------------------------------------------

proptest! {
    /// Double-couple radiation coefficients are bounded by unity. A caller scales
    /// spectra by these, so a coefficient above 1 would amplify rather than
    /// redistribute energy.
    #[test]
    fn radiation_coefficients_are_bounded_by_unity(
        strike in 0.0f32..6.3, dip in 0.0f32..1.6, rake in -3.2f32..3.2,
        azimuth in 0.0f32..6.3, takeoff in 0.0f32..3.2,
    ) {
        let coefficients = radiation_pattern(RadiationAngles {
            strike_rad: strike, dip_rad: dip, rake_rad: rake,
            azimuth_rad: azimuth, takeoff_rad: takeoff,
        });
        let (sh, sv) = (coefficients.sh, coefficients.sv);
        prop_assert!(sh.is_finite() && sv.is_finite());
        prop_assert!(sh.abs() <= 1.0 + 1e-4, "SH coefficient {sh}");
        prop_assert!(sv.abs() <= 1.0 + 1e-4, "SV coefficient {sv}");
    }

    /// Azimuth is an angle, so the pattern repeats every full turn.
    #[test]
    fn radiation_is_periodic_in_azimuth(
        strike in 0.0f32..6.3, dip in 0.0f32..1.6, rake in -3.2f32..3.2,
        azimuth in 0.0f32..6.3, takeoff in 0.0f32..3.2,
    ) {
        let at = |azimuth_rad| radiation_pattern(RadiationAngles {
            strike_rad: strike, dip_rad: dip, rake_rad: rake,
            azimuth_rad, takeoff_rad: takeoff,
        });
        let (here, turned) = (at(azimuth), at(azimuth + std::f32::consts::TAU));
        let (sh, sv) = (here.sh, here.sv);
        let (sh_turned, sv_turned) = (turned.sh, turned.sv);
        prop_assert!((sh - sh_turned).abs() < 1e-4, "SH {sh} vs {sh_turned}");
        prop_assert!((sv - sv_turned).abs() < 1e-4, "SV {sv} vs {sv_turned}");
    }
}

// ---------------------------------------------------------------------------
// Random number generation
// ---------------------------------------------------------------------------

proptest! {
    /// Uniform deviates lie in the unit interval. Downstream code indexes and takes
    /// logarithms of these, so an out-of-range value is not a cosmetic problem.
    #[test]
    fn uniform_deviates_lie_in_the_unit_interval(seed in any::<i32>(), count in 1usize..2048) {
        let mut rng = LegacyPcg::seed(seed);
        let mut out = vec![0.0; count];
        rng.fill_uniform(out.as_mut_slice());
        for (i, deviate) in out.iter().enumerate() {
            prop_assert!((0.0..1.0).contains(deviate), "deviate {i} = {deviate}");
        }
    }

    /// The normal deviates are renormalised to **unit RMS**, and `stochastic_spectrum`
    /// calibrates its amplitude on that; a plain N(0,1) generator would silently change
    /// output level. A contract rather than a statistical expectation, hence the tight
    /// tolerance.
    #[test]
    fn normal_deviates_have_unit_rms(seed in any::<i32>(), exponent in 6u32..13) {
        let count = 1usize << exponent;
        let mut rng = LegacyPcg::seed(seed);
        let mut out = vec![0.0; count];
        rng.fill_normal(out.as_mut_slice());

        let mean_square =
            out.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / count as f64;
        prop_assert!(
            (mean_square.sqrt() - 1.0).abs() < 1e-4,
            "RMS {} is not unity",
            mean_square.sqrt()
        );

        // Mean is not renormalised, so only require it to be small for the size of
        // the sample: many standard errors of slack, catching a systematic offset
        // without ever failing by luck.
        let mean = out.iter().map(|v| *v as f64).sum::<f64>() / count as f64;
        prop_assert!(
            mean.abs() < 8.0 / (count as f64).sqrt(),
            "mean {mean} too far from zero for n={count}"
        );
    }

    /// The same seed reproduces the same stream.
    #[test]
    fn a_seed_reproduces_its_stream(seed in any::<i32>(), draws in 1usize..64) {
        let mut first = LegacyPcg::seed(seed);
        let mut second = LegacyPcg::seed(seed);
        for _ in 0..draws {
            prop_assert_eq!(first.next_u32(), second.next_u32());
        }
    }

    /// Distinct seeds give distinct streams. Weak on its own, but it is what fails
    /// if seeding is ever accidentally made a no-op — which would leave every other
    /// test here passing.
    #[test]
    fn distinct_seeds_give_distinct_streams(seed in any::<i32>()) {
        let other = seed.wrapping_add(1);
        let mut a = LegacyPcg::seed(seed);
        let mut b = LegacyPcg::seed(other);
        let differs = (0..32).any(|_| a.next_u32() != b.next_u32());
        prop_assert!(differs, "seeds {seed} and {other} produced the same 32 draws");
    }
}

// ---------------------------------------------------------------------------
// Site amplification
// ---------------------------------------------------------------------------

proptest! {
    /// A flat table applies a uniform gain of `exp(level)` to every interior bin.
    ///
    /// The factors are **log** amplitudes: a constant table of 2.0 multiplies magnitudes
    /// by `e^2 = 7.389`, not by 2, so a table of ones is not the identity.
    ///
    /// The gain is uniform across the whole half-spectrum, DC and Nyquist included.
    #[test]
    fn a_flat_table_applies_a_uniform_exponential_gain(
        exponent in 4u32..9,
        level in -1.5f32..1.5,
    ) {
        let np2 = 1usize << exponent;
        let mut spec = spectrum(np2, 41);
        let original = spec.clone();
        let (frequency, log_frequency, factors) = site_table(np2, level);
        amplify(spec.as_mut_slice(), &frequency, &log_frequency[..6], &factors[..6]);

        let want = level.exp();
        for i in 2..=np2 / 2 {
            prop_assume!(original[i].norm() > 1e-3);
            let gain = spec[i].norm() / original[i].norm();
            prop_assert!(
                (gain / want - 1.0).abs() < 1e-3,
                "bin {i} gain {gain}, want exp({level}) = {want}"
            );
        }
    }

    /// Amplification is a real gain, so it scales magnitude and leaves phase alone.
    /// A complex factor slipping in — say from a mis-ordered multiply — would show
    /// up here and nowhere else.
    #[test]
    fn amplification_preserves_phase(exponent in 4u32..9, level in -1.5f32..1.5) {
        let np2 = 1usize << exponent;
        let mut spec = spectrum(np2, 41);
        let original = spec.clone();
        let (frequency, log_frequency, factors) = site_table(np2, level);
        amplify(spec.as_mut_slice(), &frequency, &log_frequency[..6], &factors[..6]);
        for i in 2..=np2 / 2 {
            prop_assume!(original[i].norm() > 1e-3);
            let before = original[i].im.atan2(original[i].re);
            let after = spec[i].im.atan2(spec[i].re);
            prop_assert!(
                angle_gap_deg(before.to_degrees(), after.to_degrees()) < 0.05,
                "bin {i} phase moved from {before} to {after}"
            );
        }
    }

    /// The output is Hermitian symmetric, so the inverse transform yields a real
    /// time series. The caller depends on this absolutely: the next thing that
    /// happens to this spectrum is an inverse transform whose imaginary part is
    /// discarded.
    #[test]
    fn amplified_spectrum_is_hermitian(exponent in 4u32..9, level in -1.5f32..1.5) {
        let np2 = 1usize << exponent;
        let mut spec = spectrum(np2, 53);
        let (frequency, log_frequency, factors) = site_table(np2, level);
        amplify(spec.as_mut_slice(), &frequency, &log_frequency[..6], &factors[..6]);
        // Bin `i` counted from DC, so `spec[i]` is the positive frequency and
        // `spec[np2 - i]` its Hermitian partner.
        for i in 1..np2 / 2 {
            let positive = spec[i];
            let negative = spec[np2 - i];
            prop_assert!(
                (positive.re - negative.re).abs() <= 1e-5 * positive.norm().max(1.0),
                "bin {i}: re {} vs {}", positive.re, negative.re
            );
            prop_assert!(
                (positive.im + negative.im).abs() <= 1e-5 * positive.norm().max(1.0),
                "bin {i}: im {} vs {}", positive.im, negative.im
            );
        }
    }
}

/// Resample the site table onto the spectrum's frequencies and apply the gain.
fn amplify(
    spectrum: &mut [Complex32],
    log_frequency_hz: &[f32],
    table_log_frequency: &[f32],
    factors: &[f32],
) {
    let mut gain = vec![0.0f32; spectrum.len() / 2 + 1];
    site_gain_curve(
        ArrayView1::from(log_frequency_hz),
        ArrayView1::from(table_log_frequency),
        ArrayView1::from(factors),
        ArrayViewMut1::from(gain.as_mut_slice()),
    );
    apply_site_amplification(spectrum, ArrayView1::from(gain.as_slice()));
}

/// A frequency axis and a flat site table at `level`, spanning the whole axis so no
/// bin falls outside the interpolation range.
fn site_table(np2: usize, level: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut frequency = vec![0.0; np2];
    for (bin, slot) in frequency.iter_mut().enumerate().take(np2 / 2 + 1) {
        *slot = bin as f32 * 0.5;
    }
    let mut log_frequency = vec![0.0; 20];
    let mut factors = vec![0.0; 20];
    for i in 0..6 {
        log_frequency[i] = (0.001f32 * 10f32.powi(i as i32)).ln();
        factors[i] = level;
    }
    (frequency, log_frequency, factors)
}

// ---------------------------------------------------------------------------
// Rupture-velocity taper
// ---------------------------------------------------------------------------

proptest! {
    /// The taper is continuous. It is built from four branches meeting at four
    /// depths, and a discontinuity there would put a step into the corner frequency
    /// of every subfault that happened to straddle a boundary.
    #[test]
    fn taper_is_continuous_across_its_band_edges(
        frac in 0.3f32..1.0, shallow in 0.2f32..1.0, deep in 0.2f32..1.0,
        hypocentre_km in 0.0f32..40.0,
    ) {
        let taper = RuptureVelocity {
            frac, shallow, deep, rv_sig1: 0.0,
        }
        .resolve(hypocentre_km);
        for edge in [
            taper.shallow_top_km,
            taper.shallow_base_km,
            taper.deep_top_km,
            taper.deep_base_km,
        ] {
            let step = 1e-3;
            let below = taper.factor(edge - step);
            let above = taper.factor(edge + step);
            // Slope is bounded by the widest band, so a genuine jump is far larger
            // than the change a small step can produce.
            prop_assert!(
                (above - below).abs() < 0.05 * frac,
                "jump at {edge} km: {below} -> {above}"
            );
        }
    }

    /// The factor is a fraction of the shear velocity, bracketed by the extremes of
    /// the three multipliers. Nothing downstream is prepared for a rupture faster
    /// than the shear wave.
    #[test]
    fn taper_stays_within_its_multipliers(
        frac in 0.3f32..1.0, shallow in 0.2f32..1.0, deep in 0.2f32..1.0,
        hypocentre_km in 0.0f32..40.0, depth_km in 0.0f32..120.0,
    ) {
        let taper = RuptureVelocity {
            frac, shallow, deep, rv_sig1: 0.0,
        }
        .resolve(hypocentre_km);
        let got = taper.factor(depth_km);
        let low = frac * shallow.min(deep).min(1.0);
        let high = frac * shallow.max(deep).max(1.0);
        prop_assert!(
            got >= low - 1e-6 && got <= high + 1e-6,
            "factor {got} outside [{low}, {high}] at {depth_km} km"
        );
        prop_assert!(got > 0.0, "factor {got} should be positive");
    }
}

// ---------------------------------------------------------------------------
// Slip model reader
// ---------------------------------------------------------------------------

/// Render a `.stoch` file for the given segment shapes, with slip `1.0` everywhere.
fn slip_model(segments: &[(usize, usize, f32, f32)]) -> StochModel {
    let built = segments
        .iter()
        .map(|&(along, down, length_km, width_km)| {
            Segment::builder()
                .fault_lon_deg(173.0)
                .fault_lat_deg(-43.0)
                .along_strike_count(along)
                .down_dip_count(down)
                .subfault_length_km(length_km)
                .subfault_width_km(width_km)
                .strike_deg(220.0)
                .dip_deg(60.0)
                .rake_deg(0.0)
                .top_depth_km(5.0)
                .hypocentre_along_strike_km(1.0)
                .hypocentre_down_dip_km(2.0)
                .subfaults(vec![
                    Subfault {
                        slip: Slip(1.0),
                        rise_time_s: 1.0,
                        rupture_time_s: 1.0
                    };
                    along * down
                ])
                .build()
        })
        .collect();
    StochModel::new(built)
}

proptest! {
    /// The totals the reader derives are exactly the sums over segments. These feed
    /// the moment normalisation and the stress-parameter adjustment, so getting them
    /// wrong rescales every waveform — quietly, because the shape stays plausible.
    #[test]
    fn derived_totals_are_the_sums_over_segments(
        shapes in proptest::collection::vec(
            (1usize..6, 1usize..5, 0.5f32..3.0, 0.5f32..3.0), 1..4,
        ),
    ) {
        let model = slip_model(&shapes);

        prop_assert_eq!(model.segments.len(), shapes.len());

        let want_subfaults: usize = shapes.iter().map(|&(a, d, _, _)| a * d).sum();
        prop_assert_eq!(model.subfault_count, want_subfaults);

        let want_area: f32 = shapes
            .iter()
            .map(|&(a, d, len, wid)| a as f32 * len * d as f32 * wid)
            .sum();
        prop_assert!(
            (model.fault_area_km2 - want_area).abs() <= 1e-3 * want_area,
            "area {} vs {want_area}",
            model.fault_area_km2
        );

        // Every segment's grid is the shape it was given, and the slip survives.
        for (segment, &(along, down, _, _)) in model.segments.iter().zip(shapes.iter()) {
            prop_assert_eq!(segment.along_strike_count, along);
            prop_assert_eq!(segment.down_dip_count, down);
            for (i, j) in segment.depth_major() {
                prop_assert_eq!(segment.at(i, j).slip, Slip(1.0));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// End to end — simulate
// ---------------------------------------------------------------------------

// Coarse end-to-end checks that a simulation runs and produces a sane record. None of
// them asserts a value.

/// A plausible layered velocity model: thin slow layers near the surface, thickening and
/// speeding up with depth, zero-thickness base as the reader expects.
fn velocity_model(layers: usize) -> hb_high::state::VelocityModelInput {
    let built: Vec<hb_high::state::InputLayer> = (0..layers)
        .map(|k| {
            let frac = k as f64 / (layers - 1) as f64;
            let vsh_km_s = 0.5 + 4.1 * frac;
            let qs = 50.0 + 150.0 * frac;
            hb_high::state::InputLayer {
                // Derived by build_velocity_model, which accumulates it down the column.
                depth_km: 0.0,
                thickness_km: if k == layers - 1 {
                    0.0
                } else {
                    (0.05 + 3.0 * frac) as f32
                },
                vp_km_s: vsh_km_s * 1.75,
                vsh_km_s,
                density_g_cm3: 1.81 + 1.5 * frac,
                attenuation_p: (2.0 * qs) as f32,
                attenuation_s: qs as f32,
            }
        })
        .collect();
    build_velocity_model(&built, 999.9).expect("valid velocity model")
}

/// Production-shaped configuration.
///
/// Deliberately *not* all production values: `czero`, the two taper multipliers and the
/// path-duration model differ, so that a property depending on a default rather than on the
/// configured value shows up here.
fn config() -> HfConfig {
    HfConfig {
        source: SourceParameters {
            stress_drop_bars: 50.0,
            czero: 2.1,
            calpha: 0.1,
            rupture_velocity: RuptureVelocity {
                frac: 0.8,
                shallow: 0.7,
                deep: 0.7,
                rv_sig1: 0.1,
            },
        },
        path: PathParameters {
            rayset: vec![RayType(1)],
            q_exponent: 0.6,
            path_duration: PathDurationModel::Bt2014Wus,
        },
        site: SiteParameters {
            kappa_s: 0.045,
            f_max_hz: 10.0,
        },
        record: RecordParameters {
            duration_s: 20.0,
            dt_s: 0.005,
        },
    }
}

/// [`config`] at a chosen record length, for the tests that vary it.
fn hf_config_with_duration(duration_s: f32) -> HfConfig {
    HfConfig {
        record: RecordParameters {
            duration_s,
            ..config().record
        },
        ..config()
    }
}

/// A simulator over the given inputs, for the tests that run more than one.
fn hf_simulator(
    config: &HfConfig,
    slip: &StochModel,
    vmod: &hb_high::state::VelocityModelInput,
) -> hb_high::sim::Simulator {
    hb_high::sim::Simulator::new(config, slip, vmod).expect("the fixture slip model is consistent")
}

/// Run one station through the whole simulation.
fn run(seed: u64) -> hb_high::sim::Simulation {
    let slip = slip_model(&[(4, 3, 1.5, 1.5)]);
    let vmod = velocity_model(20);
    let station = Station {
        longitude: 173.4,
        latitude: -43.1,
        name: "TEST".to_string(),
    };
    hb_high::sim::Simulator::new(&config(), &slip, &vmod)
        .expect("the fixture slip model is consistent")
        .run(station, seed)
}

#[test]
fn simulate_produces_a_record_of_the_requested_length() {
    let sim = run(123456789);
    // duration / dt, interleaved over three components.
    assert_eq!(sim.ndata, 4000, "ndata should be duration/dt");
    assert_eq!(
        sim.acc.dim(),
        (3, sim.ndata),
        "acc is one row per component"
    );
    assert_eq!(sim.dt, 0.005);
}

#[test]
fn simulate_produces_finite_non_zero_ground_motion() {
    let sim = run(123456789);
    assert!(
        sim.acc.iter().all(|v| v.is_finite()),
        "every sample must be finite"
    );
    let peak = sim.acc.iter().fold(0.0f32, |a, v| a.max(v.abs()));
    assert!(
        peak > 0.0,
        "the record is entirely zero -- nothing was simulated"
    );
    // Every component carries signal, which a mis-indexed accumulation could break for
    // one and not the others.
    for (component, trace) in sim.acc.rows().into_iter().enumerate() {
        let component_peak = trace.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            component_peak > 0.0,
            "component {component} is entirely zero"
        );
    }
}

#[test]
fn simulate_is_deterministic_and_seed_dependent() {
    let first = run(123456789);
    let again = run(123456789);
    assert_eq!(first.acc, again.acc, "same seed must give the same record");

    let other = run(987654321);
    assert_ne!(
        first.acc, other.acc,
        "a different seed must give a different record"
    );
}

/// A source below the whole velocity model must still produce a finite ray.
///
/// The case `ray::source_layer` returns `None` for. Reachable in production: truncating the
/// velocity model at the Moho can leave subfaults beneath the deepest layer.
#[test]
fn a_source_below_the_model_stays_finite() {
    let vmod = velocity_model(12);
    let model_bottom_km: f32 = vmod.iter().map(|l| l.thickness_km).sum();

    let mut state = hb_high::state::RayState::default();
    for depth_km in [model_bottom_km + 1.0, model_bottom_km * 2.0, 500.0] {
        let green = hb_high::ray::green_function(
            &mut state,
            &vmod
                .iter()
                .copied()
                .map(hb_high::state::Layer::from)
                .collect(),
            depth_km,
            60.0,
            1,
            hb_high::state::WaveMode::Sh,
        );
        for (name, value) in [
            ("rp0", green.rp0),
            ("stime", green.stime),
            ("rpath", green.rpath),
            ("qbar", green.qbar),
        ] {
            assert!(
                value.is_finite(),
                "source at {depth_km} km, below the {model_bottom_km} km model: {name} = {value}"
            );
        }
        assert!(
            green.stime > 0.0,
            "source at {depth_km} km: travel time {} should be positive",
            green.stime
        );
    }
}

/// A record too short to hold the arrivals says so.
///
/// Everything past `ndata` is discarded silently and mostly correctly — the envelope's
/// decayed tail routinely falls off the end. What must not be silent is the arrival *peak*
/// falling outside, which reads as a station that stopped shaking rather than a record that
/// ran out. That is reachable in production: the window has no upper cap, so a long path can
/// need a longer record than the domain asked for.
#[test]
fn a_record_too_short_for_the_arrivals_reports_clipping() {
    let slip = slip_model(&[(4, 3, 1.5, 1.5)]);
    let vmod = velocity_model(20);
    let station = Station {
        // Far enough that the S arrival is tens of seconds in.
        longitude: 176.0,
        latitude: -40.0,
        name: "FAR".to_string(),
    };

    let long_enough = hf_config_with_duration(300.0);
    let complete = hf_simulator(&long_enough, &slip, &vmod).run(station.clone(), 42);
    assert!(
        complete.clipping.is_complete(),
        "a 300 s record should hold this arrival, got {:?}",
        complete.clipping
    );

    let far_too_short = hf_config_with_duration(2.0);
    let cut = hf_simulator(&far_too_short, &slip, &vmod).run(station, 42);
    assert!(
        !cut.clipping.is_complete(),
        "a 2 s record cannot hold an arrival tens of seconds in, but reported no clipping"
    );
    assert!(
        cut.clipping.worst_overrun_s > 0.0,
        "clipping reported without an overrun: {:?}",
        cut.clipping
    );
}
