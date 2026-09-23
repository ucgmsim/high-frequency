//! The source: how strong each subfault is, and how fast the rupture reaches it.
//!
//! Everything here is a function of the slip model, the velocity model and the configuration
//! alone — nothing depends on where the station is — so [`crate::sim::Simulator::new`] computes
//! it once for a whole batch.
//!
//! * [`scale_source`] turns slip into per-subfault moment weights and derives the moment
//!   scaling of Graves & Pitarka (2010) eq. 12.
//! * [`RuptureVelocityTaper`] is the depth dependence of the rupture speed, `V_Ri` in eq. 13.
//! * [`SegmentAngles`] carries a segment's orientation and the part of eq. 13's corner
//!   frequency that is constant over it.

use crate::config::RuptureVelocity;
use crate::slip_model::{Segment, SlipModel};
use crate::velocity::{VelocityModel, layer_containing};

/// Bars·km³ to dyn·cm, the CGS moment unit Boore (1983) eq. 2 works in.
const BARS_KM3_TO_DYN_CM: f32 = 1.0e+21;

/// The subfault moments are relative, in units of `mu · area` with lengths in km; this brings
/// their sum to dyn·cm.
const RELATIVE_MOMENT_TO_DYN_CM: f32 = 1.0e+20;

/// Below this relative moment a subfault contributes nothing and is skipped outright. Applied
/// identically when counting subfaults for the normalisation and when walking them in the
/// subfault pass — the two must agree or `moment_scale`'s `N` counts subfaults that never
/// radiate.
pub const SUBFAULT_WEIGHT_THRESHOLD: MomentWeight = MomentWeight(0.001);

/// A subfault's share of the total moment, normalised so the mean over contributing
/// subfaults is one.
///
/// This is what a subfault's trace is weighted by when it is summed into the record. It is
/// derived from [`crate::slip_model::Slip`] but is not slip: the conversion multiplies by
/// rigidity and area, then rescales the whole model. A distinct type stops a caller weighting
/// by centimetres of slip.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct MomentWeight(pub f32);

/// One segment's moment weights, indexed by [`Segment::grid_index`].
pub type SegmentWeights = Vec<MomentWeight>;

/// The source's scale, derived from the slip model before any station is simulated.
#[derive(Clone, Copy, Debug)]
pub struct SourceScale {
    /// Average subfault dimension `dl = sqrt(length * width)`, averaged over segments, km.
    pub avg_subfault_km: f32,
    /// `σ_p · dl³` — the subfault moment scale, the denominator of Graves & Pitarka (2010)
    /// eq. 12's `F`, in dyn·cm.
    pub subevent_moment: f32,
    /// `F` in Graves & Pitarka (2010) eq. 12 — Frankel's (1995) finite-fault factor. It scales
    /// the subfault corner frequency towards the mainshock's while keeping the summed moment
    /// right; `crate::spectrum` is where it does its work, and the note on `frank` there shows
    /// the algebra.
    pub moment_scale: f32,
}

/// Convert slip to moment weights, and derive the source's scale from them.
///
/// The slip model is not modified: the weights come back in their own storage, one
/// [`SegmentWeights`] per segment, parallel to the subfault grid.
///
/// Three passes over the subfault grid, in this order:
///
/// 1. average subfault size
/// 2. slip → moment via the rigidity, accumulating the moment sum
/// 3. count the subfaults above [`SUBFAULT_WEIGHT_THRESHOLD`] and rescale so their mean
///    weight is 1
///
/// Pass 3's count is the `N` of Graves & Pitarka (2010) eq. 12 that reaches `moment_scale`.
pub fn scale_source(
    slip: &SlipModel,
    vmod: &VelocityModel,
    stress_drop_bars: f32,
) -> (SourceScale, Vec<SegmentWeights>) {
    let segment_count = slip.segments.len();

    // --- pass 1: average subfault size ----------------------------------------
    let avg_subfault_km = slip
        .segments
        .iter()
        .map(|s| (s.subfault_length_km * s.subfault_width_km).sqrt())
        .sum::<f32>()
        / segment_count as f32;

    // --- pass 2: relative slip to relative moment -----------------------------
    let mut relative_moment_sum = 0.0f32;
    let mut weights: Vec<SegmentWeights> = Vec::with_capacity(segment_count);

    for segment in &slip.segments {
        let mut segment_weights: SegmentWeights = Vec::with_capacity(segment.subfault_total());
        let row_depth_step_km = segment.subfault_width_km * segment.dip_deg.to_radians().sin();
        let top_depth_km = segment.top_depth_km;
        // The whole rigidity-area product is computed in f64 and narrows only at the end.
        // Narrowing earlier shifts every subfault moment by an ulp or two.
        let (length_km, width_km) = (
            segment.subfault_length_km as f64,
            segment.subfault_width_km as f64,
        );

        // Depth-major, which is storage order, so the sum accumulates without index
        // arithmetic. Rigidity is per depth row, not per subfault, so the row is the unit.
        for (row_index, row) in segment.depth_rows().enumerate() {
            let row_depth_km = top_depth_km + (row_index as f32 + 0.5) * row_depth_step_km;
            let layer = &vmod[layer_containing(vmod, row_depth_km)];
            // Rigidity times area: `mu = rho * beta^2`, so slip times this is moment.
            let rigidity_area =
                (layer.vsh_km_s * layer.vsh_km_s * layer.density_g_cm3 * length_km * width_km)
                    as f32;

            for subfault in row {
                let weight = MomentWeight(subfault.slip.0 * rigidity_area);
                relative_moment_sum += weight.0;
                segment_weights.push(weight);
            }
        }
        weights.push(segment_weights);
    }

    // `M_o` — total seismic moment, summed from the subfault moments.
    let total_moment_dyn_cm = RELATIVE_MOMENT_TO_DYN_CM * relative_moment_sum;

    // --- pass 3: normalise relative moments to average weight unity -----------
    // Depth-major again: `weights` was filled depth-major, so a flat walk of it keeps the
    // summation order.
    let mut weight_sum = 0.0f32;
    let mut radiating_subfault_count = 0usize;
    for weight in weights.iter().flatten() {
        if *weight > SUBFAULT_WEIGHT_THRESHOLD {
            weight_sum += weight.0;
            radiating_subfault_count += 1;
        }
    }
    let scale = radiating_subfault_count as f32 / weight_sum;
    for weight in weights.iter_mut().flatten() {
        weight.0 *= scale;
    }

    let subevent_moment =
        stress_drop_bars * avg_subfault_km * avg_subfault_km * avg_subfault_km * BARS_KM3_TO_DYN_CM;

    // This deviates from the published method: G&P define `F = M_o / (N * sigma_p * dl^3)`,
    // linear in subfault count, and this uses `sqrt(N)`. Neither G&P (2010) nor (2015)
    // licenses the square root. It is kept because production uses it and changing it
    // would move every waveform.
    //
    // The `1.0 *` forces the integer count through a real multiply before the sqrt.
    let moment_scale =
        total_moment_dyn_cm / (subevent_moment * (1.0 * radiating_subfault_count as f32).sqrt());

    (
        SourceScale {
            avg_subfault_km,
            subevent_moment,
            moment_scale,
        },
        weights,
    )
}

/// Top of the shallow weak zone, km. Graves & Pitarka (2010).
pub const SHALLOW_ZONE_TOP_KM: f32 = 5.0;
/// Bottom of the shallow weak zone, km — below this the taper is at the full fraction.
pub const SHALLOW_ZONE_BASE_KM: f32 = 8.0;
/// Top of the deep weak zone, km, when no hypocentre sits below it.
pub const DEEP_ZONE_TOP_KM: f32 = 15.0;
/// Bottom of the deep weak zone, km, in the same case.
pub const DEEP_ZONE_BASE_KM: f32 = 20.0;
/// How far the deep zone extends below the deepest hypocentre when it has to move down, km.
pub const DEEP_ZONE_THICKNESS_KM: f32 = 5.0;

/// Ceiling on the perturbed rupture-velocity factor, so the randomisation cannot drive the
/// rupture supershear. Not caller-settable.
pub const RUPTURE_VELOCITY_FRACTION_MAX: f32 = 1.4;

/// A [`RuptureVelocity`] with its transition depths fixed.
#[derive(Clone, Copy, Debug)]
pub struct RuptureVelocityTaper {
    pub velocity: RuptureVelocity,
    pub shallow_top_km: f32,
    pub shallow_base_km: f32,
    pub deep_top_km: f32,
    pub deep_base_km: f32,
}

impl RuptureVelocityTaper {
    /// Fix the transition depths against the deepest hypocentre in the slip model.
    ///
    /// When that hypocentre is above the default deep transition the defaults stand; otherwise
    /// the deep band moves down to start at the hypocentre, so the weak zone always sits below
    /// the nucleation point rather than cutting through it.
    pub fn new(velocity: RuptureVelocity, max_hypocentre_depth_km: f32) -> Self {
        let (deep_top_km, deep_base_km) = if max_hypocentre_depth_km > DEEP_ZONE_TOP_KM {
            (
                max_hypocentre_depth_km,
                max_hypocentre_depth_km + DEEP_ZONE_THICKNESS_KM,
            )
        } else {
            (DEEP_ZONE_TOP_KM, DEEP_ZONE_BASE_KM)
        };
        Self {
            velocity,
            shallow_top_km: SHALLOW_ZONE_TOP_KM,
            shallow_base_km: SHALLOW_ZONE_BASE_KM,
            deep_top_km,
            deep_base_km,
        }
    }

    /// The rupture-velocity factor at a given depth.
    ///
    /// Ramps up through the shallow band, sits at the base fraction in between, and ramps down
    /// through the deep band. Where the bands overlap the deep taper overrides the shallow one
    /// rather than blending with it. Overlap is possible because the deep band tracks the
    /// hypocentre.
    pub fn factor(&self, depth_km: f32) -> f32 {
        let Self {
            velocity:
                RuptureVelocity {
                    fraction,
                    shallow_factor,
                    deep_factor,
                    ..
                },
            shallow_top_km,
            shallow_base_km,
            deep_top_km,
            deep_base_km,
        } = *self;
        let shallow = if depth_km >= shallow_base_km {
            fraction
        } else if depth_km >= shallow_top_km {
            fraction
                * (shallow_factor
                    + (1.0 - shallow_factor) * (depth_km - shallow_top_km)
                        / (shallow_base_km - shallow_top_km))
        } else {
            fraction * shallow_factor
        };

        // The deep band overrides the shallow result; outside it the shallow value stands.
        if depth_km >= deep_base_km {
            fraction * deep_factor
        } else if depth_km >= deep_top_km {
            fraction
                * (1.0
                    + (deep_factor - 1.0) * (depth_km - deep_top_km) / (deep_base_km - deep_top_km))
        } else {
            shallow
        }
    }

    /// The factor at `depth_km`, randomly perturbed for one subfault.
    ///
    /// Draws exactly one normal deviate, and only when the sigma is non-zero — the
    /// perturbation and its draw are the same switch.
    pub fn perturbed_factor(&self, rng: &mut impl crate::rng::Draws, depth_km: f32) -> f32 {
        let base = self.factor(depth_km);
        let sigma = self.velocity.sigma;
        if sigma > 0.0 {
            (base * (rng.normal() * sigma).exp()).min(RUPTURE_VELOCITY_FRACTION_MAX)
        } else {
            base
        }
    }
}

/// Per-segment angles, plus the corner-frequency coefficient hoisted out of the subfault
/// loops.
pub struct SegmentAngles {
    pub strike_rad: f32,
    pub dip_rad: f32,
    pub rake_rad: f32,
    /// `c₀(1 + fcfac) / α_τ` — everything in Graves & Pitarka (2010) eq. 13's corner frequency
    /// that does not vary within a segment.
    ///
    /// `α_τ` is a pure function of per-segment constants, so evaluating it once per segment
    /// gives the same `f32` as evaluating it per subfault.
    pub corner_coeff: f32,
}

impl SegmentAngles {
    pub fn for_segment(
        segment: &Segment,
        corner_frequency_alpha: f32,
        corner_frequency_constant: f32,
    ) -> Self {
        Self {
            strike_rad: segment.strike_deg.to_radians(),
            dip_rad: segment.dip_deg.to_radians(),
            rake_rad: segment.rake_deg.to_radians(),
            corner_coeff: corner_frequency_constant
                / alpha_t(segment.dip_deg, segment.rake_deg, corner_frequency_alpha),
        }
    }
}

/// `α_τ`, the dip-and-rake corner-frequency and rise-time adjustment.
///
/// Graves & Pitarka (2015) eq. 3, `α_T = 1 + F_D·F_R·c_α`. Returns the reciprocal of that,
/// because the caller divides `c₀` by it and eq. 13 has `α_τ` in the denominator — so the value
/// returned here is `α_τ` itself, ≤ 1, and smaller for a shallow-dipping thrust. Physically that
/// means such a fault gets a *higher* corner frequency and a shorter rise time, which is the
/// observed trend (Graves & Pitarka 2010, p. 2099, citing Somerville 1998: shorter rise times for
/// thrust events imply relatively high dynamic stress drops).
///
/// Graves & Pitarka (2010) eq. 9 parameterised this on dip alone, piecewise (1 above 60°,
/// 0.82 below 45°). The continuous dip-and-rake form below is the 2015 revision.
///
/// `fD` tapers with dip above 45 degrees; `fR` peaks at a rake of 90 degrees. The rake is first
/// wrapped into `[-180, 180]`.
fn alpha_t(dip_deg: f32, rake_deg: f32, corner_frequency_alpha: f32) -> f32 {
    let dip_factor = if (45.0..=90.0).contains(&dip_deg) {
        1.0 - (dip_deg - 45.0) / 45.0
    } else if (0.0..=45.0).contains(&dip_deg) {
        1.0
    } else {
        0.0
    };

    // Wrapped into [-180, 180]: `rem_euclid` folds into [0, 360), then the top half shifts down.
    let wrapped_rake_deg = {
        let folded = rake_deg.rem_euclid(360.0);
        if folded > 180.0 {
            folded - 360.0
        } else {
            folded
        }
    };

    let rake_factor = if (0.0..=180.0).contains(&wrapped_rake_deg) {
        1.0 - (wrapped_rake_deg - 90.0).abs() / 90.0
    } else {
        0.0
    };

    1.0 / (1.0 + dip_factor * rake_factor * corner_frequency_alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `alpha_t`'s `rem_euclid` wrap agrees, to the bit, with stepping by 360 in a loop.
    ///
    /// The two differ at the boundary (`-180` wraps to `+180`), but the wrapped rake is only
    /// used as `1 - |rake - 90|/90` inside `0..=180`: `-180` falls outside and contributes 0,
    /// and `+180` falls inside and computes `1 - 1 = 0`.
    #[test]
    fn rake_wrapping_agrees_with_the_stepping_loop_it_replaced() {
        fn stepped(mut rake_deg: f32) -> f32 {
            while rake_deg < -180.0 {
                rake_deg += 360.0;
            }
            while rake_deg > 180.0 {
                rake_deg -= 360.0;
            }
            if (0.0..=180.0).contains(&rake_deg) {
                1.0 - (rake_deg - 90.0).abs() / 90.0
            } else {
                0.0
            }
        }
        fn wrapped(rake_deg: f32) -> f32 {
            let folded = rake_deg.rem_euclid(360.0);
            let rake_deg = if folded > 180.0 {
                folded - 360.0
            } else {
                folded
            };
            if (0.0..=180.0).contains(&rake_deg) {
                1.0 - (rake_deg - 90.0).abs() / 90.0
            } else {
                0.0
            }
        }
        // Every boundary and every multiple of 90 across four turns.
        for step in -720i32..=720 {
            let rake_deg = step as f32;
            assert_eq!(
                stepped(rake_deg).to_bits(),
                wrapped(rake_deg).to_bits(),
                "rake {rake_deg} deg"
            );
        }
    }

    /// The production taper values. Defaults are declared in Python, so the test states them
    /// explicitly.
    fn production_rupture_velocity() -> RuptureVelocity {
        RuptureVelocity {
            fraction: 0.8,
            shallow_factor: 0.6,
            deep_factor: 0.6,
            sigma: 0.1,
        }
    }

    #[test]
    fn deep_transition_tracks_the_deepest_hypocentre() {
        // Shallower than the default: the fixed depths stand.
        let taper = RuptureVelocityTaper::new(production_rupture_velocity(), 3.0);
        assert_eq!(
            (taper.deep_top_km, taper.deep_base_km),
            (DEEP_ZONE_TOP_KM, DEEP_ZONE_BASE_KM)
        );
        // Deeper: the band moves down and spans DEEP_ZONE_THICKNESS_KM.
        let taper = RuptureVelocityTaper::new(production_rupture_velocity(), 22.0);
        assert_eq!((taper.deep_top_km, taper.deep_base_km), (22.0, 27.0));
        assert_eq!(taper.velocity.fraction, 0.8);
    }

    #[test]
    fn taper_is_flat_between_the_two_weak_zones() {
        let taper = RuptureVelocityTaper::new(production_rupture_velocity(), 3.0);
        // Between SHALLOW_ZONE_BASE_KM and DEEP_ZONE_TOP_KM the factor is the base fraction.
        assert_eq!(taper.factor(10.0), 0.8);
        // Above the shallow zone and below the deep one it is reduced by the two multipliers.
        assert_eq!(taper.factor(0.0), 0.8 * 0.6);
        assert_eq!(taper.factor(30.0), 0.8 * 0.6);
    }
}
