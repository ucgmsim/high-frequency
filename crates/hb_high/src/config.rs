//! The simulation configuration, as a typed value rather than a positional deck.
//!
//! Every field here is supplied by the caller. There are no defaults and no `Option`s to
//! resolve: the Python dataclasses in `hf_simulation` are the single source of truth for
//! what a caller may leave unset, so a default cannot be declared in two places and drift.
//!
//! The constants that remain below are *not* defaults. They are fixed parameters of the
//! models themselves, which no caller has ever been able to set.

/// One requested ray path.
///
/// The integer is kept because the ray tracer needs it, but the two things the program actually
/// asks about it are methods.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RayType(pub i32);

/// How a [`RayType`] is interpreted.
///
/// `0` is not a ray at all: it selects a straight-line geometric path instead of a traced one.
/// For traced rays the **parity** picks the take-off direction. These are the `j = 1, M` rays
/// summed over in Graves & Pitarka (2010) eq. 10 — direct, Moho-reflected, and multiples.
///
/// Production runs `rayset = [1]`, so only `Upgoing` is exercised there.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RayKind {
    /// `0` — straight-line path, no ray tracing used.
    StraightRay,
    /// Odd — leaves the source upward, so the incidence angle is supplemented.
    Upgoing,
    /// Even and non-zero — leaves the source downward.
    Downgoing,
}

impl RayType {
    pub fn kind(self) -> RayKind {
        if self.0 == 0 {
            RayKind::StraightRay
        } else if self.0 % 2 == 1 {
            RayKind::Upgoing
        } else {
            RayKind::Downgoing
        }
    }

    /// The `itype` to trace with. A straight ray is still traced, as type 1, because the tracer
    /// runs unconditionally and the straight-line results overwrite its output afterwards.
    pub fn trace_type(self) -> i32 {
        match self.kind() {
            RayKind::StraightRay => 1,
            _ => self.0,
        }
    }
}

/// The path-duration model: how record duration grows with distance.
///
/// This is the `c₁·R` term of Graves & Pitarka (2010) eq. 17, `T_di = f_ci⁻¹ + c₁R_i`,
/// generalised to a piecewise-linear table so that the Boore & Thompson models can be selected
/// instead. See `PHYSICS.md` §7 and `papers/README.md` for what has been checked against what.
///
/// # Sources
///
/// * Graves, R. W. & Pitarka, A. (2010). Broadband ground-motion simulation using a hybrid
///   approach. *BSSA* **100**(5A), 2095–2123. doi:10.1785/0120100057 — eq. 17.
/// * Boore, D. M. & Thompson, E. M. (2014). Path durations for use in the stochastic-method
///   simulation of ground motions. *BSSA* **104**(5), 2541–2552. doi:10.1785/0120140058 —
///   Table 1, "The New Path Duration Model".
/// * Boore, D. M. & Thompson, E. M. (2015). Revisions to some parameters used in
///   stochastic-method simulations of ground motion. *BSSA* **105**(2A), 1029–1041.
///   doi:10.1785/0120140281 — Table 3, "The Path Duration Model for Stable Continental
///   Regions".
///
/// The wire encoding is a non-contiguous integer set (`0`/`1`/`2`/`11`/`12`) where every other
/// value left the table uninitialised in the original. An enum makes that unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathDurationModel {
    /// `<= 0` — Graves & Pitarka (2010) eq. 17, single segment, slope `c₁ = 0.063` s/km.
    ///
    /// Verified exact against the paper.
    Gp2010,
    /// `1` — western US, slope 0.070 s/km.
    ///
    /// **No published source.** The deck documents models 1 and 2 as "WUS/ENA modification
    /// trial/error": they are the Graves & Pitarka slope adjusted by hand, not a model from
    /// the literature, and nothing here should imply otherwise.
    Wus,
    /// `2` — eastern North America, slope 0.100 s/km. Hand-adjusted like [`Self::Wus`], with
    /// no published source.
    Ena,
    /// `11` — Boore & Thompson (2014) Table 1, active crustal regions.
    ///
    /// Breakpoints at 0, 7, 45, 125, 175 and 270 km, with durations 0, 2.4, 8.4, 10.9, 17.4
    /// and 34.2 s, linearly interpolated between — **verified against Table 1**, which
    /// specifies exactly that interpolation.
    ///
    /// **The extrapolation beyond 270 km does not match the paper**, which gives a tail slope
    /// of 0.156 s/km against the 0.177 the table builder repeats — about 13% steeper.
    ///
    /// This is the model production runs, and 270 km is well inside its working range: an
    /// Alpine Fault rupture recorded in the lower North Island has *every* subfault past that
    /// distance. See finding 7 in `papers/README.md` for the measured effect per station.
    Bt2014Wus,
    /// `12` — Boore & Thompson (2015) Table 3, stable continental regions.
    ///
    /// Breakpoints at 0, 15, 35, 50, 125, 200, 392 and 600 km, with durations 0, 2.6, 17.5,
    /// 25.1, 25.1, 28.5, 46.0 and 69.1 s — **verified against Table 3**, including its
    /// "linear interpolation … not logarithms" and its `D_P(R) = D_P(R_last) + 0.111(R −
    /// R_last)` tail.
    ///
    /// Unlike model 11 the tail is right, and by luck rather than design: the paper's 0.111
    /// s/km happens to equal the final tabulated segment's slope, which is what the table
    /// builder repeats.
    Bt2015Ena,
}

impl PathDurationModel {
    /// Decode the wire integer. Everything `<= 0` maps to `Gp2010`.
    pub fn from_deck(v: i32) -> Option<Self> {
        Some(match v {
            i32::MIN..=0 => Self::Gp2010,
            1 => Self::Wus,
            2 => Self::Ena,
            11 => Self::Bt2014Wus,
            12 => Self::Bt2015Ena,
            _ => return None,
        })
    }
}

/// The depth-dependent rupture-velocity taper.
///
/// Rupture propagates more slowly near the surface. Graves & Pitarka (2010) attribute this to
/// unconsolidated material and low effective friction at shallow depth, and model it as a
/// "shallow weak zone" above 5 km. Graves & Pitarka (2015) added an analogous **deep** weak
/// zone, which is why there are two bands here.
///
/// This feeds `V_Ri` in eq. 13's corner frequency, so a slower rupture means a lower corner
/// frequency and less high-frequency energy.
///
/// The transition depths are not settable — see [`SHALLOW_ZONE_TOP_KM`] and friends — but the
/// three reduction factors are, and the caller's values are typically locally calibrated rather
/// than the published ones. See the Python dataclass for what the defaults are and how they
/// differ from Graves & Pitarka.
#[derive(Clone, Copy, Debug)]
pub struct RuptureVelocity {
    /// Base rupture speed as a fraction of the local shear-wave velocity.
    pub frac: f32,
    /// Multiplier at the shallow end of the taper.
    pub shallow: f32,
    /// Multiplier at the deep end of the taper.
    pub deep: f32,
    /// Rupture-velocity randomisation sigma, log-normal. **0.1 in production**, so unlike the
    /// two multipliers above this path is live: it perturbs the factor per subfault, capped by
    /// [`RUPTURE_VELOCITY_FRACTION_MAX`].
    pub rv_sig1: f32,
}

/// Top of the shallow weak zone, km. Graves & Pitarka (2010).
pub const SHALLOW_ZONE_TOP_KM: f32 = 5.0;
/// Bottom of the shallow weak zone, km — below this the taper is at full `frac`.
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
    /// Base fraction of the local shear-wave velocity.
    pub frac: f32,
    /// Multiplier applied at and above [`RuptureVelocityTaper::shallow_top_km`].
    pub shallow_factor: f32,
    /// Multiplier applied at and below [`RuptureVelocityTaper::deep_base_km`].
    pub deep_factor: f32,
    pub shallow_top_km: f32,
    pub shallow_base_km: f32,
    pub deep_top_km: f32,
    pub deep_base_km: f32,
}

impl RuptureVelocity {
    /// Fix the transition depths against the deepest hypocentre in the slip model.
    ///
    /// When that hypocentre is above the default deep transition the defaults stand; otherwise
    /// the deep band moves down to start at the hypocentre, so the weak zone always sits below
    /// the nucleation point rather than cutting through it.
    pub fn resolve(self, max_hypocentre_depth_km: f32) -> RuptureVelocityTaper {
        let (deep_top_km, deep_base_km) = if max_hypocentre_depth_km > DEEP_ZONE_TOP_KM {
            (
                max_hypocentre_depth_km,
                max_hypocentre_depth_km + DEEP_ZONE_THICKNESS_KM,
            )
        } else {
            (DEEP_ZONE_TOP_KM, DEEP_ZONE_BASE_KM)
        };
        RuptureVelocityTaper {
            frac: self.frac,
            shallow_factor: self.shallow,
            deep_factor: self.deep,
            shallow_top_km: SHALLOW_ZONE_TOP_KM,
            shallow_base_km: SHALLOW_ZONE_BASE_KM,
            deep_top_km,
            deep_base_km,
        }
    }
}

impl RuptureVelocityTaper {
    /// The rupture-velocity factor at a given depth.
    ///
    /// Ramps up through the shallow band, sits at `frac` in between, and ramps down through the
    /// deep band. **The deep taper OVERWRITES the shallow one where the bands overlap** — it is
    /// not a blend of the two, and the `if`/`else if` structure below is what makes that
    /// explicit. Overlap is possible because the deep band tracks the hypocentre.
    pub fn factor(&self, depth_km: f32) -> f32 {
        let Self {
            frac,
            shallow_factor,
            deep_factor,
            shallow_top_km,
            shallow_base_km,
            deep_top_km,
            deep_base_km,
        } = *self;
        let shallow = if depth_km >= shallow_base_km {
            frac
        } else if depth_km >= shallow_top_km {
            frac * (shallow_factor
                + (1.0 - shallow_factor) * (depth_km - shallow_top_km)
                    / (shallow_base_km - shallow_top_km))
        } else {
            frac * shallow_factor
        };

        // The deep band OVERRIDES the shallow result rather than blending with it, which is
        // what the two-stage structure says: outside the deep band the shallow value stands.
        if depth_km >= deep_base_km {
            frac * deep_factor
        } else if depth_km >= deep_top_km {
            frac * (1.0
                + (deep_factor - 1.0) * (depth_km - deep_top_km) / (deep_base_km - deep_top_km))
        } else {
            shallow
        }
    }
}

/// The earthquake source: how strong the high-frequency radiation is and how fast the rupture
/// travels.
#[derive(Clone, Debug)]
pub struct SourceParameters {
    /// `Δσ` — the Brune stress parameter, bars. Graves & Pitarka use 50 bars.
    ///
    /// Boore (1983) is worth quoting on what this is: it "is best thought of here as simply a
    /// parameter controlling the strength of the high-frequency radiation, not as a measured
    /// static stress drop".
    pub stress_drop_bars: f32,
    /// `c₀`, the corner-frequency constant of Graves & Pitarka (2010) eq. 13 / (2015) eq. 1.
    pub czero: f32,
    /// `c_α`, the coefficient of the `α_τ` dip-and-rake adjustment.
    pub calpha: f32,
    pub rupture_velocity: RuptureVelocity,
}

/// The path from source to site: which rays, and how the medium attenuates along them.
#[derive(Clone, Debug)]
pub struct PathParameters {
    /// Which ray paths to sum over. Production is `[RayType(1)]`.
    pub rayset: Vec<RayType>,
    /// `x` in `Q(f) = Q₀·f^x`, the frequency exponent of the quality factor.
    pub q_exponent: f32,
    pub path_duration: PathDurationModel,
}

/// The near-surface: what happens in the last few hundred metres.
///
/// Quarter-wavelength site amplification ([`crate::site`]) is **not** a field here. It used
/// to be a `bool`, and there is no run for which it should be off, so it is applied
/// unconditionally rather than offered as a choice that only has one right answer.
#[derive(Clone, Debug)]
pub struct SiteParameters {
    /// `κ` — near-surface attenuation, seconds. Anderson & Hough (1984). Production uses 0.045.
    pub kappa_s: f32,
    /// `f_max` — the high-cut corner, Hz. See `PHYSICS.md` §3 on the `f_max`-versus-`κ`
    /// question; both parameters exist because both physical interpretations do.
    pub f_max_hz: f32,
}

/// The shape of the record to produce.
#[derive(Clone, Debug)]
pub struct RecordParameters {
    /// Record length, seconds.
    pub duration_s: f32,
    /// Sample interval, seconds.
    pub dt_s: f32,
}

/// Everything needed to simulate, with nothing about where the inputs came from or where the
/// output goes.
///
/// Nothing per-station lives here — not the seed, not the location — because one of these
/// drives a whole batch of stations through [`crate::sim::Simulator`].
#[derive(Clone, Debug)]
pub struct HfConfig {
    pub source: SourceParameters,
    pub path: PathParameters,
    pub site: SiteParameters,
    pub record: RecordParameters,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_kind_splits_on_zero_then_parity() {
        assert_eq!(RayType(0).kind(), RayKind::StraightRay);
        assert_eq!(RayType(1).kind(), RayKind::Upgoing);
        assert_eq!(RayType(3).kind(), RayKind::Upgoing);
        assert_eq!(RayType(2).kind(), RayKind::Downgoing);
        assert_eq!(RayType(4).kind(), RayKind::Downgoing);
        // A straight ray is still traced, as type 1.
        assert_eq!(RayType(0).trace_type(), 1);
        assert_eq!(RayType(3).trace_type(), 3);
    }

    #[test]
    fn path_duration_accepts_only_the_five_documented_values() {
        assert_eq!(
            PathDurationModel::from_deck(0),
            Some(PathDurationModel::Gp2010)
        );
        assert_eq!(
            PathDurationModel::from_deck(-7),
            Some(PathDurationModel::Gp2010)
        );
        assert_eq!(
            PathDurationModel::from_deck(11),
            Some(PathDurationModel::Bt2014Wus)
        );
        // The gaps are the point: these values left the duration table uninitialised.
        for bad in [3, 5, 10, 13, 99] {
            assert_eq!(
                PathDurationModel::from_deck(bad),
                None,
                "{bad} should be rejected"
            );
        }
    }

    /// The taper values here are the production ones; they are declared in Python now, so this
    /// test states them explicitly rather than reaching for a constant.
    fn production_rupture_velocity() -> RuptureVelocity {
        RuptureVelocity {
            frac: 0.8,
            shallow: 0.6,
            deep: 0.6,
            rv_sig1: 0.1,
        }
    }

    #[test]
    fn deep_transition_tracks_the_deepest_hypocentre() {
        // Shallower than the default: the fixed depths stand.
        let taper = production_rupture_velocity().resolve(3.0);
        assert_eq!(
            (taper.deep_top_km, taper.deep_base_km),
            (DEEP_ZONE_TOP_KM, DEEP_ZONE_BASE_KM)
        );
        // Deeper: the band moves down and spans DEEP_ZONE_THICKNESS_KM.
        let taper = production_rupture_velocity().resolve(22.0);
        assert_eq!((taper.deep_top_km, taper.deep_base_km), (22.0, 27.0));
        assert_eq!(taper.frac, 0.8);
    }

    #[test]
    fn taper_is_flat_between_the_two_weak_zones() {
        let taper = production_rupture_velocity().resolve(3.0);
        // Between SHALLOW_ZONE_BASE_KM and DEEP_ZONE_TOP_KM the factor is the base fraction.
        assert_eq!(taper.factor(10.0), 0.8);
        // Above the shallow zone and below the deep one it is reduced by the two multipliers.
        assert_eq!(taper.factor(0.0), 0.8 * 0.6);
        assert_eq!(taper.factor(30.0), 0.8 * 0.6);
    }
}
