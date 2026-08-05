//! The simulation configuration, as a typed value rather than a positional deck.
//!
//! Every field is named, every magic integer is an enum, and every "use the default" sentinel
//! is an `Option` with a single accessor that resolves it. Python builds one of these directly.
//!
//! # Why the types matter here more than usual
//!
//! The original was driven by a 22-line list-directed deck on stdin, where `read(5,*)` spans
//! record boundaries — so a missing or extra token silently rebinds every field after it, with
//! no error and no way to notice. **Production had a live bug of exactly that shape**, which
//! left the stress-parameter adjustment permanently disabled (see [`StressParamAdjust`]). That
//! class of bug is unrepresentable now, which is the point.

/// Pi, and the degrees-to-radians factor derived from it.
///
/// Both the slip-model reader and the simulation convert degrees, so this lives in one place to
/// keep them from drifting. The original carried an 8-digit truncation, wrong by about one `f32`
/// ulp; this is the correctly rounded value.
pub const PI: f32 = std::f32::consts::PI;

/// Degrees to radians, from the same literal.
pub const DEG_TO_RAD: f32 = PI / 180.0;

/// Built-in defaults for the fields the caller can leave unset.
///
/// Most of these are Graves & Pitarka values; see the note on [`RuptureVelocity`] for the ones
/// that differ from the published papers, and `papers/README.md` finding 5 for `CZERO`.
pub mod defaults {
    /// `c₀` in Graves & Pitarka (2010) eq. 13 / (2015) eq. 1, the corner-frequency constant.
    ///
    /// **2.0 is the 2015 value; Graves & Pitarka (2010) used 2.1.** This is a version marker:
    /// the code tracks the later parameterisation. See `papers/README.md` finding 5.
    pub const CZERO: f32 = 2.0;
    /// Rupture speed as a fraction of the local shear-wave velocity. Graves & Pitarka (2010)
    /// set "the average rupture speed at 80% of the local shear-wave velocity".
    pub const RVFAC: f32 = 0.8;
    pub const SHAL_RVFAC: f32 = 0.6;
    /// The shallow weak zone starts at 5 km, matching Graves & Pitarka (2010).
    pub const SHAL_DMIN: f32 = 5.0;
    pub const SHAL_DMAX: f32 = 8.0;
    pub const DEEP_RVFAC: f32 = 0.6;
    pub const DEEP_DMIN: f32 = 15.0;
    pub const DEEP_DMAX: f32 = 20.0;
    /// `c_α`, the coefficient of the dip-and-rake corner-frequency adjustment `α_τ`.
    pub const CALPHA: f32 = 0.1;
    /// Never supplied by any caller; hardwired to zero. Retained as a named quantity because it
    /// appears in the corner-frequency expression `c₀(1 + fcfac)`.
    pub const FCFAC: f32 = 0.0;
    /// `vsmoho` when left non-positive: high enough that no layer reaches it, i.e. "do not
    /// truncate at the Moho".
    pub const VS_MOHO: f64 = 999.9;
    /// Ceiling on the perturbed rupture-velocity factor, so the randomisation in
    /// [`crate::sim`] cannot drive the rupture supershear.
    pub const RVFMAX: f32 = 1.4;
}

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
/// instead. See `PHYSICS.md` §7.
///
/// The wire encoding is a non-contiguous integer set (`0`/`1`/`2`/`11`/`12`) where every other
/// value left the table uninitialised in the original. An enum makes that unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathDurationModel {
    /// `<= 0` — Graves & Pitarka (2010) eq. 17, single segment, slope `c₁ = 0.063` s/km.
    Gp2010,
    /// `1` — western US, slope 0.070.
    Wus,
    /// `2` — eastern North America, slope 0.100.
    Ena,
    /// `11` — Boore & Thompson (2014) Table 1, active crustal regions. Breakpoints at 0, 7, 45,
    /// 125, 175 and 270 km. **See the note in [`crate::sim`] on the extrapolation beyond
    /// 270 km, which does not match the paper.**
    Bt2014Wus,
    /// `12` — Boore & Thompson (2015), stable continental regions. Eight breakpoints.
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

/// Stress-parameter adjustment towards a target magnitude, using the Leonard (2010)
/// magnitude–area scaling relations.
///
/// # This path has never run in production
///
/// Not by configuration, but by accident: the original's deck misalignment meant the selector
/// was always read as the literal `0`, so the adjustment factor was always 1.0 regardless of
/// what the workflow config asked for. **The typed interface here fixes that**, which means the
/// branch is now reachable for the first time and has correspondingly little field exposure.
/// Treat a non-`None` value as untested rather than as supported.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StressParamAdjust {
    /// `0` or anything unrecognised — no adjustment, factor 1.
    None,
    /// `1` — Leonard (2010) scaling for active tectonic regions.
    LeonardActive,
    /// `2` — Leonard (2010) scaling for stable continental regions.
    LeonardStable,
}

impl StressParamAdjust {
    /// Anything other than 1 or 2 means no adjustment, which is what made the misaligned deck
    /// silently harmless rather than an error.
    pub fn from_deck(v: i32) -> Self {
        match v {
            1 => Self::LeonardActive,
            2 => Self::LeonardStable,
            _ => Self::None,
        }
    }
}

/// The depth-dependent rupture-velocity taper.
///
/// Rupture propagates more slowly near the surface — Graves & Pitarka (2010) attribute this to
/// unconsolidated material and low effective friction at shallow depth, and model it as a
/// "shallow weak zone" above 5 km. Graves & Pitarka (2015) added an analogous **deep** weak
/// zone, which is why there are two bands here.
///
/// This feeds `V_Ri` in eq. 13's corner frequency, so a slower rupture means a lower corner
/// frequency and less high-frequency energy.
///
/// # The default reduction factors differ from the published values
///
/// Graves & Pitarka (2010) give a 70% factor for the shallow zone and (2015) a 30% reduction
/// for the deep one; the defaults here are 0.6 for both. These are **overridable**, so a caller
/// may well be supplying locally calibrated values — but if you are reading the defaults as
/// "the paper's numbers", they are not. Worth checking against whatever calibration this
/// deployment intends.
///
/// The deep transition depths are **not** constants: they track the deepest hypocentre in the
/// slip model, so [`RuptureVelocity::resolve`] takes it as an argument.
#[derive(Clone, Copy, Debug)]
pub struct RuptureVelocity {
    /// `rvfac` — base fraction of the shear velocity.
    pub frac: Option<f32>,
    /// `shal_rvfac` — multiplier at the shallow end.
    pub shallow: Option<f32>,
    /// `deep_rvfac` — multiplier at the deep end.
    pub deep: Option<f32>,
}

/// A [`RuptureVelocity`] with its defaults applied and its transition depths fixed.
#[derive(Clone, Copy, Debug)]
pub struct RuptureVelocityTaper {
    pub rvfac: f32,
    pub shal_rvfac: f32,
    pub deep_rvfac: f32,
    pub shal_dmin: f32,
    pub shal_dmax: f32,
    pub deep_dmin: f32,
    pub deep_dmax: f32,
}

impl RuptureVelocity {
    /// Apply the defaults and set the transition depths.
    ///
    /// When the deepest hypocentre is above the default deep transition the defaults stand;
    /// otherwise the deep band moves down to start at that hypocentre and span 5 km, so the
    /// weak zone always sits below the nucleation point rather than cutting through it.
    pub fn resolve(self, max_hypocentre_depth_km: f32) -> RuptureVelocityTaper {
        let (deep_dmin, deep_dmax) = if max_hypocentre_depth_km > defaults::DEEP_DMIN {
            (max_hypocentre_depth_km, max_hypocentre_depth_km + 5.0)
        } else {
            (defaults::DEEP_DMIN, defaults::DEEP_DMAX)
        };
        RuptureVelocityTaper {
            rvfac: self.frac.unwrap_or(defaults::RVFAC),
            shal_rvfac: self.shallow.unwrap_or(defaults::SHAL_RVFAC),
            deep_rvfac: self.deep.unwrap_or(defaults::DEEP_RVFAC),
            shal_dmin: defaults::SHAL_DMIN,
            shal_dmax: defaults::SHAL_DMAX,
            deep_dmin,
            deep_dmax,
        }
    }
}

impl RuptureVelocityTaper {
    /// The rupture-velocity factor at a given depth.
    ///
    /// Ramps up through the shallow band, sits at `rvfac` in between, and ramps down through the
    /// deep band. **The deep taper OVERWRITES the shallow one where the bands overlap** — it is
    /// not a blend of the two, and the `if`/`else if` structure below is what makes that
    /// explicit. Overlap is possible because the deep band tracks the hypocentre.
    pub fn factor(&self, zdep: f32) -> f32 {
        let Self { rvfac, shal_rvfac, deep_rvfac, shal_dmin, shal_dmax, deep_dmin, deep_dmax } =
            *self;
        let mut rvf = rvfac * shal_rvfac;
        if zdep >= shal_dmin && zdep < shal_dmax {
            rvf = rvfac
                * (shal_rvfac + (1.0 - shal_rvfac) * (zdep - shal_dmin) / (shal_dmax - shal_dmin));
        } else if zdep >= shal_dmax {
            rvf = rvfac;
        }
        if zdep >= deep_dmin && zdep < deep_dmax {
            rvf = rvfac * (1.0 + (deep_rvfac - 1.0) * (zdep - deep_dmin) / (deep_dmax - deep_dmin));
        } else if zdep >= deep_dmax {
            rvf = rvfac * deep_rvfac;
        }
        rvf
    }
}

/// Everything needed to simulate one station, with nothing about where the inputs came from or
/// where the output goes.
#[derive(Clone, Debug)]
pub struct HfConfig {
    /// `Δσ` — the Brune stress parameter, bars. Graves & Pitarka use 50 bars.
    ///
    /// Boore (1983) is worth quoting on what this is: it "is best thought of here as simply a
    /// parameter controlling the strength of the high-frequency radiation", not as a measured
    /// static stress drop.
    pub stress_drop: f32,
    /// Which ray paths to sum over. Production is `[RayType(1)]`.
    pub rayset: Vec<RayType>,
    /// Whether to apply quarter-wavelength site amplification ([`crate::site`]).
    pub site_amp: bool,
    /// This station's seed, and the whole of its identity as far as the generator is
    /// concerned.
    ///
    /// **Per-station, not per-run.** Each station gets an independent PCG stream via
    /// [`crate::rng::DrawSource::for_station`], which is what makes a batch of stations safe to
    /// reorder, subset or resume. The original shared one stream across its station loop, so a
    /// multi-station run was not a concatenation of single-station runs.
    pub seed: u64,
    /// Record length, seconds.
    pub duration: f32,
    /// Sample interval, seconds.
    pub dt: f32,
    /// `f_max` — the high-cut corner, Hz. See `PHYSICS.md` §3 on the `f_max`-versus-`κ`
    /// question; both parameters exist because both physical interpretations do.
    pub fmax: f32,
    /// `κ` — near-surface attenuation, seconds. Anderson & Hough (1984). Production uses 0.045.
    pub kappa: f32,
    /// `x` in `Q(f) = Q₀·f^x`, the frequency exponent of the quality factor.
    pub qfexp: f32,
    pub rupture_velocity: RuptureVelocity,
    /// `c₀`, the corner-frequency constant of eq. 13. `None` uses [`defaults::CZERO`].
    pub czero: Option<f32>,
    /// `c_α`, the coefficient of the `α_τ` dip-and-rake adjustment.
    pub calpha: Option<f32>,
    /// `M₀` — total seismic moment. `None` derives it from the slip model.
    pub moment: Option<f32>,
    /// A constant rupture velocity, overriding the slip model's rupture times. `None` takes
    /// those times from the slip model instead, which is what production does.
    pub rupture_velocity_override: Option<f32>,
    /// Shear velocity at which to truncate the velocity model at the Moho.
    pub vs_moho: Option<f64>,
    /// Negative means the velocity model is used unperturbed. A non-negative value would route
    /// through a velocity-randomisation path that is not implemented here.
    pub nl_skip: i32,
    /// Fourier-amplitude randomisation sigmas. **Both 0.0 in production**, which is what makes
    /// that whole path dead.
    pub fa_sig1: f32,
    pub fa_sig2: f32,
    /// Rupture-velocity randomisation sigma. **0.1 in production**, so unlike the two above
    /// this path is live: it perturbs `rvf` per subfault, capped by [`defaults::RVFMAX`].
    pub rv_sig1: f32,
    pub path_duration: PathDurationModel,
    pub stress_param_adjust: StressParamAdjust,
    /// Target magnitude for the stress-parameter adjustment. `None` derives it from the moment.
    pub target_magnitude: Option<f32>,
    /// Total fault area, km². `None` takes it from the slip model.
    pub fault_area: Option<f32>,
}

impl HfConfig {
    pub fn czero(&self) -> f32 {
        self.czero.unwrap_or(defaults::CZERO)
    }
    pub fn calpha(&self) -> f32 {
        self.calpha.unwrap_or(defaults::CALPHA)
    }
    pub fn vs_moho(&self) -> f64 {
        self.vs_moho.unwrap_or(defaults::VS_MOHO)
    }
    /// Always zero. Kept as a named quantity because it appears in the corner-frequency
    /// expression `c₀(1 + fcfac)`.
    pub fn fcfac(&self) -> f32 {
        defaults::FCFAC
    }
    /// True when any randomisation sigma is set, which is the condition under which the block of
    /// normal deviates is drawn at all — and therefore affects the draw count.
    pub fn draws_normal_deviates(&self) -> bool {
        self.fa_sig1 > 0.0 || self.fa_sig2 > 0.0 || self.rv_sig1 > 0.0
    }
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
        assert_eq!(PathDurationModel::from_deck(0), Some(PathDurationModel::Gp2010));
        assert_eq!(PathDurationModel::from_deck(-7), Some(PathDurationModel::Gp2010));
        assert_eq!(PathDurationModel::from_deck(11), Some(PathDurationModel::Bt2014Wus));
        // The gaps are the point: these values left the duration table uninitialised.
        for bad in [3, 5, 10, 13, 99] {
            assert_eq!(PathDurationModel::from_deck(bad), None, "{bad} should be rejected");
        }
    }

    #[test]
    fn deep_transition_tracks_the_deepest_hypocentre() {
        // Shallower than the default: defaults stand.
        let t = RuptureVelocity { frac: None, shallow: None, deep: None }.resolve(3.0);
        assert_eq!((t.deep_dmin, t.deep_dmax), (defaults::DEEP_DMIN, defaults::DEEP_DMAX));
        // Deeper: the band moves down and spans 5 km.
        let t = RuptureVelocity { frac: None, shallow: None, deep: None }.resolve(22.0);
        assert_eq!((t.deep_dmin, t.deep_dmax), (22.0, 27.0));
        assert_eq!(t.rvfac, defaults::RVFAC);
    }

    #[test]
    fn unrecognised_stress_adjustment_is_silently_none() {
        // This is why the original's misaligned deck was inert rather than an error.
        assert_eq!(StressParamAdjust::from_deck(0), StressParamAdjust::None);
        assert_eq!(StressParamAdjust::from_deck(-1), StressParamAdjust::None);
        assert_eq!(StressParamAdjust::from_deck(7), StressParamAdjust::None);
        assert_eq!(StressParamAdjust::from_deck(1), StressParamAdjust::LeonardActive);
    }
}
