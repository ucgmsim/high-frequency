//! The simulation configuration, as a typed value rather than a positional deck.
//!
//! # Why this exists
//!
//! The Fortran is driven by a 22-line list-directed deck on stdin. That format is
//! not merely inconvenient, it is unsafe: `read(5,*)` spans record boundaries, so a
//! missing or extra token silently rebinds every field after it, with no error and
//! no way to notice. Production has a live bug of exactly this shape — see
//! `REFACTOR.md` §1.1, where `ispar_adjust` is permanently `0` and two other fields
//! are swapped, which quietly disables the whole stress-parameter-adjustment
//! feature.
//!
//! This module is the replacement: named fields, enums instead of magic integers,
//! and `Option` instead of out-of-band sentinel values. `read_deck` in the binary
//! remains the bridge from the old format, because the parity gate drives that
//! binary with generated decks and it is the only remaining tie to the Fortran
//! oracle.
//!
//! # Sentinels become `Option`
//!
//! The deck signals "use the default" with a value **below −1.0** — not merely
//! negative, which is the sort of distinction that is invisible at a call site and
//! obvious in a type. Each such field is an `Option` here, with an accessor that
//! resolves it, so the resolved value is computed in exactly one place.

/// The Fortran's pi, spelled exactly as it appears at `hb_high_ref.f:150`.
///
/// **Not** `std::f32::consts::PI`, which differs in the last bits. Both the slip
/// model reader and the simulation convert degrees with this, so it lives in one
/// place to keep them from drifting.
pub const PAI: f32 = 3.1415926;

/// Degrees to radians.
pub const PU: f32 = PAI / 180.0;

/// Built-in defaults for the fields the deck can leave unset.
///
/// `CZERO` is written 2.1 in a comment in the original and then 2.0 in code; the
/// code wins. `FCFAC` is never read from input at all — it is assigned from a
/// default and then hardwired to 0.0.
pub mod defaults {
    pub const CZERO: f32 = 2.0;
    pub const RVFAC: f32 = 0.8;
    pub const SHAL_RVFAC: f32 = 0.6;
    pub const SHAL_DMIN: f32 = 5.0;
    pub const SHAL_DMAX: f32 = 8.0;
    pub const DEEP_RVFAC: f32 = 0.6;
    pub const DEEP_DMIN: f32 = 15.0;
    pub const DEEP_DMAX: f32 = 20.0;
    pub const CALPHA: f32 = 0.1;
    pub const FCFAC: f32 = 0.0;
    /// `vsmoho` when the deck leaves it non-positive: high enough that no layer
    /// reaches it, i.e. "do not truncate at the Moho".
    pub const VS_MOHO: f64 = 999.9;
    /// Ceiling on the perturbed rupture-velocity factor.
    pub const RVFMAX: f32 = 1.4;
}

/// One entry of the deck's `rayset`.
///
/// The integer is kept because `gf_amp_tt` needs it, but the two things the program
/// actually asks about it are exposed as methods.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RayType(pub i32);

/// How a [`RayType`] is interpreted.
///
/// `0` is not a ray at all: it selects the straight-line geometric path instead of
/// a traced one. For traced rays the **parity** of the number picks the take-off
/// direction, which is why the Fortran tests `mod(irtype,2) == 1`.
///
/// Production runs `rayset = [1]`, so only `Upgoing` is exercised there; the other
/// two are reached by the `rayset=0` and `rayset=1,2` parity decks.
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

    /// The `itype` to trace with. A straight ray still gets traced, as type 1,
    /// because the Fortran calls `gf_amp_tt` before it checks for type 0 and only
    /// afterwards overwrites the results.
    pub fn trace_type(self) -> i32 {
        match self.kind() {
            RayKind::StraightRay => 1,
            _ => self.0,
        }
    }
}

/// The path-duration model: how record duration grows with distance.
///
/// The deck encodes these as `0`/`1`/`2`/`11`/`12` — a non-contiguous set where
/// every other value leaves `ndur` undefined in the Fortran, which then indexes an
/// uninitialised table. An enum makes that unrepresentable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathDurationModel {
    /// `<= 0` — Graves & Pitarka 2010, single segment, slope 0.063 s/km.
    Gp2010,
    /// `1` — western US, slope 0.070.
    Wus,
    /// `2` — eastern North America, slope 0.100.
    Ena,
    /// `11` — Boore & Thompson 2014 WUS. Six segments; its breakpoints at 7, 45,
    /// 125 and 175 km are what the Phase 2 distance ladder straddles.
    Bt2014Wus,
    /// `12` — Boore & Thompson 2015 ENA. Eight segments.
    Bt2015Ena,
}

impl PathDurationModel {
    /// Decode the deck's integer. Note `<= 0` all map to `Gp2010`, matching the
    /// Fortran's `if(ipdur_model.le.0)`.
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

/// Stress-parameter adjustment to a target magnitude.
///
/// **Inert in production**, and not by configuration: the deck misalignment
/// described in `REFACTOR.md` §1.1 means the Fortran always reads `ispar_adjust`
/// as the literal `0` that `hf_sim.py` writes on its own line, so `spar_fac` is
/// always 1.0 no matter what the workflow config says.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum StressParamAdjust {
    /// `0` or anything unrecognised — no adjustment, `spar_fac = 1`.
    None,
    /// `1` — Leonard (2010) scaling for active tectonic regions.
    LeonardActive,
    /// `2` — Leonard (2010) scaling for stable continental regions.
    LeonardStable,
}

impl StressParamAdjust {
    /// Anything other than 1 or 2 means no adjustment — the Fortran's `else`
    /// branch, which is what makes an unrecognised value silently harmless.
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
/// These seven values are always used together, at three call sites where the
/// Fortran repeats the taper character-for-character.
///
/// The deep transition depths are **not** constants: they are raised to track the
/// deepest hypocentre in the slip model, so [`RuptureVelocity::resolve`] takes it.
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
    /// `zhyp_max` is the deepest hypocentre in the slip model. When it is below the
    /// default deep transition the defaults stand; otherwise the deep band moves
    /// down to start at the hypocentre and span 5 km.
    pub fn resolve(self, zhyp_max: f32) -> RuptureVelocityTaper {
        let (deep_dmin, deep_dmax) = if zhyp_max > defaults::DEEP_DMIN {
            (zhyp_max, zhyp_max + 5.0)
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
    /// Shallow taper first, then the deep taper *overwrites* it where the depth
    /// falls in the deep band — not a blend of the two.
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

/// Everything needed to simulate one station, with nothing about where the inputs
/// came from or where the output goes.
#[derive(Clone, Debug)]
pub struct HfConfig {
    /// `sdrop` — average stress drop, bars.
    pub stress_drop: f32,
    /// `rayset`. Production is `[RayType(1)]`.
    pub rayset: Vec<RayType>,
    /// `isite_amp != 0`.
    pub site_amp: bool,
    /// `irand`. Note this is *mutated* by seeding: `init_random_seed` advances it,
    /// and the advanced value gates the rupture-time jitter.
    pub seed: i32,
    /// Record length, seconds.
    pub duration: f32,
    /// Sample interval, seconds.
    pub dt: f32,
    /// `fmax` — high-frequency cutoff, Hz.
    pub fmax: f32,
    /// `kappa` — near-surface attenuation, seconds.
    pub kappa: f32,
    /// `qfexp` — frequency exponent of Q.
    pub qfexp: f32,
    pub rupture_velocity: RuptureVelocity,
    /// `czero` — corner-frequency constant.
    pub czero: Option<f32>,
    /// `calpha` — the `alphaT` corner-frequency adjustment coefficient.
    pub calpha: Option<f32>,
    /// `mom` — total seismic moment. `None` derives it from the slip model.
    pub moment: Option<f32>,
    /// `rupv` — constant rupture velocity. `None` takes rupture times from the slip
    /// model instead, which is what production does.
    pub rupture_velocity_override: Option<f32>,
    /// `vs_moho` — shear velocity at which to truncate the model.
    pub vs_moho: Option<f64>,
    /// `nl_skip`. Negative means the velocity model is used unperturbed; a
    /// non-negative value would route through `grandvel`, which is dead here.
    pub nl_skip: i32,
    /// `fa_sig1`, `fa_sig2` — Fourier-amplitude randomisation sigmas. Both 0.0 in
    /// production, which is what makes `famprand` dead.
    pub fa_sig1: f32,
    pub fa_sig2: f32,
    /// `rv_sig1` — rupture-velocity randomisation sigma. **0.1 in production**, so
    /// unlike the two above this path is live.
    pub rv_sig1: f32,
    pub path_duration: PathDurationModel,
    pub stress_param_adjust: StressParamAdjust,
    /// `targ_mag` — target magnitude for the adjustment. `None` derives it from the
    /// moment.
    pub target_magnitude: Option<f32>,
    /// `fault_area` — km². `None` takes it from the slip model.
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
    /// `fcfac`. Read from nothing and hardwired to zero in the original; kept as a
    /// named quantity because it appears in the corner-frequency expression.
    pub fn fcfac(&self) -> f32 {
        defaults::FCFAC
    }
    /// True when any randomisation sigma is set, which is the condition under which
    /// the normal deviates are drawn at all.
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
        // The gaps are the point: 3..=10 leave ndur undefined in the Fortran.
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
        // The Fortran's else branch. This is why production's misaligned deck is
        // inert rather than an error.
        assert_eq!(StressParamAdjust::from_deck(0), StressParamAdjust::None);
        assert_eq!(StressParamAdjust::from_deck(-1), StressParamAdjust::None);
        assert_eq!(StressParamAdjust::from_deck(7), StressParamAdjust::None);
        assert_eq!(StressParamAdjust::from_deck(1), StressParamAdjust::LeonardActive);
    }
}
