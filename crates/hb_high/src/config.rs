//! The simulation configuration, as a typed value rather than a positional deck.
//!
//! Every field here is supplied by the caller. There are no defaults and no `Option`s to
//! resolve: the Python dataclasses in `hf_simulation` are the single source of truth for
//! what a caller may leave unset, so a default cannot be declared in two places and drift.
//!
//! This module is data only. The fixed parameters of the models themselves, which no caller
//! has ever been able to set, live with the models: the rupture-velocity taper's transition
//! depths in [`crate::source`], the duration tables in [`crate::path_duration`].

use crate::path_duration::PathDurationModel;
use crate::ray::RayType;

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
/// The transition depths are not settable — see [`crate::source::RuptureVelocityTaper`] — but the
/// three reduction factors are, and the caller's values are typically locally calibrated rather
/// than the published ones. See the Python dataclass for what the defaults are and how they
/// differ from Graves & Pitarka.
#[derive(Clone, Copy, Debug)]
pub struct RuptureVelocity {
    /// Base rupture speed as a fraction of the local shear-wave velocity.
    pub fraction: f32,
    /// Multiplier at the shallow end of the taper.
    pub shallow_factor: f32,
    /// Multiplier at the deep end of the taper.
    pub deep_factor: f32,
    /// Rupture-velocity randomisation sigma, log-normal (0.1 in production). Perturbs the
    /// factor per subfault, capped by [`crate::source::RUPTURE_VELOCITY_FRACTION_MAX`]. Zero
    /// disables the perturbation *and* its deviate consumption.
    pub sigma: f32,
}

/// The earthquake source: how strong the high-frequency radiation is and how fast the rupture
/// travels.
#[derive(Clone, Debug)]
pub struct SourceParameters {
    /// `c₀`, the corner-frequency constant of Graves & Pitarka (2010) eq. 13 / (2015) eq. 1.
    pub corner_frequency_constant: f32,
    /// `c_α`, the coefficient of the `α_τ` dip-and-rake adjustment.
    pub corner_frequency_alpha: f32,
    pub rupture_velocity: RuptureVelocity,
}

/// The path from source to site that is shared by every station: which rays, and how
/// duration grows with distance.
///
/// The path attenuation exponent varies by station, so it lives on [`crate::sim::Station`].
#[derive(Clone, Debug)]
pub struct PathParameters {
    /// Which ray paths to sum over. Production is the direct upgoing ray alone.
    pub rayset: Vec<RayType>,
    pub path_duration: PathDurationModel,
}

/// The near-surface parameters shared by every station.
///
/// Quarter-wavelength site amplification ([`crate::site`]) is always applied, so it has no
/// field here. `κ` varies by station, so it lives on [`crate::sim::Station`].
#[derive(Clone, Debug)]
pub struct SiteParameters {
    /// `f_max` — the high-cut corner, Hz. See `PHYSICS.md` §3 on the `f_max`-versus-`κ`
    /// question; both parameters exist because both physical interpretations do.
    pub fmax_hz: f32,
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
/// Nothing per-station lives here — not the seed, not the location, not `κ` or the `Q`
/// exponent — because one of these drives a whole batch of stations through
/// [`crate::sim::Simulator`]. Those are on [`crate::sim::Station`].
#[derive(Clone, Debug)]
pub struct HfConfig {
    pub source: SourceParameters,
    pub path: PathParameters,
    pub site: SiteParameters,
    pub record: RecordParameters,
}
