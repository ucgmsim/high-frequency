//! Double-couple radiation coefficients, and the conical average taken over them.
//!
//! A double couple does not radiate equally in all directions, and a single subfault's
//! theoretical pattern is too sharp to be realistic. Graves & Pitarka (2010) therefore use a
//! **conically averaged** pattern: perturb the five angles randomly and average. See
//! `PHYSICS.md` §5.

use ndarray::{ArrayView1, ArrayViewMut1, azip};

/// Full width of the horizontal component's perturbation cone: ±45° on each of the five
/// angles, as Graves & Pitarka (2010) specify.
pub const CONE_WIDTH_RAD: f32 = 90.0 * (std::f32::consts::PI / 180.0);
/// Half-width of the vertical component's take-off cone.
const VERTICAL_CONE_HALF_WIDTH_RAD: f32 = 40.0 * (std::f32::consts::PI / 180.0);
/// Straight down: the shallowest take-off the vertical average accepts.
const DOWNGOING_MIN_RAD: f32 = 90.0 * (std::f32::consts::PI / 180.0);
/// Straight up.
const DOWNGOING_MAX_RAD: f32 = 180.0 * (std::f32::consts::PI / 180.0);
/// One full azimuthal turn.
const FULL_TURN_RAD: f32 = 360.0 * (std::f32::consts::PI / 180.0);

/// Lower bound on the conical-average blend for a horizontal component.
///
/// At 1.0 the blend is pinned to the conical average at every frequency, which is what
/// "since using a conical average around theoretical ray, don't allow much purely theoretical
/// rad pattern" asks for.
pub const CONICAL_FLOOR: f32 = 1.0;

/// Fault orientation and the ray's arrival direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RadiationAngles {
    pub strike_rad: f32,
    pub dip_rad: f32,
    pub rake_rad: f32,
    /// Source to receiver, clockwise from north.
    pub azimuth_rad: f32,
    /// Incidence angle, measured from down.
    pub takeoff_rad: f32,
}

/// The two shear radiation coefficients for one ray leaving a double couple.
///
/// Aki & Richards write these `F^SH` and `F^SV`: how much amplitude the source radiates into
/// each of the two shear polarisations along the ray. Both are dimensionless and in `[-1, 1]`;
/// the sign is a polarity, not a magnitude.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShearRadiation {
    /// `rdsh` — the SH lobe: transverse to the ray and horizontal.
    pub sh: f32,
    /// `rdsv` — the SV lobe: transverse to the ray and in the vertical plane through it.
    pub sv: f32,
}

/// The eight sines and cosines both coefficients are built from.
///
/// Shared so that [`sv_radiation`] computes exactly the same intermediates as
/// [`radiation_pattern`] rather than a re-derived set that might round differently.
#[derive(Clone, Copy)]
struct AngleTerms {
    sin_rake: f32,
    cos_rake: f32,
    sin_dip: f32,
    cos_dip: f32,
    sin_takeoff: f32,
    cos_takeoff: f32,
    /// Azimuth measured from the strike direction, not from north.
    sin_az: f32,
    cos_az: f32,
}

impl AngleTerms {
    #[inline]
    fn of(angles: RadiationAngles) -> Self {
        let RadiationAngles {
            strike_rad,
            dip_rad,
            rake_rad,
            azimuth_rad,
            takeoff_rad,
        } = angles;
        Self {
            sin_rake: rake_rad.sin(),
            cos_rake: rake_rad.cos(),
            sin_dip: dip_rad.sin(),
            cos_dip: dip_rad.cos(),
            sin_takeoff: takeoff_rad.sin(),
            cos_takeoff: takeoff_rad.cos(),
            sin_az: (azimuth_rad - strike_rad).sin(),
            cos_az: (azimuth_rad - strike_rad).cos(),
        }
    }
}

/// # Do not regroup these expressions
///
/// They are written to a specific association order, and regrouping — even into something
/// algebraically identical — moves the last bits. `SR*(CD²-SD²)*(CT²-ST²)*SS` is
/// `((sin_rake * (cos_dip² - sin_dip²)) * (cos_takeoff² - sin_takeoff²)) * sin_az`, and the
/// parenthesisation below says so explicitly.
///
/// Double-angle identities would simplify these and are **not** bit-equivalent.
#[inline]
fn sv_from(t: AngleTerms) -> f32 {
    t.sin_rake
        * ((t.cos_dip * t.cos_dip) - (t.sin_dip * t.sin_dip))
        * ((t.cos_takeoff * t.cos_takeoff) - (t.sin_takeoff * t.sin_takeoff))
        * t.sin_az
        - t.cos_rake
            * t.cos_dip
            * ((t.cos_takeoff * t.cos_takeoff) - (t.sin_takeoff * t.sin_takeoff))
            * t.cos_az
        + t.cos_rake * t.sin_dip * t.sin_takeoff * t.cos_takeoff * 2.0 * t.sin_az * t.cos_az
        - t.sin_rake
            * t.sin_dip
            * t.cos_dip
            * 2.0
            * t.sin_takeoff
            * t.cos_takeoff
            * (1.0 + (t.sin_az * t.sin_az))
}

/// The SH lobe. Same association-order warning as [`sv_from`].
#[inline]
fn sh_from(t: AngleTerms) -> f32 {
    t.cos_rake * t.cos_dip * t.cos_takeoff * t.sin_az
        + t.cos_rake * t.sin_dip * t.sin_takeoff * ((t.cos_az * t.cos_az) - (t.sin_az * t.sin_az))
        + t.sin_rake
            * ((t.cos_dip * t.cos_dip) - (t.sin_dip * t.sin_dip))
            * t.cos_takeoff
            * t.cos_az
        - t.sin_rake * t.sin_dip * t.cos_dip * t.sin_takeoff * 2.0 * t.sin_az * t.cos_az
}

/// Both shear radiation coefficients for a double couple. (orig. `hb_high_ref.f:2275`)
///
/// Aki & Richards, *Quantitative Seismology* (2nd ed.), ch. 4.
#[inline]
pub fn radiation_pattern(angles: RadiationAngles) -> ShearRadiation {
    let terms = AngleTerms::of(angles);
    ShearRadiation {
        sh: sh_from(terms),
        sv: sv_from(terms),
    }
}

/// Only the SV coefficient.
///
/// [`vertical_radiation_spectrum`] never uses the SH lobe, so it asks for the one it wants.
///
/// **This is a separation of concerns, not an optimisation** — the compiler eliminates an
/// unused half either way.
#[inline]
pub fn sv_radiation(angles: RadiationAngles) -> f32 {
    sv_from(AngleTerms::of(angles))
}

/// Conically averaged radiation pattern for a horizontal component, per frequency bin.
/// (orig. `hb_high_ref.f:1939`)
///
/// This is `RP_ij` in Graves & Pitarka (2010) eq. 11: the pattern averaged over rays whose
/// strike, dip, rake, azimuth and take-off angle are perturbed within **±45°** of their
/// theoretical values — a cone around the theoretical ray, on the reasoning that the true
/// parameters are more likely near their nominal values than in an arbitrary orientation.
///
/// [`CONE_WIDTH_RAD`] is 90°, so `(0.5 - u)` scaled by it gives ±45° exactly as the paper
/// specifies.
///
/// # This is the dominant consumer of random numbers
///
/// **Five draws per iteration, in the order `th, fa, strX, dipX, rakX`, and `sample_count` is
/// 1000** — so 5,000 draws per call, twice per subfault per ray. The draw order and count are
/// the phase spectrum (`PHYSICS.md` §9); changing either desynchronises every waveform that
/// follows.
///
/// # The frequency blend is inert
///
/// [`CONICAL_FLOOR`] is 1.0, which forces the blend to exactly 1.0 on all three branches, so
/// the result is the conical average at every frequency and the taper never bites. The
/// expression is still written out in full, because
/// `theoretical + (conical - theoretical) * 1.0` is not bitwise equal to `conical`.
///
/// # The return value is a discarded output
///
/// The blend's lower corner is returned because the caller once assigned it to a Butterworth
/// low-cut, clobbering it. That filter is dead under production settings, so the clobber is
/// real but inert. Returned explicitly rather than hidden.
pub fn horizontal_radiation_spectrum(
    rng: &mut impl crate::rng::Draws,
    angles: &RadiationAngles,
    frequency_hz: ArrayView1<f32>,
    component_rad: f32,
    sample_count: usize,
    radiation: ArrayViewMut1<f32>,
) -> f32 {
    let &RadiationAngles {
        strike_rad,
        dip_rad,
        rake_rad,
        azimuth_rad,
        takeoff_rad,
    } = angles;

    let blend_low_hz = 0.5f32;
    let blend_high_hz = 2.0f32;

    let theoretical = radiation_pattern(*angles);

    // Project SV and SH onto the requested horizontal component. The sum is formed BEFORE the
    // magnitude is taken (below), because taking it per-term lets a negative cos or sin
    // introduce an asymmetry that is not physical.
    let projected = theoretical.sv * (component_rad - azimuth_rad).cos()
        + theoretical.sh * (component_rad - azimuth_rad).sin();

    // Sign preserved, not discarded: polarity is carried separately and reapplied at the end.
    let polarity = if projected < 0.0 { -1.0f32 } else { 1.0f32 };
    let theoretical_gain = projected.abs();

    let mut sum_of_squares = 0.0f32;
    for _sample in 0..sample_count {
        // FIVE DRAWS, AND THIS IS THE ORDER. Bound to named locals rather than written
        // straight into the struct literal below: field-initialiser order is what would
        // decide the draw order there, so reordering the fields for readability would
        // silently move every waveform. Here the order is a sequence of statements, which is
        // not something anyone reorders by accident.
        let takeoff = takeoff_rad + CONE_WIDTH_RAD * (0.5 - rng.uniform());
        let azimuth = azimuth_rad + CONE_WIDTH_RAD * (0.5 - rng.uniform());
        let strike = strike_rad + CONE_WIDTH_RAD * (0.5 - rng.uniform());
        let dip = dip_rad + CONE_WIDTH_RAD * (0.5 - rng.uniform());
        let rake = rake_rad + CONE_WIDTH_RAD * (0.5 - rng.uniform());

        let sample = radiation_pattern(RadiationAngles {
            strike_rad: strike,
            dip_rad: dip,
            rake_rad: rake,
            azimuth_rad: azimuth,
            takeoff_rad: takeoff,
        });
        let projected = sample.sv * (component_rad - azimuth).cos()
            + sample.sh * (component_rad - azimuth).sin();
        // Squared to remove the sign, so the average is an RMS; `polarity` restores the sign
        // after the sqrt below.
        sum_of_squares += projected * projected;
    }

    let conical_gain = (sum_of_squares / sample_count as f32).sqrt();

    // Piecewise in frequency: theoretical pattern below `fr1`, conical average above `fr2`,
    // log-linear blend between. Inert as written -- see `radmin` above.
    azip!((
        gain in radiation,
        &freq in frequency_hz,
    ) {
        let blend = if freq <= blend_low_hz {
            CONICAL_FLOOR
        } else if freq <= blend_high_hz {
            // The `max` applies to the QUOTIENT, not to the denominator. Written without the
            // parentheses it binds to `.ln()` and silently changes the blend.
            ((freq / blend_low_hz).ln() / (blend_high_hz / blend_low_hz).ln())
                .max(CONICAL_FLOOR)
        } else {
            1.0
        };
        *gain = polarity * (theoretical_gain + (conical_gain - theoretical_gain) * blend);
    });

    blend_low_hz
}

/// Conically averaged radiation pattern for the vertical component, per frequency bin.
/// (orig. `hb_high_ref.f:2140`)
///
/// The vertical needs no horizontal projection, so the pattern is just `RDSV * sin(takeoff)`,
/// and the average is taken over take-off angle and azimuth only.
///
/// The take-off range is clamped to `[90°, 180°]`: only downgoing directions contribute.
///
/// Like [`horizontal_radiation_spectrum`], the returned `fr1` is a value the original wrote
/// back into the caller's low-cut, and is inert for the same reason.
pub fn vertical_radiation_spectrum(
    angles: &RadiationAngles,
    frequency_hz: ArrayView1<f32>,
    uniform_a: &[f32],
    uniform_b: &[f32],
    sample_count: usize,
    radiation: ArrayViewMut1<f32>,
) -> f32 {
    let &RadiationAngles {
        strike_rad,
        dip_rad,
        rake_rad,
        takeoff_rad,
        ..
    } = angles;

    let blend_low_hz = 0.001f32;
    let blend_high_hz = 0.01f32;

    let theoretical_gain = sv_radiation(*angles) * takeoff_rad.sin();

    // Only downgoing directions contribute, so the cone is clipped to [90°, 180°].
    let takeoff_min_rad = (takeoff_rad - VERTICAL_CONE_HALF_WIDTH_RAD).max(DOWNGOING_MIN_RAD);
    let takeoff_max_rad = (takeoff_rad + VERTICAL_CONE_HALF_WIDTH_RAD).min(DOWNGOING_MAX_RAD);

    // The two uniform arrays are consumed in lockstep, one pair per sample. They are FILLED by
    // two separate sequential passes -- the first `nr` draws into `a`, the next `nr` into `b` --
    // and THAT MUST NOT BECOME ONE INTERLEAVED PASS, or every vertical component moves.
    // Zipping the consumption is free; zipping the fill is not.
    //
    // The two limits are fixed above, so their cosines are loop invariants.
    let (cos_min, cos_max) = (takeoff_min_rad.cos(), takeoff_max_rad.cos());
    let mut sum_of_magnitudes = 0.0f32;
    for (&ua, &ub) in uniform_a[..sample_count].iter().zip(uniform_b) {
        // Uniform in cos(takeoff) between the clamped limits, which samples solid angle
        // evenly rather than angle evenly.
        let takeoff_rad = ((1.0 - ua) * cos_min + ua * cos_max).acos();
        let perturbed = RadiationAngles {
            strike_rad,
            dip_rad,
            rake_rad,
            azimuth_rad: FULL_TURN_RAD * ub,
            takeoff_rad,
        };
        sum_of_magnitudes += (sv_radiation(perturbed) * takeoff_rad.sin()).abs();
    }

    // Halved because the magnitudes above are folded about zero.
    let conical_gain = sum_of_magnitudes / sample_count as f32 / 2.0;

    // Piecewise in frequency: theoretical pattern below `fr1`, conical average above `fr2`,
    // linear blend between. Written as one expression per bin so the piecewise structure is
    // visible rather than emerging from a fall-through.
    azip!((
        gain in radiation,
        &freq in frequency_hz,
    ) {
        *gain = if freq <= blend_low_hz {
            theoretical_gain
        } else if freq <= blend_high_hz {
            theoretical_gain
                + (conical_gain - theoretical_gain) * (freq - blend_low_hz)
                    / (blend_high_hz - blend_low_hz)
        } else {
            conical_gain
        };
    });

    blend_low_hz
}
