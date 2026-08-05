//! Double-couple radiation coefficients, and the conical average taken over them.
//!
//! A double couple does not radiate equally in all directions, and a single subfault's
//! theoretical pattern is too sharp to be realistic. Graves & Pitarka (2010) therefore use a
//! **conically averaged** pattern: perturb the five angles randomly and average. See
//! `PHYSICS.md` §5.

use ndarray::{azip, ArrayView1, ArrayViewMut1};

/// Fault orientation and the ray's arrival direction — the five angles every radiation
/// calculation takes, and takes in the same order.
///
/// They travelled as five positional `f32`s, which is exactly the shape a transposition
/// hides in: `(strike, dip, rake, azimuth, takeoff)` are all radians, all plausible for
/// each other, and a swap produces a wrong answer rather than a compile error.
pub struct RadiationAngles {
    pub strike_rad: f32,
    pub dip_rad: f32,
    pub rake_rad: f32,
    /// Source to receiver, clockwise from north.
    pub azimuth_rad: f32,
    /// Incidence angle, measured from down.
    pub takeoff_rad: f32,
}

/// SH and SV radiation coefficients for a double couple. (orig. `hb_high_ref.f:2275`)
///
/// Aki & Richards, *Quantitative Seismology* (2nd ed.), ch. 4. Returns `(rdsh, rdsv)`.
///
/// # Do not regroup these expressions
///
/// They are written to a specific association order, and regrouping — even into something
/// algebraically identical — moves the last bits. `SR*(CD²-SD²)*(CT²-ST²)*SS` is
/// `((sin_rake * (cos_dip² - sin_dip²)) * (cos_takeoff² - sin_takeoff²)) * sin_az`, and the
/// parenthesisation below says so explicitly.
///
/// Double-angle identities would simplify these and are **not** bit-equivalent.
pub fn radiation_pattern(strike_rad: f32, dip_rad: f32, rake_rad: f32, azimuth_rad: f32, takeoff_rad: f32) -> (f32, f32) {
    let sin_rake = rake_rad.sin();
    let cos_rake = rake_rad.cos();
    let sin_dip = dip_rad.sin();
    let cos_dip = dip_rad.cos();
    let sin_takeoff = takeoff_rad.sin();
    let cos_takeoff = takeoff_rad.cos();
    let sin_az = (azimuth_rad - strike_rad).sin();
    let cos_az = (azimuth_rad - strike_rad).cos();

    let rdsv = sin_rake * ((cos_dip * cos_dip) - (sin_dip * sin_dip)) * ((cos_takeoff * cos_takeoff) - (sin_takeoff * sin_takeoff)) * sin_az
        - cos_rake * cos_dip * ((cos_takeoff * cos_takeoff) - (sin_takeoff * sin_takeoff)) * cos_az
        + cos_rake * sin_dip * sin_takeoff * cos_takeoff * 2.0 * sin_az * cos_az
        - sin_rake * sin_dip * cos_dip * 2.0 * sin_takeoff * cos_takeoff * (1.0 + (sin_az * sin_az));

    let rdsh = cos_rake * cos_dip * cos_takeoff * sin_az
        + cos_rake * sin_dip * sin_takeoff * ((cos_az * cos_az) - (sin_az * sin_az))
        + sin_rake * ((cos_dip * cos_dip) - (sin_dip * sin_dip)) * cos_takeoff * cos_az
        - sin_rake * sin_dip * cos_dip * sin_takeoff * 2.0 * sin_az * cos_az;

    (rdsh, rdsv)
}

/// Conically averaged radiation pattern for a horizontal component, per frequency bin.
/// (orig. `hb_high_ref.f:1939`)
///
/// This is `RP_ij` in Graves & Pitarka (2010) eq. 11: the pattern averaged over rays whose
/// strike, dip, rake, azimuth and take-off angle are perturbed within **±45°** of their
/// theoretical values — a cone around the theoretical ray, on the reasoning that the true
/// parameters are more likely near their nominal values than in an arbitrary orientation.
///
/// The `9.0 * range * pu` below is 90° in radians, so `(0.5 - u)` scaled by it gives ±45°
/// exactly as the paper specifies.
///
/// # This is the dominant consumer of random numbers
///
/// **Five draws per iteration, in the order `th, fa, strX, dipX, rakX`, and `sample_count` is
/// 1000** — so 5,000 draws per call, twice per subfault per ray. The draw order and count are
/// the phase spectrum (`PHYSICS.md` §9); changing either desynchronises every waveform that
/// follows.
///
/// # `radmin = 1.0` makes the frequency blend inert
///
/// `radmin` forces `del` to exactly 1.0 on all three branches, so the result is the conical
/// average at every frequency and the `fr1`/`fr2` taper never bites. The expression is still
/// written out in full, because `rdx + (radvh - rdx) * 1.0` is not bitwise equal to `radvh`.
///
/// # The return value is a discarded output
///
/// `fr1` is returned because the original assigned it to the caller's Butterworth low-cut,
/// clobbering it. The only consumer of that value is a filter that is dead under production
/// settings, so the clobber is real but inert. Returned explicitly rather than hidden.
pub fn horizontal_radiation_spectrum(
    rng: &mut impl crate::rng::Draws,
    angles: &RadiationAngles,
    frequency_hz: &[f32],
    component_rad: f32,
    sample_count: usize,
    radiation: &mut [f32],
) -> f32 {
    let &RadiationAngles { strike_rad, dip_rad, rake_rad, azimuth_rad, takeoff_rad } = angles;
    let pu = std::f32::consts::PI / 180.0;

    let fr1 = 0.5f32;
    let fr2 = 2.0f32;
    // Set to 1.0 in 2009 with the note "since using a conical average around theoretical ray,
    // don't allow much purely theoretical rad pattern". The superseded 0.5 is the `fr1` above.
    let radmin = 1.0f32;

    let (rdsha, rdsva) = radiation_pattern(strike_rad, dip_rad, rake_rad, azimuth_rad, takeoff_rad);

    // Project SV and SH onto the requested horizontal component. The sum is formed BEFORE the
    // magnitude is taken (below), because taking it per-term lets a negative cos or sin
    // introduce an asymmetry that is not physical.
    let mut rdx = rdsva * (component_rad - azimuth_rad).cos() + rdsha * (component_rad - azimuth_rad).sin();

    // Sign preserved, not discarded: polarity is carried separately and reapplied at the end.
    let mut polarity = 1.0f32;
    if rdx < 0.0 {
        polarity = -1.0;
        rdx = -rdx;
    }

    let range = 10.0f32;
    // NOT an iterator chain, and not a `map(..).sum()`. Two invariants live here: the five
    // draws happen in a fixed order per iteration, and `radv` is a LEFT-TO-RIGHT f32 fold.
    // A `sum()` preserves both today but invites a later `rayon` or a reassociation that
    // would not, and either would move every waveform.
    let mut radv = 0.0f32;
    for _k in 1..=sample_count {
        // Five draws, in this exact order. `9 * range * pu` is 90 degrees in radians, so each
        // perturbation spans ±45°.
        let th = takeoff_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let fa = azimuth_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let strx = strike_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let dipx = dip_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let rakx = rake_rad + 9.0 * range * pu * (0.5 - rng.next_f32());

        let (rdsha, rdsva) = radiation_pattern(strx, dipx, rakx, fa, th);
        let rads = rdsva * (component_rad - fa).cos() + rdsha * (component_rad - fa).sin();
        // Squared to remove the sign, so the average is an RMS; `polarity` restores the sign
        // after the sqrt below.
        radv += rads * rads;
    }

    let radvh = (radv / sample_count as f32).sqrt();

    // Piecewise in frequency: theoretical pattern below `fr1`, conical average above `fr2`,
    // log-linear blend between. Inert as written -- see `radmin` above.
    azip!((
        gain in ArrayViewMut1::from(radiation),
        &freq in ArrayView1::from(frequency_hz),
    ) {
        let del = if freq <= fr1 {
            radmin
        } else if freq <= fr2 {
            let d = (freq / fr1).ln() / (fr2 / fr1).ln();
            if d < radmin { radmin } else { d }
        } else {
            1.0
        };
        *gain = polarity * (rdx + (radvh - rdx) * del);
    });

    fr1
}

/// Conically averaged radiation pattern for the vertical component, per frequency bin.
/// (orig. `hb_high_ref.f:2140`)
///
/// The vertical needs no horizontal projection, so the pattern is just `RDSV * sin(takeoff)`,
/// and the average is taken over take-off angle and azimuth only.
///
/// # This routine draws NOTHING from the shared stream
///
/// It reads `uniform_a`/`uniform_b`, filled once per run, where
/// [`horizontal_radiation_spectrum`] draws 5,000 numbers per call. **That asymmetry between the
/// horizontals and the vertical is load-bearing** — it is why component order is fixed
/// (`PHYSICS.md` §9), and why iterating the three components in any other order changes every
/// waveform.
///
/// The take-off range is clamped to `[90°, 180°]`: only downgoing directions contribute.
///
/// Like [`horizontal_radiation_spectrum`], the returned `fr1` is a value the original wrote
/// back into the caller's low-cut, and is inert for the same reason.
pub fn vertical_radiation_spectrum(
    angles: &RadiationAngles,
    frequency_hz: &[f32],
    uniform_a: &[f32],
    uniform_b: &[f32],
    sample_count: usize,
    radiation: &mut [f32],
) -> f32 {
    let &RadiationAngles { strike_rad, dip_rad, rake_rad, azimuth_rad, takeoff_rad } = angles;
    let pu = std::f32::consts::PI / 180.0;

    let fr1 = 0.001f32;
    let fr2 = 0.01f32;

    let (_rdsha, rdsva) = radiation_pattern(strike_rad, dip_rad, rake_rad, azimuth_rad, takeoff_rad);
    let rdx = rdsva * takeoff_rad.sin();

    let range = 40.0f32;
    let mut tha1 = takeoff_rad - range * pu;
    let mut tha2 = takeoff_rad + range * pu;
    if tha1 < 90.0 * pu {
        tha1 = 90.0 * pu;
    }
    if tha2 > 180.0 * pu {
        tha2 = 180.0 * pu;
    }

    // The two uniform arrays are consumed in lockstep, one pair per sample. They are FILLED by
    // two separate sequential passes -- the first `nr` draws into `a`, the next `nr` into `b` --
    // and THAT MUST NOT BECOME ONE INTERLEAVED PASS, or every vertical component moves.
    // Zipping the consumption is free; zipping the fill is not.
    //
    // The two limits are fixed above, so their cosines are loop invariants.
    let (cos_tha1, cos_tha2) = (tha1.cos(), tha2.cos());
    let mut radv = 0.0f32;
    for (&ua, &ub) in uniform_a[..sample_count].iter().zip(uniform_b) {
        // Uniform in cos(takeoff) between the clamped limits, which samples solid angle
        // evenly rather than angle evenly.
        let th = ((1.0 - ua) * cos_tha1 + ua * cos_tha2).acos();
        let fa = 360.0 * pu * ub;
        let (_rdsha, rdsva) = radiation_pattern(strike_rad, dip_rad, rake_rad, fa, th);
        let rads = rdsva * th.sin();
        radv += rads.abs();
    }

    let radvh = radv / sample_count as f32 / 2.0;

    // Piecewise in frequency: theoretical pattern below `fr1`, conical average above `fr2`,
    // linear blend between. Written as one expression per bin so the piecewise structure is
    // visible rather than emerging from a fall-through.
    azip!((
        gain in ArrayViewMut1::from(radiation),
        &freq in ArrayView1::from(frequency_hz),
    ) {
        *gain = if freq <= fr1 {
            rdx
        } else if freq <= fr2 {
            rdx + (radvh - rdx) * (freq - fr1) / (fr2 - fr1)
        } else {
            radvh
        };
    });

    fr1
}
