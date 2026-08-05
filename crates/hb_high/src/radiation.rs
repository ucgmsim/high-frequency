//! Double-couple far-field radiation coefficients.
//!
//! Tier 0 holds `RDATN` only; `RADFRQ_lin` and `RADV_lin` land here in tier 2.

/// `SUBROUTINE RDATN(STR,DIP,RAK,AZ,TH,RDSH,RDSV)` — `hb_high_ref.f:2275`.
///
/// SH and SV radiation coefficients for a double couple (Aki & Richards).
/// All angles in radians: `str` strike, `dip_rad` dip_rad, `rake_rad` rake, `azimuth_rad` azimuth
/// source to receiver clockwise from north, `takeoff_rad` incidence angle measured from
/// down. Returns `(rdsh, rdsv)`.
///
/// The expressions below preserve Fortran's left-to-right association exactly.
/// `SR*(CD**2-SD**2)*(CT**2-ST**2)*SS` is
/// `((sin_rake * (cos_dip*cos_dip - sin_dip*sin_dip)) * (cos_takeoff*cos_takeoff - sin_takeoff*sin_takeoff)) * sin_az` — regrouping it, even into
/// something algebraically identical, moves the last bits.
///
/// The commented-out alternative forms in the source are earlier versions using
/// double-angle identities; they are *not* bit-equivalent to what is compiled
/// and must not be substituted.
pub fn radiation_pattern(strike_rad: f32, dip_rad: f32, rake_rad: f32, azimuth_rad: f32, takeoff_rad: f32) -> (f32, f32) {
    let sin_rake = rake_rad.sin();
    let cos_rake = rake_rad.cos();
    let sin_dip = dip_rad.sin();
    let cos_dip = dip_rad.cos();
    let sin_takeoff = takeoff_rad.sin();
    let cos_takeoff = takeoff_rad.cos();
    let sin_az = (azimuth_rad - strike_rad).sin();
    let cos_az = (azimuth_rad - strike_rad).cos();

    // RDP is computed by the Fortran and then discarded -- the P radiation
    // coefficient is never returned or used. Kept so the two sources stay
    // line-comparable; see PORTING_RULES.md §7.
    let _rdp = cos_rake * sin_dip * (sin_takeoff * sin_takeoff) * 2.0 * sin_az * cos_az - cos_rake * cos_dip * 2.0 * sin_takeoff * cos_takeoff * cos_az
        + sin_rake * 2.0 * sin_dip * cos_dip * ((cos_takeoff * cos_takeoff) - (sin_takeoff * sin_takeoff) * (sin_az * sin_az))
        + sin_rake * ((cos_dip * cos_dip) - (sin_dip * sin_dip)) * 2.0 * sin_takeoff * cos_takeoff * sin_az;

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

/// `SUBROUTINE RADFRQ_lin(...)` — `hb_high_ref.f:1939`.
///
/// Conically averaged S radiation pattern for one subfault and receiver. The
/// average is taken over rays whose strike, dip, rake, azimuth and take-off
/// angle are perturbed within +/-45 degrees of the theoretical values — a cone
/// around the theoretical ray rather than a full spherical average, on the
/// reasoning that the parameters are more likely to be near their nominal
/// values than in an arbitrary orientation.
///
/// Returns the clobbered `fr1` (see below). `radiation` receives the pattern per
/// frequency bin.
///
/// # This is the dominant RNG consumer
///
/// The averaging loop draws **five** deviates per iteration, in the order
/// `th, fa, strX, dipX, rakX`, and runs `sample_count = 1000` times — so 5,000 draws per
/// call, and it is called twice per subfault per ray. Any change to that order
/// or count desynchronises the whole stream. See `PORTING_RULES.md` §5.
///
/// # `fr1` is a dummy argument the Fortran overwrites
///
/// The Fortran assigns `fr1 = 0.5` unconditionally at entry, clobbering the
/// caller's variable — which is `flol`, the Butterworth low-cut read from the
/// deck. The incoming value is never read, so this does not affect the routine
/// itself, and the only consumer of the mutated `flol` is `filter3d`, which is
/// dead under `ift = 0`. So the clobber is real but inert. Returned explicitly
/// here rather than hidden.
///
/// # `radmin = 1.0` makes the frequency blend inert
///
/// `radmin` is set to `0.5` and then immediately to `1.0`, which forces `del`
/// to exactly 1.0 on all three branches. So the result is the conical average
/// everywhere and the `fr1`/`fr2` taper never bites. The expression is still
/// written out in full: `rdx + (radvh-rdx)*1.0` is not bitwise equal to `radvh`.
///
/// `RNA` and `RNB` are declared in the Fortran signature and never read; they
/// are omitted here.
#[allow(clippy::too_many_arguments)]
pub fn horizontal_radiation_spectrum(
    rng: &mut impl crate::rng::Draws,
    strike_rad: f32,
    dip_rad: f32,
    rake_rad: f32,
    azimuth_rad: f32,
    takeoff_rad: f32,
    frequency_hz: &[f32],
    fold_count: usize,
    component_rad: f32,
    sample_count: usize,
    radiation: &mut [f32],
) -> f32 {
    let pu = std::f32::consts::PI / 180.0;

    let fr1 = 0.5f32;
    let fr2 = 2.0f32;
    // Set to 0.5 then immediately overwritten. Kept for line-comparability.
    #[allow(unused_assignments)]
    let mut radmin = 0.5f32;
    // "Since using a conical average around theoretical ray, don't allow much
    // purely theoretical rad pattern" -- 2009-02-10.
    radmin = 1.0;

    let (rdsha, rdsva) = radiation_pattern(strike_rad, dip_rad, rake_rad, azimuth_rad, takeoff_rad);

    // The Fortran computes RDX with a cos(THAA) factor and then immediately
    // recomputes it without. The first value is dead; kept so the two sources
    // line up.
    let _rdx_superseded =
        rdsva * takeoff_rad.cos() * (component_rad - azimuth_rad).cos() + rdsha * (component_rad - azimuth_rad).sin();

    // The 2004-03-19 "RADPAT FIX": take abs() after summing SV and SH, not
    // before, otherwise a negative cos or sin creates asymmetry.
    let mut rdx = rdsva * (component_rad - azimuth_rad).cos() + rdsha * (component_rad - azimuth_rad).sin();

    // 2004-12-21: preserve the sign rather than taking abs().
    let mut polarity = 1.0f32;
    if rdx < 0.0 {
        polarity = -1.0;
        rdx = -rdx;
    }

    let range = 10.0f32;
    // NOT an iterator chain. Each iteration draws five deviates from the shared stream in
    // the order th, fa, strX, dipX, rakX, and the sum is a left-to-right f32 fold; a
    // `map(..).sum()` would preserve both today but invites a later `rayon` or a
    // reordering that would not. See PORTING_RULES.md §5.
    let mut radv = 0.0f32;
    for _k in 1..=sample_count {
        // Five draws, in this exact order. 9*range*pu is 90 degrees in radians.
        let th = takeoff_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let fa = azimuth_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let strx = strike_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let dipx = dip_rad + 9.0 * range * pu * (0.5 - rng.next_f32());
        let rakx = rake_rad + 9.0 * range * pu * (0.5 - rng.next_f32());

        let (rdsha, rdsva) = radiation_pattern(strx, dipx, rakx, fa, th);
        let rads = rdsva * (component_rad - fa).cos() + rdsha * (component_rad - fa).sin();
        // Squared to remove the sign; polarity is applied at the end, hence
        // the sqrt below.
        radv += rads * rads;
    }

    let radvh = (radv / sample_count as f32).sqrt();

    for (gain, &freq) in radiation[..fold_count].iter_mut().zip(frequency_hz) {
        let del = if freq <= fr1 {
            radmin
        } else if freq <= fr2 {
            // The Fortran repeats `freq > fr1` here; the else-if already establishes it.
            let d = (freq / fr1).ln() / (fr2 / fr1).ln();
            if d < radmin { radmin } else { d }
        } else {
            1.0
        };
        *gain = polarity * (rdx + (radvh - rdx) * del);
    }

    fr1
}

/// `SUBROUTINE RADV_lin(...)` — `hb_high_ref.f:2140`.
///
/// Vertical-component radiation coefficient. Unlike [`horizontal_radiation_spectrum`] this takes
/// its random numbers from the caller-supplied `uniform_a`/`uniform_b` arrays (filled once
/// per run by `RANU2`), so it consumes **no** draws from the shared stream.
///
/// There is no `cmp` argument — the vertical component needs no horizontal
/// projection, and the pattern is `RDSV * sin(th)`.
///
/// Returns the clobbered `fr1`, which the Fortran overwrites with `0.001`. As
/// with [`horizontal_radiation_spectrum`] this mutates the caller's `flol`, and since `RADV_lin` is
/// called *after* both `RADFRQ_lin` calls, `flol` ends the subfault at 0.001
/// rather than the deck's 0.02. Inert only because `filter3d` is dead.
///
/// Two initialisations are immediately superseded and kept for comparability:
/// `fr2 = 1.5` before `fr2 = 0.01`, and `radvh = 0.7` before the computed
/// average. The take-off range is clamped to `[90, 180]` degrees.
#[allow(clippy::too_many_arguments)]
pub fn vertical_radiation_spectrum(
    strike_rad: f32,
    dip_rad: f32,
    rake_rad: f32,
    azimuth_rad: f32,
    takeoff_rad: f32,
    frequency_hz: &[f32],
    fold_count: usize,
    uniform_a: &[f32],
    uniform_b: &[f32],
    sample_count: usize,
    radiation: &mut [f32],
) -> f32 {
    let pu = std::f32::consts::PI / 180.0;

    let _fr2_superseded = 1.5f32;
    let fr1 = 0.001f32;
    let fr2 = 0.01f32;
    let _radvh_superseded = 0.7f32;

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

    // The two uniform arrays are consumed in lockstep, one pair per sample. They are
    // FILLED by two separate sequential passes -- draws 1..nr into `a`, then nr+1..2nr
    // into `b` -- and that must not become one interleaved pass, or every vertical
    // component moves. Zipping the consumption is free; zipping the fill is not.
    let mut radv = 0.0f32;
    for (&ua, &ub) in uniform_a[..sample_count].iter().zip(uniform_b) {
        // Uniform in cos(th) between the clamped limits.
        let th = ((1.0 - ua) * tha1.cos() + ua * tha2.cos()).acos();
        let fa = 360.0 * pu * ub;
        let (_rdsha, rdsva) = radiation_pattern(strike_rad, dip_rad, rake_rad, fa, th);
        let rads = rdsva * th.sin();
        radv += rads.abs();
    }

    let radvh = radv / sample_count as f32 / 2.0;

    // Below `fr1` the theoretical pattern, above `fr2` the conical average, and a linear
    // blend between. The Fortran writes `rdx` first and then overwrites or adds to it,
    // which reads as three branches only once you notice the fall-through; written as
    // one expression per bin it is visibly a piecewise function.
    for (gain, &freq) in radiation[..fold_count].iter_mut().zip(frequency_hz) {
        *gain = if freq <= fr1 {
            rdx
        } else if freq <= fr2 {
            rdx + (radvh - rdx) * (freq - fr1) / (fr2 - fr1)
        } else {
            radvh
        };
    }

    fr1
}
