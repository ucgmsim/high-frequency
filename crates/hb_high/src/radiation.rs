//! Double-couple far-field radiation coefficients.
//!
//! Tier 0 holds `RDATN` only; `RADFRQ_lin` and `RADV_lin` land here in tier 2.

/// `SUBROUTINE RDATN(STR,DIP,RAK,AZ,TH,RDSH,RDSV)` — `hb_high_ref.f:2275`.
///
/// SH and SV radiation coefficients for a double couple (Aki & Richards).
/// All angles in radians: `str` strike, `dip` dip, `rak` rake, `az` azimuth
/// source to receiver clockwise from north, `th` incidence angle measured from
/// down. Returns `(rdsh, rdsv)`.
///
/// The expressions below preserve Fortran's left-to-right association exactly.
/// `SR*(CD**2-SD**2)*(CT**2-ST**2)*SS` is
/// `((sr * (cd*cd - sd*sd)) * (ct*ct - st*st)) * ss` — regrouping it, even into
/// something algebraically identical, moves the last bits.
///
/// The commented-out alternative forms in the source are earlier versions using
/// double-angle identities; they are *not* bit-equivalent to what is compiled
/// and must not be substituted.
pub fn radiation_pattern(str_: f32, dip: f32, rak: f32, az: f32, th: f32) -> (f32, f32) {
    let sr = rak.sin();
    let vertical_slowness = rak.cos();
    let sd = dip.sin();
    let cd = dip.cos();
    let st = th.sin();
    let ct = th.cos();
    let ss = (az - str_).sin();
    let cs = (az - str_).cos();

    // RDP is computed by the Fortran and then discarded -- the P radiation
    // coefficient is never returned or used. Kept so the two sources stay
    // line-comparable; see PORTING_RULES.md §7.
    let _rdp = vertical_slowness * sd * (st * st) * 2.0 * ss * cs - vertical_slowness * cd * 2.0 * st * ct * cs
        + sr * 2.0 * sd * cd * ((ct * ct) - (st * st) * (ss * ss))
        + sr * ((cd * cd) - (sd * sd)) * 2.0 * st * ct * ss;

    let rdsv = sr * ((cd * cd) - (sd * sd)) * ((ct * ct) - (st * st)) * ss
        - vertical_slowness * cd * ((ct * ct) - (st * st)) * cs
        + vertical_slowness * sd * st * ct * 2.0 * ss * cs
        - sr * sd * cd * 2.0 * st * ct * (1.0 + (ss * ss));

    let rdsh = vertical_slowness * cd * ct * ss
        + vertical_slowness * sd * st * ((cs * cs) - (ss * ss))
        + sr * ((cd * cd) - (sd * sd)) * ct * cs
        - sr * sd * cd * st * 2.0 * ss * cs;

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
/// Returns the clobbered `fr1` (see below). `rdna` receives the pattern per
/// frequency bin.
///
/// # This is the dominant RNG consumer
///
/// The averaging loop draws **five** deviates per iteration, in the order
/// `th, fa, strX, dipX, rakX`, and runs `nr = 1000` times — so 5,000 draws per
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
    rng: &mut crate::rng::Pcg32,
    stra: f32,
    dipa: f32,
    raka: f32,
    pa: f32,
    thaa: f32,
    dfr: &crate::fort::Array1<f32>,
    nfold: usize,
    cmp: f32,
    nr: usize,
    rdna: &mut crate::fort::Array1<f32>,
) -> f32 {
    let pu = 3.1415926 / 180.0;

    let fr1 = 0.5f32;
    let fr2 = 2.0f32;
    // Set to 0.5 then immediately overwritten. Kept for line-comparability.
    #[allow(unused_assignments)]
    let mut radmin = 0.5f32;
    // "Since using a conical average around theoretical ray, don't allow much
    // purely theoretical rad pattern" -- 2009-02-10.
    radmin = 1.0;

    let (rdsha, rdsva) = radiation_pattern(stra, dipa, raka, pa, thaa);

    // The Fortran computes RDX with a cos(THAA) factor and then immediately
    // recomputes it without. The first value is dead; kept so the two sources
    // line up.
    let _rdx_superseded =
        rdsva * thaa.cos() * (cmp - pa).cos() + rdsha * (cmp - pa).sin();

    // The 2004-03-19 "RADPAT FIX": take abs() after summing SV and SH, not
    // before, otherwise a negative cos or sin creates asymmetry.
    let mut rdx = rdsva * (cmp - pa).cos() + rdsha * (cmp - pa).sin();

    // 2004-12-21: preserve the sign rather than taking abs().
    let mut polarity = 1.0f32;
    if rdx < 0.0 {
        polarity = -1.0;
        rdx = -rdx;
    }

    let range = 10.0f32;
    let mut radv = 0.0f32;
    for _k in 1..=nr {
        // Five draws, in this exact order. 9*range*pu is 90 degrees in radians.
        let th = thaa + 9.0 * range * pu * (0.5 - rng.next_f32());
        let fa = pa + 9.0 * range * pu * (0.5 - rng.next_f32());
        let strx = stra + 9.0 * range * pu * (0.5 - rng.next_f32());
        let dipx = dipa + 9.0 * range * pu * (0.5 - rng.next_f32());
        let rakx = raka + 9.0 * range * pu * (0.5 - rng.next_f32());

        let (rdsha, rdsva) = radiation_pattern(strx, dipx, rakx, fa, th);
        let rads = rdsva * (cmp - fa).cos() + rdsha * (cmp - fa).sin();
        // Squared to remove the sign; polarity is applied at the end, hence
        // the sqrt below.
        radv = radv + rads * rads;
    }

    let radvh = (radv / nr as f32).sqrt();

    for i in 1..=nfold {
        let del = if dfr[i] <= fr1 {
            radmin
        } else if dfr[i] > fr1 && dfr[i] <= fr2 {
            let d = (dfr[i] / fr1).ln() / (fr2 / fr1).ln();
            if d < radmin { radmin } else { d }
        } else {
            1.0
        };
        rdna[i] = polarity * (rdx + (radvh - rdx) * del);
    }

    fr1
}

/// `SUBROUTINE RADV_lin(...)` — `hb_high_ref.f:2140`.
///
/// Vertical-component radiation coefficient. Unlike [`horizontal_radiation_spectrum`] this takes
/// its random numbers from the caller-supplied `rna`/`rnb` arrays (filled once
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
    stra: f32,
    dipa: f32,
    raka: f32,
    pa: f32,
    thaa: f32,
    dfr: &crate::fort::Array1<f32>,
    nfold: usize,
    rna: &crate::fort::Array1<f32>,
    rnb: &crate::fort::Array1<f32>,
    nr: usize,
    rdna: &mut crate::fort::Array1<f32>,
) -> f32 {
    let pu = 3.1415926 / 180.0;

    let _fr2_superseded = 1.5f32;
    let fr1 = 0.001f32;
    let fr2 = 0.01f32;
    let _radvh_superseded = 0.7f32;

    let (_rdsha, rdsva) = radiation_pattern(stra, dipa, raka, pa, thaa);
    let rdx = rdsva * thaa.sin();

    let range = 40.0f32;
    let mut tha1 = thaa - range * pu;
    let mut tha2 = thaa + range * pu;
    if tha1 < 90.0 * pu {
        tha1 = 90.0 * pu;
    }
    if tha2 > 180.0 * pu {
        tha2 = 180.0 * pu;
    }

    let mut radv = 0.0f32;
    for k in 1..=nr {
        // Uniform in cos(th) between the clamped limits.
        let th = ((1.0 - rna[k]) * tha1.cos() + rna[k] * tha2.cos()).acos();
        let fa = 360.0 * pu * rnb[k];
        let (_rdsha, rdsva) = radiation_pattern(stra, dipa, raka, fa, th);
        let rads = rdsva * th.sin();
        radv = radv + rads.abs();
    }

    let radvh = radv / nr as f32 / 2.0;

    for i in 1..=nfold {
        rdna[i] = rdx;
        if dfr[i] <= fr1 {
            continue;
        }
        if dfr[i] > fr1 && dfr[i] <= fr2 {
            rdna[i] = rdna[i] + (radvh - rdx) * (dfr[i] - fr1) / (fr2 - fr1);
        } else {
            rdna[i] = radvh;
        }
    }

    fr1
}
