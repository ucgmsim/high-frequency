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
pub fn rdatn(str_: f32, dip: f32, rak: f32, az: f32, th: f32) -> (f32, f32) {
    let sr = rak.sin();
    let cr = rak.cos();
    let sd = dip.sin();
    let cd = dip.cos();
    let st = th.sin();
    let ct = th.cos();
    let ss = (az - str_).sin();
    let cs = (az - str_).cos();

    // RDP is computed by the Fortran and then discarded -- the P radiation
    // coefficient is never returned or used. Kept so the two sources stay
    // line-comparable; see PORTING_RULES.md §7.
    let _rdp = cr * sd * (st * st) * 2.0 * ss * cs - cr * cd * 2.0 * st * ct * cs
        + sr * 2.0 * sd * cd * ((ct * ct) - (st * st) * (ss * ss))
        + sr * ((cd * cd) - (sd * sd)) * 2.0 * st * ct * ss;

    let rdsv = sr * ((cd * cd) - (sd * sd)) * ((ct * ct) - (st * st)) * ss
        - cr * cd * ((ct * ct) - (st * st)) * cs
        + cr * sd * st * ct * 2.0 * ss * cs
        - sr * sd * cd * 2.0 * st * ct * (1.0 + (ss * ss));

    let rdsh = cr * cd * ct * ss
        + cr * sd * st * ((cs * cs) - (ss * ss))
        + sr * ((cd * cd) - (sd * sd)) * ct * cs
        - sr * sd * cd * st * 2.0 * ss * cs;

    (rdsh, rdsv)
}
