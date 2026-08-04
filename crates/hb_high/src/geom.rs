//! Source-receiver geometry. Tier 0 holds `DELAZ5`; `even_dist2` lands here in
//! tier 1.

/// Outputs of [`delaz5`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Delaz5 {
    /// Angular separation, radians.
    pub delt: f32,
    /// Angular separation, degrees.
    pub deltdg: f32,
    /// Great-circle distance, km.
    pub deltkm: f32,
    /// Azimuth event to station, radians, in `[0, 2pi)`.
    pub azes: f32,
    /// Azimuth event to station, degrees.
    pub azesdg: f32,
    /// Back azimuth station to event, radians, in `[0, 2pi)`.
    pub azse: f32,
    /// Back azimuth station to event, degrees.
    pub azsedg: f32,
}

/// `SUBROUTINE DELAZ5(...)` — `hb_high_ref.f:2673`. Geodetic distance and
/// azimuths via direction cosines.
///
/// `i <= 0` means the inputs are geographic degrees (converted here, including
/// the 0.9931177 flattening correction); `i > 0` means geocentric radians.
/// The original selects this with an arithmetic `IF(I) 50,50,51`, so zero and
/// negative both take the degrees path.
///
/// **The `i > 0` path is dead code.** `even_dist2` assigns `i=0` immediately
/// before its first call (`:2626`) and passes the literal `0` at its second
/// (`:2658`), and those are the only live call sites. The branch is kept for
/// line-by-line comparability but is deliberately not covered by the goldens.
///
/// # Precision
///
/// This routine is the clearest example of why precision is tracked per
/// expression (`PORTING_RULES.md` §2). Sixteen direction-cosine variables are
/// declared `DOUBLE PRECISION`, but everything else — including the angles fed
/// to `SIN`/`COS` and every output — is `real*4`. So `C = SIN(THE)` computes a
/// **single**-precision sine and widens the result, and `C1 = A*AP+B*BP+C*CP`
/// computes in double and narrows. Both narrowings are load-bearing.
///
/// Three separation regimes avoid catastrophic cancellation near 0 and 180
/// degrees, selected by arithmetic `IF`s on `C1-0.94` and `C1+0.94`.
pub fn delaz5(thei: f32, alei: f32, thsi: f32, alsi: f32, i: i32) -> Delaz5 {
    let (the, ale, ths, als): (f32, f32, f32, f32);

    if i <= 0 {
        // Geographic degrees. 1.745329252E-2 is the source's own truncated
        // pi/180; do not replace it with a computed constant.
        let mut the_ = 1.745329252E-2 * thei;
        let ale_ = 1.745329252E-2 * alei;
        let mut ths_ = 1.745329252E-2 * thsi;
        let als_ = 1.745329252E-2 * alsi;
        let aaa = 0.9931177 * the_.tan();
        the_ = aaa.atan();
        let aaa = 0.9931177 * ths_.tan();
        ths_ = aaa.atan();
        (the, ale, ths, als) = (the_, ale_, ths_, als_);
    } else {
        (the, ale, ths, als) = (thei, alei, thsi, alsi);
    }

    // Single-precision trig, widened into the double-precision cosines.
    let c = the.sin() as f64;
    let ak = -(the.cos() as f64);
    let d = ale.sin() as f64;
    let e = -(ale.cos() as f64);
    let a = ak * e;
    let b = -ak * d;
    let g = -c * e;
    let h = c * d;
    let cp = ths.sin() as f64;
    let akp = -(ths.cos() as f64);
    let dp = als.sin() as f64;
    let ep = -(als.cos() as f64);
    let ap = akp * ep;
    let bp = -akp * dp;
    let gp = -cp * ep;
    let hp = cp * dp;

    // Double-precision dot product narrowed to real*4.
    let c1 = (a * ap + b * bp + c * cp) as f32;

    // IF(C1-0.94) 30,31,31 -- below 0.94 goes to 30, otherwise to 31.
    let delt: f32 = if c1 - 0.94 < 0.0 {
        // IF(C1+0.94) 28,28,29 -- at or below -0.94 goes to 28.
        if c1 + 0.94 <= 0.0 {
            // Label 28: near-antipodal, use the half-chord of the sum.
            let mut c1b = ((a + ap) * (a + ap) + (b + bp) * (b + bp) + (c + cp) * (c + cp)) as f32;
            c1b = c1b.sqrt();
            c1b = c1b / 2.0;
            2.0 * c1b.acos()
        } else {
            // Label 29: the well-conditioned middle range.
            c1.acos()
        }
    } else {
        // Label 31: nearly coincident, use the half-chord of the difference.
        let mut c1b = ((a - ap) * (a - ap) + (b - bp) * (b - bp) + (c - cp) * (c - cp)) as f32;
        c1b = c1b.sqrt();
        c1b = c1b / 2.0;
        2.0 * c1b.asin()
    };

    // Label 33: common tail.
    let deltkm = 6371.0 * delt;
    let c3 = ((ap - d) * (ap - d) + (bp - e) * (bp - e) + cp * cp - 2.0) as f32;
    let c4 = ((ap - g) * (ap - g) + (bp - h) * (bp - h) + (cp - ak) * (cp - ak) - 2.0) as f32;
    let c5 = ((a - dp) * (a - dp) + (b - ep) * (b - ep) + c * c - 2.0) as f32;
    let c6 = ((a - gp) * (a - gp) + (b - hp) * (b - hp) + (c - akp) * (c - akp) - 2.0) as f32;
    let deltdg = 57.29577951 * delt;

    let mut azes = c3.atan2(c4);
    // IF(AZES) 80,81,81 -- only a strictly negative value is wrapped.
    if azes < 0.0 {
        azes = 6.283185308 + azes;
    }
    let mut azse = c5.atan2(c6);
    if azse < 0.0 {
        azse = 6.283185308 + azse;
    }
    let azesdg = 57.29577951 * azes;
    let azsedg = 57.29577951 * azse;

    Delaz5 { delt, deltdg, deltkm, azes, azesdg, azse, azsedg }
}
