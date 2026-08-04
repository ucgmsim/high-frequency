//! Source-receiver geometry. Tier 0 holds `DELAZ5`; `subfault_geometry` lands here in
//! tier 1.

/// Outputs of [`distance_azimuth`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceAzimuth {
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
/// `coord_mode <= 0` means the inputs are geographic degrees (converted here, including
/// the 0.9931177 flattening correction); `coord_mode > 0` means geocentric radians.
/// The original selects this with an arithmetic `IF(I) 50,50,51`, so zero and
/// negative both take the degrees path.
///
/// **The `coord_mode > 0` path is dead code.** `subfault_geometry` assigns `coord_mode=0` immediately
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
pub fn distance_azimuth(event_lat_deg: f32, event_lon_deg: f32, station_lat_deg: f32, station_lon_deg: f32, coord_mode: i32) -> DistanceAzimuth {
    let (the, ale, ths, als): (f32, f32, f32, f32);

    if coord_mode <= 0 {
        // Geographic degrees. 1.745329252E-2 is the source's own truncated
        // pi/180; do not replace it with a computed constant.
        let mut the_ = 1.745329252E-2 * event_lat_deg;
        let ale_ = 1.745329252E-2 * event_lon_deg;
        let mut ths_ = 1.745329252E-2 * station_lat_deg;
        let als_ = 1.745329252E-2 * station_lon_deg;
        let aaa = 0.9931177 * the_.tan();
        the_ = aaa.atan();
        let aaa = 0.9931177 * ths_.tan();
        ths_ = aaa.atan();
        (the, ale, ths, als) = (the_, ale_, ths_, als_);
    } else {
        (the, ale, ths, als) = (event_lat_deg, event_lon_deg, station_lat_deg, station_lon_deg);
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

    DistanceAzimuth { delt, deltdg, deltkm, azes, azesdg, azse, azsedg }
}

/// `subroutine subfault_geometry(...)` — `hb_high_ref.f:2593`.
///
/// Per-subfault source-to-receiver geometry for a single planar fault segment.
/// Fills five `(nq, np)` arrays, indexed `(i, j)` for along-strike and down-dip:
///
/// * `dst` — horizontal epicentral distance, km
/// * `rl`  — slant range from subfault centre to station, km
/// * `th`  — take-off angle, radians, measured as `pi - atan2(dis, depth)`
/// * `ph`  — azimuth, radians (`azes` straight from `DELAZ5`)
/// * `zet` — subfault depth, km
///
/// `astop` is half the fault length along strike, so `(i-0.5)*dx - astop`
/// centres the along-strike coordinate on the reference point.
///
/// The degree-to-km scale factors `ddx`/`ddy` are obtained empirically: two
/// `DELAZ5` calls one degree apart in longitude and in latitude respectively.
/// Both `x` and `y` are computed on each pass but only one is kept, matching the
/// Fortran.
///
/// `pi` is the source's own 9-digit `3.14159265`, not `std::f32::consts::PI`.
/// Everything here is `f32`; there is no double-precision arithmetic.
#[allow(clippy::too_many_arguments)]
/// Per-subfault source-to-station geometry, every array indexed `(i, j)`.
///
/// The Fortran keeps these as five separate `(nq, np)` arrays named `rlsu`, `phsu`,
/// `thsu`, `dst` and `zet`; they are always allocated, filled and indexed together,
/// so they are one value.
pub struct SubfaultGeometry {
    /// `rlsu` — slant distance from subfault to station, km. Includes depth, so
    /// this is what the path-duration table and `d10` are computed from.
    pub slant_km: crate::fort::Array2<f32>,
    /// `phsu` — station azimuth seen from the subfault, radians.
    pub azimuth_rad: crate::fort::Array2<f32>,
    /// `thsu` — geometric take-off angle, radians. Used in place of the traced ray
    /// parameter under the straight-ray approximation.
    pub takeoff_rad: crate::fort::Array2<f32>,
    /// `dst` — horizontal distance from subfault to station, km.
    pub horiz_km: crate::fort::Array2<f32>,
    /// `zet` — subfault depth below the surface, km.
    pub depth_km: crate::fort::Array2<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn subfault_geometry(
    fault_lon_deg: f32,
    fault_lat_deg: f32,
    station_lon_deg: f32,
    station_lat_deg: f32,
    strike_deg: f32,
    dip_deg: f32,
    top_depth_km: f32,
    along_strike_offset_km: f32,
    subfault_length_km: f32,
    subfault_width_km: f32,
    along_strike_count: usize,
    down_dip_count: usize,
) -> SubfaultGeometry {
    // Sized to the actual grid, not to the compile-time maximum. The Fortran
    // declares these `(nq, np)` = 600x100, i.e. 234 KB each and 1.14 MB for the
    // five, essentially all of it untouched -- and it allocates them per segment.
    // Every access below and in every caller is `(i, j)` within `1..=along_strike_count`/`1..=down_dip_count`,
    // so the layout is not observable and compacting them changes no arithmetic.
    // This is the same argument `input::Segment` already makes for sddp/rist/rupt.
    let mut rl = crate::fort::Array2::<f32>::new(along_strike_count, down_dip_count);
    let mut ph = crate::fort::Array2::<f32>::new(along_strike_count, down_dip_count);
    let mut th = crate::fort::Array2::<f32>::new(along_strike_count, down_dip_count);
    let mut dst = crate::fort::Array2::<f32>::new(along_strike_count, down_dip_count);
    let mut zet = crate::fort::Array2::<f32>::new(along_strike_count, down_dip_count);

    let pi = 3.14159265f32;
    let alei = 0.0f32;
    let alsi = 0.0f32;
    let thei = fault_lat_deg;

    // Degrees-to-km scale factors, one degree east and one degree north.
    let mut ddx = 0.0f32;
    let mut ddy = 0.0f32;
    for ii in 1..=2 {
        let (thsi, alsi2) = if ii == 1 {
            (thei, alei + 1.0)
        } else {
            (thei + 1.0, alsi)
        };
        let g = distance_azimuth(thei, alei, thsi, alsi2, 0);
        let az = g.azesdg;
        let dis = g.deltkm;
        let x = dis * (pi * az / 180.0).sin();
        let y = dis * (pi * az / 180.0).cos();
        if ii == 1 {
            ddx = x;
        }
        if ii == 2 {
            ddy = y;
        }
    }

    let az = strike_deg * pi / 180.0;
    let dip = dip_deg * pi / 180.0;

    let ylat = fault_lat_deg;
    let xlon = fault_lon_deg;

    // DO 20 I=1,NX / DO 20 J=1,NW share one terminator: i outer, j inner.
    for i in 1..=along_strike_count {
        for j in 1..=down_dip_count {
            let down_dip = (j - 1) as f32 * subfault_width_km + subfault_width_km / 2.0;
            let a1 = down_dip * dip.cos();
            let b1 = down_dip * dip.sin();

            let along = (i as f32 - 0.5) * subfault_length_km - along_strike_offset_km;
            let dlon = along * az.sin() + a1 * az.cos();
            let dlat = along * az.cos() - a1 * az.sin();

            let stlon = xlon + dlon / ddx;
            let stlat = ylat + dlat / ddy;

            let zm1 = top_depth_km + b1;

            let g = distance_azimuth(stlat, stlon, station_lat_deg, station_lon_deg, 0);
            let dis = g.deltkm;

            dst[(i, j)] = dis;
            rl[(i, j)] = (dis * dis + zm1 * zm1).sqrt();
            th[(i, j)] = pi - dis.atan2(zm1);
            ph[(i, j)] = g.azes;
            zet[(i, j)] = zm1;
        }
    }

    SubfaultGeometry {
        slant_km: rl,
        azimuth_rad: ph,
        takeoff_rad: th,
        horiz_km: dst,
        depth_km: zet,
    }
}
