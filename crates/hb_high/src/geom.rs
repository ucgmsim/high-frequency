//! Source-receiver geometry. Tier 0 holds `DELAZ5`; `subfault_geometry` lands here in
//! tier 1.

/// A point on the ellipsoid.
///
/// This exists because the two entry points in this file disagreed. `distance_azimuth`
/// took **lat first**; `subfault_geometry` took **lon first**, 120 lines away. Both were
/// called correctly, so it was a latent trap rather than a live bug — but transposing a
/// lat/lon pair yields a *plausible* distance and a *plausible* azimuth, which then feed
/// the path-duration table and the `1/R` geometric spreading. Nothing would look wrong.
///
/// Named fields make the order unstatable rather than merely documented.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub lat_deg: f32,
    pub lon_deg: f32,
}

/// Outputs of [`distance_azimuth`].
///
/// Four of the Fortran's seven outputs are gone with §2.5: `delt` and `deltdg` (the
/// angular separation, in radians and degrees) and `azse`/`azsedg` (the back azimuth) had
/// no reader outside the tests that checked them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceAzimuth {
    /// Great-circle distance, km.
    pub deltkm: f32,
    /// Azimuth event to station, radians, in `[0, 2pi)`.
    pub azes: f32,
    /// Azimuth event to station, degrees, in `[0, 360)`.
    pub azesdg: f32,
}

/// The WGS84 ellipsoid, built once.
///
/// `Geodesic::wgs84()` precomputes series coefficients, and this is called `2 + nx*nw`
/// times per segment per station — a few thousand times on a real fault — so building it
/// per call would put that setup in the inner loop for no reason.
static WGS84: std::sync::OnceLock<geographiclib_rs::Geodesic> = std::sync::OnceLock::new();

/// Geodesic distance and azimuth from an event to a station, on WGS84.
///
/// Replaces `SUBROUTINE DELAZ5(...)` (`hb_high_ref.f:2673`), ~80 lines of 1970s
/// direction-cosine geometry: three separation regimes to dodge catastrophic cancellation
/// near 0 and 180 degrees, a `0.9931177` tangent-scaling trick standing in for the
/// ellipsoid, a spherical Earth of radius exactly `6371.0` km, and sixteen
/// `DOUBLE PRECISION` cosines feeding `real*4` trig — `PORTING_RULES.md` §2 used it as the
/// worked example of why precision has to be tracked per expression.
///
/// `geographiclib_rs` solves the actual inverse geodesic problem, so all of that goes: no
/// regimes, no flattening approximation, no mixed precision.
///
/// # What moved, measured
///
/// Over source-station separations from 4 to 409 km around the Canterbury faults this port
/// is run on:
///
/// | | worst |
/// | --- | --- |
/// | distance | **0.058%**, about 240 m at 409 km |
/// | azimuth | **0.0065 degrees** |
///
/// `DELAZ5` is systematically **short**, consistently signed, which is what the
/// mean-radius sphere plus the tangent-scaling trick produce against a true geodesic. The
/// new values are the correct ones. Distance reaches the waveform through the
/// path-duration table and the `1/R` geometric spreading, both smooth in distance, so a
/// 0.05% shift lands far inside Tier C's +-2% equivalence band.
///
/// # The `coord_mode` argument is gone
///
/// `DELAZ5` took a flag selecting geographic degrees or geocentric radians. The radians
/// path was dead — `subfault_geometry` assigned `0` immediately before its first call and
/// passed the literal `0` at its second, and those were the only live callers.
pub fn distance_azimuth(event: GeoPoint, station: GeoPoint) -> DistanceAzimuth {
    use geographiclib_rs::InverseGeodesic;

    let geodesic = WGS84.get_or_init(geographiclib_rs::Geodesic::wgs84);

    // The four-element form is `(s12, azi1, azi2, a12)`. Note that the THREE-element form
    // is `(azi1, azi2, a12)` -- the tuple width changes what the earlier slots mean, so
    // this annotation is load-bearing and a shorter tuple silently yields azimuths where
    // a distance is expected.
    let (metres, azimuth_deg, _back_azimuth_deg, _arc_deg): (f64, f64, f64, f64) = geodesic
        .inverse(
            event.lat_deg as f64,
            event.lon_deg as f64,
            station.lat_deg as f64,
            station.lon_deg as f64,
        );

    // geographiclib reports azimuth in (-180, 180]; the callers want [0, 360). Wrapping
    // 360.0 exactly to 0.0 keeps the range half-open after the f32 narrowing, which a
    // bare `+ 360.0` does not for azimuths within an f32 ulp of zero from below.
    let mut azesdg = if azimuth_deg < 0.0 { azimuth_deg + 360.0 } else { azimuth_deg } as f32;
    if azesdg >= 360.0 {
        azesdg = 0.0;
    }
    let mut azes = azesdg.to_radians();
    if azes >= std::f32::consts::TAU {
        azes = 0.0;
    }

    DistanceAzimuth { deltkm: (metres / 1000.0) as f32, azes, azesdg }
}

/// One subfault's source-to-station geometry.
///
/// The Fortran keeps these as five separate `(nq, np)` arrays named `rlsu`, `phsu`,
/// `thsu`, `dst` and `zet`. Every read of one is at the same `(i, j)` as the other
/// four, so this is one value per subfault rather than five grids — §2.3.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubfaultRay {
    /// `rlsu` — slant distance from subfault to station, km. Includes depth, so
    /// this is what the path-duration table and `d10` are computed from.
    pub slant_km: f32,
    /// `phsu` — station azimuth seen from the subfault, radians.
    pub azimuth_rad: f32,
    /// `thsu` — geometric take-off angle, radians. Used in place of the traced ray
    /// parameter under the straight-ray approximation.
    pub takeoff_rad: f32,
    /// `dst` — horizontal distance from subfault to station, km.
    pub horiz_km: f32,
    /// `zet` — subfault depth below the surface, km.
    pub depth_km: f32,
}

/// Every subfault of one segment, as seen from one station.
///
/// # Why the accessor is 1-based when §2.3 converted everything else to 0-based
///
/// `(i, j)` here is a *subfault number* — subfault `i` along strike, `j` down dip — not
/// a storage offset, and the physical formulas that consume it are written in terms of
/// that number: the along-strike coordinate is `(i - 0.5) * length`, so the first
/// subfault sits half a cell from the edge. §2.3's win was for storage indices, where
/// 1-based-ness is pure friction and blocks vectorisation. Renumbering a domain
/// quantity from 1 to 0 would put a `+ 1` into every one of those formulas and make
/// them *less* readable, for nothing measurable — these loops are over a few hundred
/// branchy elements.
///
/// So the 1-based-ness stays, but it now lives in exactly one place — [`Self::at`] —
/// instead of being spread across five bounds-checked `Index` impls. That is the part
/// that was worth changing.
pub struct SubfaultGeometry {
    along_strike_count: usize,
    down_dip_count: usize,
    /// Strike index fastest, matching the deck's one-record-per-depth-row layout.
    rays: Vec<SubfaultRay>,
}

impl SubfaultGeometry {
    /// Subfault `along_strike` (`1..=along_strike_count`) at depth row `down_dip`
    /// (`1..=down_dip_count`).
    ///
    /// Returns by value: `SubfaultRay` is five `f32` and `Copy`, so a caller that wants
    /// four of the five fields gets them from one lookup instead of four.
    #[inline]
    pub fn at(&self, along_strike: usize, down_dip: usize) -> SubfaultRay {
        assert!(
            (1..=self.along_strike_count).contains(&along_strike)
                && (1..=self.down_dip_count).contains(&down_dip),
            "subfault ({along_strike},{down_dip}) is outside the {}x{} grid",
            self.along_strike_count,
            self.down_dip_count
        );
        self.rays[(down_dip - 1) * self.along_strike_count + (along_strike - 1)]
    }
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
/// `along_strike_offset_km` is half the fault length along strike, so
/// `(i-0.5)*subfault_length_km - along_strike_offset_km` centres the along-strike
/// coordinate on the reference point.
///
/// The degree-to-km scale factors `ddx`/`ddy` are obtained empirically: two
/// `DELAZ5` calls one degree apart in longitude and in latitude respectively.
/// Both `x` and `y` are computed on each pass but only one is kept, matching the
/// Fortran.
///
/// Everything here is `f32`; there is no double-precision arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn subfault_geometry(
    fault: GeoPoint,
    station: GeoPoint,
    strike_deg: f32,
    dip_deg: f32,
    top_depth_km: f32,
    along_strike_offset_km: f32,
    subfault_length_km: f32,
    subfault_width_km: f32,
    along_strike_count: usize,
    down_dip_count: usize,
) -> SubfaultGeometry {
    // Sized to the actual grid, not to the compile-time maximum. The Fortran declares
    // these `(nq, np)` = 600x100, i.e. 234 KB each and 1.14 MB for the five, essentially
    // all of it untouched -- and it allocates them per segment. Layout is not observable
    // (nothing indexes outside the real grid), so compacting them changes no arithmetic.
    // This is the same argument `input::Segment` already makes for its three grids.
    let mut rays = vec![SubfaultRay::default(); along_strike_count * down_dip_count];

    // Was the source's own 9-digit `3.14159265`; now the correctly rounded value.
    let pi = std::f32::consts::PI;
    let thei = fault.lat_deg;

    // Degrees-to-km scale factors, obtained empirically: one geodesic solve a degree east,
    // one a degree north.
    //
    // The Fortran writes this as `DO ii = 1, 2` whose body is an `if ii == 1 / else` on
    // both the inputs and the outputs -- a two-iteration loop that branches on its own
    // counter, which is two statements wearing a loop. Unrolled. Both `x` and `y` were
    // computed on each pass and one discarded; only the surviving one is computed here,
    // and the reference longitude (`alei`/`alsi`, both constant 0.0) is gone with it.
    //
    // Note the SCALE is what is wanted, not a position: both solves start from
    // (thei, 0.0), so only the one-degree offset matters.
    let origin = GeoPoint { lat_deg: thei, lon_deg: 0.0 };
    let east = distance_azimuth(origin, GeoPoint { lat_deg: thei, lon_deg: 1.0 });
    let north = distance_azimuth(origin, GeoPoint { lat_deg: thei + 1.0, lon_deg: 0.0 });
    let ddx = east.deltkm * (pi * east.azesdg / 180.0).sin();
    let ddy = north.deltkm * (pi * north.azesdg / 180.0).cos();

    let az = strike_deg * pi / 180.0;
    let dip = dip_deg * pi / 180.0;

    let ylat = fault.lat_deg;
    let xlon = fault.lon_deg;

    // The Fortran runs `i` outer / `j` inner while the storage is strike-fastest, so its
    // writes are strided. Unlike the subfault pass in `sim`, the order here is FREE:
    // every entry is a pure function of `(i, j)` with no accumulation and no RNG draw, so
    // nothing downstream can observe which order they were computed in. Walking depth
    // rows writes sequentially and computes no index at all.
    //
    // The 1-based subfault numbers survive as `+ 1` on the enumerations, because they are
    // physics -- the along-strike coordinate of subfault `i` is `(i - 0.5) * length`, so
    // the first subfault sits half a cell from the edge. See the note on `SubfaultRay`.
    for (row, down_dip_row) in rays.chunks_mut(along_strike_count).enumerate() {
        let j = row + 1;
        let down_dip = (j - 1) as f32 * subfault_width_km + subfault_width_km / 2.0;
        let a1 = down_dip * dip.cos();
        let b1 = down_dip * dip.sin();
        let zm1 = top_depth_km + b1;

        for (col, ray) in down_dip_row.iter_mut().enumerate() {
            let i = col + 1;
            let along = (i as f32 - 0.5) * subfault_length_km - along_strike_offset_km;
            let dlon = along * az.sin() + a1 * az.cos();
            let dlat = along * az.cos() - a1 * az.sin();

            let stlon = xlon + dlon / ddx;
            let stlat = ylat + dlat / ddy;

            let g = distance_azimuth(GeoPoint { lat_deg: stlat, lon_deg: stlon }, station);
            let dis = g.deltkm;

            *ray = SubfaultRay {
                horiz_km: dis,
                slant_km: (dis * dis + zm1 * zm1).sqrt(),
                takeoff_rad: pi - dis.atan2(zm1),
                azimuth_rad: g.azes,
                depth_km: zm1,
            };
        }
    }

    SubfaultGeometry { along_strike_count, down_dip_count, rays }
}
