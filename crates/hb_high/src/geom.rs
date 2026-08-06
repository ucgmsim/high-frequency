//! Source-receiver geometry. Tier 0 holds `DELAZ5`; `subfault_geometry` lands here in
//! tier 1.

/// A point on the ellipsoid.
///
/// Named fields make the order unstatable rather than merely documented.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub lat_deg: f32,
    pub lon_deg: f32,
}

/// Outputs of [`distance_azimuth`].
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
static WGS84: std::sync::OnceLock<geographiclib_rs::Geodesic> = std::sync::OnceLock::new();

/// Geodesic distance and azimuth from an event to a station, on WGS84.
pub fn distance_azimuth(event: GeoPoint, station: GeoPoint) -> DistanceAzimuth {
    use geographiclib_rs::InverseGeodesic;

    let geodesic = WGS84.get_or_init(geographiclib_rs::Geodesic::wgs84);

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
    let shifted = if azimuth_deg < 0.0 {
        azimuth_deg + 360.0
    } else {
        azimuth_deg
    } as f32;
    let azesdg = if shifted >= 360.0 { 0.0 } else { shifted };

    let radians = azesdg.to_radians();
    let azes = if radians >= std::f32::consts::TAU {
        0.0
    } else {
        radians
    };

    DistanceAzimuth {
        deltkm: (metres / 1000.0) as f32,
        azes,
        azesdg,
    }
}

/// One subfault's source-to-station geometry.
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
/// So the 1-based-ness stays, but it now lives in exactly one place — [`Self::at`] —
/// instead of being spread across five bounds-checked `Index` impls. That is the part
/// that was worth changing.
pub struct SubfaultGeometry {
    along_strike_count: usize,
    down_dip_count: usize,
    /// Strike index fastest.
    rays: Vec<SubfaultRay>,
}

impl SubfaultGeometry {
    /// Subfault `along_strike` (`1..=along_strike_count`) at depth row `down_dip`
    /// (`1..=down_dip_count`).
    ///
    /// # The bounds check is not redundant with the slice's
    ///
    /// It looks like one — the indexing below is checked, so why assert first? Because the
    /// two indices are folded into one offset, and an out-of-range `along_strike` **lands
    /// inside the buffer** on the wrong row rather than off the end of it. On a 4×3 grid,
    /// `at(5, 1)` computes offset 4, which is a perfectly valid index and returns subfault
    /// `(1, 2)`. The slice check cannot see the mistake; it sees a number in range.
    ///
    /// So this converts a silently wrong subfault into a panic that names the grid. See
    /// `geometry_rejects_an_index_that_would_land_on_the_wrong_row`, which is the case.
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

/// One segment's fault plane: where it is, how it is oriented, and how it is diced.
///
/// These nine travelled as nine positional arguments next to a tenth, `station`, that is
/// also a [`GeoPoint`] — so the two geographic points were adjacent and interchangeable
/// without a compile error. They are all fields of the caller's `Segment`.
#[derive(Clone, Copy)]
pub struct FaultPlane {
    /// The segment's own origin, **not** the station.
    pub origin: GeoPoint,
    pub strike_deg: f32,
    pub dip_deg: f32,
    pub top_depth_km: f32,
    pub along_strike_offset_km: f32,
    pub subfault_length_km: f32,
    pub subfault_width_km: f32,
    pub along_strike_count: usize,
    pub down_dip_count: usize,
}

/// Per-subfault source-to-receiver geometry for a single planar fault segment.
/// Fills five `(nq, np)` arrays, indexed `(i, j)` for along-strike and down-dip:
///
/// `along_strike_offset_km` is half the fault length along strike, so
/// `(i-0.5)*subfault_length_km - along_strike_offset_km` centres the along-strike
/// coordinate on the reference point.
///
pub fn subfault_geometry(plane: &FaultPlane, station: GeoPoint) -> SubfaultGeometry {
    let &FaultPlane {
        origin: fault,
        strike_deg,
        dip_deg,
        top_depth_km,
        along_strike_offset_km,
        subfault_length_km,
        subfault_width_km,
        along_strike_count,
        down_dip_count,
    } = plane;
    let mut rays = vec![SubfaultRay::default(); along_strike_count * down_dip_count];

    let pi = std::f32::consts::PI;
    let thei = fault.lat_deg;

    // Degrees-to-km scale factors, obtained empirically: one geodesic solve a degree east,
    // one a degree north.
    let origin = GeoPoint {
        lat_deg: thei,
        lon_deg: 0.0,
    };
    let east = distance_azimuth(
        origin,
        GeoPoint {
            lat_deg: thei,
            lon_deg: 1.0,
        },
    );
    let north = distance_azimuth(
        origin,
        GeoPoint {
            lat_deg: thei + 1.0,
            lon_deg: 0.0,
        },
    );
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

            let g = distance_azimuth(
                GeoPoint {
                    lat_deg: stlat,
                    lon_deg: stlon,
                },
                station,
            );
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

    SubfaultGeometry {
        along_strike_count,
        down_dip_count,
        rays,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(along_strike_count: usize, down_dip_count: usize) -> SubfaultGeometry {
        subfault_geometry(
            &FaultPlane {
                origin: GeoPoint {
                    lat_deg: -43.0,
                    lon_deg: 173.0,
                },
                strike_deg: 220.0,
                dip_deg: 70.0,
                top_depth_km: 0.0,
                along_strike_offset_km: 0.0,
                subfault_length_km: 1.5,
                subfault_width_km: 1.5,
                along_strike_count,
                down_dip_count,
            },
            GeoPoint {
                lat_deg: -43.4,
                lon_deg: 172.6,
            },
        )
    }

    /// The case that makes [`SubfaultGeometry::at`]'s assert load-bearing rather than
    /// redundant with the slice bounds check.
    ///
    /// On a 4x3 grid, `at(5, 1)` folds to flat offset 4 — in bounds, and the ray for subfault
    /// (1, 2). Without the assert this returns the wrong subfault and nothing notices.
    #[test]
    #[should_panic(expected = "outside the 4x3 grid")]
    fn geometry_rejects_an_index_that_would_land_on_the_wrong_row() {
        let geometry = grid(4, 3);
        // Proof the offset really is in range: this is the ray the caller would have got.
        let _neighbour = geometry.at(1, 2);
        let _ = geometry.at(5, 1);
    }

    /// Zero is out of range at the other end, and would underflow the `- 1` rather than
    /// overflow the buffer.
    #[test]
    #[should_panic(expected = "outside the 4x3 grid")]
    fn geometry_rejects_a_zero_index() {
        let _ = grid(4, 3).at(0, 1);
    }
}
