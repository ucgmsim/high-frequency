//! Source-receiver geometry: geodesic distance and azimuth, and per-subfault rays.

/// A point on the ellipsoid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeoPoint {
    pub lat_deg: f32,
    pub lon_deg: f32,
}

/// Outputs of [`distance_azimuth`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistanceAzimuth {
    /// Great-circle distance, km.
    pub distance_km: f32,
    /// Azimuth event to station, radians, in `[0, 2pi)`.
    pub azimuth_rad: f32,
    /// Azimuth event to station, degrees, in `[0, 360)`.
    pub azimuth_deg: f32,
}

/// The WGS84 ellipsoid, built once.
static WGS84: std::sync::OnceLock<geographiclib_rs::Geodesic> = std::sync::OnceLock::new();

/// Geodesic distance and azimuth from an event to a station, on WGS84.
pub fn distance_azimuth(event: GeoPoint, station: GeoPoint) -> DistanceAzimuth {
    use geographiclib_rs::InverseGeodesic;

    let geodesic = WGS84.get_or_init(geographiclib_rs::Geodesic::wgs84);

    let (metres, geodesic_azimuth_deg, _back_azimuth_deg, _arc_deg): (f64, f64, f64, f64) =
        geodesic.inverse(
            event.lat_deg as f64,
            event.lon_deg as f64,
            station.lat_deg as f64,
            station.lon_deg as f64,
        );

    // geographiclib reports azimuth in (-180, 180]; the callers want [0, 360). Wrapping
    // 360.0 exactly to 0.0 keeps the range half-open after the f32 narrowing, which a
    // bare `+ 360.0` does not for azimuths within an f32 ulp of zero from below.
    let shifted = if geodesic_azimuth_deg < 0.0 {
        geodesic_azimuth_deg + 360.0
    } else {
        geodesic_azimuth_deg
    } as f32;
    let azimuth_deg = if shifted >= 360.0 { 0.0 } else { shifted };

    let radians = azimuth_deg.to_radians();
    let azimuth_rad = if radians >= std::f32::consts::TAU {
        0.0
    } else {
        radians
    };

    DistanceAzimuth {
        distance_km: (metres / 1000.0) as f32,
        azimuth_rad,
        azimuth_deg,
    }
}

/// One subfault's source-to-station geometry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SubfaultRay {
    /// Slant distance from subfault to station, km. Includes depth, so
    /// this is what the path-duration table is computed from.
    pub slant_km: f32,
    /// Station azimuth seen from the subfault, radians.
    pub azimuth_rad: f32,
    /// Geometric take-off angle, radians. Used in place of the traced ray
    /// parameter under the straight-ray approximation.
    pub takeoff_rad: f32,
    /// Horizontal distance from subfault to station, km.
    pub horizontal_km: f32,
    /// Subfault depth below the surface, km.
    pub depth_km: f32,
}

/// Every subfault of one segment, as seen from one station, indexed 1-based through
/// [`Self::at`].
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
    /// # Panics
    ///
    /// If either index is outside the grid. This is not redundant with the slice's bounds
    /// check: the two indices fold into one offset, so an out-of-range `along_strike` lands
    /// on the wrong row inside the buffer (on a 4×3 grid, `at(5, 1)` returns subfault
    /// `(1, 2)`).
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
///
/// `along_strike_offset_km` is half the fault length along strike, so
/// `(i-0.5)*subfault_length_km - along_strike_offset_km` centres the along-strike
/// coordinate on the reference point.
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

    // Degrees-to-km scale factors at the fault's latitude, obtained empirically: one geodesic
    // solve a degree east, one a degree north.
    let reference = GeoPoint {
        lat_deg: fault.lat_deg,
        lon_deg: 0.0,
    };
    let east = distance_azimuth(
        reference,
        GeoPoint {
            lat_deg: fault.lat_deg,
            lon_deg: 1.0,
        },
    );
    let north = distance_azimuth(
        reference,
        GeoPoint {
            lat_deg: fault.lat_deg + 1.0,
            lon_deg: 0.0,
        },
    );
    let km_per_degree_lon = east.distance_km * (pi * east.azimuth_deg / 180.0).sin();
    let km_per_degree_lat = north.distance_km * (pi * north.azimuth_deg / 180.0).cos();

    let strike_rad = strike_deg * pi / 180.0;
    let dip_rad = dip_deg * pi / 180.0;

    // Unlike the subfault pass in `sim`, the iteration order here is free: every entry is a
    // pure function of `(i, j)` with no accumulation and no RNG draw. Walking depth rows
    // writes the strike-fastest storage sequentially.
    //
    // The 1-based subfault numbers are the `+ 1` on the enumerations: the along-strike
    // coordinate of subfault `i` is `(i - 0.5) * length`, half a cell from the edge.
    for (row, down_dip_row) in rays.chunks_mut(along_strike_count).enumerate() {
        let j = row + 1;
        let down_dip_km = (j - 1) as f32 * subfault_width_km + subfault_width_km / 2.0;
        // The subfault centre's horizontal offset perpendicular to strike, and its depth.
        let across_strike_km = down_dip_km * dip_rad.cos();
        let depth_km = top_depth_km + down_dip_km * dip_rad.sin();

        for (col, ray) in down_dip_row.iter_mut().enumerate() {
            let i = col + 1;
            let along_strike_km = (i as f32 - 0.5) * subfault_length_km - along_strike_offset_km;
            let east_km = along_strike_km * strike_rad.sin() + across_strike_km * strike_rad.cos();
            let north_km = along_strike_km * strike_rad.cos() - across_strike_km * strike_rad.sin();

            let centre = GeoPoint {
                lat_deg: fault.lat_deg + north_km / km_per_degree_lat,
                lon_deg: fault.lon_deg + east_km / km_per_degree_lon,
            };
            let to_station = distance_azimuth(centre, station);
            let horizontal_km = to_station.distance_km;

            *ray = SubfaultRay {
                horizontal_km,
                slant_km: (horizontal_km * horizontal_km + depth_km * depth_km).sqrt(),
                takeoff_rad: pi - horizontal_km.atan2(depth_km),
                azimuth_rad: to_station.azimuth_rad,
                depth_km,
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

    /// On a 4x3 grid, `at(5, 1)` folds to flat offset 4, which is in bounds and is the ray
    /// for subfault (1, 2), so only [`SubfaultGeometry::at`]'s assert catches it.
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
