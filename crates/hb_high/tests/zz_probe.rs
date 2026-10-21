//! The degree-north scale factor in `subfault_geometry` multiplies the geodesic length by
//! the cosine of the azimuth to a point one degree up the same meridian. That is only the
//! length itself if the azimuth is exactly zero, which this pins for every latitude where
//! the point one degree north is still on the globe.

use hb_high::geom::{GeoPoint, distance_azimuth};

#[test]
fn a_degree_north_is_exactly_due_north() {
    for k in -9000..=8900 {
        let lat = k as f32 / 100.0;
        let g = distance_azimuth(
            GeoPoint {
                lat_deg: lat,
                lon_deg: 0.0,
            },
            GeoPoint {
                lat_deg: lat + 1.0,
                lon_deg: 0.0,
            },
        );
        assert_eq!(
            g.azimuth_deg, 0.0,
            "a degree north of {lat} has azimuth {}",
            g.azimuth_deg
        );
        assert!(
            (100.0..125.0).contains(&g.distance_km),
            "a degree north of {lat} is {} km",
            g.distance_km
        );
    }
}
