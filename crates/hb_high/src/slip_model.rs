//! The slip model: the fault segments and the subfault grid on each.
//!
//! File parsing lives in Python (`source_modelling.stoch.StochFile`); this module holds the
//! data model and the aggregates derived from it.

use crate::geom::{FaultPlane, GeoPoint};

/// One fault segment from the `.stoch` file.
///
/// Slip, rise time and rupture time are stored as a subfault grid, along-strike by down-dip.
#[derive(Clone, Debug)]
pub struct Segment {
    /// Longitude of the segment's along-strike reference point.
    pub fault_lon_deg: f32,
    /// Latitude of the same point.
    pub fault_lat_deg: f32,
    /// Subfault count along strike.
    pub along_strike_count: usize,
    /// Subfault count down dip.
    pub down_dip_count: usize,
    /// Subfault dimension along strike, km.
    pub subfault_length_km: f32,
    /// Subfault dimension down dip, km.
    pub subfault_width_km: f32,
    /// Strike, degrees clockwise from north.
    pub strike_deg: f32,
    /// Dip, degrees from horizontal.
    pub dip_deg: f32,
    /// Rake, degrees.
    pub rake_deg: f32,
    /// Depth to the top edge of the segment, km.
    pub top_depth_km: f32,
    /// Hypocentre offset along strike from the segment centre, km.
    pub hypocentre_along_strike_km: f32,
    /// Hypocentre offset down dip from the top edge, km.
    pub hypocentre_down_dip_km: f32,
    /// Half the fault length along strike,
    /// `0.5 * along_strike_count * subfault_length_km`. Not read from the file;
    /// derived here because every consumer wants it.
    pub along_strike_offset_km: f32,
    /// The subfault grid, strike index fastest. One record per down-dip row, which is
    /// the order the file stores it in and the order every accumulation over it runs.
    ///
    /// Private so the layout cannot leak: reach it through [`Segment::at`],
    /// [`Segment::depth_rows`] or [`Segment::depth_rows_mut`].
    subfaults: Vec<Subfault>,
}

/// How far one subfault slipped, as the `.stoch` file gives it.
///
/// A newtype because the quantity derived from it — a subfault's share of the total moment,
/// `source::MomentWeight` — is also an `f32` and means something else entirely.
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
pub struct Slip(pub f32);

/// What the `.stoch` file says about one subfault.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Subfault {
    /// Slip. Read from the file and never modified.
    pub slip: Slip,
    /// Rise time, s.
    pub rise_time_s: f32,
    /// Rupture time relative to origin, s.
    pub rupture_time_s: f32,
}

#[bon::bon]
impl Segment {
    /// Build a segment from its geometry and the subfault grid.
    ///
    /// The grid is strike-index-fastest, one row per down-dip index — the order the
    /// `.stoch` file stores it in and the order every accumulation over it runs.
    #[builder]
    pub fn new(
        fault_lon_deg: f32,
        fault_lat_deg: f32,
        along_strike_count: usize,
        down_dip_count: usize,
        subfault_length_km: f32,
        subfault_width_km: f32,
        strike_deg: f32,
        dip_deg: f32,
        rake_deg: f32,
        top_depth_km: f32,
        hypocentre_along_strike_km: f32,
        hypocentre_down_dip_km: f32,
        subfaults: Vec<Subfault>,
    ) -> Self {
        assert_eq!(
            subfaults.len(),
            along_strike_count * down_dip_count,
            "subfault grid holds {} entries, expected {along_strike_count}x{down_dip_count}",
            subfaults.len()
        );
        Self {
            fault_lon_deg,
            fault_lat_deg,
            along_strike_count,
            down_dip_count,
            subfault_length_km,
            subfault_width_km,
            strike_deg,
            dip_deg,
            rake_deg,
            top_depth_km,
            hypocentre_along_strike_km,
            hypocentre_down_dip_km,
            along_strike_offset_km: 0.5 * along_strike_count as f32 * subfault_length_km,
            subfaults,
        }
    }

    /// The segment's plane, as the geometry sees it.
    pub fn fault_plane(&self) -> FaultPlane {
        FaultPlane {
            origin: GeoPoint {
                lat_deg: self.fault_lat_deg,
                lon_deg: self.fault_lon_deg,
            },
            strike_deg: self.strike_deg,
            dip_deg: self.dip_deg,
            top_depth_km: self.top_depth_km,
            along_strike_offset_km: self.along_strike_offset_km,
            subfault_length_km: self.subfault_length_km,
            subfault_width_km: self.subfault_width_km,
            along_strike_count: self.along_strike_count,
            down_dip_count: self.down_dip_count,
        }
    }

    /// Subfault count, `along_strike_count * down_dip_count`.
    pub fn subfault_total(&self) -> usize {
        self.subfaults.len()
    }

    /// Flat offset of subfault `(i, j)`. This is a 1-based scheme, see [`Segment::at`].
    #[inline]
    pub fn grid_index(&self, along_strike: usize, down_dip: usize) -> usize {
        assert!(
            (1..=self.along_strike_count).contains(&along_strike)
                && (1..=self.down_dip_count).contains(&down_dip),
            "subfault ({along_strike},{down_dip}) is outside the {}x{} grid",
            self.along_strike_count,
            self.down_dip_count
        );
        (down_dip - 1) * self.along_strike_count + (along_strike - 1)
    }

    /// Subfault `along_strike` (`1..=along_strike_count`) at depth row `down_dip`
    /// (`1..=down_dip_count`).
    #[inline]
    pub fn at(&self, along_strike: usize, down_dip: usize) -> Subfault {
        self.subfaults[self.grid_index(along_strike, down_dip)]
    }

    /// The grid as one contiguous run per depth row, shallowest first.
    pub fn depth_rows(&self) -> impl Iterator<Item = &[Subfault]> {
        self.subfaults.chunks(self.along_strike_count)
    }

    /// [`Segment::depth_rows`], mutably.
    pub fn depth_rows_mut(&mut self) -> impl Iterator<Item = &mut [Subfault]> {
        self.subfaults.chunks_mut(self.along_strike_count)
    }

    /// Subfault indices `(i, j)` with the depth index outermost: `j` varies
    /// slowest, `i` fastest.
    pub fn depth_major(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (along_strike_count, down_dip_count) = (self.along_strike_count, self.down_dip_count);
        (1..=down_dip_count).flat_map(move |j| (1..=along_strike_count).map(move |i| (i, j)))
    }

    /// Subfault indices `(i, j)` with the strike index outermost: `i` varies
    /// slowest, `j` fastest.
    pub fn strike_major(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (along_strike_count, down_dip_count) = (self.along_strike_count, self.down_dip_count);
        (1..=along_strike_count).flat_map(move |i| (1..=down_dip_count).map(move |j| (i, j)))
    }
}

/// The whole slip model.
#[derive(Clone, Debug)]
pub struct SlipModel {
    pub segments: Vec<Segment>,
    /// Total subfault count across all segments.
    pub subfault_count: usize,
    /// Total fault area, km².
    pub fault_area_km2: f32,
    /// Deepest hypocentre over the segments.
    pub max_hypocentre_depth_km: f32,
}

impl SlipModel {
    /// Assemble a slip model from its segments, deriving the three aggregates.
    pub fn new(segments: Vec<Segment>) -> Self {
        let mut subfault_count = 0usize;
        let mut fault_area_km2 = 0.0f32;
        let mut max_hypocentre_depth_km = 0.0f32;

        for seg in &segments {
            subfault_count += seg.along_strike_count * seg.down_dip_count;
            fault_area_km2 += seg.along_strike_count as f32
                * seg.subfault_length_km
                * seg.down_dip_count as f32
                * seg.subfault_width_km;

            let zhyp =
                seg.top_depth_km + seg.hypocentre_down_dip_km / seg.dip_deg.to_radians().sin();
            if zhyp > max_hypocentre_depth_km {
                max_hypocentre_depth_km = zhyp;
            }
        }

        Self {
            segments,
            subfault_count,
            fault_area_km2,
            max_hypocentre_depth_km,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_subfault_orders_are_transposes_of_each_other() {
        // A 3x2 grid: same set of indices, opposite traversal. The orders are not
        // interchangeable at the call sites -- see Segment::strike_major -- so this
        // pins which is which.
        let s = Segment {
            fault_lon_deg: 0.0,
            fault_lat_deg: 0.0,
            along_strike_count: 3,
            down_dip_count: 2,
            subfault_length_km: 1.0,
            subfault_width_km: 1.0,
            strike_deg: 0.0,
            dip_deg: 90.0,
            rake_deg: 0.0,
            top_depth_km: 0.0,
            hypocentre_along_strike_km: 0.0,
            hypocentre_down_dip_km: 0.0,
            along_strike_offset_km: 0.0,
            subfaults: vec![Subfault::default(); 6],
        };
        assert_eq!(
            s.depth_major().collect::<Vec<_>>(),
            [(1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)],
            "depth_major must vary i fastest"
        );
        assert_eq!(
            s.strike_major().collect::<Vec<_>>(),
            [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)],
            "strike_major must vary j fastest"
        );
        // Same index set either way.
        let mut a: Vec<_> = s.depth_major().collect();
        let mut b: Vec<_> = s.strike_major().collect();
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }
}
