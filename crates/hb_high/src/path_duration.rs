//! The path-duration model: how the shaping window lengthens with distance.

/// The path-duration model: how record duration grows with distance.
///
/// This is the `c₁·R` term of Graves & Pitarka (2010) eq. 17, `T_di = f_ci⁻¹ + c₁R_i`,
/// generalised to a piecewise-linear table so that the Boore & Thompson models can be selected
/// instead. See `PHYSICS.md` §7.
///
/// # Sources
///
/// * Graves, R. W. & Pitarka, A. (2010). Broadband ground-motion simulation using a hybrid
///   approach. *BSSA* **100**(5A), 2095–2123. doi:10.1785/0120100057 — eq. 17.
/// * Boore, D. M. & Thompson, E. M. (2014). Path durations for use in the stochastic-method
///   simulation of ground motions. *BSSA* **104**(5), 2541–2552. doi:10.1785/0120140058 —
///   Table 1, "The New Path Duration Model".
/// * Boore, D. M. & Thompson, E. M. (2015). Revisions to some parameters used in
///   stochastic-method simulations of ground motion. *BSSA* **105**(2A), 1029–1041.
///   doi:10.1785/0120140281 — Table 3, "The Path Duration Model for Stable Continental
///   Regions".
///
/// The wire encoding is the non-contiguous integer set `0`/`1`/`2`/`11`/`12`; see
/// [`PathDurationModel::from_code`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PathDurationModel {
    /// `<= 0` — Graves & Pitarka (2010) eq. 17, single segment, slope `c₁ = 0.063` s/km.
    ///
    /// Verified exact against the paper.
    Gp2010,
    /// `1` — western US, slope 0.070 s/km.
    ///
    /// No published source: models 1 and 2 are the Graves & Pitarka slope adjusted by hand
    /// ("WUS/ENA modification trial/error"), not a model from the literature.
    Wus,
    /// `2` — eastern North America, slope 0.100 s/km. Hand-adjusted like [`Self::Wus`], with
    /// no published source.
    Ena,
    /// `11` — Boore & Thompson (2014) Table 1, active crustal regions.
    ///
    /// Breakpoints at 0, 7, 45, 125, 175 and 270 km, with durations 0, 2.4, 8.4, 10.9, 17.4
    /// and 34.2 s, linearly interpolated between, as Table 1 specifies.
    ///
    /// The extrapolation beyond 270 km does not match the paper, which gives a tail slope of
    /// 0.156 s/km against the 0.177 the table builder repeats (the final segment's slope).
    /// Production behaves the same way, so this is kept to match the archived catalogue. It
    /// lengthens path durations past 270 km (about +6.9 s at 600 km) and lowers peak
    /// amplitudes by a few percent at stations where most subfaults are that far away.
    ///
    /// This is the model production runs.
    Bt2014Wus,
    /// `12` — Boore & Thompson (2015) Table 3, stable continental regions.
    ///
    /// Breakpoints at 0, 15, 35, 50, 125, 200, 392 and 600 km, with durations 0, 2.6, 17.5,
    /// 25.1, 25.1, 28.5, 46.0 and 69.1 s, linearly interpolated, with the `D_P(R) =
    /// D_P(R_last) + 0.111(R − R_last)` tail of Table 3.
    ///
    /// Unlike model 11 the tail matches the paper, because its 0.111 s/km equals the final
    /// tabulated segment's slope, which is what the table builder repeats.
    Bt2015Ena,
}

impl PathDurationModel {
    /// Decode the wire integer. Everything `<= 0` maps to `Gp2010`; any other value outside
    /// the set is rejected.
    pub fn from_code(v: i32) -> Option<Self> {
        Some(match v {
            i32::MIN..=0 => Self::Gp2010,
            1 => Self::Wus,
            2 => Self::Ena,
            11 => Self::Bt2014Wus,
            12 => Self::Bt2015Ena,
            _ => return None,
        })
    }
}

/// One segment of the piecewise-linear duration-versus-distance table.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DurationSegment {
    /// Distance at which this segment starts, km.
    start_km: f32,
    /// Duration at that distance, s.
    duration_s: f32,
    /// Slope of this segment, s/km.
    slope_s_per_km: f32,
}

/// A path-duration model as a piecewise-linear duration-versus-distance table, ascending in
/// distance. The largest model uses eight segments.
#[derive(Clone, Debug, PartialEq)]
pub struct PathDuration {
    segments: Vec<DurationSegment>,
}

impl PathDuration {
    /// Build the table. See [`PathDurationModel`] for the sources of each model and what has
    /// been verified.
    pub fn new(model: PathDurationModel) -> Self {
        let segments = match model {
            // Graves & Pitarka (2010) eq. 17: `T_di = f_ci^-1 + c1*R_i` with `c1 = 0.063`.
            PathDurationModel::Gp2010 => constant_slope(0.063),
            PathDurationModel::Wus => constant_slope(0.07),
            PathDurationModel::Ena => constant_slope(0.1),
            // Boore & Thompson (2014) Table 1, "The New Path Duration Model" (p. 2546):
            // breakpoints at 0, 7, 45, 125, 175, 270 km with durations 0, 2.4, 8.4, 10.9, 17.4,
            // 34.2 s, linearly interpolated.
            //
            // Deviates from the paper beyond 270 km. Table 1 gives "slope of last segment 0.156"
            // s/km past the last breakpoint; `from_breakpoints` instead copies the final
            // tabulated segment's slope, `(34.2 - 17.4)/(270 - 175) = 0.177`, about 13%
            // steeper. Production does the same, so this is kept to match the archived
            // catalogue. It is not an edge case: an Alpine Fault rupture recorded past Cook
            // Strait has every subfault beyond 270 km.
            PathDurationModel::Bt2014Wus => from_breakpoints(
                [0.0, 7.0, 45.0, 125.0, 175.0, 270.0],
                [0.0, 2.4, 8.4, 10.9, 17.4, 34.2],
            ),
            // Boore & Thompson (2015) Table 3 (p. 1034), stable continental regions. The
            // paper's tail is `D_P(R_last) + 0.111(R - R_last)`, and the final tabulated
            // segment's slope is `(69.1 - 46.0)/(600 - 392) = 0.1111` -- so repeating it is
            // correct here, where for model 11 above it is not.
            PathDurationModel::Bt2015Ena => from_breakpoints(
                [0.0, 15.0, 35.0, 50.0, 125.0, 200.0, 392.0, 600.0],
                [0.0, 2.6, 17.5, 25.1, 25.1, 28.5, 46.0, 69.1],
            ),
        };
        Self { segments }
    }

    /// The path duration at `distance_km`, s — the `c₁R` term of Graves & Pitarka (2010)
    /// eq. 17, generalised.
    ///
    /// Reads the last table segment this distance is past (`.last()`: the table ascends). The
    /// comparison is strict `>`, so a distance of exactly the first breakpoint (0.0) matches
    /// nothing and the duration is zero.
    pub fn at(&self, distance_km: f32) -> f32 {
        let segment = self
            .segments
            .iter()
            .take_while(|segment| distance_km > segment.start_km)
            .last()
            .copied()
            .unwrap_or(DurationSegment {
                start_km: 0.0,
                duration_s: 0.0,
                slope_s_per_km: 0.0,
            });
        segment.duration_s + segment.slope_s_per_km * (distance_km - segment.start_km)
    }
}

/// The single-segment models give their slope directly and have no breakpoints.
fn constant_slope(slope_s_per_km: f32) -> Vec<DurationSegment> {
    vec![DurationSegment {
        start_km: 0.0,
        duration_s: 0.0,
        slope_s_per_km,
    }]
}

/// The multi-segment models give (distance, duration) breakpoints; the slopes are
/// differenced from consecutive pairs.
///
/// The last segment repeats the previous slope so distances past the table
/// extrapolate rather than flatten — which is why the fold looks one short and then
/// pushes a copy.
fn from_breakpoints<const N: usize>(
    start_km: [f32; N],
    duration_s: [f32; N],
) -> Vec<DurationSegment> {
    let mut table: Vec<DurationSegment> = start_km
        .windows(2)
        .zip(duration_s.windows(2))
        .map(|(r, d)| DurationSegment {
            start_km: r[0],
            duration_s: d[0],
            slope_s_per_km: (d[1] - d[0]) / (r[1] - r[0]),
        })
        .collect();
    let last_slope = table
        .last()
        .expect("a breakpoint table has at least two entries")
        .slope_s_per_km;
    table.push(DurationSegment {
        start_km: start_km[N - 1],
        duration_s: duration_s[N - 1],
        slope_s_per_km: last_slope,
    });
    table
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_duration_accepts_only_the_five_documented_values() {
        assert_eq!(
            PathDurationModel::from_code(0),
            Some(PathDurationModel::Gp2010)
        );
        assert_eq!(
            PathDurationModel::from_code(-7),
            Some(PathDurationModel::Gp2010)
        );
        assert_eq!(
            PathDurationModel::from_code(11),
            Some(PathDurationModel::Bt2014Wus)
        );
        // Values outside the documented set are rejected.
        for bad in [3, 5, 10, 13, 99] {
            assert_eq!(
                PathDurationModel::from_code(bad),
                None,
                "{bad} should be rejected"
            );
        }
    }
}
