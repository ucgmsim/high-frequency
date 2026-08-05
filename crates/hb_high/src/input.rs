//! The source and receiver data model: the slip model, the 1-D velocity model, and a
//! station.
//!
//! This was `input.rs` because it held the readers for the three text files the Fortran
//! deck named. §4.3 deleted them -- Python owns every file format now, reusing
//! `source_modelling.stoch.StochFile` and `workflow.realisations.HFVelocityModel1D` -- and
//! what is left is the data model plus the two derivations that belong next to the physics
//! consuming them: Moho truncation and the air layer.

use crate::state::{params, VelocityModelInput};

/// One fault segment from the `.stoch` file.
///
/// Slip, rise time and rupture time are stored as a subfault grid — along-strike by
/// down-dip — rather than in the Fortran's `(lv, nq, np)` block.
///
/// That layout change is deliberate and safe. The Fortran declares
/// `sddp(lv,nq,np)` with `lv=1000, nq=600, np=100`: **240 MB per array, 720 MB
/// for the three**, almost all of it untouched. Layout is only observable where
/// the original indexes out of bounds (as it does for `stdd`, see
/// `PORTING_RULES.md` §7), and these three are always indexed inside the real grid.
/// So compacting them changes no arithmetic.
///
/// Field names are the port's, not the Fortran's; each one's original spelling is in
/// its doc comment so a reader can still find it in `hb_high_ref.f`.
#[derive(Clone, Debug)]
pub struct Segment {
    /// `elonq` — longitude of the segment's along-strike reference point.
    pub fault_lon_deg: f32,
    /// `elatq` — latitude of the same point.
    pub fault_lat_deg: f32,
    /// `nx` — subfault count along strike.
    pub along_strike_count: usize,
    /// `nw` — subfault count down dip.
    pub down_dip_count: usize,
    /// `dx` — subfault dimension along strike, km.
    pub subfault_length_km: f32,
    /// `dw` — subfault dimension down dip, km.
    pub subfault_width_km: f32,
    /// `strq` — strike, degrees clockwise from north.
    pub strike_deg: f32,
    /// `dipq` — dip, degrees from horizontal.
    pub dip_deg: f32,
    /// `rakeq` — rake, degrees.
    pub rake_deg: f32,
    /// `dtop` — depth to the top edge of the segment, km.
    pub top_depth_km: f32,
    /// `shyp` — hypocentre offset along strike from the segment centre, km.
    pub hypocentre_along_strike_km: f32,
    /// `dhyp` — hypocentre offset down dip from the top edge, km.
    pub hypocentre_down_dip_km: f32,
    /// `astop` — half the fault length along strike,
    /// `0.5 * along_strike_count * subfault_length_km`. Not read from the file;
    /// derived here because every consumer wants it.
    pub along_strike_offset_km: f32,
    /// The subfault grid, strike index fastest — one record per down-dip row, which is
    /// the order the file stores it in and the order every accumulation over it runs.
    ///
    /// Private so the layout cannot leak: reach it through [`Segment::at`],
    /// [`Segment::depth_rows`] or [`Segment::depth_rows_mut`].
    subfaults: Vec<Subfault>,
}

/// What the `.stoch` file says about one subfault.
///
/// The Fortran keeps `sddp`, `rist` and `rupt` as three separate `(lv, nq, np)` blocks,
/// but every read of one is at the same `(i, j)` as the other two, so this is one value
/// per subfault — §2.3.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Subfault {
    /// `sddp` — slip.
    ///
    /// **This field changes meaning partway through a run.** It holds slip as read from
    /// the file until `sim::normalise_source`, which converts it in place to
    /// relative moment and then rescales it to unit mean. Everything downstream of that
    /// call is reading moment weights, not slip. The Fortran does the same thing to the
    /// same array; naming it `slip_cm` would be a lie for most of the program's life,
    /// which is why the units tag is absent here.
    pub slip: f32,
    /// `rist` — rise time, s.
    pub rise_time_s: f32,
    /// `rupt` — rupture time relative to origin, s.
    pub rupture_time_s: f32,
}

/// `#[bon::bon]` transforms only the `#[builder]`-marked function below; every other
/// method in this block is left exactly as written.
///
/// A segment needs twelve geometry scalars and a grid. As positional arguments that is a
/// row of bare floats in which `strike_deg`, `dip_deg` and `rake_deg` are mutually
/// swappable without a type error — the exact shape of bug the deck format produced for
/// twenty years. `Segment::builder().strike_deg(...)` cannot make that mistake.
#[bon::bon]
impl Segment {
    /// Build a segment from its geometry and the subfault grid.
    ///
    /// The grid is strike-index-fastest, one row per down-dip index — the order the
    /// `.stoch` file stores it in and the order every accumulation over it runs.
    /// `along_strike_offset_km` is derived here rather than supplied, because it is a
    /// function of the geometry and every consumer wants it.
    ///
    /// This is the constructor the Python boundary uses. `read_stoch` was previously
    /// the only way to obtain a `Segment`, which tied the data model to a text format.
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

    /// Subfault count, `nx * nw`.
    pub fn subfault_total(&self) -> usize {
        self.subfaults.len()
    }

    /// Flat offset of subfault `(i, j)` — 1-based, as [`Segment::at`] documents.
    ///
    /// Public because `sim` lays its own per-subfault grids (the time windows) over the
    /// same shape and must agree on the ordering. Nothing else should need it.
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

    /// Subfault `along_strike` (`1..=nx`) at depth row `down_dip` (`1..=nw`).
    ///
    /// **1-based, deliberately.** These are subfault *numbers*, not storage offsets:
    /// the along-strike coordinate of subfault `i` is `(i - 0.5) * length`, so the
    /// numbering is part of the physics rather than an artifact of Fortran. See
    /// [`crate::geom::SubfaultGeometry`], which makes the same choice for the same
    /// reason and explains it at length.
    #[inline]
    pub fn at(&self, along_strike: usize, down_dip: usize) -> Subfault {
        self.subfaults[self.grid_index(along_strike, down_dip)]
    }

    /// The grid as one contiguous run per depth row, shallowest first.
    ///
    /// This is what the two source-normalisation accumulations want: they sum in
    /// depth-major order, which *is* storage order, so they can walk the slice and never
    /// compute an index. Floating-point summation is order-dependent, so that
    /// correspondence is load-bearing, not a convenience.
    pub fn depth_rows(&self) -> impl Iterator<Item = &[Subfault]> {
        self.subfaults.chunks(self.along_strike_count)
    }

    /// [`Segment::depth_rows`], mutably.
    pub fn depth_rows_mut(&mut self) -> impl Iterator<Item = &mut [Subfault]> {
        self.subfaults.chunks_mut(self.along_strike_count)
    }

    /// Subfault indices `(i, j)` with the **depth** index outermost: `j` varies
    /// slowest, `i` fastest.
    ///
    /// This is the order the time-window pass walks the grid.
    pub fn depth_major(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (along_strike_count, down_dip_count) = (self.along_strike_count, self.down_dip_count);
        (1..=down_dip_count).flat_map(move |j| (1..=along_strike_count).map(move |i| (i, j)))
    }

    /// Subfault indices `(i, j)` with the **strike** index outermost: `i` varies
    /// slowest, `j` fastest.
    ///
    /// This is the order the subfault pass walks the grid, and the fact that it is
    /// the *opposite* of [`Segment::depth_major`] is load-bearing rather than
    /// incidental: the subfault pass advances `irandcnt` once per surviving
    /// subfault and uses it to index `fgrand`, so walking the grid the other way
    /// would pair a different normal deviate with each subfault's rupture-velocity
    /// perturbation, and every waveform would change.
    ///
    /// Neither iterator borrows the segment — both capture the two counts by value — so a
    /// caller can mutate `slip` while iterating.
    pub fn strike_major(&self) -> impl Iterator<Item = (usize, usize)> + use<> {
        let (along_strike_count, down_dip_count) = (self.along_strike_count, self.down_dip_count);
        (1..=along_strike_count).flat_map(move |i| (1..=down_dip_count).map(move |j| (i, j)))
    }
}

/// The whole slip model.
#[derive(Clone, Debug)]
pub struct StochModel {
    pub segments: Vec<Segment>,
    /// Total subfault count across all segments.
    pub subfault_count: usize,
    /// Total fault area, km².
    pub fault_area_km2: f32,
    /// Deepest hypocentre over the segments.
    pub max_hypocentre_depth_km: f32,
}

impl StochModel {
    /// Assemble a slip model from its segments, deriving the three aggregates.
    ///
    /// Each aggregate accumulates in segment order. Integer addition is exact and the
    /// area sum is a reduction over a handful of terms, so the order is not delicate —
    /// but it is the order `read_stoch` used, and keeping it means the two agree bit for
    /// bit on any model either can express.
    ///
    /// `deg_to_rad` is a parameter rather than a constant because the hypocentre depth
    /// needs `sin(dip)`, and the caller owns the degree convention.
    pub fn new(segments: Vec<Segment>, deg_to_rad: f32) -> Self {
        let mut subfault_count = 0usize;
        let mut fault_area_km2 = 0.0f32;
        let mut max_hypocentre_depth_km = 0.0f32;

        for seg in &segments {
            subfault_count += seg.along_strike_count * seg.down_dip_count;
            fault_area_km2 += seg.along_strike_count as f32 * seg.subfault_length_km
                * seg.down_dip_count as f32 * seg.subfault_width_km;

            // 2014-12-19: this was '*' and should have been '/'; fixed upstream.
            let zhyp = seg.top_depth_km
                + seg.hypocentre_down_dip_km / (seg.dip_deg * deg_to_rad).sin();
            if zhyp > max_hypocentre_depth_km {
                max_hypocentre_depth_km = zhyp;
            }
        }

        Self { segments, subfault_count, fault_area_km2, max_hypocentre_depth_km }
    }
}


/// Why a velocity model cannot be used as given.
///
/// Deliberately **not** a `DeckError`: that type describes a list-directed text read and
/// lives in `deck.rs`, which §4.3 deletes. The surviving data model must not depend on the
/// dying one.
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("velocity model has no layers")]
    NoLayers,
    #[error("velocity model has {count} layers, exceeding nlaymax = {max}")]
    TooManyLayers { count: usize, max: usize },
    #[error(
        "the first layer already reaches vs_moho = {vs_moho_km_s} km/s, so there is no \
         model above the Moho to simulate"
    )]
    MohoAtFirstLayer { vs_moho_km_s: f64 },
}

/// Build the velocity model from layer records, returning the layer count after Moho
/// truncation.
///
/// Depth accumulation, truncation at the first layer reaching `vs_moho_km_s`, and the
/// zero-thickness bottom layer that makes reflected rays come out right (2016-08-03).
///
/// This briefly duplicated `read_velocity_model`, which did the same three things
/// interleaved with parsing. §4.3 deleted the reader, so this is now the only
/// implementation — and `the_array_path_and_the_text_path_agree_field_for_field` proved
/// they agreed field-for-field before the reader went.
pub fn build_velocity_model(
    vmod_in: &mut VelocityModelInput,
    layers: &[crate::state::InputLayer],
    vs_moho_km_s: f64,
) -> Result<usize, ModelError> {
    if layers.is_empty() {
        return Err(ModelError::NoLayers);
    }
    if layers.len() > params::NLAYMAX {
        return Err(ModelError::TooManyLayers {
            count: layers.len(),
            max: params::NLAYMAX,
        });
    }

    let mut layer_count = layers.len();
    for (i, layer) in layers.iter().enumerate() {
        vmod_in[i] = *layer;
        vmod_in[i].depth_km = layer.thickness_km;
        if i > 0 {
            vmod_in[i].depth_km += vmod_in[i - 1].depth_km;
        }

        if layer.vsh_km_s >= vs_moho_km_s {
            if i == 0 {
                // The Fortran reads depth_km(0) here, one before the array start. Not
                // reachable with the production vs_moho of 999.9, and an error rather
                // than a silent out-of-bounds read.
                return Err(ModelError::MohoAtFirstLayer { vs_moho_km_s });
            }
            // A LAYER COUNT, so one more than the 0-based index that reached it.
            layer_count = i + 1;
            vmod_in[i].thickness_km = 0.0;
            vmod_in[i].depth_km = vmod_in[i - 1].depth_km;
            break;
        }
    }

    vmod_in[layer_count - 1].thickness_km = 0.0;
    Ok(layer_count)
}


/// Insert the thin "air" layer at the top of the model — `hb_high_ref.f:494-513`.
///
/// Needed to get the correct free-surface reflection coefficient for
/// surface-reflected rays. Returns the updated `(layer_count, skip_layers)`.
///
/// This fires in production: the standard 34-layer model has
/// `depth_km(1) = 0.05` and `vp_km_s(1) = 1.8`, so `layer_count` becomes 35 and `skip_layers` goes
/// from -99 to -98 (still negative, so `grandvel` stays dead).
///
/// # The air layer's Q is never set, and it does not matter
///
/// The shift copies seven fields down but only **five** are overwritten at index 0, so
/// `attenuation_p` and `attenuation_s` there keep the original first layer's values
/// instead of air-like ones. §3.4 looked at fixing this and found there is nothing to
/// fix: **neither field is ever read at index 0.** `attenuation_p` has no live reader at
/// all, and `attenuation_s` is read only at `vmod[nh1]` and `vmod[nhj]` in
/// `geometric_spreading`, where the layer indices come from `green_function`'s ray
/// building and are never below `krec = 1`.
///
/// So the value is arbitrary and unobservable. Left alone rather than changed, because
/// changing data nothing reads is risk without benefit.
pub fn insert_air_layer(vmod_in: &mut VelocityModelInput, layer_count: usize, skip_layers: i32) -> (usize, i32) {
    if !(vmod_in[0].depth_km > 0.001 && vmod_in[0].vp_km_s > 0.01) {
        return (layer_count, skip_layers);
    }
    let layer_count = layer_count + 1;
    let skip_layers = skip_layers + 1;

    // Shift down from the back so nothing is overwritten before it is copied. 0-based,
    // the Fortran's `DO i = layer_count, 2, -1` is `layer_count - 1` down to `1`.
    //
    // One whole layer per step, where this used to be seven field assignments. That is
    // the difference that matters below: the shift moves ALL seven fields, and only five
    // are then overwritten at index 0, so `attenuation_p`/`attenuation_s` there keep the
    // original first layer's values. With parallel arrays that was an absence a reader
    // had to notice; here it is a visibly partial update of a whole struct.
    for i in (1..layer_count).rev() {
        vmod_in[i] = vmod_in[i - 1];
    }

    // depth_km and thickness_km are real*4, so these literals are already f32.
    vmod_in[0].depth_km = 0.0001;
    vmod_in[0].thickness_km = 0.0001;
    // vp_km_s, vsh_km_s and density_g_cm3 are real*8, but the Fortran literals are UNSUFFIXED
    // and therefore only carry f32 precision -- PORTING_RULES.md §1b. Writing
    // 0.001f64 here gives 0.001 exactly; the Fortran stores
    // 0.0010000000474974513. Caught by the reader golden.
    vmod_in[0].vp_km_s = 0.001f32 as f64;
    vmod_in[0].vsh_km_s = 0.0005f32 as f64;
    vmod_in[0].density_g_cm3 = 0.001f32 as f64;
    // attenuation_p/attenuation_s at index 0 are deliberately not set. Nothing reads
    // them -- see the doc comment.

    (layer_count, skip_layers)
}

/// One station.
#[derive(Clone, Debug)]
pub struct Station {
    pub stlon: f32,
    pub stlat: f32,
    /// Name, `character*12` in the Fortran.
    pub cap: String,
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Layer records built the way a Python caller builds them, for the tests that used to
    /// start from a text model: `(thickness_km, vp, vsh, density, qp, qs)` per layer.
    fn layers(rows: &[(f32, f64, f64, f64, f32, f32)]) -> Vec<crate::state::InputLayer> {
        rows.iter()
            .map(|&(thickness_km, vp_km_s, vsh_km_s, density_g_cm3, qp, qs)| {
                crate::state::InputLayer {
                    // Derived by build_velocity_model, so deliberately not supplied.
                    depth_km: 0.0,
                    thickness_km,
                    vp_km_s,
                    vsh_km_s,
                    density_g_cm3,
                    attenuation_p: qp,
                    attenuation_s: qs,
                }
            })
            .collect()
    }

    #[test]
    fn the_two_subfault_orders_are_transposes_of_each_other() {
        // A 3x2 grid: same set of indices, opposite traversal. The orders are not
        // interchangeable at the call sites -- see Segment::strike_major -- so this
        // pins which is which.
        let s = Segment {
            fault_lon_deg: 0.0, fault_lat_deg: 0.0,
            along_strike_count: 3, down_dip_count: 2,
            subfault_length_km: 1.0, subfault_width_km: 1.0,
            strike_deg: 0.0, dip_deg: 90.0, rake_deg: 0.0, top_depth_km: 0.0,
            hypocentre_along_strike_km: 0.0, hypocentre_down_dip_km: 0.0,
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
        a.sort(); b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn a_moho_in_the_first_layer_is_an_error_not_a_panic() {
        // The Fortran reads depth_km(0) here, out of bounds. The array path
        // returns an error instead: it faces untrusted input from Python.
        let layers = [crate::state::InputLayer {
            depth_km: 0.0,
            thickness_km: 1.0,
            vp_km_s: 8.0,
            vsh_km_s: 4.6,
            density_g_cm3: 3.3,
            attenuation_p: 400.0,
            attenuation_s: 200.0,
        }];
        let mut v = VelocityModelInput::new();
        assert!(matches!(
            build_velocity_model(&mut v, &layers, 4.0),
            Err(ModelError::MohoAtFirstLayer { .. })
        ));
        assert!(matches!(
            build_velocity_model(&mut v, &[], 999.9),
            Err(ModelError::NoLayers)
        ));
    }

    #[test]
    fn velocity_model_truncates_at_the_moho_and_zeroes_the_base() {
        let model = layers(&[
            (1.0, 2.0, 1.0, 2.0, 100.0, 50.0),
            (2.0, 4.0, 2.5, 2.5, 200.0, 100.0),
            (3.0, 8.0, 4.6, 3.3, 400.0, 200.0),
        ]);
        let mut v = VelocityModelInput::new();
        // vsmoho below the third layer's 4.6 truncates there.
        let j0 = build_velocity_model(&mut v, &model, 4.0).unwrap();
        assert_eq!(j0, 3);
        // `j0` is a COUNT, so the Moho layer it stops at is index `j0 - 1`. Note the
        // first of these would also have passed against index 3, on an element the
        // reader never wrote -- an NLAYMAX-sized buffer will happily read zero one past
        // the model, which is why the second assertion compares two real values.
        assert_eq!(v[j0 - 1].thickness_km, 0.0, "the Moho layer is zeroed");
        assert_eq!(v[j0 - 1].depth_km, v[j0 - 2].depth_km);
    }

    #[test]
    fn velocity_model_without_moho_still_zeroes_the_base() {
        let model = layers(&[(1.0, 2.0, 1.0, 2.0, 100.0, 50.0), (2.0, 4.0, 2.5, 2.5, 200.0, 100.0)]);
        let mut v = VelocityModelInput::new();
        let j0 = build_velocity_model(&mut v, &model, 999.9).unwrap();
        assert_eq!(j0, 2);
        // 0-based since §2.3: the base of a 2-layer model is index 1, and the first
        // layer's cumulative depth is index 0.
        assert_eq!(v[1].thickness_km, 0.0);
        assert_eq!(v[0].depth_km, 1.0);
    }

    #[test]
    fn air_layer_is_inserted_for_a_realistic_model() {
        let model = layers(&[(0.05, 1.8, 0.5, 1.81, 116.0, 58.0), (2.0, 4.0, 2.5, 2.5, 200.0, 100.0)]);
        let mut v = VelocityModelInput::new();
        let j0 = build_velocity_model(&mut v, &model, 999.9).unwrap();
        let qp1_before = v[0].attenuation_p;
        let (j0b, nlskip) = insert_air_layer(&mut v, j0, -99);
        assert_eq!(j0b, j0 + 1, "production models do get the air layer");
        assert_eq!(nlskip, -98, "still negative, so grandvel stays dead");
        assert_eq!(v[0].thickness_km, 0.0001);
        // Not 0.001f64: the Fortran literal is unsuffixed in a real*8 context,
        // so it carries only f32 precision. See PORTING_RULES.md §1b.
        assert_eq!(v[0].vp_km_s, 0.001f32 as f64);
        assert_eq!(v[0].vsh_km_s, 0.0005f32 as f64);
        assert_eq!(v[0].density_g_cm3, 0.001f32 as f64);
        assert_eq!(v[1].thickness_km, 0.05, "the original first layer shifted down");
        // The shift copies seven fields but only five are overwritten, so Q
        // stays put.
        assert_eq!(
            v[0].attenuation_p, qp1_before,
            "the air layer's Q is left as-is; nothing reads it, so this pins the shift \
             rather than a physical choice"
        );
    }

    #[test]
    fn air_layer_is_skipped_when_the_model_starts_at_the_surface() {
        let model = layers(&[(0.0, 1.8, 0.5, 1.81, 116.0, 58.0), (2.0, 4.0, 2.5, 2.5, 200.0, 100.0)]);
        let mut v = VelocityModelInput::new();
        let j0 = build_velocity_model(&mut v, &model, 999.9).unwrap();
        let (j0b, nlskip) = insert_air_layer(&mut v, j0, -99);
        assert_eq!((j0b, nlskip), (j0, -99));
    }
}
