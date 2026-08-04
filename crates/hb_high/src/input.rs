//! Readers for the three input files: the `.stoch` slip model, the 1-D velocity
//! model, and the station list.

use crate::deck::{DeckError, ListReader};
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
    /// the file until [`crate::sim::normalise_source`], which converts it in place to
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

impl Segment {
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

/// Read a `.stoch` file — `hb_high_ref.f:253-285`.
///
/// Produced from an SRF by `srf2stoch`. Format: segment count, then per segment
/// a two-line header followed by three blocks of one record per down-dip row, each
/// record holding one value per along-strike column (slip, rise time, rupture time).
///
/// `deg_to_rad` is the caller's degrees-to-radians factor; the Fortran uses its own
/// `3.1415926/180` from `:150`.
pub fn read_stoch(text: &str, deg_to_rad: f32) -> Result<StochModel, DeckError> {
    let mut r = ListReader::new(text);
    let nevnt = r.i32()? as usize;

    let mut segments = Vec::with_capacity(nevnt);
    let mut subfault_count = 0usize;
    let mut fault_area_km2 = 0.0f32;
    let mut max_hypocentre_depth_km = 0.0f32;

    for _ in 0..nevnt {
        let v = r.read_values(6)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        let fault_lon_deg = crate::deck::parse_f32(g(0))?;
        let fault_lat_deg = crate::deck::parse_f32(g(1))?;
        let along_strike_count = crate::deck::parse_i32(g(2))? as usize;
        let down_dip_count = crate::deck::parse_i32(g(3))? as usize;
        let subfault_length_km = crate::deck::parse_f32(g(4))?;
        let subfault_width_km = crate::deck::parse_f32(g(5))?;

        let v = r.read_values(6)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        let strike_deg = crate::deck::parse_f32(g(0))?;
        let dip_deg = crate::deck::parse_f32(g(1))?;
        let rake_deg = crate::deck::parse_f32(g(2))?;
        let top_depth_km = crate::deck::parse_f32(g(3))?;
        let hypocentre_along_strike_km = crate::deck::parse_f32(g(4))?;
        let hypocentre_down_dip_km = crate::deck::parse_f32(g(5))?;

        // Accumulated in the Fortran's order: `nx*nw + nstot`, not `nstot + nx*nw`.
        subfault_count = along_strike_count * down_dip_count + subfault_count;
        fault_area_km2 = along_strike_count as f32 * subfault_length_km
            * down_dip_count as f32 * subfault_width_km
            + fault_area_km2;
        let along_strike_offset_km = 0.5 * along_strike_count as f32 * subfault_length_km;

        // 2014-12-19: this was '*' and should have been '/'; fixed upstream.
        let zhyp = top_depth_km + hypocentre_down_dip_km / (dip_deg * deg_to_rad).sin();
        if zhyp > max_hypocentre_depth_km {
            max_hypocentre_depth_km = zhyp;
        }

        // Three blocks, each one record per depth row -- so the file's own order is this
        // grid's storage order, and the reader fills it front to back three times over,
        // once per field.
        let mut subfaults = vec![Subfault::default(); along_strike_count * down_dip_count];
        let fields: [fn(&mut Subfault) -> &mut f32; 3] = [
            |s| &mut s.slip,
            |s| &mut s.rise_time_s,
            |s| &mut s.rupture_time_s,
        ];
        for field in fields {
            for row in subfaults.chunks_mut(along_strike_count) {
                // One record per down-dip row, one value per along-strike column.
                let values = r.read_values(along_strike_count)?;
                for (subfault, value) in row.iter_mut().zip(&values) {
                    *field(subfault) = crate::deck::parse_f32(value.as_deref().unwrap_or(""))?;
                }
            }
        }

        segments.push(Segment {
            fault_lon_deg, fault_lat_deg,
            along_strike_count, down_dip_count, subfault_length_km, subfault_width_km,
            strike_deg, dip_deg, rake_deg, top_depth_km,
            hypocentre_along_strike_km, hypocentre_down_dip_km, along_strike_offset_km,
            subfaults,
        });
    }

    Ok(StochModel { segments, subfault_count, fault_area_km2, max_hypocentre_depth_km })
}

/// Read the 1-D velocity model into `/vmod_in/` — `hb_high_ref.f:322-349`.
///
/// Returns the layer count after Moho truncation. Layers at or below the first
/// one with `vsh_km_s >= vsmoho` are dropped, and the bottom layer is forced to zero
/// thickness so reflected rays are computed correctly (2016-08-03).
///
/// Note the mixed types on each record: `thickness_km`, `attenuation_p` and `attenuation_s` are `real*4`
/// while `vp_km_s`, `vsh_km_s` and `density_g_cm3` are `real*8` — see `state::VelocityModelInput`.
///
/// If the *first* layer already exceeds `vsmoho` the Fortran reads `depth_km(0)`,
/// one before the array start. Not reachable with the production `vsmoho` of
/// 999.9, and reproduced as a panic rather than a silent read.
///
/// Layers are 0-based since §2.3, so the Fortran's layer `i` is index `i - 1` here.
pub fn read_velocity_model(
    text: &str,
    vmod_in: &mut VelocityModelInput,
    vsmoho: f64,
) -> Result<usize, DeckError> {
    let mut r = ListReader::new(text);
    let mut j0 = r.i32()? as usize;
    assert!(
        j0 <= params::NLAYMAX,
        "velocity model has {j0} layers, exceeding nlaymax = {}",
        params::NLAYMAX
    );

    let mut jmoho = j0;
    for i in 0..j0 {
        let v = r.read_values(6)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        vmod_in.thickness_km[i] = crate::deck::parse_f32(g(0))?;
        vmod_in.vp_km_s[i] = crate::deck::parse_f64(g(1))?;
        vmod_in.vsh_km_s[i] = crate::deck::parse_f64(g(2))?;
        vmod_in.density_g_cm3[i] = crate::deck::parse_f64(g(3))?;
        vmod_in.attenuation_p[i] = crate::deck::parse_f32(g(4))?;
        vmod_in.attenuation_s[i] = crate::deck::parse_f32(g(5))?;

        vmod_in.depth_km[i] = vmod_in.thickness_km[i];
        if i > 0 {
            vmod_in.depth_km[i] += vmod_in.depth_km[i - 1];
        }

        if vmod_in.vsh_km_s[i] >= vsmoho {
            // A LAYER COUNT, so it is one more than the 0-based index that reached it.
            jmoho = i + 1;
            vmod_in.thickness_km[i] = 0.0;
            assert!(i > 0, "vsmoho reached in layer 1; the Fortran would read depth_km(0)");
            vmod_in.depth_km[i] = vmod_in.depth_km[i - 1];
            break;
        }
    }

    j0 = jmoho;
    vmod_in.thickness_km[j0 - 1] = 0.0;
    Ok(j0)
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
/// Note the shift copies seven fields down but only **five** are overwritten at
/// index 1. `attenuation_p(1)` and `attenuation_s(1)` therefore keep the original first layer's Q
/// values rather than getting air-like ones. Faithful to the Fortran.
pub fn insert_air_layer(vmod_in: &mut VelocityModelInput, layer_count: usize, skip_layers: i32) -> (usize, i32) {
    if !(vmod_in.depth_km[0] > 0.001 && vmod_in.vp_km_s[0] > 0.01) {
        return (layer_count, skip_layers);
    }
    let layer_count = layer_count + 1;
    let skip_layers = skip_layers + 1;

    // Shift down from the back so nothing is overwritten before it is copied. 0-based, the
    // Fortran's `DO i = layer_count, 2, -1` is `layer_count - 1` down to `1`.
    for i in (1..layer_count).rev() {
        vmod_in.depth_km[i] = vmod_in.depth_km[i - 1];
        vmod_in.thickness_km[i] = vmod_in.thickness_km[i - 1];
        vmod_in.vp_km_s[i] = vmod_in.vp_km_s[i - 1];
        vmod_in.vsh_km_s[i] = vmod_in.vsh_km_s[i - 1];
        vmod_in.density_g_cm3[i] = vmod_in.density_g_cm3[i - 1];
        vmod_in.attenuation_p[i] = vmod_in.attenuation_p[i - 1];
        vmod_in.attenuation_s[i] = vmod_in.attenuation_s[i - 1];
    }

    // depth_km and thickness_km are real*4, so these literals are already f32.
    vmod_in.depth_km[0] = 0.0001;
    vmod_in.thickness_km[0] = 0.0001;
    // vp_km_s, vsh_km_s and density_g_cm3 are real*8, but the Fortran literals are UNSUFFIXED
    // and therefore only carry f32 precision -- PORTING_RULES.md §1b. Writing
    // 0.001f64 here gives 0.001 exactly; the Fortran stores
    // 0.0010000000474974513. Caught by the reader golden.
    vmod_in.vp_km_s[0] = 0.001f32 as f64;
    vmod_in.vsh_km_s[0] = 0.0005f32 as f64;
    vmod_in.density_g_cm3[0] = 0.001f32 as f64;
    // attenuation_p(1) and attenuation_s(1) are deliberately not set; see the note above.

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

/// Read the station list — `hb_high_ref.f:835-857`.
///
/// Leading comment lines beginning `#` or `%` are counted and skipped. The
/// Fortran detects them with a list-directed read into a `character*256`, which
/// takes the first blank-delimited token, then tests its first character.
///
/// Records are `lon lat name`. Reading stops early at end of file — the Fortran
/// uses `end=1` to branch straight to the program's `END`, which notably skips
/// `close(22)`.
pub fn read_stations(text: &str, nsite: usize) -> Result<Vec<Station>, DeckError> {
    // Header sniff.
    let mut head_lines = 0usize;
    for line in text.lines() {
        let first_token = line.split_whitespace().next().unwrap_or("");
        let c = first_token.chars().next().unwrap_or(' ');
        if c != '#' && c != '%' {
            break;
        }
        head_lines += 1;
    }

    let mut out = Vec::with_capacity(nsite);
    for line in text.lines().skip(head_lines) {
        if out.len() == nsite {
            break;
        }
        let mut r = ListReader::new(line);
        let v = match r.read_values(3) {
            Ok(v) => v,
            // end=1: run out of stations and stop, rather than erroring.
            Err(_) => break,
        };
        out.push(Station {
            stlon: crate::deck::parse_f32(v[0].as_deref().unwrap_or(""))?,
            stlat: crate::deck::parse_f32(v[1].as_deref().unwrap_or(""))?,
            cap: v[2].as_deref().unwrap_or("").to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINI_STOCH: &str = "\
1
 -179.7826   -37.6329     2     2     1.64     1.48
 187   68  166    25.63     0.00     1.48
  7.38758e+00  5.38111e+00
  8.36237e+00  9.36874e+00
  1.32973e-01  1.25785e-01
  1.57674e-01  1.71956e-01
  6.64584e-01  6.61278e-01
  6.37069e-01  5.81083e-01
";

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
    fn reads_the_minimal_stoch_fixture() {
        let pu = 3.1415926f32 / 180.0;
        let m = read_stoch(MINI_STOCH, pu).unwrap();
        assert_eq!(m.segments.len(), 1);
        let s = &m.segments[0];
        assert_eq!((s.along_strike_count, s.down_dip_count), (2, 2));
        assert_eq!(s.fault_lon_deg, -179.7826);
        assert_eq!(s.strike_deg, 187.0);
        assert_eq!(m.subfault_count, 4);
        // Slip rows are down-dip, values along strike.
        assert_eq!(s.at(1, 1).slip, 7.38758e0);
        assert_eq!(s.at(2, 1).slip, 5.38111e0);
        assert_eq!(s.at(1, 2).slip, 8.36237e0);
        assert_eq!(s.at(1, 1).rise_time_s, 1.32973e-1);
        assert_eq!(s.at(2, 2).rupture_time_s, 5.81083e-1);
        assert_eq!(s.along_strike_offset_km, 0.5 * 2.0 * 1.64);
    }

    #[test]
    fn velocity_model_truncates_at_the_moho_and_zeroes_the_base() {
        let text = "3\n1.0 2.0 1.0 2.0 100 50\n2.0 4.0 2.5 2.5 200 100\n3.0 8.0 4.6 3.3 400 200\n";
        let mut v = VelocityModelInput::new();
        // vsmoho below the third layer's 4.6 truncates there.
        let j0 = read_velocity_model(text, &mut v, 4.0).unwrap();
        assert_eq!(j0, 3);
        // `j0` is a COUNT, so the Moho layer it stops at is index `j0 - 1`. Note the
        // first of these would also have passed against index 3, on an element the
        // reader never wrote -- an NLAYMAX-sized buffer will happily read zero one past
        // the model, which is why the second assertion compares two real values.
        assert_eq!(v.thickness_km[j0 - 1], 0.0, "the Moho layer is zeroed");
        assert_eq!(v.depth_km[j0 - 1], v.depth_km[j0 - 2]);
    }

    #[test]
    fn velocity_model_without_moho_still_zeroes_the_base() {
        let text = "2\n1.0 2.0 1.0 2.0 100 50\n2.0 4.0 2.5 2.5 200 100\n";
        let mut v = VelocityModelInput::new();
        let j0 = read_velocity_model(text, &mut v, 999.9).unwrap();
        assert_eq!(j0, 2);
        // 0-based since §2.3: the base of a 2-layer model is index 1, and the first
        // layer's cumulative depth is index 0.
        assert_eq!(v.thickness_km[1], 0.0);
        assert_eq!(v.depth_km[0], 1.0);
    }

    #[test]
    fn air_layer_is_inserted_for_a_realistic_model() {
        let text = "2\n0.05 1.8 0.5 1.81 116.0 58.0\n2.0 4.0 2.5 2.5 200 100\n";
        let mut v = VelocityModelInput::new();
        let j0 = read_velocity_model(text, &mut v, 999.9).unwrap();
        let qp1_before = v.attenuation_p[0];
        let (j0b, nlskip) = insert_air_layer(&mut v, j0, -99);
        assert_eq!(j0b, j0 + 1, "production models do get the air layer");
        assert_eq!(nlskip, -98, "still negative, so grandvel stays dead");
        assert_eq!(v.thickness_km[0], 0.0001);
        // Not 0.001f64: the Fortran literal is unsuffixed in a real*8 context,
        // so it carries only f32 precision. See PORTING_RULES.md §1b.
        assert_eq!(v.vp_km_s[0], 0.001f32 as f64);
        assert_eq!(v.vsh_km_s[0], 0.0005f32 as f64);
        assert_eq!(v.density_g_cm3[0], 0.001f32 as f64);
        assert_eq!(v.thickness_km[1], 0.05, "the original first layer shifted down");
        // The shift copies seven fields but only five are overwritten, so Q
        // stays put.
        assert_eq!(v.attenuation_p[0], qp1_before, "attenuation_p(1) is deliberately not air-like");
    }

    #[test]
    fn air_layer_is_skipped_when_the_model_starts_at_the_surface() {
        let text = "2\n0.0 1.8 0.5 1.81 116.0 58.0\n2.0 4.0 2.5 2.5 200 100\n";
        let mut v = VelocityModelInput::new();
        let j0 = read_velocity_model(text, &mut v, 999.9).unwrap();
        let (j0b, nlskip) = insert_air_layer(&mut v, j0, -99);
        assert_eq!((j0b, nlskip), (j0, -99));
    }

    #[test]
    fn station_list_skips_comment_headers() {
        let s = read_stations("# a comment\n% another\n176 -40 STATX\n177 -41 STATY\n", 2).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0].cap, "STATX");
        assert_eq!(s[1].stlat, -41.0);
    }

    #[test]
    fn station_list_stops_at_end_of_file() {
        // nsite larger than the file: the Fortran's end=1 stops the loop.
        let s = read_stations("176 -40 STATX\n", 5).unwrap();
        assert_eq!(s.len(), 1);
    }
}
