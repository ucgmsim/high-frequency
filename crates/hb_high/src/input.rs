//! Readers for the three input files: the `.stoch` slip model, the 1-D velocity
//! model, and the station list.

use crate::deck::{DeckError, ListReader};
use crate::fort::Array2;
use crate::state::{params, VmodIn};

/// One fault segment from the `.stoch` file.
///
/// Slip, rise time and rupture time are stored `(nx, nw)` — along-strike by
/// down-dip — rather than in the Fortran's `(lv, nq, np)` block.
///
/// That layout change is deliberate and safe. The Fortran declares
/// `sddp(lv,nq,np)` with `lv=1000, nq=600, np=100`: **240 MB per array, 720 MB
/// for the three**, almost all of it untouched. Layout is only observable where
/// the original indexes out of bounds (as it does for `stdd`, see
/// `PORTING_RULES.md` §7), and these three are always indexed within
/// `1..=nx`/`1..=nw`. So compacting them changes no arithmetic.
#[derive(Clone, Debug)]
pub struct Segment {
    pub elonq: f32,
    pub elatq: f32,
    pub nx: usize,
    pub nw: usize,
    pub dx: f32,
    pub dw: f32,
    pub strq: f32,
    pub dipq: f32,
    pub rakeq: f32,
    pub dtop: f32,
    pub shyp: f32,
    pub dhyp: f32,
    /// Along-strike extent of the reference point, `0.5*nx*dx`.
    pub astop: f32,
    /// Slip, indexed `(i, j)`.
    pub sddp: Array2<f32>,
    /// Rise time.
    pub rist: Array2<f32>,
    /// Rupture time.
    pub rupt: Array2<f32>,
}

/// The whole slip model.
#[derive(Clone, Debug)]
pub struct StochModel {
    pub segments: Vec<Segment>,
    /// Total subfault count across all segments.
    pub nstot: usize,
    /// Total fault area, km².
    pub farea_in: f32,
    /// Deepest hypocentre over the segments.
    pub zhyp_max: f32,
}

/// Read a `.stoch` file — `hb_high_ref.f:253-285`.
///
/// Produced from an SRF by `srf2stoch`. Format: segment count, then per segment
/// a two-line header followed by three `nw`-row blocks of `nx` values each
/// (slip, rise time, rupture time).
///
/// `pu` is the caller's degrees-to-radians factor; the Fortran uses its own
/// `3.1415926/180` from `:150`.
pub fn read_stoch(text: &str, pu: f32) -> Result<StochModel, DeckError> {
    let mut r = ListReader::new(text);
    let nevnt = r.i32()? as usize;

    let mut segments = Vec::with_capacity(nevnt);
    let mut nstot = 0usize;
    let mut farea_in = 0.0f32;
    let mut zhyp_max = 0.0f32;

    for _ in 0..nevnt {
        let v = r.read_values(6)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        let elonq = crate::deck::parse_f32(g(0))?;
        let elatq = crate::deck::parse_f32(g(1))?;
        let nx = crate::deck::parse_i32(g(2))? as usize;
        let nw = crate::deck::parse_i32(g(3))? as usize;
        let dx = crate::deck::parse_f32(g(4))?;
        let dw = crate::deck::parse_f32(g(5))?;

        let v = r.read_values(6)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        let strq = crate::deck::parse_f32(g(0))?;
        let dipq = crate::deck::parse_f32(g(1))?;
        let rakeq = crate::deck::parse_f32(g(2))?;
        let dtop = crate::deck::parse_f32(g(3))?;
        let shyp = crate::deck::parse_f32(g(4))?;
        let dhyp = crate::deck::parse_f32(g(5))?;

        // Accumulated in the Fortran's order: nx*nw + nstot, not nstot + nx*nw.
        nstot = nx * nw + nstot;
        farea_in = nx as f32 * dx * nw as f32 * dw + farea_in;
        let astop = 0.5 * nx as f32 * dx;

        // 2014-12-19: this was '*' and should have been '/'; fixed upstream.
        let zhyp = dtop + dhyp / (dipq * pu).sin();
        if zhyp > zhyp_max {
            zhyp_max = zhyp;
        }

        let mut sddp = Array2::<f32>::new(nx, nw);
        let mut rist = Array2::<f32>::new(nx, nw);
        let mut rupt = Array2::<f32>::new(nx, nw);
        for arr in [&mut sddp, &mut rist, &mut rupt] {
            for j in 1..=nw {
                // One record per down-dip row, nx values along strike.
                let row = r.read_values(nx)?;
                for i in 1..=nx {
                    arr[(i, j)] = crate::deck::parse_f32(row[i - 1].as_deref().unwrap_or(""))?;
                }
            }
        }

        segments.push(Segment {
            elonq, elatq, nx, nw, dx, dw,
            strq, dipq, rakeq, dtop, shyp, dhyp, astop,
            sddp, rist, rupt,
        });
    }

    Ok(StochModel { segments, nstot, farea_in, zhyp_max })
}

/// Read the 1-D velocity model into `/vmod_in/` — `hb_high_ref.f:322-349`.
///
/// Returns the layer count after Moho truncation. Layers at or below the first
/// one with `vsh >= vsmoho` are dropped, and the bottom layer is forced to zero
/// thickness so reflected rays are computed correctly (2016-08-03).
///
/// Note the mixed types on each record: `thic0`, `qp0` and `qs0` are `real*4`
/// while `vp0`, `vsh0` and `rho0` are `real*8` — see `state::VmodIn`.
///
/// If the *first* layer already exceeds `vsmoho` the Fortran reads `depth0(0)`,
/// one before the array start. Not reachable with the production `vsmoho` of
/// 999.9, and reproduced as a panic rather than a silent read.
pub fn read_velocity_model(
    text: &str,
    vmod_in: &mut VmodIn,
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
    for i in 1..=j0 {
        let v = r.read_values(6)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        vmod_in.thic0[i] = crate::deck::parse_f32(g(0))?;
        vmod_in.vp0[i] = crate::deck::parse_f64(g(1))?;
        vmod_in.vsh0[i] = crate::deck::parse_f64(g(2))?;
        vmod_in.rho0[i] = crate::deck::parse_f64(g(3))?;
        vmod_in.qp0[i] = crate::deck::parse_f32(g(4))?;
        vmod_in.qs0[i] = crate::deck::parse_f32(g(5))?;

        vmod_in.depth0[i] = vmod_in.thic0[i];
        if i > 1 {
            vmod_in.depth0[i] += vmod_in.depth0[i - 1];
        }

        if vmod_in.vsh0[i] >= vsmoho {
            jmoho = i;
            vmod_in.thic0[i] = 0.0;
            assert!(i > 1, "vsmoho reached in layer 1; the Fortran would read depth0(0)");
            vmod_in.depth0[i] = vmod_in.depth0[i - 1];
            break;
        }
    }

    j0 = jmoho;
    vmod_in.thic0[j0] = 0.0;
    Ok(j0)
}

/// Insert the thin "air" layer at the top of the model — `hb_high_ref.f:494-513`.
///
/// Needed to get the correct free-surface reflection coefficient for
/// surface-reflected rays. Returns the updated `(j0, nlskip)`.
///
/// This fires in production: the standard 34-layer model has
/// `depth0(1) = 0.05` and `vp0(1) = 1.8`, so `j0` becomes 35 and `nlskip` goes
/// from -99 to -98 (still negative, so `grandvel` stays dead).
///
/// Note the shift copies seven fields down but only **five** are overwritten at
/// index 1. `qp0(1)` and `qs0(1)` therefore keep the original first layer's Q
/// values rather than getting air-like ones. Faithful to the Fortran.
pub fn insert_air_layer(vmod_in: &mut VmodIn, j0: usize, nlskip: i32) -> (usize, i32) {
    if !(vmod_in.depth0[1] > 0.001 && vmod_in.vp0[1] > 0.01) {
        return (j0, nlskip);
    }
    let j0 = j0 + 1;
    let nlskip = nlskip + 1;

    for i in (2..=j0).rev() {
        vmod_in.depth0[i] = vmod_in.depth0[i - 1];
        vmod_in.thic0[i] = vmod_in.thic0[i - 1];
        vmod_in.vp0[i] = vmod_in.vp0[i - 1];
        vmod_in.vsh0[i] = vmod_in.vsh0[i - 1];
        vmod_in.rho0[i] = vmod_in.rho0[i - 1];
        vmod_in.qp0[i] = vmod_in.qp0[i - 1];
        vmod_in.qs0[i] = vmod_in.qs0[i - 1];
    }

    // depth0 and thic0 are real*4, so these literals are already f32.
    vmod_in.depth0[1] = 0.0001;
    vmod_in.thic0[1] = 0.0001;
    // vp0, vsh0 and rho0 are real*8, but the Fortran literals are UNSUFFIXED
    // and therefore only carry f32 precision -- PORTING_RULES.md §1b. Writing
    // 0.001f64 here gives 0.001 exactly; the Fortran stores
    // 0.0010000000474974513. Caught by the reader golden.
    vmod_in.vp0[1] = 0.001f32 as f64;
    vmod_in.vsh0[1] = 0.0005f32 as f64;
    vmod_in.rho0[1] = 0.001f32 as f64;
    // qp0(1) and qs0(1) are deliberately not set; see the note above.

    (j0, nlskip)
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
    fn reads_the_minimal_stoch_fixture() {
        let pu = 3.1415926f32 / 180.0;
        let m = read_stoch(MINI_STOCH, pu).unwrap();
        assert_eq!(m.segments.len(), 1);
        let s = &m.segments[0];
        assert_eq!((s.nx, s.nw), (2, 2));
        assert_eq!(s.elonq, -179.7826);
        assert_eq!(s.strq, 187.0);
        assert_eq!(m.nstot, 4);
        // Slip rows are down-dip, values along strike.
        assert_eq!(s.sddp[(1, 1)], 7.38758e0);
        assert_eq!(s.sddp[(2, 1)], 5.38111e0);
        assert_eq!(s.sddp[(1, 2)], 8.36237e0);
        assert_eq!(s.rist[(1, 1)], 1.32973e-1);
        assert_eq!(s.rupt[(2, 2)], 5.81083e-1);
        assert_eq!(s.astop, 0.5 * 2.0 * 1.64);
    }

    #[test]
    fn velocity_model_truncates_at_the_moho_and_zeroes_the_base() {
        let text = "3\n1.0 2.0 1.0 2.0 100 50\n2.0 4.0 2.5 2.5 200 100\n3.0 8.0 4.6 3.3 400 200\n";
        let mut v = VmodIn::new();
        // vsmoho below the third layer's 4.6 truncates there.
        let j0 = read_velocity_model(text, &mut v, 4.0).unwrap();
        assert_eq!(j0, 3);
        assert_eq!(v.thic0[3], 0.0, "the Moho layer is zeroed");
        assert_eq!(v.depth0[3], v.depth0[2]);
    }

    #[test]
    fn velocity_model_without_moho_still_zeroes_the_base() {
        let text = "2\n1.0 2.0 1.0 2.0 100 50\n2.0 4.0 2.5 2.5 200 100\n";
        let mut v = VmodIn::new();
        let j0 = read_velocity_model(text, &mut v, 999.9).unwrap();
        assert_eq!(j0, 2);
        assert_eq!(v.thic0[2], 0.0);
        assert_eq!(v.depth0[1], 1.0);
    }

    #[test]
    fn air_layer_is_inserted_for_a_realistic_model() {
        let text = "2\n0.05 1.8 0.5 1.81 116.0 58.0\n2.0 4.0 2.5 2.5 200 100\n";
        let mut v = VmodIn::new();
        let j0 = read_velocity_model(text, &mut v, 999.9).unwrap();
        let qp1_before = v.qp0[1];
        let (j0b, nlskip) = insert_air_layer(&mut v, j0, -99);
        assert_eq!(j0b, j0 + 1, "production models do get the air layer");
        assert_eq!(nlskip, -98, "still negative, so grandvel stays dead");
        assert_eq!(v.thic0[1], 0.0001);
        // Not 0.001f64: the Fortran literal is unsuffixed in a real*8 context,
        // so it carries only f32 precision. See PORTING_RULES.md §1b.
        assert_eq!(v.vp0[1], 0.001f32 as f64);
        assert_eq!(v.vsh0[1], 0.0005f32 as f64);
        assert_eq!(v.rho0[1], 0.001f32 as f64);
        assert_eq!(v.thic0[2], 0.05, "the original first layer shifted down");
        // The shift copies seven fields but only five are overwritten, so Q
        // stays put.
        assert_eq!(v.qp0[1], qp1_before, "qp0(1) is deliberately not air-like");
    }

    #[test]
    fn air_layer_is_skipped_when_the_model_starts_at_the_surface() {
        let text = "2\n0.0 1.8 0.5 1.81 116.0 58.0\n2.0 4.0 2.5 2.5 200 100\n";
        let mut v = VmodIn::new();
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
