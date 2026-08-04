//! Intensity measures, for scientific equivalence testing of `hb_high` output.
//!
//! # Why this exists
//!
//! The port is bit-identical to a pinned Fortran oracle, which is the right gate
//! while transliterating but the wrong one for refactoring and optimisation. Once
//! the last bits are allowed to move, the question becomes whether the *science*
//! is unchanged — and that is answered in intensity measures, not bytes.
//!
//! # Units
//!
//! `hb_high` writes acceleration in **cm/s²**, so every function here takes
//! cm/s² and the constants are in that system (`G_CM = 981.0`). Passing g or
//! m/s² silently rescales Arias intensity and CAV.
//!
//! Computation is in `f64` throughout even though the input is `f32`. These are
//! measurement instruments, not part of the port: there is no reason to
//! reproduce single-precision accumulation here, and doing so would add noise to
//! the thing doing the measuring.
//!
//! # Provenance
//!
//! The pSA solver follows the formulation in `ucgmsim/IM_calculation`'s Rust
//! branch: Newmark-β with the integration constant chosen from a stability
//! criterion, solving implicitly for **displacement** and differentiating,
//! rather than solving for acceleration and integrating twice. The latter
//! amplifies differencing noise in the input, which matters for stochastic
//! high-frequency records. Reimplemented here against plain slices to avoid a
//! cross-repository dependency, and per instruction it is taken as correct
//! rather than separately validated.

pub mod arias;
pub mod cav;
pub mod duration;
pub mod fas;
pub mod peak;
pub mod psa;

/// Gravitational acceleration in cm/s² (Gal), matching `hb_high`'s output units.
pub const G_CM: f64 = 981.0;

/// Standard 5% damping.
pub const DAMPING: f64 = 0.05;

/// pSA periods, seconds. Spans 0.01–5 s as agreed.
///
/// The low end is where the stochastic high-frequency method carries the most
/// weight; the high end is included for completeness but is where the hybrid
/// hands over to the low-frequency model, so ratios there are the least
/// meaningful. Reported per period so that can be seen rather than averaged away.
pub const PERIODS: &[f64] = &[
    0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
];

/// Component order in `hb_high`'s output stream: 090, 000, ver.
pub const COMPONENTS: [&str; 3] = ["090", "000", "ver"];

/// Every scalar IM computed for one component of one record.
#[derive(Clone, Debug, PartialEq)]
pub struct Measures {
    /// Peak ground acceleration, cm/s².
    pub pga: f64,
    /// Peak ground velocity, cm/s.
    pub pgv: f64,
    /// Arias intensity, cm/s.
    pub ai: f64,
    /// Cumulative absolute velocity, cm/s.
    pub cav: f64,
    /// Significant duration 5–75%, s.
    pub ds575: f64,
    /// Significant duration 5–95%, s.
    pub ds595: f64,
    /// pSA at [`PERIODS`], cm/s².
    pub psa: Vec<f64>,
}

impl Measures {
    /// Compute every scalar IM for one component.
    pub fn compute(acc_cm_s2: &[f32], dt: f64) -> Self {
        let a: Vec<f64> = acc_cm_s2.iter().map(|&x| x as f64).collect();
        let (ds575, ds595) = duration::significant_durations(&a, dt);
        Self {
            pga: peak::pga(&a),
            pgv: peak::pgv(&a, dt),
            ai: arias::arias_intensity(&a, dt),
            cav: cav::cav(&a, dt),
            ds575,
            ds595,
            psa: psa::psa_set(&a, dt, PERIODS, DAMPING),
        }
    }

    /// Flat (name, value) pairs, for CSV emission and endpoint iteration.
    pub fn endpoints(&self) -> Vec<(String, f64)> {
        let mut v = vec![
            ("PGA".to_string(), self.pga),
            ("PGV".to_string(), self.pgv),
            ("AI".to_string(), self.ai),
            ("CAV".to_string(), self.cav),
            ("Ds575".to_string(), self.ds575),
            ("Ds595".to_string(), self.ds595),
        ];
        for (t, p) in PERIODS.iter().zip(&self.psa) {
            v.push((format!("pSA({t})"), *p));
        }
        v
    }
}

/// Split `hb_high`'s raw output into three component channels.
///
/// The stream is `ndata * 3` `f32` with the component index fastest, i.e.
/// `090,000,ver` per time sample — see `PORTING_RULES.md` §9.
pub fn split_components(raw: &[f32]) -> [Vec<f32>; 3] {
    assert_eq!(
        raw.len() % 3,
        0,
        "output length {} is not a multiple of 3",
        raw.len()
    );
    let n = raw.len() / 3;
    let mut out = [Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n)];
    for s in raw.chunks_exact(3) {
        out[0].push(s[0]);
        out[1].push(s[1]);
        out[2].push(s[2]);
    }
    out
}

/// Read a raw little-endian `f32` output file.
pub fn read_output(path: &std::path::Path) -> std::io::Result<Vec<f32>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("{}: {} bytes is not a whole number of f32", path.display(), bytes.len()),
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}
