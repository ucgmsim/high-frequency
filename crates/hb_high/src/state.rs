//! The common blocks, as explicit state.
//!
//! Field names are the canonical ones from `PORTING_RULES.md` §6, chosen once
//! rather than per-routine — the Fortran calls the same storage `dep`/`dpt`,
//! `th`/`thic`, `vs`/`vsh`/`s`, and `dn`/`rho`/`d`/`rh` in different routines.
//! Layouts are positionally identical across every declaration, so only the
//! naming needed resolving.

use crate::fort::Array1;

/// `params.h` / `params_no_window.h`.
///
/// `nq`, `np`, `nlaymax` and `lv` are identical in both headers. `mm` and `mmv`
/// are **not** — the main program under `VERSION1` gets `params_no_window.h`
/// (`262144`) while every subroutine unconditionally includes `params.h`
/// (`32769`/`180000`). That mismatch does not affect the live subprogram set,
/// which only uses the four constants below, but see `reference/PROVENANCE.md`.
pub mod params {
    /// Maximum layers in the velocity model.
    pub const NLAYMAX: usize = 500;
    /// Maximum subfaults along strike.
    pub const NQ: usize = 600;
    /// Maximum subfaults down dip.
    pub const NP: usize = 100;
    /// Maximum fault segments.
    pub const LV: usize = 1000;
    /// `mm` from `params_no_window.h` (main program under VERSION1).
    pub const MM: usize = 262144;
    /// `mmv` from `params_no_window.h`.
    pub const MMV: usize = 262144;
}

use params::NLAYMAX;

/// `common /vmod/` — the working velocity model, as perturbed and truncated.
///
/// Mixed precision within one block: the first five arrays are `real*8`, `qp`
/// and `qs` are `real*4`. Note that five of the fourteen declarations get their
/// `real*8`-ness solely from `implicit real*8 (a-h,o-z)`, so the types here are
/// not negotiable.
#[derive(Clone, Debug)]
pub struct Vmod {
    /// `dep` / `dpt` — cumulative depth to the base of each layer.
    pub depth: Array1<f64>,
    /// `th` — layer thickness.
    pub thic: Array1<f64>,
    /// P velocity. Named `c` in the dead `gencof`.
    pub vp: Array1<f64>,
    /// S velocity. Named `vs` or `s` elsewhere.
    pub vsh: Array1<f64>,
    /// Density. Named `dn`, `d` or `rh` elsewhere.
    pub rho: Array1<f64>,
    pub qp: Array1<f32>,
    pub qs: Array1<f32>,
}

impl Default for Vmod {
    fn default() -> Self {
        Self::new()
    }
}

impl Vmod {
    pub fn new() -> Self {
        Self {
            depth: Array1::new(NLAYMAX),
            thic: Array1::new(NLAYMAX),
            vp: Array1::new(NLAYMAX),
            vsh: Array1::new(NLAYMAX),
            rho: Array1::new(NLAYMAX),
            qp: Array1::new(NLAYMAX),
            qs: Array1::new(NLAYMAX),
        }
    }
}

/// `common /vmod_in/` — the unperturbed model as read from file.
///
/// `depth0`, `thic0` and `qp0` are `real*4` here while the corresponding
/// `/vmod/` fields are `real*8`. That asymmetry is not a mistake in the port:
/// those three are undeclared in *both* scopes that declare the block, so they
/// fall to implicit `real*4`. Adding `implicit none` to either Fortran scope
/// would shift the whole block. See `PORTING_RULES.md` §2.
#[derive(Clone, Debug)]
pub struct VmodIn {
    pub depth0: Array1<f32>,
    pub thic0: Array1<f32>,
    pub vp0: Array1<f64>,
    pub vsh0: Array1<f64>,
    pub rho0: Array1<f64>,
    pub qp0: Array1<f32>,
    pub qs0: Array1<f32>,
    /// `grand` / `gr` — RNG scratch shared with `grandvel` (dead in production).
    pub grand: Array1<f32>,
}

impl Default for VmodIn {
    fn default() -> Self {
        Self::new()
    }
}

impl VmodIn {
    pub fn new() -> Self {
        Self {
            depth0: Array1::new(NLAYMAX),
            thic0: Array1::new(NLAYMAX),
            vp0: Array1::new(NLAYMAX),
            vsh0: Array1::new(NLAYMAX),
            rho0: Array1::new(NLAYMAX),
            qp0: Array1::new(NLAYMAX),
            qs0: Array1::new(NLAYMAX),
            grand: Array1::new(3000),
        }
    }
}

/// `common /rays/` — the ray segment description. Written only by `gf_amp_tt`.
///
/// The Fortran declares `nh(1,nlaymax)` and `nm(1,nlaymax)` with a degenerate
/// leading dimension, and every routine hardwires the ray index to 1. The
/// leading dimension is dropped here; routines still take an `ir` argument so
/// call sites match the Fortran, and assert it is 1.
#[derive(Clone, Debug)]
pub struct Rays {
    /// Layer index of each ray segment.
    pub nh: Array1<i32>,
    /// Wave mode of each segment: 3 = SV, 4 = SH, 5 = P.
    pub nm: Array1<i32>,
    /// Ray degeneracy; negative means the ray is upgoing.
    pub ndeg: Array1<i32>,
    /// Number of segments in the ray.
    pub nd: Array1<i32>,
}

impl Default for Rays {
    fn default() -> Self {
        Self::new()
    }
}

impl Rays {
    pub fn new() -> Self {
        Self {
            nh: Array1::new(NLAYMAX),
            nm: Array1::new(NLAYMAX),
            ndeg: Array1::new(NLAYMAX),
            nd: Array1::new(NLAYMAX),
        }
    }
}

/// `common /travel/` — per-layer path multipliers. Written only by `trav`.
///
/// `alp` and `als` are `real*4` by *explicit* declaration inside routines that
/// are otherwise `implicit real*8`; that explicit declaration is load-bearing.
///
/// The third slot is written by `trav` as `ndeep` but read as `nd` by `cagcon`
/// and `dtdp` and as `ndp` by `pnot` — and it is **not** the same quantity as
/// `Rays::nd`, which those same routines also declare. See `PORTING_RULES.md`
/// §6.
#[derive(Clone, Debug)]
pub struct Travel {
    /// P path multiplier per layer.
    pub alp: Array1<f32>,
    /// S path multiplier per layer.
    pub als: Array1<f32>,
    /// Deepest layer the ray penetrates. Read as `nd`/`ndp` by consumers.
    pub ndeep: i32,
    /// Written by `trav`; read by nothing.
    pub nup: i32,
}

impl Default for Travel {
    fn default() -> Self {
        Self::new()
    }
}

impl Travel {
    pub fn new() -> Self {
        Self { alp: Array1::new(NLAYMAX), als: Array1::new(NLAYMAX), ndeep: 0, nup: 0 }
    }
}

/// `common /coff/` — interface interaction types. Written only by `trav`.
///
/// Do not conflate with the dead `gencof`'s dummy argument, also named `it`.
#[derive(Clone, Debug)]
pub struct Coff {
    /// 0 = transmission, 1 = reflection, 2 = direct ray.
    pub it: Array1<i32>,
    /// Segment direction: +1 up, -1 down.
    pub nup1: Array1<i32>,
}

impl Default for Coff {
    fn default() -> Self {
        Self::new()
    }
}

impl Coff {
    pub fn new() -> Self {
        Self { it: Array1::new(NLAYMAX), nup1: Array1::new(NLAYMAX) }
    }
}

/// The full ray-tracing state, threaded through the `gf_amp_tt` cluster.
///
/// The Fortran passes none of this in arguments — `gf_amp_tt` writes `/rays/`,
/// `trav` writes `/travel/` and `/coff/`, and `pnot`/`ttime`/`geom_terms` read
/// them back. Making the dataflow explicit is the point of this struct.
///
/// `/rmode/love` is included for fidelity: `trav` writes it, but its only
/// readers (`refft`, `tranm`) are unreachable, so nothing live consumes it.
#[derive(Clone, Debug, Default)]
pub struct RayState {
    pub rays: Rays,
    pub travel: Travel,
    pub coff: Coff,
    /// `/rmode/love` — 1 for P-SV, 2 for SH. Written by `trav`, never read.
    pub love: i32,
}
