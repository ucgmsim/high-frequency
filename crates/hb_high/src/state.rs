//! The common blocks, as explicit state.
//!
//! Field names are the canonical ones from `PORTING_RULES.md` §6, chosen once
//! rather than per-routine — the Fortran calls the same storage `dep`/`dpt`,
//! `th`/`thickness_km`, `vs`/`vsh_km_s`/`s`, and `dn`/`density_g_cm3`/`d`/`rh` in different routines.
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
    /// `mmv` from `params_no_window.h`.
    ///
    /// The last compile-time ceiling in the port, and it is not a ceiling on capability:
    /// it is the number of normal deviates `simulate` draws per station, which is part of
    /// the RNG stream and therefore load-bearing rather than a size. See §2.6b.
    ///
    /// `NQ` (600 subfaults along strike), `NP` (100 down dip), `LV` (1000 segments) and
    /// `MM` (262144 samples) were deleted in §2.3/§2.6b: every buffer they sized is now
    /// sized from the deck, so a longer record no longer needs a recompile.
    pub const MMV: usize = 262144;
}

use params::NLAYMAX;

/// `common /vmod/` — the working velocity model, as perturbed and truncated.
///
/// Mixed precision within one block: the first five arrays are `real*8`, `attenuation_p`
/// and `attenuation_s` are `real*4`. Note that five of the fourteen declarations get their
/// `real*8`-ness solely from `implicit real*8 (a-h,o-z)`, so the types here are
/// not negotiable.
#[derive(Clone, Debug)]
pub struct VelocityModel {
    /// `dep` / `dpt` — cumulative depth to the base of each layer.
    pub depth_km: Array1<f64>,
    /// `th` — layer thickness.
    pub thickness_km: Array1<f64>,
    /// P velocity. Named `c` in the dead `gencof`.
    pub vp_km_s: Array1<f64>,
    /// S velocity. Named `vs` or `s` elsewhere.
    pub vsh_km_s: Array1<f64>,
    /// Density. Named `dn`, `d` or `rh` elsewhere.
    pub density_g_cm3: Array1<f64>,
    pub attenuation_p: Array1<f32>,
    pub attenuation_s: Array1<f32>,
}

impl Default for VelocityModel {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityModel {
    pub fn new() -> Self {
        Self {
            depth_km: Array1::new(NLAYMAX),
            thickness_km: Array1::new(NLAYMAX),
            vp_km_s: Array1::new(NLAYMAX),
            vsh_km_s: Array1::new(NLAYMAX),
            density_g_cm3: Array1::new(NLAYMAX),
            attenuation_p: Array1::new(NLAYMAX),
            attenuation_s: Array1::new(NLAYMAX),
        }
    }
}

/// `common /vmod_in/` — the unperturbed model as read from file.
///
/// `depth_km`, `thickness_km` and `attenuation_p` are `real*4` here while the corresponding
/// `/vmod/` fields are `real*8`. That asymmetry is not a mistake in the port:
/// those three are undeclared in *both* scopes that declare the block, so they
/// fall to implicit `real*4`. Adding `implicit none` to either Fortran scope
/// would shift the whole block. See `PORTING_RULES.md` §2.
#[derive(Clone, Debug)]
pub struct VelocityModelInput {
    pub depth_km: Array1<f32>,
    pub thickness_km: Array1<f32>,
    pub vp_km_s: Array1<f64>,
    pub vsh_km_s: Array1<f64>,
    pub density_g_cm3: Array1<f64>,
    pub attenuation_p: Array1<f32>,
    pub attenuation_s: Array1<f32>,
    /// `grand` / `gr` — RNG scratch shared with `grandvel` (dead in production).
    pub grand: Array1<f32>,
}

impl Default for VelocityModelInput {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityModelInput {
    pub fn new() -> Self {
        Self {
            depth_km: Array1::new(NLAYMAX),
            thickness_km: Array1::new(NLAYMAX),
            vp_km_s: Array1::new(NLAYMAX),
            vsh_km_s: Array1::new(NLAYMAX),
            density_g_cm3: Array1::new(NLAYMAX),
            attenuation_p: Array1::new(NLAYMAX),
            attenuation_s: Array1::new(NLAYMAX),
            grand: Array1::new(3000),
        }
    }
}

/// `common /rays/` — the ray segment description. Written only by `green_function`.
///
/// The Fortran declares `nh(1,nlaymax)` and `nm(1,nlaymax)` with a degenerate
/// leading dimension, and every routine hardwires the ray index to 1. The
/// leading dimension is dropped here; routines still take an `ir` argument so
/// call sites match the Fortran, and assert it is 1.
#[derive(Clone, Debug)]
pub struct Rays {
    /// Layer index of each ray segment. **Indexed 0-based by segment**, `0..nd`.
    pub nh: Vec<i32>,
    /// Wave mode of each segment: 3 = SV, 4 = SH, 5 = P. 0-based by segment.
    pub nm: Vec<i32>,
    /// Ray degeneracy; negative means the ray is upgoing.
    ///
    /// A scalar, not an array. The Fortran declares `ndeg(1)` and `nd(1)` -- indexed by
    /// the same degenerate ray dimension the struct header describes -- and every routine
    /// hardwires that index to 1, so an array here only invited the reader to wonder what
    /// the other elements meant.
    pub ndeg: i32,
    /// Number of segments in the ray. A scalar, for the same reason as `ndeg`.
    pub nd: i32,
}

impl Default for Rays {
    fn default() -> Self {
        Self::new()
    }
}

impl Rays {
    pub fn new() -> Self {
        Self {
            nh: vec![0; NLAYMAX],
            nm: vec![0; NLAYMAX],
            ndeg: 0,
            nd: 0,
        }
    }
}

/// `common /travel/` — per-layer path multipliers. Written only by `build_ray_path`.
///
/// `alp` and `als` are `real*4` by *explicit* declaration inside routines that
/// are otherwise `implicit real*8`; that explicit declaration is load-bearing.
///
/// The third slot is written by `build_ray_path` as `ndeep` but read as `nd` by `cagniard_time`
/// and `cagniard_time_derivative` and as `ndp` by `stationary_ray_parameter` — and it is **not** the same quantity as
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
    /// Written by `build_ray_path`; read by nothing.
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

/// `common /coff/` — interface interaction types. Written only by `build_ray_path`.
///
/// Do not conflate with the dead `gencof`'s dummy argument, also named `it`.
#[derive(Clone, Debug)]
pub struct Coefficients {
    /// 0 = transmission, 1 = reflection, 2 = direct ray. 0-based by segment.
    pub it: Vec<i32>,
    /// Segment direction: +1 up, -1 down. 0-based by segment.
    pub nup1: Vec<i32>,
}

impl Default for Coefficients {
    fn default() -> Self {
        Self::new()
    }
}

impl Coefficients {
    pub fn new() -> Self {
        Self { it: vec![0; NLAYMAX], nup1: vec![0; NLAYMAX] }
    }
}

/// The full ray-tracing state, threaded through the `green_function` cluster.
///
/// The Fortran passes none of this in arguments — `green_function` writes `/rays/`,
/// `build_ray_path` writes `/travel/` and `/coff/`, and `stationary_ray_parameter`/`travel_time`/`geometric_spreading` read
/// them back. Making the dataflow explicit is the point of this struct.
///
/// `/rmode/love` is included for fidelity: `build_ray_path` writes it, but its only
/// readers (`refft`, `tranm`) are unreachable, so nothing live consumes it.
#[derive(Clone, Debug, Default)]
pub struct RayState {
    pub rays: Rays,
    pub travel: Travel,
    pub coff: Coefficients,
    /// `/rmode/love` — 1 for P-SV, 2 for SH. Written by `build_ray_path`, never read.
    pub love: i32,
}
