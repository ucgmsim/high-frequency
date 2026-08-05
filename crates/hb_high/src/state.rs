//! The common blocks, as explicit state.
//!
//! Field names are the canonical ones from `PORTING_RULES.md` §6, chosen once
//! rather than per-routine — the Fortran calls the same storage `dep`/`dpt`,
//! `th`/`thickness_km`, `vs`/`vsh_km_s`/`s`, and `dn`/`density_g_cm3`/`d`/`rh` in different routines.
//! Layouts are positionally identical across every declaration, so only the
//! naming needed resolving.

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
///
/// **Layers are indexed 0-based**, `0..layer_count` — §2.3. The Fortran numbers them from
/// 1 and so did this port until then. The allocation stays at `NLAYMAX` rather than
/// shrinking to `layer_count`, and that is load-bearing rather than lazy: two lookups read
/// one element PAST the model when a source is below every layer, which the Fortran did
/// too, and the surrounding code depends on getting the zero there rather than a panic.
/// See `PORTING_RULES.md` §7.
/// One layer of the working velocity model.
///
/// The mixed precision is not negotiable and not tidyable: the first five are `real*8`
/// and the two `attenuation` fields `real*4` in the Fortran, five of the fourteen
/// declarations getting their `real*8`-ness solely from `implicit real*8 (a-h,o-z)`.
/// Widening `attenuation_s` to `f64` would change `geometric_spreading`'s deliberately
/// single-precision accumulation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Layer {
    /// `dep` / `dpt` — cumulative depth to the base of this layer.
    pub depth_km: f64,
    /// `th` — layer thickness.
    pub thickness_km: f64,
    /// P velocity. Named `c` in the dead `gencof`.
    pub vp_km_s: f64,
    /// S velocity. Named `vs` or `s` elsewhere.
    pub vsh_km_s: f64,
    /// Density. Named `dn`, `d` or `rh` elsewhere.
    pub density_g_cm3: f64,
    pub attenuation_p: f32,
    pub attenuation_s: f32,
}

#[derive(Clone, Debug)]
pub struct VelocityModel {
    /// Indexed through [`Index`], so a caller writes `vmod[k].vsh_km_s`.
    layers: Vec<Layer>,
}

impl Default for VelocityModel {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityModel {
    pub fn new() -> Self {
        Self { layers: vec![Layer::default(); NLAYMAX] }
    }

    /// The layers as a slice, for the reductions that want a range rather than one index.
    ///
    /// Deliberately not `Deref<Target = [Layer]>`: that would also expose `len()`, which
    /// is `NLAYMAX` and not the layer count. Every caller here already carries the real
    /// count, and confusing the two is exactly the `j0` hazard `PORTING_RULES.md` §7
    /// describes.
    #[inline]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
}

impl std::ops::Index<usize> for VelocityModel {
    type Output = Layer;
    #[inline]
    fn index(&self, layer: usize) -> &Layer {
        &self.layers[layer]
    }
}

impl std::ops::IndexMut<usize> for VelocityModel {
    #[inline]
    fn index_mut(&mut self, layer: usize) -> &mut Layer {
        &mut self.layers[layer]
    }
}

/// `common /vmod_in/` — the unperturbed model as read from file.
///
/// `depth_km`, `thickness_km` and `attenuation_p` are `real*4` here while the corresponding
/// `/vmod/` fields are `real*8`. That asymmetry is not a mistake in the port:
/// those three are undeclared in *both* scopes that declare the block, so they
/// fall to implicit `real*4`. Adding `implicit none` to either Fortran scope
/// would shift the whole block. See `PORTING_RULES.md` §2.
/// One layer as read from file.
///
/// **`depth_km` and `thickness_km` are `f32` here and `f64` in [`Layer`].** That is not
/// an inconsistency to tidy: they are undeclared in *both* Fortran scopes that declare
/// this block, so they fall to implicit `real*4`, while the corresponding `/vmod/`
/// fields are `real*8`. Adding `implicit none` to either scope would shift the whole
/// block. See `PORTING_RULES.md` §2.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InputLayer {
    pub depth_km: f32,
    pub thickness_km: f32,
    pub vp_km_s: f64,
    pub vsh_km_s: f64,
    pub density_g_cm3: f64,
    pub attenuation_p: f32,
    pub attenuation_s: f32,
}

#[derive(Clone, Debug)]
pub struct VelocityModelInput {
    layers: Vec<InputLayer>,
    // `grand`/`gr` -- 3000 floats of RNG scratch for `grandvel` -- lived here until
    // §2.8. `grandvel` is dead under the production deck (`nl_skip < 0`) and is not
    // ported, so nothing ever read the field, but `simulate` deep-cloned it once per
    // call to carry it.
}

impl Default for VelocityModelInput {
    fn default() -> Self {
        Self::new()
    }
}

impl VelocityModelInput {
    pub fn new() -> Self {
        Self { layers: vec![InputLayer::default(); NLAYMAX] }
    }
}

impl From<InputLayer> for Layer {
    /// The unperturbed path: `/vmod_in/` to `/vmod/` verbatim, widening the two `real*4`
    /// fields the working model holds in double.
    ///
    /// The Fortran spells this out field by field inside the station loop. It is a
    /// conversion, and now says so.
    fn from(l: InputLayer) -> Self {
        Self {
            depth_km: l.depth_km as f64,
            thickness_km: l.thickness_km as f64,
            vp_km_s: l.vp_km_s,
            vsh_km_s: l.vsh_km_s,
            density_g_cm3: l.density_g_cm3,
            attenuation_p: l.attenuation_p,
            attenuation_s: l.attenuation_s,
        }
    }
}

impl std::ops::Index<usize> for VelocityModelInput {
    type Output = InputLayer;
    #[inline]
    fn index(&self, layer: usize) -> &InputLayer {
        &self.layers[layer]
    }
}

impl std::ops::IndexMut<usize> for VelocityModelInput {
    #[inline]
    fn index_mut(&mut self, layer: usize) -> &mut InputLayer {
        &mut self.layers[layer]
    }
}


// ---------------------------------------------------------------------------
// The three-valued and two-valued quantities the Fortran spells as integers
// ---------------------------------------------------------------------------

/// Wave mode of a ray segment — the Fortran's `nm`.
///
/// Never added, subtracted, ordered or used as a magnitude; only ever compared against
/// the literals 3, 4 and 5 at six sites. The `md` in every golden driver takes exactly
/// these three values, so the enum is total over the test corpus as well as production
/// (which is hardwired to SH).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveMode {
    /// `3` — vertically polarised shear.
    Sv,
    /// `4` — horizontally polarised shear. What production runs.
    Sh,
    /// `5` — compressional.
    P,
}

impl WaveMode {
    /// True for `Sv` and `Sh` — the test the Fortran writes out as
    /// `nm == 3 .or. nm == 4` in four separate places.
    #[inline]
    pub fn is_shear(self) -> bool {
        matches!(self, Self::Sv | Self::Sh)
    }

    /// Decode a golden fixture's stored mode.
    pub fn from_fortran(v: i32) -> Self {
        match v {
            3 => Self::Sv,
            4 => Self::Sh,
            5 => Self::P,
            _ => panic!("wave mode {v} is not one of 3 (SV), 4 (SH) or 5 (P)"),
        }
    }

    /// Re-encode for comparison against a golden fixture.
    pub fn as_fortran(self) -> i32 {
        match self {
            Self::Sv => 3,
            Self::Sh => 4,
            Self::P => 5,
        }
    }
}

impl Default for WaveMode {
    /// Arbitrary, and unobservable. `Rays::nm` is allocated at `NLAYMAX` but every read
    /// is inside `0..nd`, which `green_function` always writes in full, so no consumer
    /// can reach an unwritten slot. `Sh` is chosen because it is what production writes.
    fn default() -> Self {
        Self::Sh
    }
}

/// What happens at the interface below a ray segment — the Fortran's `it`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interaction {
    /// `0` — the ray passes into the next layer. No second velocity clamp in
    /// `travel_time`.
    #[default]
    Transmission,
    /// `1` — the ray reflects, so `travel_time` must also clamp against the layer on the
    /// far side of the interface.
    Reflection,
    /// `2` — a single-segment direct ray. Written by `build_ray_path`, read by nothing
    /// live; the tier-1 golden compares it.
    Direct,
}

impl Interaction {

    pub fn as_fortran(self) -> i32 {
        match self {
            Self::Transmission => 0,
            Self::Reflection => 1,
            Self::Direct => 2,
        }
    }
}

/// Direction a ray segment travels — the Fortran's `nup` and `nup1`, `+1` and `-1`.
///
/// Nothing about these is numeric. The Fortran derives the first from
/// `(-1)**nl` — integer exponentiation used to extract a parity bit — and negates it to
/// mean "flip", not "negate a quantity".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    /// `+1`.
    #[default]
    Up,
    /// `-1`.
    Down,
}

impl Direction {
    /// The Fortran's `(-1)**nl`: an even count gives `+1` (up), an odd count `-1`.
    #[inline]
    pub fn from_parity(crossings: i32) -> Self {
        if crossings % 2 == 0 { Self::Up } else { Self::Down }
    }

    #[inline]
    pub fn flipped(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }

    /// The layer on the other side of the interface below a segment in `layer`.
    ///
    /// A layer STEP, so it is the same +-1 in either index base. Upgoing from layer 0
    /// would underflow, as the Fortran read `vs(0)` there; unreachable, and loud if it
    /// ever is not.
    #[inline]
    pub fn step_from(self, layer: usize) -> usize {
        match self {
            Self::Up => layer - 1,
            Self::Down => layer + 1,
        }
    }


    pub fn as_fortran(self) -> i32 {
        match self {
            Self::Up => 1,
            Self::Down => -1,
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
    /// Wave mode of each segment. 0-based by segment.
    pub nm: Vec<WaveMode>,
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
            nm: vec![WaveMode::default(); NLAYMAX],
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
    /// P path multiplier per layer. 0-based by layer.
    pub alp: Vec<f32>,
    /// S path multiplier per layer. 0-based by layer.
    pub als: Vec<f32>,
    /// Deepest layer the ray penetrates, as a **0-based layer index**. Read as `nd`/`ndp`
    /// by consumers, which iterate `0..=ndeep`.
    pub ndeep: i32,
    /// Direction the ray leaves the source. Written by `build_ray_path`; read by nothing
    /// live, but the tier-1 golden compares it.
    pub nup: Direction,
}

impl Default for Travel {
    fn default() -> Self {
        Self::new()
    }
}

impl Travel {
    pub fn new() -> Self {
        Self {
            alp: vec![0.0; NLAYMAX],
            als: vec![0.0; NLAYMAX],
            ndeep: 0,
            nup: Direction::default(),
        }
    }
}

/// `common /coff/` — interface interaction types. Written only by `build_ray_path`.
///
/// Do not conflate with the dead `gencof`'s dummy argument, also named `it`.
#[derive(Clone, Debug)]
pub struct Coefficients {
    /// What happens at the interface below each segment. 0-based by segment.
    pub it: Vec<Interaction>,
    /// Direction of each segment. 0-based by segment.
    pub nup1: Vec<Direction>,
}

impl Default for Coefficients {
    fn default() -> Self {
        Self::new()
    }
}

impl Coefficients {
    pub fn new() -> Self {
        Self {
            it: vec![Interaction::default(); NLAYMAX],
            nup1: vec![Direction::default(); NLAYMAX],
        }
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
