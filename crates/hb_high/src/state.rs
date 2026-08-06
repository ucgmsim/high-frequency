//! The common blocks, as explicit state.
//!
//! Field names are the canonical ones from `PORTING_RULES.md` §6, chosen once
//! rather than per-routine — the Fortran calls the same storage `dep`/`dpt`,
//! `th`/`thickness_km`, `vs`/`vsh_km_s`/`s`, and `dn`/`density_g_cm3`/`d`/`rh` in different routines.
//! Layouts are positionally identical across every declaration, so only the
//! naming needed resolving.


/// One layer of the working velocity model, `common /vmod/`.
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

/// The working velocity model: layers from the surface down, **0-based**.
///
/// A plain `Vec`, so `len()` is the layer count and means it. This was a newtype wrapping a
/// fixed 500-element buffer, with `len()` deliberately hidden because it reported the
/// ceiling rather than the model — every caller had to carry the real count alongside. The
/// ceiling existed to absorb one out-of-range read, which `ray::source_layer` now names
/// instead.
pub type VelocityModel = Vec<Layer>;

/// One layer as read from file, `common /vmod_in/`.
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

/// The velocity model as read, before the air layer and the working-model widening.
pub type VelocityModelInput = Vec<InputLayer>;

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

// ---------------------------------------------------------------------------
// The three-valued and two-valued quantities the Fortran spells as integers
// ---------------------------------------------------------------------------

/// Wave mode of a ray segment — the Fortran's `nm`.
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
    /// Arbitrary, and unobservable: [`Rays::wave_modes`] is grown one entry per segment as
    /// the path is built, so there is no unwritten slot for a consumer to reach. `Sh` is
    /// chosen because it is what production writes.
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
        if crossings % 2 == 0 {
            Self::Up
        } else {
            Self::Down
        }
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
#[derive(Clone, Debug, Default)]
pub struct Rays {
    /// Layer index of each ray segment. **Indexed 0-based by segment.**
    ///
    /// Grown as the path is built, so its length IS the segment count -- the Fortran's
    /// separate `nd` counter alongside a fixed-size buffer is gone.
    pub layer_indices: Vec<usize>,
    /// Wave mode of each segment, parallel to [`Rays::layer_indices`].
    pub wave_modes: Vec<WaveMode>,
    /// Ray degeneracy; negative means the ray is upgoing.
    ///
    /// A scalar, not an array. The Fortran declares `ndeg(1)` indexed by the same
    /// degenerate ray dimension the struct header describes, and every routine hardwires
    /// that index to 1, so an array here only invited the reader to wonder what the other
    /// elements meant.
    pub degeneracy: i32,
}

impl Rays {
    /// Number of segments in the ray — the Fortran's `nd`.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.layer_indices.len()
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
#[derive(Clone, Debug, Default)]
pub struct Travel {
    /// P path multiplier per layer. 0-based by layer, one entry per layer in the model.
    pub p_traversals: Vec<f32>,
    /// S path multiplier per layer, parallel to [`Travel::p_traversals`].
    pub s_traversals: Vec<f32>,
    /// Deepest layer the ray penetrates, as a **0-based layer index**. Consumers iterate
    /// `0..=deepest_layer`.
    pub deepest_layer: usize,
    /// Direction the ray leaves the source. Written by `build_ray_path`; read by nothing
    /// live, but the tier-1 golden compares it.
    pub takeoff: Direction,
}

impl Travel {
    /// Resize both multiplier tables to one entry per layer and zero them.
    ///
    /// The Fortran zeroed `alp(1:100)` while the arrays were dimensioned 500, so a model
    /// deeper than 100 layers inherited multipliers from the PREVIOUS ray and silently
    /// produced wrong travel times. Sizing to the model makes that unrepresentable.
    pub fn reset_for(&mut self, layer_count: usize) {
        self.p_traversals.clear();
        self.p_traversals.resize(layer_count, 0.0);
        self.s_traversals.clear();
        self.s_traversals.resize(layer_count, 0.0);
    }
}

/// `common /coff/` — interface interaction types. Written only by `build_ray_path`.
///
/// Do not conflate with the dead `gencof`'s dummy argument, also named `it`.
#[derive(Clone, Debug, Default)]
pub struct Coefficients {
    /// What happens at the interface below each segment. 0-based by segment.
    pub interactions: Vec<Interaction>,
    /// Direction of each segment. 0-based by segment.
    pub directions: Vec<Direction>,
}

impl Coefficients {
    /// Resize both tables to one entry per ray segment and reset them.
    pub fn reset_for(&mut self, segment_count: usize) {
        self.interactions.clear();
        self.interactions.resize(segment_count, Interaction::default());
        self.directions.clear();
        self.directions.resize(segment_count, Direction::default());
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
    pub coefficients: Coefficients,
    /// `/rmode/love` — 1 for P-SV, 2 for SH. Written by `build_ray_path`, never read.
    pub love: i32,
}
