//! The velocity model and the ray-tracing state, as explicit structs.

/// One layer of the working velocity model.
///
/// The mixed precision is deliberate: widening `attenuation_s` to `f64` would change
/// `geometric_spreading`'s single-precision accumulation, which matches the original code.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Layer {
    /// Cumulative depth to the base of this layer.
    pub depth_km: f64,
    /// Layer thickness.
    pub thickness_km: f64,
    /// P velocity.
    pub vp_km_s: f64,
    /// S velocity.
    pub vsh_km_s: f64,
    /// Density.
    pub density_g_cm3: f64,
    pub attenuation_p: f32,
    pub attenuation_s: f32,
}

/// The working velocity model: layers from the surface down, 0-based. `len()` is the layer
/// count.
pub type VelocityModel = Vec<Layer>;

/// One layer as read from file.
///
/// `depth_km` and `thickness_km` are `f32` here and `f64` in [`Layer`] deliberately: the
/// original code held them in single precision as read, and rounding through `f32` preserves
/// its output.
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
    /// The unperturbed path: the input layer verbatim, widening the two `f32` fields the
    /// working model holds in `f64`.
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
// Enumerated ray quantities, with their integer codes in the golden fixtures
// ---------------------------------------------------------------------------

/// Wave mode of a ray segment.
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
    /// True for `Sv` and `Sh`.
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

/// What happens at the interface below a ray segment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Interaction {
    /// `0` — the ray passes into the next layer.
    #[default]
    Transmission,
    /// `1` — the ray reflects, reversing the direction of the next segment.
    Reflection,
    /// `2` — a single-segment direct ray. Written by `build_ray_path`; nothing reads it.
    Direct,
}

/// Direction a ray segment travels, encoded `+1` (up) and `-1` (down).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Direction {
    /// `+1`.
    #[default]
    Up,
    /// `-1`.
    Down,
}

impl Direction {
    /// `(-1)^crossings`: an even count gives `+1` (up), an odd count `-1`.
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
}

/// The ray segment description. Written only by `green_function`.
#[derive(Clone, Debug, Default)]
pub struct Rays {
    /// Layer index of each ray segment, 0-based by segment.
    ///
    /// Grown as the path is built, so its length is the segment count.
    pub layer_indices: Vec<usize>,
    /// Wave mode of each segment, parallel to [`Rays::layer_indices`].
    pub wave_modes: Vec<WaveMode>,
    /// Ray degeneracy; negative means the ray is upgoing.
    pub degeneracy: i32,
}

impl Rays {
    /// Number of segments in the ray.
    #[inline]
    pub fn segment_count(&self) -> usize {
        self.layer_indices.len()
    }
}

/// Per-layer path multipliers. Written only by `build_ray_path`.
///
/// The multipliers are `f32` deliberately, matching the single precision of the original
/// code. [`Travel::deepest_layer`] is a layer index, not the segment count
/// [`Rays::segment_count`].
#[derive(Clone, Debug, Default)]
pub struct Travel {
    /// P path multiplier per layer. 0-based by layer, one entry per layer in the model.
    pub p_traversals: Vec<f32>,
    /// S path multiplier per layer, parallel to [`Travel::p_traversals`].
    pub s_traversals: Vec<f32>,
    /// Deepest layer the ray penetrates, as a 0-based layer index. Consumers iterate
    /// `0..=deepest_layer`.
    pub deepest_layer: usize,
    /// Direction the ray leaves the source. Written by `build_ray_path`; nothing reads it.
    pub takeoff: Direction,
}

impl Travel {
    /// Resize both multiplier tables to one entry per layer and zero them, so no multiplier
    /// carries over from the previous ray.
    pub fn reset_for(&mut self, layer_count: usize) {
        self.p_traversals.clear();
        self.p_traversals.resize(layer_count, 0.0);
        self.s_traversals.clear();
        self.s_traversals.resize(layer_count, 0.0);
    }
}

/// Interface interaction types and segment directions. Written only by `build_ray_path`.
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
        self.interactions
            .resize(segment_count, Interaction::default());
        self.directions.clear();
        self.directions.resize(segment_count, Direction::default());
    }
}

/// The full ray-tracing state, threaded through the `green_function` cluster.
///
/// `green_function` writes `rays`, `build_ray_path` writes `travel` and `coefficients`, and
/// `stationary_ray_parameter` and `geometric_spreading` read them back.
#[derive(Clone, Debug, Default)]
pub struct RayState {
    pub rays: Rays,
    pub travel: Travel,
    pub coefficients: Coefficients,
    /// 1 for P-SV, 2 for SH. Written by `build_ray_path`; only the golden test reads it.
    pub love: i32,
}
