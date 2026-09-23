//! The ray tracer's working state: the segment list, the per-layer path multipliers and the
//! interface interactions, as explicit structs.

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
}
