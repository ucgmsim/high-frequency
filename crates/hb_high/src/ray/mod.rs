//! Ray theory: travel times, geometric spreading and path attenuation through a 1-D
//! layered medium.
//!
//! # Where to read the theory
//!
//! The slowness formulation — ray parameter `p`, vertical slowness
//! `eta = sqrt(1/v^2 - p^2)`, and the branch-cut choice [`vertical_slowness`] makes — is
//! standard, and Shearer, *Introduction to Seismology*, ch. 4 ("Ray Theory: Travel Times")
//! is enough to follow everything here.
//!
//! [`cagniard_time`] and [`cagniard_time_derivative`] evaluate the Cagniard–De Hoop
//! integrand: the transform-domain response is written so that the inverse transform can be
//! read off as a time-domain one along a path where the phase is real. De Hoop (1960), "A
//! modification of Cagniard's method for solving seismic pulse problems", *Applied Scientific
//! Research* B8, 349–356, is the paper the method is named for.
//!
//! These citations are signposts rather than verified equation references, so nothing below
//! is annotated with an equation number.
use std::f32::consts::PI;

use crate::fft::Complex64;
use crate::geom::SubfaultRay;
use crate::velocity::{Layer, VelocityModel};

mod state;
pub use state::{Coefficients, Direction, Interaction, RayState, Rays, Travel, WaveMode};

/// One requested ray path: the `j` of Graves & Pitarka (2010) eq. 10's sum over paths — direct,
/// Moho-reflected, and multiples.
///
/// Production runs the direct upgoing ray alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayType {
    /// Not traced: a straight line from subfault to station, with a nominal medium. See
    /// [`trace`].
    StraightLine,
    /// Traced through the layered model.
    Traced(RayShape),
}

/// The topology of a traced ray.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RayShape {
    /// Leaves the source upward, then bounces off the Moho `multiples` extra times.
    Upgoing { multiples: u32 },
    /// Leaves the source downward and reflects off the Moho, then bounces `multiples` extra
    /// times.
    DownToMoho { multiples: u32 },
}

impl RayType {
    /// Decode the integer ray code the configuration carries: `0` is the straight line, and a
    /// positive code is a traced ray, decoded by [`RayShape::from_code`]. Negative codes name
    /// nothing and give `None`.
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::StraightLine),
            _ => RayShape::from_code(code).map(Self::Traced),
        }
    }
}

impl RayShape {
    /// Decode a positive ray code. The **parity** picks the take-off direction — odd leaves
    /// upward, even downward — and each further pair adds one Moho multiple: `1` is the direct
    /// ray, `2` the Moho reflection, `3` the direct ray plus one multiple, and so on.
    pub fn from_code(code: i32) -> Option<Self> {
        let code = u32::try_from(code).ok().filter(|&code| code > 0)?;
        Some(if code % 2 == 1 {
            Self::Upgoing {
                multiples: (code - 1) / 2,
            }
        } else {
            Self::DownToMoho {
                multiples: (code - 2) / 2,
            }
        })
    }

    /// Which way the ray leaves the source.
    pub fn takeoff(self) -> Takeoff {
        match self {
            Self::Upgoing { .. } => Takeoff::Up,
            Self::DownToMoho { .. } => Takeoff::Down,
        }
    }
}

/// Complex vertical slowness `eta = sqrt(1/velocity_km_s^2 - ray_parameter^2)`, with an
/// explicit branch-cut choice.
pub fn vertical_slowness(ray_parameter: Complex64, velocity_km_s: f64) -> Complex64 {
    let t1 = 1.0e-08f64;
    let rsq = 1.0f64 / (velocity_km_s * velocity_km_s);
    let pr = ray_parameter.re;
    // `pi` here is Im(ray_parameter), not the constant.
    let pi = ray_parameter.im;
    let a = rsq - pr * pr + pi * pi;
    let b = -2.0f64 * pi * pr;
    let d = (a * a + b * b).sqrt().sqrt();

    // Near the real axis the phase is forced to 0 or pi rather than taken from
    // atan2, which would be ill-conditioned there.
    let phi = if pi.abs() < t1 {
        if a < 0.0 { std::f64::consts::PI } else { 0.0 }
    } else {
        b.atan2(a)
    };

    let cos_half = (phi / 2.0f64).cos();
    let sin_half = (phi / 2.0f64).sin();

    // The branch choice: negate unless (sin <= t1 and cos > 0).
    let (e, f) = if sin_half > t1 || cos_half <= 0.0 {
        (-cos_half, -sin_half)
    } else {
        (cos_half, sin_half)
    };

    Complex64::new(d * e, d * f)
}

/// Builds the per-layer path multipliers for one ray.
///
/// `source_depth_km` is the source depth, `receiver_depth_km` the receiver depth.
pub fn build_ray_path(
    state: &mut RayState,
    vmod: &VelocityModel,
    source_depth_km: f64,
    receiver_depth_km: f64,
) {
    let n = state.rays.segment_count();
    // A zero-segment ray would underflow every `n - 1` below.
    assert!(n >= 1, "build_ray_path needs at least one ray segment");

    state.travel.reset_for(vmod.len());
    state.coefficients.reset_for(n);

    // Count how many times each layer is traversed, by wave mode.
    for (&layer, &mode) in state.rays.layer_indices.iter().zip(&state.rays.wave_modes) {
        if mode == WaveMode::P {
            state.travel.p_traversals[layer] += 1.0;
        }
        if mode.is_shear() {
            state.travel.s_traversals[layer] += 1.0;
        }
    }

    // Ray direction from the source: nup = +1 up, -1 down. ndeg < 0 forces
    // upgoing, which resolves the ambiguity when source and receiver share a
    // layer.
    let source_layer = state.rays.layer_indices[0];
    let receiver_layer = state.rays.layer_indices[n - 1];
    // Starts at 1, not 0.
    let nl = 1 + state
        .rays
        .layer_indices
        .iter()
        .filter(|&&layer| layer == source_layer)
        .count() as i32;
    let mut nup = Direction::from_parity(nl);
    if receiver_layer > source_layer {
        nup = nup.flipped();
    }
    if state.rays.degeneracy < 0 {
        nup = Direction::Up;
    }
    if n == 1 && receiver_depth_km >= source_depth_km {
        nup = Direction::Down;
    }

    // Interaction type at each interface and direction of each segment.
    state.coefficients.directions[0] = nup;
    if n == 1 {
        state.coefficients.interactions[0] = Interaction::Direct;
    } else {
        for i in 0..n - 1 {
            // Consecutive segments in the same layer means the ray turned around.
            state.coefficients.interactions[i] =
                if state.rays.layer_indices[i + 1] == state.rays.layer_indices[i] {
                    Interaction::Reflection
                } else {
                    Interaction::Transmission
                };
            state.coefficients.directions[i + 1] = match state.coefficients.interactions[i] {
                Interaction::Reflection => state.coefficients.directions[i].flipped(),
                Interaction::Transmission | Interaction::Direct => state.coefficients.directions[i],
            };
        }
    }

    // Both endpoints sit part-way through their layer, so the whole-layer traversal
    // counted above is trimmed by the fraction the ray does not travel.
    //
    // The two calls are not symmetric: `Up` trims by `below` at the receiver and
    // by `above` at the source -- the roles are swapped. Making them symmetric changes every
    // travel time.
    let receiver = layer_fractions(vmod, receiver_layer, receiver_depth_km);
    trim_traversal(
        state,
        receiver_layer,
        n - 1,
        match state.coefficients.directions[n - 1] {
            Direction::Up => receiver.above,
            Direction::Down => receiver.below,
        },
    );

    let source = layer_fractions(vmod, source_layer, source_depth_km);
    trim_traversal(
        state,
        source_layer,
        0,
        match nup {
            Direction::Up => source.below,
            Direction::Down => source.above,
        },
    );

    state.travel.deepest_layer = state.rays.layer_indices.iter().copied().max().unwrap_or(0);
}

/// Where an endpoint sits within its layer, as the two complementary fractions.
struct LayerFractions {
    /// Fraction of the layer above the point.
    above: f64,
    /// Fraction below it. `above + below == 1`.
    below: f64,
}

fn layer_fractions(vmod: &VelocityModel, layer: usize, depth_km: f64) -> LayerFractions {
    let above_km: f64 = vmod[..layer].iter().map(|l| l.thickness_km).sum();
    let into_layer_km = depth_km - above_km;
    let thickness_km = vmod[layer].thickness_km;
    LayerFractions {
        above: into_layer_km / thickness_km,
        below: (thickness_km - into_layer_km) / thickness_km,
    }
}

/// Subtract the part of `layer` that segment `segment` does not actually travel.
///
/// A shear mode takes the S multiplier, anything else the P one.
fn trim_traversal(state: &mut RayState, layer: usize, segment: usize, fraction: f64) {
    let traversals = if state.rays.wave_modes[segment].is_shear() {
        &mut state.travel.s_traversals
    } else {
        &mut state.travel.p_traversals
    };
    traversals[layer] = (traversals[layer] as f64 - fraction) as f32;
}

/// `sin(i)` substituted for a post-critical ray, where the real value would make the
/// geometric-spreading denominator imaginary. Rounded through `f32` to reproduce the
/// single-precision constant of the original code.
const SIN_INCIDENCE_CLAMP: f64 = 0.999999f32 as f64;

/// How far inside the branch cut [`stationary_ray_parameter`] starts its search.
///
/// The ray parameter must sit strictly below `1/v_max`, because [`vertical_slowness`] is
/// singular on the cut.
///
/// One value suffices for every physical velocity. `(1/v - eps) * v` fails to fall below 1
/// only when `eps * v` is under half an ulp of 1.0, which at this magnitude needs `v < 1e-7`
/// km/s — a fastest-traversed-layer velocity of a tenth of a millimetre per second.
const BRANCH_CUT_CLEARANCE: f64 = 1.0e-10;

/// One layer as the Cagniard integrals see it: the medium, and how many times the ray
/// crosses it in each mode.
struct TraversedLayer<'a> {
    layer: &'a Layer,
    p: f32,
    s: f32,
}

/// The layers the ray penetrates, `0..=deepest_layer`, zipped with their multipliers.
fn traversed_layers<'a>(
    state: &'a RayState,
    vmod: &'a VelocityModel,
) -> impl Iterator<Item = TraversedLayer<'a>> {
    let depth = state.travel.deepest_layer;
    vmod[..=depth]
        .iter()
        .zip(&state.travel.p_traversals[..=depth])
        .zip(&state.travel.s_traversals[..=depth])
        .map(|((layer, &p), &s)| TraversedLayer { layer, p, s })
}

/// Returns `(path_length_km, qbar)`: total ray path length, and the path-integrated
/// attenuation operator `sum(t_i / Qs_i)`.
pub fn geometric_spreading(
    state: &RayState,
    vmod: &VelocityModel,
    source_depth_km: f64,
    ray_parameter: f64,
    takeoff: Takeoff,
) -> (f64, f32) {
    let source_layer = state.rays.layer_indices[0];

    // Layers above the source layer, skipping the air layer at index 0. `1..source_layer`
    // rather than `1..=source_layer - 1` so a source in layer 0 cannot underflow.
    let above_source_km: f64 = vmod[1..source_layer].iter().map(|l| l.thickness_km).sum();

    // How far the first segment travels vertically within the source layer.
    let first_segment_km = match takeoff {
        Takeoff::Up => source_depth_km - above_source_km,
        Takeoff::Down => above_source_km + vmod[source_layer].thickness_km - source_depth_km,
    };

    let first = &vmod[source_layer];
    let length_km = first_segment_km * incidence_secant(ray_parameter, first.vsh_km_s);
    let time_s = length_km / first.vsh_km_s;

    let mut path_length_km = length_km;
    let mut qbar = (time_s / first.attenuation_s as f64) as f32;

    for &layer_index in &state.rays.layer_indices[1..] {
        let layer = &vmod[layer_index];
        let length_km = layer.thickness_km * incidence_secant(ray_parameter, layer.vsh_km_s);
        let time_s = length_km / layer.vsh_km_s;

        path_length_km += length_km;
        // Narrowed on every iteration: single-precision accumulation, as in the original code.
        qbar = (qbar as f64 + time_s / layer.attenuation_s as f64) as f32;
    }

    if path_length_km == 0.0 {
        path_length_km = 0.001f32 as f64;
    }
    (path_length_km, qbar)
}

/// `1/cos(i)` for a segment at `velocity_km_s`, from Snell's law `sin(i)/v = p` — the factor
/// that turns a segment's vertical extent into its length along the ray.
///
/// `sin(i)` is replaced by [`SIN_INCIDENCE_CLAMP`] only for a post-critical ray, where
/// `sin(i) >= 1` would make the root imaginary. This is not `min(sin_i, SIN_INCIDENCE_CLAMP)`:
/// a value between the clamp and 1 -- 0.9999995, say -- is post-critical by neither test and is
/// kept exactly, where `min` would pull it down to the clamp and move the spreading.
#[inline]
fn incidence_secant(ray_parameter: f64, velocity_km_s: f64) -> f64 {
    let incidence_sine = ray_parameter * velocity_km_s;
    let sine = if incidence_sine >= 1.0 {
        SIN_INCIDENCE_CLAMP
    } else {
        incidence_sine
    };
    1.0 / (1.0 - sine * sine).sqrt()
}

/// Cagniard complex travel time as a function of complex ray parameter:
/// `tau(ray_parameter) = ray_parameter*range_km + sum_i [eta_p(i)*alp(i)*th(i) + eta_s(i)*als(i)*th(i)]`.
///
/// Note the guard is `alp(i) > 0`, whereas [`cagniard_time_derivative`] uses `alp(i) /= 0`. `alp`
/// can be negative after `build_ray_path`'s source- and receiver-layer adjustments, so
/// the two routines disagree about negative multipliers: `cagniard_time` skips them,
/// `cagniard_time_derivative` does not. Kept so outputs match the original code.
pub fn cagniard_time(
    state: &RayState,
    vmod: &VelocityModel,
    ray_parameter: Complex64,
    range_km: f64,
) -> Complex64 {
    let mut a = Complex64::ZERO;
    for TraversedLayer { layer, p, s } in traversed_layers(state, vmod) {
        let ea = if p > 0.0 {
            vertical_slowness(ray_parameter, layer.vp_km_s)
        } else {
            Complex64::ZERO
        };
        let eb = if s > 0.0 {
            vertical_slowness(ray_parameter, layer.vsh_km_s)
        } else {
            Complex64::ZERO
        };
        a = a + ea * (p as f64) * layer.thickness_km + eb * (s as f64) * layer.thickness_km;
    }
    ray_parameter * range_km + a
}

/// `dtau/dp = range_km - ray_parameter * sum_i [th(i)*alp(i)/eta_p(i) + th(i)*als(i)/eta_s(i)]`.
///
/// The divisions promote the real numerator to complex and do a full complex division —
/// Smith's algorithm, not `(ac+bd)/(c^2+d^2)`. See [`crate::fft::Complex`]'s `Div`.
///
/// The guard here is `/= 0` rather than `> 0`; see [`cagniard_time`].
pub fn cagniard_time_derivative(
    state: &RayState,
    vmod: &VelocityModel,
    ray_parameter: Complex64,
    range_km: f64,
) -> Complex64 {
    let mut a = Complex64::ZERO;
    for TraversedLayer { layer, p, s } in traversed_layers(state, vmod) {
        let b = if p != 0.0 {
            Complex64::from(layer.thickness_km * p as f64)
                / vertical_slowness(ray_parameter, layer.vp_km_s)
        } else {
            Complex64::ZERO
        };
        let c = if s != 0.0 {
            Complex64::from(layer.thickness_km * s as f64)
                / vertical_slowness(ray_parameter, layer.vsh_km_s)
        } else {
            Complex64::ZERO
        };
        a = a + b + c;
    }
    Complex64::from(range_km) - ray_parameter * a
}

/// Finds the geometric ray parameter `p0` and its travel time `t0`, returned as
/// `(p0, t0)`.
///
/// The strategy is to start just inside the nearest branch cut (`1/v_max` over
/// the layers the ray actually traverses) and, if `dtau/dp` is negative there,
/// bisect down towards zero until `|dtau/dp| <= 0.01` or 40 iterations.
pub fn stationary_ray_parameter(
    state: &RayState,
    vmod: &VelocityModel,
    range_km: f64,
) -> (f64, f64) {
    // Closest branch cut, i.e. the highest velocity the ray samples.
    let v =
        traversed_layers(state, vmod).fold(0.0f64, |fastest, TraversedLayer { layer, p, s }| {
            let fastest = if p > 0.0 {
                fastest.max(layer.vp_km_s)
            } else {
                fastest
            };
            if s > 0.0 {
                fastest.max(layer.vsh_km_s)
            } else {
                fastest
            }
        });

    let ptest = 1.0 / v;
    let mut p = Complex64::from(ptest - 10.0 * BRANCH_CUT_CLEARANCE);

    let mut a = cagniard_time_derivative(state, vmod, p, range_km).re;

    if a < 0.0 {
        let mut k = 0;
        let mut pn = p.re;
        let mut pp = 0.0f64;
        loop {
            k += 1;
            p = Complex64::from((pn + pp) / 2.0);
            a = cagniard_time_derivative(state, vmod, p, range_km).re;
            if a.abs() <= 0.01 || k >= 40 {
                break;
            }
            if a > 0.0 {
                pp = p.re;
            } else {
                pn = p.re;
            }
        }
    }

    let p0 = p.re;
    let t = cagniard_time(state, vmod, p, range_km);
    (p0, t.re)
}

/// Take-off direction from the source. See [`RayShape::takeoff`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Takeoff {
    /// The ray leaves the source upward.
    Up,
    /// Down to the Moho, then back up.
    Down,
}

/// The ray segment list under construction.
struct SegmentList<'a> {
    rays: &'a mut Rays,
    mode: WaveMode,
}

impl<'a> SegmentList<'a> {
    fn new(rays: &'a mut Rays, mode: WaveMode) -> Self {
        rays.layer_indices.clear();
        rays.wave_modes.clear();
        Self { rays, mode }
    }

    /// Append one segment. The segment count is the list length, so there is no separate
    /// counter to keep in step.
    fn push(&mut self, layer: usize) {
        self.rays.layer_indices.push(layer);
        self.rays.wave_modes.push(self.mode);
    }

    /// Segments from `from` up to `receiver` inclusive, shallowing.
    fn ascend_to(&mut self, from: usize, receiver: usize) {
        for layer in (receiver..=from).rev() {
            self.push(layer);
        }
    }

    /// Segments from `from` down to the layer above the Moho, deepening. Returns the
    /// deepest layer pushed, which the caller ascends back from.
    ///
    /// The Moho is above the first layer of zero thickness; `build_velocity_model` forces
    /// the bottom layer to zero thickness, so the scan always terminates.
    ///
    /// The two exhaustion cases return different things and both are read: a scan that runs
    /// to completion returns one past the last layer tested, an empty range returns `from`
    /// unchanged. Collapsing them silently changes the ray.
    fn descend_to_moho(&mut self, vmod: &VelocityModel, from: usize, bottom_layer: usize) -> usize {
        let last = bottom_layer - 1;
        // Empty range: nothing to descend through, so the caller ascends from where it is.
        if from > last {
            return from;
        }
        for layer in from..=last {
            self.push(layer);
            if vmod[layer + 1].thickness_km == 0.0 {
                return layer;
            }
        }
        // Ran to completion: one past the last layer tested.
        last + 1
    }

    fn moho_multiple(&mut self, vmod: &VelocityModel, receiver: usize, bottom_layer: usize) {
        let turning_layer = self.descend_to_moho(vmod, receiver, bottom_layer);
        self.ascend_to(turning_layer, receiver);
    }
}

/// How far a source is nudged clear of a layer interface, km.
///
/// A ray starting exactly on a boundary is ambiguous about which layer it is in, so the
/// source depth is moved to one side. Rounded through `f32` to reproduce the single-precision
/// constant of the original code.
const INTERFACE_CLEARANCE_KM: f64 = 0.02f32 as f64;

/// Which layer a source at `depth_km` sits in, and the depth nudged clear of any interface.
///
/// `None` means the source is below every layer, i.e. in the half-space. That is a real case:
/// truncating the velocity model at the Moho can leave subfaults beneath the deepest layer.
///
/// An `Option` rather than a `layer_count` sentinel, because that sentinel is a valid index
/// into anything sized with slack and would silently read a zeroed layer.
fn source_layer(vmod: &VelocityModel, mut depth_km: f64) -> (Option<usize>, f64) {
    let mut interface_km = 0.0f64;
    for (layer, entry) in vmod.iter().enumerate() {
        interface_km += entry.thickness_km;
        // Resting on the interface from above: push down past it.
        if depth_km >= interface_km && (depth_km - interface_km) < INTERFACE_CLEARANCE_KM {
            depth_km = interface_km + INTERFACE_CLEARANCE_KM;
        }
        if depth_km < interface_km {
            // Sitting just under it: pull back up.
            if (interface_km - depth_km) < INTERFACE_CLEARANCE_KM {
                depth_km = interface_km - INTERFACE_CLEARANCE_KM;
            }
            return (Some(layer), depth_km);
        }
    }
    (None, depth_km)
}

/// The deepest layer a source can be placed in, and the depth of its base.
///
/// Not simply the last layer: `build_velocity_model` forces the bottom layer to zero
/// thickness as a half-space marker, and [`build_ray_path`] places a source within its layer
/// by dividing by that thickness. Putting a source in the zero-thickness layer divides by
/// zero and NaNs the travel time.
fn deepest_layer_with_thickness(vmod: &VelocityModel) -> (usize, f64) {
    vmod.iter()
        .enumerate()
        .filter(|(_, layer)| layer.thickness_km > 0.0)
        .fold((0, 0.0), |(_, base_km), (index, layer)| {
            (index, base_km + layer.thickness_km)
        })
}

/// Outputs of [`green_function`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GreenFunction {
    /// The stationary ray parameter `p`, s/km.
    pub ray_parameter_s_per_km: f32,
    pub travel_time_s: f32,
    pub path_length_km: f32,
    /// Path-integrated attenuation, `sum(t_i/Qs_i)`.
    pub qbar: f32,
}

/// Builds the ray segment description for a given source depth and ray shape,
/// then drives [`build_ray_path`], [`stationary_ray_parameter`] and
/// [`geometric_spreading`] to return ray parameter, travel time, path length and path
/// attenuation.
///
/// Sole writer of `state.rays`. Production passes [`WaveMode::Sh`] and the direct upgoing
/// ray, so the multiple loops never run.
///
/// The Moho is taken to be above the first layer of zero thickness, or the
/// deepest layer if none is zero.
///
/// A source below the whole model is in the half-space, so it is placed just inside the
/// deepest layer that has thickness.
///
/// Note that if the source layer is shallower than the receiver layer (index 1) the upgoing
/// segment loop produces zero segments, which [`build_ray_path`] rejects with a panic.
pub fn green_function(
    state: &mut RayState,
    vmod: &VelocityModel,
    source_depth_km: f32,
    range_km: f32,
    shape: RayShape,
    wave_mode: WaveMode,
) -> GreenFunction {
    // The receiver sits in the second layer, index 1, below the air layer.
    let receiver_layer = 1usize;
    let layer_count = vmod.len();
    // Deepest layer, 0-based. The Moho loops below run to `bottom_layer - 1` because they
    // test the thickness of the layer below the one they are on, and the bottom layer is
    // forced to zero thickness by `build_velocity_model`.
    let bottom_layer = layer_count - 1;

    state.rays.degeneracy = 1;
    let receiver_depth_km = vmod[0].thickness_km;

    // A source below every layer is in the half-space. `velocity::layer_containing` already
    // reads that case as "the half-space is the bottom layer"; here it also has to be
    // somewhere the ray geometry can start from, so it goes just inside the base of the
    // deepest layer that has any thickness.
    let (source_layer, source_depth_km) = match source_layer(vmod, source_depth_km as f64) {
        (Some(layer), depth_km) => (layer, depth_km),
        (None, _) => {
            let (deepest, base_km) = deepest_layer_with_thickness(vmod);
            (deepest, base_km - INTERFACE_CLEARANCE_KM)
        }
    };

    let mut path = SegmentList::new(&mut state.rays, wave_mode);
    // Both arms are the same two operations in a different order, plus the same Moho
    // bounce repeated `multiples` times.
    match shape {
        RayShape::Upgoing { multiples } => {
            path.ascend_to(source_layer, receiver_layer);
            for _ in 0..multiples {
                path.moho_multiple(vmod, receiver_layer, bottom_layer);
            }
        }
        RayShape::DownToMoho { multiples } => {
            let turning_layer = path.descend_to_moho(vmod, source_layer, bottom_layer);
            path.ascend_to(turning_layer, receiver_layer);
            for _ in 0..multiples {
                path.moho_multiple(vmod, receiver_layer, bottom_layer);
            }
        }
    }

    build_ray_path(state, vmod, source_depth_km, receiver_depth_km);
    let (ray_parameter, travel_time_s) = stationary_ray_parameter(state, vmod, range_km as f64);
    let (path_length_km, qbar) =
        geometric_spreading(state, vmod, source_depth_km, ray_parameter, shape.takeoff());

    GreenFunction {
        ray_parameter_s_per_km: ray_parameter as f32,
        travel_time_s: travel_time_s as f32,
        path_length_km: path_length_km as f32,
        qbar,
    }
}

/// Nominal `Q₀` for the straight-line path, which does not trace the medium and so cannot
/// integrate the real per-layer attenuation.
const STRAIGHT_LINE_Q: f32 = 150.0;
/// Nominal shear velocity for the same, km/s.
const STRAIGHT_LINE_VELOCITY_KM_S: f32 = 3.7;
/// Where the window starts, as a fraction of the straight-line travel time.
const STRAIGHT_LINE_WINDOW_START_FRACTION: f32 = 0.7;

/// One ray path from a subfault to the station, after tracing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TracedPath {
    /// Path length along the ray, km. Not epicentral distance.
    pub path_length_km: f32,
    /// `q̄`, the travel-time weighted `Σ t/q` along the path (Ou & Herrmann 1990).
    pub qbar: f32,
    /// Take-off angle at the source, radians.
    pub takeoff_rad: f32,
    pub onset: Onset,
}

/// When a path's contribution starts, which the two path models answer differently.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Onset {
    /// A traced ray arrives this long after the subfault ruptures. The shaping window starts
    /// before that, by the envelope's lead-in to its peak.
    Arrival { travel_time_s: f32 },
    /// The straight-line model places the window start directly, with no lead-in.
    WindowStart { window_start_s: f32 },
}

/// Trace one ray from a subfault to the station.
///
/// `shear_velocity_km_s` is `β` at the subfault: the straight-line model's attenuation uses it,
/// and a traced ray's incidence angle follows from it by Snell's law.
pub fn trace(
    state: &mut RayState,
    vmod: &VelocityModel,
    ray_type: RayType,
    geometry: &SubfaultRay,
    shear_velocity_km_s: f32,
) -> TracedPath {
    match ray_type {
        RayType::StraightLine => {
            let path_length_km = geometry.slant_km;
            TracedPath {
                path_length_km,
                qbar: path_length_km / (shear_velocity_km_s * STRAIGHT_LINE_Q),
                takeoff_rad: geometry.takeoff_rad,
                onset: Onset::WindowStart {
                    window_start_s: STRAIGHT_LINE_WINDOW_START_FRACTION * path_length_km
                        / STRAIGHT_LINE_VELOCITY_KM_S,
                },
            }
        }
        RayType::Traced(shape) => {
            let green = green_function(
                state,
                vmod,
                geometry.depth_km,
                geometry.horizontal_km,
                shape,
                WaveMode::Sh,
            );
            // Incidence angle from the ray parameter: `sin(i)/β = p`.
            let sine = shear_velocity_km_s * green.ray_parameter_s_per_km;
            let incidence = if sine > 1.0 { 0.5 * PI } else { sine.asin() };
            TracedPath {
                path_length_km: green.path_length_km,
                qbar: green.qbar,
                // An upgoing ray's take-off is measured from the other pole.
                takeoff_rad: match shape.takeoff() {
                    Takeoff::Up => PI - incidence,
                    Takeoff::Down => incidence,
                },
                onset: Onset::Arrival {
                    travel_time_s: green.travel_time_s,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_codes_split_on_zero_then_parity() {
        assert_eq!(RayType::from_code(0), Some(RayType::StraightLine));
        for (code, shape) in [
            (1, RayShape::Upgoing { multiples: 0 }),
            (2, RayShape::DownToMoho { multiples: 0 }),
            (3, RayShape::Upgoing { multiples: 1 }),
            (4, RayShape::DownToMoho { multiples: 1 }),
        ] {
            assert_eq!(RayType::from_code(code), Some(RayType::Traced(shape)));
        }
        assert_eq!(RayShape::from_code(0), None, "0 is not a traced ray");
        assert_eq!(RayType::from_code(-1), None, "negative codes name nothing");
    }
}
