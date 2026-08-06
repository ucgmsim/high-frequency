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
//! **Neither source is in `papers/`, so unlike the physics modules these citations are
//! signposts rather than verified equation references** — nothing below is annotated with an
//! equation number, because §3 says to cite a paper only after reading it.
use crate::fft::Complex64;
use crate::state::{Direction, Interaction, Layer, RayState, Rays, VelocityModel, WaveMode};

/// `function vertical_slowness(ray_parameter,velocity_km_s)` — `hb_high_ref.f:3349`. Complex vertical slowness
/// `eta = sqrt(1/velocity_km_s^2 - ray_parameter^2)`, with an explicit branch-cut choice.
pub fn vertical_slowness(ray_parameter: Complex64, velocity_km_s: f64) -> Complex64 {
    let t1 = 1.0e-08f64;
    let rsq = 1.0f64 / (velocity_km_s * velocity_km_s);
    let pr = ray_parameter.re;
    // `pi` here is Im(ray_parameter), matching the Fortran's variable name.
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

    // The branch choice, as one expression rather than four reassignments. Negate unless
    // (sin <= t1 and cos > 0):
    //   IF(F.GT.T1) GO TO 13      -> sin > t1 negates
    //   IF(E.GT.0.0D0) GO TO 12   -> otherwise cos > 0 skips the negation
    //   13: e = -e; f = -f
    let (e, f) = if sin_half > t1 || cos_half <= 0.0 {
        (-cos_half, -sin_half)
    } else {
        (cos_half, sin_half)
    };

    Complex64::new(d * e, d * f)
}

/// Builds the per-layer path multipliers for one ray `hb_high_ref.f:3507`.
///
/// `source_depth_km` is the source depth, `receiver_depth_km` the receiver depth.
pub fn build_ray_path(
    state: &mut RayState,
    vmod: &VelocityModel,
    source_depth_km: f64,
    receiver_depth_km: f64,
) {
    state.love = if state.rays.wave_modes[0] == WaveMode::Sh {
        2
    } else {
        1
    };
    let n = state.rays.segment_count();
    // A zero-segment ray would underflow every `n - 1` below. The Fortran read past the
    // array start instead; both are broken, but a named panic beats index arithmetic
    // wrapping.
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
    state.travel.takeoff = nup;

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
    // **The two calls are not the same function.** `Up` trims by `below` at the receiver and
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
/// geometric-spreading denominator imaginary. Unsuffixed literal, so `f32` precision.
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

/// Returns `(rp, qb)`: total ray path length in km, and the path-integrated
/// attenuation operator `sum(t_i / Qs_i)`.
/// `subroutine geometric_spreading(source_depth_km,ray_parameter,ray_type,rp,qb)` — `hb_high_ref.f:3918`.
pub fn geometric_spreading(
    state: &RayState,
    vmod: &VelocityModel,
    source_depth_km: f64,
    ray_parameter: f64,
    takeoff: Takeoff,
) -> (f64, f32) {
    let nh1 = state.rays.layer_indices[0];

    // Layers above the source layer, skipping the air layer at index 0. 0-based this is
    // `1..nh1`, not `1..=nh1 - 1`: same range, but the first spelling cannot underflow
    // when the source is in layer 0 and does not need the `saturating_sub` that hid it.
    let dep: f64 = vmod[1..nh1].iter().map(|l| l.thickness_km).sum();

    let th1 = match takeoff {
        Takeoff::Up => source_depth_km - dep,
        Takeoff::Down => dep + vmod[nh1].thickness_km - source_depth_km,
    };

    // Substituted only for a post-critical ray, where `sin(i) >= 1` would make `denom`
    // imaginary.
    //
    // **This is not `min(sini, SIN_INCIDENCE_CLAMP)`**, tempting as it looks. A value between
    // the clamp and 1 -- 0.9999995, say -- is post-critical by neither test and is kept
    // exactly, where `min` would pull it down to the clamp and move the spreading.
    // Snell's law: sin(i)/v = p.
    let incidence_sine = ray_parameter * vmod[nh1].vsh_km_s;
    let sini = if incidence_sine >= 1.0 {
        SIN_INCIDENCE_CLAMP
    } else {
        incidence_sine
    };
    let denom = 1.0 / (1.0 - sini * sini).sqrt();

    let ri = th1 * denom;
    let ti = ri / vmod[nh1].vsh_km_s;

    let mut rsum = ri;
    let mut qb = (ti / vmod[nh1].attenuation_s as f64) as f32;

    for j in 1..state.rays.segment_count() {
        let nhj = state.rays.layer_indices[j];
        let incidence_sine = ray_parameter * vmod[nhj].vsh_km_s;
        let sini = if incidence_sine >= 1.0 {
            SIN_INCIDENCE_CLAMP
        } else {
            incidence_sine
        };
        let denom = 1.0 / (1.0 - sini * sini).sqrt();

        let ri = vmod[nhj].thickness_km * denom;
        let ti = ri / vmod[nhj].vsh_km_s;

        rsum += ri;
        // Narrowed on every iteration: single-precision accumulation.
        qb = (qb as f64 + ti / vmod[nhj].attenuation_s as f64) as f32;
    }

    if rsum == 0.0 {
        rsum = 0.001f32 as f64;
    }
    (rsum, qb)
}

/// `function cagniard_time(ray_parameter,ir,range_km)` — `hb_high_ref.f:3327`.
///
/// Cagniard complex travel time as a function of complex ray parameter:
/// `tau(ray_parameter) = ray_parameter*range_km + sum_i [eta_p(i)*alp(i)*th(i) + eta_s(i)*als(i)*th(i)]`.
///
/// Note the guard is `alp(i) > 0`, whereas [`cagniard_time_derivative`] uses `alp(i) /= 0`. `alp`
/// can be negative after `build_ray_path`'s source- and receiver-layer adjustments, so
/// the two routines genuinely disagree about negative multipliers: `cagniard_time`
/// skips them, `cagniard_time_derivative` does not. Preserved as-is.
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

/// `function cagniard_time_derivative(ray_parameter,ir,range_km)` — `hb_high_ref.f:3413`.
///
/// `dtau/dp = range_km - ray_parameter * sum_i [th(i)*alp(i)/eta_p(i) + th(i)*als(i)/eta_s(i)]`.
///
/// The divisions are `real*8 / complex*16`, which Fortran evaluates by promoting
/// the numerator to complex and doing a full complex division — Smith's
/// algorithm, not `(ac+bd)/(c^2+d^2)`. See [`crate::fft::Complex`]'s `Div`.
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

/// `subroutine stationary_ray_parameter(ir,p0,t0,range_km)` — `hb_high_ref.f:3441`.
///
/// Finds the geometric ray parameter `p0` and its travel time `t0`, returned as
/// `(p0, t0)`.
///
/// The strategy is to start just inside the nearest branch cut (`1/v_max` over
/// the layers the ray actually traverses) and, if `dtau/dp` is negative there,
/// bisect down towards zero until `|dtau/dp| <= 0.01` or 40 iterations.
///
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

    // Label 12.
    let p0 = p.re;
    let t = cagniard_time(state, vmod, p, range_km);
    (p0, t.re)
}

/// Take-off direction from the source — the parity of the Fortran's `itype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Takeoff {
    /// Odd `itype` — the ray leaves the source upward.
    Up,
    /// Even `itype` — down to the Moho, then back up.
    Down,
}

impl Takeoff {
    /// Every call site passes `ray_type >= 0`. A negative value leaves the take-off angle
    /// undefined, so it is rejected at the boundary rather than deep inside a formula.
    pub fn from_ray_type(ray_type: i32) -> Self {
        assert!(
            ray_type >= 0,
            "ray type {ray_type} is negative; th1 would be undefined"
        );
        if ray_type % 2 == 1 {
            Self::Up
        } else {
            Self::Down
        }
    }
}

/// The ray topology `green_function` builds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RayShape {
    /// Odd. `multiples = (itype - 1) / 2`.
    Upgoing { multiples: i32 },
    /// Even. `multiples = (itype - 2) / 2`.
    DownToMoho { multiples: i32 },
}

impl RayShape {
    fn from_ray_type(ray_type: i32) -> Self {
        match Takeoff::from_ray_type(ray_type) {
            Takeoff::Up => Self::Upgoing {
                multiples: (ray_type - 1) / 2,
            },
            Takeoff::Down => Self::DownToMoho {
                multiples: (ray_type - 2) / 2,
            },
        }
    }
}

/// The ray segment list under construction.
///
/// Replaces a `push` closure plus a loose `l` counter, and gives the four copy-pasted
/// descending loops and three copy-pasted Moho descents one home. `green_function`'s
/// segment-building block was 60 lines of which roughly 45 were duplicates.
struct RayPath<'a> {
    rays: &'a mut Rays,
    mode: WaveMode,
}

impl<'a> RayPath<'a> {
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
    /// The Moho is above the first layer of zero thickness; `read_velocity_model` forces
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
        // Ran to completion: `high + 1`.
        last + 1
    }

    fn moho_multiple(&mut self, vmod: &VelocityModel, receiver: usize, bottom_layer: usize) {
        let kbot = self.descend_to_moho(vmod, receiver, bottom_layer);
        self.ascend_to(kbot, receiver);
    }
}

/// How far a source is nudged clear of a layer interface, km.
///
/// A ray starting exactly on a boundary is ambiguous about which layer it is in, so the
/// source depth is moved to one side. Unsuffixed literal in an `implicit real*8` routine, so
/// it carries only `f32` precision — `PORTING_RULES.md` §1b.
const INTERFACE_CLEARANCE_KM: f64 = 0.02f32 as f64;

/// Which layer a source at `depth_km` sits in, and the depth nudged clear of any interface.
///
/// `None` means the source is below every layer, i.e. in the half-space. That is a real case:
/// truncating the velocity model at the Moho can leave subfaults beneath the deepest layer.
///
/// # Why this is an `Option` and not a sentinel
///
/// A "not found" spelled as `layer_count` is a valid index into anything sized to hold the
/// model plus slack, so it does not trap — it reads a zeroed layer, zero velocity and zero
/// density, and puts NaN travel times into the waveform for those subfaults. The `Option`
/// makes that case impossible to use by accident.
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
    /// Ray parameter.
    pub rp0: f32,
    /// Travel time, seconds.
    pub stime: f32,
    /// Ray path length, km.
    pub rpath: f32,
    /// Path-integrated attenuation, `sum(t_i/Qs_i)`.
    pub qbar: f32,
}

/// Builds the ray segment description for a given source depth and ray type,
/// then drives [`build_ray_path`], [`stationary_ray_parameter`] and
/// [`geometric_spreading`] to return ray parameter, travel time, path length and path
/// attenuation.
/// `subroutine green_function(...)` — `hb_high_ref.f:3174`.
///
///
/// Sole writer of `/rays/`. Production passes [`WaveMode::Sh`]. `ray_type` odd means an upgoing ray, even means
/// down-going then Moho-reflected, and values above 2 add Moho multiples —
/// production passes 1, so the multiple loops never run.
///
/// The Moho is taken to be above the first layer of zero thickness, or the
/// deepest layer if none is zero.
///
/// A source below the whole model is in the half-space, so it is placed just inside the
/// deepest layer that has thickness.
///
/// Note that if the source layer is shallower than `krec` (layer 2) the upgoing segment loop
/// produces zero segments and `nd` is 0, which [`build_ray_path`] is not written to handle.
pub fn green_function(
    state: &mut RayState,
    vmod: &VelocityModel,
    src_depth: f32,
    range: f32,
    ray_type: i32,
    wave_mode: WaveMode,
) -> GreenFunction {
    // The receiver sits in the second layer -- index 1 since §2.3, not 2.
    let krec = 1usize;
    let layer_count = vmod.len();
    // Deepest layer, 0-based. The Moho loops below run to `bottom_layer - 1` because they
    // test the thickness of the layer BELOW the one they are on, and the bottom layer is
    // forced to zero thickness by `read_velocity_model`.
    let bottom_layer = layer_count - 1;

    state.rays.degeneracy = 1;
    let hr = vmod[0].thickness_km;
    let rr = range as f64;

    // A source below every layer is in the half-space. `sim::source_layer_for` already reads
    // that case as "the half-space is the bottom layer"; here it also has to be somewhere the
    // ray geometry can start from, so it goes just inside the base of the deepest layer that
    // has any thickness.
    let (ksrc, hs) = match source_layer(vmod, src_depth as f64) {
        (Some(layer), depth_km) => (layer, depth_km),
        (None, _) => {
            let (deepest, base_km) = deepest_layer_with_thickness(vmod);
            (deepest, base_km - INTERFACE_CLEARANCE_KM)
        }
    };

    let mut path = RayPath::new(&mut state.rays, wave_mode);
    // Both arms are the same two operations in a different order, plus the same Moho
    // bounce repeated `multiples` times. Production passes itype = 1, so `multiples` is
    // 0 and the bounce loops never run.
    match RayShape::from_ray_type(ray_type) {
        RayShape::Upgoing { multiples } => {
            path.ascend_to(ksrc, krec);
            for _ in 0..multiples {
                path.moho_multiple(vmod, krec, bottom_layer);
            }
        }
        RayShape::DownToMoho { multiples } => {
            let kbot = path.descend_to_moho(vmod, ksrc, bottom_layer);
            path.ascend_to(kbot, krec);
            for _ in 0..multiples {
                path.moho_multiple(vmod, krec, bottom_layer);
            }
        }
    }

    build_ray_path(state, vmod, hs, hr);
    let (p0, t0) = stationary_ray_parameter(state, vmod, rr);
    let (rpd, qbar) = geometric_spreading(state, vmod, hs, p0, Takeoff::from_ray_type(ray_type));

    GreenFunction {
        rp0: p0 as f32,
        stime: t0 as f32,
        rpath: rpd as f32,
        qbar,
    }
}
