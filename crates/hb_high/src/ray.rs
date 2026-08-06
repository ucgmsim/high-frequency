//! Ray theoretical calculations.
use crate::fft::Complex64;
use crate::state::{Direction, Interaction, RayState, Rays, VelocityModel, WaveMode};

/// `function vertical_slowness(ray_parameter,velocity_km_s)` — `hb_high_ref.f:3349`. Complex vertical slowness
/// `eta = sqrt(1/velocity_km_s^2 - ray_parameter^2)`, with an explicit branch-cut choice.
pub fn vertical_slowness(ray_parameter: Complex64, velocity_km_s: f64) -> Complex64 {
    let t1 = 1.0e-08f64;
    let rsq = 1.0f64 / (velocity_km_s * velocity_km_s);
    let pr = ray_parameter.re;
    // `pi` here is Im(ray_parameter), matching the Fortran's variable name.
    let pi = ray_parameter.im;
    let mut a = rsq - pr * pr + pi * pi;
    let mut b = -2.0f64 * pi * pr;
    let d = (a * a + b * b).sqrt().sqrt();

    // Near the real axis the phase is forced to 0 or pi rather than taken from
    // atan2, which would be ill-conditioned there.
    let phi = if pi.abs() < t1 {
        if a < 0.0 {
            std::f64::consts::PI
        } else {
            0.0
        }
    } else {
        b.atan2(a)
    };

    let mut e = (phi / 2.0f64).cos();
    let mut f = (phi / 2.0f64).sin();

    // Labels 13/12: negate unless (f <= t1 and e > 0).
    //   IF(F.GT.T1) GO TO 13      -> f > t1 negates
    //   IF(E.GT.0.0D0) GO TO 12   -> otherwise e > 0 skips the negation
    //   13: e = -e; f = -f
    if f > t1 || e <= 0.0 {
        e = -e;
        f = -f;
    }

    a = d * e;
    b = d * f;
    Complex64::new(a, b)
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
    state.love = if state.rays.nm[0] == WaveMode::Sh {
        2
    } else {
        1
    };
    let n = state.rays.nd as usize;
    // The n == 0 case the doc comment describes would underflow every `n - 1` below.
    // The Fortran read past the array start instead; both are broken, but a named panic
    // beats an index arithmetic overflow.
    assert!(
        n >= 1,
        "build_ray_path needs at least one ray segment, got nd = 0"
    );

    // FIXED (§3.4). The Fortran zeroes `alp(1:100)` while the arrays are dimensioned
    // `nlaymax = 500`, so a model deeper than 100 layers inherits multipliers from the
    // PREVIOUS ray and silently produces wrong travel times. Production uses 34 layers,
    // so this was harmless there and is why it survived -- but it is wrong for any deeper
    // model, and nothing warned.
    state.travel.alp.fill(0.0);
    state.travel.als.fill(0.0);

    // Count how many times each layer is traversed, by wave mode. Both indices are
    // 0-based since §2.3: `i` over segments, and the layer numbers stored in `nh`.
    for (&layer, &mode) in state.rays.nh[..n].iter().zip(&state.rays.nm[..n]) {
        let h = layer as usize;
        if mode == WaveMode::P {
            state.travel.alp[h] += 1.0;
        }
        if mode.is_shear() {
            state.travel.als[h] += 1.0;
        }
    }

    // Ray direction from the source: nup = +1 up, -1 down. ndeg < 0 forces
    // upgoing, which resolves the ambiguity when source and receiver share a
    // layer.
    let lis = state.rays.nh[0] as usize;
    let lir = state.rays.nh[n - 1] as usize;
    // Starts at 1, not 0: the Fortran's `nl = 1` before the count.
    let nl = 1 + state.rays.nh[..n]
        .iter()
        .filter(|&&h| h as usize == lis)
        .count() as i32;
    // `(-1)**nl` in the Fortran -- integer exponentiation extracting a parity bit.
    let mut nup = Direction::from_parity(nl);
    if lir > lis {
        nup = nup.flipped();
    }
    if state.rays.ndeg < 0 {
        nup = Direction::Up;
    }
    if n == 1 && receiver_depth_km >= source_depth_km {
        nup = Direction::Down;
    }
    state.travel.nup = nup;

    // Interaction type at each interface and direction of each segment.
    let n1 = n - 1;
    state.coff.nup1[0] = nup;
    if n != 1 {
        for i in 0..n1 {
            let k = state.rays.nh[i];
            let m = state.rays.nh[i + 1];
            // Consecutive segments in the same layer means the ray turned around.
            state.coff.it[i] = if m == k {
                Interaction::Reflection
            } else {
                Interaction::Transmission
            };
            state.coff.nup1[i + 1] = match state.coff.it[i] {
                Interaction::Reflection => state.coff.nup1[i].flipped(),
                Interaction::Transmission | Interaction::Direct => state.coff.nup1[i],
            };
        }
    }
    if n == 1 {
        state.coff.it[0] = Interaction::Direct;
    }

    // Receiver position within its layer: total thickness of everything above it.
    let thtot: f64 = vmod.layers()[..lir].iter().map(|l| l.thickness_km).sum();
    let hrl = receiver_depth_km - thtot;
    let a1 = hrl / vmod[lir].thickness_km;
    let a2 = (vmod[lir].thickness_km - hrl) / vmod[lir].thickness_km;
    let nupa = state.coff.nup1[n - 1];
    // Labels 23/24: a shear mode takes the S multiplier, anything else the P one.
    let trim = match nupa {
        Direction::Up => a1,
        Direction::Down => a2,
    };
    let multiplier = if state.rays.nm[n - 1].is_shear() {
        &mut state.travel.als
    } else {
        &mut state.travel.alp
    };
    multiplier[lir] = (multiplier[lir] as f64 - trim) as f32;

    // Source position within its layer, same as the receiver block above.
    let thtot: f64 = vmod.layers()[..lis].iter().map(|l| l.thickness_km).sum();
    let hsl = source_depth_km - thtot;
    let a1 = hsl / vmod[lis].thickness_km;
    let a2 = (vmod[lis].thickness_km - hsl) / vmod[lis].thickness_km;
    // Note the a1/a2 roles are SWAPPED relative to the receiver block above: `Up`
    // subtracts a2 here but a1 there. That is what the Fortran does, and it is the one
    // asymmetry that makes these two blocks not quite the same function.
    let trim = match nup {
        Direction::Up => a2,
        Direction::Down => a1,
    };
    let multiplier = if state.rays.nm[0].is_shear() {
        &mut state.travel.als
    } else {
        &mut state.travel.alp
    };
    multiplier[lis] = (multiplier[lis] as f64 - trim) as f32;

    // Deepest layer the ray penetrates.
    // Folded from 0 rather than `max().unwrap()`: the Fortran seeds `ndeep = 0`, so a ray
    // whose layers were all negative would keep the 0. Unreachable, but preserved.
    state.travel.ndeep = state.rays.nh[..n].iter().copied().fold(0i32, i32::max);
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
    let nh1 = state.rays.nh[0] as usize;

    // Layers above the source layer, skipping the air layer at index 0. 0-based this is
    // `1..nh1`, not `1..=nh1 - 1`: same range, but the first spelling cannot underflow
    // when the source is in layer 0 and does not need the `saturating_sub` that hid it.
    let dep: f64 = vmod.layers()[1..nh1].iter().map(|l| l.thickness_km).sum();

    let th1 = match takeoff {
        Takeoff::Up => source_depth_km - dep,
        Takeoff::Down => dep + vmod[nh1].thickness_km - source_depth_km,
    };

    let clamp = 0.999999f32 as f64;

    let mut sini = ray_parameter * vmod[nh1].vsh_km_s;
    if sini >= 1.0 {
        sini = clamp;
    }
    let denom = 1.0 / (1.0 - sini * sini).sqrt();

    let ri = th1 * denom;
    let ti = ri / vmod[nh1].vsh_km_s;

    let mut rsum = ri;
    let mut qb = (ti / vmod[nh1].attenuation_s as f64) as f32;

    for j in 1..state.rays.nd as usize {
        let nhj = state.rays.nh[j] as usize;
        let mut sini = ray_parameter * vmod[nhj].vsh_km_s;
        if sini >= 1.0 {
            sini = clamp;
        }
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
    for i in 0..=state.travel.ndeep as usize {
        let mut ea = Complex64::ZERO;
        let mut eb = Complex64::ZERO;
        if state.travel.alp[i] > 0.0 {
            ea = vertical_slowness(ray_parameter, vmod[i].vp_km_s);
        }
        if state.travel.als[i] > 0.0 {
            eb = vertical_slowness(ray_parameter, vmod[i].vsh_km_s);
        }
        a = a
            + ea * (state.travel.alp[i] as f64) * vmod[i].thickness_km
            + eb * (state.travel.als[i] as f64) * vmod[i].thickness_km;
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
    for i in 0..=state.travel.ndeep as usize {
        let mut b = Complex64::ZERO;
        let mut c = Complex64::ZERO;
        if state.travel.alp[i] != 0.0 {
            let ea = vertical_slowness(ray_parameter, vmod[i].vp_km_s);
            b = Complex64::from(vmod[i].thickness_km * state.travel.alp[i] as f64) / ea;
        }
        if state.travel.als[i] != 0.0 {
            let eb = vertical_slowness(ray_parameter, vmod[i].vsh_km_s);
            c = Complex64::from(vmod[i].thickness_km * state.travel.als[i] as f64) / eb;
        }
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
    let mut v = 0.0f64;
    for i in 0..=state.travel.ndeep as usize {
        if state.travel.alp[i] > 0.0 {
            v = v.max(vmod[i].vp_km_s);
        }
        if state.travel.als[i] > 0.0 {
            v = v.max(vmod[i].vsh_km_s);
        }
    }

    let mut eps = 1.0e-10f64;
    let ptest = 1.0 / v;

    loop {
        let rp = (ptest - eps) * v;
        if rp >= 1.0 {
            eps *= 10.0;
        } else {
            break;
        }
    }

    let mut p = Complex64::from(ptest - 10.0 * eps);

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
    /// Every call site passes `itype >= 1`; a negative value is what the Fortran leaves
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
    len: usize,
    mode: WaveMode,
}

impl<'a> RayPath<'a> {
    fn new(rays: &'a mut Rays, mode: WaveMode) -> Self {
        Self { rays, len: 0, mode }
    }

    /// 0-based: write at the current count, then advance it. The Fortran pre-increments,
    /// so `l` ends at the same segment count either way.
    fn push(&mut self, layer: usize) {
        self.rays.nh[self.len] = layer as i32;
        self.rays.nm[self.len] = self.mode;
        self.len += 1;
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
    /// The return value carries the Fortran's `DO` post-loop semantics, which is why this
    /// is a method and not an inlined loop: a range that runs to completion leaves
    /// `high + 1`, an EMPTY range leaves `low` untouched, and `green_function` reads the
    /// loop variable afterwards as `kbot`. Getting that wrong silently changes the ray.
    fn descend_to_moho(&mut self, vmod: &VelocityModel, from: usize, bottom_layer: usize) -> usize {
        let last = bottom_layer - 1;
        // The empty-range case: `low > high` leaves the Fortran's loop variable at `low`.
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

    fn finish(self) -> usize {
        self.len
    }
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
/// then drives [`build_ray_path`], [`stationary_ray_parameter`], [`travel_time`] and [`geometric_spreading`] to return ray
/// parameter, travel time, path length and path attenuation.
/// `subroutine green_function(...)` — `hb_high_ref.f:3174`.
///
///
/// Sole writer of `/rays/`. Production passes [`WaveMode::Sh`]. `ray_type` odd means an upgoing ray, even means
/// down-going then Moho-reflected, and values above 2 add Moho multiples —
/// production passes 1, so the multiple loops never run.
///
/// The `sgc` argument is declared `complex` in the Fortran and never referenced;
/// omitted here. `rcv`, `cp0` and `gc` are likewise declared and unused.
///
/// The Moho is taken to be above the first layer of zero thickness, or the
/// deepest layer if none is zero.
///
/// # `ksrc` can come out as `layer_count + 1`
///
/// The source-layer search is a `DO ksrc = 1, layer_count` that exits early via `goto`
/// once the accumulated depth passes `hs`. If it never does — a source below the
/// whole model — the loop runs to completion and Fortran leaves the loop variable
/// at `layer_count + 1`, which then becomes the first ray segment's layer index. That is
/// a latent out-of-range read in the original. It is reproduced rather than
/// clamped; in Rust it surfaces as a bounds panic instead of silently reading
/// past the model. See `PORTING_RULES.md` §7.
///
/// Note also that if `ksrc < krec` (a source shallower than layer 2) the upgoing
/// segment loop produces zero segments and `nd` is 0, which `build_ray_path` is not
/// written to handle.
///
/// `hs_tol = 0.02` is an unsuffixed literal in an `implicit real*8` routine, so
/// it carries only `f32` precision — see `PORTING_RULES.md` §1b.
pub fn green_function(
    state: &mut RayState,
    vmod: &VelocityModel,
    layer_count: usize,
    src_depth: f32,
    range: f32,
    ray_type: i32,
    wave_mode: WaveMode,
) -> GreenFunction {
    // The receiver sits in the second layer -- index 1 since §2.3, not 2.
    let krec = 1usize;
    // Deepest layer, 0-based. The Moho loops below run to `bottom_layer - 1` because they
    // test the thickness of the layer BELOW the one they are on, and the bottom layer is
    // forced to zero thickness by `read_velocity_model`.
    let bottom_layer = layer_count - 1;

    state.rays.ndeg = 1;
    let hr = vmod[0].thickness_km;
    let mut hs = src_depth as f64;
    let rr = range as f64;

    // Find the source layer, nudging hs off an interface by hs_tol either way
    // so the ray does not start exactly on a boundary.
    let hs_tol = 0.02f32 as f64;
    let mut dep = 0.0f64;
    // Loop-completion value. The Fortran's DO ksrc = 1, layer_count leaves layer_count+1;
    // 0-based that is `layer_count`, one past the last layer. It is READ in that state --
    // see the doc comment -- which is why the arrays stay NLAYMAX-sized.
    // The Fortran's `DO ksrc = 1, layer_count` leaves the loop variable at
    // `layer_count + 1` when it runs to completion; 0-based that is `layer_count`, one
    // past the last layer. It is READ in that state -- see the doc comment -- which is
    // why the arrays stay NLAYMAX-sized.
    let mut ksrc = layer_count;
    for k in 0..layer_count {
        dep += vmod[k].thickness_km;
        if hs >= dep && (hs - dep) < hs_tol {
            hs = dep + hs_tol;
        }
        if hs < dep {
            if (dep - hs) < hs_tol {
                hs = dep - hs_tol;
            }
            ksrc = k;
            break;
        }
    }

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
    state.rays.nd = path.finish() as i32;

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
