//! Ray theory. `cagniard_time`, `cagniard_time_derivative`, `stationary_ray_parameter`, `travel_time` and `green_function` follow in
//! tiers 2-4.

use crate::fort::Complex64;
use crate::state::{RayState, VelocityModel};

/// `function vertical_slowness(ray_parameter,velocity_km_s)` — `hb_high_ref.f:3349`. Complex vertical slowness
/// `eta = sqrt(1/velocity_km_s^2 - ray_parameter^2)`, with an explicit branch-cut choice.
///
/// This is the numerically delicate heart of the ray code. It evaluates the
/// square root in polar form rather than algebraically so the branch can be
/// selected deliberately, and the selection at labels 12/13 must be
/// transliterated literally — see `PORTING_RULES.md` §2.
///
/// Two traps in the original worth naming:
///
/// * `pr = ray_parameter` assigns a `complex*16` to a `real*8`, which silently takes the
///   real part. It is not a typo for `dreal(ray_parameter)`.
/// * the local named `pi` is `dimag(ray_parameter)`, the **imaginary part of ray_parameter**, not
///   3.14159. The actual pi appears separately, as the Fortran's truncated 10-digit
///   literal `3.141592654d0`; it is `std::f64::consts::PI` here.
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
        if a < 0.0 { std::f64::consts::PI } else { 0.0 }
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

/// `subroutine build_ray_path(ray_index,source_depth_km,receiver_depth_km)` — `hb_high_ref.f:3507`.
///
/// Builds the per-layer path multipliers for one ray. Sole writer of
/// `/travel/` (`alp`, `als`, `ndeep`, `nup`), `/coff/` (`it`, `nup1`) and
/// `/rmode/` (`love`); reads `/rays/` and `/vmod/thickness_km`.
///
/// `source_depth_km` is the source depth, `receiver_depth_km` the receiver depth. `ray_index` is retained to match
/// the Fortran signature but is always 1 — `/rays/` has a degenerate leading
/// dimension of 1.
///
/// # Known bug, reproduced
///
/// The zeroing loop runs `i = 1,100` while `alp`/`als` are dimensioned
/// `nlaymax = 500`. Layers above 100 therefore retain multipliers from the
/// previous ray. Harmless at the ~34 layers production uses, but it is not
/// widened here: doing so would change results for any deeper model, silently.
/// See `PORTING_RULES.md` §7.
pub fn build_ray_path(state: &mut RayState, vmod: &VelocityModel, ray_index: usize, source_depth_km: f64, receiver_depth_km: f64) {
    assert_eq!(ray_index, 1, "/rays/ has a degenerate leading dimension; ray_index must be 1");

    state.love = 1;
    if state.rays.nm[0] == 4 {
        state.love = 2;
    }
    let n = state.rays.nd as usize;
    // The n == 0 case the doc comment describes would underflow every `n - 1` below.
    // The Fortran read past the array start instead; both are broken, but a named panic
    // beats an index arithmetic overflow.
    assert!(n >= 1, "build_ray_path needs at least one ray segment, got nd = 0");

    // DO 10 I=1,100 -- deliberately not NLAYMAX. See the note above. 0-based, so this is
    // layers 0..100, the same hundred layers the Fortran zeroed.
    for i in 0..100 {
        state.travel.alp[i] = 0.0;
        state.travel.als[i] = 0.0;
    }

    // Count how many times each layer is traversed, by wave mode. Both indices are
    // 0-based since §2.3: `i` over segments, and the layer numbers stored in `nh`.
    for i in 0..n {
        let h = state.rays.nh[i] as usize;
        if state.rays.nm[i] == 5 {
            state.travel.alp[h] += 1.0;
        }
        if state.rays.nm[i] == 3 || state.rays.nm[i] == 4 {
            state.travel.als[h] += 1.0;
        }
    }

    // Ray direction from the source: nup = +1 up, -1 down. ndeg < 0 forces
    // upgoing, which resolves the ambiguity when source and receiver share a
    // layer.
    let lis = state.rays.nh[0] as usize;
    let lir = state.rays.nh[n - 1] as usize;
    let mut nl = 1i32;
    for i in 0..n {
        if state.rays.nh[i] as usize == lis {
            nl += 1;
        }
    }
    let mut nup = (-1i32).pow(nl as u32);
    if lir > lis {
        nup = -nup;
    }
    if state.rays.ndeg < 0 {
        nup = 1;
    }
    if n == 1 && receiver_depth_km >= source_depth_km {
        nup = -1;
    }
    state.travel.nup = nup;

    // Interaction type at each interface and direction of each segment.
    let n1 = n - 1;
    state.coff.nup1[0] = nup;
    if n != 1 {
        for i in 0..n1 {
            let k = state.rays.nh[i];
            let m = state.rays.nh[i + 1];
            state.coff.it[i] = if m == k { 1 } else { 0 };
            state.coff.nup1[i + 1] = match (state.coff.nup1[i], state.coff.it[i]) {
                (1, 1) => -1,
                (-1, 1) => 1,
                (1, 0) => 1,
                (-1, 0) => -1,
                // The Fortran is four independent IFs with no else, so an
                // unexpected pair would leave nup1(i+1) at its previous value.
                // That cannot arise: it is 0 or 1 by construction just above,
                // and nup1 is +-1 by induction from nup.
                (a, b) => panic!("unreachable nup1/it combination ({a},{b})"),
            };
        }
    }
    if n == 1 {
        state.coff.it[0] = 2;
    }

    // Receiver position within its layer: total thickness of everything above it.
    // The Fortran sums layers 1..lir-1, which 0-based is indices 0..lir-1 -- so `0..lir`,
    // NOT `1..=lir - 1`. Getting this wrong drops the air layer from the sum and moves the
    // receiver, which is exactly the kind of silent one-layer error §2.3 is prone to.
    let mut thtot = 0.0f64;
    for i in 0..lir {
        thtot = vmod.thickness_km[i] + thtot;
    }
    let hrl = receiver_depth_km - thtot;
    let a1 = hrl / vmod.thickness_km[lir];
    let a2 = (vmod.thickness_km[lir] - hrl) / vmod.thickness_km[lir];
    let nupa = state.coff.nup1[n - 1];
    // Labels 23/24: mode 5 takes the P multiplier, modes 3 and 4 the S one,
    // and anything else falls through to P.
    if state.rays.nm[n - 1] == 3 || state.rays.nm[n - 1] == 4 {
        if nupa == 1 {
            state.travel.als[lir] = (state.travel.als[lir] as f64 - a1) as f32;
        }
        if nupa == -1 {
            state.travel.als[lir] = (state.travel.als[lir] as f64 - a2) as f32;
        }
    } else {
        if nupa == 1 {
            state.travel.alp[lir] = (state.travel.alp[lir] as f64 - a1) as f32;
        }
        if nupa == -1 {
            state.travel.alp[lir] = (state.travel.alp[lir] as f64 - a2) as f32;
        }
    }

    // Source position within its layer, same as the receiver block above.
    let mut thtot = 0.0f64;
    for i in 0..lis {
        thtot = vmod.thickness_km[i] + thtot;
    }
    let hsl = source_depth_km - thtot;
    let a1 = hsl / vmod.thickness_km[lis];
    let a2 = (vmod.thickness_km[lis] - hsl) / vmod.thickness_km[lis];
    // Note the a1/a2 roles are swapped relative to the receiver block above:
    // nup == 1 subtracts a2 here but a1 there. That is what the Fortran does.
    if state.rays.nm[0] == 3 || state.rays.nm[0] == 4 {
        if nup == 1 {
            state.travel.als[lis] = (state.travel.als[lis] as f64 - a2) as f32;
        }
        if nup == -1 {
            state.travel.als[lis] = (state.travel.als[lis] as f64 - a1) as f32;
        }
    } else {
        if nup == 1 {
            state.travel.alp[lis] = (state.travel.alp[lis] as f64 - a2) as f32;
        }
        if nup == -1 {
            state.travel.alp[lis] = (state.travel.alp[lis] as f64 - a1) as f32;
        }
    }

    // Deepest layer the ray penetrates.
    let mut ndeep = 0i32;
    for i in 0..n {
        ndeep = ndeep.max(state.rays.nh[i]);
    }
    state.travel.ndeep = ndeep;
}

/// `subroutine geometric_spreading(source_depth_km,ray_parameter,ray_type,rp,qb)` — `hb_high_ref.f:3918`.
///
/// Returns `(rp, qb)`: total ray path length in km, and the path-integrated
/// attenuation operator `sum(t_i / Qs_i)`.
///
/// `ray_type` odd means upgoing, even means downgoing/Moho-reflected — the source
/// comments call this "hardwired to direct and 1 down-going Moho".
///
/// # Precision
///
/// `qb` is `real*4` while every other local is `real*8` under
/// `implicit real*8 (a-h,o-z)`, so **the attenuation sum accumulates in single
/// precision**: each `qb = qb + ti/attenuation_s(...)` promotes, adds in double, and
/// narrows straight back. Accumulating in `f64` and narrowing once at the end
/// would be more accurate and would not match.
///
/// Both magic literals are unsuffixed in the Fortran, so they carry only `f32`
/// precision even in this `real*8` routine — see `PORTING_RULES.md` §1b. This
/// is why they are written `0.999999f32 as f64` rather than as plain `f64`
/// literals; the difference shows up around the 30th bit.
pub fn geometric_spreading(
    state: &RayState,
    vmod: &VelocityModel,
    source_depth_km: f64,
    ray_parameter: f64,
    ray_type: i32,
) -> (f64, f32) {
    let nh1 = state.rays.nh[0] as usize;

    // Layers above the source layer, skipping the air layer at index 0. 0-based this is
    // `1..nh1`, not `1..=nh1 - 1`: same range, but the first spelling cannot underflow
    // when the source is in layer 0 and does not need the `saturating_sub` that hid it.
    let mut dep = 0.0f64;
    for j in 1..nh1 {
        dep += vmod.thickness_km[j];
    }

    let m = ray_type % 2;
    let th1 = if m == 1 {
        source_depth_km - dep
    } else if m == 0 {
        dep + vmod.thickness_km[nh1] - source_depth_km
    } else {
        // The Fortran has two IFs and no else, so a negative odd ray_type would
        // leave th1 undefined. Every call site passes ray_type >= 1.
        panic!("geometric_spreading: ray_type {ray_type} gives mod {m}, leaving th1 undefined");
    };

    let clamp = 0.999999f32 as f64;

    let mut sini = ray_parameter * vmod.vsh_km_s[nh1];
    if sini >= 1.0 {
        sini = clamp;
    }
    let denom = 1.0 / (1.0 - sini * sini).sqrt();

    let ri = th1 * denom;
    let ti = ri / vmod.vsh_km_s[nh1];

    let mut rsum = ri;
    let mut qb = (ti / vmod.attenuation_s[nh1] as f64) as f32;

    for j in 1..state.rays.nd as usize {
        let nhj = state.rays.nh[j] as usize;
        let mut sini = ray_parameter * vmod.vsh_km_s[nhj];
        if sini >= 1.0 {
            sini = clamp;
        }
        let denom = 1.0 / (1.0 - sini * sini).sqrt();

        let ri = vmod.thickness_km[nhj] * denom;
        let ti = ri / vmod.vsh_km_s[nhj];

        rsum += ri;
        // Narrowed on every iteration: single-precision accumulation.
        qb = (qb as f64 + ti / vmod.attenuation_s[nhj] as f64) as f32;
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
/// The loop bound is `/travel/` slot 3, which `build_ray_path` writes as `ndeep` and this
/// routine declares as `nd`. It is **not** `/rays/nd`, which this routine also
/// has in scope. See `PORTING_RULES.md` §6.
///
/// `ir` is unused — kept to match the Fortran signature.
///
/// Note the guard is `alp(i) > 0`, whereas [`cagniard_time_derivative`] uses `alp(i) /= 0`. `alp`
/// can be negative after `build_ray_path`'s source- and receiver-layer adjustments, so
/// the two routines genuinely disagree about negative multipliers: `cagniard_time`
/// skips them, `cagniard_time_derivative` does not. Preserved as-is.
pub fn cagniard_time(state: &RayState, vmod: &VelocityModel, ray_parameter: Complex64, _ray_index: usize, range_km: f64) -> Complex64 {
    let mut a = Complex64::ZERO;
    for i in 0..=state.travel.ndeep as usize {
        let mut ea = Complex64::ZERO;
        let mut eb = Complex64::ZERO;
        if state.travel.alp[i] > 0.0 {
            ea = vertical_slowness(ray_parameter, vmod.vp_km_s[i]);
        }
        if state.travel.als[i] > 0.0 {
            eb = vertical_slowness(ray_parameter, vmod.vsh_km_s[i]);
        }
        a = a + ea * (state.travel.alp[i] as f64) * vmod.thickness_km[i]
              + eb * (state.travel.als[i] as f64) * vmod.thickness_km[i];
    }
    ray_parameter * range_km + a
}

/// `function cagniard_time_derivative(ray_parameter,ir,range_km)` — `hb_high_ref.f:3413`.
///
/// `dtau/dp = range_km - ray_parameter * sum_i [th(i)*alp(i)/eta_p(i) + th(i)*als(i)/eta_s(i)]`.
///
/// The divisions are `real*8 / complex*16`, which Fortran evaluates by promoting
/// the numerator to complex and doing a full complex division — Smith's
/// algorithm, not `(ac+bd)/(c^2+d^2)`. See [`crate::fort::Complex`]'s `Div`.
///
/// `ir` is unused. The guard here is `/= 0` rather than `> 0`; see [`cagniard_time`].
pub fn cagniard_time_derivative(state: &RayState, vmod: &VelocityModel, ray_parameter: Complex64, _ray_index: usize, range_km: f64) -> Complex64 {
    let mut a = Complex64::ZERO;
    for i in 0..=state.travel.ndeep as usize {
        let mut b = Complex64::ZERO;
        let mut c = Complex64::ZERO;
        if state.travel.alp[i] != 0.0 {
            let ea = vertical_slowness(ray_parameter, vmod.vp_km_s[i]);
            b = Complex64::from(vmod.thickness_km[i] * state.travel.alp[i] as f64) / ea;
        }
        if state.travel.als[i] != 0.0 {
            let eb = vertical_slowness(ray_parameter, vmod.vsh_km_s[i]);
            c = Complex64::from(vmod.thickness_km[i] * state.travel.als[i] as f64) / eb;
        }
        a = a + b + c;
    }
    Complex64::from(range_km) - ray_parameter * a
}

/// `subroutine stationary_ray_parameter(ray_index,p0,t0,range_km)` — `hb_high_ref.f:3441`.
///
/// Finds the geometric ray parameter `p0` and its travel time `t0`, returned as
/// `(p0, t0)`.
///
/// The strategy is to start just inside the nearest branch cut (`1/v_max` over
/// the layers the ray actually traverses) and, if `dtau/dp` is negative there,
/// bisect down towards zero until `|dtau/dp| <= 0.01` or 40 iterations.
///
/// # Silent real-part extraction, three times
///
/// `a`, `pn`, `p0` and `t0` are all `real*8` under `implicit real*8 (a-h,o-z)`
/// while `cagniard_time_derivative` and `cagniard_time` return `complex*16`. Fortran assigns the real part
/// without comment. These are not typos for `dreal(...)` — they are the
/// intended behaviour, and the `.re` accesses below are the same operation made
/// visible.
///
/// # The `222` loop is a rounding workaround, not a physical one
///
/// `eps` is set to `1.0d-20` and then immediately `1.0d-10`. The loop then grows
/// it by factors of ten until `(ptest - eps)*v` is genuinely below 1, and a
/// final factor of ten is applied on top — the source comment says
/// "add another factor of 10 just to be sure-> problems on Linux". Reproduced
/// exactly, including the redundant first assignment.
///
/// The `> 0` guard on `alp`/`als` matches [`cagniard_time`], not [`cagniard_time_derivative`]. That
/// mismatch has a consequence: `v` is the largest velocity among layers with a
/// *positive* multiplier, while `cagniard_time_derivative` sums over every layer with a *nonzero*
/// one. So the layer defining `v` is always in `cagniard_time_derivative`'s sum, and as `p`
/// approaches `1/v` that layer's `eta` approaches zero and its term diverges.
///
/// # The immediate-return path is unreachable
///
/// Consequently `a` is always large and negative at the starting point and the
/// bisection always runs — `if(a.lt.0.) go to 11` is effectively unconditional.
/// Measured over 72 cases with `range_km` from 0.5 to 400 km, the largest `a` seen was
/// -2106. Every case also exits on the `|a| <= 0.01` tolerance; the
/// 40-iteration cap never fires. Both facts are pinned in `tier3_golden.rs`.
pub fn stationary_ray_parameter(state: &RayState, vmod: &VelocityModel, ray_index: usize, range_km: f64) -> (f64, f64) {
    // Closest branch cut, i.e. the highest velocity the ray samples.
    let mut v = 0.0f64;
    for i in 0..=state.travel.ndeep as usize {
        if state.travel.alp[i] > 0.0 {
            v = v.max(vmod.vp_km_s[i]);
        }
        if state.travel.als[i] > 0.0 {
            v = v.max(vmod.vsh_km_s[i]);
        }
    }

    #[allow(unused_assignments)]
    let mut eps = 1.0e-20f64;
    eps = 1.0e-10;
    let ptest = 1.0 / v;

    // Label 222: grow eps until backing off from the cut actually lands below
    // it in floating point.
    loop {
        let rp = (ptest - eps) * v;
        if rp >= 1.0 {
            eps = 10.0 * eps;
        } else {
            break;
        }
    }

    let mut p = Complex64::from(ptest - 10.0 * eps);

    // Real part of a complex*16, assigned to a real*8.
    let mut a = cagniard_time_derivative(state, vmod, p, ray_index, range_km).re;

    if a < 0.0 {
        // Label 11: bisect between pn (where dtau/dp < 0) and pp.
        let mut k = 0;
        let mut pn = p.re;
        let mut pp = 0.0f64;
        loop {
            k += 1;
            p = Complex64::from((pn + pp) / 2.0);
            a = cagniard_time_derivative(state, vmod, p, ray_index, range_km).re;
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
    let t = cagniard_time(state, vmod, p, ray_index, range_km);
    (p0, t.re)
}

/// `subroutine travel_time(ray_index,ray_parameter,t0,p1,t1,range_km)` — `hb_high_ref.f:3610`.
///
/// Clamps the ray parameter to the smallest `1/v` over every segment and both
/// sides of each reflecting interface, then evaluates the travel time there.
/// Returns `(p1, t1)`.
///
/// # Both outputs are discarded by the only caller
///
/// `green_function` passes `p1`/`t1` at `:3313` and never reads them. The call is
/// side-effect-free — `travel_time` writes no common block — so it could be elided
/// entirely. It is kept so the two sources stay line-comparable, and because
/// removing it would be a behaviour-neutral change that still deserves to be
/// recorded rather than assumed. See `PORTING_RULES.md` §7.
///
/// The `t0` argument is likewise never read by the Fortran.
///
/// # Mostly inert under the production ray
///
/// The interface clamp only runs where `it(i) == 1`, i.e. a reflection, which
/// `build_ray_path` sets only when consecutive segments share a layer. The production ray
/// is strictly descending (`nh` running `ksrc` down to 2), so `it` is 0
/// throughout and only the first clamp applies. The branch matters for the
/// Moho-multiple ray shapes.
///
/// Note `nm(ray_index,1)` — the mode of the *first* segment governs whether P
/// velocities are considered, for every segment.
pub fn travel_time(
    state: &RayState,
    vmod: &VelocityModel,
    ray_index: usize,
    ray_parameter: f64,
    _time_guess: f64,
    range_km: f64,
) -> (f64, f64) {
    let n = state.rays.nd as usize;
    let mut p1 = ray_parameter;

    for i in 0..n {
        let nup = state.coff.nup1[i];
        let nhi = state.rays.nh[i] as usize;

        let mut vb = vmod.vsh_km_s[nhi];
        let mut va = vb;
        if state.rays.nm[0] != 4 {
            va = vmod.vp_km_s[nhi];
        }
        p1 = p1.min(1.0 / va).min(1.0 / vb);

        // The last segment has no interface below it.
        if i == n - 1 {
            continue;
        }
        // Transmission needs no second clamp; only reflections do.
        if state.coff.it[i] == 0 {
            continue;
        }

        // Label 10 for upgoing, otherwise the layer below. A layer STEP, so it is the
        // same +-1 in either index base. Upgoing from layer 0 would underflow, as the
        // Fortran read `vs(0)` there; unreachable, and loud if it ever is not.
        let k = if nup == 1 { nhi - 1 } else { nhi + 1 };
        vb = vmod.vsh_km_s[k];
        va = vb;
        if state.rays.nm[0] != 4 {
            va = vmod.vp_km_s[k];
        }
        p1 = p1.min(1.0 / va).min(1.0 / vb);
    }

    let p = Complex64::from(p1);
    let t = cagniard_time(state, vmod, p, ray_index, range_km);
    (p1, t.re)
}

/// Value of a Fortran `DO j = lo, hi` variable after the loop, with step +1.
///
/// Two cases matter and they differ: a loop that runs to completion leaves
/// `hi + 1`, while a loop whose range is empty leaves `lo` untouched.
/// `green_function` reads the loop variable after the loop (`kbot = j`), so getting
/// this wrong silently changes the ray description.
fn do_end(low: usize, high: usize) -> usize {
    if low > high { low } else { high + 1 }
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

/// `subroutine green_function(...)` — `hb_high_ref.f:3174`.
///
/// Builds the ray segment description for a given source depth and ray type,
/// then drives [`build_ray_path`], [`stationary_ray_parameter`], [`travel_time`] and [`geometric_spreading`] to return ray
/// parameter, travel time, path length and path attenuation.
///
/// Sole writer of `/rays/`. `wave_mode` is the wave mode (3 = SV, 4 = SH, 5 = P);
/// production passes 4. `ray_type` odd means an upgoing ray, even means
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
/// past the model. See `PORTING_RULES.wave_mode` §7.
///
/// Note also that if `ksrc < krec` (a source shallower than layer 2) the upgoing
/// segment loop produces zero segments and `nd` is 0, which `build_ray_path` is not
/// written to handle.
///
/// `hs_tol = 0.02` is an unsuffixed literal in an `implicit real*8` routine, so
/// it carries only `f32` precision — see `PORTING_RULES.wave_mode` §1b.
pub fn green_function(
    state: &mut RayState,
    vmod: &VelocityModel,
    layer_count: usize,
    src_depth: f32,
    range: f32,
    ray_type: i32,
    wave_mode: i32,
) -> GreenFunction {
    // The receiver sits in the second layer -- index 1 since §2.3, not 2.
    let krec = 1usize;
    // Deepest layer, 0-based. The Moho loops below run to `bottom_layer - 1` because they
    // test the thickness of the layer BELOW the one they are on, and the bottom layer is
    // forced to zero thickness by `read_velocity_model`.
    let bottom_layer = layer_count - 1;
    let ir = 1usize;

    state.rays.ndeg = 1;
    let hr = vmod.thickness_km[0];
    let mut hs = src_depth as f64;
    let rr = range as f64;

    // Find the source layer, nudging hs off an interface by hs_tol either way
    // so the ray does not start exactly on a boundary.
    let hs_tol = 0.02f32 as f64;
    let mut dep = 0.0f64;
    // Loop-completion value. The Fortran's DO ksrc = 1, layer_count leaves layer_count+1;
    // 0-based that is `layer_count`, one past the last layer. It is READ in that state --
    // see the doc comment -- which is why the arrays stay NLAYMAX-sized.
    let mut ksrc = do_end(0, layer_count - 1);
    for k in 0..layer_count {
        dep += vmod.thickness_km[k];
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

    let mut l = 0usize;
    // 0-based: write at the current count, then advance it. The Fortran pre-increments,
    // so `l` ends at the same segment count either way.
    let push = |state: &mut RayState, l: &mut usize, layer: usize| {
        state.rays.nh[*l] = layer as i32;
        state.rays.nm[*l] = wave_mode;
        *l += 1;
    };

    if ray_type % 2 == 1 {
        // Upgoing: ksrc down to krec.
        let mut j = ksrc as i64;
        while j >= krec as i64 {
            push(state, &mut l, j as usize);
            j -= 1;
        }
        // Moho multiples, if any. ktn is 0 for ray_type == 1.
        let ktn = (ray_type - 1) / 2;
        for _kt in 1..=ktn {
            let mut jv = do_end(krec, bottom_layer - 1);
            for jj in krec..=(bottom_layer - 1) {
                push(state, &mut l, jj);
                if vmod.thickness_km[jj + 1] == 0.0 {
                    jv = jj;
                    break;
                }
            }
            let kbot = jv;
            let mut j = kbot as i64;
            while j >= krec as i64 {
                push(state, &mut l, j as usize);
                j -= 1;
            }
        }
    } else {
        // Down-going to the Moho, then back up.
        let mut jv = do_end(ksrc, bottom_layer - 1);
        for jj in ksrc..=(bottom_layer - 1) {
            push(state, &mut l, jj);
            if vmod.thickness_km[jj + 1] == 0.0 {
                jv = jj;
                break;
            }
        }
        let kbot = jv;
        let mut j = kbot as i64;
        while j >= krec as i64 {
            push(state, &mut l, j as usize);
            j -= 1;
        }

        let ktn = (ray_type - 2) / 2;
        for _kt in 1..=ktn {
            let mut jv = do_end(krec, bottom_layer - 1);
            for jj in krec..=(bottom_layer - 1) {
                push(state, &mut l, jj);
                if vmod.thickness_km[jj + 1] == 0.0 {
                    jv = jj;
                    break;
                }
            }
            let kbot = jv;
            let mut j = kbot as i64;
            while j >= krec as i64 {
                push(state, &mut l, j as usize);
                j -= 1;
            }
        }
    }
    state.rays.nd = l as i32;

    build_ray_path(state, vmod, ir, hs, hr);
    let (p0, t0) = stationary_ray_parameter(state, vmod, ir, rr);
    // Outputs discarded by the Fortran; the call is kept for comparability.
    let (_p1, _t1) = travel_time(state, vmod, ir, p0, t0, rr);

    let (rpd, qbar) = geometric_spreading(state, vmod, hs, p0, ray_type);

    GreenFunction {
        rp0: p0 as f32,
        stime: t0 as f32,
        rpath: rpd as f32,
        qbar,
    }
}
