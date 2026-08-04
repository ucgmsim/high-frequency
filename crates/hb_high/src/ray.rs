//! Ray theory. `cagcon`, `dtdp`, `pnot`, `ttime` and `gf_amp_tt` follow in
//! tiers 2-4.

use crate::fort::Complex64;
use crate::state::{RayState, Vmod};

/// `function cr(p,v)` — `hb_high_ref.f:3349`. Complex vertical slowness
/// `eta = sqrt(1/v^2 - p^2)`, with an explicit branch-cut choice.
///
/// This is the numerically delicate heart of the ray code. It evaluates the
/// square root in polar form rather than algebraically so the branch can be
/// selected deliberately, and the selection at labels 12/13 must be
/// transliterated literally — see `PORTING_RULES.md` §2.
///
/// Two traps in the original worth naming:
///
/// * `pr = p` assigns a `complex*16` to a `real*8`, which silently takes the
///   real part. It is not a typo for `dreal(p)`.
/// * the local named `pi` is `dimag(p)`, the **imaginary part of p**, not
///   3.14159. The actual pi appears separately as the truncated 10-digit
///   literal `3.141592654d0`, which is copied verbatim.
pub fn cr(p: Complex64, v: f64) -> Complex64 {
    let t1 = 1.0e-08f64;
    let rsq = 1.0f64 / (v * v);
    let pr = p.re;
    // `pi` here is Im(p), matching the Fortran's variable name.
    let pi = p.im;
    let mut a = rsq - pr * pr + pi * pi;
    let mut b = -2.0f64 * pi * pr;
    let d = (a * a + b * b).sqrt().sqrt();

    // Near the real axis the phase is forced to 0 or pi rather than taken from
    // atan2, which would be ill-conditioned there.
    let phi = if pi.abs() < t1 {
        if a < 0.0 { 3.141592654f64 } else { 0.0 }
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

/// `subroutine trav(ir,hs,hr)` — `hb_high_ref.f:3507`.
///
/// Builds the per-layer path multipliers for one ray. Sole writer of
/// `/travel/` (`alp`, `als`, `ndeep`, `nup`), `/coff/` (`it`, `nup1`) and
/// `/rmode/` (`love`); reads `/rays/` and `/vmod/thic`.
///
/// `hs` is the source depth, `hr` the receiver depth. `ir` is retained to match
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
pub fn trav(st: &mut RayState, vmod: &Vmod, ir: usize, hs: f64, hr: f64) {
    assert_eq!(ir, 1, "/rays/ has a degenerate leading dimension; ir must be 1");

    st.love = 1;
    if st.rays.nm[1] == 4 {
        st.love = 2;
    }
    let n = st.rays.nd[ir] as usize;

    // DO 10 I=1,100 -- deliberately not 1..=NLAYMAX. See the note above.
    for i in 1..=100 {
        st.travel.alp[i] = 0.0;
        st.travel.als[i] = 0.0;
    }

    // Count how many times each layer is traversed, by wave mode.
    for i in 1..=n {
        let h = st.rays.nh[i] as usize;
        if st.rays.nm[i] == 5 {
            st.travel.alp[h] += 1.0;
        }
        if st.rays.nm[i] == 3 || st.rays.nm[i] == 4 {
            st.travel.als[h] += 1.0;
        }
    }

    // Ray direction from the source: nup = +1 up, -1 down. ndeg < 0 forces
    // upgoing, which resolves the ambiguity when source and receiver share a
    // layer.
    let lis = st.rays.nh[1] as usize;
    let lir = st.rays.nh[n] as usize;
    let mut nl = 1i32;
    for i in 1..=n {
        if st.rays.nh[i] as usize == lis {
            nl += 1;
        }
    }
    let mut nup = (-1i32).pow(nl as u32);
    if lir > lis {
        nup = -nup;
    }
    if st.rays.ndeg[ir] < 0 {
        nup = 1;
    }
    if n == 1 && hr >= hs {
        nup = -1;
    }
    st.travel.nup = nup;

    // Interaction type at each interface and direction of each segment.
    let n1 = n - 1;
    st.coff.nup1[1] = nup;
    if n != 1 {
        for i in 1..=n1 {
            let k = st.rays.nh[i];
            let m = st.rays.nh[i + 1];
            st.coff.it[i] = if m == k { 1 } else { 0 };
            st.coff.nup1[i + 1] = match (st.coff.nup1[i], st.coff.it[i]) {
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
        st.coff.it[1] = 2;
    }

    // Receiver position within its layer.
    let lir1 = lir - 1;
    let mut thtot = 0.0f64;
    for i in 1..=lir1 {
        thtot = vmod.thic[i] + thtot;
    }
    let hrl = hr - thtot;
    let a1 = hrl / vmod.thic[lir];
    let a2 = (vmod.thic[lir] - hrl) / vmod.thic[lir];
    let nupa = st.coff.nup1[n];
    // Labels 23/24: mode 5 takes the P multiplier, modes 3 and 4 the S one,
    // and anything else falls through to P.
    if st.rays.nm[n] == 3 || st.rays.nm[n] == 4 {
        if nupa == 1 {
            st.travel.als[lir] = (st.travel.als[lir] as f64 - a1) as f32;
        }
        if nupa == -1 {
            st.travel.als[lir] = (st.travel.als[lir] as f64 - a2) as f32;
        }
    } else {
        if nupa == 1 {
            st.travel.alp[lir] = (st.travel.alp[lir] as f64 - a1) as f32;
        }
        if nupa == -1 {
            st.travel.alp[lir] = (st.travel.alp[lir] as f64 - a2) as f32;
        }
    }

    // Source position within its layer.
    let lis1 = lis - 1;
    let mut thtot = 0.0f64;
    for i in 1..=lis1 {
        thtot = vmod.thic[i] + thtot;
    }
    let hsl = hs - thtot;
    let a1 = hsl / vmod.thic[lis];
    let a2 = (vmod.thic[lis] - hsl) / vmod.thic[lis];
    // Note the a1/a2 roles are swapped relative to the receiver block above:
    // nup == 1 subtracts a2 here but a1 there. That is what the Fortran does.
    if st.rays.nm[1] == 3 || st.rays.nm[1] == 4 {
        if nup == 1 {
            st.travel.als[lis] = (st.travel.als[lis] as f64 - a2) as f32;
        }
        if nup == -1 {
            st.travel.als[lis] = (st.travel.als[lis] as f64 - a1) as f32;
        }
    } else {
        if nup == 1 {
            st.travel.alp[lis] = (st.travel.alp[lis] as f64 - a2) as f32;
        }
        if nup == -1 {
            st.travel.alp[lis] = (st.travel.alp[lis] as f64 - a1) as f32;
        }
    }

    // Deepest layer the ray penetrates.
    let mut ndeep = 0i32;
    for i in 1..=n {
        ndeep = ndeep.max(st.rays.nh[i]);
    }
    st.travel.ndeep = ndeep;
}

/// `subroutine geom_terms(hs,p0,itype,rp,qb)` — `hb_high_ref.f:3918`.
///
/// Returns `(rp, qb)`: total ray path length in km, and the path-integrated
/// attenuation operator `sum(t_i / Qs_i)`.
///
/// `itype` odd means upgoing, even means downgoing/Moho-reflected — the source
/// comments call this "hardwired to direct and 1 down-going Moho".
///
/// # Precision
///
/// `qb` is `real*4` while every other local is `real*8` under
/// `implicit real*8 (a-h,o-z)`, so **the attenuation sum accumulates in single
/// precision**: each `qb = qb + ti/qs(...)` promotes, adds in double, and
/// narrows straight back. Accumulating in `f64` and narrowing once at the end
/// would be more accurate and would not match.
///
/// Both magic literals are unsuffixed in the Fortran, so they carry only `f32`
/// precision even in this `real*8` routine — see `PORTING_RULES.md` §1b. This
/// is why they are written `0.999999f32 as f64` rather than as plain `f64`
/// literals; the difference shows up around the 30th bit.
pub fn geom_terms(
    st: &RayState,
    vmod: &Vmod,
    hs: f64,
    p0: f64,
    itype: i32,
) -> (f64, f32) {
    let nh1 = st.rays.nh[1] as usize;

    let mut dep = 0.0f64;
    for j in 2..=nh1.saturating_sub(1) {
        dep += vmod.thic[j];
    }

    let m = itype % 2;
    let th1 = if m == 1 {
        hs - dep
    } else if m == 0 {
        dep + vmod.thic[nh1] - hs
    } else {
        // The Fortran has two IFs and no else, so a negative odd itype would
        // leave th1 undefined. Every call site passes itype >= 1.
        panic!("geom_terms: itype {itype} gives mod {m}, leaving th1 undefined");
    };

    let clamp = 0.999999f32 as f64;

    let mut sini = p0 * vmod.vsh[nh1];
    if sini >= 1.0 {
        sini = clamp;
    }
    let denom = 1.0 / (1.0 - sini * sini).sqrt();

    let ri = th1 * denom;
    let ti = ri / vmod.vsh[nh1];

    let mut rsum = ri;
    let mut qb = (ti / vmod.qs[nh1] as f64) as f32;

    for j in 2..=st.rays.nd[1] as usize {
        let nhj = st.rays.nh[j] as usize;
        let mut sini = p0 * vmod.vsh[nhj];
        if sini >= 1.0 {
            sini = clamp;
        }
        let denom = 1.0 / (1.0 - sini * sini).sqrt();

        let ri = vmod.thic[nhj] * denom;
        let ti = ri / vmod.vsh[nhj];

        rsum += ri;
        // Narrowed on every iteration: single-precision accumulation.
        qb = (qb as f64 + ti / vmod.qs[nhj] as f64) as f32;
    }

    if rsum == 0.0 {
        rsum = 0.001f32 as f64;
    }
    (rsum, qb)
}
