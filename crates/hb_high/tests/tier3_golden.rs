//! Bit-identity gate for `stationary_ray_parameter` and `travel_time`.
//!
//! The driver runs the real `build_ray_path` first and dumps the resulting `/travel/` and
//! `/coff/` state, so these tests load that state directly rather than
//! re-deriving it. That keeps each kernel's gate independent — `build_ray_path` has its
//! own in `tier1_golden.rs`.
//!
//! Regenerate with `harness/kernels/gen_tier3_golden.sh`.
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.

use hb_high::ray::{stationary_ray_parameter, travel_time};

mod common;
use common::*;

#[test]
fn pnot_matches_fortran() {
    let mut r = Golden::open("tier3", "pnot.bin");
    let mut n = 0;
    // stationary_ray_parameter appears to have two paths: return immediately at the branch cut when
    // dtau/dp >= 0 there, or bisect down towards zero. Measured across this
    // corpus, the immediate path NEVER fires, and it looks structurally
    // unreachable rather than merely uncovered:
    //
    //   v is the largest velocity among layers with a POSITIVE multiplier, and
    //   cagniard_time_derivative sums th(i)*mult(i)/eta(i) over every layer with a NONZERO
    //   multiplier. So the layer that defines v is necessarily in cagniard_time_derivative's sum,
    //   and as p approaches 1/v that layer's eta approaches 0, making its term
    //   diverge. Hence a = r - p*sum is large and negative at the starting
    //   point, for any finite r.
    //
    // Empirically the largest `a` over 72 cases spanning r from 0.5 to 400 km
    // is -2106, nowhere near zero. The assertion below therefore pins the
    // *unreachability*: if a future change makes that path fire, this test says
    // so rather than silently leaving it unverified.
    let mut immediate = 0;
    let mut bisected = 0;

    while !r.done() {
        let ndeep = r.usize();
        let rr = r.f64();
        let (st, vmod) = r.ray_seam_state(ndeep);
        let want_p0 = r.f64();
        let want_t0 = r.f64();

        let (p0, t0) = stationary_ray_parameter(&st, &vmod, rr);
        let tag = format!("stationary_ray_parameter case {n} (ndeep={ndeep} r={rr})");
        // p0 pins the whole search: the eps growth loop, the 0.01 tolerance and
        // the 40-iteration cap all feed into it.
        eq64(&format!("{tag} p0"), p0, want_p0);
        eq64(&format!("{tag} t0"), t0, want_t0);

        // Reconstruct the starting point to classify which path ran.
        let mut v = 0.0f64;
        for i in 0..ndeep {
            if st.travel.alp[i] > 0.0 {
                v = v.max(vmod[i].vp_km_s);
            }
            if st.travel.als[i] > 0.0 {
                v = v.max(vmod[i].vsh_km_s);
            }
        }
        let mut eps = 1.0e-10f64;
        let ptest = 1.0 / v;
        while (ptest - eps) * v >= 1.0 {
            eps *= 10.0;
        }
        if p0 == ptest - 10.0 * eps {
            immediate += 1;
        } else {
            bisected += 1;
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 72);
    assert!(bisected > 0, "no case entered the bisection; that path is untested");
    assert_eq!(
        immediate, 0,
        "a case returned at the branch cut. That path was believed unreachable \
         (see the note above); if it is now reachable the reasoning needs \
         revisiting and the path needs its own coverage."
    );
    // Not asserted, but recorded: every case exits the bisection on the
    // |a| <= 0.01 tolerance. The 40-iteration cap never fires here, so it
    // remains an uncovered safety net.
}

#[test]
fn ttime_matches_fortran() {
    let mut r = Golden::open("tier3", "ttime.bin");
    let mut n = 0;
    while !r.done() {
        let ndeep = r.usize();
        let nseg = r.usize();
        let p0 = r.f64();
        let rr = r.f64();
        let (mut st, vmod) = r.ray_seam_state(ndeep);

        for k in 0..nseg { st.rays.nh[k] = r.i32() - 1; }
        for k in 0..nseg { st.rays.nm[k] = r.i32(); }
        for k in 0..nseg { st.coff.it[k] = r.i32(); }
        for k in 0..nseg { st.coff.nup1[k] = r.i32(); }
        st.rays.nd = nseg as i32;

        let want_p1 = r.f64();
        let want_t1 = r.f64();

        // t0 is passed and never read by the Fortran; 0.0 stands in.
        let (p1, t1) = travel_time(&st, &vmod, p0, 0.0, rr);
        let tag = format!("travel_time case {n} (ndeep={ndeep} n={nseg})");
        eq64(&format!("{tag} p1"), p1, want_p1);
        eq64(&format!("{tag} t1"), t1, want_t1);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 72);
}
