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
use hb_high::state::{RayState, VelocityModel};
use std::path::PathBuf;

struct Reader {
    buf: Vec<u8>,
    pos: usize,
    name: String,
}

impl Reader {
    fn open(name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../harness/golden/tier3")
            .join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("reading {}: {e}. Run harness/kernels/gen_tier3_golden.sh", path.display())
        });
        Self { buf, pos: 0, name: name.to_string() }
    }
    fn take<const N: usize>(&mut self) -> [u8; N] {
        assert!(self.pos + N <= self.buf.len(),
                "{}: ran off the end at byte {} of {}", self.name, self.pos, self.buf.len());
        let out = self.buf[self.pos..self.pos + N].try_into().unwrap();
        self.pos += N;
        out
    }
    fn f64(&mut self) -> f64 { f64::from_le_bytes(self.take::<8>()) }
    fn f32(&mut self) -> f32 { f32::from_le_bytes(self.take::<4>()) }
    fn i32(&mut self) -> i32 { i32::from_le_bytes(self.take::<4>()) }
    fn usize(&mut self) -> usize { self.i32() as usize }
    fn done(&self) -> bool { self.pos >= self.buf.len() }
    fn assert_exhausted(&self) {
        assert_eq!(self.pos, self.buf.len(),
                   "{}: consumed {} of {} bytes; record layout disagrees with the driver",
                   self.name, self.pos, self.buf.len());
    }

    /// `dump_state`: th, vp, vs (`f64`) then alp, als (`f32`), for `1..=ndeep`.
    fn state(&mut self, ndeep: usize) -> (RayState, VelocityModel) {
        let mut vmod = VelocityModel::new();
        for k in 1..=ndeep { vmod.thic[k] = self.f64(); }
        for k in 1..=ndeep { vmod.vp[k] = self.f64(); }
        for k in 1..=ndeep { vmod.vsh[k] = self.f64(); }
        let mut st = RayState::default();
        for k in 1..=ndeep { st.travel.alp[k] = self.f32(); }
        for k in 1..=ndeep { st.travel.als[k] = self.f32(); }
        st.travel.ndeep = ndeep as i32;
        (st, vmod)
    }
}

#[track_caller]
fn eq64(what: &str, got: f64, want: f64) {
    assert_eq!(got.to_bits(), want.to_bits(),
               "{what}: rust {got:?} (0x{:016x}) vs fortran {want:?} (0x{:016x})",
               got.to_bits(), want.to_bits());
}

#[test]
fn pnot_matches_fortran() {
    let mut r = Reader::open("pnot.bin");
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
        let (st, vmod) = r.state(ndeep);
        let want_p0 = r.f64();
        let want_t0 = r.f64();

        let (p0, t0) = stationary_ray_parameter(&st, &vmod, 1, rr);
        let tag = format!("stationary_ray_parameter case {n} (ndeep={ndeep} r={rr})");
        // p0 pins the whole search: the eps growth loop, the 0.01 tolerance and
        // the 40-iteration cap all feed into it.
        eq64(&format!("{tag} p0"), p0, want_p0);
        eq64(&format!("{tag} t0"), t0, want_t0);

        // Reconstruct the starting point to classify which path ran.
        let mut v = 0.0f64;
        for i in 1..=ndeep {
            if st.travel.alp[i] > 0.0 {
                v = v.max(vmod.vp[i]);
            }
            if st.travel.als[i] > 0.0 {
                v = v.max(vmod.vsh[i]);
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
    let mut r = Reader::open("ttime.bin");
    let mut n = 0;
    while !r.done() {
        let ndeep = r.usize();
        let nseg = r.usize();
        let p0 = r.f64();
        let rr = r.f64();
        let (mut st, vmod) = r.state(ndeep);

        for k in 1..=nseg { st.rays.nh[k] = r.i32(); }
        for k in 1..=nseg { st.rays.nm[k] = r.i32(); }
        for k in 1..=nseg { st.coff.it[k] = r.i32(); }
        for k in 1..=nseg { st.coff.nup1[k] = r.i32(); }
        st.rays.nd[1] = nseg as i32;

        let want_p1 = r.f64();
        let want_t1 = r.f64();

        // t0 is passed and never read by the Fortran; 0.0 stands in.
        let (p1, t1) = travel_time(&st, &vmod, 1, p0, 0.0, rr);
        let tag = format!("travel_time case {n} (ndeep={ndeep} n={nseg})");
        eq64(&format!("{tag} p1"), p1, want_p1);
        eq64(&format!("{tag} t1"), t1, want_t1);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 72);
}
