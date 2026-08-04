//! Bit-identity gate for the tier-1 kernels.
//!
//! These routines communicate through common blocks, so each golden record
//! carries the input state, the arguments, **and** the emitted state — the
//! golden is a specification of the whole seam, not just of return values.
//!
//! Regenerate with `harness/kernels/gen_tier1_golden.sh`.
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.

use hb_high::fort::Array1;
use hb_high::geom::subfault_geometry;
use hb_high::ray::{geometric_spreading, build_ray_path};
use hb_high::site::site_amplification_factors;
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
            .join("../../harness/golden/tier1")
            .join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("reading {}: {e}. Run harness/kernels/gen_tier1_golden.sh", path.display())
        });
        Self { buf, pos: 0, name: name.to_string() }
    }

    fn take<const N: usize>(&mut self) -> [u8; N] {
        assert!(
            self.pos + N <= self.buf.len(),
            "{}: ran off the end at byte {} of {}",
            self.name, self.pos, self.buf.len()
        );
        let out = self.buf[self.pos..self.pos + N].try_into().unwrap();
        self.pos += N;
        out
    }

    fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take::<4>())
    }
    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take::<8>())
    }
    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take::<4>())
    }
    fn usize(&mut self) -> usize {
        self.i32() as usize
    }
    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }
    fn assert_exhausted(&self) {
        assert_eq!(
            self.pos, self.buf.len(),
            "{}: consumed {} of {} bytes; record layout disagrees with the driver",
            self.name, self.pos, self.buf.len()
        );
    }

    /// Read `th`, `vsh_km_s`, `density_g_cm3` for layers `1..=j0` into a fresh `VelocityModel`,
    /// matching the driver's `dump_vmod`.
    fn vmod(&mut self, j0: usize) -> VelocityModel {
        let mut v = VelocityModel::new();
        for k in 1..=j0 {
            v.thickness_km[k] = self.f64();
        }
        for k in 1..=j0 {
            v.vsh_km_s[k] = self.f64();
        }
        for k in 1..=j0 {
            v.density_g_cm3[k] = self.f64();
        }
        v
    }
}

#[track_caller]
fn eq32(what: &str, got: f32, want: f32) {
    assert_eq!(
        got.to_bits(), want.to_bits(),
        "{what}: rust {got:?} (0x{:08x}) vs fortran {want:?} (0x{:08x})",
        got.to_bits(), want.to_bits()
    );
}

#[track_caller]
fn eq64(what: &str, got: f64, want: f64) {
    assert_eq!(
        got.to_bits(), want.to_bits(),
        "{what}: rust {got:?} (0x{:016x}) vs fortran {want:?} (0x{:016x})",
        got.to_bits(), want.to_bits()
    );
}

#[test]
fn get_sitefacs_matches_fortran() {
    let mut r = Reader::open("get_sitefacs.bin");
    let mut cases = 0;
    while !r.done() {
        let j0 = r.usize();
        let nfreq = r.usize();
        let vmod = r.vmod(j0);
        let mut fn_ = Array1::<f32>::new(nfreq);
        for k in 1..=nfreq {
            fn_[k] = r.f32();
        }
        let want: Vec<f32> = (0..nfreq).map(|_| r.f32()).collect();

        let mut an = Array1::<f32>::new(nfreq);
        site_amplification_factors(&vmod, j0, nfreq, fn_.as_slice(), an.as_mut_slice());
        for k in 1..=nfreq {
            eq32(&format!("site_amplification_factors j0={j0} an[{k}]"), an[k], want[k - 1]);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}

#[test]
fn trav_matches_fortran() {
    let mut r = Reader::open("trav.bin");

    // ONE state across every case, mirroring the Fortran's persistent common
    // block. This is deliberate: `build_ray_path` zeroes only alp(1:100) of 500, so
    // whether higher indices carry values from a previous ray is part of the
    // behaviour under test. A fresh state per case would not exercise it.
    let mut st = RayState::default();

    let mut cases = 0;
    while !r.done() {
        let j0 = r.usize();
        let n = r.usize();
        let ir = r.usize();
        let ndeg = r.i32();
        let hs = r.f64();
        let hr = r.f64();
        let vmod = r.vmod(j0);

        for k in 1..=n {
            st.rays.nh[k] = r.i32();
        }
        for k in 1..=n {
            st.rays.nm[k] = r.i32();
        }
        st.rays.nd = n as i32;
        st.rays.ndeg = ndeg;

        let (w_love, w_nup, w_ndeep) = (r.i32(), r.i32(), r.i32());
        let w_it: Vec<i32> = (0..n).map(|_| r.i32()).collect();
        let w_nup1: Vec<i32> = (0..n).map(|_| r.i32()).collect();
        let w_alp: Vec<f32> = (0..j0).map(|_| r.f32()).collect();
        let w_als: Vec<f32> = (0..j0).map(|_| r.f32()).collect();

        build_ray_path(&mut st, &vmod, ir, hs, hr);

        let tag = format!("build_ray_path case {cases} (j0={j0} n={n} ndeg={ndeg})");
        assert_eq!(st.love, w_love, "{tag} love");
        assert_eq!(st.travel.nup, w_nup, "{tag} nup");
        assert_eq!(st.travel.ndeep, w_ndeep, "{tag} ndeep");
        for k in 1..=n {
            assert_eq!(st.coff.it[k], w_it[k - 1], "{tag} it[{k}]");
            assert_eq!(st.coff.nup1[k], w_nup1[k - 1], "{tag} nup1[{k}]");
        }
        for k in 1..=j0 {
            eq32(&format!("{tag} alp[{k}]"), st.travel.alp[k], w_alp[k - 1]);
            eq32(&format!("{tag} als[{k}]"), st.travel.als[k], w_als[k - 1]);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 8);
}

#[test]
fn geom_terms_matches_fortran() {
    let mut r = Reader::open("geom_terms.bin");
    let mut cases = 0;
    while !r.done() {
        let j0 = r.usize();
        let n = r.usize();
        let itype = r.i32();
        let hs = r.f64();
        let p0 = r.f64();

        let mut vmod = VelocityModel::new();
        for k in 1..=j0 {
            vmod.thickness_km[k] = r.f64();
        }
        for k in 1..=j0 {
            vmod.vsh_km_s[k] = r.f64();
        }
        for k in 1..=j0 {
            vmod.attenuation_s[k] = r.f32();
        }

        let mut st = RayState::default();
        for k in 1..=n {
            st.rays.nh[k] = r.i32();
        }
        st.rays.nd = n as i32;

        let w_rp = r.f64();
        let w_qb = r.f32();

        let (rp, qb) = geometric_spreading(&st, &vmod, hs, p0, itype);
        let tag = format!("geometric_spreading case {cases} (itype={itype} p0={p0})");
        eq64(&format!("{tag} rp"), rp, w_rp);
        // The single-precision accumulation of qb is exactly what this pins.
        eq32(&format!("{tag} qb"), qb, w_qb);
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn even_dist2_matches_fortran() {
    let mut r = Reader::open("even_dist2.bin");
    let mut cases = 0;
    while !r.done() {
        let nx = r.usize();
        let nw = r.usize();
        let (xlonq, ylatq, slon, slat) = (r.f32(), r.f32(), r.f32(), r.f32());
        let (azmq, dipangq, zm, astop) = (r.f32(), r.f32(), r.f32(), r.f32());
        let (dx, dy) = (r.f32(), r.f32());

        let g = subfault_geometry(
            xlonq, ylatq, slon, slat, azmq, dipangq, zm, astop, dx, dy, nx, nw,
        );

        // Driver dump order: ((dst,rl,th,ph,zet), j=1,nw), i=1,nx)
        for i in 1..=nx {
            for j in 1..=nw {
                let tag = format!("subfault_geometry case {cases} ({i},{j})");
                let ray = g.at(i, j);
                eq32(&format!("{tag} dst"), ray.horiz_km, r.f32());
                eq32(&format!("{tag} rl"), ray.slant_km, r.f32());
                eq32(&format!("{tag} th"), ray.takeoff_rad, r.f32());
                eq32(&format!("{tag} ph"), ray.azimuth_rad, r.f32());
                eq32(&format!("{tag} zet"), ray.depth_km, r.f32());
            }
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 5);
}
