//! Bit-identity gate for `stochastic_spectrum` and `green_function`.
//!
//! Regenerate with `harness/kernels/gen_tier4_golden.sh`.
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.

use hb_high::fort::Complex32;
use hb_high::ray::green_function;
use hb_high::rng::Pcg32;
use hb_high::state::{RayState, VelocityModel};
use hb_high::stoc::stochastic_spectrum;
use std::path::PathBuf;

struct Reader {
    buf: Vec<u8>,
    pos: usize,
    name: String,
}

impl Reader {
    fn open(name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../harness/golden/tier4")
            .join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("reading {}: {e}. Run harness/kernels/gen_tier4_golden.sh", path.display())
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
    fn f32(&mut self) -> f32 { f32::from_le_bytes(self.take::<4>()) }
    fn f64(&mut self) -> f64 { f64::from_le_bytes(self.take::<8>()) }
    fn i32(&mut self) -> i32 { i32::from_le_bytes(self.take::<4>()) }
    fn usize(&mut self) -> usize { self.i32() as usize }
    fn done(&self) -> bool { self.pos >= self.buf.len() }
    fn assert_exhausted(&self) {
        assert_eq!(self.pos, self.buf.len(),
                   "{}: consumed {} of {} bytes; record layout disagrees with the driver",
                   self.name, self.pos, self.buf.len());
    }
}

#[track_caller]
fn eq32(what: &str, got: f32, want: f32) {
    assert_eq!(got.to_bits(), want.to_bits(),
               "{what}: rust {got:?} (0x{:08x}) vs fortran {want:?} (0x{:08x})",
               got.to_bits(), want.to_bits());
}

/// Relative comparison against a per-record scale, for the values that pass through
/// the transform.
///
/// `REFACTOR.md` §2.1 replaced the vendored radix-2 kernel with `rustfft`, which sums
/// the butterflies in a different order. The physics either side of the transform is
/// unchanged and still worth checking against the Fortran, so these comparisons are
/// loosened rather than deleted — but they can no longer be exact.
///
/// `1e-4` of the record's peak. The measured whole-program deviation from the swap is
/// ~1e-6 of peak, so this is 100x headroom against rounding while still catching
/// anything structural: a wrong scale factor, a dropped taper, a mirrored half.
fn near32(what: &str, got: f32, want: f32, scale: f32) {
    let tol = 1e-4 * scale.max(f32::MIN_POSITIVE);
    assert!(
        (got - want).abs() <= tol,
        "{what}: rust {got:?} vs fortran {want:?} (delta {:.3e}, tolerance {tol:.3e})",
        (got - want).abs()
    );
}

#[test]
fn stoc_f_matches_fortran() {
    let mut r = Reader::open("stoc_f.bin");
    let mut cases = 0;
    let mut saw_negative_kappa = false;

    while !r.done() {
        let np2 = r.usize();
        let seed = r.i32();
        let nf = np2 / 2 + 1;

        let (rr, tw, eps, eta, betvs) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (row, dt, smt, dlm, fc) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (fmx, akapp, qb, qfe, bigc) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        if akapp <= 0.0 {
            saw_negative_kappa = true;
        }

        let dfr: Vec<f32> = (0..nf).map(|_| r.f32()).collect();
        let want: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want_after: Vec<f32> = (0..8).map(|_| r.f32()).collect();

        let (mut rng, _) = Pcg32::seed(seed);
        let mut cw = vec![Complex32::ZERO; np2];
        stochastic_spectrum(&mut rng, np2, rr, tw, eps, eta, betvs, row, dt, smt, dlm,
               fc, fmx, akapp, cw.as_mut_slice(), dfr.as_slice(), qb, qfe, bigc);

        let tag = format!("stochastic_spectrum case {cases} (np2={np2} akapp={akapp})");
        // Scale from the Fortran record, so the tolerance does not float with our
        // own output.
        let scale = want.iter().fold(0.0f32, |a, c| a.max(c.re.abs()).max(c.im.abs()));
        for i in 0..np2 {
            let at = i + 1;
            near32(&format!("{tag} cw[{at}].re"), cw[i].re, want[i].re, scale);
            near32(&format!("{tag} cw[{at}].im"), cw[i].im, want[i].im, scale);
        }
        // Generator position: stochastic_spectrum consumes np2 deviates via
        // normal_deviates, and the shared stream must stay in step.
        for (k, w) in want_after.iter().enumerate() {
            eq32(&format!("{tag} post-call draw {k} (generator position)"),
                 rng.next_f32(), *w);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
    assert!(saw_negative_kappa,
            "no case had akapp <= 0, so the alternative high-cut branch \
             ((1+omg/omgm)**(-1.0) rather than exp(-pi*f*kappa)) is untested");
}

#[test]
fn gf_amp_tt_matches_fortran() {
    let mut r = Reader::open("gf_amp_tt.bin");
    let mut cases = 0;
    // Which ray topologies the corpus actually reached.
    let mut itypes = std::collections::BTreeSet::new();
    let mut mds = std::collections::BTreeSet::new();

    while !r.done() {
        let j0 = r.usize();
        let itype = r.i32();
        let md = r.i32();
        itypes.insert(itype);
        mds.insert(md);
        let src_depth = r.f32();
        let range = r.f32();

        let mut vmod = VelocityModel::new();
        for k in 0..j0 { vmod.thickness_km[k] = r.f64(); }
        for k in 0..j0 { vmod.vp_km_s[k] = r.f64(); }
        for k in 0..j0 { vmod.vsh_km_s[k] = r.f64(); }
        for k in 0..j0 { vmod.attenuation_s[k] = r.f32(); }

        let want_nd = r.usize();
        let want_nh: Vec<i32> = (0..want_nd).map(|_| r.i32()).collect();
        let want_nm: Vec<i32> = (0..want_nd).map(|_| r.i32()).collect();
        let want_ndeep = r.i32();
        let want_love = r.i32();
        let (w_rp0, w_stime, w_rpath, w_qbar) = (r.f32(), r.f32(), r.f32(), r.f32());

        let mut st = RayState::default();
        let g = green_function(&mut st, &vmod, j0, src_depth, range, itype, md);

        let tag = format!("green_function case {cases} (itype={itype} md={md} \
                           depth={src_depth} range={range})");
        // The ray description itself, before the derived quantities: a wrong
        // segment list would otherwise only show up as a wrong travel time.
        assert_eq!(st.rays.nd as usize, want_nd, "{tag} nd");
        for k in 0..want_nd {
            // 0-based layer index against the golden's 1-based layer number.
            assert_eq!(st.rays.nh[k] + 1, want_nh[k], "{tag} nh[{k}]");
            assert_eq!(st.rays.nm[k], want_nm[k], "{tag} nm[{k}]");
        }
        assert_eq!(st.travel.ndeep + 1, want_ndeep, "{tag} ndeep");
        assert_eq!(st.love, want_love, "{tag} love");

        eq32(&format!("{tag} rp0"), g.rp0, w_rp0);
        eq32(&format!("{tag} stime"), g.stime, w_stime);
        eq32(&format!("{tag} rpath"), g.rpath, w_rpath);
        eq32(&format!("{tag} qbar"), g.qbar, w_qbar);
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 48);
    // Production only ever passes itype=1 and md=4; the corpus deliberately
    // covers the down-going and Moho-multiple topologies too.
    assert!(itypes.contains(&1) && itypes.contains(&2),
            "need both upgoing and down-going ray types, saw {itypes:?}");
    assert!(itypes.iter().any(|&t| t > 2),
            "no itype > 2, so the Moho-multiple loops are untested; saw {itypes:?}");
    assert!(mds.len() >= 2, "only one wave mode exercised: {mds:?}");
}
