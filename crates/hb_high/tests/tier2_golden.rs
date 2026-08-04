//! Bit-identity gate for the tier-2 kernels.
//!
//! Regenerate with `harness/kernels/gen_tier2_golden.sh`.
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.

use hb_high::fort::{Complex32, Complex64};
use hb_high::highcor::apply_radiation_and_invert;
use hb_high::radiation::{horizontal_radiation_spectrum, vertical_radiation_spectrum};
use hb_high::ray::{cagniard_time, cagniard_time_derivative};
use hb_high::rng::Pcg32;
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
            .join("../../harness/golden/tier2")
            .join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("reading {}: {e}. Run harness/kernels/gen_tier2_golden.sh", path.display())
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

// `eq64` is gone: both `f64` goldens in this tier now go through `near64`, since the pi
// correction in `vertical_slowness` reaches every value they compare. The `f32` goldens
// below are untouched by it and are still exact.

/// Relative comparison for the values that no longer match the oracle bit for bit.
///
/// Two independent reasons, both understood and both bounded:
///
/// * **Complex division.** `REFACTOR.md` §2.2 replaced the hand-written complex
///   arithmetic with `num-complex`. Its `norm`, `exp` and `mul` are bit-identical to the
///   gfortran forms; its **division** is not — it does not use gfortran's
///   Smith-with-range-reduction branch, and differs by 1-2 ulps. Only
///   `cagniard_time_derivative` reaches a complex/complex division.
/// * **Pi.** `vertical_slowness` used to force the branch-cut phase to the Fortran's
///   truncated `3.141592654d0` and now uses `std::f64::consts::PI`, which moves the real
///   part of `eta` by up to 2.05e-10 of its magnitude on branch-cut cases — see
///   `tier0_golden::cr_stays_close_to_fortran`. Both `cagniard_time` and
///   `cagniard_time_derivative` sum `eta` over layers, so both inherit it.
///
/// The pi term dominates — ~1e-16 for the division ulps against these measured worsts:
///
/// ```text
/// cagcon  1.082e-10   (cagniard_time, a plain sum of eta over layers)
/// dtdp    7.012e-10   (cagniard_time_derivative, which divides BY eta)
/// ```
///
/// `eta`'s own worst is 2.05e-10 (`tier0_golden::cr_stays_close_to_fortran`). `cagcon`
/// stays under that, as a sum should. `dtdp` amplifies it ~3.4x, which is expected
/// rather than alarming: it divides by `eta`, and near the branch cut `eta` is small, so
/// a fixed relative perturbation upstream is magnified by however close that case sits
/// to the cut. That amplification is data-dependent, so the bound needs real headroom
/// over the observed worst rather than a snug fit to it.
///
/// `5e-9` is ~7x the measured worst, and still four orders tighter than anything that
/// could hide a wrong branch, a swapped operand or a lost term.
fn near64(what: &str, got: f64, want: f64) {
    let tol = 5e-9 * want.abs().max(f64::MIN_POSITIVE);
    assert!(
        (got - want).abs() <= tol,
        "{what}: rust {got:?} vs fortran {want:?} (delta {:.3e}, tolerance {tol:.3e})",
        (got - want).abs()
    );
}

/// Track the worst relative divergence across a fixture, so the number that justifies
/// [`near64`]'s tolerance stays measured rather than remembered.
#[derive(Default)]
struct Divergence {
    worst: f64,
    at: String,
}

impl Divergence {
    fn note(&mut self, what: &str, got: f64, want: f64) {
        let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        if rel > self.worst {
            self.worst = rel;
            self.at = what.to_string();
        }
    }

    fn report(&self, fixture: &str) {
        println!("{fixture}: worst relative divergence {:.3e} at {}", self.worst, self.at);
    }
}

/// Shared record layout for the `cagniard_time`/`cagniard_time_derivative` seam.
fn read_ray_seam(r: &mut Reader) -> (RayState, VelocityModel, Complex64, f64, usize) {
    let ndp = r.usize();
    let p = Complex64::new(r.f64(), r.f64());
    let rr = r.f64();
    let mut vmod = VelocityModel::new();
    for k in 0..ndp { vmod.thickness_km[k] = r.f64(); }
    for k in 0..ndp { vmod.vp_km_s[k] = r.f64(); }
    for k in 0..ndp { vmod.vsh_km_s[k] = r.f64(); }
    let mut st = RayState::default();
    for k in 0..ndp { st.travel.alp[k] = r.f32(); }
    for k in 0..ndp { st.travel.als[k] = r.f32(); }
    // The golden's `ndp` is the Fortran's deepest LAYER NUMBER, which doubles as a count
    // of layers from 1. `ndeep` is a 0-based index since §2.3, so it is one lower.
    st.travel.ndeep = ndp as i32 - 1;
    (st, vmod, p, rr, ndp)
}

/// Was exact; now bounded. `cagniard_time` sums `vertical_slowness` over layers, so it
/// inherits the pi correction described on [`near64`].
#[test]
fn cagcon_stays_close_to_fortran() {
    let mut r = Reader::open("cagcon.bin");
    let mut n = 0;
    let mut div = Divergence::default();
    while !r.done() {
        let (st, vmod, p, rr, ndp) = read_ray_seam(&mut r);
        let want = Complex64::new(r.f64(), r.f64());
        let got = cagniard_time(&st, &vmod, p, rr);
        let re = format!("cagniard_time case {n} (ndeep={ndp}) re");
        let im = format!("cagniard_time case {n} (ndeep={ndp}) im");
        div.note(&re, got.re, want.re);
        div.note(&im, got.im, want.im);
        near64(&re, got.re, want.re);
        near64(&im, got.im, want.im);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 200);
    div.report("cagcon");
}

#[test]
fn dtdp_stays_close_to_fortran() {
    let mut r = Reader::open("dtdp.bin");
    let mut n = 0;
    let mut div = Divergence::default();
    while !r.done() {
        let (st, vmod, p, rr, ndp) = read_ray_seam(&mut r);
        let want = Complex64::new(r.f64(), r.f64());
        // Exercises complex division: Smith's algorithm, not (ac+bd)/(c^2+d^2).
        let got = cagniard_time_derivative(&st, &vmod, p, rr);
        let re = format!("cagniard_time_derivative case {n} (ndeep={ndp}) re");
        let im = format!("cagniard_time_derivative case {n} (ndeep={ndp}) im");
        div.note(&re, got.re, want.re);
        div.note(&im, got.im, want.im);
        near64(&re, got.re, want.re);
        near64(&im, got.im, want.im);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 200);
    div.report("dtdp");
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
fn highcor_f_matches_fortran() {
    let mut r = Reader::open("highcor_f.bin");
    let mut cases = 0;
    while !r.done() {
        let nf = r.usize();
        let mf = r.usize();
        let np2 = r.usize();

        // 0-based since §2.3; labels keep the Fortran's 1-based index.
        let rdna: Vec<f32> = (0..nf).map(|_| r.f32()).collect();
        let mut cw1: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want_cw: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want_stdd: Vec<f32> = (0..np2).map(|_| r.f32()).collect();

        let mut stdd = vec![0.0; np2];
        apply_radiation_and_invert(nf, mf, cw1.as_mut_slice(), stdd.as_mut_slice(), rdna.as_slice());

        let cw_scale = want_cw.iter().fold(0.0f32, |a, c| a.max(c.re.abs()).max(c.im.abs()));
        let stdd_scale = want_stdd.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        for i in 0..np2 {
            let at = i + 1;
            near32(&format!("apply_radiation_and_invert np2={np2} cw1[{at}].re"), cw1[i].re, want_cw[i].re, cw_scale);
            near32(&format!("apply_radiation_and_invert np2={np2} cw1[{at}].im"), cw1[i].im, want_cw[i].im, cw_scale);
            near32(&format!("apply_radiation_and_invert np2={np2} stdd[{at}]"), stdd[i], want_stdd[i], stdd_scale);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}

#[test]
fn radfrq_lin_matches_fortran() {
    let mut r = Reader::open("radfrq_lin.bin");
    let mut cases = 0;
    while !r.done() {
        let (stra, dipa, raka) = (r.f32(), r.f32(), r.f32());
        let (pa, thaa, cmp) = (r.f32(), r.f32(), r.f32());
        let nfold = r.usize();
        let nr = r.usize();
        let seed = r.i32();

        let dfr: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();
        let want_fr1 = r.f32();
        let want_rdna: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();
        let want_after: Vec<f32> = (0..8).map(|_| r.f32()).collect();

        let (mut rng, _) = Pcg32::seed(seed);
        let mut rdna = vec![0.0; nfold];
        let fr1 = horizontal_radiation_spectrum(&mut rng, stra, dipa, raka, pa, thaa, dfr.as_slice(), nfold, cmp, nr, rdna.as_mut_slice());

        let tag = format!("horizontal_radiation_spectrum case {cases} (cmp={cmp})");
        eq32(&format!("{tag} fr1 (clobbered)"), fr1, want_fr1);
        for i in 0..nfold {
            eq32(&format!("{tag} rdna[{}]", i + 1), rdna[i], want_rdna[i]);
        }
        // The stream position after the call. This is what catches a version
        // that gets the pattern right while drawing the wrong number of
        // deviates -- 5 per iteration, in the order th, fa, strX, dipX, rakX.
        for (k, w) in want_after.iter().enumerate() {
            eq32(&format!("{tag} post-call draw {k} (generator position)"),
                 rng.next_f32(), *w);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn radv_lin_matches_fortran() {
    let mut r = Reader::open("radv_lin.bin");
    let mut cases = 0;
    while !r.done() {
        let (stra, dipa, raka, pa, thaa) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let nfold = r.usize();
        let nr = r.usize();

        let dfr: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();
        let rna: Vec<f32> = (0..nr).map(|_| r.f32()).collect();
        let rnb: Vec<f32> = (0..nr).map(|_| r.f32()).collect();
        let want_fr1 = r.f32();
        let want_rdna: Vec<f32> = (0..nfold).map(|_| r.f32()).collect();

        let mut rdna = vec![0.0; nfold];
        let fr1 = vertical_radiation_spectrum(stra, dipa, raka, pa, thaa, dfr.as_slice(), nfold, rna.as_slice(), rnb.as_slice(), nr, rdna.as_mut_slice());

        let tag = format!("vertical_radiation_spectrum case {cases}");
        eq32(&format!("{tag} fr1 (clobbered to 0.001)"), fr1, want_fr1);
        for i in 0..nfold {
            eq32(&format!("{tag} rdna[{}]", i + 1), rdna[i], want_rdna[i]);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}
