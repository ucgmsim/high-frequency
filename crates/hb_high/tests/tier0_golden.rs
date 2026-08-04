//! Bit-identity gate for the tier-0 leaf kernels.
//!
//! Goldens are produced by `harness/kernels/tier0_driver.f` linked against
//! `reference/hb_high_subs.f`, so they come from the same Fortran the oracle
//! binary runs. Each record holds the **inputs as well as the outputs**, and
//! these tests read the inputs from the file rather than regenerating them — a
//! kernel test must not be able to pass by comparing two different input sets.
//!
//! Regenerate with `harness/kernels/gen_tier0_golden.sh`.
//!
//!
//! **Fixture filenames are the FORTRAN routine names**, not this port's. They are
//! written by the Fortran driver, which dumps one file per subprogram it exercises,
//! so `cr.bin` holds the golden for what is now `ray::vertical_slowness`. Renaming
//! them would mean editing the drivers and regenerating every golden, and the names
//! are useful provenance where they are. See `REFACTOR.md` §1.4b.
//!
//! **`fast` no longer has a golden here either.** §2.1 replaced the radix-2 kernel
//! with `rustfft`, and unlike `stoc_f` and `highcor_f` — where the transform is one
//! step of five and the surrounding physics is still worth checking — this test's
//! entire content was "does our FFT match the Fortran's FFT". That question is now
//! answered by `tests/properties.rs`: round-trip proportionality, linearity, and a real
//! DC bin for real input, all of which held across the swap unchanged.
//!
//! **`gamma` no longer has a golden here.** `REFACTOR.md` §2.2b replaced the
//! transcribed `DGAMM` with `libm::tgamma`, so a bit-for-bit comparison against the
//! old series is a comparison against code that no longer exists. Its contract is now
//! carried by property tests in `tests/properties.rs` — the functional equation
//! `Gamma(x+1) = x*Gamma(x)`, the factorials, and `Gamma(1/2) = sqrt(pi)` — which held
//! across the swap without modification. `harness/golden/tier0/dgamm.bin` is left in
//! place as a record of what the Fortran produced; nothing reads it.
//! Every comparison is exact. See `PORTING_RULES.md` §10.

use hb_high::fft::remove_quadratic_trend;
use hb_high::fort::{Array1, Complex32, Complex64};
use hb_high::geom::distance_azimuth;
use hb_high::radiation::radiation_pattern;
use hb_high::ray::vertical_slowness;
use hb_high::site::apply_site_amplification;
use std::path::PathBuf;

/// Sequential reader over an `access='stream'` Fortran file.
struct Reader {
    buf: Vec<u8>,
    pos: usize,
    name: String,
}

impl Reader {
    fn open(name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../harness/golden/tier0")
            .join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!("reading {}: {e}. Run harness/kernels/gen_tier0_golden.sh", path.display())
        });
        Self { buf, pos: 0, name: name.to_string() }
    }

    fn take<const N: usize>(&mut self) -> [u8; N] {
        assert!(
            self.pos + N <= self.buf.len(),
            "{}: ran off the end at byte {} (file is {} bytes)",
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

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Assert the whole file was consumed — catches a record-layout
    /// misunderstanding that would otherwise pass silently on a prefix.
    fn assert_exhausted(&self) {
        assert_eq!(
            self.pos, self.buf.len(),
            "{}: consumed {} of {} bytes; record layout disagrees with the driver",
            self.name, self.pos, self.buf.len()
        );
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
fn rdatn_matches_fortran() {
    let mut r = Reader::open("rdatn.bin");
    let mut n = 0;
    while !r.done() {
        let (str_, dip, rak, az, th) = (r.f32(), r.f32(), r.f32(), r.f32(), r.f32());
        let (w_sh, w_sv) = (r.f32(), r.f32());
        let (sh, sv) = radiation_pattern(str_, dip, rak, az, th);
        eq32(&format!("radiation_pattern case {n} rdsh"), sh, w_sh);
        eq32(&format!("radiation_pattern case {n} rdsv"), sv, w_sv);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 2000);
}

#[test]
fn delaz5_matches_fortran() {
    let mut r = Reader::open("delaz5.bin");
    let mut n = 0;
    while !r.done() {
        let (thei, alei, thsi, alsi) = (r.f32(), r.f32(), r.f32(), r.f32());
        let iflag = r.i32();
        let want = [r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32()];
        let g = distance_azimuth(thei, alei, thsi, alsi, iflag);
        let got = [g.delt, g.deltdg, g.deltkm, g.azes, g.azesdg, g.azse, g.azsedg];
        let names = ["delt", "deltdg", "deltkm", "azes", "azesdg", "azse", "azsedg"];
        for k in 0..7 {
            eq32(
                &format!("distance_azimuth case {n} ({thei},{alei})->({thsi},{alsi}) {}", names[k]),
                got[k], want[k],
            );
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 2000);
}

#[test]
fn cr_matches_fortran() {
    let mut r = Reader::open("cr.bin");
    let mut n = 0;
    while !r.done() {
        let p = Complex64::new(r.f64(), r.f64());
        let v = r.f64();
        let want = Complex64::new(r.f64(), r.f64());
        let got = vertical_slowness(p, v);
        eq64(&format!("vertical_slowness({p:?},{v}) re"), got.re, want.re);
        eq64(&format!("vertical_slowness({p:?},{v}) im"), got.im, want.im);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 1500);
}

#[test]
fn flzero_matches_fortran() {
    let mut r = Reader::open("flzero.bin");
    let mut cases = 0;
    while !r.done() {
        let n = r.i32() as usize;
        let dt = r.f32();
        let mut a = Array1::<f32>::new(n);
        for i in 1..=n {
            a[i] = r.f32();
        }
        let a_in = a.clone();
        let want: Vec<f32> = (0..n).map(|_| r.f32()).collect();

        remove_quadratic_trend(n, dt, &mut a);
        for i in 1..=n {
            eq32(&format!("remove_quadratic_trend n={n} dt={dt} a[{i}]"), a[i], want[i - 1]);
        }
        // The correction loop starts at I=3, so the first two samples must come
        // back untouched. Pinned explicitly because it is easy to "fix".
        eq32(&format!("remove_quadratic_trend n={n} a[1] must be untouched"), a[1], a_in[1]);
        eq32(&format!("remove_quadratic_trend n={n} a[2] must be untouched"), a[2], a_in[2]);
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn siteamp_matches_fortran() {
    let mut r = Reader::open("siteamp.bin");
    let mut cases = 0;
    while !r.done() {
        let np2 = r.i32() as usize;
        let nn = r.i32() as usize;
        let np = np2 / 2;

        let mut dfr = Array1::<f32>::new(np + 1);
        for i in 1..=np + 1 {
            dfr[i] = r.f32();
        }
        let mut fn_ = Array1::<f32>::new(nn);
        for i in 1..=nn {
            fn_[i] = r.f32();
        }
        let mut an = Array1::<f32>::new(nn);
        for i in 1..=nn {
            an[i] = r.f32();
        }
        let mut cw = Array1::<Complex32>::filled(np2, Complex32::ZERO);
        for i in 1..=np2 {
            cw[i] = Complex32::new(r.f32(), r.f32());
        }
        let want: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();

        apply_site_amplification(np2, &mut cw, &dfr, nn, &fn_, &an);

        // §2.6 defect 2: the Fortran scales the DC bin (1) and the Nyquist bin
        // (np2/2 + 1) by the raw factor while exponentiating every bin between, two
        // conventions in one routine. That is fixed, so those two bins DELIBERATELY no
        // longer match the Fortran and are excluded here rather than the whole
        // comparison being loosened -- the rest of this golden checks the
        // log-frequency interpolation across the table, which is untouched and still
        // worth an exact check.
        //
        // The Hermitian mirror means the negative-frequency partner of Nyquist is the
        // same bin, so only these two indices move.
        let nyquist = np2 / 2 + 1;
        for i in 1..=np2 {
            if i == 1 || i == nyquist {
                continue;
            }
            eq32(&format!("apply_site_amplification np2={np2} [{i}].re"), cw[i].re, want[i - 1].re);
            eq32(&format!("apply_site_amplification np2={np2} [{i}].im"), cw[i].im, want[i - 1].im);
        }

        // And assert the two excluded bins differ in exactly the way intended: the
        // fixed code applies exp(factor) where the Fortran applied factor. A silent
        // agreement here would mean the fix did not take.
        for (i, factor) in [(1usize, an[1]), (nyquist, an[nn])] {
            let fortran_gain = factor;
            let fixed_gain = factor.exp();
            if want[i - 1].re.abs() > 1e-20 && (fortran_gain - fixed_gain).abs() > 1e-6 {
                let ratio = cw[i].re / want[i - 1].re;
                let expected = fixed_gain / fortran_gain;
                assert!(
                    (ratio / expected - 1.0).abs() < 1e-3,
                    "bin {i}: gain ratio {ratio} vs expected exp({factor})/{factor} = {expected}"
                );
            }
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}
