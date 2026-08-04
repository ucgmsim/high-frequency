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
//! Every comparison is exact. See `PORTING_RULES.md` §10.

use hb_high::fft::{fast, flzero};
use hb_high::fort::{Array1, Complex32, Complex64};
use hb_high::geom::delaz5;
use hb_high::radiation::rdatn;
use hb_high::ray::cr;
use hb_high::site::siteamp;
use hb_high::special::dgamm;
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
        let (sh, sv) = rdatn(str_, dip, rak, az, th);
        eq32(&format!("rdatn case {n} rdsh"), sh, w_sh);
        eq32(&format!("rdatn case {n} rdsv"), sv, w_sv);
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
        let g = delaz5(thei, alei, thsi, alsi, iflag);
        let got = [g.delt, g.deltdg, g.deltkm, g.azes, g.azesdg, g.azse, g.azsedg];
        let names = ["delt", "deltdg", "deltkm", "azes", "azesdg", "azse", "azsedg"];
        for k in 0..7 {
            eq32(
                &format!("delaz5 case {n} ({thei},{alei})->({thsi},{alsi}) {}", names[k]),
                got[k], want[k],
            );
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 2000);
}

#[test]
fn dgamm_matches_fortran() {
    let mut r = Reader::open("dgamm.bin");
    let mut n = 0;
    while !r.done() {
        let x = r.f64();
        let want = r.f64();
        eq64(&format!("dgamm({x})"), dgamm(x), want);
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 1003, "expected 1000 random cases plus 3 error paths");
}

#[test]
fn cr_matches_fortran() {
    let mut r = Reader::open("cr.bin");
    let mut n = 0;
    while !r.done() {
        let p = Complex64::new(r.f64(), r.f64());
        let v = r.f64();
        let want = Complex64::new(r.f64(), r.f64());
        let got = cr(p, v);
        eq64(&format!("cr({p:?},{v}) re"), got.re, want.re);
        eq64(&format!("cr({p:?},{v}) im"), got.im, want.im);
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

        flzero(n, dt, &mut a);
        for i in 1..=n {
            eq32(&format!("flzero n={n} dt={dt} a[{i}]"), a[i], want[i - 1]);
        }
        // The correction loop starts at I=3, so the first two samples must come
        // back untouched. Pinned explicitly because it is easy to "fix".
        eq32(&format!("flzero n={n} a[1] must be untouched"), a[1], a_in[1]);
        eq32(&format!("flzero n={n} a[2] must be untouched"), a[2], a_in[2]);
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 6);
}

#[test]
fn fast_matches_fortran() {
    let mut r = Reader::open("fast.bin");
    let mut cases = 0;
    while !r.done() {
        let nnn = r.i32() as usize;
        let ind = r.i32();
        let mut ace = Array1::<Complex32>::filled(nnn, Complex32::ZERO);
        for i in 1..=nnn {
            ace[i] = Complex32::new(r.f32(), r.f32());
        }
        let want: Vec<Complex32> =
            (0..nnn).map(|_| Complex32::new(r.f32(), r.f32())).collect();

        fast(nnn, &mut ace, ind);
        for i in 1..=nnn {
            eq32(&format!("fast n={nnn} ind={ind} [{i}].re"), ace[i].re, want[i - 1].re);
            eq32(&format!("fast n={nnn} ind={ind} [{i}].im"), ace[i].im, want[i - 1].im);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 26, "13 lengths x 2 directions");
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

        siteamp(np2, &mut cw, &dfr, nn, &fn_, &an);
        for i in 1..=np2 {
            eq32(&format!("siteamp np2={np2} [{i}].re"), cw[i].re, want[i - 1].re);
            eq32(&format!("siteamp np2={np2} [{i}].im"), cw[i].im, want[i - 1].im);
        }
        cases += 1;
    }
    r.assert_exhausted();
    assert_eq!(cases, 4);
}
