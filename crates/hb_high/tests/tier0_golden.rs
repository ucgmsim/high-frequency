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
use hb_high::fort::{Complex32, Complex64};
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
fn distance_azimuth_stays_close_to_fortran() {
    // §2.5 replaced DELAZ5 with a WGS84 geodesic, so this CANNOT be bit-exact any more.
    // It is kept rather than deleted because the interesting question changed from "is it
    // identical" to "is the deliberate change the size we said it was" -- and the same
    // oracle data answers that. It would still catch a gross error: a degrees/radians
    // mix-up, a transposed forward/back azimuth, or reading the wrong slot out of
    // geographiclib's width-dependent return tuple.
    //
    // Only the three live outputs are compared; §2.5 dropped `delt`, `deltdg`, `azse` and
    // `azsedg`, which nothing outside this test read.
    //
    // THE BOUNDS ARE STRATIFIED BY SEPARATION, because the two formulations disagree by
    // very different amounts at the two ends and only one end is the port's business:
    //
    //   * under 500 km -- every distance this program will ever see, source subfault to
    //     station -- azimuth agrees to 0.045 degrees.
    //   * near-antipodal (this fixture reaches 18,729 km, about 168 degrees of arc)
    //     azimuth disagrees by up to 1.1 degrees, because the azimuth of a geodesic is
    //     ill-conditioned there: the path direction becomes arbitrary as the endpoints
    //     approach antipodes. Asserting a tight bound on that would be asserting
    //     something about neither implementation's accuracy.
    //
    // So the production regime is bounded tightly and the global figure is printed, not
    // asserted. Widening a single global tolerance until it passed would have hidden which
    // of the two effects was which -- the §2.4c mistake, from the other direction.
    let mut r = Reader::open("delaz5.bin");
    let mut n = 0;
    let mut worst_km_rel = 0.0f64;
    let mut worst_km_at = String::new();
    let mut worst_az_global = 0.0f64;
    let mut worst_az_global_at = String::new();
    let mut worst_az_near = 0.0f64;
    let mut worst_az_near_at = String::new();
    while !r.done() {
        let (thei, alei, thsi, alsi) = (r.f32(), r.f32(), r.f32(), r.f32());
        let iflag = r.i32();
        let [_delt, _deltdg, want_km, _azes, want_azdg, _azse, _azsedg] =
            [r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32(), r.f32()];
        // The geocentric-radians branch was dead and is gone; assert the oracle data never
        // exercised it rather than trusting the old comment that said so.
        assert!(iflag <= 0, "case {n} used the dead coord_mode > 0 path");

        let g = distance_azimuth(thei, alei, thsi, alsi);

        // At zero separation the azimuth is arbitrary in both formulations.
        if want_km > 1.0 {
            let where_ = || format!("case {n} ({thei},{alei})->({thsi},{alsi}) {want_km} km");
            let rel = ((g.deltkm - want_km) / want_km).abs() as f64;
            if rel > worst_km_rel {
                worst_km_rel = rel;
                worst_km_at = where_();
            }
            let gap = {
                let d = (g.azesdg - want_azdg).abs() as f64;
                d.min(360.0 - d)
            };
            if gap > worst_az_global {
                worst_az_global = gap;
                worst_az_global_at = where_();
            }
            if want_km < 500.0 && gap > worst_az_near {
                worst_az_near = gap;
                worst_az_near_at = where_();
            }
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 2000);

    println!(
        "delaz5 fixture, 2000 cases:\n  \
         worst distance      {:.4}% at {worst_km_at}\n  \
         worst azimuth <500km {worst_az_near:.4} deg at {worst_az_near_at}\n  \
         worst azimuth global {worst_az_global:.4} deg at {worst_az_global_at}",
        worst_km_rel * 100.0
    );

    // DELAZ5 is systematically SHORT: a 6371.0 km mean-radius sphere plus a
    // tangent-scaling stand-in for the ellipsoid, against a true geodesic. The largest
    // relative distance error is at the SHORT end -- 0.32% at 1.04 km, i.e. 3 metres --
    // where DELAZ5 switches to its near-coincident half-chord formulation.
    assert!(
        worst_km_rel < 0.005,
        "distance moved by {:.4}% at {worst_km_at}, more than the 0.5% §2.5 allows for",
        worst_km_rel * 100.0
    );
    assert!(
        worst_az_near < 0.1,
        "azimuth moved by {worst_az_near:.4} degrees at {worst_az_near_at}; under 500 km \
         the two formulations should agree to 0.1 degrees"
    );
}

/// `vertical_slowness` no longer matches the Fortran bit for bit, and the reason is pi.
///
/// On the branch cut — `|Im(p)| < 1e-8` with `a < 0` — the phase is forced to exactly
/// pi rather than taken from `atan2`. The Fortran forced it to its own truncated
/// `3.141592654d0`, which is 4.1e-10 short of pi, so the `cos(phi/2)` that should have
/// been an exact zero came out at ~5.7e-11 instead. That error *was* the golden.
///
/// With `std::f64::consts::PI` the same cosine lands at -1.7e-17, six orders of
/// magnitude closer to the true zero. So this test now measures how far the port has
/// moved *away* from the oracle and asserts the move is confined to the component that
/// should be zero, rather than pinning a value that is known to be wrong.
///
/// The imaginary part — which carries the whole magnitude of `eta` on this branch —
/// is still compared exactly, and still passes on all 1500 cases. Off the branch cut
/// nothing changed at all: `atan2` never sees the constant.
#[test]
fn cr_stays_close_to_fortran() {
    let mut r = Reader::open("cr.bin");
    let mut n = 0;
    // Worst divergence, relative to the magnitude of eta for that case.
    let mut worst_rel = 0.0f64;
    let mut worst_at = String::new();
    // Cases where the port and the oracle still agree bit for bit.
    let mut exact = 0;

    while !r.done() {
        let p = Complex64::new(r.f64(), r.f64());
        let v = r.f64();
        let want = Complex64::new(r.f64(), r.f64());
        let got = vertical_slowness(p, v);

        // The magnitude is untouched by the constant and is the natural scale for the
        // real part's departure from zero.
        eq64(&format!("vertical_slowness({p:?},{v}) im"), got.im, want.im);

        if got.re.to_bits() == want.re.to_bits() {
            exact += 1;
        } else {
            let scale = want.norm().max(f64::MIN_POSITIVE);
            let rel = (got.re - want.re).abs() / scale;
            if rel > worst_rel {
                worst_rel = rel;
                worst_at = format!("p={p:?} v={v}");
            }
            // Every divergent case must be one where the oracle's own value was
            // numerical noise around zero, and ours is smaller noise. If the port ever
            // moves a real quantity here, this is what catches it.
            assert!(
                got.re.abs() < want.re.abs(),
                "vertical_slowness({p:?},{v}) re: rust {got:?} is not closer to zero \
                 than fortran {want:?}; the pi fix should only ever shrink this term"
            );
        }
        n += 1;
    }
    r.assert_exhausted();
    assert_eq!(n, 1500);

    println!(
        "cr fixture, {n} cases: {exact} still bit-exact, \
         worst real-part divergence {worst_rel:.3e} of |eta| at {worst_at}"
    );

    // The Fortran's pi error is 4.1e-10 absolute and reaches `cos(phi/2)` halved, so the
    // departure it induces is bounded by ~2.05e-10 of |eta|. Measured worst over the
    // fixture: 2.051e-10, on 450 of the 1500 cases (the ones that land on the branch
    // cut; the other 1050 are still bit-exact). Analysis and measurement agree to three
    // digits, so this bound is 5x headroom over a well-understood number rather than a
    // tolerance widened until the test passed.
    assert!(
        worst_rel < 1.0e-9,
        "vertical_slowness moved by {worst_rel:.3e} of |eta| at {worst_at}, more than \
         the Fortran's own pi error can account for"
    );
}

#[test]
fn flzero_matches_fortran() {
    let mut r = Reader::open("flzero.bin");
    let mut cases = 0;
    while !r.done() {
        let n = r.i32() as usize;
        let dt = r.f32();
        // 0-based since §2.3. The labels still report the Fortran's 1-based sample
        // number, so a failure can be looked up in the oracle's own dump.
        let mut a = vec![0.0; n];
        for slot in a.iter_mut() {
            *slot = r.f32();
        }
        let a_in = a.clone();
        let want: Vec<f32> = (0..n).map(|_| r.f32()).collect();

        remove_quadratic_trend(dt, a.as_mut_slice());
        for i in 0..n {
            eq32(
                &format!("remove_quadratic_trend n={n} dt={dt} a[{}]", i + 1),
                a[i], want[i],
            );
        }
        // The correction loop starts at Fortran I=3, so the first two samples must come
        // back untouched. Pinned explicitly because it is easy to "fix".
        eq32(&format!("remove_quadratic_trend n={n} a[1] must be untouched"), a[0], a_in[0]);
        eq32(&format!("remove_quadratic_trend n={n} a[2] must be untouched"), a[1], a_in[1]);
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

        let dfr: Vec<f32> = (0..=np).map(|_| r.f32()).collect();
        let fn_: Vec<f32> = (0..nn).map(|_| r.f32()).collect();
        let an: Vec<f32> = (0..nn).map(|_| r.f32()).collect();
        let mut cw: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();
        let want: Vec<Complex32> =
            (0..np2).map(|_| Complex32::new(r.f32(), r.f32())).collect();

        apply_site_amplification(cw.as_mut_slice(), dfr.as_slice(), nn, fn_.as_slice(), an.as_slice());

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
        // 0-based: the Fortran's bin 1 is index 0 and its np2/2 + 1 is index np2/2.
        let nyquist = np2 / 2;
        for i in 0..np2 {
            if i == 0 || i == nyquist {
                continue;
            }
            eq32(
                &format!("apply_site_amplification np2={np2} [{}].re", i + 1),
                cw[i].re, want[i].re,
            );
            eq32(
                &format!("apply_site_amplification np2={np2} [{}].im", i + 1),
                cw[i].im, want[i].im,
            );
        }

        // And assert the two excluded bins differ in exactly the way intended: the
        // fixed code applies exp(factor) where the Fortran applied factor. A silent
        // agreement here would mean the fix did not take.
        // DC takes the first table entry and Nyquist the clamped last one -- 0-based,
        // `an[0]` and `an[nn - 1]`.
        for (i, factor) in [(0usize, an[0]), (nyquist, an[nn - 1])] {
            let fortran_gain = factor;
            let fixed_gain = factor.exp();
            if want[i].re.abs() > 1e-20 && (fortran_gain - fixed_gain).abs() > 1e-6 {
                let ratio = cw[i].re / want[i].re;
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
