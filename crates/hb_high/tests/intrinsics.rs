//! Does Rust's libm agree with gfortran's, bit for bit, on the transcendentals
//! this port depends on?
//!
//! Not a kernel gate. This validates the assumption every other gate rests on.
//! `PORTING_RULES.md` §10 names libm divergence and `**` expansion as the two
//! likeliest residual causes of bit-identity failure, so they get their own
//! sweep: if a future toolchain changes one of these, this test points straight
//! at the cause instead of the failure surfacing inside `stochastic_spectrum`.
//!
//! Inputs span the magnitudes the program actually sees — frequencies from 1e-6
//! to 100 Hz and the very large `Rxx` (range in cm) that appears in `stochastic_spectrum`'s
//! exponentials.
//!
//! Regenerate with `harness/kernels/gen_intrinsics_golden.sh`.

use std::path::PathBuf;

/// Force a genuine libm `powf` call.
///
/// Without this, LLVM rewrites `powf(x, <const>)` into the folded form at `-O2`
/// but not at `-O0`, so the same source compares against different functions in
/// the two profiles. That is not hypothetical: it is how the release build first
/// caught `powf(0.5)` being folded to `sqrt` and disagreeing with gfortran.
#[inline(never)]
fn powf_rt(x: f32, e: f32) -> f32 {
    x.powf(e)
}

#[test]
fn rust_libm_matches_gfortran() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/golden/intrinsics/sweep.bin");
    let buf = std::fs::read(&path).unwrap_or_else(|e| {
        panic!("reading {}: {e}. Run harness/kernels/gen_intrinsics_golden.sh", path.display())
    });

    let mut pos = 0usize;
    let f32r = |p: &mut usize| -> f32 {
        let v = f32::from_le_bytes(buf[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    // Tally per operation so a failure names the intrinsic rather than an index.
    let names_f32 = ["x**e", "x**0.5", "x**(-0.5)", "sqrt",
                     "x**(-1.0) as reciprocal", "1.0/x", "ln",
                     "exp(-x*1e-3)", "sin", "cos", "atan2", "hypot-as-sqrt"];
    let names_f64 = ["dsqrt", "dlog", "dexp", "dcos", "dsin", "datan2"];
    let mut bad: Vec<(String, usize)> = Vec::new();
    let mut n = 0;

    while pos < buf.len() {
        let x = f32r(&mut pos);
        let e = f32r(&mut pos);
        let got_f32: [f32; 12] = [
            powf_rt(x, e),
            powf_rt(x, 0.5),
            powf_rt(x, -0.5),
            x.sqrt(),
            // gfortran folds x**(-1.0) into a division, so the port must too.
            1.0 / x,
            1.0 / x,
            x.ln(),
            (-x * 1.0e-3).exp(),
            e.sin(),
            e.cos(),
            e.atan2(x),
            (x * x + e * e).sqrt(),
        ];
        for (k, name) in names_f32.iter().enumerate() {
            let want = f32r(&mut pos);
            if got_f32[k].to_bits() != want.to_bits() {
                if bad.iter().all(|(nm, _)| nm != name) {
                    bad.push((name.to_string(), n));
                }
            }
        }

        let dx = f64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let de = f64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
        pos += 8;
        let got_f64: [f64; 6] = [
            dx.sqrt(),
            dx.ln(),
            (-dx * 1.0e-2).exp(),
            de.cos(),
            de.sin(),
            de.atan2(dx),
        ];
        for (k, name) in names_f64.iter().enumerate() {
            let want = f64::from_le_bytes(buf[pos..pos + 8].try_into().unwrap());
            pos += 8;
            if got_f64[k].to_bits() != want.to_bits() {
                if bad.iter().all(|(nm, _)| nm != name) {
                    bad.push((name.to_string(), n));
                }
            }
        }
        n += 1;
    }

    assert_eq!(pos, buf.len(), "record layout disagrees with the driver");
    assert_eq!(n, 20000);
    assert!(
        bad.is_empty(),
        "Rust and gfortran disagree on: {bad:?}\n\
         Each entry is (intrinsic, first differing case). This does not \
         necessarily mean the port is wrong -- it means an affected kernel \
         cannot be made bit-identical through that intrinsic, and needs a \
         documented ulp bound instead. See PORTING_RULES.md §10."
    );
}

/// Constant exponents do not all map the same way, and guessing costs a real
/// bug — this cost one in `stochastic_spectrum`.
///
/// gfortran folds `x**(-1.0)` into a reciprocal but leaves `x**0.5` as a `powf`
/// call. Rust's `powf` agrees with neither folded form reliably. Measured over
/// 200,000 values:
///
/// | expression | agrees with | disagreement rate |
/// | --- | --- | --- |
/// | gfortran `x**(-1.0)` | `1.0/x` | 0 |
/// | Rust `powf(x,-1.0)` | `1.0/x` | 126 / 200000 |
/// | gfortran `x**0.5` | `sqrt(x)` | 108 / 200000 |
/// | Rust `powf(x,0.5)` | `sqrt(x)` | 108 / 200000 |
///
/// So `**(-1.0)` must be ported as a division and `**0.5` as `powf`. Note the
/// last two rows agree with each other, which is why `powf(0.5)` is correct
/// *because* both call libm — not because either equals `sqrt`.
///
/// The `#[inline(never)]` is essential: without it LLVM rewrites
/// `powf(x, const)` into the folded form, which hides the difference and makes
/// the result depend on optimisation level.
#[test]
fn constant_exponent_powers_do_not_all_fold() {
    let mut pow_recip_diff = 0;
    let mut pow_sqrt_diff = 0;
    for i in 1..=200_000u32 {
        let x = 1.0f32 + (i as f32) * 0.0001;
        if powf_rt(x, -1.0).to_bits() != (1.0f32 / x).to_bits() {
            pow_recip_diff += 1;
        }
        if powf_rt(x, 0.5).to_bits() != x.sqrt().to_bits() {
            pow_sqrt_diff += 1;
        }
    }
    assert!(
        pow_recip_diff > 0,
        "powf(x,-1.0) now agrees with 1.0/x everywhere. If Rust's libm changed,          re-check the **(-1.0) sites -- but keep writing them as divisions,          since that is what gfortran folds to."
    );
    assert!(
        pow_sqrt_diff > 0,
        "powf(x,0.5) now agrees with sqrt(x) everywhere; re-check the **0.5 sites."
    );
}
