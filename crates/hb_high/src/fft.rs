//! `FAST` — the radix-2 FFT — and `FLZERO`, the baseline correction.
//!
//! Transliterated from `hb_high_ref.f`. The FFTW variant of `FAST` is not
//! ported; see `reference/PROVENANCE.md` and `harness/PHASE0C_RESULTS.md` for
//! the measurement that justified dropping it.

use crate::fort::{Array1, Complex32};

/// `SUBROUTINE FAST(NNN,ACE,IND)` — in-place unnormalised complex radix-2 FFT.
///
/// `ind = -1` is the analysis transform (`e^{-i w t}`), `ind = +1` synthesis.
/// Neither direction is scaled; callers divide by `NNN` where they need to.
///
/// `nnn` must be a power of two. Every call site satisfies this — `np2` is built
/// by doubling from 2 at `:1089-1092` — but the assertion makes a future
/// violation loud instead of silently producing garbage, since the bit-reversal
/// permutation below is only a permutation for powers of two.
///
/// The `3.141593` in the twiddle argument is a 7-digit truncation of pi, about
/// 2 `f32` ulps off. It is load-bearing: substituting a more accurate value
/// changes the last bits of every transform. See `PORTING_RULES.md` §1.
pub fn fast(nnn: usize, ace: &mut Array1<Complex32>, ind: i32) {
    assert!(nnn.is_power_of_two(), "FAST requires a power-of-two length, got {nnn}");

    // Bit-reversal permutation (DO 100 / labels 110, 120, 130).
    let mut j = 1usize;
    for i in 1..=nnn {
        if i < j {
            let temp = ace[j];
            ace[j] = ace[i];
            ace[i] = temp;
        }
        let mut m = nnn / 2;
        // 120: IF(J.LE.M) GO TO 130 / J=J-M / M=M/2 / IF(M.GE.2) GO TO 120
        while j > m {
            j -= m;
            m /= 2;
            if m < 2 {
                break;
            }
        }
        j += m;
    }

    // Butterflies (label 140 loop).
    let mut kmax = 1usize;
    while kmax < nnn {
        let istep = kmax * 2;
        for k in 1..=kmax {
            let theta = Complex32::new(
                0.0,
                3.141593 * ((ind * (k as i32 - 1)) as f32) / (kmax as f32),
            );
            // Hoisted out of the inner loop. gfortran does this itself at -O2
            // (confirmed by disassembly in the EMOD3D profiling work), and
            // since CEXP is a pure function of THETA the value is identical
            // either way.
            let w = theta.exp();
            let mut i = k;
            while i <= nnn {
                let jj = i + kmax;
                let temp = ace[jj] * w;
                ace[jj] = ace[i] - temp;
                ace[i] = ace[i] + temp;
                i += istep;
            }
        }
        kmax = istep;
    }
}

/// `SUBROUTINE FLZERO(N,DT,A)` — remove the quadratic acceleration trend that
/// leaves final velocity and displacement at zero.
///
/// Note it modifies only `a(3..=n)`: `a(1)` and `a(2)` are left untouched
/// because the correction loop starts at `I=3`. That asymmetry is preserved.
pub fn flzero(n: usize, dt: f32, a: &mut Array1<f32>) {
    let mut ve = 0.0f32;
    let mut de = 0.0f32;
    let a1 = dt / 2.0;
    let a2 = a1 * dt / 3.0;
    let nstps = n - 1;

    for i in 1..=nstps {
        // DE uses the value of VE from *before* this iteration's update; the
        // two statements are both inside DO 1 and their order matters.
        de = de + ve * dt + a2 * (2.0 * a[i] + a[i + 1]);
        ve = ve + a1 * (a[i] + a[i + 1]);
    }

    let rnstp = nstps as f32;
    let t = rnstp * dt;
    let c1 = 2.0 / t * (ve - 3.0 / t * de);
    let c2 = 6.0 / t * (2.0 / t * de - ve) / t;

    for i in 3..=n {
        let a3 = (i - 1) as f32;
        a[i] = a[i] + c1 + c2 * a3 * dt;
    }
}
