//! `stochastic_spectrum` — the stochastic source spectrum for one subfault.

use ndarray::{azip, s, Array1, ArrayView1};
use crate::fft::{forward, inverse, remove_quadratic_trend};
use crate::fort::{Complex32, Complex64};
use crate::rng::{fill_normal_deviates, Draws};
use crate::special::gamma;

/// `subroutine stochastic_spectrum(...)` — `hb_high_ref.f:1670`.
///
/// Builds the complex Fourier spectrum of one subfault's stochastic S-wave
/// motion: a Brune omega-squared source, a kappa/fmax high-cut, path Q, and the
/// Frankel two-corner operator, multiplied by a unit-power random phase
/// spectrum and mirrored to Hermitian symmetry. Returns `np2` values.
///
/// The spectrum was an out-parameter — a caller-owned scratch buffer refilled once per
/// subfault per component. Returning it lets the value flow straight into
/// [`radiate_and_invert`], which consumes it, so the caller no longer clones.
///
/// `dlm` is declared and never used; kept in the signature for call-site parity.
///
/// # Precision layout
///
/// `a1`, `a2`, `a3`, `gsa`, `gm` and the `as` array are `real*8`; everything
/// else is `real*4`. So each `as(i)` term is *computed* in single precision and
/// then widened, and only the final product `a1*a2*a3*frank` accumulates in
/// double. The `as f64` casts below are exactly those widening points.
///
/// Two subtleties verified against gfortran 16.1.1 rather than assumed:
///
/// * `complex*8 * real*8` promotes the **complex** operand to `complex*16` and
///   multiplies in double, narrowing only on assignment. Doing the whole product
///   in `f32` differs in the last bit, so `spectrum(i) = ac(i)*as(i)*amp` is built
///   through [`Complex64`] here.
/// * Constant exponents need care, and the two cases here differ. gfortran folds
///   `x**(-1.0)` into a reciprocal — verified identical to `1.0/x` over 200,000
///   values — but Rust's `powf(-1.0)` is a libm call that disagrees with `1.0/x`
///   in about 1 case in 1,600. So `**(-1.0)` is written as an explicit division.
///   `x**0.5`, by contrast, gfortran does *not* fold: it calls `powf`. Rust
///   cannot express that portably — LLVM folds `powf(x, 0.5)` to `sqrt(x)` at
///   `-O2` but not at `-O0`, so the result would depend on optimisation level.
///   The only `x**0.5` in this routine feeds a dead store and is simply not
///   computed. See `PORTING_RULES.md` §4b.
///
/// # The power normalisation is self-referential, which makes it robust
///
/// `amp = 1/(dt*sqrt(fsa/fold_count))` normalises so the average *power* spectrum is
/// unity, per Boore (1983) — a 2009-03-18 change from normalising the amplitude
/// spectrum, which reduced motions about 10% and was offset by raising the default
/// corner frequency 5%.
///
/// This comment used to claim the calibration depends on `fill_normal_deviates`'s
/// unit-RMS rescale, so that a plain N(0,1) generator would change the output level.
/// **That is wrong.** `fsa` is measured from the very sequence that was rescaled, so if
/// the deviates carry a scale factor `s`, then `ac` does too, `fsa` carries `s^2`, and
/// `amp` carries `1/s`. The product `ac * as_ * amp` is invariant. See
/// [`crate::rng::fill_normal_deviates`] for the full trace.
///
/// The practical consequence is the opposite of what was documented: this routine is
/// **indifferent** to the deviate source's scale, which is one less thing tying it to a
/// particular generator.
#[allow(clippy::too_many_arguments)]
pub fn stochastic_spectrum(
    rng: &mut impl Draws,
    np2: usize,
    distance_km: f32,
    window_s: f32,
    window_eps: f32,
    window_eta: f32,
    shear_velocity_km_s: f32,
    density_g_cm3: f32,
    dt: f32,
    subevent_moment: f32,
    _avg_subfault_km: f32,
    corner_frequency_hz: f32,
    fmax_hz: f32,
    kappa_s: f32,
    frequency_hz: &[f32],
    // `frequency_hz[i]^(1 - qfexp)`, precomputed per segment -- see `SpectrumPlan`.
    path_exponent: &[f32],
    // `(i * dt)^b`, precomputed per segment -- see `SpectrumPlan`.
    envelope_power: &[f32],
    qbar: f32,
    moment_scale: f32,
) -> Array1<Complex32> {
    let pai = std::f32::consts::PI;
    let rp = 0.63f32;

    let fc2 = corner_frequency_hz * corner_frequency_hz;

    // fs is the free-surface factor; prtitn the vector partition factor for two
    // orthogonal components, nominally 1/sqrt(2) but written as two digits.
    let fs = 2.0f32;
    let prtitn = 0.71f32;

    let fold_count = np2 / 2 + 1;
    let distance_cm = distance_km * 100000.0;

    // Saragoni-Hart style envelope: b and c from the (window_eps, window_eta) window shape.
    let b = -window_eps * window_eta.ln() / (1.0 + window_eps * (window_eps.ln() - 1.0));
    let c = b / window_eps / window_s;
    // Computed in real*4, then widened -- gsa is real*8 but 2*b+1.0 is not.
    let gsa = (2.0 * b + 1.0) as f64;
    let gm = gamma(gsa);
    // The power is real*4; the division by gm and the sqrt are real*8; the
    // result narrows back to real*4.
    let aa = (((2.0 * c).powf(2.0 * b + 1.0) as f64) / gm).sqrt() as f32;

    // Saragoni-Hart envelope, `aa * t^b * exp(-c*t)` on the evenly spaced grid
    // `t = (i-1)*dt`.
    //
    // `exp(-c*t)` on that grid is a geometric sequence with ratio `exp(-c*dt)`, so it
    // advances by one multiply per sample instead of one `expf` per sample. That is
    // `np2` transcendentals removed per call, three calls per subfault.
    // `PROFILE.md` item 4 ruled this out for bit-identity; Stage 2 allows it.
    //
    // The ratio is accumulated in `f64` deliberately. Relative error grows like
    // `n * eps`, which over 16384 samples is ~1e-3 in `f32` — visible — against
    // ~2e-12 in `f64`. Underflow is harmless and matches the direct form: once the
    // product reaches zero it stays there, exactly as `expf` of a large negative
    // argument would.
    //
    // `t^b` has no recurrence for real `b`, but it does not need one: `b` comes from the
    // window shape, which is fixed for the whole run, and `t` is `index * dt` on a fixed
    // grid. So the whole table is a per-segment constant and arrives precomputed. That
    // removed 5.5M `powf` per medium-fault run computing at most `np2` distinct values.
    let decay_per_sample = (-(c as f64) * dt as f64).exp();
    let mut decay = 1.0f64; // exp(0) at the first sample
    let mut w = vec![0.0f32; np2];
    for (envelope, &power) in w.iter_mut().zip(envelope_power) {
        *envelope = aa * power * decay as f32;
        decay *= decay_per_sample;
    }

    let beta = shear_velocity_km_s * 100000.0;
    let cc = rp * fs * prtitn / (4.0 * pai * density_g_cm3 * (beta * beta * beta));
    let omgc = 2.0 * pai * corner_frequency_hz;
    let omgm = 2.0 * pai * fmax_hz;

    // Bin 0 (DC) stays zero; bins 1..fold_count get the shape. Slicing both from 1 keeps
    // the two arrays' correspondence in the types instead of in two matching `[i]`s.
    // Sized at `np2` even though the highest index ever read is `fold_count - 1`, i.e.
    // half of it is never touched. Shrinking it to `fold_count` MEASURED SLOWER: at np2 =
    // 16384 the `f64` buffer is exactly 128 KB, which is glibc's mmap threshold, so
    // `calloc` hands back fresh already-zero pages and the zeroing costs nothing. At
    // `fold_count` it is 64 KB, comes off the heap, and has to be memset for real.
    // +5.4M instructions per run for "using less memory". See REFACTOR.md §2.6b, which
    // measured the same effect from the other direction.
    let mut as_ = vec![0.0f64; np2];
    for ((shape, &fr), &path_fr) in as_[1..fold_count]
        .iter_mut()
        .zip(&frequency_hz[1..])
        .zip(&path_exponent[1..])
    {
        let fr2 = fr * fr;

        // The Q model qv = 150.0*fr**0.5 is computed by the Fortran but feeds
        // only the first, dead, a3 form below, so it is not computed here.
        // Earlier variants in the source: 100+10*fr**1.70 and 270*fr**0.5
        // ("Beresnev Northridge").
        //
        // Dropping it also removes the port's last CONSTANT-exponent powf.
        // That matters: LLVM folds powf(x, 0.5) into sqrt(x) at -O2 but not at
        // -O0, and gfortran's x**0.5 is a real powf call, so keeping it would
        // have made the port's output depend on optimisation level. See
        // PORTING_RULES.md §4b.

        let omg = 2.0 * pai * fr;
        let a1 = (cc * subevent_moment * (omg * omg / (1.0 + (omg / omgc) * (omg / omgc)))) as f64;

        // `a2` (near-surface attenuation) and `a3` (path attenuation) were two
        // separate `expf` calls per frequency bin. Two simplifications, both
        // arithmetic identities verified numerically to double-precision epsilon
        // before being applied:
        //
        //   a3's argument   -0.5*omg*qbar*fr^-qfe  with omg = 2*pi*fr
        //                 = -pi*qbar*fr^(1-qfe)              one fewer multiply
        //   a2 * a3       = exp(-pi*fr*kappa) * exp(-pi*qbar*fr^(1-qfe))
        //                 = exp(-pi*(fr*kappa + qbar*fr^(1-qfe)))   one fewer expf
        //
        // Only the `kappa > 0` branch can be combined; the `kappa <= 0` form of `a2`
        // is a rational function, not an exponential, and production always has
        // `kappa = 0.045`. Both branches are exercised — the tier-4 golden includes a
        // negative-kappa case.
        //
        // No assumption is made about the frequency axis being evenly spaced, unlike
        // the envelope recurrence above. `frequency_hz` is caller-supplied data.
        let path = qbar * path_fr;
        let a2a3 = if kappa_s <= 0.0 {
            let a2 = (1.0 / (1.0 + (omg / omgm))) as f64;
            let a3 = ((-pai * path).exp() / distance_cm) as f64;
            a2 * a3
        } else {
            ((-pai * (fr * kappa_s + path)).exp() / distance_cm) as f64
        };

        let frank = moment_scale * (fc2 + fr2) / (fc2 + moment_scale * fr2);

        *shape = a1 * a2a3 * frank as f64;
    }

    let mut a = vec![0.0f32; np2];
    fill_normal_deviates(rng, np2, &mut a);
    remove_quadratic_trend(dt, &mut a);

    // Built by collecting rather than zero-filling then overwriting every element: `ac` is
    // np2 complex values allocated three times per subfault, and the zeroing pass was
    // pure waste.
    let mut ac: Vec<Complex32> = a
        .iter()
        .zip(&w)
        .map(|(&deviate, &envelope)| Complex32::new(deviate * envelope, 0.0))
        .collect();

    forward(&mut ac);

    // Average POWER spectrum to unity (2009-03-18), not amplitude.
    //
    // `norm_sqr()` is `re^2 + im^2`. The Fortran wrote `cabs(ac(i))*cabs(ac(i))`,
    // which takes a square root and then squares it away again -- one `hypotf` per
    // frequency bin, and `hypot` is not cheap. That was 4.5% of total runtime, and
    // `PORTING_RULES.md` §4 / `PROFILE.md` item 3 recorded that it could not be
    // simplified because `hypot(re,im)^2` and `re^2 + im^2` differ in the last bits.
    // Under Stage 2 it can: this is the same quantity, computed without the detour.
    // `Sum for f32` folds left to right, matching the Fortran's `fsa = fsa + ..`. Any
    // reassociating form (chunked, pairwise, parallel) would not -- see REFACTOR.md §1.3b.
    let fsa: f32 = ac[..fold_count].iter().map(Complex32::norm_sqr).sum();
    let amp = 1.0 / (dt * (fsa / fold_count as f32).sqrt());

    // complex*8 * real*8 goes through complex*16; see the note above.
    let scale = |z: Complex32, s: f64, m: f32| -> Complex32 {
        let d = Complex64::new(z.re as f64, z.im as f64) * s * m as f64;
        Complex32::new(d.re as f32, d.im as f32)
    };

    // 0-based. The Fortran writes spectrum(i) and its conjugate partner
    // spectrum(np2 - i + 1) from a 1-based i; with j = i - 1 the partner is
    // np2 - i + 1 - 1 = np2 - j - 1. Checked on np2 = 16: Fortran i = 1 writes
    // spectrum(16), storage element 15, and j = 0 gives 16 - 0 - 1 = 15.
    //
    // Note the partner of the LAST iteration and the Nyquist store below are the same
    // element -- Fortran i = np writes spectrum(np + 1) = spectrum(fold_count) -- so the
    // Nyquist assignment overwrites it. That ordering is the original's and is kept.
    // TEMPORARY second buffer. The mirror below reads `ac` and writes `spectrum`, and an
    // in-place merge is possible -- every index is read before it is written -- but it
    // needs the Nyquist value saved before the loop, and that index reasoning is §5.5's
    // job, isolated so that a red snapshot there means the indices rather than this
    // plumbing. §5.5 folds the two together and this allocation goes.
    //
    // The zero fill is not waste that survives: the loop writes `0..np`, `np..np2` and
    // the Nyquist, which is every element.
    let mut spectrum = Array1::from_elem(np2, Complex32::ZERO);

    let np = np2 / 2;
    for j in 0..np {
        spectrum[j] = scale(ac[j], as_[j], amp);
        // conjg() is applied to the complex*16 product, before narrowing.
        let d = Complex64::new(ac[j + 1].re as f64, ac[j + 1].im as f64) * as_[j + 1];
        let d = d.conj() * amp as f64;
        spectrum[np2 - j - 1] = Complex32::new(d.re as f32, d.im as f32);
    }
    spectrum[fold_count - 1] = scale(ac[fold_count - 1], as_[fold_count - 1], amp);

    spectrum
}

/// Multiply a spectrum by its radiation pattern, invert, scale and taper.
///
/// `hb_high_ref.f:2234` (`HIGHCOR`). Returns the real time series; the spectrum goes in by
/// value because the inverse transform consumes it.
///
/// Lengths carry the information the Fortran passed as `fold_count`, `mirror_count` and
/// `np2`: the radiation pattern covers the positive frequencies, so `radiation.len()` *is*
/// `fold_count`, and the mirrored half is `radiation[1..fold_count - 1]` walked backwards.
/// The Fortran wrote that index as `radiation(2*fold_count - i)`, which needed a worked
/// example on `np2 = 16` to believe; a reversed view cannot be off by one.
///
/// `RADIATION_NORM` is the radiation-pattern normalisation and `PARTITION_FACTOR` the vector
/// partition factor for two orthogonal components — nominally `1/sqrt(2)`, written as two
/// digits in the original and kept that way because it is a calibration choice, not an
/// approximation of anything.
///
/// # The taper constant was a typo, and it is now fixed
///
/// The taper used `dd = 3.14159625/n0` (`:2266`). That is **not** pi — the last digits of
/// `3.14159265` are transposed. Every other occurrence in the file is some truncation of the
/// correct value (`3.1415926`, `3.14159265`, `3.141592654`), so this one was a genuine slip
/// rather than a deliberate approximation.
///
/// It was reproduced verbatim for as long as bit-identity was the contract. The error is
/// about 1.1e-6 relative — roughly thirty times the worst of the file's honest truncations —
/// and it left the taper fractionally short of a half cosine, so the final sample was not
/// exactly zero. It is now `std::f32::consts::PI`, and the taper closes properly.
pub fn radiate_and_invert(
    mut spectrum: Array1<Complex32>,
    radiation: ArrayView1<f32>,
) -> Array1<f32> {
    const RADIATION_NORM: f32 = 0.63;
    const PARTITION_FACTOR: f32 = 0.71;

    let np2 = spectrum.len();
    let fold_count = radiation.len();
    // `mirror_count` was a parameter and is always `fold_count - 2`; the old code asserted
    // exactly this. Sliced explicitly rather than as `fold_count..` because the two are only
    // incidentally equal for the np2 the program uses.
    let mirror_count = fold_count - 2;

    // Positive frequencies, signed radiation pattern -- sign preserved since 2004-12-21,
    // where the older code took abs().
    // `azip!` rather than `*=`: ndarray's operator overloads require both sides to have the
    // same element type, and this is Complex32 scaled by f32. Same iteration, same
    // bit-exactness -- element-wise either way.
    azip!((bin in &mut spectrum.slice_mut(s![..fold_count]), &gain in &radiation) *bin *= gain);
    azip!(
        (bin in &mut spectrum.slice_mut(s![fold_count..fold_count + mirror_count]),
         &gain in &radiation.slice(s![1..fold_count - 1; -1]))
        *bin *= gain
    );

    inverse(spectrum.as_slice_mut().expect("an owned Array1 is contiguous"));

    let scale = 1.0 / (RADIATION_NORM * PARTITION_FACTOR * np2 as f32);
    let mut samples = spectrum.mapv(|bin| scale * bin.re);

    // Raised-cosine taper over the final tenth. `i + 1` keeps the Fortran's 1-based step
    // number, which is what makes the last sample land on cos(pi) and the taper close.
    let taper_len = np2 / 10;
    let step = std::f32::consts::PI / taper_len as f32;
    let taper =
        Array1::from_shape_fn(taper_len, |i| 0.5 * (1.0 + ((i + 1) as f32 * step).cos()));
    let mut tail = samples.slice_mut(s![np2 - taper_len..]);
    tail *= &taper;

    samples
}
