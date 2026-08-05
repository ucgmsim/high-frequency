//! `stochastic_spectrum` — the stochastic source spectrum for one subfault.

use ndarray::{azip, s, Array1, ArrayView1, ArrayViewMut1, Axis};
use crate::fft::{forward, inverse, remove_quadratic_trend};
use crate::fft::{Complex32, Complex64};
use crate::rng::{fill_normal_deviates, Draws};
use crate::special::gamma;

/// Transform length and frequency axis for one segment.
///
/// Lived in `sim.rs` until §5.3. It is *the spectrum plan* — every field exists to serve
/// [`stochastic_spectrum`] and the two routines that decorate its output — so it belongs
/// beside them, which is the same argument that moved `highcor.rs` here in §5.1a.
pub struct SpectrumPlan {
    pub np2: usize,
    /// `nfold` — positive-frequency bin count, `np2/2 + 1`, and the length of every
    /// positive-frequency table below.
    pub fold_count: usize,
    pub frequency_hz: Vec<f32>,

    // ---- precomputed transcendentals -------------------------------------------------
    //
    // The three tables below were, between them, the largest single cost in the program:
    // 5.5M `powf`, 2.75M `powf` and 2.75M `ln` per medium-fault run, computing at most
    // `np2`, `fold_count` and `fold_count` DISTINCT values respectively. Every one was a
    // pure function of quantities that do not change within a segment, recomputed once
    // per subfault per ray per component.
    //
    // Hoisting them is bit-exact by construction: `powf` and `ln` are deterministic, so
    // evaluating a pure function once and reusing it gives the identical `f32`.
    /// `ln(frequency_hz[i])`. Index 0 is `-inf` and is never read — the site-amplification
    /// interpolation starts at bin 1, because `ln(0)` has no meaning as a frequency.
    pub log_frequency_hz: Vec<f32>,
    /// `frequency_hz[i]^(1 - q_exponent)` — the path-attenuation frequency dependence.
    pub path_exponent: Vec<f32>,
    /// `(i * dt)^b` — the power-law factor of the Saragoni-Hart envelope.
    ///
    /// `b` comes from the window shape `(window_eps, window_eta)`, which is fixed for the
    /// whole run, so this is `np2` values that were being recomputed 336 times on the
    /// medium fault.
    pub envelope_power: Vec<f32>,
}

impl SpectrumPlan {
    /// Smallest power of two at or above `2 * tmax_s / dt`, and the axis that goes with it.
    pub fn new(tmax_s: f32, dt: f32, q_exponent: f32, window_eps: f32, window_eta: f32) -> Self {
        let ntmax = (2.0 * tmax_s / dt).trunc() as usize;
        let mut np2 = 2usize;
        while np2 < ntmax {
            np2 *= 2;
        }
        let fold_count = np2 / 2 + 1;

        let df = 1.0 / (np2 as f32 * dt);
        // 0-based, which also removes the `- 1`: the axis is `df * bin`.
        let frequency_hz: Vec<f32> = (0..fold_count).map(|bin| df * bin as f32).collect();

        let log_frequency_hz: Vec<f32> = frequency_hz.iter().map(|f| f.ln()).collect();
        let path_exponent: Vec<f32> =
            frequency_hz.iter().map(|f| f.powf(1.0 - q_exponent)).collect();

        // The Saragoni-Hart shape parameter, from the window shape alone. Identical to the
        // expression in `stochastic_spectrum`, which is where it used to live.
        let b = -window_eps * window_eta.ln() / (1.0 + window_eps * (window_eps.ln() - 1.0));
        let envelope_power: Vec<f32> = (0..np2).map(|i| (i as f32 * dt).powf(b)).collect();

        SpectrumPlan {
            np2,
            fold_count,
            frequency_hz,
            log_frequency_hz,
            path_exponent,
            envelope_power,
        }
    }
}

/// Spectral-model constants that are **fixed for the whole run**.
///
/// The cut between this and [`RayPath`] is the useful one: everything here is the same on
/// every one of the hundreds of thousands of calls in a run, and everything there changes
/// on each. Nineteen positional arguments hid that distinction completely.
pub struct SourceModel {
    pub dt: f32,
    /// `tw_eps` / `tw_eta` — the Saragoni-Hart window shape.
    pub window_eps: f32,
    pub window_eta: f32,
    pub subevent_moment: f32,
    pub kappa_s: f32,
    pub moment_scale: f32,
}

/// One `(subfault, ray, component)`'s own path, and the medium at its source.
pub struct RayPath {
    pub distance_km: f32,
    pub window_s: f32,
    pub shear_velocity_km_s: f32,
    pub density_g_cm3: f32,
    pub corner_frequency_hz: f32,
    /// Capped at 15 Hz for the vertical component, which is why this is per-call rather
    /// than a [`SourceModel`] constant.
    pub fmax_hz: f32,
    pub qbar: f32,
}

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
/// `dlm` — the average subfault dimension — was declared and never used, and was kept in
/// the signature for parity with the Fortran call site. That reason expired with Stage 5,
/// and it is gone.
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
pub fn stochastic_spectrum(
    rng: &mut impl Draws,
    plan: &SpectrumPlan,
    model: &SourceModel,
    path: &RayPath,
) -> Array1<Complex32> {
    // Destructured rather than read through the structs field by field, so that the
    // arithmetic below reads as arithmetic. The names are the ones the derivation uses.
    let &SourceModel { dt, window_eps, window_eta, subevent_moment, kappa_s, moment_scale } =
        model;
    let &RayPath {
        distance_km,
        window_s,
        shear_velocity_km_s,
        density_g_cm3,
        corner_frequency_hz,
        fmax_hz,
        qbar,
    } = path;
    let SpectrumPlan { np2, fold_count, frequency_hz, path_exponent, envelope_power, .. } = plan;
    let (np2, fold_count) = (*np2, *fold_count);

    let pai = std::f32::consts::PI;
    let rp = 0.63f32;

    let fc2 = corner_frequency_hz * corner_frequency_hz;

    // fs is the free-surface factor; prtitn the vector partition factor for two
    // orthogonal components, nominally 1/sqrt(2) but written as two digits.
    let fs = 2.0f32;
    let prtitn = 0.71f32;

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
    // The recurrence is what stops this being a plain elementwise expression, and `scan`
    // is the shape that says so: `aa * power` is per-element, `decay` is carried. Built by
    // scanning rather than zero-filling then overwriting -- `np2` floats were being written
    // twice per call, three calls per subfault.
    let decay_per_sample = (-(c as f64) * dt as f64).exp();
    let w: Vec<f32> = envelope_power
        .iter()
        .scan(1.0f64, |decay, &power| {
            let envelope = aa * power * *decay as f32;
            *decay *= decay_per_sample; // exp(0) at the first sample, so this advances after
            Some(envelope)
        })
        .collect();

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
    // Kept a `Vec` rather than an `Array1` precisely so the allocation above stays the one
    // that was measured: `vec![0.0f64; n]` reaches `alloc_zeroed`, which is what puts it on
    // the mmap path at 128 KB.
    //
    // `azip!` rather than a nested `.zip().zip()`: three arrays walked together read as
    // three named bindings instead of a `((shape, fr), path_fr)` tuple unpacked in the
    // pattern. It also ASSERTS the three lengths agree, where `zip` silently stops at the
    // shortest -- a real check, since `frequency_hz` and `path_exponent` are caller-supplied.
    let mut as_ = vec![0.0f64; np2];
    azip!((
        shape in ArrayViewMut1::from(&mut as_[1..fold_count]),
        &fr in ArrayView1::from(&frequency_hz[1..fold_count]),
        &path_fr in ArrayView1::from(&path_exponent[1..fold_count]),
    ) {
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
        let path_attenuation = qbar * path_fr;
        let a2a3 = if kappa_s <= 0.0 {
            let a2 = (1.0 / (1.0 + (omg / omgm))) as f64;
            let a3 = ((-pai * path_attenuation).exp() / distance_cm) as f64;
            a2 * a3
        } else {
            ((-pai * (fr * kappa_s + path_attenuation)).exp() / distance_cm) as f64
        };

        let frank = moment_scale * (fc2 + fr2) / (fc2 + moment_scale * fr2);

        *shape = a1 * a2a3 * frank as f64;
    });


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

    // The mirror is not a second computation. It is the Hermitian symmetry of the first,
    // and separating the two says so.
    //
    // The old loop scaled `ac[j]` into `spectrum[j]` and, in the same iteration, built
    // `conj(ac[j+1] * as_[j+1]) * amp` into `spectrum[np2-j-1]`. That second quantity IS
    // the first at index `j+1`, conjugated: `as_` and `amp` are real, so conjugation
    // commutes with both, and negating an imaginary part is exact in `f32` and `f64`
    // alike. So `conj` before narrowing and `conj` after give the same bits, and the two
    // halves separate. Tier 4 is the check on that claim, not this comment.
    //
    // In place on `ac`, which the returned array now IS. §5.2's second buffer and its zero
    // fill are both gone, and with them the extra allocation §5.2 flagged as temporary.
    let mut spectrum = Array1::from(ac);
    let np = np2 / 2;

    // Positive frequencies, Nyquist included. complex*8 * real*8 goes through complex*16;
    // see the note above. Bin 0 comes out zero because `as_[0]` is never written, which is
    // what the old `j = 0` iteration did too.
    azip!((
        bin in spectrum.slice_mut(s![..fold_count]),
        &shape in ArrayView1::from(&as_[..fold_count]),
    ) {
        let d = Complex64::new(bin.re as f64, bin.im as f64) * shape * amp as f64;
        *bin = Complex32::new(d.re as f32, d.im as f32);
    });

    // Negative frequencies: bin `np2 - k` is `conj(bin k)` for k in `1..np`. A REVERSED
    // view of the head against the tail, rather than the index `np2 - j - 1` whose old
    // comment needed a worked example on np2 = 16 to be believable. Checked the same way:
    // np2 = 16 gives np = 8, so dest runs 9..15 while src runs 7 down to 1 -- dest 9 takes
    // src 7, dest 15 takes src 1.
    //
    // The old loop's last iteration ALSO wrote `spectrum[np]`, which the Nyquist store
    // then overwrote. That write was dead. Dropping it is what leaves the two halves
    // disjoint, which is the only reason this can be a pair of views at all -- and it
    // removes the separate Nyquist line, since the positive half already covers bin `np`.
    let (positive, mut negative) = spectrum.view_mut().split_at(Axis(0), np + 1);
    azip!((dest in &mut negative, &src in positive.slice(s![1..np; -1])) *dest = src.conj());

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
