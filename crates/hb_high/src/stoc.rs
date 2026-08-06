//! One subfault's stochastic spectrum, and its inverse transform to a time series.
//!
//! This module is **Boore (1983)**, "Stochastic simulation of high-frequency ground motions
//! based on seismological models of the radiated spectra", *BSSA* 73(6A), 1865–1894 — the
//! point-source stochastic method, equations 1 through 11. [`crate::sim`] is the finite-fault
//! layer around it, from Graves & Pitarka (2010).
//!
//! The idea, which explains the shape of everything here: high-frequency ground motion looks
//! like filtered noise, so rather than solving a wave equation you specify the Fourier
//! *amplitude* spectrum from seismology and pair it with a *random phase* spectrum. See
//! `PHYSICS.md` §1 and §6; `papers/README.md` records the equation-by-equation verification.

use crate::fft::{Complex32, Complex64};
use crate::fft::{forward, inverse, remove_quadratic_trend};
use crate::rng::{Draws, fill_normal_deviates};
use ndarray::{Array1, ArrayView1, ArrayViewMut1, Axis, azip, s};
use std::f32::consts::{PI, TAU};

/// The three factors of Boore (1983) eq. 2's constant
/// `C = R_θφ · FS · PRTITN / (4πρβ³)`.
///
/// **Two of them exist only to be cancelled.** [`stochastic_spectrum`] multiplies them in and
/// [`radiate_and_invert`] divides them straight back out, which leaves the conically averaged
/// pattern from [`crate::radiation`] standing where Boore's scalar average would have been —
/// and that is `RP_ij` in Graves & Pitarka (2010) eq. 11. They are shared constants rather
/// than four literals in two functions precisely so that "change one and you must change its
/// partner" is not something a reader has to be told. See `PHYSICS.md` §5.
///
/// `R_θφ` — the shear-wave radiation pattern averaged over the focal sphere. Cancelled.
pub const AVERAGE_RADIATION_PATTERN: f32 = 0.63;

/// `PRTITN` — the partition of energy between the two horizontal components. Nominally
/// `1/√2`, written to two digits, and cancelled along with [`AVERAGE_RADIATION_PATTERN`].
pub const HORIZONTAL_PARTITION: f32 = 0.71;

/// `FS` — free-surface amplification. A shear wave arriving at a free surface doubles. Not
/// cancelled: this one is real.
const FREE_SURFACE_AMPLIFICATION: f32 = 2.0;

/// Boore (1983) eq. 2 is CGS throughout, so distances and velocities convert on the way in.
const CM_PER_KM: f32 = 100_000.0;

/// Transform length, frequency axis, and the transcendentals that depend only on them.
///
/// One of these is built per fault segment and reused across every subfault, ray and component
/// in it. The three precomputed tables were between them the largest single cost in the
/// program — 5.5M `powf`, 2.75M `powf` and 2.75M `ln` per medium-fault run, each computing at
/// most `np2` or `fold_count` *distinct* values. Hoisting them is exact: `powf` and `ln` are
/// deterministic, so evaluating a pure function once and reusing it gives the identical `f32`.
pub struct SpectrumPlan {
    pub np2: usize,
    /// Positive-frequency bin count, `np2/2 + 1`, and the length of every table below.
    pub fold_count: usize,
    pub frequency_hz: Array1<f32>,

    /// `ln(frequency_hz[i])`, for the site-amplification interpolation. Index 0 is `-inf` and
    /// is never read — `ln(0)` has no meaning as a frequency.
    pub log_frequency_hz: Array1<f32>,
    /// `frequency_hz[i]^(1 - q_exponent)` — the frequency dependence of path attenuation,
    /// which arises because `Q(f) = Q₀·f^x`. See `PHYSICS.md` §3.
    pub path_exponent: Array1<f32>,
    /// `(i·dt)^b` — the power-law factor of the Saragoni–Hart envelope, Boore (1983) eq. 7.
    ///
    /// `b` comes from the window shape, fixed for the whole run, so this is `np2` values that
    /// were otherwise recomputed for every subfault, ray and component.
    pub envelope_power: Array1<f32>,
}

impl SpectrumPlan {
    /// Smallest power of two at or above `2 · tmax_s / dt`, and the frequency axis to match.
    ///
    /// The factor of two is Boore (1983, p. 1869): the record is made about twice the duration
    /// of strong shaking, so that the windowed transient fits inside it with room to decay.
    ///
    /// # `np2` IS THE DRAW COUNT, so it is not a buffer size to tune
    ///
    /// [`stochastic_spectrum`] draws exactly `np2` normal deviates per call. Rounding to a
    /// different length — a 5-smooth "fast" size, say, which for a 131 s window would be
    /// 52488 against this 65536, a 20% saving — would change how many numbers come off
    /// the shared generator, and therefore **every waveform computed after it**. It also
    /// moves `df = 1/(np2·dt)`, the frequency axis the spectral shape is evaluated on.
    ///
    /// So this is physics wearing the costume of an allocation, and `PHYSICS.md` §9 and
    /// `ENGINEERING_RULES` §5 both say the draw count is part of the answer. Changing it is
    /// a LONG-tier question, not an optimisation.
    ///
    /// Nothing here needs a *power* of two in particular — the Hermitian mirror below only
    /// needs `np2` even, and `rustfft` plans any length (mixed-radix, Bluestein for primes),
    /// which is also why it offers no `next_fast_len` to delegate to.
    pub fn new(tmax_s: f32, dt: f32, q_exponent: f32, window_eps: f32, window_eta: f32) -> Self {
        // At least 2 bins, so a degenerate `tmax_s` still gives a transform the rest of the
        // module can index. `next_power_of_two` is exact where the doubling loop was a search:
        // both give the smallest power of two at or above `ntmax`.
        let ntmax = (2.0 * tmax_s / dt).trunc() as usize;
        let np2 = ntmax.next_power_of_two().max(2);
        let fold_count = np2 / 2 + 1;

        let df = 1.0 / (np2 as f32 * dt);
        let frequency_hz = Array1::from_iter((0..fold_count).map(|bin| df * bin as f32));

        let log_frequency_hz = frequency_hz.mapv(f32::ln);
        let path_exponent = frequency_hz.mapv(|f| f.powf(1.0 - q_exponent));

        // Boore (1983) eq. 8, the envelope shape parameter, from the window shape alone.
        // Duplicated in `stochastic_spectrum`, which needs `b` again to form `c`.
        let b = -window_eps * window_eta.ln() / (1.0 + window_eps * (window_eps.ln() - 1.0));
        let envelope_power = Array1::from_iter((0..np2).map(|i| (i as f32 * dt).powf(b)));

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
/// The cut between this and [`RayPath`] is the useful one: everything here is the same on every
/// one of the hundreds of thousands of calls in a run, and everything there changes on each.
pub struct SourceModel {
    pub dt: f32,
    /// Envelope shape: `ε` is where the peak sits as a fraction of the duration, `η` how far
    /// the envelope has decayed by the end. Boore (1983) uses 0.2 and 0.05, and so does this.
    pub window_eps: f32,
    pub window_eta: f32,
    /// `σ_p · dl³` — the subfault moment scale, in dyn·cm.
    pub subevent_moment: f32,
    /// `κ`, the near-surface attenuation operator's decay constant, in seconds. Production
    /// uses 0.045. Anderson & Hough (1984).
    pub kappa_s: f32,
    /// `F` in Graves & Pitarka (2010) eq. 12 — Frankel's finite-fault factor. See the note on
    /// `frank` in [`stochastic_spectrum`], which is where it does its work.
    pub moment_scale: f32,
}

/// One `(subfault, ray, component)`'s own path, and the medium at its source.
pub struct RayPath {
    /// Ray path length, **not** epicentral distance.
    pub distance_km: f32,
    /// Shaping-window length, Boore (1983) `T_w`.
    pub window_s: f32,
    /// `β` and `ρ` at the subfault, not at the station.
    pub shear_velocity_km_s: f32,
    pub density_g_cm3: f32,
    /// `f_ci`, Graves & Pitarka (2010) eq. 13.
    pub corner_frequency_hz: f32,
    /// Capped at 15 Hz for the vertical component, which is why this is per-call rather
    /// than a [`SourceModel`] constant.
    pub fmax_hz: f32,
    /// `q̄`, the travel-time weighted `Σ t/q` along the ray (Ou & Herrmann 1990). For a
    /// straight ray this reduces to `R/(βQ)`.
    pub qbar: f32,
}

/// The complex Fourier spectrum of one subfault's stochastic S-wave motion.
/// (orig. `hb_high_ref.f:1670`)
///
/// Boore (1983) eq. 1 — a product of source, path and site terms, multiplied by the spectrum of
/// windowed random noise and mirrored to Hermitian symmetry:
///
/// ```text
/// A(ω) = C · M₀ · S(ω,ω_c) · P(ω,ω_m) · exp(−ωR/2Qβ) / R
///        └─────────────┘   └──────┘   └────────────┘ └─┘
///           source          high-cut    path Q        spreading
/// ```
///
/// with the finite-fault correction of Graves & Pitarka (2010) eq. 12 folded in.
///
/// Writes `np2` bins into `spectrum`, which must be contiguous — the transform is in place.
/// The caller owns the storage so that the three components can be one `(3, np2)` block
/// rather than three separate allocations per subfault per ray.
///
/// See `PHYSICS.md` §2–§3 and §6 for the physics, and `papers/README.md` for the verification.
///
/// # Precision layout is load-bearing
///
/// The `as f64` casts below are not decoration. Each per-bin term is computed in `f32` and then
/// widened; only the final product accumulates in `f64`. In particular the complex product goes
/// through [`Complex64`] deliberately — doing it entirely in `f32` differs in the last bit.
///
/// # No constant-exponent `powf`, ever
///
/// LLVM folds `powf(x, 0.5)` into `sqrt(x)` at `-O2` but not at `-O0`, so a constant exponent
/// would make this routine's output depend on the optimisation level. There is none here, and
/// there must not be one added. `x^(-1)` is likewise written as an explicit division rather
/// than `powf(x, -1.0)`, which is a libm call that disagrees with `1.0/x` about one time in
/// 1,600.
///
/// # The power normalisation is self-referential, which makes it robust
///
/// `amp = 1/(dt·√(fsa/fold_count))` scales the random sequence so its average **power**
/// spectrum is unity. Boore (1983, p. 1867) specifies unit average *amplitude*, achieved by
/// choosing the noise variance; measuring the realised spectrum and correcting it is the same
/// intent, implemented differently, and matches the RMS averaging of his Figure 1.
///
/// The useful consequence is that this routine is **indifferent to the deviate source's
/// scale**: `fsa` is measured from the very sequence that produced `ac`, so a scale factor `s`
/// in the generator gives `ac` a factor `s`, `fsa` a factor `s²`, and `amp` a factor `1/s`.
/// The product is invariant. One less thing tying the result to a particular generator.
pub fn stochastic_spectrum(
    rng: &mut impl Draws,
    plan: &SpectrumPlan,
    model: &SourceModel,
    path: &RayPath,
    mut spectrum: ArrayViewMut1<Complex32>,
) {
    // Destructured so that the arithmetic below reads as arithmetic, under the names the
    // derivation uses.
    let &SourceModel {
        dt,
        window_eps,
        window_eta,
        subevent_moment,
        kappa_s,
        moment_scale,
    } = model;
    let &RayPath {
        distance_km,
        window_s,
        shear_velocity_km_s,
        density_g_cm3,
        corner_frequency_hz,
        fmax_hz,
        qbar,
    } = path;
    let SpectrumPlan {
        np2,
        fold_count,
        frequency_hz,
        path_exponent,
        envelope_power,
        ..
    } = plan;
    let (np2, fold_count) = (*np2, *fold_count);

    let fc2 = corner_frequency_hz * corner_frequency_hz;
    let distance_cm = distance_km * CM_PER_KM;

    // The Saragoni & Hart (1974) shaping window, `w(t) = a·t^b·e^(−ct)·H(t)`, in the
    // parameterisation of Boore (1983) eq. 7–11. `b` and `c` are eq. 8 and 9; they place the
    // envelope peak at a fraction `ε` of the duration and bring it down to a fraction `η` of
    // the peak by the end.
    let b = -window_eps * window_eta.ln() / (1.0 + window_eps * (window_eps.ln() - 1.0));
    let c = b / window_eps / window_s;
    // Boore (1983) eq. 11, `a = [(2c)^(2b+1) / Γ(2b+1)]^(1/2)`, which normalises the envelope
    // to unit squared area. The mixed precision is deliberate: `2b+1` is formed in `f32`, the
    // division and sqrt happen in `f64`, and the result narrows back.
    let gsa = (2.0 * b + 1.0) as f64;
    let gm = libm::tgamma(gsa);
    let aa = (((2.0 * c).powf(2.0 * b + 1.0) as f64) / gm).sqrt() as f32;

    // Evaluate the envelope on the sample grid `t = i·dt`.
    //
    // `exp(-c·t)` on an evenly spaced grid is a geometric sequence with ratio `exp(-c·dt)`, so
    // it advances by one multiply per sample instead of one `expf` per sample -- `np2`
    // transcendentals removed per call, three calls per subfault. THE RATIO IS ACCUMULATED IN
    // `f64` DELIBERATELY: relative error grows like `n·eps`, which over 16384 samples is ~1e-3
    // in `f32` (visible) against ~2e-12 in `f64`. Underflow is harmless and matches the direct
    // form -- once the product reaches zero it stays there, as `expf` of a large negative
    // argument would.
    //
    // `t^b` has no such recurrence for real `b` and does not need one: it is a per-segment
    // constant and arrives precomputed in `envelope_power`.
    //
    // `scan` rather than a loop because that is the shape of the computation: `aa * power` is
    // per-element, `decay` is carried. Building by scan also avoids zero-filling `np2` floats
    // and immediately overwriting them.
    let decay_per_sample = (-(c as f64) * dt as f64).exp();
    let w: Vec<f32> = envelope_power
        .iter()
        .scan(1.0f64, |decay, &power| {
            let envelope = aa * power * *decay as f32;
            *decay *= decay_per_sample; // exp(0) at the first sample, so this advances after
            Some(envelope)
        })
        .collect();

    // Boore (1983) eq. 2's `C`, in CGS.
    let beta = shear_velocity_km_s * CM_PER_KM;
    let cc = AVERAGE_RADIATION_PATTERN * FREE_SURFACE_AMPLIFICATION * HORIZONTAL_PARTITION
        / (4.0 * PI * density_g_cm3 * (beta * beta * beta));
    let omgc = TAU * corner_frequency_hz;
    let omgm = TAU * fmax_hz;

    // The spectral shape, bin by bin. DC stays zero; bins `1..fold_count` get the shape.
    //
    // SIZED AT `np2`, NOT `fold_count`, ON PURPOSE, even though the top half is never read.
    // Shrinking it MEASURED SLOWER: at np2 = 16384 the `f64` buffer is exactly 128 KB, glibc's
    // mmap threshold, so `alloc_zeroed` hands back fresh already-zero pages for free. At
    // `fold_count` it is 64 KB, comes off the heap, and must be memset for real -- +5.4M
    // instructions per run in exchange for using less memory. It stays a `Vec` rather than an
    // `Array1` for the same reason: `vec![0.0f64; n]` is what reaches `alloc_zeroed`.
    //
    // `azip!` asserts the three lengths agree, where a nested `zip` would silently stop at the
    // shortest -- a real check, since `frequency_hz` and `path_exponent` are caller-supplied.
    let mut as_ = vec![0.0f64; np2];
    azip!((
        shape in ArrayViewMut1::from(&mut as_[1..fold_count]),
        &fr in frequency_hz.slice(s![1..fold_count]),
        &path_fr in path_exponent.slice(s![1..fold_count]),
    ) {
        let fr2 = fr * fr;

        // Boore (1983) eq. 3, the ω-squared source spectrum, "following Aki (1967) and Brune
        // (1970)": `S(ω,ω_c) = ω²/(1 + (ω/ω_c)²)`. Rises as f² below the corner, flat above.
        let omg = TAU * fr;
        let a1 = (cc * subevent_moment * (omg * omg / (1.0 + (omg / omgc) * (omg / omgc)))) as f64;

        // Near-surface and whole-path attenuation, plus 1/R geometric spreading.
        //
        // `κ > 0` -- the production branch -- is Anderson & Hough (1984), `exp(−πκf)`, and
        // Graves & Pitarka (2010) eq. 16. It is combined with the path term in a single
        // `exp`, which is an arithmetic identity rather than an approximation:
        //
        //   path Q      exp(−ωR/2Qβ)  with  Q(f) = Q₀f^x  and  q̄ = R/(Q₀β)
        //             = exp(−π·q̄·f^(1−x))              G&P eq. 14
        //   combined    exp(−πfκ)·exp(−π·q̄·f^(1−x))
        //             = exp(−π(fκ + q̄·f^(1−x)))       one `expf` instead of two
        //
        // `κ ≤ 0` IS A DIFFERENT FILTER, and not Boore's. Boore (1983) eq. 4 is an eight-pole
        // form `[1+(ω/ω_m)^8]^(−1/2)`; this is a single pole, `1/(1 + ω/ω_m)`. Production
        // always has `κ = 0.045`, so only the tier-4 golden's negative-κ case reaches it.
        //
        // Unlike the envelope recurrence above, nothing here assumes the frequency axis is
        // evenly spaced -- `frequency_hz` is caller-supplied data.
        let path_attenuation = qbar * path_fr;
        let a2a3 = if kappa_s <= 0.0 {
            let a2 = (1.0 / (1.0 + (omg / omgm))) as f64;
            let a3 = ((-PI * path_attenuation).exp() / distance_cm) as f64;
            a2 * a3
        } else {
            ((-PI * (fr * kappa_s + path_attenuation)).exp() / distance_cm) as f64
        };

        // Frankel's (1995) finite-fault factor, as given by Graves & Pitarka (2010) eq. 12:
        // it scales the subfault corner frequency towards the mainshock's while keeping the
        // summed moment right.
        //
        // NOT a two-corner spectrum, despite the shape of the expression. Multiplied into
        // `a1` the `(1 + (f/f_c)²)` cancels exactly, leaving a SINGLE-corner spectrum of
        // moment `F·M₀` and corner `f_c/√F`:
        //
        //   a1 ∝ M₀f²/(1+x),  frank = F(1+x)/(1+Fx),  x = (f/f_c)²
        //   a1·frank ∝ (F·M₀)·f² / (1 + (f/(f_c/√F))²)
        //
        // which is what G&P describe in words. Looking for a sag between two corners here --
        // as in Boore, Di Alessandro & Abrahamson (2014) eq. 4 -- will not find one. The
        // cancellation is exact in real arithmetic but is NOT performed, so simplifying it
        // would move the last bits. See `PHYSICS.md` §2.
        let frank = moment_scale * (fc2 + fr2) / (fc2 + moment_scale * fr2);

        *shape = a1 * a2a3 * frank as f64;
    });

    // The random phase spectrum. `remove_quadratic_trend` removes the quadratic acceleration
    // trend so that final velocity and displacement come out at zero.
    let mut a = vec![0.0f32; np2];
    fill_normal_deviates(rng, np2, &mut a);
    remove_quadratic_trend(dt, &mut a);

    // Windowed noise: the envelope times the deviates, as the real part of the signal,
    // written straight into the caller's storage.
    azip!((
        bin in &mut spectrum,
        &deviate in ArrayView1::from(&a[..]),
        &envelope in ArrayView1::from(&w[..]),
    ) {
        *bin = Complex32::new(deviate * envelope, 0.0);
    });

    forward(
        spectrum
            .as_slice_mut()
            .expect("the caller's spectrum row must be contiguous"),
    );

    // Measure the realised average power of the noise spectrum, so it can be normalised out.
    // `norm_sqr()` is `re² + im²` -- the same quantity as `|z|²` without the `hypot` and the
    // squaring that undoes it, which was 4.5% of total runtime.
    let fsa: f32 = spectrum
        .slice(s![..fold_count])
        .iter()
        .map(Complex32::norm_sqr)
        .sum();
    let amp = 1.0 / (dt * (fsa / fold_count as f32).sqrt());

    // Apply the shape, then mirror to Hermitian symmetry. The mirror is not a second
    // computation -- it is the symmetry of the first, and `as_` and `amp` being real is what
    // lets the two separate: conjugation commutes with real scaling, and negating an imaginary
    // part is exact, so conjugating before or after narrowing gives the same bits.
    //
    let np = np2 / 2;

    // Positive frequencies, Nyquist included. The `Complex64` intermediate is the precision
    // note above. Bin 0 comes out zero because `as_[0]` is never written.
    azip!((
        bin in spectrum.slice_mut(s![..fold_count]),
        &shape in ArrayView1::from(&as_[..fold_count]),
    ) {
        let d = Complex64::new(bin.re as f64, bin.im as f64) * shape * amp as f64;
        *bin = Complex32::new(d.re as f32, d.im as f32);
    });

    // Negative frequencies: bin `np2 - k` is `conj(bin k)` for k in `1..np`. A reversed view of
    // the head assigned into the tail, which cannot be off by one the way an index expression
    // can. Checked on np2 = 16, where np = 8: dest runs 9..15 while src runs 7 down to 1, so
    // dest 9 takes src 7 and dest 15 takes src 1.
    //
    // Bin `np` is its own mirror and belongs to the positive half, which is why the two halves
    // are disjoint and this can be a pair of views at all.
    let (positive, mut negative) = spectrum.view_mut().split_at(Axis(0), np + 1);
    azip!((dest in &mut negative, &src in positive.slice(s![1..np; -1])) *dest = src.conj());
}

/// Apply the radiation pattern, invert to a time series, scale and taper.
/// (orig. `hb_high_ref.f:2234`)
///
/// Completes Graves & Pitarka (2010) eq. 10 for one component: the spectrum from
/// [`stochastic_spectrum`] carries `C·S·G·P`, and the conically averaged pattern `RP_ij` from
/// [`crate::radiation`] goes on here. The inverse transform runs in place on `spectrum`,
/// which is therefore left as scratch, and the real samples land in `time_series`.
///
/// Lengths carry what the original passed as three separate counts: the radiation pattern
/// covers the positive frequencies, so `radiation.len()` *is* `fold_count`, and the mirrored
/// half is `radiation[1..fold_count-1]` walked backwards.
///
/// # The division here is the cancellation
///
/// [`AVERAGE_RADIATION_PATTERN`] and [`HORIZONTAL_PARTITION`] are multiplied in by
/// [`stochastic_spectrum`] as part of Boore (1983) eq. 2's constant `C`, and divided back out
/// here so that the conically averaged pattern stands in their place — which is what Graves &
/// Pitarka (2010) eq. 11 asks for. They are two shared constants rather than four literals, so
/// the pairing cannot drift.
///
/// # The taper constant was a typo in the original, and is fixed here
///
/// The taper step used `3.14159625`, which is **not** pi — the last digits of `3.14159265` are
/// transposed. Every other occurrence in the source was some honest truncation of the correct
/// value, so this one was a slip. The error is about 1.1e-6 relative, roughly thirty times the
/// worst of those truncations, and it left the taper fractionally short of a half cosine so the
/// final sample was not exactly zero. It is `std::f32::consts::PI` now and the taper closes.
pub fn radiate_and_invert(
    mut spectrum: ArrayViewMut1<Complex32>,
    radiation: ArrayView1<f32>,
    mut time_series: ArrayViewMut1<f32>,
) {
    let np2 = spectrum.len();
    let fold_count = radiation.len();
    // Always `fold_count - 2`. Sliced explicitly rather than as `fold_count..` because the two
    // are only incidentally equal for the `np2` this program uses.
    let mirror_count = fold_count - 2;

    // Positive frequencies, then the mirrored half. The pattern is SIGNED -- the polarity from
    // `crate::radiation` is carried through rather than discarded.
    //
    // `azip!` rather than `*=` because ndarray's operator overloads want matching element
    // types, and this is `Complex32` scaled by `f32`. Same iteration either way.
    azip!((bin in &mut spectrum.slice_mut(s![..fold_count]), &gain in &radiation) *bin *= gain);
    azip!(
        (bin in &mut spectrum.slice_mut(s![fold_count..fold_count + mirror_count]),
         &gain in &radiation.slice(s![1..fold_count - 1; -1]))
        *bin *= gain
    );

    inverse(
        spectrum
            .as_slice_mut()
            .expect("the caller's spectrum row must be contiguous"),
    );

    // Assigns rather than accumulates, so `time_series` needs no pre-zeroing: every element
    // is written before anything reads it.
    let scale = 1.0 / (AVERAGE_RADIATION_PATTERN * HORIZONTAL_PARTITION * np2 as f32);
    azip!((sample in &mut time_series, &bin in &spectrum) *sample = scale * bin.re);

    // Raised-cosine taper over the final tenth, so the transient closes smoothly instead of
    // being truncated. The `i + 1` is what makes the last sample land on `cos(π)` and the
    // taper reach zero.
    let taper_len = np2 / 10;
    let step = std::f32::consts::PI / taper_len as f32;
    let taper = Array1::from_shape_fn(taper_len, |i| 0.5 * (1.0 + ((i + 1) as f32 * step).cos()));
    let mut tail = time_series.slice_mut(s![np2 - taper_len..]);
    tail *= &taper;
}

#[cfg(test)]
mod tests {
    use libm::tgamma as gamma;

    /// The one gamma argument production actually evaluates, pinned so that a future change of
    /// implementation has to look at the value that matters rather than at the thousand that
    /// do not. `3.5062997341156006` is `2b+1` for the fixed window shape.
    ///
    /// The reference value is what the original's hand-rolled series returned; the two differ
    /// by three ulps of `f64`, and the assertion below shows that difference does not survive
    /// the narrowing to `f32` that the consumer applies.
    #[test]
    fn the_production_gamma_argument_survives_narrowing_to_f32() {
        let gsa = 3.5062997341156006f64;
        let reference = 3.346549271566832f64;
        let got = gamma(gsa);
        assert!(
            (got - reference).abs() / reference < 1e-15,
            "gamma({gsa}) = {got}, reference value {reference}"
        );
        // The consumer narrows to f32; show the difference does not survive that.
        assert_eq!((got as f32).to_bits(), (reference as f32).to_bits());
    }

    /// A pole must be loud. The caller never checks, so a finite sentinel would propagate a
    /// plausible-looking wrong number into the spectrum.
    #[test]
    fn gamma_poles_are_not_finite() {
        for pole in [0.0, -1.0, -2.0, -3.0] {
            assert!(
                !gamma(pole).is_finite(),
                "gamma({pole}) = {} should not be a usable value",
                gamma(pole)
            );
        }
    }
}
