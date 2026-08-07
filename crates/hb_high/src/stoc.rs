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
use std::collections::HashMap;

use crate::rng::Draws;
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
/// One of these is built per **distinct transform length** and reused across every subfault,
/// ray and component that needs that length — see [`PlanCache`]. The three precomputed tables
/// were between them the largest single cost in the program — 5.5M `powf`, 2.75M `powf` and
/// 2.75M `ln` per medium-fault run, each computing at most `np2` or `fold_count` *distinct*
/// values. Hoisting them is exact: `powf` and `ln` are deterministic, so evaluating a pure
/// function once and reusing it gives the identical `f32`.
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
    /// The transform length a window of `window_s` seconds needs.
    ///
    /// The factor of two is Boore (1983, p. 1869): the record is made about twice the duration
    /// of strong shaking, so that the windowed transient fits inside it with room to decay.
    ///
    /// # `np2` IS THE DRAW COUNT, so it is not a buffer size to tune casually
    ///
    /// [`stochastic_spectrum`] draws exactly `np2` normal deviates per call, and `np2` also
    /// sets `df = 1/(np2·dt)`, the frequency axis the spectral shape is evaluated on. So this
    /// is physics wearing the costume of an allocation: changing it moves **every waveform
    /// computed after it**, and `ENGINEERING_RULES` §4 requires the reasoning to be written
    /// down rather than assumed.
    ///
    /// It has been changed twice, deliberately, and here is the reasoning.
    ///
    /// * **The length is now 7-smooth rather than a power of two** — see
    ///   [`crate::fft::good_length`]. Nothing here ever needed a power of two; the Hermitian
    ///   mirror needs `np2` even and `rustfft` plans any length.
    ///
    /// * **The window is the subfault's own, not the segment's longest.** This used to be
    ///   called once per segment with the maximum over all its subfaults, so a subfault whose
    ///   envelope had decayed to `η²` of its peak by sample 8,000 still drew 131,072 deviates
    ///   and transformed all of them. Every subfault now gets the same *relative* headroom —
    ///   twice its own window — that the longest one always had, which is what the factor of
    ///   two means in the first place. The waste it removes is the difference between the two,
    ///   and on an Alpine Fault deck recorded at 600 km that is most of the run.
    ///
    /// The consequence to be aware of is that `df` is now coarser for a short-window subfault
    /// than it was. That is the intended reading of Boore's construction rather than a
    /// degradation of it: the frequency resolution a transient needs is set by its own
    /// duration, and resolving an 8,000-sample envelope on a 131,072-point grid buys nothing
    /// the envelope contains.
    #[must_use]
    pub fn length_for(window_s: f32, dt: f32) -> usize {
        crate::fft::good_length((2.0 * window_s / dt).trunc() as usize)
    }

    /// Build the tables for a transform of `np2` points.
    ///
    /// `np2` comes from [`Self::length_for`] on the production path. It is taken directly
    /// rather than derived here so that the tier-4 golden — whose fixtures record `np2` as an
    /// input — can build a plan for exactly the length the Fortran used.
    pub fn new(np2: usize, dt: f32, q_exponent: f32, window_eps: f32, window_eta: f32) -> Self {
        assert!(
            np2 >= 2 && np2.is_multiple_of(2),
            "a spectrum plan needs an even length of at least 2, got {np2}"
        );
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

/// The plans a station needs, built on demand and keyed by transform length.
///
/// # Why a cache, and why it is per station rather than per run
///
/// [`SpectrumPlan::length_for`] gives a *different* length for nearly every subfault, and a
/// plan costs `np2` `powf` calls plus `fold_count` more `powf` and `ln`. Rebuilding one per
/// subfault would put those transcendentals straight back on the hot path — the exact cost
/// the tables were hoisted out of it to avoid. Sharing by length recovers the hoist: the
/// lengths repeat heavily, because [`crate::fft::good_length`] quantises them onto a ladder
/// of four rungs per octave. An Alpine Fault station lands 1,854 subfaults on **23**
/// distinct lengths, holding about 5.7 MB of tables — see that function for why the ladder
/// is as coarse as it is, which is a decision this cache's memory footprint drove.
///
/// It is per station, and deliberately not on [`crate::sim::Simulator`], because a
/// `Simulator` is shared across a dask thread pool behind `&self`. Caching there would need
/// either a lock on the hottest path in the program or interior mutability that is not
/// `Sync`, and the cost of not doing it is one plan build per distinct length per station —
/// measured at under 2% of a station, against 1,854 subfaults' worth of tables it saves.
/// `crate::fft`'s own plan cache is thread-local for the same reason.
pub struct PlanCache {
    dt: f32,
    q_exponent: f32,
    window_eps: f32,
    window_eta: f32,
    /// Keyed by `np2`. A `HashMap` rather than a sorted `Vec` because the lengths arrive in
    /// no useful order and the count is small either way.
    plans: HashMap<usize, SpectrumPlan>,
}

impl PlanCache {
    /// A cache for a run with these fixed window and path parameters.
    #[must_use]
    pub fn new(dt: f32, q_exponent: f32, window_eps: f32, window_eta: f32) -> Self {
        Self {
            dt,
            q_exponent,
            window_eps,
            window_eta,
            plans: HashMap::new(),
        }
    }

    /// The plan for a subfault whose window is `window_s` seconds long.
    pub fn for_window(&mut self, window_s: f32) -> &SpectrumPlan {
        let np2 = SpectrumPlan::length_for(window_s, self.dt);
        self.for_length(np2)
    }

    /// The plan for an explicit transform length.
    ///
    /// Public so that a caller sizing scratch buffers can ask for the longest plan it will
    /// need before the subfault loop starts.
    pub fn for_length(&mut self, np2: usize) -> &SpectrumPlan {
        let (dt, q_exponent, window_eps, window_eta) =
            (self.dt, self.q_exponent, self.window_eps, self.window_eta);
        self.plans
            .entry(np2)
            .or_insert_with(|| SpectrumPlan::new(np2, dt, q_exponent, window_eps, window_eta))
    }

    /// How many distinct lengths have been built. For tests and profiling.
    #[must_use]
    pub fn len(&self) -> usize {
        self.plans.len()
    }

    /// Whether nothing has been built yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.plans.is_empty()
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
/// The deterministic half of one subfault's spectrum, plus the scratch its phase
/// realisations are drawn into.
///
/// # Why the shape is a value rather than a step
///
/// Nothing in the envelope or the amplitude depends on the random phase, and for the **two
/// horizontals nothing in them differs at all**: `f_max` is the only per-component input to
/// either, and only the vertical caps it. Both were nevertheless rebuilt for each of the three
/// components of every subfault and every ray. Building them once and drawing three phase
/// realisations against them is the same arithmetic in the same order.
///
/// # Why it owns its buffers
///
/// The three vectors were allocated and freed **per call**, so at a realistic `np2` of ~144,000
/// a station churned tens of gigabytes through the allocator to hold numbers it overwrote
/// immediately. They are sized once for the longest transform a segment will use and then used
/// a prefix at a time, which is what every other buffer on this path already does.
///
/// The tail beyond the current prefix holds the previous subfault's numbers. Every read below
/// goes through a slice of the current length, which is what makes the stale tail unreachable
/// rather than merely unread.
pub struct SpectrumShape {
    /// The Saragoni–Hart envelope on the sample grid, `w(t) = a·t^b·e^(−ct)`.
    envelope: Vec<f32>,
    /// Boore (1983) eq. 1's amplitude spectrum.
    ///
    /// **Bin 0 is never written and must stay zero.** It is zeroed at construction and the
    /// fill below starts at 1, which is what leaves DC at zero in the finished spectrum.
    amplitude: Vec<f64>,
    /// The normal deviates one realisation draws, held so the draw does not allocate.
    deviates: Vec<f32>,
    /// Sample interval, carried from the last [`SpectrumShape::refresh`] because the phase
    /// realisation needs it too.
    dt: f32,
}

impl SpectrumShape {
    /// Buffers for transforms of up to `np2_max` points.
    #[must_use]
    pub fn with_capacity(np2_max: usize) -> Self {
        Self {
            envelope: vec![0.0; np2_max],
            amplitude: vec![0.0; np2_max],
            deviates: vec![0.0; np2_max],
            dt: 0.0,
        }
    }

    /// Rebuild the envelope and the amplitude for one `(subfault, ray, f_max)`.
    ///
    /// See [`stochastic_spectrum`] for the physics; everything here was its first half.
    pub fn refresh(&mut self, plan: &SpectrumPlan, model: &SourceModel, path: &RayPath) {
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
        let mut decay = 1.0f64;
        for (slot, &power) in self.envelope[..np2].iter_mut().zip(envelope_power.iter()) {
            *slot = aa * power * decay as f32;
            decay *= decay_per_sample; // exp(0) at the first sample, so this advances after
        }

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
        azip!((
            shape in ArrayViewMut1::from(&mut self.amplitude[1..fold_count]),
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

        self.dt = dt;
    }
}

/// One phase realisation against a prepared [`SpectrumShape`].
/// (orig. `hb_high_ref.f:1670`)
///
/// Draws `np2` normal deviates, windows them with the envelope, transforms, normalises the
/// realised power to unity and applies the amplitude — which is the half of Boore (1983) eq. 1
/// that a seed changes. The deterministic half is [`SpectrumShape::refresh`], and the doc there
/// says why the two are separate.
///
/// `spectrum` must be contiguous and `np2` long — the transform is in place. The caller owns it
/// so the three components can be one `(3, np2)` block rather than three allocations per
/// subfault per ray.
///
/// # `np2` IS THE DRAW COUNT
///
/// Exactly `np2` deviates, which is why [`SpectrumPlan::length_for`] is physics rather than a
/// buffer size. See the note there.
pub fn stochastic_spectrum(
    rng: &mut impl Draws,
    plan: &SpectrumPlan,
    shape: &mut SpectrumShape,
    mut spectrum: ArrayViewMut1<Complex32>,
) {
    let (np2, fold_count) = (plan.np2, plan.fold_count);
    let dt = shape.dt;

    // The random phase spectrum. `remove_quadratic_trend` removes the quadratic acceleration
    // trend so that final velocity and displacement come out at zero.
    let deviates = &mut shape.deviates[..np2];
    rng.fill_normal(deviates);
    remove_quadratic_trend(dt, deviates);

    // Windowed noise: the envelope times the deviates, as the real part of the signal,
    // written straight into the caller's storage.
    azip!((
        bin in &mut spectrum,
        &deviate in ArrayView1::from(&deviates[..]),
        &envelope in ArrayView1::from(&shape.envelope[..np2]),
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
        &amplitude in ArrayView1::from(&shape.amplitude[..fold_count]),
    ) {
        let d = Complex64::new(bin.re as f64, bin.im as f64) * amplitude * amp as f64;
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
    use super::PlanCache;
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

    /// A cache built for a run's parameters.
    fn cache() -> PlanCache {
        PlanCache::new(0.005, 0.6, 0.2, 0.05)
    }

    /// A shorter window gets a shorter transform. This is the whole point of the change, so
    /// it is asserted rather than assumed.
    #[test]
    fn a_shorter_window_gets_a_shorter_transform() {
        let mut plans = cache();
        let short = plans.for_window(2.0).np2;
        let long = plans.for_window(200.0).np2;
        assert!(
            short < long,
            "a 2 s window got {short} points and a 200 s window {long}"
        );
        // `long` is what the segment maximum used to impose on EVERY subfault, including
        // the 2 s one. That ratio is the waste this change removes.
        assert_eq!((short, long), (896, 81_920));
    }

    /// The cache returns one plan per length, not one per request.
    ///
    /// If this regresses, every subfault rebuilds `np2` `powf` calls and the tables are no
    /// longer hoisted out of the hot path at all.
    #[test]
    fn the_cache_builds_one_plan_per_distinct_length() {
        let mut plans = cache();
        // Windows chosen to straddle the 7-smooth ladder: many of them, few lengths.
        for step in 0..400 {
            plans.for_window(10.0 + step as f32 * 0.01);
        }
        assert!(
            plans.len() < 20,
            "400 nearby windows produced {} plans",
            plans.len()
        );
        // And the same window twice is free.
        let before = plans.len();
        plans.for_window(10.0);
        assert_eq!(plans.len(), before, "a repeated window built a second plan");
    }

    /// Every plan's tables agree with its length, whichever way it was asked for.
    #[test]
    fn a_plans_tables_match_its_length() {
        let mut plans = cache();
        for window_s in [0.001, 0.5, 3.0, 40.0, 175.0, 400.0] {
            let plan = plans.for_window(window_s);
            assert_eq!(plan.fold_count, plan.np2 / 2 + 1);
            assert_eq!(plan.frequency_hz.len(), plan.fold_count);
            assert_eq!(plan.log_frequency_hz.len(), plan.fold_count);
            assert_eq!(plan.path_exponent.len(), plan.fold_count);
            assert_eq!(plan.envelope_power.len(), plan.np2);
            assert_eq!(plan.np2 % 2, 0, "np2 must be even for the Hermitian mirror");
        }
    }

    /// A window so short it rounds to nothing still gives an indexable transform.
    ///
    /// Reachable: `window_s` comes from a corner frequency and a path duration, and a
    /// subfault directly beneath a station has both small.
    #[test]
    fn a_degenerate_window_still_gives_a_usable_plan() {
        let mut plans = cache();
        let plan = plans.for_window(0.0);
        assert_eq!(plan.np2, 2);
        assert_eq!(plan.fold_count, 2);
    }
}
