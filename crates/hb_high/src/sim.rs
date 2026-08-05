//! Simulating one station.
//!
//! This is the physics, separated from the deck that used to drive it and from the
//! file the result used to be written to. [`simulate`] takes a [`HfConfig`], a slip
//! model, a velocity model and one station, and returns samples.
//!
//! # One station per call
//!
//! The Fortran loops over `nsite` stations sharing a single generator: `uniform_deviates` fills
//! the `vertical_radiation_spectrum` uniforms once before the loop, and each station's
//! `normal_deviates` draw continues from wherever the previous station left the
//! stream. A multi-station run is therefore **not** a concatenation of
//! single-station runs.
//!
//! Production never relies on that — `hf_sim.py` runs one process per station — so
//! this takes one station and seeds from `config.seed`, which reproduces the
//! `nsite = 1` case exactly. The driver refuses anything else rather than silently
//! computing something different.
//!
//! # Cost note
//!
//! Each call re-does the slip-model normalisation and the air-layer insertion, both
//! of which are station-independent. That is deliberate for now: it keeps the
//! signature honest about what it needs. A `Simulator` type holding the prepared
//! model and the reusable `mmv`-sized buffers is the obvious next step once the
//! Python wrapper starts looping over stations.

use crate::config::{
    HfConfig, PathDurationModel, RayKind, RuptureVelocityTaper, StressParamAdjust,
};
use crate::fort::{truncate_toward_zero, Complex32};
use crate::geom::{subfault_geometry, GeoPoint, SubfaultGeometry};
use crate::highcor::apply_radiation_and_invert;
use crate::input::{insert_air_layer, Segment, StochModel};
use crate::radiation::{horizontal_radiation_spectrum, vertical_radiation_spectrum};
use crate::ray::green_function;
use crate::rng::{fill_normal_deviates, fill_uniform_deviates, Pcg32};
use crate::site::{site_amplification_factors, apply_site_amplification};
use crate::state::{params, RayState, VelocityModel, VelocityModelInput, WaveMode};
use crate::stoc::stochastic_spectrum;

/// `mm` and `mmv` as the **main program** sees them.
///
/// Under `VERSION1` the main program includes `params_no_window.h`, so both are
/// 262144 — not the 32769/180000 that every subroutine gets from `params.h`.
/// This matters here because `ndata` is clamped to `mmv` and
/// `fill_normal_deviates(mmv, ...)` draws exactly this many deviates.
const MMV: usize = params::MMV;

/// The three output components, in the order the Fortran computes them.
///
/// **The order is load-bearing and this enum does not make it safe to change.** The two
/// horizontals each draw 5,000 deviates from the shared stream inside
/// `horizontal_radiation_spectrum`; the vertical draws none, reading the uniforms filled
/// once before the station loop. Reordering these three, or iterating them in anything
/// that does not preserve declaration order, moves every waveform. `REFACTOR.md`'s Tier D
/// finding turns on exactly that asymmetry.
///
/// `REFACTOR.md` §1.3b called for this and argued against the alternative: a trait would
/// have to smuggle the horizontal/vertical difference through an associated type and
/// would read worse. **Prefer the enum**, and match on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    /// `090` — east, the Fortran's first.
    E090,
    /// `000` — north.
    N000,
    /// The vertical. Capped at 15 Hz where the horizontals are not.
    Vertical,
}

impl Component {
    /// Declaration order, which is stream order. Iterate this, never a collection built
    /// some other way.
    const ALL: [Component; 3] = [Self::E090, Self::N000, Self::Vertical];

    /// Index into the three-element per-component arrays.
    #[inline]
    fn index(self) -> usize {
        match self {
            Self::E090 => 0,
            Self::N000 => 1,
            Self::Vertical => 2,
        }
    }

    /// Horizontal projection angle, in degrees. `None` for the vertical, which needs no
    /// projection — and that `None` is what selects the other radiation routine.
    #[inline]
    fn azimuth_offset_deg(self) -> Option<f32> {
        match self {
            Self::E090 => Some(-90.0),
            Self::N000 => Some(0.0),
            Self::Vertical => None,
        }
    }

    /// `fmax`, capped at 15 Hz for the vertical only. The Fortran writes this as
    /// `if (fmx1 > 15.0 .and. kf == 3) fmx1 = 15.0`, where `kf` is simultaneously a
    /// 1-based array index and a behaviour flag.
    #[inline]
    fn capped_fmax(self, fmax_hz: f32) -> f32 {
        match self {
            Self::Vertical => fmax_hz.min(15.0),
            _ => fmax_hz,
        }
    }
}

/// One station's synthetic record.
pub struct Simulation {
    /// Samples per component.
    pub ndata: usize,
    pub dt: f32,
    /// Ground motion, **interleaved** 090/000/ver — `ndata * 3` values, component
    /// fastest, which is the order the Fortran streams to disk.
    pub acc: Vec<f32>,
    /// `d10` — the closest subfault distance, km. The Fortran writes this to stderr
    /// as `(1x,f10.4)` and `hf_sim.py` parses it back; returning it instead makes
    /// the text format the driver's problem.
    ///
    /// Note it is re-initialised per fault segment in the original, so on a
    /// multi-segment model this describes the **last** segment only. Reproduced.
    pub d10_km: f32,
}

/// Why a simulation could not be produced.
#[derive(Debug)]
pub enum SimError {
    /// Segment dimensions disagree, which the Fortran refuses.
    InconsistentSegments(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::InconsistentSegments(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for SimError {}

/// Simulate one station.
pub fn simulate(
    config: &HfConfig,
    slip: &StochModel,
    vmod_in: &VelocityModelInput,
    j0_in: usize,
    station: crate::input::Station,
) -> Result<Simulation, SimError> {
    let (deg_to_rad, pi) = (crate::config::DEG_TO_RAD, crate::config::PI);

    let tw_eps = 0.2f32; // 0.4 first, then 0.2
    let tw_eta = 0.05f32;

    let nr = 1000usize;
    let nsfac = 20usize;

    // Site-amplification frequency table, log-transformed in place (:196-218).
    let fn_hz: [f32; 20] = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 0.70,
        1.00, 2.00, 3.00, 5.00, 7.00, 10.00, 20.00, 30.00, 50.00, 70.00,
    ];
    // `nsfac` entries, not `NLAYMAX` = 500. The Fortran declared this and `siteamp_factors`
    // over the layer ceiling because they sat in a common block sized for the velocity
    // model, but both are indexed `0..nsfac` -- a frequency table, not a layer table.
    let siteamp_log_freq: Vec<f32> = fn_hz[..nsfac].iter().map(|hz| hz.ln()).collect();

    // Resolved-default accessors are called once, here; the body below then reads
    // under the Fortran's names as the transliteration it still largely is.
    let czero = config.czero();
    let calpha = config.calpha();
    let fcfac = config.fcfac();
    let rvfmax = crate::config::defaults::RVFMAX;
    let (duration, dt, fmx, akapp, qfexp) =
        (config.duration, config.dt, config.fmax, config.kappa, config.qfexp);
    let rvsig1 = config.rv_sig1;
    let mut stress_average = config.stress_drop;
    let mut irand = config.seed;

    // The slip model is normalised in place and the velocity model gains an air
    // layer, so both are worked on as copies.
    let mut stoch = slip.clone();
    let mut vmod_in = vmod_in.clone();
    let mut j0 = j0_in;
    let mut nlskip = config.nl_skip;

    // Every segment must agree with the FIRST on subfault size, so the first is the
    // reference and the rest are the candidates -- `split_first` says that, where
    // re-indexing `segments[0]` inside a loop over `segments` left it to the reader to
    // notice the index was constant. The reported numbers stay 1-based, matching the
    // Fortran's message.
    if let Some((reference, rest)) = stoch.segments.split_first() {
        for (k, s) in rest.iter().enumerate() {
            if s.subfault_length_km != reference.subfault_length_km {
                return Err(SimError::InconsistentSegments(format!(
                    "dx({}) = {} not equal to dx(1) = {}, exiting...",
                    k + 2, s.subfault_length_km, reference.subfault_length_km
                )));
            }
            if s.subfault_width_km != reference.subfault_width_km {
                return Err(SimError::InconsistentSegments(format!(
                    "dw({}) = {} not equal to dw(1) = {}, exiting...",
                    k + 2, s.subfault_width_km, reference.subfault_width_km
                )));
            }
        }
    }

    // ------------------------------------------------- path duration model ---
    let path_duration = path_duration_table(config.path_duration);

    // ------------------------------------------------------- air layer -------
    let (j0_air, nlskip_air) = insert_air_layer(&mut vmod_in, j0, nlskip);
    j0 = j0_air;
    nlskip = nlskip_air;

    // Resolved here rather than at parse time: the deep transition depths depend on
    // the deepest hypocentre, which is only known once the slip model is read.
    let rv = config.rupture_velocity.resolve(stoch.max_hypocentre_depth_km);

    // ------------------------------------------------ source normalisation ---
    let SourceScale { dlm, sm, subfault_count } =
        normalise_source(&mut stoch, j0, &vmod_in, deg_to_rad, config.moment);

    // ---------------------------------------- stress parameter adjustment ----
    let targ_mag = config
        .target_magnitude
        .unwrap_or_else(|| 2.0 * (sm.ln() / 10.0f32.ln()) / 3.0 - 10.7);
    let fault_area = config.fault_area.unwrap_or(stoch.fault_area_km2);
    let mut spar_fac = match config.stress_param_adjust {
        // Leonard (2010), active tectonic.
        StressParamAdjust::LeonardActive => ((targ_mag - 3.99) * 10.0f32.ln()).exp() / fault_area,
        // Leonard (2010), stable continent.
        StressParamAdjust::LeonardStable => ((targ_mag - 4.19) * 10.0f32.ln()).exp() / fault_area,
        StressParamAdjust::None => 1.0,
    };
    spar_fac = spar_fac.sqrt();
    stress_average *= spar_fac;

    // Seismic moment of the subevent, from the average stress on the fault.
    let subevent_moment = stress_average * dlm * dlm * dlm * 1.0e+21;
    // The Fortran's `nsum` -- the sub-event count -- is computed from `ratio` and then
    // forced to 1 (2004-04-20), which is why the Frankel operator in
    // stochastic_spectrum carries the scaling instead. It is not bound here because
    // nothing reads it; the one place it reached is documented at its use site in the
    // subfault pass, where the draw it used to gate still has to happen.

    // The Fortran computes four candidate moment scalings in a row and lets the
    // last assignment win, leaving the other three as documentation of what was
    // tried. Reproduced with the names attached to their formulae rather than to
    // their order, and only the surviving one bound.
    //
    //   by_count      sm / (subevent_moment * subfault_count)          -- linear in subfault count
    //   by_sqrt_count sm / (subevent_moment * sqrt(subfault_count))    -- THE LIVE ONE
    //   by_two_thirds (sm/subevent_moment)^(2/3)
    //   by_corner_sq  (fce_avg / fcmain)^2
    //
    // `1.0 *` in by_sqrt_count is the Fortran's, and it matters: it forces the
    // integer subfault_count through a real multiply before the sqrt.
    let moment_scale = sm / (subevent_moment * (1.0 * subfault_count as f32).sqrt());

    // ------------------------------------------------------------ stations ---
    // No ceiling. The Fortran clamped this to `mmv`, which SILENTLY TRUNCATED a record
    // longer than the compiled array rather than reporting anything -- arguably worse
    // than the `np2 > mm` abort below it, which at least said something. Both are gone;
    // the buffers are sized from the deck.
    let ndata = truncate_toward_zero(duration / dt) as usize;

    let (mut rng, deviates) = seed_and_predraw(config, irand, nr, config.draws_normal_deviates());
    // init_random_seed mutates its argument, and the mutated value gates the
    // rupture-time jitter below.
    irand = deviates.seeded_irand;

    let mut vmod = VelocityModel::new();
    // `ndata` samples, not `mmv`: the output loop reads `1..=ndata` and nothing else
    // touches this. Three component traces, not a 2-D array -- the Fortran's `DS(3, mmv)`
    // was a 2-D block because Fortran had no better option.
    let mut acc: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; ndata]);

    // Everything the two passes need that is constant for the whole run, gathered once
    // so the extracted functions take one reference instead of eighteen scalars.
    let run = RunScalars {
        dt, fmax_hz: fmx, kappa_s: akapp, q_exponent: qfexp,
        pi, deg_to_rad,
        window_eps: tw_eps, window_eta: tw_eta,
        corner_const: czero * (1.0 + fcfac),
        calpha,
        rvfmax, rv_sig1: rvsig1,
        avg_subfault_km: dlm,
        subevent_moment, moment_scale,
        radv_sample_count: nr,
        site_table_len: nsfac,
        ndata,
        layer_count: j0,
        jitter_enabled: irand > 0,
    };

    // ------------------------------------------------- the single station ---
    // Only read after the segment loop, and only meaningful if there was one: a
    // zero-segment model returns this sentinel, which `main` then prints as a distance.
    let mut d10 = 1000.0f32;

    // A non-negative `nl_skip` would route the model through `grandvel`, the
    // velocity-model perturbation, which is dead under the production deck and not
    // ported. Asserted rather than branched on, so the live path is not an `else`.
    assert!(nlskip < 0, "grandvel is dead under the production deck (nl_skip < 0)");
    for k in 0..j0 {
        vmod[k] = vmod_in[k].into();
    }

    for seg in &stoch.segments {
        let angles = SegmentAngles::for_segment(seg, run.calpha, run.corner_const, deg_to_rad);

        let geom = subfault_geometry(
            GeoPoint { lat_deg: seg.fault_lat_deg, lon_deg: seg.fault_lon_deg },
            GeoPoint { lat_deg: station.stlat, lon_deg: station.stlon },
            seg.strike_deg, seg.dip_deg, seg.top_depth_km, seg.along_strike_offset_km,
            seg.subfault_length_km, seg.subfault_width_km,
            seg.along_strike_count, seg.down_dip_count,
        );

        let windows = time_window_pass(seg, &geom, &vmod, &rv, &path_duration, &angles, &run);
        // Re-initialised per segment, which is why the stderr distance reports only the
        // LAST segment on a multi-segment model. Reproduced.
        d10 = windows.d10_km;

        let plan = plan_segment_spectrum(windows.tmax, dt);

        subfault_pass(
            &mut rng, &mut acc, seg, &geom, &windows, &plan, &vmod, &rv, &angles, &run,
            config, &deviates, &siteamp_log_freq,
        );
    }

    // filter3d is not reachable from here at all: the switch that would enable it
    // (`ift`) is a driver-level deck field, and it is dead under production. The
    // driver refuses a non-zero value rather than passing it through.

    // The Fortran scans all three components for their peak amplitude here. Under
    // BINMOD nothing writes the result -- the only consumer is the commented-out
    // `!WRITE(6,*) 'ACC.MAX='` at hb_high_ref.f:1423 -- so the scan is dropped.
    // Interleaved, component fastest: 090/000/ver per time sample.
    // Interleaved, component fastest: 090/000/ver per time sample. Destructuring the
    // three traces first is what lets this be a zip -- `acc[c][sample]` over a loop nest
    // says the same thing while hiding that the three are walked in lockstep.
    let [e090, n000, vertical] = &acc;
    let out: Vec<f32> = e090
        .iter()
        .zip(n000)
        .zip(vertical)
        .flat_map(|((&e, &n), &v)| [e, n, v])
        .collect();
    debug_assert_eq!(out.len(), ndata * 3);

    Ok(Simulation { ndata, dt, acc: out, d10_km: d10 })
}


// ---------------------------------------------------------------------------
// The pieces `simulate` is made of
// ---------------------------------------------------------------------------

/// Everything derived from the deck and the slip model that is constant for the whole
/// run.
///
/// This exists because the alternative is an eighteen-argument function. Grouping does
/// not make the coupling smaller, but it does put every one of these under a name with a
/// unit, in one place, instead of spread across a 500-line body.
struct RunScalars {
    dt: f32,
    fmax_hz: f32,
    kappa_s: f32,
    q_exponent: f32,
    /// The source's own pi, and degrees-to-radians derived from it.
    pi: f32,
    deg_to_rad: f32,
    /// `tw_eps` / `tw_eta` — the Saragoni-Hart window shape.
    window_eps: f32,
    window_eta: f32,
    /// `czero * (1 + fcfac)` — the numerator of the corner-frequency coefficient.
    corner_const: f32,
    calpha: f32,
    /// Ceiling on the perturbed rupture-velocity factor.
    rvfmax: f32,
    /// Rupture-velocity randomisation sigma. Zero disables the perturbation *and* its
    /// deviate consumption.
    rv_sig1: f32,
    /// `dlm` — average subfault dimension, km.
    avg_subfault_km: f32,
    subevent_moment: f32,
    moment_scale: f32,
    /// `nr` — sample count for the conical radiation average, and a DRAW COUNT.
    radv_sample_count: usize,
    /// `nsfac` — length of the site-amplification frequency table.
    site_table_len: usize,
    ndata: usize,
    /// `j0` — layer count. Read as an INDEX where a source is below the model; see
    /// `PORTING_RULES.md` §7.
    layer_count: usize,
    /// Whether the rupture-time jitter draw happens.
    ///
    /// The Fortran tests `irand > 0` on the seed AFTER `init_random_seed` advanced it by
    /// `SEED_WORDS = 8`, so this is really `config.seed > -8`. Deciding it once, at the
    /// seeding site, keeps that eight-off comparison from looking like a seed test at the
    /// point of use.
    jitter_enabled: bool,
}

/// Per-segment angles, plus the one quantity the Fortran recomputes per subfault and
/// needn't.
struct SegmentAngles {
    strike_rad: f32,
    dip_rad: f32,
    rake_rad: f32,
    /// `czero * (1 + fcfac) / alphaT`.
    ///
    /// **Hoisted.** The Fortran evaluates `alphaT` inside BOTH subfault loops, from three
    /// per-segment constants — a sine, a square root and four arithmetic ops per subfault,
    /// producing the same value every time. Bit-identical to compute it once.
    corner_coeff: f32,
}

impl SegmentAngles {
    fn for_segment(seg: &Segment, calpha: f32, corner_const: f32, deg_to_rad: f32) -> Self {
        Self {
            strike_rad: seg.strike_deg * deg_to_rad,
            dip_rad: seg.dip_deg * deg_to_rad,
            rake_rad: seg.rake_deg * deg_to_rad,
            corner_coeff: corner_const / alpha_t(seg.dip_deg, seg.rake_deg, calpha),
        }
    }
}

/// The generator and the three pre-drawn blocks, in the order the Fortran draws them.
struct Deviates {
    /// `irand` after `init_random_seed` mutated it.
    seeded_irand: i32,
    /// `fgrand` — one block of `MMV`, indexed by `irandcnt`.
    normal: Vec<f32>,
    /// `rna` / `rnb` — the vertical component's uniforms.
    radv_uniform_a: Vec<f32>,
    radv_uniform_b: Vec<f32>,
}

/// Seed, then make the three pre-draws.
///
/// **The order and the counts are the contract, not an implementation detail.** Seed,
/// then `nr` uniforms into `a`, then `nr` uniforms into `b`, then `MMV` normals. Changing
/// any of the three moves every sample downstream, which is why this is one function
/// rather than three calls spread through the setup. See `REFACTOR.md` §2.6b.
///
/// The two uniform blocks must stay two sequential fills. Interleaving them into one pass
/// would put different deviates in different slots and change every vertical component.
fn seed_and_predraw(
    config: &HfConfig,
    irand: i32,
    radv_sample_count: usize,
    draw_normals: bool,
) -> (Pcg32, Deviates) {
    let (mut rng, seeded_irand) = Pcg32::seed(irand);

    // `nr` values, not `mmv`. `vertical_radiation_spectrum` reads exactly this many, and
    // the Fortran reserved and zeroed 262144 to use 1000 of them.
    let mut radv_uniform_a = vec![0.0f32; radv_sample_count];
    let mut radv_uniform_b = vec![0.0f32; radv_sample_count];
    fill_uniform_deviates(&mut rng, radv_sample_count, &mut radv_uniform_a);
    fill_uniform_deviates(&mut rng, radv_sample_count, &mut radv_uniform_b);

    // This one STAYS at `MMV`, and not from laziness: the number drawn is part of the
    // stream. Shrinking the allocation without shrinking the draw is a buffer overrun;
    // shrinking the draw changes every waveform. See REFACTOR.md §2.6b.
    let mut normal = vec![0.0f32; MMV];
    if draw_normals {
        fill_normal_deviates(&mut rng, MMV, &mut normal);
    }
    let _ = config;

    (rng, Deviates { seeded_irand, normal, radv_uniform_a, radv_uniform_b })
}

/// What the time-window pass produces for one segment.
struct WindowPass {
    /// One entry per real subfault, indexed by [`Segment::grid_index`].
    window_s: Vec<f32>,
    /// Longest window over the segment; sizes the transform.
    tmax: f32,
    /// `d10` — closest subfault slant distance, km.
    d10_km: f32,
}

/// Time-window pass — `hb_high_ref.f`'s first subfault loop.
///
/// **Depth-major: `j` outer, `i` inner.** That is the OPPOSITE order to
/// [`subfault_pass`], and the difference is load-bearing there, not here: this pass draws
/// nothing from the generator, so its order affects only the `tmax` max-reduction and the
/// `d10` min-reduction, both of which are order-independent in exact arithmetic and
/// preserved here anyway.
fn time_window_pass(
    seg: &Segment,
    geom: &SubfaultGeometry,
    vmod: &VelocityModel,
    rupture: &RuptureVelocityTaper,
    path_duration: &PathDuration,
    angles: &SegmentAngles,
    run: &RunScalars,
) -> WindowPass {
    let mut tmax = 0.0f32;
    let mut d10_km = 10000.0f32;
    let mut window_s = vec![0.0f32; seg.subfault_total()];

    // Carried ACROSS subfaults deliberately. Unlike the subfault pass, this one has no
    // `vsh(1)` default before the lookup: if the subfault depth exceeds every layer,
    // the previous subfault's velocity is reused. Undefined on the very first subfault
    // of the first station in the Fortran; zero here.
    let mut shear_velocity_km_s = 0.0f32;

    for (i, j) in seg.depth_major() {
        let ray = geom.at(i, j);
        if let Some(ksrc) =
            (0..run.layer_count).find(|&k| vmod[k].depth_km >= ray.depth_km as f64)
        {
            shear_velocity_km_s = vmod[ksrc].vsh_km_s as f32;
        }

        let rvf = rupture.factor(ray.depth_km);

        // The last table segment this distance is past. The Fortran scans the whole
        // table letting later matches overwrite earlier ones, which is `.last()` -- NOT
        // `.find()`, since the table ascends and the first match is the wrong end.
        //
        // Strict `>`, so a distance of exactly the first breakpoint (0.0) matches
        // NOTHING and the duration terms stay zero. The Fortran leaves them undefined
        // there; zero is this port's choice, and `unwrap_or` is where it lives.
        let bin = path_duration
            .iter()
            .take_while(|s| ray.slant_km > s.start_km)
            .last()
            .copied()
            .unwrap_or(DurationSegment { start_km: 0.0, duration_s: 0.0, slope_s_per_km: 0.0 });

        let fce = angles.corner_coeff * rvf * shear_velocity_km_s
            / (run.avg_subfault_km * run.pi);
        let tw0 = run.moment_scale.sqrt() * (1.0 / fce);
        let dpath = bin.duration_s + bin.slope_s_per_km * (ray.slant_km - bin.start_km);
        // VERSION1: no 81.92 s cap.
        let window = 2.12 * (tw0 + dpath);
        window_s[seg.grid_index(i, j)] = window;

        if window > tmax {
            tmax = window;
        }
        d10_km = d10_km.min(ray.slant_km);
    }

    WindowPass { window_s, tmax, d10_km }
}

/// Transform length and frequency axis for one segment.
struct SpectrumPlan {
    np2: usize,
    /// `nfold` — positive-frequency bin count, `np2/2 + 1`.
    fold_count: usize,
    /// `mfold` — mirrored bin count, `np2/2 - 1`.
    mirror_count: usize,
    frequency_hz: Vec<f32>,
}

/// Smallest power of two at or above `2 * tmax / dt`, and the axis that goes with it.
fn plan_segment_spectrum(tmax: f32, dt: f32) -> SpectrumPlan {
    let ntmax = truncate_toward_zero(2.0 * tmax / dt) as usize;
    let mut np2 = 2usize;
    while np2 < ntmax {
        np2 *= 2;
    }
    let fold_count = np2 / 2 + 1;
    let mirror_count = np2 / 2 - 1;

    let df = 1.0 / (np2 as f32 * dt);
    // 0-based, which also removes the `- 1`: the axis is `df * bin`.
    let frequency_hz: Vec<f32> = (0..fold_count).map(|bin| df * bin as f32).collect();

    SpectrumPlan { np2, fold_count, mirror_count, frequency_hz }
}

/// Subfault pass — `hb_high_ref.f`'s second subfault loop.
///
/// **Strike-major: `i` outer, `j` inner, the OPPOSITE of [`time_window_pass`], and here
/// the order IS the contract.** `irandcnt` advances once per surviving subfault and
/// indexes the pre-drawn normals, and every `stochastic_spectrum` and
/// `horizontal_radiation_spectrum` call draws from the live stream. Walking the grid the
/// other way pairs a different deviate with every subfault and changes every waveform.
/// See `PORTING_RULES.md` §5.
#[allow(clippy::too_many_arguments)]
fn subfault_pass(
    rng: &mut Pcg32,
    acc: &mut [Vec<f32>; 3],
    seg: &Segment,
    geom: &SubfaultGeometry,
    windows: &WindowPass,
    plan: &SpectrumPlan,
    vmod: &VelocityModel,
    rupture: &RuptureVelocityTaper,
    angles: &SegmentAngles,
    run: &RunScalars,
    config: &HfConfig,
    deviates: &Deviates,
    siteamp_log_freq: &[f32],
) {
    let np2 = plan.np2;
    let mut spectrum: [Vec<Complex32>; 3] =
        std::array::from_fn(|_| vec![Complex32::ZERO; np2]);
    let mut subfault_acc: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; np2]);
    let mut radiation = vec![0.0f32; plan.fold_count];
    let mut siteamp_factors = vec![0.0f32; run.site_table_len];
    let mut ray = RayState::default();

    // 0-based. The Fortran starts this at 1 and PRE-increments, so its first read is
    // index 2, i.e. storage element 1 -- element 0 is never read. Starting at 0 and
    // pre-incrementing lands on that same element.
    let mut irandcnt = 0usize;

    for (i, j) in seg.strike_major() {
        let subfault = seg.at(i, j);
        if subfault.slip < 0.001 {
            continue; // goto 4 lands on the inner loop's terminator
        }
        let ray_geometry = geom.at(i, j);
        let subfault_window_s = windows.window_s[seg.grid_index(i, j)];

        for component in &mut subfault_acc {
            component.fill(0.0);
        }

        // This pass DOES default the velocity and density before the lookup, unlike the
        // window pass, which carries the previous subfault's value.
        let mut shear_velocity_km_s = vmod[0].vsh_km_s as f32;
        let mut density_g_cm3 = vmod[0].density_g_cm3 as f32;
        let ksrc = match (0..run.layer_count)
            .find(|&k| vmod[k].depth_km >= ray_geometry.depth_km as f64)
        {
            Some(k) => {
                shear_velocity_km_s = vmod[k].vsh_km_s as f32;
                density_g_cm3 = vmod[k].density_g_cm3 as f32;
                k
            }
            // Keeps the first layer's defaults, and `layer_count` is one PAST the last
            // layer -- the Fortran's `j0 + 1`, read as such below and by
            // `site_amplification_factors`. See PORTING_RULES.md §7.
            None => run.layer_count,
        };
        if ksrc == run.layer_count {
            // The Fortran prints 'wrong!' and carries on with ksrc = j0+1.
            println!(" wrong!");
        }

        let base_rvf = rupture.factor(ray_geometry.depth_km);
        let mut rvf = base_rvf;
        if run.rv_sig1 > 0.0 {
            irandcnt += 1;
            rvf = base_rvf * (deviates.normal[irandcnt] * run.rv_sig1).exp();
            if rvf > run.rvfmax {
                rvf = run.rvfmax;
            }
        }

        let fce = angles.corner_coeff * rvf * shear_velocity_km_s / run.avg_subfault_km / run.pi;

        for &ray_type in &config.rayset {
            let kind = ray_type.kind();

            // The tracing runs even for a straight ray: the Fortran calls
            // green_function unconditionally and overwrites the results below, and
            // type 0 borrows type 1's tracing to do it.
            let g = green_function(
                &mut ray, vmod, run.layer_count, ray_geometry.depth_km,
                ray_geometry.horiz_km, ray_type.trace_type(), WaveMode::Sh,
            );
            let mut stime = g.stime;
            let mut rpath = g.rpath;
            let mut qbar = g.qbar;
            let mut sub_tstart = stime - run.window_eps * subfault_window_s;

            if kind == RayKind::StraightRay {
                rpath = ray_geometry.slant_km;
                qbar = rpath / (shear_velocity_km_s * 150.0);
                stime = rpath / 3.7;
                sub_tstart = 0.7 * stime;
            }

            // Three calls in component order: each draws `np2` normal deviates.
            for (component, spec) in Component::ALL.into_iter().zip(spectrum.iter_mut()) {
                stochastic_spectrum(
                    rng, np2, rpath, subfault_window_s, run.window_eps, run.window_eta,
                    shear_velocity_km_s, density_g_cm3, run.dt,
                    run.subevent_moment, run.avg_subfault_km, fce,
                    component.capped_fmax(run.fmax_hz), run.kappa_s,
                    spec, &plan.frequency_hz, qbar, run.q_exponent,
                    run.moment_scale,
                );
            }

            if config.site_amp {
                site_amplification_factors(
                    vmod, ksrc, run.site_table_len, siteamp_log_freq, &mut siteamp_factors,
                );
                for spec in &mut spectrum {
                    apply_site_amplification(
                        spec, &plan.frequency_hz, run.site_table_len,
                        siteamp_log_freq, &siteamp_factors,
                    );
                }
            }
            // famprand is dead: fasig1 = fasig2 = 0.

            // Incidence angle from the ray parameter: sin(i)/vs = p0.
            let p0 = g.rp0;
            let incidence = if shear_velocity_km_s * p0 > 1.0 {
                0.5 * run.pi
            } else {
                (shear_velocity_km_s * p0).asin()
            };
            let th = match kind {
                // The straight-ray approximation ignores the traced ray parameter and
                // uses the geometric take-off angle.
                RayKind::StraightRay => ray_geometry.takeoff_rad,
                RayKind::Upgoing => run.pi - incidence,
                RayKind::Downgoing => incidence,
            };
            let pa = ray_geometry.azimuth_rad;

            // The ONLY difference between the three components is which radiation
            // routine runs, and the `Option` carries it: `Some` is a horizontal, which
            // draws 5,000 deviates; `None` is the vertical, which draws none.
            for component in Component::ALL {
                match component.azimuth_offset_deg() {
                    Some(offset_deg) => horizontal_radiation_spectrum(
                        rng, angles.strike_rad, angles.dip_rad, angles.rake_rad, pa, th,
                        &plan.frequency_hz, plan.fold_count, offset_deg * run.deg_to_rad,
                        run.radv_sample_count, &mut radiation,
                    ),
                    None => vertical_radiation_spectrum(
                        angles.strike_rad, angles.dip_rad, angles.rake_rad, pa, th,
                        &plan.frequency_hz, plan.fold_count,
                        &deviates.radv_uniform_a, &deviates.radv_uniform_b,
                        run.radv_sample_count, &mut radiation,
                    ),
                };
                let k = component.index();
                apply_radiation_and_invert(
                    plan.fold_count, plan.mirror_count,
                    &mut spectrum[k], &mut subfault_acc[k], &radiation,
                );
            }

            // Rupture time at this subfault.
            let ratim = match config.rupture_velocity_override {
                Some(vr) => {
                    let along_strike_centre = 0.5 * (seg.along_strike_count as f32 + 1.0);
                    let xra = seg.hypocentre_along_strike_km
                        - (i as f32 - along_strike_centre) * seg.subfault_length_km;
                    let yra =
                        seg.hypocentre_down_dip_km - (j as f32 - 0.5) * seg.subfault_width_km;
                    let mut t = (xra * xra + yra * yra).sqrt() / vr;
                    if run.jitter_enabled {
                        t += (rng.next_f32() - 0.5) * 0.1 * t;
                    }
                    t
                }
                None => subfault.rupture_time_s,
            };

            // Both terms truncate TOWARD ZERO, not toward negative infinity, so a
            // negative `sub_tstart` makes `kst` smaller and possibly negative.
            let kst = truncate_toward_zero(ratim / run.dt)
                + truncate_toward_zero(sub_tstart / run.dt);

            // ONE DRAW, AND IT MUST STAY. The Fortran loops `k = 1, nsum` here, draws a
            // uniform, and turns it into a sub-event time offset. But `nsum` was frozen
            // at 1 in 2004, so the loop ran once and the next statement was
            // `if (nsum.eq.1) k2 = 0` -- the offset was computed and then thrown away.
            //
            // The arithmetic is gone. The draw is not: it advances the shared generator
            // once per (subfault, ray), and every sample after it depends on where the
            // stream ends up. Deleting this as "obviously dead code" changes every
            // waveform in the program.
            let _stream_advance = rng.next_f32();

            accumulate_subfault(acc, &subfault_acc, subfault.slip, kst, np2, run.ndata);
        }
    }
}

/// Place one subfault's `np2`-sample contribution into the station accumulator.
///
/// `start_sample` is the Fortran's 1-based sample number and **can be negative**:
/// `int()` truncates toward zero and `sub_tstart` can be negative. Samples landing
/// before sample 1 are discarded rather than written, matching the Fortran, whose output
/// only ever reads `DS(1..ndata)`.
///
/// §2.6 defect 1 lives here: sample 1 of the subfault's trace lands on `start_sample`,
/// not on `start_sample + 1`. The Fortran read `stdd(li - k2, l)`, so its first iteration
/// read index 0 -- one element before the column, which nothing writes -- and every
/// subfault's contribution arrived one sample late. See `REFACTOR.md` §2.6 for the
/// analysis and `PORTING_RULES.md` §7 for the aliasing that made that read return zero
/// rather than crash.
fn accumulate_subfault(
    acc: &mut [Vec<f32>; 3],
    subfault_acc: &[Vec<f32>; 3],
    weight: f32,
    start_sample: i32,
    np2: usize,
    ndata: usize,
) {
    // The contribution is `subfault_acc[0..np2]` placed at
    // `acc[start_sample ..= start_sample + np2 - 1]`, both clipped to the record.
    let last_sample = (start_sample + np2 as i32 - 1).min(ndata as i32);
    // Clip the low end to sample 1. `skip` is how many of the subfault's own samples fall
    // before the record starts.
    let first_sample = start_sample.max(1);

    // BOTH ends can put the window entirely outside the record, and they are different
    // cases: `last < 1` is a contribution that ends before the record begins (a large
    // negative `sub_tstart`), and `last < first` is one that begins after it ends
    // (`start_sample > ndata`, which a long-path ray at a far station reaches). The
    // Fortran's `do while (li <= kend)` covers both by simply not iterating. Written as
    // an explicit range, the second case computes a negative length and must be rejected
    // before it is cast — this is exactly what the parity gate caught when it was not.
    if last_sample < first_sample {
        return;
    }

    let skip = (first_sample - start_sample) as usize;
    let count = (last_sample - first_sample + 1) as usize;
    let dst = first_sample as usize - 1;

    for (out, contribution) in acc.iter_mut().zip(subfault_acc) {
        for (slot, &value) in
            out[dst..dst + count].iter_mut().zip(&contribution[skip..skip + count])
        {
            *slot += weight * value;
        }
    }
}

/// One segment of the piecewise-linear duration-versus-distance table.
///
/// The Fortran keeps these as three parallel `real dur(50)` arrays plus an `ndur` count.
/// Every read of one is at the same index as the other two, so this is one value per
/// segment — the same argument `Subfault` and `SubfaultRay` already make (§2.3).
#[derive(Clone, Copy, Debug, PartialEq)]
struct DurationSegment {
    /// `rdur` — distance at which this segment starts, km.
    start_km: f32,
    /// `dpth` — duration at that distance, s.
    duration_s: f32,
    /// `dpdr` — slope of this segment, s/km.
    slope_s_per_km: f32,
}

/// The path-duration model: a piecewise-linear duration-versus-distance table.
///
/// The `50` the Fortran sized its arrays at is gone with the parallel arrays. It was a
/// ceiling nothing enforced and the largest model uses 8 of it; `len()` is now the count,
/// so `ndur` is gone too — a length and a capacity can no longer disagree.
type PathDuration = Vec<DurationSegment>;

/// Build the path-duration table.
///
/// Total by construction now that the model is an enum: the Fortran's
/// undefined-`ndur` path is unrepresentable, and rejecting a bad integer happens
/// once, in `PathDurationModel::from_deck`.
fn path_duration_table(model: PathDurationModel) -> PathDuration {
    /// The single-segment models give their slope directly and have no breakpoints.
    fn constant_slope(slope_s_per_km: f32) -> PathDuration {
        vec![DurationSegment { start_km: 0.0, duration_s: 0.0, slope_s_per_km }]
    }

    /// The multi-segment models give (distance, duration) breakpoints; the slopes are
    /// differenced from consecutive pairs.
    ///
    /// The last segment repeats the previous slope so distances past the table
    /// extrapolate rather than flatten — which is why the fold looks one short and then
    /// pushes a copy.
    fn from_breakpoints<const N: usize>(start_km: [f32; N], duration_s: [f32; N]) -> PathDuration {
        let mut table: PathDuration = start_km
            .windows(2)
            .zip(duration_s.windows(2))
            .map(|(r, d)| DurationSegment {
                start_km: r[0],
                duration_s: d[0],
                slope_s_per_km: (d[1] - d[0]) / (r[1] - r[0]),
            })
            .collect();
        let last_slope = table.last().expect("a breakpoint table has at least two entries")
            .slope_s_per_km;
        table.push(DurationSegment {
            start_km: start_km[N - 1],
            duration_s: duration_s[N - 1],
            slope_s_per_km: last_slope,
        });
        table
    }

    match model {
        PathDurationModel::Gp2010 => constant_slope(0.063),
        PathDurationModel::Wus => constant_slope(0.07),
        PathDurationModel::Ena => constant_slope(0.1),
        // BT2014 WUS. The breakpoints at 7, 45, 125 and 175 km are what the Phase 2
        // distance ladder is chosen to straddle.
        PathDurationModel::Bt2014Wus => from_breakpoints(
            [0.0, 7.0, 45.0, 125.0, 175.0, 270.0],
            [0.0, 2.4, 8.4, 10.9, 17.4, 34.2],
        ),
        PathDurationModel::Bt2015Ena => from_breakpoints(
            [0.0, 15.0, 35.0, 50.0, 125.0, 200.0, 392.0, 600.0],
            [0.0, 2.6, 17.5, 25.1, 25.1, 28.5, 46.0, 69.1],
        ),
    }
}


/// Scalars derived from the slip model before any station is simulated.
struct SourceScale {
    /// Average subfault dimension `sqrt(length * width)`, averaged over segments, km.
    dlm: f32,
    /// Total seismic moment. Derived from the summed subfault moments when the
    /// deck asked for it with a negative value.
    sm: f32,
    /// Count of subfaults whose relative moment exceeds 0.001 — the same
    /// threshold the subfault pass uses to skip a subfault entirely.
    subfault_count: usize,
}

/// Convert relative slip to relative moment, then normalise to unit average
/// weight, mutating `stoch.segments[..].slip` in place.
///
/// Three passes over the subfault grid, in the Fortran's order:
///
/// 1. average subfault size, and the maximum absolute slip (computed and discarded)
/// 2. slip → moment via the rigidity `xmu`, accumulating `xsum`, `fce_avg`,
///    `trise_avg` and a count of *all* subfaults
/// 3. re-count only the subfaults above 0.001 and rescale so their mean weight is 1
///
/// The two counts are different and both matter: pass 2's count normalises the
/// averages, pass 3's is the one that reaches `moment_scale`. The Fortran shadows one
/// `subfault_count` with the other, so only the second survives — hence only that one is
/// returned.
#[allow(clippy::too_many_arguments)]
fn normalise_source(
    stoch: &mut StochModel,
    layer_count: usize,
    vmod_in: &VelocityModelInput,
    deg_to_rad: f32,
    moment: Option<f32>,
) -> SourceScale {
    let nevnt = stoch.segments.len();

    // --- pass 1: average subfault size ----------------------------------------
    // The Fortran also accumulates `slip_max` over every subfault here. Nothing
    // reads it: its only consumer is the commented-out `!print*,'Maximum slip '`
    // at hb_high_ref.f:683. Dropped, along with the O(subfault_count) loop that fed it.
    let mut dlm = 0.0f32;
    for s in &stoch.segments {
        dlm += (s.subfault_length_km * s.subfault_width_km).sqrt();
    }
    dlm /= nevnt as f32;

    // --- pass 2: relative slip to relative moment -----------------------------
    // The Fortran also accumulates `fce_avg` and `trise_avg` in this loop, doing a
    // rupture-velocity taper, an `alphaT` evaluation (with a sine and a square root)
    // and four more arithmetic ops PER SUBFAULT. All of it is dead:
    //
    //   fce_avg, trise_avg -> fcmain -> bigC3  (hb_high_ref.f:805)
    //
    // and `bigC` is assigned five times at :807-811 with `bigC = bigC1b` last, so
    // bigC3 never survives. Their only other consumers are the commented-out prints
    // at :684-685 and :817. Dropping them also removes `xnorm`, this pass's own
    // `subfault_count` count (which only normalised them), and `islip_weight_avg` -- a frozen
    // switch whose two branches only ever weighted these dead quantities.
    //
    // What remains live: the in-place slip-to-moment conversion, and `xsum`, which
    // supplies the total moment when the deck asks for it to be derived.
    let mut xsum = 0.0f32;

    for segment in &mut stoch.segments {
        let dwdj = segment.subfault_width_km * (segment.dip_deg * deg_to_rad).sin();
        let top_depth_km = segment.top_depth_km;
        // vsh_km_s and density_g_cm3 are real*8 and the Fortran's dx/dw real*4, so the
        // WHOLE product below is computed in double (dx/dw promoted) and narrows only on
        // assignment to xmu, which is implicit real*4. Narrowing earlier shifts every
        // subfault moment by an ulp or two.
        let (length_km, width_km) =
            (segment.subfault_length_km as f64, segment.subfault_width_km as f64);

        // Depth-major, which is storage order, so `xsum` accumulates in the Fortran's
        // order without any index arithmetic. Rigidity is per depth row, not per
        // subfault, which is why the row is the unit here.
        for (row_index, row) in segment.depth_rows_mut().enumerate() {
            let zdep = top_depth_km + (row_index as f32 + 0.5) * dwdj;
            // Layer lookup. Falls through with k = j0+1 if zdep is below the
            // model, which the Fortran then indexes -- so the fall-through is
            // load-bearing, not an error path.
            let k = (0..layer_count).find(|&kk| zdep <= vmod_in[kk].depth_km).unwrap_or(layer_count);
            let xmu = (vmod_in[k].vsh_km_s * vmod_in[k].vsh_km_s * vmod_in[k].density_g_cm3
                * length_km
                * width_km) as f32;

            for subfault in row {
                subfault.slip *= xmu;
                xsum += subfault.slip;
            }
        }
    }

    // `None` means the deck asked for the moment to be derived from the summed
    // subfault moments.
    let sm = moment.unwrap_or(1.0e+20 * xsum);

    // --- pass 3: normalise relative moments to average weight unity -----------
    let mut wsum = 0.0f32;
    let mut subfault_count = 0usize;
    for segment in &stoch.segments {
        // Depth-major again, so the slice order is the summation order.
        for subfault in segment.depth_rows().flatten() {
            if subfault.slip > 0.001 {
                wsum += subfault.slip;
                subfault_count += 1;
            }
        }
    }
    let scale = subfault_count as f32 / wsum;
    for segment in &mut stoch.segments {
        for subfault in segment.depth_rows_mut().flatten() {
            subfault.slip *= scale;
        }
    }

    SourceScale { dlm, sm, subfault_count }
}

/// The `alphaT` corner-frequency adjustment (2013-11-20).
///
/// Also appears three times identically. `fD` tapers with dip above 45 degrees;
/// `fR` peaks at a rake of 90 degrees. The rake is first wrapped into
/// `[-180, 180]` by repeated addition or subtraction of 360, which the Fortran
/// does with backward `goto`s.
fn alpha_t(avgdip: f32, rake_deg: f32, calpha: f32) -> f32 {
    let mut fd = 0.0f32;
    if avgdip <= 90.0 && avgdip > 45.0 {
        fd = 1.0 - (avgdip - 45.0) / 45.0;
    } else if (0.0..=45.0).contains(&avgdip) {
        fd = 1.0;
    }

    let mut avgrak = rake_deg;
    while avgrak < -180.0 {
        avgrak += 360.0;
    }
    while avgrak > 180.0 {
        avgrak -= 360.0;
    }

    let mut fr = 0.0f32;
    if (0.0..=180.0).contains(&avgrak) {
        // sqrt(x*x) rather than abs(x); the Fortran writes it this way.
        fr = 1.0 - ((avgrak - 90.0) * (avgrak - 90.0)).sqrt() / 90.0;
    }

    1.0 / (1.0 + fd * fr * calpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: the Fortran's own loop, transliterated.
    fn accumulate_reference(
        acc: &mut [Vec<f32>; 3],
        subfault_acc: &[Vec<f32>; 3],
        weight: f32,
        start_sample: i32,
        np2: usize,
        ndata: usize,
    ) {
        let kend = (start_sample + np2 as i32 - 1).min(ndata as i32);
        let mut li = start_sample;
        while li <= kend {
            if li >= 1 {
                let idx = (li - start_sample) as usize;
                let sample = li as usize - 1;
                for component in 0..3 {
                    acc[component][sample] += weight * subfault_acc[component][idx];
                }
            }
            li += 1;
        }
    }

    /// The slice form must agree with the loop form at every alignment, including the
    /// two that put the window entirely outside the record.
    ///
    /// `start_sample > ndata` is the case §2.8 got wrong: it computes a negative length,
    /// and casting that to `usize` wraps. The parity gate caught it on one deck
    /// (`rayset=1,3`, where the Moho multiple makes the path long enough to start past
    /// the end of the record); this pins it without needing a 22-deck run.
    #[test]
    fn accumulate_matches_the_fortran_loop_at_every_alignment() {
        let np2 = 8usize;
        let ndata = 10usize;
        let subfault_acc: [Vec<f32>; 3] =
            std::array::from_fn(|c| (0..np2).map(|i| (c * 100 + i + 1) as f32).collect());

        // Well before the record, straddling both edges, and well past the end.
        for start in -12i32..=14 {
            let mut got: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; ndata]);
            let mut want: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; ndata]);
            accumulate_subfault(&mut got, &subfault_acc, 2.0, start, np2, ndata);
            accumulate_reference(&mut want, &subfault_acc, 2.0, start, np2, ndata);
            assert_eq!(got, want, "start_sample = {start}");
        }
    }
}
