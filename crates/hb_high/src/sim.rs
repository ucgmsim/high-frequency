//! Simulating one station: the finite-fault layer.
//!
//! This module is **Graves & Pitarka (2010)**, "Broadband ground-motion simulation using a
//! hybrid approach", *BSSA* 100(5A), 2095–2123 — specifically its high-frequency module,
//! equations 10 through 17. [`crate::stoc`] builds one subfault's spectrum (Boore 1983); this
//! walks the rupture and sums them.
//!
//! ```text
//! A_i(f) = Σ_j  C_ij · S_i(f) · G_ij(f) · P(f)          eq. 10
//! ```
//!
//! The loops here are that sum: over subfaults `i`, over ray paths `j`, and over the three
//! output components. See `PHYSICS.md` §7 for the assembly, and `papers/README.md` for the
//! equation-by-equation verification.
//!
//! # One station per call
//!
//! **A multi-station run is not a concatenation of single-station runs**, at least not in the
//! original: it shared one generator across its station loop, so each station's draws continued
//! from wherever the previous one stopped. Here each station gets an independent stream seeded
//! from `config.seed`, which makes a batch safe to reorder, subset or resume.
//!
//! # Cost note
//!
//! Each call re-does the slip-model normalisation and the air-layer insertion, both of which are
//! station-independent. Deliberate for now — it keeps the signature honest about what it needs.
//! A `Simulator` type holding the prepared model and the reusable buffers is the obvious next
//! step once a caller starts looping over many stations.

use crate::config::{
    HfConfig, PathDurationModel, RayKind, RuptureVelocityTaper, StressParamAdjust,
};
use ndarray::{s, Array1, ArrayView1, ArrayViewMut1};

use crate::fft::Complex32;
use crate::geom::{subfault_geometry, FaultPlane, GeoPoint, SubfaultGeometry};
use crate::input::{insert_air_layer, Segment, StochModel};
use crate::radiation::{horizontal_radiation_spectrum, vertical_radiation_spectrum, RadiationAngles};
use crate::ray::green_function;
use crate::rng::{fill_uniform_deviates, normal_deviate, Draws, DrawSource};
use crate::site::{site_amplification_factors, apply_site_amplification};
use crate::state::{RayState, VelocityModel, VelocityModelInput, WaveMode};
use crate::stoc::{radiate_and_invert, stochastic_spectrum, RayPath, SourceModel, SpectrumPlan};

/// The three output components, in the order they are computed — which is also the order a
/// caller receives them in.
///
/// **The order is load-bearing, and this enum does not make it safe to change.** The two
/// horizontals each draw 5,000 deviates from the shared stream inside
/// [`crate::radiation::horizontal_radiation_spectrum`]; the vertical draws none, reading a
/// pre-filled table instead. So the three are *not* interchangeable positions in a loop:
/// reordering them, or iterating them in anything that does not preserve declaration order,
/// moves every waveform. See `PHYSICS.md` §9.
///
/// An enum rather than a trait deliberately — a trait would have to smuggle the
/// horizontal/vertical difference through an associated type and would read worse. Match on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    /// `090` — east, and the first computed.
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

    /// `f_max`, capped at 15 Hz for the vertical only.
    ///
    /// The cap is empirical: vertical-component spectra fall off from a lower corner than the
    /// horizontals do. It is the one place the component identity changes the *physics* rather
    /// than just which radiation routine runs.
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
    /// Ground motion, **interleaved** 090/000/ver — `ndata * 3` values, component fastest.
    /// Callers consume the three channels positionally, so the order is part of the interface.
    pub acc: Vec<f32>,
}

/// Why a simulation could not be produced.
#[derive(Debug)]
pub enum SimError {
    /// Segment dimensions disagree. Every segment must share the first one's subfault size,
    /// because `dl` is a single per-run quantity in the corner-frequency and duration models.
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

    // Boore (1983, p. 1869)'s own envelope-shape values, and the ones Graves & Pitarka (2010)
    // use: the peak sits at 0.2 of the duration, decayed to 0.05 of the peak by the end.
    let tw_eps = 0.2f32;
    let tw_eta = 0.05f32;

    let nr = 1000usize;
    let nsfac = 20usize;

    let fn_hz: [f32; 20] = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 0.70,
        1.00, 2.00, 3.00, 5.00, 7.00, 10.00, 20.00, 30.00, 50.00, 70.00,
    ];
    // The 20-entry site-amplification frequency table, log-transformed once. It is a FREQUENCY
    // table, not a layer table -- sized by its own entry count, not by the velocity model.
    let siteamp_log_freq: Vec<f32> = fn_hz[..nsfac].iter().map(|hz| hz.ln()).collect();

    // Resolved-default accessors are called once, here, so each default is applied in exactly
    // one place.
    let czero = config.czero();
    let calpha = config.calpha();
    let fcfac = config.fcfac();
    let rvfmax = crate::config::defaults::RVFMAX;
    let (duration, dt, fmx, akapp, qfexp) =
        (config.duration, config.dt, config.fmax, config.kappa, config.qfexp);
    let rvsig1 = config.rv_sig1;
    let mut stress_average = config.stress_drop;
    let irand = config.seed;

    // The slip model is normalised in place and the velocity model gains an air
    // layer, so both are worked on as copies.
    let mut stoch = slip.clone();
    let mut vmod_in = vmod_in.clone();
    let mut j0 = j0_in;
    let mut nlskip = config.nl_skip;

    // Every segment must agree with the FIRST on subfault size, so the first is the
    // reference and the rest are the candidates -- `split_first` says that, where
    // re-indexing `segments[0]` inside a loop over `segments` left it to the reader to
    // notice the index was constant. The reported numbers stay 1-based, to match the input
    // file's own numbering.
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

    // `sigma_p * dl^3` -- the subfault moment scale, the denominator of Graves & Pitarka (2010)
    // eq. 12's `F`. The 1e21 converts bars*km^3 to dyn*cm.
    let subevent_moment = stress_average * dlm * dlm * dlm * 1.0e+21;

    // `F` in Graves & Pitarka (2010) eq. 12 -- Frankel's (1995) finite-fault factor. It scales
    // the subfault corner frequency towards the mainshock's while keeping the summed moment
    // right; `crate::stoc` is where it does its work, and the note on `frank` there shows the
    // algebra.
    //
    // DEVIATION FROM THE PUBLISHED METHOD, AND IT IS A PHYSICS CHOICE. G&P define
    // `F = M_o / (N * sigma_p * dl^3)` -- LINEAR in subfault count. This uses `sqrt(N)`.
    // Graves & Pitarka (2015) does not revise `F`, so neither paper licenses the square root;
    // it may come from the subfault-summation scheme instead. Three other candidate scalings
    // were tried and abandoned upstream, which is some evidence this was tuned rather than
    // derived:
    //
    //   sm / (subevent_moment * subfault_count)          linear in N -- what G&P specify
    //   sm / (subevent_moment * sqrt(subfault_count))    THE LIVE ONE
    //   (sm / subevent_moment)^(2/3)
    //   (fce_avg / fcmain)^2
    //
    // Flagged in `papers/README.md` finding 4 and left alone: changing it would move every
    // waveform, and it wants a domain judgement rather than a tidy-up.
    //
    // The `1.0 *` forces the integer count through a real multiply before the sqrt, and is
    // load-bearing for the exact result.
    let moment_scale = sm / (subevent_moment * (1.0 * subfault_count as f32).sqrt());

    // ------------------------------------------------------------ stations ---
    // No ceiling on the record length: buffers are sized from the requested duration. The
    // original clamped this to a compile-time maximum and SILENTLY TRUNCATED anything longer.
    let ndata = (duration / dt).trunc() as usize;

    // The draw source and whether the rupture-time jitter applies are decided together,
    // because under legacy seeding the jitter gate is a side effect of the seeding ritual.
    let (mut rng, deviates) = seed_and_predraw(config, irand, nr, config.draws_normal_deviates());

    let mut vmod = VelocityModel::new();
    // Three separate component traces rather than one interleaved 2-D block. The interleaving
    // that the output format wants happens once, at the end.
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
        jitter_enabled: deviates.jitter_enabled,
    };

    // ------------------------------------------------- the single station ---
    // Starts at infinity so the first segment's value wins the `min` below.

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
            &FaultPlane {
                origin: GeoPoint { lat_deg: seg.fault_lat_deg, lon_deg: seg.fault_lon_deg },
                strike_deg: seg.strike_deg,
                dip_deg: seg.dip_deg,
                top_depth_km: seg.top_depth_km,
                along_strike_offset_km: seg.along_strike_offset_km,
                subfault_length_km: seg.subfault_length_km,
                subfault_width_km: seg.subfault_width_km,
                along_strike_count: seg.along_strike_count,
                down_dip_count: seg.down_dip_count,
            },
            GeoPoint { lat_deg: station.stlat, lon_deg: station.stlon },
        );

        let windows = time_window_pass(seg, &geom, &vmod, &rv, &path_duration, &angles, &run);
        // The minimum over ALL segments. The original re-initialised this inside the segment
        // loop, so on a multi-segment model the reported distance described only the last
        // segment -- invisible while every fixture was single-segment. Now the minimum,
        // which is what "closest subfault distance" means.

        let plan =
            SpectrumPlan::new(windows.tmax, dt, run.q_exponent, run.window_eps, run.window_eta);

        subfault_pass(
            &mut rng,
            &mut acc,
            SegmentPass { seg, geom: &geom, windows: &windows, plan: &plan, angles: &angles },
            RunContext {
                vmod: &vmod,
                rupture: &rv,
                run: &run,
                config,
                deviates: &deviates,
                siteamp_log_freq: &siteamp_log_freq,
            },
        );
    }

    // An optional output filter is not reachable from here: the switch that would enable it is
    // a caller-level field, and a non-zero value is refused rather than passed through.

    // A peak-amplitude scan over all three components used to happen here. Nothing consumed
    // the result, so it is gone.
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

    Ok(Simulation { ndata, dt, acc: out })
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
    /// Layer count. Also read as an INDEX where a source sits below the model — see
    /// [`source_layer_for`], which is where that case is resolved.
    layer_count: usize,
    /// Whether the rupture-time jitter draw happens.
    ///
    /// Under legacy seeding this is a sign test on the seed *after* the seeding ritual has
    /// advanced it by 8, i.e. really `seed > -8`. Decided once, at the seeding site, so that
    /// eight-off comparison does not look like a plain seed test at the point of use.
    jitter_enabled: bool,
}

/// Per-segment angles, plus the corner-frequency coefficient hoisted out of the subfault
/// loops.
struct SegmentAngles {
    strike_rad: f32,
    dip_rad: f32,
    rake_rad: f32,
    /// `c₀(1 + fcfac) / α_τ` — everything in Graves & Pitarka (2010) eq. 13's corner frequency
    /// that does not vary within a segment.
    ///
    /// **Hoisted**, and bit-identically so: `α_τ` is a pure function of three per-segment
    /// constants, so evaluating it once per segment rather than once per subfault per loop gives
    /// the identical `f32`.
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

/// The generator and the pre-drawn blocks. **The fill order is the draw order**, and the draw
/// order is part of the answer.
struct Deviates {
    /// Whether the rupture-time jitter draw happens. Under modern seeding it always does;
    /// legacy reproduces the old `seed + 8 > 0` sign test.
    jitter_enabled: bool,

    /// `rna` / `rnb` — the vertical component's uniforms.
    radv_uniform_a: Vec<f32>,
    radv_uniform_b: Vec<f32>,
}

/// Seed, then make the three pre-draws.
///
/// **The order and the counts are the contract, not an implementation detail.** Seed,
/// then `nr` uniforms into `a`, then `nr` uniforms into `b`. Changing either moves every
/// sample downstream, which is why this is one function rather than two calls spread
/// through the setup.
///
/// The two uniform blocks must stay two sequential fills. Interleaving them into one pass
/// would put different deviates in different slots and change every vertical component.
fn seed_and_predraw(
    config: &HfConfig,
    seed: u64,
    radv_sample_count: usize,
    draw_normals: bool,
) -> (DrawSource, Deviates) {
    let (mut rng, jitter_enabled) = DrawSource::for_station(seed);

    // Exactly `radv_sample_count` values, which is what `vertical_radiation_spectrum` reads.
    // Filled by two SEPARATE sequential passes -- see the note there on why they must not be
    // interleaved.
    let mut radv_uniform_a = vec![0.0f32; radv_sample_count];
    let mut radv_uniform_b = vec![0.0f32; radv_sample_count];
    fill_uniform_deviates(&mut rng, radv_sample_count, &mut radv_uniform_a);
    fill_uniform_deviates(&mut rng, radv_sample_count, &mut radv_uniform_b);

    // The `MMV` block is GONE. It drew 262,144 normal deviates per station into a buffer
    // with exactly ONE read site, and `irandcnt` advanced once per surviving subfault:
    // 4 reads on the mini fault, 2,827 on the alpine -- 0.0015% and 1.1% of what was
    // drawn. It cost 3.0 ms and 1.0 MiB resident per process, which is 27% of total
    // runtime on the mini fault, and the mini fault is the common case when a 1000-station
    // run gives every station its own process.
    //
    // It also carried a subtler defect. `fill_normal_deviates` renormalises the whole
    // block so its sum of squares equals its length, so the handful of values actually
    // used were scaled by a factor derived from ~262,140 values that were never read.
    // That factor is not physics; it is an artifact of a buffer size.
    //
    // The perturbation now draws on demand, at its point of use. See `subfault_pass`.
    let _ = (draw_normals, config);

    (rng, Deviates { jitter_enabled, radv_uniform_a, radv_uniform_b })
}

/// The velocity-model layer a source at `depth_km` sits in.
///
/// # The defect this replaces
///
/// The two subfault passes disagreed about what to do when a source is deeper than the
/// whole model, and **both answers were wrong**:
///
/// * the window pass had no fallback at all, so it silently reused the *previous
///   subfault's* velocity — and on the very first subfault of the first station the
///   original read uninitialised memory. Pinning that to 0.0 would be a choice rather than
///   a behaviour, and no more defensible.
/// * the subfault pass fell back to layer 0 for the velocity — which after
///   `insert_air_layer` is the **air layer**, `vsh = 0.0005 km/s` — and set `ksrc` to
///   `layer_count`, one PAST the model. That index then reached
///   `site_amplification_factors`, which read a zeroed `Layer`, computed
///   `ln(0 / (bz*pz)) = -inf`, and exponentiated it back to a gain of **zero**. A
///   subfault below the model contributed nothing at all.
///
/// Both now take the deepest real layer, which is the only physically sensible reading:
/// a source below the model is in the half-space, and the half-space is the bottom layer.
///
/// This is reachable — the `vs_moho=4.2` deck truncates the model at the Moho and lands
/// subfaults beneath it, and it is the one deck of 22 whose output moves.
fn source_layer_for(vmod: &VelocityModel, run: &RunScalars, depth_km: f32) -> usize {
    (0..run.layer_count)
        .find(|&k| vmod[k].depth_km >= depth_km as f64)
        .unwrap_or(run.layer_count - 1)
}

/// What the time-window pass produces for one segment.
struct WindowPass {
    /// One entry per real subfault, indexed by [`Segment::grid_index`].
    window_s: Vec<f32>,
    /// Longest window over the segment; sizes the transform.
    tmax: f32,
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
    let mut window_s = vec![0.0f32; seg.subfault_total()];

    for (i, j) in seg.depth_major() {
        let ray = geom.at(i, j);
        // A subfault below the whole model takes the DEEPEST layer -- see
        // [`source_layer_for`] for why neither of the original fallbacks was defensible.
        let shear_velocity_km_s = vmod[source_layer_for(vmod, run, ray.depth_km)].vsh_km_s as f32;

        let rvf = rupture.factor(ray.depth_km);

        // The last table segment this distance is past. `.last()`, NOT `.find()` -- the table
        // ascends, so the first match is the wrong end.
        //
        // Strict `>`, so a distance of exactly the first breakpoint (0.0) matches NOTHING and
        // the duration terms stay zero. That zero is a deliberate choice for an undefined
        // case, and `unwrap_or` is where it lives.
        let bin = path_duration
            .iter()
            .take_while(|s| ray.slant_km > s.start_km)
            .last()
            .copied()
            .unwrap_or(DurationSegment { start_km: 0.0, duration_s: 0.0, slope_s_per_km: 0.0 });

        // Graves & Pitarka (2010) eq. 13: `f_ci = c0 * V_Ri / (alpha_tau * pi * dl)`, with
        // `corner_coeff` carrying `c0 / alpha_tau` and `rvf * beta` being the local rupture
        // speed `V_Ri`.
        let fce = angles.corner_coeff * rvf * shear_velocity_km_s
            / (run.avg_subfault_km * run.pi);
        // The window duration, eq. 17: `T_di = f_ci^-1 + c1*R_i`, a source term plus a path
        // term. Two details worth stating:
        //
        //   * the source term uses `sqrt(F)/f_ci`, which is `1/f_c_effective` for the RESCALED
        //     corner of eq. 12 -- see the `frank` note in `crate::stoc`. So the duration
        //     follows the mainshock-scaled corner, not the raw subfault corner.
        //   * the 2.12 is Boore (1983, p. 1869), who sets the record length to about twice the
        //     duration of strong shaking so the windowed transient has room to decay.
        //
        // No upper cap on the window length.
        let tw0 = run.moment_scale.sqrt() * (1.0 / fce);
        let dpath = bin.duration_s + bin.slope_s_per_km * (ray.slant_km - bin.start_km);
        let window = 2.12 * (tw0 + dpath);
        window_s[seg.grid_index(i, j)] = window;

        if window > tmax {
            tmax = window;
        }
    }

    WindowPass { window_s, tmax }
}

/// The five per-segment things the subfault pass reads.
///
/// Bundled with [`RunContext`] to get `subfault_pass` from thirteen positional arguments
/// to four. The two bundles are the natural cut: this one changes once per segment, that
/// one not at all.
#[derive(Clone, Copy)]
struct SegmentPass<'a> {
    seg: &'a Segment,
    geom: &'a SubfaultGeometry,
    windows: &'a WindowPass,
    plan: &'a SpectrumPlan,
    angles: &'a SegmentAngles,
}

/// Everything the subfault pass reads that is fixed for the whole run.
#[derive(Clone, Copy)]
struct RunContext<'a> {
    vmod: &'a VelocityModel,
    rupture: &'a RuptureVelocityTaper,
    run: &'a RunScalars,
    config: &'a HfConfig,
    deviates: &'a Deviates,
    siteamp_log_freq: &'a [f32],
}

/// Subfault pass — `hb_high_ref.f`'s second subfault loop.
///
/// **Strike-major: `i` outer, `j` inner, the OPPOSITE of [`time_window_pass`], and here
/// the order IS the contract.** `irandcnt` advances once per surviving subfault and
/// indexes the pre-drawn normals, and every `stochastic_spectrum` and
/// `horizontal_radiation_spectrum` call draws from the live stream. Walking the grid the
/// other way pairs a different deviate with every subfault and changes every waveform.
/// See `PORTING_RULES.md` §5.
fn subfault_pass(
    rng: &mut impl Draws,
    acc: &mut [Vec<f32>; 3],
    segment: SegmentPass<'_>,
    ctx: RunContext<'_>,
) {
    // Destructured immediately so the body below reads exactly as it did when these were
    // thirteen positional parameters. The bundles exist to make the CALL safe, not to be
    // threaded through the body field by field.
    let SegmentPass { seg, geom, windows, plan, angles } = segment;
    let RunContext { vmod, rupture, run, config, deviates, siteamp_log_freq } = ctx;

    let np2 = plan.np2;
    // Fixed for the whole run, so it is built once here rather than per call. Six of
    // `stochastic_spectrum`'s nineteen former arguments were these, re-passed on every one
    // of the hundreds of thousands of calls in a run.
    let model = SourceModel {
        dt: run.dt,
        window_eps: run.window_eps,
        window_eta: run.window_eta,
        subevent_moment: run.subevent_moment,
        kappa_s: run.kappa_s,
        moment_scale: run.moment_scale,
    };
    // Holds three owned spectra between the two component loops. A `Vec` rather than a
    // `[_; 3]` because the values MOVE out at the end -- `drain` hands each one to
    // `radiate_and_invert`, which consumes it -- and because it must be filled by an
    // explicit sequential loop: the fill order is the RNG stream, and neither
    // `array::from_fn` nor `array::map` documents its evaluation order. The Vec's own
    // three-pointer allocation is made once here and reused; `drain` leaves the capacity.
    let mut spectrum: Vec<Array1<Complex32>> = Vec::with_capacity(3);
    let mut subfault_acc: [Array1<f32>; 3] = std::array::from_fn(|_| Array1::zeros(np2));
    let mut radiation = vec![0.0f32; plan.fold_count];
    let mut siteamp_factors = vec![0.0f32; run.site_table_len];
    let mut ray = RayState::default();

    for (i, j) in seg.strike_major() {
        let subfault = seg.at(i, j);
        if subfault.slip < 0.001 {
            continue; // goto 4 lands on the inner loop's terminator
        }
        let ray_geometry = geom.at(i, j);
        let subfault_window_s = windows.window_s[seg.grid_index(i, j)];

        // No pre-zeroing. `apply_radiation_and_invert` ASSIGNS over `time_series[..np2]`
        // -- `*sample = fac * bin.re`, not `+=` -- for all three components before
        // `accumulate_subfault` reads any of them, so every element is written before it
        // is read. The fill was 192 KB of memset per subfault that nothing could observe.

        let ksrc = source_layer_for(vmod, run, ray_geometry.depth_km);
        let shear_velocity_km_s = vmod[ksrc].vsh_km_s as f32;
        let density_g_cm3 = vmod[ksrc].density_g_cm3 as f32;

        let base_rvf = rupture.factor(ray_geometry.depth_km);
        let mut rvf = base_rvf;
        if run.rv_sig1 > 0.0 {
            // Drawn here rather than read from a pre-filled block. One deviate per
            // surviving subfault is what was ever used; `irandcnt` existed only to index
            // into 262,144 of them.
            rvf = base_rvf * (normal_deviate(rng) * run.rv_sig1).exp();
            if rvf > run.rvfmax {
                rvf = run.rvfmax;
            }
        }

        let fce = angles.corner_coeff * rvf * shear_velocity_km_s / run.avg_subfault_km / run.pi;

        for &ray_type in &config.rayset {
            let kind = ray_type.kind();

            // The tracing runs even for a straight ray -- type 0 borrows type 1's tracing and
            // then the straight-line values below overwrite the results. Wasteful, but the
            // tracer also advances no random state, so removing it is safe only if you are
            // sure of that.
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

            // Three calls in component order: each draws `np2` normal deviates. The only
            // field that differs between them is `fmax_hz`, which the vertical caps.
            spectrum.clear();
            for component in Component::ALL {
                spectrum.push(stochastic_spectrum(
                    rng,
                    plan,
                    &model,
                    &RayPath {
                        distance_km: rpath,
                        window_s: subfault_window_s,
                        shear_velocity_km_s,
                        density_g_cm3,
                        corner_frequency_hz: fce,
                        fmax_hz: component.capped_fmax(run.fmax_hz),
                        qbar,
                    },
                ));
            }

            if config.site_amp {
                site_amplification_factors(
                    vmod, ksrc, siteamp_log_freq, &mut siteamp_factors,
                );
                for spec in &mut spectrum {
                    apply_site_amplification(
                        spec.as_slice_mut().expect("an owned Array1 is contiguous"),
                        &plan.log_frequency_hz, run.site_table_len,
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
            // `drain` moves each spectrum out. The zip is sound because the fill loop
            // above pushes in `Component::ALL` order and this walks the same order; the
            // Vec is left empty and reusable for the next ray type.
            let arrival = RadiationAngles {
                strike_rad: angles.strike_rad,
                dip_rad: angles.dip_rad,
                rake_rad: angles.rake_rad,
                azimuth_rad: pa,
                takeoff_rad: th,
            };
            for (component, spec) in Component::ALL.into_iter().zip(spectrum.drain(..)) {
                match component.azimuth_offset_deg() {
                    Some(offset_deg) => horizontal_radiation_spectrum(
                        rng, &arrival, &plan.frequency_hz, offset_deg * run.deg_to_rad,
                        run.radv_sample_count, &mut radiation,
                    ),
                    None => vertical_radiation_spectrum(
                        &arrival, &plan.frequency_hz,
                        &deviates.radv_uniform_a, &deviates.radv_uniform_b,
                        run.radv_sample_count, &mut radiation,
                    ),
                };
                // The spectrum moves: out of the Vec, into `radiate_and_invert`, which
                // consumes it because the inverse transform is in place, and the samples
                // move on into the accumulator. No copy anywhere on this path.
                subfault_acc[component.index()] =
                    radiate_and_invert(spec, ArrayView1::from(&radiation[..]));
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

            // Both terms truncate TOWARD ZERO, not toward negative infinity, so a negative
            // `sub_tstart` makes `kst` smaller and possibly negative. `accumulate_subfault`
            // relies on that and clips; see `PORTING_RULES.md` §7.
            //
            // `trunc()` is written explicitly even though `as i32` alone would round the
            // same way, because it is the rounding MODE that is load-bearing here and a bare
            // cast does not say so.
            let kst = (ratim / run.dt).trunc() as i32 + (sub_tstart / run.dt).trunc() as i32;

            // ONE DRAW, AND IT MUST STAY. A sub-event loop here once drew a uniform and
            // turned it into a time offset, but the sub-event count was frozen at 1 and the
            // offset was then unconditionally zeroed -- computed and thrown away.
            //
            // The arithmetic is gone. THE DRAW IS NOT. It advances the shared generator once
            // per (subfault, ray), and every sample drawn after it depends on where the stream
            // ends up. Deleting this as obviously-dead code changes every waveform in the
            // program. See `PHYSICS.md` §9.
            let _stream_advance = rng.next_f32();

            accumulate_subfault(acc, &subfault_acc, subfault.slip, kst, np2, run.ndata);
        }
    }
}

/// Place one subfault's `np2`-sample contribution into the station accumulator.
///
/// `start_sample` is a 1-based sample number and **can be negative**: truncation is toward
/// zero and the window start can precede the origin time. Samples landing before sample 1 are
/// discarded rather than written.
///
/// # A fixed off-by-one
///
/// Sample 1 of the subfault's trace lands on `start_sample`, not on `start_sample + 1`. The
/// original indexed one element before the column -- which nothing wrote, and which aliasing
/// made read as zero rather than crash -- so every subfault's contribution arrived one sample
/// late. Corrected here.
fn accumulate_subfault(
    acc: &mut [Vec<f32>; 3],
    subfault_acc: &[Array1<f32>; 3],
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

    // BOTH ends can put the window entirely outside the record, and they are different cases:
    // one contribution ends before the record begins (a large negative start), another begins
    // after it ends (a long-path ray at a far station). AS AN EXPLICIT RANGE THE SECOND CASE
    // COMPUTES A NEGATIVE LENGTH, which must be rejected before it is cast to `usize` or it
    // wraps to something enormous. A gate caught exactly that once.
    if last_sample < first_sample {
        return;
    }

    let skip = (first_sample - start_sample) as usize;
    let count = (last_sample - first_sample + 1) as usize;
    let dst = first_sample as usize - 1;

    // `scaled_add` IS this operation: `y += alpha * x`, the axpy every linear-algebra library
    // names. The clipping above is the part that carries the actual thought.
    for (out, contribution) in acc.iter_mut().zip(subfault_acc) {
        ArrayViewMut1::from(&mut out[dst..dst + count])
            .scaled_add(weight, &contribution.slice(s![skip..skip + count]));
    }
}

/// One segment of the piecewise-linear duration-versus-distance table.
///
/// One value per segment rather than three parallel arrays: every read of one field is at the
/// same index as the other two, so they belong together.
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
/// `len()` is the segment count, so a length and a capacity can no longer disagree. The largest
/// model here uses eight segments.
type PathDuration = Vec<DurationSegment>;

/// Build the path-duration table.
///
/// Total by construction, now that the model is an enum: there is no way to reach an
/// uninitialised table, and rejecting a bad selector happens once, in
/// [`PathDurationModel::from_deck`].
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
        // Graves & Pitarka (2010) eq. 17: `T_di = f_ci^-1 + c1*R_i` with `c1 = 0.063`.
        PathDurationModel::Gp2010 => constant_slope(0.063),
        PathDurationModel::Wus => constant_slope(0.07),
        PathDurationModel::Ena => constant_slope(0.1),
        // Boore & Thompson (2014) Table 1, reproduced exactly: breakpoints at 0, 7, 45, 125,
        // 175, 270 km with durations 0, 2.4, 8.4, 10.9, 17.4, 34.2 s.
        //
        // KNOWN DEVIATION FROM THE PAPER, BEYOND 270 km. Table 1 specifies a tail slope of
        // 0.156 s/km for `R` past the last breakpoint. `from_breakpoints` instead copies the
        // slope of the final tabulated segment, `(34.2 - 17.4)/(270 - 175) = 0.177` -- about
        // 13% steeper. This reproduces the original faithfully (it did
        // `dpdr(ndur) = dpdr(ndur-1)`), so the deviation is in the model as implemented rather
        // than in this port, and it only bites for ray paths longer than 270 km. Recorded in
        // `papers/README.md`; not changed here, because it would move every long-path waveform.
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
/// Three passes over the subfault grid, in this order:
///
/// 1. average subfault size, and the maximum absolute slip (computed and discarded)
/// 2. slip → moment via the rigidity `xmu`, accumulating `xsum`, `fce_avg`,
///    `trise_avg` and a count of *all* subfaults
/// 3. re-count only the subfaults above 0.001 and rescale so their mean weight is 1
///
/// The two counts are different and both matter: pass 2's normalises the averages, pass 3's is
/// the one that reaches `moment_scale` — the `N` of Graves & Pitarka (2010) eq. 12. Only the
/// second is returned, because only it is read downstream.
fn normalise_source(
    stoch: &mut StochModel,
    layer_count: usize,
    vmod_in: &VelocityModelInput,
    deg_to_rad: f32,
    moment: Option<f32>,
) -> SourceScale {
    let nevnt = stoch.segments.len();

    // --- pass 1: average subfault size ----------------------------------------
    // A maximum-slip accumulation over every subfault used to happen here. Nothing read it, so
    // it is gone along with the whole loop that fed it.
    let mut dlm = 0.0f32;
    for s in &stoch.segments {
        dlm += (s.subfault_length_km * s.subfault_width_km).sqrt();
    }
    dlm /= nevnt as f32;

    // --- pass 2: relative slip to relative moment -----------------------------
    // An average corner frequency and rise time were also accumulated here, at the cost of a
    // rupture-velocity taper, an `alpha_tau` evaluation and several more operations PER
    // SUBFAULT. All of it fed a quantity nothing reads:
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
        // THE WHOLE PRODUCT BELOW IS COMPUTED IN f64 and narrows only on assignment to `xmu`.
        // Narrowing earlier -- for instance by keeping the dimensions in `f32` -- shifts every
        // subfault moment by an ulp or two.
        let (length_km, width_km) =
            (segment.subfault_length_km as f64, segment.subfault_width_km as f64);

        // Depth-major, which is storage order, so `xsum` accumulates without index arithmetic.
        // Rigidity is per depth row, not per subfault, which is why the row is the unit.
        for (row_index, row) in segment.depth_rows_mut().enumerate() {
            let zdep = top_depth_km + (row_index as f32 + 0.5) * dwdj;
            // Layer lookup. FALLS THROUGH to `layer_count` -- one past the model -- when the
            // depth is below every layer, and that index is then used. The fall-through is
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

/// `α_τ`, the dip-and-rake corner-frequency and rise-time adjustment.
///
/// Graves & Pitarka (2015) eq. 3, `α_T = 1 + F_D·F_R·c_α`. **Returns the RECIPROCAL of that**,
/// because the caller divides `c₀` by it and eq. 13 has `α_τ` in the denominator — so the value
/// returned here is `α_τ` itself, ≤ 1, and smaller for a shallow-dipping thrust. Physically that
/// means such a fault gets a *higher* corner frequency and a shorter rise time, which is the
/// observed trend (Graves & Pitarka 2010, p. 2099, citing Somerville 1998: shorter rise times for
/// thrust events imply relatively high dynamic stress drops).
///
/// Graves & Pitarka (2010) eq. 9 parameterised this on **dip alone**, piecewise — 1 above 60°,
/// 0.82 below 45°. The continuous dip-and-rake form below is the 2015 revision, and is another
/// marker that this code tracks the later method (see [`crate::config::defaults::CZERO`]).
///
/// `fD` tapers with dip above 45 degrees; `fR` peaks at a rake of 90 degrees. The rake is first
/// wrapped into
/// `[-180, 180]` by repeated addition or subtraction of 360.
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
        // `sqrt(x*x)` rather than `abs(x)`: kept because the two can differ in the last bit.
        fr = 1.0 - ((avgrak - 90.0) * (avgrak - 90.0)).sqrt() / 90.0;
    }

    1.0 / (1.0 + fd * fr * calpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward sample-at-a-time loop, as an independent check on the slice form.
    fn accumulate_reference(
        acc: &mut [Vec<f32>; 3],
        subfault_acc: &[Array1<f32>; 3],
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
    /// `start_sample > ndata` is the case that was once wrong: it computes a negative length,
    /// and casting that to `usize` wraps. It is reachable -- a Moho multiple can make the path
    /// long enough to start past the end of the record -- and this pins it directly rather
    /// than relying on a deck that happens to trigger it.
    #[test]
    fn accumulate_matches_the_reference_loop_at_every_alignment() {
        let np2 = 8usize;
        let ndata = 10usize;
        let subfault_acc: [Array1<f32>; 3] =
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
