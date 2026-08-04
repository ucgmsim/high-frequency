//! `hb_high` driver — transliteration of the main program, `hb_high_ref.f:102-1463`.
//!
//! Reads a 22-line parameter deck on stdin plus a `.stoch` slip model, a 1-D
//! velocity model and a station list; writes `ndata * 3` interleaved `f32` to the
//! named output file and one `(1x,f10.4)` epicentral distance per station to
//! stderr.
//!
//! Structured as the Fortran is, in the Fortran's order. See `PORTING_RULES.md`;
//! in particular the loop nest below has two passes over the subfaults that
//! iterate in **opposite** order, and that is load-bearing.

use std::io::Write;

use hb_high::config::{
    defaults, HfConfig, PathDurationModel, RayKind, RayType, RuptureVelocity,
    RuptureVelocityTaper, StressParamAdjust,
};
use hb_high::deck::ListReader;
use hb_high::fort::{nint, Array1, Array2, Complex32};
use hb_high::geom::even_dist2;
use hb_high::highcor::highcor_f;
use hb_high::input::{
    insert_air_layer, read_stations, read_stoch, read_velocity_model, StochModel,
};
use hb_high::radiation::{radfrq_lin, radv_lin};
use hb_high::ray::gf_amp_tt;
use hb_high::rng::{normal_random_number, ranu2, Pcg32};
use hb_high::site::{get_sitefacs, siteamp};
use hb_high::state::{params, RayState, Vmod, VmodIn};
use hb_high::stoc::stoc_f;

/// `mm` and `mmv` as the **main program** sees them.
///
/// Under `VERSION1` the main program includes `params_no_window.h`, so both are
/// 262144 — not the 32769/180000 that every subroutine gets from `params.h`.
/// This matters here because `ndata` is clamped to `mmv` and
/// `normal_random_number(mmv, normal_deviates)` draws exactly this many deviates.
const MM: usize = params::MM;
const MMV: usize = params::MMV;


/// The parts of the deck that say where data comes from and where it goes, as
/// opposed to what to simulate. Everything physical lives in [`HfConfig`].
///
/// This split is the point of the exercise: `HfConfig` is what a library caller or
/// the future Python wrapper supplies, while `DeckIo` is an artifact of being driven
/// by a text deck that writes to a file.
struct DeckIo {
    /// `asite` — station list path.
    station_file: String,
    /// Opened **without** truncation and seeked into, so several invocations can
    /// fill disjoint ranges of one shared file. See `PORTING_RULES.md` §9.
    output: String,
    slip_model: String,
    velocity_model: String,
    /// Always 1 in production — `hf_sim.py` runs one process per station. Anything
    /// else is refused; see `run`.
    nsite: usize,
    seek_bytes: i64,
    /// `ift`. Non-zero would call `filter3d`, which is dead here.
    iftt: i32,
    /// `fhi` — reaches only the `fhigh must be <` warning.
    fhil: f32,
}

/// The path-duration model: a piecewise-linear duration-versus-distance table.
///
/// `dpdr` is the slope of each segment. For the single-segment models it is given
/// directly; for the multi-segment ones it is differenced from `rdur`/`dpth`, and
/// the last segment repeats the second-to-last slope so distances beyond the table
/// extrapolate rather than flatten.
struct PathDuration {
    ndur: usize,
    /// Segment start distances, km.
    rdur: Array1<f32>,
    /// Duration at each segment start, s.
    dpth: Array1<f32>,
    /// Slope of each segment, s/km.
    dpdr: Array1<f32>,
}

/// Build the path-duration table.
///
/// Total by construction now that the model is an enum: the Fortran's
/// undefined-`ndur` path is unrepresentable, and rejecting a bad integer happens
/// once, in `PathDurationModel::from_deck`.
fn path_duration_table(model: PathDurationModel) -> PathDuration {
    let mut rdur = Array1::<f32>::new(50);
    let mut dpth = Array1::<f32>::new(50);
    let mut dpdr = Array1::<f32>::new(50);

    // (distances, durations) for the multi-segment models; slope-only for the rest.
    let ndur = match model {
        PathDurationModel::Gp2010 => { rdur[1] = 0.0; dpth[1] = 0.0; dpdr[1] = 0.063; 1 }
        PathDurationModel::Wus => { rdur[1] = 0.0; dpth[1] = 0.0; dpdr[1] = 0.07; 1 }
        PathDurationModel::Ena => { rdur[1] = 0.0; dpth[1] = 0.0; dpdr[1] = 0.1; 1 }
        PathDurationModel::Bt2014Wus => {
            // BT2014 WUS. The breakpoints at 7, 45, 125 and 175 km are what the
            // Phase 2 distance ladder is chosen to straddle.
            let r = [0.0, 7.0, 45.0, 125.0, 175.0, 270.0];
            let d = [0.0, 2.4, 8.4, 10.9, 17.4, 34.2];
            for (i, (&ri, &di)) in r.iter().zip(d.iter()).enumerate() {
                rdur[i + 1] = ri;
                dpth[i + 1] = di;
            }
            r.len()
        }
        PathDurationModel::Bt2015Ena => {
            // BT2015 ENA.
            let r = [0.0, 15.0, 35.0, 50.0, 125.0, 200.0, 392.0, 600.0];
            let d = [0.0, 2.6, 17.5, 25.1, 25.1, 28.5, 46.0, 69.1];
            for (i, (&ri, &di)) in r.iter().zip(d.iter()).enumerate() {
                rdur[i + 1] = ri;
                dpth[i + 1] = di;
            }
            r.len()
        }
    };

    if ndur != 1 {
        for i in 1..=ndur - 1 {
            dpdr[i] = (dpth[i + 1] - dpth[i]) / (rdur[i + 1] - rdur[i]);
        }
        dpdr[ndur] = dpdr[ndur - 1];
    }
    PathDuration { ndur, rdur, dpth, dpdr }
}

/// Read the deck, in the Fortran's order.
///
/// Loading the files it names (`slip_model`, `velfile`, `asite`) is deliberately
/// *not* done here. The Fortran interleaves those reads with the deck reads, but
/// nothing in deck parsing depends on their contents, so hoisting the whole deck
/// ahead of them is behaviour-preserving and makes both halves legible.
fn read_deck(
    deck: &mut ListReader,
) -> Result<(HfConfig, DeckIo), Box<dyn std::error::Error>> {
    use hb_high::deck::{parse_f32, parse_f64, parse_i32};

    /// "Use the default" is signalled by a value below **-1.0**, not merely a
    /// negative one. Getting that boundary wrong would substitute a default for a
    /// legitimate small negative input, silently.
    fn defaulted(v: f32) -> Option<f32> {
        if v < -1.0 { None } else { Some(v) }
    }
    /// The other convention in this deck: non-positive means "derive this".
    fn derived(v: f32) -> Option<f32> {
        if v <= 0.0 { None } else { Some(v) }
    }

    let stress_average = deck.f32()?;
    let asite = deck.read_filename()?;
    let outname = deck.read_filename()?;

    // `read(5,*) nrtyp,(irtype(i),i=1,nrtyp)` is ONE read whose length depends
    // on its own first item.
    let (nrtyp, irtype_items) = deck.read_count_and_list()?;
    let mut rayset = Vec::with_capacity(nrtyp as usize);
    for item in irtype_items.iter().take(nrtyp as usize) {
        rayset.push(RayType(parse_i32(item.as_deref().unwrap_or(""))?));
    }

    let isite_amp = deck.i32()?;
    let (iftt, fhil) = {
        let v = deck.read_values(4)?;
        let g = |k: usize| v[k].as_deref().unwrap_or("");
        let _nbu = parse_i32(g(0))?;
        let iftt = parse_i32(g(1))?;
        let _flol = parse_f32(g(2))?;
        (iftt, parse_f32(g(3))?)
    };
    let irand = deck.i32()?;
    let nsite = deck.i32()? as usize;

    let (duration, dt, fmx, akapp, qfexp) = {
        let v = deck.read_values(5)?;
        let g = |k: usize| parse_f32(v[k].as_deref().unwrap_or(""));
        (g(0)?, g(1)?, g(2)?, g(3)?, g(4)?)
    };

    let (rupture_velocity, czero, calpha) = {
        let v = deck.read_values(5)?;
        let g = |k: usize| parse_f32(v[k].as_deref().unwrap_or(""));
        (
            RuptureVelocity {
                frac: defaulted(g(0)?),
                shallow: defaulted(g(1)?),
                deep: defaulted(g(2)?),
            },
            defaulted(g(3)?),
            defaulted(g(4)?),
        )
    };

    let (moment, rupture_velocity_override) = {
        let v = deck.read_values(2)?;
        let g = |k: usize| parse_f32(v[k].as_deref().unwrap_or(""));
        // A negative moment means derive it from the slip model. Note the boundary
        // differs from `derived`: the Fortran tests `sm < 0.0`, so an explicit zero
        // is honoured rather than derived.
        let m = g(0)?;
        (if m < 0.0 { None } else { Some(m) }, derived(g(1)?))
    };

    let slip_model = deck.read_filename()?;
    let velocity_model = deck.read_filename()?;
    let vsmoho = deck.f64()?; // pre-set to -1.0, then read
    let vs_moho = if vsmoho <= 0.0 { None } else { Some(vsmoho) };

    let nl_skip = {
        let v = deck.read_values(6)?;
        let gf = |k: usize| parse_f32(v[k].as_deref().unwrap_or(""));
        let nlskip = parse_i32(v[0].as_deref().unwrap_or(""))?;
        // Velocity-model perturbation sigmas and icflag: all dead under the
        // production deck (every sigma is 0.0), so read past them. The original
        // also normalises icflag to 1 when it is neither 0 nor 1, which cannot
        // matter once nothing reads it.
        let (_vpsig, _vshsig, _rhosig, _qssig) = (gf(1)?, gf(2)?, gf(3)?, gf(4)?);
        let _icflag = parse_i32(v[5].as_deref().unwrap_or(""))?;
        nlskip
    };
    let _velname = deck.read_filename()?;

    let (fa_sig1, fa_sig2, rv_sig1) = {
        let v = deck.read_values(3)?;
        let g = |k: usize| parse_f32(v[k].as_deref().unwrap_or(""));
        (g(0)?, g(1)?, g(2)?)
    };

    let ipdur_model = deck.i32()?;
    let path_duration = PathDurationModel::from_deck(ipdur_model).ok_or_else(|| {
        format!(
            "ipdur_model = {ipdur_model} leaves ndur undefined in the Fortran \
             (accepted values are <=0, 1, 2, 11, 12)"
        )
    })?;

    // NOTE: this read wants THREE items and the deck supplies a bare `0` line
    // followed by three more, so it takes the `0` as ispar_adjust and two from the
    // next record -- leaving the third to be discarded when the next read starts a
    // fresh record. That is why production's tect_type never arrives and why
    // targ_mag and fault_area are swapped relative to what hf_sim.py intends.
    // Reproduced deliberately; see REFACTOR.md §1.1.
    let (stress_param_adjust, target_magnitude, fault_area) = {
        let v = deck.read_values(3)?;
        (
            StressParamAdjust::from_deck(parse_i32(v[0].as_deref().unwrap_or(""))?),
            derived(parse_f32(v[1].as_deref().unwrap_or(""))?),
            derived(parse_f32(v[2].as_deref().unwrap_or(""))?),
        )
    };

    let seek_bytes = {
        // Pre-set to 0, then read.
        let v = deck.read_values(1)?;
        v[0].as_deref()
            .map(|s| parse_f64(s).map(|x| x as i64))
            .transpose()?
            .unwrap_or(0)
    };

    Ok((
        HfConfig {
            stress_drop: stress_average,
            rayset,
            site_amp: isite_amp != 0,
            seed: irand,
            duration,
            dt,
            fmax: fmx,
            kappa: akapp,
            qfexp,
            rupture_velocity,
            czero,
            calpha,
            moment,
            rupture_velocity_override,
            vs_moho,
            nl_skip,
            fa_sig1,
            fa_sig2,
            rv_sig1,
            path_duration,
            stress_param_adjust,
            target_magnitude,
            fault_area,
        },
        DeckIo {
            station_file: asite,
            output: outname,
            slip_model,
            velocity_model,
            nsite,
            seek_bytes,
            iftt,
            fhil,
        },
    ))
}

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hb_high: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

/// Read one subfault's accumulated trace, component `l`, at Fortran index `idx`,
/// reproducing the original's out-of-bounds read at `idx == 0`.
///
/// Names below are the **Fortran's** (`stdd`), not this port's, because the whole
/// point of the function is to model what the original does to its own storage.
///
/// The accumulation loop runs `li = k2, kend` and reads `stdd(li-k2, l)`, so the
/// first iteration reads **index 0** — one before the column. `highcor_f` fills
/// only `1..=np2`.
///
/// In the Fortran `stdd` is `stdd(mmv,3)` column-major, so `stdd(0,2)` aliases
/// `stdd(mmv,1)` and `stdd(0,3)` aliases `stdd(mmv,2)`; both are untouched (the
/// zeroing loop covers only `1..np2`) and live in `.bss`, hence zero.
/// `stdd(0,1)` is genuinely before the array. Modelled as zero for all three.
///
/// The observable effect is that each subfault's contribution is delayed one
/// sample: `DS(l,k2)` gets nothing and `DS(l,k2+1)` gets `stdd(1)`.
/// See `PORTING_RULES.md` §7 — and `REFACTOR.md` §2.6, which asks whether this
/// bug should be kept for production compatibility or fixed.
#[inline]
fn subfault_acc_at(subfault_acc: &[Array1<f32>; 3], l: usize, idx: usize) -> f32 {
    if idx == 0 {
        0.0
    } else {
        subfault_acc[l - 1][idx]
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut stdin = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut stdin)?;
    let mut deck = ListReader::new(&stdin);

    let pu = 3.1415926f32 / 180.0;
    let pai = 3.1415926f32;

    let tw_eps = 0.2f32; // 0.4 first, then 0.2
    let tw_eta = 0.05f32;

    let nr = 1000usize;
    let delay = 0.0f32;
    let nsfac = 20usize;

    // Site-amplification frequency table, log-transformed in place (:196-218).
    let fn_hz: [f32; 20] = [
        0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.20, 0.30, 0.50, 0.70,
        1.00, 2.00, 3.00, 5.00, 7.00, 10.00, 20.00, 30.00, 50.00, 70.00,
    ];
    let mut siteamp_log_freq = Array1::<f32>::new(params::NLAYMAX);
    for i in 1..=nsfac {
        siteamp_log_freq[i] = fn_hz[i - 1].ln();
    }

    // ---------------------------------------------------------------- deck ---
    let (config, io) = read_deck(&mut deck)?;

    // The resolved-default accessors are called once, here, and the body below then
    // reads under the Fortran's names as the transliteration it still is.
    let czero = config.czero();
    let calpha = config.calpha();
    let fcfac = config.fcfac();
    let rvfmax = defaults::RVFMAX;
    let (duration, dt, fmx, akapp, qfexp) =
        (config.duration, config.dt, config.fmax, config.kappa, config.qfexp);
    let rvsig1 = config.rv_sig1;
    let mut stress_average = config.stress_drop;
    let mut irand = config.seed;
    let mut nlskip = config.nl_skip;

    if io.nsite != 1 {
        // The station loop shares one generator: `ranu2` fills radv_rand_a/b once
        // before it, and each station's `normal_random_number` draw continues from
        // wherever the previous station left off. So a multi-station run is NOT a
        // concatenation of single-station runs, and the one-station-per-call library
        // API cannot express it. Production always writes nsite = 1 (hf_sim.py runs
        // one process per station) and no parity deck uses anything else, so this is
        // refused rather than silently computed differently.
        return Err(format!(
            "nsite = {} is not supported: the station loop shares one RNG stream, so \
             stations are not independent. Production runs one station per process. \
             See REFACTOR.md §1.1.",
            io.nsite
        )
        .into());
    }

    let stoch_text = std::fs::read_to_string(&io.slip_model)
        .map_err(|e| format!("opening slip model {}: {e}", io.slip_model))?;
    let mut stoch = read_stoch(&stoch_text, pu)?;
    let nevnt = stoch.segments.len();

    for (k, s) in stoch.segments.iter().enumerate() {
        if s.dx != stoch.segments[0].dx {
            return Err(format!(
                "dx({}) = {} not equal to dx(1) = {}, exiting...",
                k + 1, s.dx, stoch.segments[0].dx
            ).into());
        }
        if s.dw != stoch.segments[0].dw {
            return Err(format!(
                "dw({}) = {} not equal to dw(1) = {}, exiting...",
                k + 1, s.dw, stoch.segments[0].dw
            ).into());
        }
    }

    let mut vmod_in = VmodIn::new();
    let vel_text = std::fs::read_to_string(&io.velocity_model)
        .map_err(|e| format!("opening velocity model {}: {e}", io.velocity_model))?;
    let mut j0 = read_velocity_model(&vel_text, &mut vmod_in, config.vs_moho())?;

    // ------------------------------------------------- path duration model ---
    let PathDuration { ndur, rdur, dpth, dpdr } = path_duration_table(config.path_duration);

    // ------------------------------------------------------- air layer -------
    let (j0_air, nlskip_air) = insert_air_layer(&mut vmod_in, j0, nlskip);
    j0 = j0_air;
    nlskip = nlskip_air;

    if io.fhil > 1.0 / 2.0 / dt {
        eprintln!("fhigh must be < {} {} {}", 1.0 / 2.0 / dt, io.fhil, dt);
    }

    // Resolved here rather than at parse time: the deep transition depths depend on
    // the deepest hypocentre, which is only known once the slip model is read.
    let rv = config.rupture_velocity.resolve(stoch.zhyp_max);

    // ------------------------------------------------ source normalisation ---
    let SourceScale { dlm, sm, fce_avg, fcmain, nstot } =
        normalise_source(&mut stoch, j0, &vmod_in, &rv, pu, pai, czero, calpha, fcfac, config.moment);

    // ---------------------------------------- stress parameter adjustment ----
    let targ_mag = config
        .target_magnitude
        .unwrap_or_else(|| 2.0 * (sm.ln() / 10.0f32.ln()) / 3.0 - 10.7);
    let fault_area = config.fault_area.unwrap_or(stoch.farea_in);
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
    // nsum is computed from `ratio` and then forced to 1 (2004-04-20), which is
    // why the Frankel operator in stoc_f carries the scaling instead.
    let nsum = 1usize;

    // The Fortran computes four candidate moment scalings in a row and lets the
    // last assignment win, leaving the other three as documentation of what was
    // tried. Reproduced with the names attached to their formulae rather than to
    // their order, and only the surviving one bound.
    //
    //   by_count      sm / (subevent_moment * nstot)          -- linear in subfault count
    //   by_sqrt_count sm / (subevent_moment * sqrt(nstot))    -- THE LIVE ONE
    //   by_two_thirds (sm/subevent_moment)^(2/3)
    //   by_corner_sq  (fce_avg / fcmain)^2
    //
    // `1.0 *` in by_sqrt_count is the Fortran's, and it matters: it forces the
    // integer nstot through a real multiply before the sqrt.
    let _by_count = sm / (subevent_moment * (nstot * nsum) as f32);
    let _by_two_thirds = ((2.0 / 3.0) * (sm / subevent_moment).ln()).exp();
    let _by_corner_sq = (fce_avg / fcmain) * (fce_avg / fcmain);
    let moment_scale = sm / (subevent_moment * (1.0 * nstot as f32).sqrt());

    // ------------------------------------------------------------ stations ---
    let ndata = {
        let n = (duration / dt) as i32 as usize;
        n.min(MMV)
    };

    let _ifu = irand;
    let (mut rng, irand_after) = Pcg32::seed(irand);
    // init_random_seed mutates its argument, and the mutated value gates the
    // rupture-time jitter below.
    irand = irand_after;

    let mut radv_rand_a = Array1::<f32>::new(MMV);
    let mut radv_rand_b = Array1::<f32>::new(MMV);
    ranu2(&mut rng, nr, &mut radv_rand_a);
    ranu2(&mut rng, nr, &mut radv_rand_b);

    let station_text = std::fs::read_to_string(&io.station_file)
        .map_err(|e| format!("opening station file {}: {e}", io.station_file))?;
    let stations = read_stations(&station_text, io.nsite)?;

    let mut out = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&io.output)
        .map_err(|e| format!("opening output {}: {e}", io.output))?;
    // No status= in the Fortran open, so an existing file is NOT truncated;
    // combined with the seek this lets several invocations fill disjoint ranges
    // of one shared file. See PORTING_RULES.md §9.
    std::io::Seek::seek(&mut out, std::io::SeekFrom::Start(io.seek_bytes as u64))?;

    let mut vmod = Vmod::new();
    let mut acc = Array2::<f32>::new(3, MMV);
    let mut normal_deviates = Array1::<f32>::new(MMV);
    let mut freq = Array1::<f32>::new(MM);
    let mut radiation = Array1::<f32>::new(MM);
    let mut siteamp_factors = Array1::<f32>::new(params::NLAYMAX);
    let mut stderr = std::io::stderr();

    for station in &stations {
        let mut d10 = 1000.0f32;
        acc.fill(0.0);

        if nlskip >= 0 {
            unreachable!("grandvel is dead under the production deck (nl_skip < 0)");
        } else {
            for k in 1..=j0 {
                vmod.depth[k] = vmod_in.depth0[k] as f64;
                vmod.thic[k] = vmod_in.thic0[k] as f64;
                vmod.vp[k] = vmod_in.vp0[k];
                vmod.vsh[k] = vmod_in.vsh0[k];
                vmod.rho[k] = vmod_in.rho0[k];
                vmod.qp[k] = vmod_in.qp0[k];
                vmod.qs[k] = vmod_in.qs0[k];
            }
        }

        if config.draws_normal_deviates() {
            // mmv deviates, not np2: this is the full 262144 under VERSION1.
            normal_random_number(&mut rng, MMV, &mut normal_deviates);
        }

        for iv in 0..nevnt {
            let seg = &stoch.segments[iv];
            let stra = seg.strq * pu;
            let dipa = seg.dipq * pu;
            let raka = seg.rakeq * pu;

            let geom = even_dist2(
                seg.elonq, seg.elatq, station.stlon, station.stlat,
                seg.strq, seg.dipq, seg.dtop, seg.astop, seg.dx, seg.dw,
                seg.nx, seg.nw,
            );

            // --- time-window pass. NOTE: j outer, i inner. --------------------
            let mut tmax = 0.0f32;
            // Re-initialised per segment, which is why the stderr distance below
            // reports only the last segment. Reproduced.
            d10 = 10000.0;
            let mut window_s = Array2::<f32>::new(params::NQ, params::NP);
            let mut bet = 0.0f32;
            // Depth-major: j slowest. The subfault pass below goes the other way.
            for (i, j) in seg.depth_major() {
                // No `bet = vsh(1)` default here, unlike the subfault pass
                // below: if zet exceeds every depth, bet keeps its previous
                // value. Undefined on the very first subfault of the first
                // station in the Fortran; zero here.
                if let Some(ksrc) = (1..=j0).find(|&k| vmod.depth[k] >= geom.depth_km[(i, j)] as f64) {
                    bet = vmod.vsh[ksrc] as f32;
                }

                let rvf = rv.factor(geom.depth_km[(i, j)]);
                let alphat = alpha_t(seg.dipq, seg.rakeq, calpha);
                let fc_coeff = czero * (1.0 + fcfac) / alphat;

                // Path duration bin. Strict `>` means r0/d0/slp stay unset
                // if rlsu is exactly rdur(1) = 0.0; zero here rather than
                // the Fortran's undefined.
                let mut r0 = 0.0f32;
                let mut d0 = 0.0f32;
                let mut slp = 0.0f32;
                for kk in 1..=ndur {
                    if geom.slant_km[(i, j)] > rdur[kk] {
                        r0 = rdur[kk];
                        d0 = dpth[kk];
                        slp = dpdr[kk];
                    }
                }

                let fce = fc_coeff * rvf * bet / (dlm * pai);
                let tw0 = 1.0 / fce;
                let tw0 = moment_scale.sqrt() * tw0;
                let dpath = d0 + slp * (geom.slant_km[(i, j)] - r0);
                // VERSION1: no 81.92 s cap.
                window_s[(i, j)] = 2.12 * (tw0 + dpath);

                if window_s[(i, j)] > tmax {
                    tmax = window_s[(i, j)];
                }
                d10 = d10.min(geom.slant_km[(i, j)]);
            }

            let ntmax = (2.0 * tmax / dt) as i32 as usize;
            let mut np2 = 2usize;
            while np2 < ntmax {
                np2 *= 2;
            }
            if np2 > MM {
                eprintln!("np2= {np2} > array dimension mm= {MM}");
                eprintln!("need to recompile with larger array size, exiting...");
                // go to 9555: skips close(22).
                return Ok(());
            }

            let nfold = np2 / 2 + 1;
            let mfold = np2 / 2 - 1;
            let df = 1.0 / (np2 as f32 * dt);
            for i in 1..=nfold {
                freq[i] = df * (i - 1) as f32;
            }

            let mut spectrum: [Array1<Complex32>; 3] = [
                Array1::filled(np2, Complex32::ZERO),
                Array1::filled(np2, Complex32::ZERO),
                Array1::filled(np2, Complex32::ZERO),
            ];
            let mut subfault_acc: [Array1<f32>; 3] = [
                Array1::new(np2), Array1::new(np2), Array1::new(np2),
            ];
            let mut ray = RayState::default();

            let mut irandcnt = 1usize;

            // --- subfault pass. NOTE: i outer, j inner -- the OPPOSITE order to
            // the window pass above. irandcnt is consumed in THIS order. -------
            for (i, j) in seg.strike_major() {
                if seg.sddp[(i, j)] < 0.001 {
                    continue; // goto 4 lands on the inner loop's terminator
                }

                for il in 1..=np2 {
                    subfault_acc[0][il] = 0.0;
                    subfault_acc[1][il] = 0.0;
                    subfault_acc[2][il] = 0.0;
                }

                // This pass DOES default bet/ro before the lookup.
                let mut bet = vmod.vsh[1] as f32;
                let mut ro = vmod.rho[1] as f32;
                let ksrc = match (1..=j0).find(|&k| vmod.depth[k] >= geom.depth_km[(i, j)] as f64) {
                    Some(k) => {
                        bet = vmod.vsh[k] as f32;
                        ro = vmod.rho[k] as f32;
                        k
                    }
                    // bet and ro keep the layer-1 defaults set just above.
                    None => j0 + 1,
                };
                if ksrc == j0 + 1 {
                    // The Fortran prints 'wrong!' and carries on with
                    // ksrc = j0+1, which it then passes to get_sitefacs.
                    println!(" wrong!");
                }

                let rvf0 = rv.factor(geom.depth_km[(i, j)]);
                let mut rvf = rvf0;
                if rvsig1 > 0.0 {
                    irandcnt += 1;
                    rvf = rvf0 * (normal_deviates[irandcnt] * rvsig1).exp();
                    if rvf > rvfmax {
                        rvf = rvfmax;
                    }
                }

                let alphat = alpha_t(seg.dipq, seg.rakeq, calpha);
                let fc_coeff = czero * (1.0 + fcfac) / alphat;
                let fce = fc_coeff * rvf * bet / dlm / pai;
                let rise = seg.rist[(i, j)];

                let mode = 4; // hardwired SH
                for &ray_type in &config.rayset {
                    let kind = ray_type.kind();

                    // The tracing runs even for a straight ray: the Fortran calls
                    // gf_amp_tt unconditionally and overwrites the results below,
                    // and type 0 borrows type 1's tracing to do it.
                    let g = gf_amp_tt(
                        &mut ray, &vmod, j0, geom.depth_km[(i, j)], geom.horiz_km[(i, j)],
                        ray_type.trace_type(), mode,
                    );
                    let mut stime = g.stime;
                    let mut rpath = g.rpath;
                    let mut qbar = g.qbar;
                    let mut sub_tstart = stime - tw_eps * window_s[(i, j)];

                    if kind == RayKind::StraightRay {
                        rpath = geom.slant_km[(i, j)];
                        qbar = rpath / (bet * 150.0);
                        stime = rpath / 3.7;
                        sub_tstart = 0.7 * stime;
                    }

                    let tw = window_s[(i, j)];
                    for kf in 1..=3 {
                        let mut fmx1 = fmx;
                        if fmx1 > 15.0 && kf == 3 {
                            fmx1 = 15.0;
                        }
                        stoc_f(
                            &mut rng, np2, rpath, tw, tw_eps, tw_eta, bet, ro, dt,
                            subevent_moment, dlm, fce, fmx1, akapp,
                            &mut spectrum[kf - 1], &freq, qbar, qfexp, moment_scale,
                        );
                    }

                    if config.site_amp {
                        get_sitefacs(&vmod, ksrc, nsfac, &siteamp_log_freq, &mut siteamp_factors);
                        for k in 0..3 {
                            siteamp(np2, &mut spectrum[k], &freq, nsfac, &siteamp_log_freq, &siteamp_factors);
                        }
                    }
                    // famprand is dead: fasig1 = fasig2 = 0.

                    // Incidence angle from the ray parameter: sin(i)/vs = p0.
                    // th = i for a downgoing ray, pi - i for upgoing.
                    let p0 = g.rp0;
                    let incidence =
                        if bet * p0 > 1.0 { 0.5 * pai } else { (bet * p0).asin() };
                    let th = match kind {
                        // The straight-ray approximation ignores the traced ray
                        // parameter and uses the geometric take-off angle.
                        RayKind::StraightRay => geom.takeoff_rad[(i, j)],
                        RayKind::Upgoing => pai - incidence,
                        RayKind::Downgoing => incidence,
                    };
                    let pa = geom.azimuth_rad[(i, j)];

                    let cmp = -90.0 * pu;
                    radfrq_lin(&mut rng, stra, dipa, raka, pa, th, &freq, nfold, cmp, nr, &mut radiation);
                    highcor_f(nfold, mfold, np2, &mut spectrum[0], &mut subfault_acc[0], &radiation);

                    let cmp = 0.0f32;
                    radfrq_lin(&mut rng, stra, dipa, raka, pa, th, &freq, nfold, cmp, nr, &mut radiation);
                    highcor_f(nfold, mfold, np2, &mut spectrum[1], &mut subfault_acc[1], &radiation);

                    radv_lin(stra, dipa, raka, pa, th, &freq, nfold, &radv_rand_a, &radv_rand_b, nr, &mut radiation);
                    highcor_f(nfold, mfold, np2, &mut spectrum[2], &mut subfault_acc[2], &radiation);

                    // Rupture time at this subfault.
                    let mut ratim;
                    if let Some(vr) = config.rupture_velocity_override {
                        let xra = seg.shyp - (i as f32 - 0.5 * (seg.nx as f32 + 1.0)) * seg.dx;
                        let yra = seg.dhyp - (j as f32 - 0.5) * seg.dw;
                        ratim = (xra * xra + yra * yra).sqrt() / vr;
                        if irand > 0 {
                            ratim += (rng.rand_numb() - 0.5) * 0.1 * ratim;
                        }
                    } else {
                        ratim = seg.rupt[(i, j)];
                    }

                    // int() truncates toward zero, so a negative
                    // sub_tstart makes kst smaller, possibly negative.
                    let kst = (ratim / dt) as i32 + (sub_tstart / dt) as i32;

                    for _k in 1..=nsum {
                        let si = rng.rand_numb();
                        let dris = si * rise / dt;
                        let mut k2 = nint(dris);
                        if nsum == 1 {
                            k2 = 0;
                        }
                        let k2 = k2 + kst;
                        let kend = (k2 + np2 as i32).min(ndata as i32);

                        let sd = seg.sddp[(i, j)];
                        let mut li = k2;
                        while li <= kend {
                            // Writes below index 1 go before DS in the
                            // Fortran and are never read back, since the
                            // output reads DS(1..ndata). Discarded.
                            if li >= 1 {
                                let idx = (li - k2) as usize;
                                let lu = li as usize;
                                acc[(1, lu)] += sd * subfault_acc_at(&subfault_acc, 1, idx);
                                acc[(2, lu)] += sd * subfault_acc_at(&subfault_acc, 2, idx);
                                acc[(3, lu)] += sd * subfault_acc_at(&subfault_acc, 3, idx);
                            }
                            li += 1;
                        }
                    }
                }
            }
        }

        // iftt = 0, so filter3d is not called.
        if io.iftt > 0 {
            unreachable!("filter3d is dead under the production deck (ift = 0)");
        }

        // Peak amplitudes: computed and, under BINMOD, not written anywhere.
        let mut amx = [0.0f32; 3];
        for i in 1..=ndata {
            for l in 0..3 {
                amx[l] = amx[l].max(acc[(l + 1, i)].abs());
            }
        }
        let _ = amx;
        let _ = delay;

        // Stream write: component fastest, so 090/000/ver per time sample.
        let mut bytes = Vec::with_capacity(ndata * 3 * 4);
        for i in 1..=ndata {
            for l in 1..=3 {
                bytes.extend_from_slice(&acc[(l, i)].to_le_bytes());
            }
        }
        out.write_all(&bytes)?;

        // c(2) = d10, written as (1x,f10.4): a leading blank from the 1x, THEN
        // a width-10 field. Dropping the 1x gives 10 characters instead of 11 --
        // harmless to hf_sim.py, which does float(stderr.strip()), but it is
        // still an interface difference and the parity gate flags it.
        writeln!(stderr, " {:>10.4}", d10)?;
    }

    Ok(())
}

/// Scalars derived from the slip model before any station is simulated.
struct SourceScale {
    /// Average subfault dimension `sqrt(dx*dw)`, averaged over segments, km.
    dlm: f32,
    /// Total seismic moment. Derived from the summed subfault moments when the
    /// deck asked for it with a negative value.
    sm: f32,
    /// Mean subfault corner frequency, normalised. Feeds `bigc3`.
    fce_avg: f32,
    /// Corner frequency of the whole event, from the mean rise time.
    fcmain: f32,
    /// Count of subfaults whose relative moment exceeds 0.001 — the same
    /// threshold the subfault pass uses to skip a subfault entirely.
    nstot: usize,
}

/// Convert relative slip to relative moment, then normalise to unit average
/// weight, mutating `stoch.segments[..].sddp` in place.
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
/// `nstot` with the other, so only the second survives — hence only that one is
/// returned.
#[allow(clippy::too_many_arguments)]
fn normalise_source(
    stoch: &mut StochModel,
    j0: usize,
    vmod_in: &VmodIn,
    rv: &RuptureVelocityTaper,
    pu: f32,
    pai: f32,
    czero: f32,
    calpha: f32,
    fcfac: f32,
    moment: Option<f32>,
) -> SourceScale {
    let nevnt = stoch.segments.len();

    // --- pass 1: average subfault size, and a max slip that goes nowhere ------
    let mut dlm = 0.0f32;
    let mut amx2 = 0.0f32;
    for s in &stoch.segments {
        dlm = (s.dx * s.dw).sqrt() + dlm;
        for (i, j) in s.depth_major() {
            amx2 = amx2.max(s.sddp[(i, j)].abs());
        }
    }
    let _slip_max = amx2;
    dlm /= nevnt as f32;

    // --- pass 2: relative slip to relative moment; average fc and rise time ---
    // Frozen switch: the Fortran sets this to 1 and then to 0, so 0 wins and both
    // slip-weighted branches below are unreachable. Kept because collapsing it
    // bakes in a decision someone once left adjustable -- see REFACTOR.md §2.6.
    let islip_weight_avg = 0;
    let mut nstot = 0usize;
    let mut xsum = 0.0f32;
    let mut fce_avg = 0.0f32;
    let mut trise_avg = 0.0f32;

    for iv in 0..nevnt {
        let dwdj = stoch.segments[iv].dw * (stoch.segments[iv].dipq * pu).sin();
        let nw = stoch.segments[iv].nw;
        let nx = stoch.segments[iv].nx;
        for j in 1..=nw {
            let zdep = stoch.segments[iv].dtop + (j as f32 - 0.5) * dwdj;
            // Layer lookup. Falls through with k = j0+1 if zdep is below the
            // model, which the Fortran then indexes -- so the fall-through is
            // load-bearing, not an error path.
            let k = (1..=j0).find(|&kk| zdep <= vmod_in.depth0[kk]).unwrap_or(j0 + 1);
            let bet = vmod_in.vsh0[k] as f32;
            // vsh0 and rho0 are real*8 and dx/dw are real*4, so the WHOLE
            // product is computed in double (dx/dw promoted) and narrows only on
            // assignment to xmu, which is implicit real*4. Narrowing earlier
            // shifts every subfault moment by an ulp or two.
            let xmu = (vmod_in.vsh0[k] * vmod_in.vsh0[k] * vmod_in.rho0[k]
                * stoch.segments[iv].dx as f64
                * stoch.segments[iv].dw as f64) as f32;

            for i in 1..=nx {
                let v = xmu * stoch.segments[iv].sddp[(i, j)];
                stoch.segments[iv].sddp[(i, j)] = v;
                xsum += v;

                let rvf = rv.factor(zdep);
                let alphat =
                    alpha_t(stoch.segments[iv].dipq, stoch.segments[iv].rakeq, calpha);
                let fc_coeff = czero * (1.0 + fcfac) / alphat;
                let mut fce = fc_coeff * rvf * bet / (dlm * pai);
                let mut trise = stoch.segments[iv].rist[(i, j)];
                if islip_weight_avg == 1 {
                    fce = stoch.segments[iv].sddp[(i, j)] * fce;
                    trise = stoch.segments[iv].sddp[(i, j)] * trise;
                }
                fce_avg += fce;
                trise_avg += trise;
                nstot += 1;
            }
        }
    }

    // `None` means the deck asked for the moment to be derived from the summed
    // subfault moments.
    let sm = moment.unwrap_or(1.0e+20 * xsum);
    let xnorm = if islip_weight_avg == 1 { 1.0 / xsum } else { 1.0 / nstot as f32 };
    fce_avg *= xnorm;
    trise_avg *= xnorm;

    let fcoef = czero / (2.0 * pai);
    let fcmain = fcoef / trise_avg;

    // --- pass 3: normalise relative moments to average weight unity -----------
    let mut wsum = 0.0f32;
    let mut nstot = 0usize;
    for s in &stoch.segments {
        for (i, j) in s.depth_major() {
            if s.sddp[(i, j)] > 0.001 {
                wsum += s.sddp[(i, j)];
                nstot += 1;
            }
        }
    }
    let scale = nstot as f32 / wsum;
    for s in &mut stoch.segments {
        for (i, j) in s.depth_major() {
            s.sddp[(i, j)] *= scale;
        }
    }

    SourceScale { dlm, sm, fce_avg, fcmain, nstot }
}





/// The `alphaT` corner-frequency adjustment (2013-11-20).
///
/// Also appears three times identically. `fD` tapers with dip above 45 degrees;
/// `fR` peaks at a rake of 90 degrees. The rake is first wrapped into
/// `[-180, 180]` by repeated addition or subtraction of 360, which the Fortran
/// does with backward `goto`s.
fn alpha_t(avgdip: f32, rakeq: f32, calpha: f32) -> f32 {
    let mut fd = 0.0f32;
    if avgdip <= 90.0 && avgdip > 45.0 {
        fd = 1.0 - (avgdip - 45.0) / 45.0;
    } else if avgdip <= 45.0 && avgdip >= 0.0 {
        fd = 1.0;
    }

    let mut avgrak = rakeq;
    while avgrak < -180.0 {
        avgrak += 360.0;
    }
    while avgrak > 180.0 {
        avgrak -= 360.0;
    }

    let mut fr = 0.0f32;
    if avgrak <= 180.0 && avgrak >= 0.0 {
        // sqrt(x*x) rather than abs(x); the Fortran writes it this way.
        fr = 1.0 - ((avgrak - 90.0) * (avgrak - 90.0)).sqrt() / 90.0;
    }

    1.0 / (1.0 + fd * fr * calpha)
}
