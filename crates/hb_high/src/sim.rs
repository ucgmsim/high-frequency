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
    HfConfig, PathDurationModel, RayKind, StressParamAdjust,
};
use crate::fort::{truncate_toward_zero, Complex32};
use crate::geom::subfault_geometry;
use crate::highcor::apply_radiation_and_invert;
use crate::input::{insert_air_layer, StochModel};
use crate::radiation::{horizontal_radiation_spectrum, vertical_radiation_spectrum};
use crate::ray::green_function;
use crate::rng::{fill_normal_deviates, fill_uniform_deviates, Pcg32};
use crate::site::{site_amplification_factors, apply_site_amplification};
use crate::state::{params, RayState, VelocityModel, VelocityModelInput};
use crate::stoc::stochastic_spectrum;

/// `mm` and `mmv` as the **main program** sees them.
///
/// Under `VERSION1` the main program includes `params_no_window.h`, so both are
/// 262144 — not the 32769/180000 that every subroutine gets from `params.h`.
/// This matters here because `ndata` is clamped to `mmv` and
/// `fill_normal_deviates(mmv, ...)` draws exactly this many deviates.
const MMV: usize = params::MMV;

/// Index of the vertical component in the three-element per-component arrays.
///
/// The three are ordered 090, 000, vertical throughout, and that order is load-bearing:
/// the two horizontals draw 5,000 deviates each from the shared stream and the vertical
/// draws none. A `Component` enum is the right home for this — §2.8 batch 5 — but the
/// magic `3` it replaces was worth removing on its own.
const VERTICAL: usize = 2;

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

    let (mut rng, irand_after) = Pcg32::seed(irand);
    // init_random_seed mutates its argument, and the mutated value gates the
    // rupture-time jitter below.
    irand = irand_after;

    // `nr` = 1000 values, not `mmv` = 262144. `vertical_radiation_spectrum` reads
    // exactly `nr` of these, and `fill_uniform_deviates` only ever drew that many, so
    // the other 99.6% of each array was reserved, zeroed and never touched.
    let mut radv_rand_a = vec![0.0f32; nr];
    let mut radv_rand_b = vec![0.0f32; nr];
    fill_uniform_deviates(&mut rng, nr, &mut radv_rand_a);
    fill_uniform_deviates(&mut rng, nr, &mut radv_rand_b);

    let mut vmod = VelocityModel::new();
    // `ndata` samples, not `mmv`: the output loop reads `1..=ndata` and nothing else
    // touches this.
    // Three component traces, not a 2-D array. The Fortran's `DS(3, mmv)` was a 2-D
    // block because Fortran had no better option; here it is what it actually is, and it
    // now matches `spectrum` and `subfault_acc` beside it.
    //
    // §2.3 could only do this once §2.6's defect-1 fix landed: the column-major layout of
    // the `fort::Array2` this used to be was load-bearing for exactly one thing, the
    // `stdd(0,l)` alias across columns, and that read is gone. `Array2` itself is now gone
    // too -- this was one of its last two uses.
    let mut acc: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; ndata]);
    // This one STAYS at `mmv`, and the reason is not laziness. `fill_normal_deviates`
    // is called with `MMV` below, and the number of deviates drawn is part of the RNG
    // stream -- every subsequent draw depends on where the generator ended up. Shrinking
    // the allocation without shrinking the draw would be a buffer overrun; shrinking the
    // draw would change every waveform. See REFACTOR.md §2.6b.
    let mut normal_deviates = vec![0.0f32; MMV];
    let mut siteamp_factors = vec![0.0f32; nsfac];

    // ------------------------------------------------- the single station ---
    let mut d10 = 1000.0f32;

    if nlskip >= 0 {
        unreachable!("grandvel is dead under the production deck (nl_skip < 0)");
    } else {
        for k in 0..j0 {
            vmod[k].depth_km = vmod_in[k].depth_km as f64;
            vmod[k].thickness_km = vmod_in[k].thickness_km as f64;
            vmod[k].vp_km_s = vmod_in[k].vp_km_s;
            vmod[k].vsh_km_s = vmod_in[k].vsh_km_s;
            vmod[k].density_g_cm3 = vmod_in[k].density_g_cm3;
            vmod[k].attenuation_p = vmod_in[k].attenuation_p;
            vmod[k].attenuation_s = vmod_in[k].attenuation_s;
        }
    }

    if config.draws_normal_deviates() {
        // mmv deviates, not np2: this is the full 262144 under VERSION1.
        fill_normal_deviates(&mut rng, MMV, &mut normal_deviates);
    }

    for seg in &stoch.segments {
        let strike_rad = seg.strike_deg * deg_to_rad;
        let dip_rad = seg.dip_deg * deg_to_rad;
        let rake_rad = seg.rake_deg * deg_to_rad;

        let geom = subfault_geometry(
            seg.fault_lon_deg, seg.fault_lat_deg, station.stlon, station.stlat,
            seg.strike_deg, seg.dip_deg, seg.top_depth_km, seg.along_strike_offset_km,
            seg.subfault_length_km, seg.subfault_width_km,
            seg.along_strike_count, seg.down_dip_count,
        );

        // --- time-window pass. NOTE: j outer, i inner. --------------------
        let mut tmax = 0.0f32;
        // Re-initialised per segment, which is why the stderr distance below
        // reports only the last segment. Reproduced.
        d10 = 10000.0;
        // One entry per real subfault. Was `(NQ, NP)` = 600x100 regardless of the fault,
        // the last of the compile-time-ceiling allocations §2.6b set out to remove.
        let mut window_s = vec![0.0f32; seg.subfault_total()];
        let mut shear_velocity_km_s = 0.0f32;
        // Depth-major: j slowest. The subfault pass below goes the other way.
        for (i, j) in seg.depth_major() {
            let ray = geom.at(i, j);
            // No `shear_velocity_km_s = vsh_km_s(1)` default here, unlike the subfault pass
            // below: if zet exceeds every depth, shear_velocity_km_s keeps its previous
            // value. Undefined on the very first subfault of the first
            // station in the Fortran; zero here.
            if let Some(ksrc) = (0..j0).find(|&k| vmod[k].depth_km >= ray.depth_km as f64) {
                shear_velocity_km_s = vmod[ksrc].vsh_km_s as f32;
            }

            let rvf = rv.factor(ray.depth_km);
            let alphat = alpha_t(seg.dip_deg, seg.rake_deg, calpha);
            let fc_coeff = czero * (1.0 + fcfac) / alphat;

            // The last segment this distance is past. The Fortran scans the whole table
            // letting later matches overwrite earlier ones, which is `.last()` -- NOT
            // `.find()`, and the difference matters because the table is ascending so the
            // first match is the wrong end.
            //
            // Strict `>`, so a distance of exactly the first breakpoint (0.0) matches
            // NOTHING and the duration terms stay zero. The Fortran leaves them
            // undefined there; zero is this port's choice, and `unwrap_or` is where it
            // now lives rather than three loose initialisers.
            let bin = path_duration
                .iter()
                .take_while(|s| ray.slant_km > s.start_km)
                .last()
                .copied()
                .unwrap_or(DurationSegment { start_km: 0.0, duration_s: 0.0, slope_s_per_km: 0.0 });

            let fce = fc_coeff * rvf * shear_velocity_km_s / (dlm * pi);
            let tw0 = 1.0 / fce;
            let tw0 = moment_scale.sqrt() * tw0;
            let dpath = bin.duration_s + bin.slope_s_per_km * (ray.slant_km - bin.start_km);
            // VERSION1: no 81.92 s cap.
            let window = 2.12 * (tw0 + dpath);
            window_s[seg.grid_index(i, j)] = window;

            if window > tmax {
                tmax = window;
            }
            d10 = d10.min(ray.slant_km);
        }

        let ntmax = truncate_toward_zero(2.0 * tmax / dt) as usize;
        let mut np2 = 2usize;
        while np2 < ntmax {
            np2 *= 2;
        }
        let nfold = np2 / 2 + 1;
        let mfold = np2 / 2 - 1;
        // Sized from `np2` and allocated here rather than at `mm` before the loop:
        // `np2` is not known until the time-window pass above has produced `tmax`, and
        // both of these are per-segment quantities that are fully rewritten each time
        // round, so nothing carries across segments.
        let mut freq = vec![0.0f32; nfold];
        let mut radiation = vec![0.0f32; nfold];
        let df = 1.0 / (np2 as f32 * dt);
        // 0-based, which also removes the `- 1`: the axis is `df * bin`.
        for (bin, f) in freq.iter_mut().enumerate() {
            *f = df * bin as f32;
        }

        let mut spectrum: [Vec<Complex32>; 3] =
            std::array::from_fn(|_| vec![Complex32::ZERO; np2]);
        let mut subfault_acc: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.0f32; np2]);
        let mut ray = RayState::default();

        // 0-based. The Fortran starts this at 1 and PRE-increments, so its first read
        // is index 2, i.e. storage element 1 -- element 0 is never read. Starting at 0
        // and pre-incrementing lands on that same element.
        let mut irandcnt = 0usize;

        // --- subfault pass. NOTE: i outer, j inner -- the OPPOSITE order to
        // the window pass above. irandcnt is consumed in THIS order. -------
        for (i, j) in seg.strike_major() {
            let subfault = seg.at(i, j);
            if subfault.slip < 0.001 {
                continue; // goto 4 lands on the inner loop's terminator
            }
            let ray_geometry = geom.at(i, j);
            let subfault_window_s = window_s[seg.grid_index(i, j)];

            for component in &mut subfault_acc {
                component.fill(0.0);
            }

            // This pass DOES default shear_velocity_km_s/density_g_cm3 before the lookup.
            let mut shear_velocity_km_s = vmod[0].vsh_km_s as f32;
            let mut density_g_cm3 = vmod[0].density_g_cm3 as f32;
            let ksrc = match (0..j0).find(|&k| vmod[k].depth_km >= ray_geometry.depth_km as f64) {
                Some(k) => {
                    shear_velocity_km_s = vmod[k].vsh_km_s as f32;
                    density_g_cm3 = vmod[k].density_g_cm3 as f32;
                    k
                }
                // shear_velocity_km_s and density_g_cm3 keep the first layer's defaults.
                // `j0` is one PAST the last layer once 0-based, which is the Fortran's
                // `j0 + 1` and is read as such below and by site_amplification_factors.
                None => j0,
            };
            if ksrc == j0 {
                // The Fortran prints 'wrong!' and carries on with
                // ksrc = j0+1, which it then passes to site_amplification_factors.
                println!(" wrong!");
            }

            let base_rvf = rv.factor(ray_geometry.depth_km);
            let mut rvf = base_rvf;
            if rvsig1 > 0.0 {
                irandcnt += 1;
                rvf = base_rvf * (normal_deviates[irandcnt] * rvsig1).exp();
                if rvf > rvfmax {
                    rvf = rvfmax;
                }
            }

            let alphat = alpha_t(seg.dip_deg, seg.rake_deg, calpha);
            let fc_coeff = czero * (1.0 + fcfac) / alphat;
            let fce = fc_coeff * rvf * shear_velocity_km_s / dlm / pi;

            let mode = 4; // hardwired SH
            for &ray_type in &config.rayset {
                let kind = ray_type.kind();

                // The tracing runs even for a straight ray: the Fortran calls
                // green_function unconditionally and overwrites the results below,
                // and type 0 borrows type 1's tracing to do it.
                let g = green_function(
                    &mut ray, &vmod, j0, ray_geometry.depth_km, ray_geometry.horiz_km,
                    ray_type.trace_type(), mode,
                );
                let mut stime = g.stime;
                let mut rpath = g.rpath;
                let mut qbar = g.qbar;
                let mut sub_tstart = stime - tw_eps * subfault_window_s;

                if kind == RayKind::StraightRay {
                    rpath = ray_geometry.slant_km;
                    qbar = rpath / (shear_velocity_km_s * 150.0);
                    stime = rpath / 3.7;
                    sub_tstart = 0.7 * stime;
                }

                // 0-based, so the vertical is component 2 rather than the Fortran's
                // `kf == 3`. The three calls stay in this order: each draws `np2` normal
                // deviates from the shared stream.
                for (component, spec) in spectrum.iter_mut().enumerate() {
                    // Only the vertical is capped at 15 Hz.
                    let fmx1 = if component == VERTICAL { fmx.min(15.0) } else { fmx };
                    stochastic_spectrum(
                        &mut rng, np2, rpath, subfault_window_s, tw_eps, tw_eta,
                        shear_velocity_km_s, density_g_cm3, dt,
                        subevent_moment, dlm, fce, fmx1, akapp,
                        spec, &freq, qbar, qfexp,
                        moment_scale,
                    );
                }

                if config.site_amp {
                    site_amplification_factors(
                        &vmod, ksrc, nsfac, &siteamp_log_freq,
                        &mut siteamp_factors,
                    );
                    for component in &mut spectrum {
                        apply_site_amplification(
                            component, &freq, nsfac,
                            &siteamp_log_freq, &siteamp_factors,
                        );
                    }
                }
                // famprand is dead: fasig1 = fasig2 = 0.

                // Incidence angle from the ray parameter: sin(i)/vs = p0.
                // th = i for a downgoing ray, pi - i for upgoing.
                let p0 = g.rp0;
                let incidence =
                    if shear_velocity_km_s * p0 > 1.0 { 0.5 * pi } else { (shear_velocity_km_s * p0).asin() };
                let th = match kind {
                    // The straight-ray approximation ignores the traced ray
                    // parameter and uses the geometric take-off angle.
                    RayKind::StraightRay => ray_geometry.takeoff_rad,
                    RayKind::Upgoing => pi - incidence,
                    RayKind::Downgoing => incidence,
                };
                let pa = ray_geometry.azimuth_rad;

                let component_rad = -90.0 * deg_to_rad;
                horizontal_radiation_spectrum(
                        &mut rng, strike_rad, dip_rad, rake_rad, pa, th, &freq,
                        nfold, component_rad, nr, &mut radiation,
                    );
                apply_radiation_and_invert(nfold, mfold, &mut spectrum[0], &mut subfault_acc[0], &radiation);

                let component_rad = 0.0f32;
                horizontal_radiation_spectrum(
                        &mut rng, strike_rad, dip_rad, rake_rad, pa, th, &freq,
                        nfold, component_rad, nr, &mut radiation,
                    );
                apply_radiation_and_invert(nfold, mfold, &mut spectrum[1], &mut subfault_acc[1], &radiation);

                vertical_radiation_spectrum(
                        strike_rad, dip_rad, rake_rad, pa, th, &freq, nfold,
                        &radv_rand_a, &radv_rand_b, nr,
                        &mut radiation,
                    );
                apply_radiation_and_invert(nfold, mfold, &mut spectrum[2], &mut subfault_acc[2], &radiation);

                // Rupture time at this subfault.
                let mut ratim;
                if let Some(vr) = config.rupture_velocity_override {
                    let along_strike_centre = 0.5 * (seg.along_strike_count as f32 + 1.0);
                    let xra = seg.hypocentre_along_strike_km
                        - (i as f32 - along_strike_centre) * seg.subfault_length_km;
                    let yra = seg.hypocentre_down_dip_km - (j as f32 - 0.5) * seg.subfault_width_km;
                    ratim = (xra * xra + yra * yra).sqrt() / vr;
                    if irand > 0 {
                        ratim += (rng.next_f32() - 0.5) * 0.1 * ratim;
                    }
                } else {
                    ratim = subfault.rupture_time_s;
                }

                // Both terms truncate TOWARD ZERO, not toward negative infinity, so a
                // negative `sub_tstart` makes `kst` smaller and possibly negative. The
                // named shim is the point: a future edit to `.floor()` here would be
                // silent, and `kst` is the sample index the whole subfault lands on.
                let kst = truncate_toward_zero(ratim / dt) + truncate_toward_zero(sub_tstart / dt);

                // ONE DRAW, AND IT MUST STAY. The Fortran loops `k = 1, nsum` here,
                // draws a uniform, and turns it into a sub-event time offset `k2`. But
                // `nsum` was frozen at 1 in 2004, so the loop ran once and the very next
                // statement was `if (nsum.eq.1) k2 = 0` -- the offset was computed and
                // then unconditionally thrown away, taking the rise time with it.
                //
                // So the arithmetic is dead and is gone. The draw is not: it advances
                // the shared generator once per (subfault, ray), and every sample
                // produced after it depends on where the stream ends up. Deleting this
                // line as "obviously dead code" changes every waveform in the program.
                //
                // The offset it used to compute is a frozen switch, not a defect -- see
                // REFACTOR.md "Not defects: frozen switches". Reviving it needs the same
                // explicit sign-off collapsing it would have needed.
                let _stream_advance = rng.next_f32();

                // §2.6 defect 1 is fixed here: sample 1 of the subfault's trace lands on
                // `k2`, not on `k2 + 1`.
                //
                // The Fortran read `stdd(li - k2, l)`, so its first iteration read index
                // 0 -- one element before the column, which nothing ever writes -- and
                // every subfault's contribution arrived one sample late. See REFACTOR.md
                // §2.6 for the analysis and PORTING_RULES §7 for the aliasing that made
                // that read return zero rather than crash.
                //
                // The upper bound moves with it: reading `idx + 1` over the old range
                // would reach `subfault_acc[np2 + 1]`, past the end. The contribution is
                // `subfault_acc[1..=np2]` placed at `acc[k2 ..= k2 + np2 - 1]`.
                let k2 = kst;
                let kend = (k2 + np2 as i32 - 1).min(ndata as i32);

                let sd = subfault.slip;
                let mut li = k2;
                while li <= kend {
                    // `k2` can be negative. Writes below index 1 land before DS in the
                    // Fortran and are never read back, since the output reads
                    // DS(1..ndata), so they are discarded rather than reproduced.
                    if li >= 1 {
                        let idx = (li - k2) as usize;
                        // `li` is the Fortran's 1-based sample number, so the 0-based
                        // slot is one lower. The `fort::Array2` this used to be had a
                        // 1-based second subscript, which hid this.
                        let sample = li as usize - 1;
                        for component in 0..3 {
                            acc[component][sample] += sd * subfault_acc[component][idx];
                        }
                    }
                    li += 1;
                }
            }
        }
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
