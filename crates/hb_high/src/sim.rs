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
use crate::fort::{round_half_away_from_zero, Array1, Array2, Complex32};
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
const MM: usize = params::MM;
const MMV: usize = params::MMV;

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
    /// The transform length the fault needs exceeds the compiled array bound. The
    /// Fortran prints two lines and jumps to `9555`, which exits *without* closing
    /// the output unit — so it is a clean exit having written nothing, not a crash.
    TransformTooLong { np2: usize, mm: usize },
    /// Segment dimensions disagree, which the Fortran refuses.
    InconsistentSegments(String),
}

impl std::fmt::Display for SimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimError::TransformTooLong { np2, mm } => write!(
                f,
                "np2= {np2} > array dimension mm= {mm}\nneed to recompile with larger \
                 array size, exiting..."
            ),
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
    let mut siteamp_log_freq = Array1::<f32>::new(params::NLAYMAX);
    for i in 1..=nsfac {
        siteamp_log_freq[i] = fn_hz[i - 1].ln();
    }

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

    let nevnt = stoch.segments.len();

    for (k, s) in stoch.segments.iter().enumerate() {
        if s.dx != stoch.segments[0].dx {
            return Err(SimError::InconsistentSegments(format!(
                "dx({}) = {} not equal to dx(1) = {}, exiting...",
                k + 1, s.dx, stoch.segments[0].dx
            )));
        }
        if s.dw != stoch.segments[0].dw {
            return Err(SimError::InconsistentSegments(format!(
                "dw({}) = {} not equal to dw(1) = {}, exiting...",
                k + 1, s.dw, stoch.segments[0].dw
            )));
        }
    }

    // ------------------------------------------------- path duration model ---
    let PathDuration { ndur, rdur, dpth, dpdr } = path_duration_table(config.path_duration);

    // ------------------------------------------------------- air layer -------
    let (j0_air, nlskip_air) = insert_air_layer(&mut vmod_in, j0, nlskip);
    j0 = j0_air;
    nlskip = nlskip_air;

    // Resolved here rather than at parse time: the deep transition depths depend on
    // the deepest hypocentre, which is only known once the slip model is read.
    let rv = config.rupture_velocity.resolve(stoch.zhyp_max);

    // ------------------------------------------------ source normalisation ---
    let SourceScale { dlm, sm, nstot } =
        normalise_source(&mut stoch, j0, &vmod_in, deg_to_rad, config.moment);

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
    // why the Frankel operator in stochastic_spectrum carries the scaling instead.
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
    let moment_scale = sm / (subevent_moment * (1.0 * nstot as f32).sqrt());

    // ------------------------------------------------------------ stations ---
    let ndata = {
        let n = (duration / dt) as i32 as usize;
        n.min(MMV)
    };

    let (mut rng, irand_after) = Pcg32::seed(irand);
    // init_random_seed mutates its argument, and the mutated value gates the
    // rupture-time jitter below.
    irand = irand_after;

    // `nr` = 1000 values, not `mmv` = 262144. `vertical_radiation_spectrum` reads
    // exactly `nr` of these, and `fill_uniform_deviates` only ever drew that many, so
    // the other 99.6% of each array was reserved, zeroed and never touched.
    let mut radv_rand_a = Array1::<f32>::new(nr);
    let mut radv_rand_b = Array1::<f32>::new(nr);
    fill_uniform_deviates(&mut rng, nr, &mut radv_rand_a);
    fill_uniform_deviates(&mut rng, nr, &mut radv_rand_b);

    let mut vmod = VelocityModel::new();
    // `ndata` samples, not `mmv`: the output loop reads `1..=ndata` and nothing else
    // touches this.
    let mut acc = Array2::<f32>::new(3, ndata);
    // This one STAYS at `mmv`, and the reason is not laziness. `fill_normal_deviates`
    // is called with `MMV` below, and the number of deviates drawn is part of the RNG
    // stream -- every subsequent draw depends on where the generator ended up. Shrinking
    // the allocation without shrinking the draw would be a buffer overrun; shrinking the
    // draw would change every waveform. See REFACTOR.md §2.6b.
    let mut normal_deviates = Array1::<f32>::new(MMV);
    let mut siteamp_factors = Array1::<f32>::new(params::NLAYMAX);

    // ------------------------------------------------- the single station ---
    let mut d10 = 1000.0f32;

    if nlskip >= 0 {
        unreachable!("grandvel is dead under the production deck (nl_skip < 0)");
    } else {
        for k in 1..=j0 {
            vmod.depth_km[k] = vmod_in.depth_km[k] as f64;
            vmod.thickness_km[k] = vmod_in.thickness_km[k] as f64;
            vmod.vp_km_s[k] = vmod_in.vp_km_s[k];
            vmod.vsh_km_s[k] = vmod_in.vsh_km_s[k];
            vmod.density_g_cm3[k] = vmod_in.density_g_cm3[k];
            vmod.attenuation_p[k] = vmod_in.attenuation_p[k];
            vmod.attenuation_s[k] = vmod_in.attenuation_s[k];
        }
    }

    if config.draws_normal_deviates() {
        // mmv deviates, not np2: this is the full 262144 under VERSION1.
        fill_normal_deviates(&mut rng, MMV, &mut normal_deviates);
    }

    for iv in 0..nevnt {
        let seg = &stoch.segments[iv];
        let strike_rad = seg.strq * deg_to_rad;
        let dip_rad = seg.dipq * deg_to_rad;
        let rake_rad = seg.rakeq * deg_to_rad;

        let geom = subfault_geometry(
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
        let mut shear_velocity_km_s = 0.0f32;
        // Depth-major: j slowest. The subfault pass below goes the other way.
        for (i, j) in seg.depth_major() {
            // No `shear_velocity_km_s = vsh_km_s(1)` default here, unlike the subfault pass
            // below: if zet exceeds every depth, shear_velocity_km_s keeps its previous
            // value. Undefined on the very first subfault of the first
            // station in the Fortran; zero here.
            if let Some(ksrc) = (1..=j0).find(|&k| vmod.depth_km[k] >= geom.depth_km[(i, j)] as f64) {
                shear_velocity_km_s = vmod.vsh_km_s[ksrc] as f32;
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

            let fce = fc_coeff * rvf * shear_velocity_km_s / (dlm * pi);
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
            // `go to 9555` in the original: it prints and exits WITHOUT closing the
            // output unit, i.e. a clean exit having written nothing for this station.
            // The driver reproduces that from this error rather than doing it here,
            // because "print and exit successfully" is a driver decision.
            return Err(SimError::TransformTooLong { np2, mm: MM });
        }

        let nfold = np2 / 2 + 1;
        let mfold = np2 / 2 - 1;
        // Sized from `np2` and allocated here rather than at `mm` before the loop:
        // `np2` is not known until the time-window pass above has produced `tmax`, and
        // both of these are per-segment quantities that are fully rewritten each time
        // round, so nothing carries across segments.
        let mut freq = Array1::<f32>::new(nfold);
        let mut radiation = Array1::<f32>::new(nfold);
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

            // This pass DOES default shear_velocity_km_s/density_g_cm3 before the lookup.
            let mut shear_velocity_km_s = vmod.vsh_km_s[1] as f32;
            let mut density_g_cm3 = vmod.density_g_cm3[1] as f32;
            let ksrc = match (1..=j0).find(|&k| vmod.depth_km[k] >= geom.depth_km[(i, j)] as f64) {
                Some(k) => {
                    shear_velocity_km_s = vmod.vsh_km_s[k] as f32;
                    density_g_cm3 = vmod.density_g_cm3[k] as f32;
                    k
                }
                // shear_velocity_km_s and density_g_cm3 keep the layer-1 defaults set just above.
                None => j0 + 1,
            };
            if ksrc == j0 + 1 {
                // The Fortran prints 'wrong!' and carries on with
                // ksrc = j0+1, which it then passes to site_amplification_factors.
                println!(" wrong!");
            }

            let base_rvf = rv.factor(geom.depth_km[(i, j)]);
            let mut rvf = base_rvf;
            if rvsig1 > 0.0 {
                irandcnt += 1;
                rvf = base_rvf * (normal_deviates[irandcnt] * rvsig1).exp();
                if rvf > rvfmax {
                    rvf = rvfmax;
                }
            }

            let alphat = alpha_t(seg.dipq, seg.rakeq, calpha);
            let fc_coeff = czero * (1.0 + fcfac) / alphat;
            let fce = fc_coeff * rvf * shear_velocity_km_s / dlm / pi;
            let rise = seg.rist[(i, j)];

            let mode = 4; // hardwired SH
            for &ray_type in &config.rayset {
                let kind = ray_type.kind();

                // The tracing runs even for a straight ray: the Fortran calls
                // green_function unconditionally and overwrites the results below,
                // and type 0 borrows type 1's tracing to do it.
                let g = green_function(
                    &mut ray, &vmod, j0, geom.depth_km[(i, j)], geom.horiz_km[(i, j)],
                    ray_type.trace_type(), mode,
                );
                let mut stime = g.stime;
                let mut rpath = g.rpath;
                let mut qbar = g.qbar;
                let mut sub_tstart = stime - tw_eps * window_s[(i, j)];

                if kind == RayKind::StraightRay {
                    rpath = geom.slant_km[(i, j)];
                    qbar = rpath / (shear_velocity_km_s * 150.0);
                    stime = rpath / 3.7;
                    sub_tstart = 0.7 * stime;
                }

                let tw = window_s[(i, j)];
                for kf in 1..=3 {
                    let mut fmx1 = fmx;
                    if fmx1 > 15.0 && kf == 3 {
                        fmx1 = 15.0;
                    }
                    stochastic_spectrum(
                        &mut rng, np2, rpath, tw, tw_eps, tw_eta, shear_velocity_km_s, density_g_cm3, dt,
                        subevent_moment, dlm, fce, fmx1, akapp,
                        &mut spectrum[kf - 1], &freq, qbar, qfexp, moment_scale,
                    );
                }

                if config.site_amp {
                    site_amplification_factors(&vmod, ksrc, nsfac, &siteamp_log_freq, &mut siteamp_factors);
                    for k in 0..3 {
                        apply_site_amplification(np2, &mut spectrum[k], &freq, nsfac, &siteamp_log_freq, &siteamp_factors);
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
                    RayKind::StraightRay => geom.takeoff_rad[(i, j)],
                    RayKind::Upgoing => pi - incidence,
                    RayKind::Downgoing => incidence,
                };
                let pa = geom.azimuth_rad[(i, j)];

                let component_rad = -90.0 * deg_to_rad;
                horizontal_radiation_spectrum(&mut rng, strike_rad, dip_rad, rake_rad, pa, th, &freq, nfold, component_rad, nr, &mut radiation);
                apply_radiation_and_invert(nfold, mfold, np2, &mut spectrum[0], &mut subfault_acc[0], &radiation);

                let component_rad = 0.0f32;
                horizontal_radiation_spectrum(&mut rng, strike_rad, dip_rad, rake_rad, pa, th, &freq, nfold, component_rad, nr, &mut radiation);
                apply_radiation_and_invert(nfold, mfold, np2, &mut spectrum[1], &mut subfault_acc[1], &radiation);

                vertical_radiation_spectrum(strike_rad, dip_rad, rake_rad, pa, th, &freq, nfold, &radv_rand_a, &radv_rand_b, nr, &mut radiation);
                apply_radiation_and_invert(nfold, mfold, np2, &mut spectrum[2], &mut subfault_acc[2], &radiation);

                // Rupture time at this subfault.
                let mut ratim;
                if let Some(vr) = config.rupture_velocity_override {
                    let xra = seg.shyp - (i as f32 - 0.5 * (seg.nx as f32 + 1.0)) * seg.dx;
                    let yra = seg.dhyp - (j as f32 - 0.5) * seg.dw;
                    ratim = (xra * xra + yra * yra).sqrt() / vr;
                    if irand > 0 {
                        ratim += (rng.next_f32() - 0.5) * 0.1 * ratim;
                    }
                } else {
                    ratim = seg.rupt[(i, j)];
                }

                // int() truncates toward zero, so a negative
                // sub_tstart makes kst smaller, possibly negative.
                let kst = (ratim / dt) as i32 + (sub_tstart / dt) as i32;

                for _k in 1..=nsum {
                    let si = rng.next_f32();
                    let dris = si * rise / dt;
                    let mut k2 = round_half_away_from_zero(dris);
                    if nsum == 1 {
                        k2 = 0;
                    }
                    let k2 = k2 + kst;
                    // §2.6 defect 1 is fixed here: sample 1 of the subfault's trace
                    // now lands on `k2`, not on `k2 + 1`.
                    //
                    // The Fortran read `stdd(li - k2, l)`, so its first iteration read
                    // index 0 -- one element before the column, which nothing ever
                    // writes -- and every subfault's contribution arrived one sample
                    // late. See REFACTOR.md §2.6 for the analysis and PORTING_RULES §7
                    // for the aliasing that made that read return zero rather than
                    // crash.
                    //
                    // The upper bound moves with it: reading `idx + 1` over the old
                    // range would reach `subfault_acc[np2 + 1]`, past the end. The
                    // contribution is `subfault_acc[1..=np2]` placed at
                    // `acc[k2 ..= k2 + np2 - 1]`.
                    let kend = (k2 + np2 as i32 - 1).min(ndata as i32);

                    let sd = seg.sddp[(i, j)];
                    let mut li = k2;
                    while li <= kend {
                        // `k2` can be negative. Writes below index 1 land before DS in
                        // the Fortran and are never read back, since the output reads
                        // DS(1..ndata), so they are discarded rather than reproduced.
                        if li >= 1 {
                            let idx = (li - k2) as usize + 1;
                            let lu = li as usize;
                            acc[(1, lu)] += sd * subfault_acc[0][idx];
                            acc[(2, lu)] += sd * subfault_acc[1][idx];
                            acc[(3, lu)] += sd * subfault_acc[2][idx];
                        }
                        li += 1;
                    }
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
    let mut out = Vec::with_capacity(ndata * 3);
    for i in 1..=ndata {
        for l in 1..=3 {
            out.push(acc[(l, i)]);
        }
    }

    Ok(Simulation { ndata, dt, acc: out, d10_km: d10 })
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


/// Scalars derived from the slip model before any station is simulated.
struct SourceScale {
    /// Average subfault dimension `sqrt(dx*dw)`, averaged over segments, km.
    dlm: f32,
    /// Total seismic moment. Derived from the summed subfault moments when the
    /// deck asked for it with a negative value.
    sm: f32,
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
    layer_count: usize,
    vmod_in: &VelocityModelInput,
    deg_to_rad: f32,
    moment: Option<f32>,
) -> SourceScale {
    let nevnt = stoch.segments.len();

    // --- pass 1: average subfault size ----------------------------------------
    // The Fortran also accumulates `slip_max` over every subfault here. Nothing
    // reads it: its only consumer is the commented-out `!print*,'Maximum slip '`
    // at hb_high_ref.f:683. Dropped, along with the O(nstot) loop that fed it.
    let mut dlm = 0.0f32;
    for s in &stoch.segments {
        dlm = (s.dx * s.dw).sqrt() + dlm;
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
    // `nstot` count (which only normalised them), and `islip_weight_avg` -- a frozen
    // switch whose two branches only ever weighted these dead quantities.
    //
    // What remains live: the in-place slip-to-moment conversion, and `xsum`, which
    // supplies the total moment when the deck asks for it to be derived.
    let mut xsum = 0.0f32;

    for iv in 0..nevnt {
        let dwdj = stoch.segments[iv].dw * (stoch.segments[iv].dipq * deg_to_rad).sin();
        let nw = stoch.segments[iv].nw;
        let nx = stoch.segments[iv].nx;
        for j in 1..=nw {
            let zdep = stoch.segments[iv].dtop + (j as f32 - 0.5) * dwdj;
            // Layer lookup. Falls through with k = j0+1 if zdep is below the
            // model, which the Fortran then indexes -- so the fall-through is
            // load-bearing, not an error path.
            let k = (1..=layer_count).find(|&kk| zdep <= vmod_in.depth_km[kk]).unwrap_or(layer_count + 1);
            // vsh_km_s and density_g_cm3 are real*8 and dx/dw are real*4, so the WHOLE
            // product is computed in double (dx/dw promoted) and narrows only on
            // assignment to xmu, which is implicit real*4. Narrowing earlier
            // shifts every subfault moment by an ulp or two.
            let xmu = (vmod_in.vsh_km_s[k] * vmod_in.vsh_km_s[k] * vmod_in.density_g_cm3[k]
                * stoch.segments[iv].dx as f64
                * stoch.segments[iv].dw as f64) as f32;

            for i in 1..=nx {
                let v = xmu * stoch.segments[iv].sddp[(i, j)];
                stoch.segments[iv].sddp[(i, j)] = v;
                xsum += v;
            }
        }
    }

    // `None` means the deck asked for the moment to be derived from the summed
    // subfault moments.
    let sm = moment.unwrap_or(1.0e+20 * xsum);

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

    SourceScale { dlm, sm, nstot }
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
