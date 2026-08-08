//! The production-scale baseline: one real Alpine Fault deck, five stations, and the
//! counters every optimisation stage is judged against.
//!
//! # Why this is not in `whole.rs`
//!
//! `whole.rs` builds its faults in code, because a benchmark wants fixed inputs of a known
//! size. That reasoning holds for the three grid shapes there and breaks here: the thing
//! being measured is a *real* rupture's structure — 189 segments whose rupture times run to
//! 383 s, diced 1.6 km, recorded for 470 s — and no constructor in a source file is going to
//! reproduce that honestly. So the slip model is read, and the two inputs that can be pinned
//! are pinned: the velocity model is the crate's own committed fixture, and the configuration
//! is written out below rather than parsed.
//!
//! # What it prints, and which numbers matter
//!
//! Wall clock drifts 1–3% on a loaded box, which is the same size as several of the effects
//! this is meant to adjudicate — `PROFILE.md` says so and it is why nothing marginal in this
//! repo is settled on a clock. So the load-bearing output is the **census**: pairs attempted,
//! pairs whose window fell entirely outside the record, and transform samples computed
//! against samples that reached it. Those are integers, they are deterministic, and they are
//! immune to whatever else the machine is doing.
//!
//! # Running it
//!
//! ```text
//! HB_STOCH=~/tmp/alpine_hf_test/realisation.stoch cargo bench --bench alpine
//! perf stat -e instructions,cycles -- cargo bench --bench alpine
//! ```
//!
//! `HB_STOCH` defaults to the committed `alpine_base_r1.stoch`, which is a single 257x11
//! segment — enough to exercise the path, an order of magnitude short of the real thing.
//! `HB_STATIONS` takes a comma-separated subset of the names below; a full five-station run
//! on the real deck is minutes, not seconds.

use std::path::{Path, PathBuf};
use std::time::Instant;

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::input::{Segment, Slip, Station, StochModel, Subfault, build_velocity_model};
use hb_high::state::{InputLayer, VelocityModelInput};

/// The `hf` block of `Rupture 71072`, the realisation this baseline is taken from, and the
/// `domain.duration` and `dt` its production run used.
///
/// Written out rather than parsed because a benchmark's inputs should be readable in the
/// benchmark. Two of these are not the file's literal values and both are the deck reader's
/// job on the production path:
///
/// * `calpha` is `-99.0` in the file, which is the "use the default" sentinel, so 0.1 stands
///   here. Getting this wrong makes every non-strike-slip corner frequency negative.
/// * `vs_moho` is `999.9`, i.e. above every layer, so the model is not truncated.
const DURATION_S: f32 = 470.0;
const DT_S: f32 = 0.005;
const VS_MOHO_KM_S: f64 = 999.9;

fn config() -> HfConfig {
    HfConfig {
        source: SourceParameters {
            stress_drop_bars: 50.0,
            czero: 2.1,
            calpha: 0.1,
            rupture_velocity: RuptureVelocity {
                frac: 0.8,
                shallow: 0.6,
                deep: 0.6,
                rv_sig1: 0.1,
            },
        },
        path: PathParameters {
            rayset: vec![RayType(1)],
            q_exponent: 0.6,
            path_duration: PathDurationModel::from_deck(11)
                .expect("11 is Boore & Thompson (2014) WUS"),
        },
        site: SiteParameters {
            kappa_s: 0.045,
            f_max_hz: 10.0,
        },
        record: RecordParameters {
            duration_s: DURATION_S,
            dt_s: DT_S,
        },
    }
}

/// Five stations spanning the geometry, nearest grid point to each named place.
///
/// A random sample would be the wrong instrument. `PROFILE.md` records the same 112 subfaults
/// costing 8.6x more under one geometry than another, so runtime is a function of where the
/// station is, and a mean over a sample hides exactly the effect every stage here targets.
/// These bracket it: on the fault, near it, and 250, 500 and 800 km out. Taupo is the
/// northern limit worth simulating for this rupture.
const STATIONS: &[(&str, f32, f32)] = &[
    ("FJDS", -43.389137, 170.184_23),    // Franz Josef, on the fault
    ("HMCS", -42.716922, 170.963_96),    // Hokitika
    ("dJRHSCA", -43.531956, 172.635_47), // Christchurch
    ("kBN8GWE", -41.288177, 174.776_98), // Wellington
    ("TPPS", -38.686325, 176.067_47),    // Taupo
];

/// `hf_seed` from the realisation, offset per station.
///
/// Every station gets an independent stream from its own seed, so the offset is an identity
/// rather than an ordering — `Simulator::run` takes `&self` and nothing crosses between them.
const HF_SEED: u64 = 362_950_150;

/// Record RMS at one station over `count` seeds, with the scatter that makes the mean
/// interpretable.
///
/// **The scatter is the point, not decoration.** A single seed's record energy can move 20%
/// between two correct realisations, so a before-and-after pair at one seed says nothing. What
/// can be compared is the mean against the standard error on the difference, which is what
/// this prints.
fn seed_scan(simulator: &hb_high::sim::Simulator, wanted: &Option<Vec<String>>, count: u64) {
    println!(
        "{:<10} {:>6}  {:>12} {:>12} {:>9}",
        "station", "seeds", "mean RMS", "sd", "sd/mean"
    );

    for (index, &(name, latitude, longitude)) in STATIONS.iter().enumerate() {
        if let Some(wanted) = wanted
            && !wanted.iter().any(|w| w == name)
        {
            continue;
        }

        let root = HF_SEED.wrapping_add((index as u64) << 32);
        let values: Vec<f64> = (0..count)
            .map(|k| {
                let station = Station {
                    name: name.to_owned(),
                    latitude,
                    longitude,
                };
                let acc = simulator.run(station, root.wrapping_add(k)).acc;
                // Over all three components together: one number per record, which is what
                // "record RMS" means and what the earlier adjudication compared.
                let sum: f64 = acc.iter().map(|&x| x as f64 * x as f64).sum();
                (sum / acc.len() as f64).sqrt()
            })
            .collect();

        let mean = values.iter().sum::<f64>() / count as f64;
        // Sample standard deviation: the seeds are a sample of the realisation ensemble, not
        // the whole of it.
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (count as f64 - 1.0);
        let sd = variance.sqrt();
        println!(
            "{name:<10} {count:>6}  {mean:>12.6} {sd:>12.6} {:>8.2}%",
            100.0 * sd / mean
        );
    }
}

/// How runtime scales with the two things a campaign can choose: how much fault to rupture and
/// how many stations to record it at. Emits CSV on stdout.
///
/// # The two sweeps measure different things and neither substitutes for the other
///
/// **Fault size** is swept by taking the first `k` segments — a shorter rupture on the same
/// fault system, which is what a smaller event actually looks like — against a fixed station
/// set. It is the sweep with something to discover: runtime is not linear in subfault count,
/// because a longer rupture also puts subfaults further from the station and lengthens their
/// windows.
///
/// **Station count** is swept against a fixed fault, over a nested prefix of one shuffled
/// sample so each larger set contains the smaller. It should be linear by construction —
/// `Simulator::run` takes `&self` and shares nothing between stations — so what it is really
/// testing is that claim, and the per-station scatter it reports is the honest measure of how
/// badly a single station predicts a campaign.
fn scaling(
    slip: &StochModel,
    vmod: &VelocityModelInput,
    config: &HfConfig,
    stations: &[(String, f32, f32)],
) {
    println!(
        "sweep,segments,subfaults,stations,wall_s,wall_s_per_station,pairs,samples,useful_pct"
    );

    let run = |sweep: &str, segments: usize, count: usize| {
        let subset = StochModel::new(slip.segments[..segments].to_vec());
        let subfaults: usize = subset.segments.iter().map(Segment::subfault_total).sum();
        let simulator = hb_high::sim::Simulator::new(config, &subset, vmod)
            .expect("a subset of a stoch file is diced the same way as the whole");

        let start = Instant::now();
        let (mut pairs, mut computed, mut landed) = (0usize, 0usize, 0usize);
        for (index, (name, latitude, longitude)) in stations.iter().take(count).enumerate() {
            let station = Station {
                name: name.clone(),
                latitude: *latitude,
                longitude: *longitude,
            };
            let census = simulator
                .run(station, HF_SEED.wrapping_add(index as u64))
                .census;
            pairs += census.pairs_attempted;
            computed += census.samples_computed;
            landed += census.samples_accumulated;
        }
        let wall = start.elapsed().as_secs_f64();
        let useful = if computed == 0 {
            0.0
        } else {
            100.0 * landed as f64 / computed as f64
        };
        println!(
            "{sweep},{segments},{subfaults},{count},{wall:.3},{:.3},{pairs},{computed},{useful:.2}",
            wall / count as f64
        );
    };

    // Roughly halving the fault each step, down to a single segment.
    let total = slip.segments.len();
    let mut sizes: Vec<usize> = Vec::new();
    let mut size = total;
    while size >= 1 {
        sizes.push(size);
        size /= 2;
    }
    sizes.reverse();
    for &segments in &sizes {
        run("fault", segments, 4.min(stations.len()));
    }

    // A fault big enough to be representative and small enough that 31 station-runs is
    // minutes rather than an hour.
    let mid = (total / 4).max(1);
    let mut count = 1;
    while count <= stations.len() {
        run("stations", mid, count);
        count *= 2;
    }
}

/// Read `name lat lon` per line. Used by the scaling sweep, which needs more stations than the
/// five named ones and needs them nested.
fn stations_from_file(path: &Path) -> Vec<(String, f32, f32)> {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let mut field = line.split_ascii_whitespace();
            let mut next = || field.next().expect("each line is `name lat lon`");
            let name = next().to_owned();
            let latitude: f32 = next().parse().expect("latitude");
            let longitude: f32 = next().parse().expect("longitude");
            (name, latitude, longitude)
        })
        .collect()
}

fn repository_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Read the crate's committed velocity-model fixture: a layer count, then one
/// `thickness vp vs rho qp qs` row each.
///
/// This fixture is bit-identical to `Rupture 71072`'s own `hf_velocity_model_1d`, checked
/// field by field over all 34 layers, which is why the baseline needs only the slip model
/// from outside the repository.
fn velocity_model() -> VelocityModelInput {
    let path = repository_root().join("harness/fixtures/velocity_model");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut tokens = text.split_ascii_whitespace();
    let count: usize = tokens
        .next()
        .and_then(|t| t.parse().ok())
        .expect("the fixture starts with a layer count");

    let layers: Vec<InputLayer> = (0..count)
        .map(|k| {
            let mut next = || -> f64 {
                tokens
                    .next()
                    .and_then(|t| t.parse().ok())
                    .unwrap_or_else(|| panic!("layer {k} of {count} is short or malformed"))
            };
            InputLayer {
                // Filled by `build_velocity_model`, which accumulates it from the thicknesses.
                depth_km: 0.0,
                thickness_km: next() as f32,
                vp_km_s: next(),
                vsh_km_s: next(),
                density_g_cm3: next(),
                attenuation_p: next() as f32,
                attenuation_s: next() as f32,
            }
        })
        .collect();

    build_velocity_model(&layers, VS_MOHO_KM_S).expect("the fixture model is well formed")
}

/// Read a `.stoch` slip model.
///
/// The format is whitespace-delimited throughout, so it is one token stream rather than a
/// line parser:
///
/// ```text
/// segment_count
/// per segment:  lon lat nx nw dx dw
///               strike dip rake top_depth hyp_along hyp_down
///               nx*nw slip, then nx*nw rise time, then nx*nw rupture time
/// ```
///
/// Each block runs down-dip rows outermost with the along-strike index fastest, which is the
/// order [`Segment::new`] wants its grid in — so the values go straight in with no transpose.
fn slip_model(path: &Path) -> StochModel {
    let text =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
    let mut tokens = text.split_ascii_whitespace();
    let mut next = |what: &str| -> f64 {
        tokens
            .next()
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| panic!("{} ran out at {what}", path.display()))
    };

    let segment_count = next("the segment count") as usize;
    let segments: Vec<Segment> = (0..segment_count)
        .map(|_| {
            let fault_lon_deg = next("longitude") as f32;
            let fault_lat_deg = next("latitude") as f32;
            let along_strike_count = next("nx") as usize;
            let down_dip_count = next("nw") as usize;
            let subfault_length_km = next("dx") as f32;
            let subfault_width_km = next("dw") as f32;
            let strike_deg = next("strike") as f32;
            let dip_deg = next("dip") as f32;
            let rake_deg = next("rake") as f32;
            let top_depth_km = next("top depth") as f32;
            let hypocentre_along_strike_km = next("along-strike hypocentre") as f32;
            let hypocentre_down_dip_km = next("down-dip hypocentre") as f32;

            let total = along_strike_count * down_dip_count;
            let mut block = |what: &'static str| -> Vec<f32> {
                (0..total).map(|_| next(what) as f32).collect()
            };
            let (slip, rise, rupture) = (block("slip"), block("rise time"), block("rupture time"));

            Segment::builder()
                .fault_lon_deg(fault_lon_deg)
                .fault_lat_deg(fault_lat_deg)
                .along_strike_count(along_strike_count)
                .down_dip_count(down_dip_count)
                .subfault_length_km(subfault_length_km)
                .subfault_width_km(subfault_width_km)
                .strike_deg(strike_deg)
                .dip_deg(dip_deg)
                .rake_deg(rake_deg)
                .top_depth_km(top_depth_km)
                .hypocentre_along_strike_km(hypocentre_along_strike_km)
                .hypocentre_down_dip_km(hypocentre_down_dip_km)
                .subfaults(
                    (0..total)
                        .map(|k| Subfault {
                            slip: Slip(slip[k]),
                            rise_time_s: rise[k],
                            rupture_time_s: rupture[k],
                        })
                        .collect(),
                )
                .build()
        })
        .collect();

    StochModel::new(segments)
}

fn main() {
    let stoch = std::env::var("HB_STOCH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repository_root().join("harness/fixtures/stoch/alpine_base_r1.stoch"));
    let wanted: Option<Vec<String>> = std::env::var("HB_STATIONS")
        .ok()
        .map(|s| s.split(',').map(|n| n.trim().to_owned()).collect());

    let slip = slip_model(&stoch);
    let subfaults: usize = slip.segments.iter().map(Segment::subfault_total).sum();
    let vmod = velocity_model();
    let config = config();

    let build = Instant::now();
    let simulator = hb_high::sim::Simulator::new(&config, &slip, &vmod)
        .expect("every segment of a stoch file is diced the same way");
    let setup_ms = build.elapsed().as_secs_f64() * 1e3;

    println!("deck      {}", stoch.display());
    println!(
        "          {} segments, {subfaults} subfaults, {} rays, {:.0} s record at dt {}",
        slip.segments.len(),
        config.path.rayset.len(),
        DURATION_S,
        DT_S,
    );
    println!(
        "          ndata {}, setup {setup_ms:.1} ms",
        simulator.ndata()
    );
    println!();

    // Scaling sweeps, which need their own station set -- more than five, and nested.
    if let Ok(list) = std::env::var("HB_STATION_LIST") {
        let stations = stations_from_file(Path::new(&list));
        scaling(&slip, &vmod, &config, &stations);
        return;
    }

    // A change that alters which deviates a subfault receives moves every waveform without
    // being wrong. `ENGINEERING_RULES` §4 wants an argument for why the new numbers are right,
    // and the argument this mode supplies is the one `PROFILE.md` used when the transform
    // length changed: the *level* is a statistic over seeds, and it should not move.
    if let Ok(seeds) = std::env::var("HB_SEEDS") {
        let count: u64 = seeds.parse().expect("HB_SEEDS is a seed count");
        seed_scan(&simulator, &wanted, count);
        return;
    }

    println!(
        "{:<10} {:>8}  {:>9} {:>8}  {:>13} {:>7}  {:>5} {:>8}",
        "station", "wall s", "pairs", "outside", "samples", "useful", "plans", "peaks lost"
    );

    for (index, &(name, latitude, longitude)) in STATIONS.iter().enumerate() {
        if let Some(wanted) = &wanted
            && !wanted.iter().any(|w| w == name)
        {
            continue;
        }
        let station = Station {
            name: name.to_owned(),
            latitude,
            longitude,
        };

        let start = Instant::now();
        let simulation = simulator.run(station, HF_SEED.wrapping_add(index as u64));
        let wall = start.elapsed().as_secs_f64();

        let census = simulation.census;
        println!(
            "{name:<10} {wall:>8.2}  {:>9} {:>8}  {:>13} {:>6.1}%  {:>5} {:>8}",
            census.pairs_attempted,
            census.pairs_outside_record,
            census.samples_computed,
            100.0 * census.useful_fraction(),
            census.transform_lengths,
            simulation.clipping.peaks_lost,
        );
    }
}
