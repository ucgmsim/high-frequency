//! End-to-end benchmarks: the whole simulation in-process, per fault size.
//!
//! These time the simulation only, with no process startup, input parsing or output, so
//! they are not comparable with `harness/bench_baseline.csv`, which timed a whole
//! executable run.
//!
//! The faults are built in code: a benchmark wants fixed inputs of a known size, and
//! `subfault_count` is what runtime scales with.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::input::{Segment, Slip, Station, StochModel, Subfault, build_velocity_model};
use hb_high::state::{InputLayer, VelocityModelInput};

/// Grid shapes spanning three orders of magnitude in subfault count. The alpine-scale case is
/// seconds per iteration, so it stays opt-in via `HB_BENCH_SLOW=1`.
const FAULTS: &[(&str, usize, usize)] = &[("mini", 4, 1), ("medium", 14, 8), ("alpine", 257, 11)];

fn uniform_fault(along: usize, down: usize) -> StochModel {
    let segment = Segment::builder()
        .fault_lon_deg(173.0)
        .fault_lat_deg(-43.0)
        .along_strike_count(along)
        .down_dip_count(down)
        .subfault_length_km(1.5)
        .subfault_width_km(1.5)
        .strike_deg(220.0)
        .dip_deg(60.0)
        .rake_deg(160.0)
        .top_depth_km(1.0)
        .hypocentre_along_strike_km(0.0)
        .hypocentre_down_dip_km(1.5)
        .subfaults(vec![
            Subfault {
                slip: Slip(50.0),
                rise_time_s: 0.5,
                rupture_time_s: 0.0
            };
            along * down
        ])
        .build();
    StochModel::new(vec![segment])
}

fn crustal_model(layers: usize) -> VelocityModelInput {
    let built: Vec<InputLayer> = (0..layers)
        .map(|k| {
            let frac = k as f64 / (layers - 1) as f64;
            let vsh_km_s = 0.5 + 4.1 * frac;
            let qs = 50.0 + 150.0 * frac;
            InputLayer {
                depth_km: 0.0,
                thickness_km: if k == layers - 1 {
                    0.0
                } else {
                    (0.05 + 3.0 * frac) as f32
                },
                vp_km_s: vsh_km_s * 1.75,
                vsh_km_s,
                density_g_cm3: 1.81 + 1.5 * frac,
                attenuation_p: (2.0 * qs) as f32,
                attenuation_s: qs as f32,
            }
        })
        .collect();
    build_velocity_model(&built, 999.9).expect("valid velocity model")
}

fn production_config() -> HfConfig {
    HfConfig {
        source: SourceParameters {
            stress_drop_bars: 50.0,
            czero: 2.0,
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
            path_duration: PathDurationModel::Gp2010,
        },
        site: SiteParameters {
            kappa_s: 0.045,
            f_max_hz: 10.0,
        },
        record: RecordParameters {
            duration_s: 40.0,
            dt_s: 0.005,
        },
    }
}

fn bench_whole(c: &mut Criterion) {
    let vmod = crustal_model(20);
    let config = production_config();
    let slow = std::env::var("HB_BENCH_SLOW").is_ok_and(|v| v == "1");

    let mut group = c.benchmark_group("whole_program");
    // Even the smallest fault is milliseconds and the largest is seconds, so criterion's
    // default 100 samples would take minutes for no extra precision.
    group.sample_size(10);

    for &(name, along, down) in FAULTS {
        if name == "alpine" && !slow {
            eprintln!("skipping whole_program/alpine (set HB_BENCH_SLOW=1 to include)");
            continue;
        }
        let slip = uniform_fault(along, down);
        let station = Station {
            longitude: 173.3,
            latitude: -42.7,
            name: "BENCH".to_string(),
        };
        group.bench_with_input(
            BenchmarkId::new(name, slip.subfault_count),
            &slip,
            |b, slip| {
                b.iter(|| {
                    black_box(
                        hb_high::sim::Simulator::new(&config, slip, &vmod)
                            .expect("the fixture slip model is consistent")
                            .run(station.clone(), 12345),
                    )
                })
            },
        );
    }
    group.finish();
}

/// Stations per batch. Small enough that the setup still shows against it, large enough to
/// be a plausible chunk for one dask task.
const BATCH_STATIONS: usize = 16;

/// What `Simulator` was built for: setup once, N stations against it.
///
/// `bench_whole` simulates one station per iteration, so `Simulator::new` is inside every
/// sample and the construct-once saving is invisible.
///
/// The two arms are the same work in two orders. `shared` builds one simulator and runs the
/// batch against it; `per_station` rebuilds it for every station. The gap between them is
/// the setup -- the air layer, the slip-model normalisation, the moment scaling and the
/// per-segment angles -- divided across the batch instead of paid each time.
///
/// It should widen with subfault count, because the normalisation is the part that scales
/// with it and the per-station work is dominated by the geometry.
fn bench_batch(c: &mut Criterion) {
    let vmod = crustal_model(20);
    let config = production_config();
    let slow = std::env::var("HB_BENCH_SLOW").is_ok_and(|v| v == "1");

    let stations: Vec<Station> = (0..BATCH_STATIONS)
        .map(|i| Station {
            // Spread over about a degree, so the batch is not one geometry repeated.
            longitude: 173.3 + 0.05 * i as f32,
            latitude: -42.7 - 0.05 * i as f32,
            name: format!("BENCH{i}"),
        })
        .collect();

    let mut group = c.benchmark_group("batch");
    group.sample_size(10);

    for &(name, along, down) in FAULTS {
        if name == "alpine" && !slow {
            continue;
        }
        let slip = uniform_fault(along, down);

        group.bench_with_input(
            BenchmarkId::new(format!("{name}/shared"), slip.subfault_count),
            &slip,
            |b, slip| {
                b.iter(|| {
                    let simulator = hb_high::sim::Simulator::new(&config, slip, &vmod)
                        .expect("the fixture slip model is consistent");
                    for station in &stations {
                        black_box(simulator.run(station.clone(), 12345));
                    }
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("{name}/per_station"), slip.subfault_count),
            &slip,
            |b, slip| {
                b.iter(|| {
                    for station in &stations {
                        black_box(
                            hb_high::sim::Simulator::new(&config, slip, &vmod)
                                .expect("the fixture slip model is consistent")
                                .run(station.clone(), 12345),
                        );
                    }
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_whole, bench_batch);
criterion_main!(benches);
