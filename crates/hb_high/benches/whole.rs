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

use hb_high::geom::GeoPoint;

#[path = "../tests/common/fixtures.rs"]
mod fixtures;
use fixtures::{crustal_model, production_config, uniform_fault};

/// Grid shapes spanning three orders of magnitude in subfault count. The alpine-scale case is
/// seconds per iteration, so it stays opt-in via `HB_BENCH_SLOW=1`.
const FAULTS: &[(&str, usize, usize)] = &[("mini", 4, 1), ("medium", 14, 8), ("alpine", 257, 11)];

fn bench_whole(c: &mut Criterion) {
    let vmod = crustal_model(20);
    let config = production_config(40.0);
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
        let station = GeoPoint {
            lon_deg: 173.3,
            lat_deg: -42.7,
        };
        group.bench_with_input(
            BenchmarkId::new(name, slip.subfault_count),
            &slip,
            |b, slip| {
                b.iter(|| {
                    black_box(
                        hb_high::sim::Simulator::new(&config, slip, &vmod)
                            .expect("the fixture slip model is consistent")
                            .run(station, 12345),
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
    let config = production_config(40.0);
    let slow = std::env::var("HB_BENCH_SLOW").is_ok_and(|v| v == "1");

    let stations: Vec<GeoPoint> = (0..BATCH_STATIONS)
        .map(|i| GeoPoint {
            // Spread over about a degree, so the batch is not one geometry repeated.
            lon_deg: 173.3 + 0.05 * i as f32,
            lat_deg: -42.7 - 0.05 * i as f32,
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
                        black_box(simulator.run(*station, 12345));
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
                                .run(*station, 12345),
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
