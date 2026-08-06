//! End-to-end benchmarks: one station, whole pipeline, per fault size.
//!
//! # What changed in §4.3, and why it is an improvement
//!
//! These used to drive the built binary through `std::process::Command`, feeding it a deck on
//! stdin — because `run()` in `main.rs` read stdin and wrote files, so there was nothing else
//! to call. The note here used to say that benching in-process "would require restructuring
//! it, which is Phase 3 work".
//!
//! That restructuring is done: `main.rs` and the deck reader are gone and `simulate` takes
//! typed values. So these now call the library directly, which removes process startup, deck
//! generation via `python3 harness/mkdeck.py`, text parsing and a file write from the
//! measurement. What is left is the simulation, which is the thing worth timing — so these
//! numbers are NOT comparable with `harness/bench_baseline.csv`, which timed all of it.
//!
//! The faults are built in code rather than read from fixtures, for the same reason the
//! snapshot test builds its own: a benchmark wants *fixed* inputs of a known size, and
//! `subfault_count` is what runtime scales with.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::input::{build_velocity_model, Segment, Station, StochModel, Subfault};
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
                slip: 50.0,
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
            apply_quarter_wavelength_site_amplification: true,
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
                        hb_high::sim::simulate(&config, slip, &vmod, station.clone(), 12345)
                            .expect("simulation succeeds"),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_whole);
criterion_main!(benches);
