//! End-to-end snapshot of the whole pipeline, against summary statistics recorded in
//! `harness/golden/snapshot.txt`.
//!
//! Inputs are fixed and built in code, and draws come from the frozen fixture generator, so
//! the result is reproducible. It runs in milliseconds with no subprocess.
//!
//! # Why summary statistics rather than the samples
//!
//! A full record per component per fixture is a megabyte of golden data no reviewer can
//! read. Seven numbers per component fit on a line, and a diff says *how* the waveform
//! moved.
//!
//! The sampled positions are offsets from the peak, so every one lands inside the signal:
//! a pin in the silence before the S arrival or after the coda would record zero and never
//! fail. `argmax` is the most sensitive field -- any change to timing, ray topology or
//! window placement moves it -- while peak and RMS are invariant under a permutation of the
//! record.
//!
//! # This goes red by design
//!
//! Any change to the draw structure (count, order, or generator) moves these numbers.
//! Re-record with `UPDATE_SNAPSHOT=1` once the change has been validated, and say why in
//! the commit message.

use hb_high::config::{
    HfConfig, PathDurationModel, PathParameters, RayType, RecordParameters, RuptureVelocity,
    SiteParameters, SourceParameters,
};
use hb_high::input::{Segment, Slip, Station, StochModel, Subfault, build_velocity_model};
use hb_high::sim::Simulator;
use hb_high::state::VelocityModelInput;
use ndarray::Array2;

const GOLDEN: &str = "../../harness/golden/snapshot.txt";
const COMPONENT_COUNT: usize = 3;

/// Seven numbers per component that summarise one record without storing it.
fn summarise(acc: &Array2<f32>, ndata: usize) -> String {
    let mut fields = Vec::new();
    for component in 0..COMPONENT_COUNT {
        let samples: Vec<f32> = acc.row(component).to_vec();
        assert_eq!(samples.len(), ndata);

        let (argmax, peak) =
            samples
                .iter()
                .enumerate()
                .fold((0usize, 0.0f32), |(at, best), (i, v)| {
                    if v.abs() > best {
                        (i, v.abs())
                    } else {
                        (at, best)
                    }
                });
        // f64 accumulation, so the statistics do not add f32 rounding of their own.
        let energy = samples.iter().map(|v| *v as f64 * *v as f64).sum::<f64>();
        let mean = samples.iter().map(|v| *v as f64).sum::<f64>() / ndata as f64;

        fields.push(format!("{peak:.6e}"));
        fields.push(argmax.to_string());
        fields.push(format!("{:.6e}", (energy / ndata as f64).sqrt()));
        fields.push(format!("{mean:.6e}"));
        // Offsets from the peak, so every one lands inside the signal. Coprime-ish strides
        // rather than 1/2/3 so they do not all sit inside one half-cycle of the waveform.
        for offset in [1usize, 7, 53] {
            fields.push(format!("{:.6e}", samples[(argmax + offset) % ndata]));
        }
    }
    fields.join(" ")
}

/// A uniform-slip single-segment fault, `along` by `down` subfaults.
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

/// A smoothly graded crustal model with a thin near-surface layer, so `insert_air_layer`
/// fires as it does on every production model.
fn crustal_model(layers: usize) -> VelocityModelInput {
    let built: Vec<hb_high::state::InputLayer> = (0..layers)
        .map(|k| {
            let frac = k as f64 / (layers - 1) as f64;
            let vsh_km_s = 0.5 + 4.1 * frac;
            let qs = 50.0 + 150.0 * frac;
            hb_high::state::InputLayer {
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

fn production_config(duration: f32) -> HfConfig {
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
            duration_s: duration,
            dt_s: 0.005,
        },
    }
}

#[test]
fn the_whole_pipeline_matches_the_recorded_snapshot() {
    // The frozen fixture draw source, not the production generator: a snapshot is only a
    // pin if the numbers are reproducible, and `FixtureDraws` exists to never change.
    // SAFETY: set before any simulation runs, and this test is the only reader.
    unsafe { std::env::set_var("HB_FIXTURE_RNG", "1") };

    // Fixed inputs spanning two grid shapes, not a realistic earthquake.
    let vmod = crustal_model(20);

    let mut lines = Vec::new();
    for (label, along, down) in [("small", 4usize, 1usize), ("medium", 14, 8)] {
        let slip = uniform_fault(along, down);
        let origin = &slip.segments[0];

        for (seed, duration) in [(12345u64, 40.0f32), (987654321, 60.0)] {
            let station = Station {
                longitude: origin.fault_lon_deg + 0.3,
                latitude: origin.fault_lat_deg + 0.3,
                name: "TEST".to_string(),
            };
            let sim = Simulator::new(&production_config(duration), &slip, &vmod)
                .expect("the fixture slip model is consistent")
                .run(station, seed);

            // A snapshot of an all-zero record is a gate that cannot fail.
            let peak = sim.acc.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(
                peak > 0.0,
                "{label} seed {seed} at {duration} s produced silence -- snapshotting \
                 zeros would pin nothing"
            );
            lines.push(format!(
                "{label} {seed} {duration} {}",
                summarise(&sim.acc, sim.ndata)
            ));
        }
    }
    let produced = lines.join("\n") + "\n";

    if std::env::var_os("UPDATE_SNAPSHOT").is_some() {
        std::fs::write(GOLDEN, &produced).expect("writing the snapshot");
        eprintln!("snapshot re-recorded -- explain why in the commit message");
        return;
    }

    let recorded = std::fs::read_to_string(GOLDEN)
        .unwrap_or_else(|e| panic!("cannot read {GOLDEN}: {e}\nrecord it with UPDATE_SNAPSHOT=1"));
    if recorded != produced {
        // Line-by-line, because "one of 4 cases moved" is the first thing to know and a
        // whole-string diff buries it.
        for (want, got) in recorded.lines().zip(produced.lines()) {
            if want != got {
                eprintln!("  recorded: {want}\n  produced: {got}");
            }
        }
        panic!(
            "the pipeline no longer reproduces harness/golden/snapshot.txt.\n\
             If this commit deliberately changed the draw structure -- count, order, or \
             generator -- that is expected: check the output is still correct, then re-record \
             with UPDATE_SNAPSHOT=1 and say so in the commit message."
        );
    }
}
