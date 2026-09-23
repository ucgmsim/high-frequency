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
//!
//! Integers (the `argmax` fields) must match exactly. Floats may differ by a relative
//! [`FLOAT_TOLERANCE`], because the platform `libm` rounds transcendentals differently
//! from one system to another. A change to the draw structure moves them by far more.

use hb_high::geom::GeoPoint;
use hb_high::rng::FixtureDraws;
use hb_high::sim::Simulator;
use ndarray::Array2;

#[path = "common/fixtures.rs"]
mod fixtures;
use fixtures::{crustal_model, production_config, uniform_fault};

const GOLDEN: &str = "../../harness/golden/snapshot.txt";
const COMPONENT_COUNT: usize = 3;
const FLOAT_TOLERANCE: f64 = 1e-5;

/// Whether two snapshot lines agree: integers exactly, floats to [`FLOAT_TOLERANCE`].
fn lines_match(want: &str, got: &str) -> bool {
    let (want, got): (Vec<_>, Vec<_>) = (
        want.split_whitespace().collect(),
        got.split_whitespace().collect(),
    );
    want.len() == got.len()
        && want.iter().zip(&got).all(|(w, g)| {
            w == g
                || (w.contains('e')
                    && match (w.parse::<f64>(), g.parse::<f64>()) {
                        (Ok(w), Ok(g)) => (w - g).abs() <= FLOAT_TOLERANCE * w.abs().max(g.abs()),
                        _ => false,
                    })
        })
}

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

#[test]
fn the_whole_pipeline_matches_the_recorded_snapshot() {
    // Fixed inputs spanning two grid shapes, not a realistic earthquake.
    let vmod = crustal_model(20);

    let mut lines = Vec::new();
    for (label, along, down) in [("small", 4usize, 1usize), ("medium", 14, 8)] {
        let slip = uniform_fault(along, down);
        let origin = &slip.segments[0];

        for (seed, duration) in [(12345u64, 40.0f32), (987654321, 60.0)] {
            let station = GeoPoint {
                lon_deg: origin.fault_lon_deg + 0.3,
                lat_deg: origin.fault_lat_deg + 0.3,
            };
            // The frozen fixture draw source, not the production generator: a snapshot is
            // only a pin if the numbers are reproducible, and `FixtureDraws` exists to never
            // change.
            let sim = Simulator::new(&production_config(duration), &slip, &vmod)
                .expect("the fixture slip model is consistent")
                .run_with::<FixtureDraws>(station, seed);

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
    let mismatched = recorded.lines().count() != produced.lines().count()
        || recorded
            .lines()
            .zip(produced.lines())
            .any(|(w, g)| !lines_match(w, g));
    if mismatched {
        // Line-by-line, because "one of 4 cases moved" is the first thing to know and a
        // whole-string diff buries it.
        for (want, got) in recorded.lines().zip(produced.lines()) {
            if !lines_match(want, got) {
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
