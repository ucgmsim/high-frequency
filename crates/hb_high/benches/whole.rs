//! End-to-end benchmarks: the whole program, per fault.
//!
//! These drive the built binary through `std::process::Command` rather than
//! calling into the library. `run()` in `main.rs` reads stdin and writes files, so
//! benching it in-process would require restructuring it — which is Phase 3 work,
//! not a prerequisite for a baseline. Process startup is well under a millisecond
//! against runs of tens of milliseconds upward, so the measurement is honest.
//!
//! `CARGO_BIN_EXE_hb_high` is set by Cargo for benches, so the binary is located
//! without guessing at target paths or profiles.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Faults spanning three orders of magnitude in subfault count. The alpine case
/// is minutes per run, so it is opt-in via `HB_BENCH_SLOW=1` rather than part of
/// the default set.
const FAULTS: &[(&str, &str, usize)] = &[
    ("mini", "2012p578973", 4),
    ("medium", "2013p543824", 112),
    ("alpine", "alpine_base_r1", 2827),
];

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Build a deck via the harness generator, so benches and the parity gate cannot
/// drift apart in what they consider a production deck.
fn deck(fault: &str, out_dir: &Path) -> String {
    let root = repo_root();
    let station = out_dir.join("station.ll");
    let output = out_dir.join("bench_out.bin");
    let o = Command::new("python3")
        .arg(root.join("harness/mkdeck.py"))
        .arg("--stoch")
        .arg(root.join(format!("harness/fixtures/stoch/{fault}.stoch")))
        .arg("--velmod")
        .arg(root.join("harness/fixtures/velocity_model"))
        .arg("--station-file")
        .arg(&station)
        .arg("--output-file")
        .arg(&output)
        .arg("--write-station")
        .output()
        .expect("running harness/mkdeck.py");
    assert!(
        o.status.success(),
        "mkdeck.py failed: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    String::from_utf8(o.stdout).expect("deck is utf8")
}

fn run_once(exe: &str, deck: &str) {
    let mut child = Command::new(exe)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawning hb_high");
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(deck.as_bytes())
        .expect("writing deck");
    let status = child.wait().expect("waiting for hb_high");
    assert!(status.success(), "hb_high exited with {status}");
}

fn bench_whole(c: &mut Criterion) {
    let exe = env!("CARGO_BIN_EXE_hb_high");
    let out_dir = repo_root().join("harness/out/bench");
    std::fs::create_dir_all(&out_dir).expect("creating harness/out/bench");
    let slow = std::env::var("HB_BENCH_SLOW").is_ok_and(|v| v == "1");

    let mut group = c.benchmark_group("whole_program");
    // Even the smallest fault is ~50 ms, so criterion's default 100 samples would
    // take minutes for no extra precision.
    group.sample_size(10);

    for &(name, fault, subfaults) in FAULTS {
        if name == "alpine" && !slow {
            eprintln!("skipping whole_program/alpine (set HB_BENCH_SLOW=1 to include)");
            continue;
        }
        let d = deck(fault, &out_dir);
        group.bench_with_input(
            BenchmarkId::new(name, subfaults),
            &d,
            |b, d| b.iter(|| run_once(exe, d)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_whole);
criterion_main!(benches);
