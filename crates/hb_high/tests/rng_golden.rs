//! Bit-identity gate for the RNG kernels.
//!
//! Goldens in `harness/golden/rng/` are produced by `harness/kernels/rng_driver.f`
//! linked against `reference/pcg32.f` and `reference/hb_high_subs.f`, so they
//! come from the same Fortran the oracle binary runs. Regenerate with
//! `harness/kernels/gen_rng_golden.sh`.
//!
//! Every assertion here is exact. There is no tolerance: a single differing bit
//! in the generator desynchronises the entire draw stream, and because the
//! program's output is noise, the resulting waveform would still look
//! plausible. See `PORTING_RULES.md` §10.

use hb_high::rng::{fill_normal_deviates, fill_uniform_deviates, Pcg32, SEED_WORDS};
use std::path::PathBuf;

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../harness/golden/rng")
        .canonicalize()
        .expect("harness/golden/rng missing; run harness/kernels/gen_rng_golden.sh")
}

fn read_u64s(name: &str) -> Vec<u64> {
    let raw = std::fs::read(golden_dir().join(name))
        .unwrap_or_else(|e| panic!("reading golden {name}: {e}"));
    assert_eq!(raw.len() % 8, 0, "{name}: not a whole number of integer*8");
    raw.chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_f32s(name: &str) -> Vec<f32> {
    let raw = std::fs::read(golden_dir().join(name))
        .unwrap_or_else(|e| panic!("reading golden {name}: {e}"));
    assert_eq!(raw.len() % 4, 0, "{name}: not a whole number of real*4");
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Compare bitwise, and on failure report where the streams first parted and
/// what the values were — the only diagnostic that makes an RNG mismatch
/// tractable.
fn assert_f32_bit_identical(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{name}: first divergence at index {i}: rust {g:?} (0x{:08x}) vs \
             fortran {w:?} (0x{:08x})",
            g.to_bits(),
            w.to_bits()
        );
    }
}

const SEEDS: [i32; 4] = [123456789, -3, -2000000000, 0];

#[test]
fn next_u32_matches_fortran() {
    for seed in SEEDS {
        let want = read_u64s(&format!("next_u32_{seed}.bin"));
        let (mut g, _) = Pcg32::seed(seed);
        for (i, &w) in want.iter().enumerate() {
            assert!(w < 1u64 << 32, "golden value {w} exceeds 32 bits");
            let got = g.next_u32();
            assert_eq!(
                got as u64, w,
                "seed {seed}: generator diverged at draw {i}: \
                 rust {got} vs fortran {w}"
            );
        }
        assert_eq!(want.len(), 4096);
    }
}

#[test]
fn rand_numb_matches_fortran() {
    for seed in SEEDS {
        let want = read_f32s(&format!("rand_numb_{seed}.bin"));
        let (mut g, _) = Pcg32::seed(seed);
        let got: Vec<f32> = (0..want.len()).map(|_| g.next_f32()).collect();
        assert_f32_bit_identical(&format!("next_f32 seed {seed}"), &got, &want);
    }
}

#[test]
fn ranu2_matches_fortran() {
    for seed in SEEDS {
        let want = read_f32s(&format!("ranu2_{seed}.bin"));
        let (mut g, _) = Pcg32::seed(seed);
        let mut rn = vec![0.0; want.len()];
        fill_uniform_deviates(&mut g, want.len(), rn.as_mut_slice());
        assert_f32_bit_identical(
            &format!("uniform_deviates seed {seed}"),
            rn.as_slice(),
            &want,
        );
    }
}

#[test]
fn normal_random_number_matches_fortran() {
    // Odd counts exercise the discarded sine partner of the final Box-Muller
    // pair; 1000 and 4096 are sizes the program actually uses.
    for n in [1usize, 2, 3, 15, 16, 1000, 4096] {
        let want = read_f32s(&format!("normal_{n}.bin"));
        assert_eq!(want.len(), n);
        let (mut g, _) = Pcg32::seed(123456789);
        let mut acc = vec![0.0; n];
        fill_normal_deviates(&mut g, n, acc.as_mut_slice());
        assert_f32_bit_identical(&format!("normal nr={n}"), acc.as_slice(), &want);
    }
}

#[test]
fn seed_mutation_matches_fortran() {
    // init_random_seed increments its argument once per seed word, and
    // hb_high reads the mutated value at line 1366 to gate rupture-time
    // jitter. The golden is the Fortran's own before/after pairs.
    let text = std::fs::read_to_string(golden_dir().join("seed_mutation.txt"))
        .expect("seed_mutation.txt missing");
    let mut checked = 0;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut it = line.split_whitespace();
        let before: i32 = it.next().unwrap().parse().unwrap();
        let after: i32 = it.next().unwrap().parse().unwrap();
        assert_eq!(
            after - before, SEED_WORDS,
            "fortran mutated {before} by {}, but SEED_WORDS is {SEED_WORDS}",
            after - before
        );
        let (_, got) = Pcg32::seed(before);
        assert_eq!(got, after, "seed mutation mismatch for {before}");
        checked += 1;
    }
    assert!(checked >= 5, "expected at least 5 golden seed pairs, got {checked}");

    // The case that actually changes program behaviour: a small negative seed
    // becomes positive after mutation, flipping the line-1366 branch.
    let (_, after) = Pcg32::seed(-3);
    assert!(after > 0, "seed -3 must mutate to a positive value");
}
