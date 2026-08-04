//! Shared scaffolding for the golden tests.
//!
//! Six test files carried a near-identical `Reader` struct, five copies of `eq32`, four
//! of `eq64` and two of `near32` — around 200 duplicated lines whose only real variation
//! was the `harness/golden/<tier>` subdirectory and the name of the regeneration script
//! in the panic message. Both are parameters now.
//!
//! The duplication was not merely untidy. `io_golden`'s copy had drifted: it sliced
//! `buf[pos..pos + 4]` with no bounds check, so a short or truncated fixture panicked
//! with a slice-index message instead of the "ran off the end at byte N of M" the other
//! five produce, and its `eq64` dropped the hex rendering that makes an ulp-level
//! mismatch readable. Converging on one implementation fixes both for free.
//!
//! # Why the readers are small composable pieces, not one fixture builder
//!
//! The obvious factoring of the four velocity-model loaders is a single
//! `vmod(count, fields)` taking a set of which fields to read. That would be a flag
//! argument — the exact smell §2.8 is removing elsewhere in this crate — and it would be
//! worse than the duplication, because the field *order* in each fixture is dictated by
//! its Fortran driver's `write` statement and is not a property a caller should be
//! choosing at all. So the shared pieces are `f64s`/`f32s`, and each tier spells out its
//! own driver's dump order in terms of them. [`Golden::ray_seam_state`] is the one
//! exception: tiers 2 and 3 read the *same* record layout, so it is one layout, not two.

// Each test binary links this module separately and uses a different subset, so anything
// not used by *every* binary would otherwise warn in the others.
#![allow(dead_code)]

use std::path::PathBuf;

use hb_high::state::{RayState, VelocityModel};

/// Sequential reader over a Fortran `access='stream'` file.
pub struct Golden {
    buf: Vec<u8>,
    pos: usize,
    name: String,
}

impl Golden {
    /// Open `harness/golden/<tier>/<name>`.
    ///
    /// `tier` is both the subdirectory and the middle of the regeneration script's name,
    /// so a missing fixture says exactly which script to run.
    pub fn open(tier: &str, name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../harness/golden")
            .join(tier)
            .join(name);
        let buf = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "reading {}: {e}. Run harness/kernels/gen_{tier}_golden.sh",
                path.display()
            )
        });
        Self { buf, pos: 0, name: format!("{tier}/{name}") }
    }

    fn take<const N: usize>(&mut self) -> [u8; N] {
        assert!(
            self.pos + N <= self.buf.len(),
            "{}: ran off the end at byte {} of {}",
            self.name,
            self.pos,
            self.buf.len()
        );
        let out = self.buf[self.pos..self.pos + N].try_into().unwrap();
        self.pos += N;
        out
    }

    pub fn f32(&mut self) -> f32 {
        f32::from_le_bytes(self.take::<4>())
    }

    pub fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.take::<8>())
    }

    pub fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.take::<4>())
    }

    pub fn usize(&mut self) -> usize {
        self.i32() as usize
    }

    /// `n` bytes as text — the drivers write `character*n` fields unpadded.
    pub fn chars(&mut self, n: usize) -> String {
        assert!(
            self.pos + n <= self.buf.len(),
            "{}: ran off the end at byte {} of {}",
            self.name,
            self.pos,
            self.buf.len()
        );
        let s = String::from_utf8_lossy(&self.buf[self.pos..self.pos + n]).to_string();
        self.pos += n;
        s
    }

    pub fn f32s(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.f32()).collect()
    }

    pub fn f64s(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.f64()).collect()
    }

    pub fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Assert the whole file was consumed — catches a record-layout misunderstanding
    /// that would otherwise pass silently on a prefix.
    pub fn assert_exhausted(&self) {
        assert_eq!(
            self.pos,
            self.buf.len(),
            "{}: consumed {} of {} bytes; record layout disagrees with the driver",
            self.name,
            self.pos,
            self.buf.len()
        );
    }

    /// The `dump_state` record shared by the tier-2 and tier-3 drivers: `thickness_km`,
    /// `vp_km_s` and `vsh_km_s` as `f64`, then `alp` and `als` as `f32`, for `layers`
    /// layers.
    ///
    /// `layers` is the Fortran's deepest layer *number*, which doubles as a count from 1.
    /// `Travel::ndeep` is a 0-based index since §2.3, so it is one lower — that `- 1` is
    /// the whole reason this is shared rather than written out twice.
    pub fn ray_seam_state(&mut self, layers: usize) -> (RayState, VelocityModel) {
        let mut vmod = VelocityModel::new();
        for k in 0..layers {
            vmod[k].thickness_km = self.f64();
        }
        for k in 0..layers {
            vmod[k].vp_km_s = self.f64();
        }
        for k in 0..layers {
            vmod[k].vsh_km_s = self.f64();
        }
        let mut st = RayState::default();
        for k in 0..layers {
            st.travel.alp[k] = self.f32();
        }
        for k in 0..layers {
            st.travel.als[k] = self.f32();
        }
        st.travel.ndeep = layers as i32 - 1;
        (st, vmod)
    }
}

/// Bit-for-bit `f32` comparison. The hex is not decoration: at this level the decimal
/// rendering of two values an ulp apart is usually identical.
#[track_caller]
pub fn eq32(what: &str, got: f32, want: f32) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: rust {got:?} (0x{:08x}) vs fortran {want:?} (0x{:08x})",
        got.to_bits(),
        want.to_bits()
    );
}

/// Bit-for-bit `f64` comparison.
#[track_caller]
pub fn eq64(what: &str, got: f64, want: f64) {
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "{what}: rust {got:?} (0x{:016x}) vs fortran {want:?} (0x{:016x})",
        got.to_bits(),
        want.to_bits()
    );
}

/// Relative `f32` comparison against a per-record scale.
///
/// §2.1 replaced the vendored radix-2 kernel with `rustfft`, which sums the butterflies
/// in a different order, so anything downstream of a transform can no longer be exact.
/// The physics either side of it is unchanged and still worth checking, hence a loosened
/// comparison rather than a deleted one.
///
/// The scale is the record's peak rather than the individual value, so a bin that is
/// legitimately near zero is not held to an impossible relative tolerance.
#[track_caller]
pub fn near32(what: &str, got: f32, want: f32, scale: f32) {
    let tol = 1e-4 * scale.max(f32::MIN_POSITIVE);
    assert!(
        (got - want).abs() <= tol,
        "{what}: rust {got:?} vs fortran {want:?} (delta {:.3e}, tolerance {tol:.3e})",
        (got - want).abs()
    );
}

/// Relative `f64` comparison at a caller-supplied tolerance.
///
/// The tolerance is a parameter, not a constant, because each tier's divergence has a
/// different cause and a different measured size. Whoever passes it owns the argument for
/// it — see `tier2_golden`'s, which records both the mechanism and the measured worst.
#[track_caller]
pub fn near64(what: &str, got: f64, want: f64, relative_tolerance: f64) {
    let tol = relative_tolerance * want.abs().max(f64::MIN_POSITIVE);
    assert!(
        (got - want).abs() <= tol,
        "{what}: rust {got:?} vs fortran {want:?} (delta {:.3e}, tolerance {tol:.3e})",
        (got - want).abs()
    );
}

/// Tracks the worst relative divergence across a fixture, so the number justifying a
/// [`near64`] tolerance stays measured rather than remembered.
#[derive(Default)]
pub struct Divergence {
    worst: f64,
    at: String,
}

impl Divergence {
    pub fn note(&mut self, what: &str, got: f64, want: f64) {
        let rel = (got - want).abs() / want.abs().max(f64::MIN_POSITIVE);
        if rel > self.worst {
            self.worst = rel;
            self.at = what.to_string();
        }
    }

    pub fn report(&self, fixture: &str) {
        println!("{fixture}: worst relative divergence {:.3e} at {}", self.worst, self.at);
    }
}
