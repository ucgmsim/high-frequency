//! Shared scaffolding for the golden tests.
//!
//! Field order in each fixture is fixed by the Fortran driver that wrote it, so each test
//! spells out its own record layout in terms of the primitive readers (`f32s`, `f64s`,
//! ...).

// Each test binary links this module separately and uses a different subset, so anything
// not used by *every* binary would otherwise warn in the others.
#![allow(dead_code)]

use std::path::PathBuf;

/// Sequential reader over a Fortran `access='stream'` file.
pub struct Golden {
    buf: Vec<u8>,
    pos: usize,
    name: String,
}

impl Golden {
    /// Open `harness/golden/<tier>/<name>`.
    pub fn open(tier: &str, name: &str) -> Self {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../harness/golden")
            .join(tier)
            .join(name);
        let buf =
            std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()));
        Self {
            buf,
            pos: 0,
            name: format!("{tier}/{name}"),
        }
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
/// For anything downstream of an FFT: `rustfft` sums the butterflies in a different order
/// from the Fortran's transform, so these values cannot be bit-exact.
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
