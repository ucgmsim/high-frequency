//! Scientific equivalence campaign for the `hb_high` port.
//!
//! Tier A (bit-identity) lives in `harness/run_parity.sh`. This crate implements
//! the tiers above it:
//!
//! * **B** — paired IM equivalence at matched seeds, for changes that alter
//!   arithmetic but leave the RNG stream intact. Matched seeds give matched
//!   realisations, so the comparison has almost no variance and is correspondingly
//!   sensitive.
//! * **C** — distributional IM equivalence against production Fortran, which uses
//!   a different RNG and so can only be compared unpaired.
//! * **D** — inter-frequency correlation of Fourier amplitude residuals.
//!
//! See `stats` for why this is framed as equivalence testing rather than as a
//! null-hypothesis test.

pub mod stats;
