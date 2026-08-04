//! Rust port of EMOD3D `hb_high_v6.0.3` — stochastic high-frequency seismogram
//! generator, `BINMOD` + `VERSION1` configuration only.
//!
//! # Porting contract
//!
//! This crate is a *transliteration*, not a rewrite. Until the whole-program
//! parity gate is green, every module here is expected to look like Fortran:
//! 1-based indexing, column-major 2-D arrays, `goto`s rendered as labelled
//! loops, and literal constants copied character-for-character from the source.
//! See `PORTING_RULES.md` at the repo root before changing anything.
//!
//! The single hard rule: **output must be byte-identical to
//! `reference/hb_high_ref.f`** on every deck in `harness/decks/`.

pub mod fort;
