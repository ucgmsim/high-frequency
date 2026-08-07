//! High-frequency stochastic seismogram synthesis.
//!
//! This crate is the **high-frequency (f > 1 Hz) module of the Graves & Pitarka hybrid
//! broadband method**. Give it a finite-fault rupture, a 1-D velocity model and a station, and
//! it returns three components of ground acceleration.
//!
//! # Start here
//!
//! **`PHYSICS.md`** explains what is computed and why, for a reader with a general geophysics
//! background and no familiarity with this code. It has a symbol glossary, because the
//! identifiers are inherited six-character names and the literature uses Greek.
//!
//! **`papers/README.md`** records which paper substantiates which equation, and how far each
//! claim has been checked. Two papers carry almost all of it:
//!
//! * **Boore (1983)**, *BSSA* 73(6A), 1865–1894 — the point-source stochastic method,
//!   equations 1–11. That is [`stoc`].
//! * **Graves & Pitarka (2010)**, *BSSA* 100(5A), 2095–2123 — the finite-fault wrapper,
//!   equations 10–17. That is [`sim`].
//!
//! # The method in one paragraph
//!
//! High-frequency ground motion looks like filtered noise. So instead of solving a wave
//! equation, specify the Fourier *amplitude* spectrum from seismology — source, path and site
//! as separate multiplicative factors — pair it with a *random phase* spectrum, and inverse
//! transform. One seed gives one plausible accelerogram; another seed gives another. This is
//! why the crate is full of random number generation, and why the *number* of random draws is
//! treated as part of the answer rather than an implementation detail.
//!
//! # Module map
//!
//! | module | what it does |
//! | --- | --- |
//! | [`sim`] | walks the rupture and sums subfault contributions — the entry point |
//! | [`stoc`] | one subfault's spectrum, and its inverse transform |
//! | [`radiation`] | double-couple pattern and its conical average |
//! | [`site`] | quarter-wavelength site amplification |
//! | [`ray`] | ray tracing, travel times and path attenuation |
//! | [`geom`] | subfault geometry on a WGS84 geodesic |
//! | [`config`] | typed configuration, with defaults resolved in one place |
//! | [`input`] | the source and receiver data model |
//! | [`state`] | the velocity model and ray-tracing state |
//! | [`fft`] | the transform, and the baseline correction that shares its callers |
//! | [`rng`] | the generators, and the draw-count contract |
//!
//! # Three things that cannot move
//!
//! * **Component order** — 090, 000, vertical, interleaved component-fastest. Callers consume
//!   the channels positionally, and the two horizontals draw from the shared random stream
//!   while the vertical does not.
//! * **Same-version determinism** — one seed, one build, one answer. Free to change *across*
//!   versions, since results are regenerable by pinning a commit, but never within one.
//! * **[`rng::Draws::uniform`]'s 24-bit conversion**, which keeps deviates inside `[0, 1)`.
//!   Box-Muller's zero-rejection loops in [`rng::LegacyPcg`] depend on it.
//!
//! `PHYSICS.md` §9 lists the rest of what looks like a bug and is not.

pub mod config;
pub mod fft;
pub mod geom;
pub mod input;
pub mod radiation;
pub mod ray;
pub mod rng;
pub mod sim;
pub mod site;
pub mod state;
pub mod stoc;
