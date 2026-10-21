//! High-frequency stochastic seismogram synthesis.
//!
//! This crate is the **high-frequency (f > 1 Hz) module of the Graves & Pitarka hybrid
//! broadband method**. Give it a finite-fault rupture, a 1-D velocity model and a station, and
//! it returns three components of ground acceleration.
//!
//! # Start here
//!
//! **`PHYSICS.md`** explains what is computed and why, for a reader with a general geophysics
//! background and no familiarity with this code. It has a symbol glossary mapping the
//! identifiers to the literature's notation.
//!
//! Two papers carry almost all of the method:
//!
//! * **Boore (1983)**, *BSSA* 73(6A), 1865–1894 — the point-source stochastic method,
//!   equations 1–11. That is [`spectrum`].
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
//! | [`spectrum`] | one subfault's spectrum, and its inverse transform |
//! | [`source`] | slip to moment weights, the moment scaling and the rupture-velocity taper |
//! | [`radiation`] | double-couple pattern and its conical average |
//! | [`site`] | quarter-wavelength site amplification |
//! | [`ray`] | ray tracing, travel times and path attenuation |
//! | [`path_duration`] | how the shaping window lengthens with distance |
//! | [`geom`] | subfault geometry on a WGS84 geodesic |
//! | [`record`] | the output record, where a contribution lands in it, and what did not fit |
//! | [`config`] | the caller's configuration, as plain data |
//! | [`slip_model`] | the fault segments and their subfault grids |
//! | [`velocity`] | the 1-D velocity model: Moho truncation, the air layer, layer lookup |
//! | [`fft`] | the transform, and the baseline correction that shares its callers |
//! | [`rng`] | the generators, the draw-count contract and sub-stream seeding |
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
pub mod path_duration;
pub mod radiation;
pub mod ray;
pub mod record;
pub mod rng;
pub mod sim;
pub mod site;
pub mod slip_model;
pub mod source;
pub mod spectrum;
pub mod velocity;
