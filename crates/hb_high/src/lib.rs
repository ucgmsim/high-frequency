//! Rust port of EMOD3D `hb_high_v6.0.3` — stochastic high-frequency seismogram
//! generator, `BINMOD` + `VERSION1` configuration only.
//!
//! # What this crate is, as of Stage 3
//!
//! It began as a *transliteration* of `hb_high_ref.f`, gated on byte-identical output. **That contract has expired**, deliberately and in stages: §2.1 replaced the
//! FFT, §2.2b the gamma function, §2.5 turned `DELAZ5` into a WGS84 geodesic, §2.6 fixed
//! two reproduced defects, §2.8 corrected the truncated pi literals. The Fortran-shaped
//! code went with it — storage is 0-based, the 2-D array wrappers are gone, magic
//! integers are enums.
//!
//! What the port is gated on now is **scientific equivalence, not identity**: the
//! distributions of intensity measures must match production Fortran within ±2%, which
//! is roughly 0.04 of a typical ground-motion-model aleatory sigma. See `REFACTOR.md`
//! for the tier structure and `ENGINEERING_RULES.md` for what may and may not change.
//!
//! `PORTING_RULES.md` is retained as **archaeology**. It explains why the Fortran does
//! what it does, which is still needed when reading the oracle — but most of its rules
//! are marked expired, and it is no longer a description of this crate.
//!
//! Two things genuinely cannot move, and neither is fidelity:
//!
//! * the **component order** of `Simulation::acc` — 090, 000, vertical, interleaved
//!   component-fastest. §4.3 deleted the raw-`f32`-at-a-seek-offset output format and the
//!   `(1x,f10.4)` distance on stderr along with the CLI driver, because a Python caller
//!   gets an array and a `PyErr`; what survives is the order the three channels come in;
//! * **same-version determinism** — one seed, one build, one answer;
//! * `next_f32`'s **24-bit** conversion, which keeps deviates inside `[0, 1)`.

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
pub mod special;
