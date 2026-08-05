# hf_port

Rust port of EMOD3D's `hb_high_v6.0.3` stochastic high-frequency seismogram
generator, `BINMOD` + `VERSION1` configuration.

**Status: scientifically equivalent to production Fortran**, certified at ±2% on
the intensity measures — 324 of 375 endpoints certified, **0 refuted**, the
remainder undetermined for want of sample size rather than agreement.

It is **not** bit-identical, and has not been since §2.1. That was the Stage 1
contract and it was retired on purpose, one change at a time, each with its
measured effect recorded: the FFT (§2.1), the gamma function (§2.2b), a WGS84
geodesic in place of `DELAZ5` (§2.5), two reproduced defects (§2.6), and the
Fortran's truncated pi literals (§2.8). `REFACTOR.md` is the log.

See `reference/PROVENANCE.md` for what the oracle is and how it differs from
production, `ENGINEERING_RULES.md` for what may and may not change, and
`PORTING_RULES.md` — retained as archaeology — for why the Fortran does what it
does. The latter describes the *original*, not this crate.

```
cargo test                        kernel + reader goldens, property tests
harness/run_selfparity.sh REF     compare against an earlier commit
harness/run_cheap.sh              per-commit gate (~25 s)
harness/run_long.sh               the statistical equivalence campaign
harness/build_ref.sh              build the Fortran oracle + A/B binaries
harness/cov.sh                    coverage of the reference under the deck ladder
```

`harness/run_parity.sh` still exists and is **red by design** — it is byte
comparison against the oracle. It is kept because it is the only remaining
mechanical link to the original, and because its deck ladder is what
`run_selfparity.sh` reuses.

Debug and release must still agree with **each other**. Neither is bit-identical
to the Fortran any more, but a disagreement between the two profiles means the
port depends on optimisation-level float behaviour, which has already caught one
real bug.

EMOD3D is not modified by this project.
