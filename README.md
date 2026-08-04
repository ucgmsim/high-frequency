# hf_port

Rust port of EMOD3D's `hb_high_v6.0.3` stochastic high-frequency seismogram
generator, `BINMOD` + `VERSION1` configuration.

**Status: whole-program bit-identical to the Fortran oracle** across the deck
ladder in `harness/run_parity.sh`, in both debug and release.

The porting method is bit-identity against a pinned Fortran reference. See
`reference/PROVENANCE.md` for what the oracle is and how it differs from
production, `harness/PHASE0C_RESULTS.md` for the validation justifying those
differences, and `PORTING_RULES.md` for the transliteration rules.

```
harness/build_ref.sh              build the Fortran oracle + A/B binaries
harness/ab_fft.sh                 Phase 0c leg 1: FFT swap
python3 harness/ab_rng.py         Phase 0c leg 2: RNG swap
cargo test                        kernel + reader goldens (bit-identical)
harness/run_parity.sh [--debug]   whole-program parity ladder
SLOW=1 harness/run_parity.sh      ... including the 2827-subfault alpine fault
harness/cov.sh                    coverage of the reference under that ladder
```

Both `cargo test` and `run_parity.sh` must pass in **both** profiles: a
debug/release disagreement means the port depends on optimisation-level float
behaviour, which has already caught one real bug (see `PORTING_RULES.md` §4b).

EMOD3D is not modified by this project.
