# hf_port

Rust port of EMOD3D's `hb_high_v6.0.3` stochastic high-frequency seismogram
generator, `BINMOD` + `VERSION1` configuration.

The porting method is bit-identity against a pinned Fortran oracle. See
`reference/PROVENANCE.md` for what the oracle is and how it differs from
production, `harness/PHASE0C_RESULTS.md` for the validation that justifies those
differences, and `PORTING_RULES.md` for the transliteration rules.

```
harness/build_ref.sh              build the Fortran oracle + A/B binaries
harness/ab_fft.sh                 Phase 0c leg 1: FFT swap
python3 harness/ab_rng.py         Phase 0c leg 2: RNG swap
cargo test                        kernel goldens (bit-identical)
```

EMOD3D is not modified by this project.
