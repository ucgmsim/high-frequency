# Phase 0c validation results

Three one-time checks establishing that `reference/build/hb_ref` is a legitimate
bit-exact oracle for the Rust port. Run on gfortran 16.1.1 / FFTW 3.3.11,
x86-64, with the flags in `harness/build_ref.sh`.

Fixture: `harness/fixtures/stoch/2012p578973.stoch` (single segment, `nx=2`,
`nw=2`), `harness/fixtures/velocity_model` (34 layers), one station at +0.1°
longitude from the fault reference point, `duration=20`, `dt=0.005`
(`ndata=4000`, output 48000 bytes).

All three pass. **Status: cleared to proceed to Phase 1.**

---

## Leg 1 — FFT swap (HARD GATE)

`harness/ab_fft.sh`. Unpatched source built twice, differing only in
`USE_FFTW`. Same generator, same seed, so the FFT is the only variable.

The concern: the two `FAST` implementations use **opposite Fourier sign
conventions**. `fftw3.f` defines `FFTW_FORWARD=-1`, and the wrapper maps
`IND==1` to the forward plan; the radix-2 kernel's `IND=-1` is the analysis
transform. Callers use `-1` for analysis (`stoc_f:1760`) and `+1` for synthesis
(`highcor_f:2258`), matching radix-2 and inverted relative to FFTW. So the FFTW
build computes the conjugate transform at both ends.

**Result: the flip cancels, as predicted.**

| metric | value |
| --- | --- |
| bit-identical | no |
| max abs diff | 1.573563e-05 |
| waveform peak | 1.936250e+01 |
| max rel to peak | 8.127e-07 |
| ulp max / mean / median | 172032 / 51.42 / 4 |
| samples differing at all | 61.1% |
| correlation (090/000/ver) | 0.9999999999996 / 0.9999999999996 / 0.9999999999996 |

The mechanism: every operation between the forward and inverse transform is
either multiplication by a real factor (`AS`, `AMP`, `rdna`, `an`) or an
explicit conjugate-symmetric mirror, so applying `conj` at both ends leaves the
real part of the output unchanged up to rounding. What remains is
single-precision rounding accumulation.

The large max-ulp figure is not a counter-indication — ulp distance explodes
near zero crossings. The meaningful numbers are the 8.1e-07 relative difference
and the correlation. A genuine convention error would have destroyed the
correlation, not left it at twelve nines.

**Side observation, and a useful one:** re-running this leg gives slightly
different numbers (max rel 8.127e-07 on the first run, 8.373e-07 on the second;
max ulp 172032 vs 196608). The radix-2 side is deterministic, so the movement is
entirely on the FFTW side — `FFTW_MEASURE` selected a different plan on the
second run. That is finding 1's non-reproducibility happening in front of us,
and it is why the oracle cannot be an FFTW build.

**Conclusion: dropping FFTW costs nothing observable.** The port targets the
radix-2 transform.

## Leg 2 — RNG swap

`harness/ab_rng.py --seeds 24`. Unpatched radix-2 build (gfortran intrinsic
generator) vs patched build (PCG32). Same FFT, so the generator is the only
variable. These cannot agree sample-by-sample — a different generator is a
different noise realisation — so this compares distributions over 24 seeds.

Peak acceleration, geometric-mean ratio (orig/PCG32) with 95% CI:

| component | ratio | 95% CI |
| --- | --- | --- |
| 090 | 0.9840 | [0.8728, 1.1093] |
| 000 | 1.0561 | [0.9772, 1.1415] |
| ver | 0.9654 | [0.8838, 1.0545] |

Fourier amplitude, geometric-mean ratio per band:

| band (Hz) | 090 | 000 | ver |
| --- | --- | --- | --- |
| 0.10–0.25 | 1.0478 | 1.0776 | 0.9040 |
| 0.25–0.50 | 1.0398 | 0.8525 | 0.9327 |
| 0.50–1.00 | 0.9154 | 0.8731 | 0.8382 |
| 1.00–2.00 | 0.9652 | 1.0272 | 1.0185 |
| 2.00–4.00 | 1.0278 | 0.9856 | 0.9874 |
| 4.00–8.00 | 0.9278 | 0.9843 | 0.9688 |
| 8.00–16.00 | 1.0153 | 1.0119 | 0.9901 |

All three PGA confidence intervals straddle 1. FAS ratios sit near 1 across all
seven bands with **no systematic trend** across frequency — which is the
diagnostic that matters. A wrong variance or a broken unit-RMS renormalisation
would show as a flat offset; consuming the wrong number of deviates in the
spectral loop would show as a tilt. Neither is present.

Two of the 21 band CIs marginally exclude 1 (`ver` 0.50–1.00 Hz at
[0.724, 0.970]; `090` 4–8 Hz at [0.861, 1.000]). At 95% confidence across 21
comparisons roughly one exclusion is expected by chance, so this is consistent
with sampling noise. The low-frequency bands have wide CIs because they contain
few FFT bins at `np2` for a 20 s record.

## Leg 3 — Determinism of the oracle

Two runs of `hb_ref` on an identical deck: **bit-identical**, 0 differing
samples, correlation exactly 1.

This is the check that retires the `FFTW_MEASURE` concern from finding 1. With
FFTW the plan is chosen by timing it, so the production binary is not reliably
reproducible against itself; the patched reference is.

---

## Reproducing

```bash
harness/build_ref.sh
harness/ab_fft.sh                    # leg 1
python3 harness/ab_rng.py --seeds 24 # leg 2
# leg 3:
python3 harness/mkdeck.py --stoch harness/fixtures/stoch/2012p578973.stoch \
  --velmod harness/fixtures/velocity_model \
  --station-file /tmp/s.ll --output-file /tmp/r1.bin --write-station > /tmp/d1.txt
sed 's|r1.bin|r2.bin|' /tmp/d1.txt > /tmp/d2.txt
reference/build/hb_ref < /tmp/d1.txt && reference/build/hb_ref < /tmp/d2.txt
python3 harness/compare.py /tmp/r1.bin /tmp/r2.bin
```
