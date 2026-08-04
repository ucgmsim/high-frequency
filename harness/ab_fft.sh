#!/usr/bin/env bash
# Phase 0c, leg 1 — isolate the FFT swap.
#
# Builds the UNPATCHED vendored Fortran twice, differing only in USE_FFTW, and
# compares the waveforms. Both binaries use gfortran's intrinsic RNG with the
# same seed, so the random realisation is identical and the FFT is the only
# variable.
#
# This is a HARD GATE on the whole porting plan. The two FAST implementations
# use opposite Fourier sign conventions (fftw3.f sets FFTW_FORWARD=-1 and the
# wrapper maps IND==1 to the forward plan, whereas the radix-2 kernel's IND=-1
# is the analysis transform). The flip is expected to cancel, because every
# spectral operation between the forward and inverse transform is either
# multiplication by a real factor or an explicit conjugate-symmetric mirror.
# If it does not cancel, the port would be bit-identical to something
# production does not compute, and the plan must be revisited.
#
# Pass criterion: max |diff| / waveform peak < 1e-5 and per-component
# correlation > 0.999999. A real convention error destroys the correlation.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=harness/out/ab_fft
mkdir -p "$OUT"

harness/build_ref.sh >/dev/null

STOCH=${1:-harness/fixtures/stoch/2012p578973.stoch}

python3 harness/mkdeck.py --stoch "$STOCH" --velmod harness/fixtures/velocity_model \
    --station-file "$OUT/station.ll" --output-file "$OUT/radix2.bin" \
    --seed 123456789 --duration 20 --dt 0.005 --write-station > "$OUT/deck_radix2.txt"
sed 's|radix2\.bin|fftw.bin|' "$OUT/deck_radix2.txt" > "$OUT/deck_fftw.txt"

./reference/build/hb_orig_radix2 < "$OUT/deck_radix2.txt"
./reference/build/hb_orig_fftw   < "$OUT/deck_fftw.txt"

python3 harness/compare.py "$OUT/radix2.bin" "$OUT/fftw.bin" \
    --max-rel 1e-5 --min-corr 0.999999
