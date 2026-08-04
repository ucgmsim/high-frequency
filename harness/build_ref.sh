#!/usr/bin/env bash
# Build the three Fortran binaries the harness needs.
#
#   hb_orig_radix2  unpatched vendored source, radix-2 FFT, gfortran RNG
#   hb_orig_fftw    unpatched vendored source, FFTW FFT, gfortran RNG
#   hb_ref          THE ORACLE: PCG32 + radix-2, the target for bit-identity
#
# Flags are part of the golden contract, not a preference:
#   -O0 -ffp-contract=off   no reassociation, no FMA fusion. Production builds
#                           -O2 with contraction enabled, which reorders float
#                           arithmetic; goldens are defined against these flags.
#   -fwrapv                 pcg32.f relies on wrapping signed 64-bit multiply.
#   -DVERSION1              HF_TIME_WINDOW=off, as production builds.
set -euo pipefail
cd "$(dirname "$0")/.."

FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
# fftw3.f lives alongside the C header; only the hb_orig_fftw leg needs it.
# pkg-config reports no -I when FFTW is in a default *C* path, but gfortran's
# INCLUDE directive does not search those, so /usr/include is always appended.
FFTW_INC=($(pkg-config --cflags-only-I fftw3f 2>/dev/null) -I/usr/include)
mkdir -p reference/build

python3 reference/make_ref.py

gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 \
    -o reference/build/hb_orig_radix2 reference/hb_high_orig.f -lm
gfortran "${FFLAGS[@]}" "${FFTW_INC[@]}" -DBINMOD -DVERSION1 -DUSE_FFTW \
    -o reference/build/hb_orig_fftw reference/hb_high_orig.f -lfftw3f -lm
gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 \
    -o reference/build/hb_ref reference/hb_high_ref.f reference/pcg32.f -lm

echo "built:"
ls -1 reference/build/
