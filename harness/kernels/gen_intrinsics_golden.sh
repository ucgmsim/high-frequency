#!/usr/bin/env bash
# Sweep gfortran's intrinsics so the Rust side can be checked against them.
set -euo pipefail
cd "$(dirname "$0")/../.."
FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/intrinsics
mkdir -p "$G" harness/out
python3 reference/make_ref.py >/dev/null
gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/intrinsics_driver \
    harness/kernels/intrinsics_driver.f reference/pcg32.f -lm
printf '%s\n' "$G/sweep.bin" | ./harness/out/intrinsics_driver
printf '  sweep.bin %s bytes\n' "$(stat -c%s "$G/sweep.bin")"
