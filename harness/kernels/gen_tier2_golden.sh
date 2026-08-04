#!/usr/bin/env bash
# Generate tier-2 kernel goldens from the Fortran reference. Committed.
set -euo pipefail
cd "$(dirname "$0")/../.."

FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/tier2
mkdir -p "$G" harness/out

python3 reference/make_ref.py >/dev/null

gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/tier2_driver \
    harness/kernels/tier2_driver.f reference/hb_high_subs.f reference/pcg32.f -lm

names=(cagcon dtdp highcor_f radfrq_lin radv_lin)
for i in 1 2 3 4 5; do
    n=${names[$((i-1))]}
    printf '%s\n%s\n' "$i" "$G/$n.bin" | ./harness/out/tier2_driver >/dev/null
    printf '  %-12s %s bytes\n' "$n" "$(stat -c%s "$G/$n.bin")"
done
