#!/usr/bin/env bash
# Generate tier-1 kernel goldens from the Fortran reference. Committed.
set -euo pipefail
cd "$(dirname "$0")/../.."

FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/tier1
mkdir -p "$G" harness/out

python3 reference/make_ref.py >/dev/null

gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/tier1_driver \
    harness/kernels/tier1_driver.f reference/hb_high_subs.f reference/pcg32.f -lm

names=(get_sitefacs trav geom_terms even_dist2)
for i in 1 2 3 4; do
    n=${names[$((i-1))]}
    printf '%s\n%s\n' "$i" "$G/$n.bin" | ./harness/out/tier1_driver >/dev/null
    printf '  %-12s %s bytes\n' "$n" "$(stat -c%s "$G/$n.bin")"
done
