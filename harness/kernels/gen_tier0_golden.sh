#!/usr/bin/env bash
# Generate tier-0 kernel goldens from the Fortran reference.
# Committed, so `cargo test` works without gfortran.
set -euo pipefail
cd "$(dirname "$0")/../.."

FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/tier0
mkdir -p "$G" harness/out

python3 reference/make_ref.py >/dev/null

gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/tier0_driver \
    harness/kernels/tier0_driver.f reference/hb_high_subs.f reference/pcg32.f -lm

names=(rdatn delaz5 dgamm cr flzero fast siteamp)
for i in 1 2 3 4 5 6 7; do
    n=${names[$((i-1))]}
    printf '%s\n%s\n' "$i" "$G/$n.bin" | ./harness/out/tier0_driver >/dev/null
    printf '  %-8s %s bytes\n' "$n" "$(stat -c%s "$G/$n.bin")"
done
