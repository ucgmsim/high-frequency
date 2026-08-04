#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/tier4
mkdir -p "$G" harness/out
python3 reference/make_ref.py >/dev/null
gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/tier4_driver \
    harness/kernels/tier4_driver.f reference/hb_high_subs.f reference/pcg32.f -lm
names=(stoc_f gf_amp_tt)
for i in 1 2; do
    n=${names[$((i-1))]}
    printf '%s\n%s\n' "$i" "$G/$n.bin" | ./harness/out/tier4_driver >/dev/null
    printf '  %-10s %s bytes\n' "$n" "$(stat -c%s "$G/$n.bin")"
done
