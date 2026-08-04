#!/usr/bin/env bash
# Phase 0g: coverage of the Fortran reference under the deck ladder.
#
# Whole-file gcov figures are misleading because hb_high_ref.f still contains the
# 27 subprograms that are dead in this configuration, so this splits live from
# dead. The dead set reading 0% is an independent check on the dead-code
# analysis, which was otherwise derived only by reading call sites.
#
# Results and interpretation: harness/COVERAGE.md
set -euo pipefail
cd "$(dirname "$0")/.."
python3 reference/make_ref.py >/dev/null

C=harness/out/cov
rm -rf "$C"; mkdir -p "$C"
gfortran -cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w \
    -fprofile-arcs -ftest-coverage -DBINMOD -DVERSION1 -Ireference \
    -o "$C/hb_cov" reference/hb_high_ref.f reference/pcg32.f -lm

F=$PWD/harness/fixtures
MK=$PWD/harness/mkdeck.py
cd "$C"
run() {
    python3 "$MK" --velmod "$F/velocity_model" --station-file s.ll \
        --output-file o.bin --write-station "$@" > d.txt
    ./hb_cov < d.txt >/dev/null 2>&1 || true
}
for seed in 123456789 -3 -2000000000 0 987654321; do
    run --stoch "$F/stoch/2012p578973.stoch" --seed "$seed"
done
for ip in 0 1 2 11 12; do run --stoch "$F/stoch/2012p578973.stoch" --ipdur "$ip"; done
run --stoch "$F/stoch/2012p578973.stoch" --siteamp 0
run --stoch "$F/stoch/2012p578973.stoch" --rayset 1,2
run --stoch "$F/stoch/2012p578973.stoch" --rayset 1,3
run --stoch "$F/stoch/2012p578973.stoch" --rayset 0
for seed in 123456789 -3 -2000000000; do
    run --stoch "$F/stoch/2012p578973.stoch" --rupv 2.5 --seed "$seed"
done
run --stoch "$F/stoch/2012p578973.stoch" --kappa -1
run --stoch "$F/stoch/2012p578973.stoch" --vs-moho 4.2
run --stoch "$F/stoch/2013p543824.stoch"
run --stoch "$F/stoch/2012p578973.stoch" --duration 40 --dt 0.01
run --stoch "$F/stoch/2012p578973.stoch" --duration 100

# gcov resolves the source path relative to the COMPILE directory, so it has to
# run from the repo root or it emits a .gcov with no source text.
cd "$OLDPWD"
gcov -b -o "$C" "$C/hb_cov-hb_high_ref" >/dev/null 2>&1
mv -f hb_high_ref.f.gcov params_no_window.h.gcov "$C/" 2>/dev/null || true
python3 harness/cov_report.py "$C/hb_high_ref.f.gcov"
