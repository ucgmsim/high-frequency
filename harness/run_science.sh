#!/usr/bin/env bash
# Phase 2: the scientific equivalence campaign.
#
# Tier A (bit-identity) is harness/run_parity.sh and remains the default gate for
# any change that should not alter behaviour. This script runs the tiers above it:
#
#   B  paired IMs at matched seeds against the ORACLE (shared PCG32 stream, so the
#      realisations match and the test is extremely sensitive)
#   C  distributional IMs against PRODUCTION Fortran (different RNG, so unpaired)
#   D  inter-frequency correlation of Fourier amplitude residuals
#
# All three must pass on a bit-identical build. A failure there means the
# measurement is wrong, not the port.
#
# Usage: harness/run_science.sh [--quick]
#   --quick uses a reduced sample size for a plumbing check. It is NOT a
#   certification: see the note on sample size below.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1

# Sample sizes. These are sized to have a realistic chance of PASSING, not merely
# for the interval to fit the band -- see stats::sample_size_for. Sizing from
# "half-width < band" gives n=600 for a +/-2% band, which passes only about 7% of
# endpoints even when the two codes are statistically identical.
SEEDS_C=${SEEDS_C:-2500}
SEEDS_D=${SEEDS_D:-600}
SEEDS_B=${SEEDS_B:-50}
if [ "$QUICK" = 1 ]; then
    SEEDS_C=100; SEEDS_D=120; SEEDS_B=20
    echo "QUICK MODE: reduced sample sizes, results are indicative only"
    echo
fi

cargo build --release --offline -q
[ -x reference/build/hb_ref ]  || harness/build_ref.sh >/dev/null
[ -x reference/build/hb_prod ] || { echo "building hb_prod..."; harness/bench_vs_fortran.sh >/dev/null; }

fail=0

echo "############ Tier B -- paired, matched seeds, vs the oracle ############"
./target/release/validate --tier b --cell a --seeds "$SEEDS_B" || fail=1

echo
echo "############ Tier C -- distributional, vs production Fortran ############"
./target/release/validate --tier c --cell a --seeds "$SEEDS_C" || fail=1

echo
echo "############ Tier D -- inter-frequency correlation ############"
# Tier D needs enough realisations for a stable correlation matrix: SE(rho) is
# about (1-rho^2)/sqrt(n-3), so n in the hundreds. It does not need Tier C's n,
# because the permutation test calibrates itself against the sample it is given.
./target/release/validate --tier d --cell a --seeds "$SEEDS_D" || fail=1

if [ "${CELL_B:-0}" = "1" ]; then
    echo
    echo "############ Cell B -- 112-subfault fault ############"
    ./target/release/validate --tier c --cell b --seeds "${SEEDS_CB:-800}" || fail=1
fi

echo
if [ "$fail" = 0 ]; then
    echo "PASS: all tiers equivalent"
else
    echo "FAIL: see the NOT EQUIVALENT lines above and harness/science_*.csv"
    exit 1
fi
