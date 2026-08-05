#!/usr/bin/env bash
# Stage 3's LONG tier -- the scientific equivalence campaign.
#
# Run ONCE at the end of a stage, not per commit. `harness/run_cheap.sh` is the
# per-commit gate; this is what says whether the accumulated result is still the same
# science.
#
#   C     distributional IMs against PRODUCTION Fortran, unpaired. The acceptance gate.
#         Now also gates the distribution SHAPE -- the scatter ratio and the 5th/50th/95th
#         percentiles -- because a change in sampling moves shape while leaving the mean
#         alone, and the mean was all that was ever tested.
#   D     inter-frequency correlation of Fourier-amplitude residuals.
#   D A/A the same test against ONE binary split in half, so every flag is a known false
#         alarm. Mandatory: a verdict is not reportable without its own false-alarm floor.
#
# Tier B is gone. It paired against the oracle by sharing a PCG32 stream, which Stage 3's
# RNG replacement destroys -- and a desynced paired test does not fail, it goes
# Undetermined everywhere and PASSES. See stats.rs where equivalence_paired used to be.
#
# Usage: harness/run_long.sh [--quick]
#
#   --quick is the BISECTION tool, not a certification. It drops to n=100 and widens the
#   band to +/-10%, because n=100 resolves only about +/-4.7% and the resolution gate
#   would (correctly) refuse to decide a +/-2% band on it. Use it to find WHICH commit
#   moved something, then confirm with a full run.
set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=0
BAND=""
[ "${1:-}" = "--quick" ] && QUICK=1

# Sample sizes. These are sized to have a realistic chance of PASSING, not merely
# for the interval to fit the band -- see stats::sample_size_for. Sizing from
# "half-width < band" gives n=600 for a +/-2% band, which passes only about 7% of
# endpoints even when the two codes are statistically identical.
SEEDS_C=${SEEDS_C:-5100}
SEEDS_D=${SEEDS_D:-600}
if [ "$QUICK" = 1 ]; then
    SEEDS_C=100; SEEDS_D=120; BAND="--band 0.10"
    echo "QUICK MODE: reduced sample sizes, results are indicative only"
    echo
fi

cargo build --release --offline -q
[ -x reference/build/hb_ref ]  || harness/build_ref.sh >/dev/null
[ -x reference/build/hb_prod ] || { echo "building hb_prod..."; harness/bench_vs_fortran.sh >/dev/null; }

fail=0

echo "############ Tier C -- distributional, vs production Fortran ############"
./target/release/validate --tier c --cell a --seeds "$SEEDS_C" $BAND --baseline || fail=1

echo
echo "############ Tier D -- inter-frequency correlation ############"
# Tier D needs enough realisations for a stable correlation matrix: SE(rho) is
# about (1-rho^2)/sqrt(n-3), so n in the hundreds. It does not need Tier C's n,
# because the permutation test calibrates itself against the sample it is given.
#
# The verdict is Holm-corrected across the whole family. Gating on raw p would
# give 15 tests a family-wise false-alarm rate of 1 - 0.95^15 = 53.7%, i.e. it
# would fail more often than not on a correct port. See stats::holm_adjusted.
#
./target/release/validate --tier d --cell a --seeds "$SEEDS_D" --baseline || fail=1

echo
echo "############ Tier D A/A -- is the TEST calibrated? ############"
# MANDATORY since Stage 3, where it used to be a comment nobody ran.
#
# It splits ONE binary's realisations in half, so every flag is a known false alarm and
# the flag rate measures whether the permutation test over-rejects on a max-over-435-pairs
# statistic. Without it a Tier D failure cannot be attributed between "the codes differ"
# and "the test is noisy" -- and it is the only reason the production 4-of-15 is
# attributable to the generator at all.
#
# A verdict is not reportable without knowing its own false-alarm floor.
./target/release/validate --tier d --cell a --seeds $((SEEDS_D * 2)) --aa --baseline || fail=1

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
