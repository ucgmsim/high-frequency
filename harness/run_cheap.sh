#!/usr/bin/env bash
# Stage 3's per-commit gate. Target: under 60 seconds.
#
# Stage 2's gate was `run_selfparity.sh` in bit-exact mode. Stage 3 retires it for two
# independent reasons: the RNG is being replaced, and accumulation-order fidelity is no
# longer binding, so reassociation is legal. Nothing is bit-exact any more.
#
# What replaces it is REPLAY-PARITY. Both binaries are driven by `HB_FIXTURE_RNG`, a draw
# source that is frozen forever and is not the production generator. Because the draws
# provably do not move, any difference in output is attributable to the code — which is
# exactly the question a per-commit gate should answer, and exactly the question a
# production-RNG comparison stops being able to answer the moment the engine changes.
#
# It goes RED BY DESIGN when a commit changes the draw *structure* (how many draws, or in
# what order). That is not a false alarm: it is the gate saying "this one needs the long
# tier", which is the correct answer for exactly those commits.
#
# Usage:
#   harness/run_cheap.sh              compare against harness/CHEAP_BASELINE
#   harness/run_cheap.sh <git-ref>    compare against something else
set -euo pipefail
cd "$(dirname "$0")/.."

TOL=${CHEAP_TOL:-1e-6}
BASELINE_FILE=harness/CHEAP_BASELINE
REF=${1:-}
if [ -z "$REF" ]; then
    [ -f "$BASELINE_FILE" ] || {
        echo "no $BASELINE_FILE and no ref given" >&2
        echo "  freeze one with:  git rev-parse HEAD > $BASELINE_FILE" >&2
        exit 2
    }
    REF=$(grep -v '^#' "$BASELINE_FILE" | tr -d '[:space:]' | head -1)
fi

start=$(date +%s)
fail=0

echo "############ 1. unit, golden and property tests ############"
cargo test --workspace --quiet 2>&1 | grep -E "^(test result|error|failures)" || true
cargo test --workspace --quiet >/dev/null 2>&1 || fail=1

echo
echo "############ 2. clippy ############"
if cargo clippy -p hb_high --all-targets --message-format short 2>&1 | grep -q "^crates"; then
    echo "FAIL: clippy is not at zero"
    cargo clippy -p hb_high --all-targets --message-format short 2>&1 | grep "^crates" | head
    fail=1
else
    echo "ok    zero warnings"
fi

echo
echo "############ 3. replay-parity vs $REF ############"
# The env var is exported so BOTH binaries inherit it -- run_parity.sh spawns the
# reference and the working tree, and a gate where only one side used the fixture source
# would compare two different programs.
#
# The reference must be a commit that HAS the fixture source. Commits before it cannot be
# replay-compared, which is why the baseline is frozen at or after §3.0c.
export HB_FIXTURE_RNG=1
if harness/run_selfparity.sh "$REF" "$TOL" 2>&1 | tail -3; then
    :
else
    fail=1
fi

echo
elapsed=$(( $(date +%s) - start ))
if [ "$fail" = 0 ]; then
    echo "CHEAP PASS  (${elapsed}s)"
else
    echo "CHEAP FAIL  (${elapsed}s)"
    echo
    echo "If leg 3 failed and this commit deliberately changed the draw structure --"
    echo "count, order, or the generator itself -- that is expected. Re-freeze the"
    echo "baseline and let the long tier adjudicate:"
    echo "    git rev-parse HEAD > $BASELINE_FILE"
    exit 1
fi
