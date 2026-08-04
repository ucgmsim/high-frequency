#!/usr/bin/env bash
# Stage 2 per-commit gate: compare this tree's output against an EARLIER COMMIT's.
#
# `run_parity.sh` compared against the Fortran oracle and was the Stage 1 gate. Since
# §2.1 replaced the FFT it is red on every deck by design, so it can no longer answer
# "did this commit change anything unexpectedly". This does.
#
# It builds the binary from a git ref into a throwaway worktree and diffs the two over
# the same deck ladder. Two ways to use it:
#
#   harness/run_selfparity.sh HEAD~1              was this commit bit-exact?
#   harness/run_selfparity.sh pre-loop 1e-4       how far has the whole loop drifted?
#
# Bit-exact mode is the useful one per commit: a change meant to be exact that comes
# back red is a bug, and that is the signal the statistical gates are far too slow to
# give. Tolerance mode is for asking how much accumulated drift a run of commits has
# produced.
set -euo pipefail
cd "$(dirname "$0")/.."

REF=${1:-HEAD~1}
TOL=${2:-}
WT=${SELFPARITY_WORKTREE:-target/selfparity}

SHA=$(git rev-parse --short "$REF")
echo "reference: $REF ($SHA)"

# A detached worktree at the reference commit, reused across runs. Its own target/ dir
# would double every build, so it shares this one via CARGO_TARGET_DIR with a distinct
# subdirectory -- sharing target/ outright would make the two builds evict each other.
if [ -d "$WT" ]; then
    git -C "$WT" checkout -q --detach "$SHA" 2>/dev/null || {
        git worktree remove --force "$WT" 2>/dev/null || rm -rf "$WT"
        git worktree add -q --detach "$WT" "$SHA"
    }
else
    git worktree add -q --detach "$WT" "$SHA"
fi

echo "building reference binary..."
( cd "$WT" && CARGO_TARGET_DIR=../selfparity-target cargo build --release -q )
REF_BUILT="$(pwd)/target/selfparity-target/release/hb_high"
[ -x "$REF_BUILT" ] || { echo "reference build produced no binary" >&2; exit 1; }

cargo build --release -q

# Keep the two runs' scratch separate from run_parity.sh's so a concurrent Fortran
# comparison cannot clobber this one.
export REF_BIN="$REF_BUILT"
export PARITY_OUT=harness/out/selfparity
[ -n "$TOL" ] && export PARITY_MAX_REL="$TOL"
exec harness/run_parity.sh --release
