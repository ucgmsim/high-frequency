#!/usr/bin/env bash
# Phase 2 gate: whole-program bit-identity against the Fortran oracle.
#
# Runs a ladder of decks through both binaries and `cmp`s the raw output. The
# ladder is not just "more cases": each tier switches on a part of the program
# that the previous one did not reach, and several tiers exercise branches the
# PRODUCTION deck never takes (see the comments per tier).
#
# Usage: harness/run_parity.sh [--release|--debug]
#
# Two env hooks, added in Stage 2:
#   REF_BIN          reference binary to compare against. Defaults to the Fortran
#                    oracle. harness/run_selfparity.sh points this at a Rust binary
#                    built from an earlier commit, which is the only cheap per-commit
#                    gate left now that bit-identity to the Fortran is gone.
#   PARITY_MAX_REL   if set, compare within this relative tolerance instead of
#                    bit-exactly (max |diff| / waveform peak).
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=${1:---release}
case "$PROFILE" in
  --release) cargo build --release -q; RUST=target/release/hb_high ;;
  --debug)   cargo build -q;           RUST=target/debug/hb_high ;;
  *) echo "usage: $0 [--release|--debug]" >&2; exit 2 ;;
esac

REF_BIN=${REF_BIN:-reference/build/hb_ref}
if [ "$REF_BIN" = "reference/build/hb_ref" ]; then
    [ -x reference/build/hb_ref ] || harness/build_ref.sh >/dev/null
fi
[ -x "$REF_BIN" ] || { echo "reference binary $REF_BIN is missing" >&2; exit 2; }

O=${PARITY_OUT:-harness/out/parity}
F=harness/fixtures
mkdir -p "$O"

pass=0; fail=0; failed_cases=()

run_case() {
    local name=$1; shift
    rm -f "$O/f.bin" "$O/r.bin"
    python3 harness/mkdeck.py --velmod "$F/velocity_model" \
        --station-file "$O/s.ll" --output-file "$O/f.bin" --write-station "$@" > "$O/df.txt"
    sed 's|f\.bin|r.bin|' "$O/df.txt" > "$O/dr.txt"

    local fe re
    fe=$("$REF_BIN" < "$O/df.txt" 2>&1 >/dev/null) || true
    re=$(./"$RUST"        < "$O/dr.txt" 2>&1 >/dev/null) || true

    if [ -n "${PARITY_MAX_REL:-}" ]; then
        # Tolerance mode: report the actual deviation, not just pass/fail, so a
        # per-commit run says HOW FAR the change moved the waveform.
        if ! python3 harness/compare.py "$O/f.bin" "$O/r.bin" \
                --max-rel "$PARITY_MAX_REL" --quiet; then
            fail=$((fail+1)); failed_cases+=("$name")
            printf '  FAIL  %s (beyond %s relative)\n' "$name" "$PARITY_MAX_REL"
            return
        fi
    elif ! cmp -s "$O/f.bin" "$O/r.bin"; then
        fail=$((fail+1)); failed_cases+=("$name")
        printf '  FAIL  %s\n' "$name"
        python3 harness/compare.py "$O/f.bin" "$O/r.bin" --quiet || true
        return
    fi
    # The stderr epicentral distance is part of the interface: hf_sim.py parses
    # it with float(stderr.strip()).
    if [ "$fe" != "$re" ]; then
        fail=$((fail+1)); failed_cases+=("$name (stderr)")
        printf '  FAIL  %s -- stderr differs: %q vs %q\n' "$name" "$fe" "$re"
        return
    fi
    pass=$((pass+1))
    printf '  ok    %s\n' "$name"
}

echo "tier 1 -- production defaults, minimal fault"
for seed in 123456789 -3 -2000000000 0 987654321; do
    run_case "tier1 mini seed=$seed" --stoch "$F/stoch/2012p578973.stoch" --seed "$seed"
done

echo "tier 2 -- one switch at a time; several are dead in production"
# ipdur_model selects a different path-duration table (production uses 11).
for ip in 0 1 2 11 12; do
    run_case "tier2 ipdur=$ip" --stoch "$F/stoch/2012p578973.stoch" --ipdur "$ip"
done
# isite_amp=0 skips get_sitefacs/siteamp entirely.
run_case "tier2 siteamp=0" --stoch "$F/stoch/2012p578973.stoch" --siteamp 0
# rayset with more than one entry runs the ray loop more than once; ray type 2
# is the down-going/Moho-reflected topology, which production never requests.
run_case "tier2 rayset=1,2" --stoch "$F/stoch/2012p578973.stoch" --rayset 1,2
run_case "tier2 rayset=1,3" --stoch "$F/stoch/2012p578973.stoch" --rayset 1,3
# rayset=0 selects the straight-ray approximation, bypassing gf_amp_tt's result.
run_case "tier2 rayset=0"   --stoch "$F/stoch/2012p578973.stoch" --rayset 0
# rupv > 0 takes the geometric rupture-time branch, the ONLY way to reach the
# irand jitter test at line 1366. Production passes rupv=-1, so that branch is
# unreachable there. Paired with seeds either side of the mutated-irand sign.
for seed in 123456789 -3 -2000000000; do
    run_case "tier2 rupv=2.5 seed=$seed" --stoch "$F/stoch/2012p578973.stoch" \
        --rupv 2.5 --seed "$seed"
done
# kappa <= 0 selects stoc_f's alternative high-cut branch.
run_case "tier2 kappa=-1" --stoch "$F/stoch/2012p578973.stoch" --kappa -1
# A vs_moho low enough to actually truncate the model at the Moho, but not so
# low that the model ends above the source. 4.2 truncates at layer 33 of 34.
#
# NOTE: vs_moho=3.5 truncates the model to about 5 km while the source sits at
# about 26 km. The reference then SEGFAULTS: the layer search falls through (it
# prints 'wrong!'), garbage velocities give a nonsense travel time, and the
# accumulation index goes far enough negative to write outside DS. That input is
# outside the program's domain, so there is nothing to be bit-identical to and
# it is deliberately not in the ladder. The port does not crash there -- it
# discards the out-of-range writes -- which is a robustness difference, not a
# parity one.
run_case "tier2 vs_moho=4.2" --stoch "$F/stoch/2012p578973.stoch" --vs-moho 4.2

echo "tier 3 -- larger faults and longer records"
run_case "tier3 2013p543824"   --stoch "$F/stoch/2013p543824.stoch"
run_case "tier3 mini dt=0.01"  --stoch "$F/stoch/2012p578973.stoch" --duration 40 --dt 0.01
run_case "tier3 mini long"     --stoch "$F/stoch/2012p578973.stoch" --duration 100
# The alpine fault is 2827 subfaults at np2=65536 -- minutes per binary, so it
# is opt-in rather than part of the default gate. Run it with SLOW=1.
if [ "${SLOW:-0}" = "1" ]; then
    run_case "tier3 alpine"    --stoch "$F/stoch/alpine_base_r1.stoch"
else
    echo "  skip  tier3 alpine (set SLOW=1 to include; takes minutes)"
fi

echo
if [ "$fail" -eq 0 ]; then
    if [ -n "${PARITY_MAX_REL:-}" ]; then
        echo "PASS: $pass/$pass decks within $PARITY_MAX_REL relative ($PROFILE, ref $REF_BIN)"
    else
        echo "PASS: $pass/$pass decks bit-identical ($PROFILE, ref $REF_BIN)"
    fi
else
    echo "FAIL: $fail of $((pass+fail)) decks differ ($PROFILE)"
    for c in "${failed_cases[@]}"; do echo "  - $c"; done
    exit 1
fi
