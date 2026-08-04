#!/usr/bin/env bash
# Wall-clock comparison: Rust release vs PRODUCTION Fortran.
#
# Criterion cannot meaningfully drive an external process, so this is a plain
# best-of-N timer.
#
# The reference here is deliberately the PRODUCTION Fortran build:
#   -O2, USE_FFTW=ON, VERSION1, gfortran's intrinsic RNG
#
# NOT reference/build/hb_ref, which is the bit-identity oracle built at -O0 with
# the radix-2 FFT. Timing against that would flatter the port by an order of
# magnitude and tell us nothing about whether the port is fast enough to deploy.
#
# The two binaries therefore produce different realisations (different RNG) and
# slightly different numerics (FFTW vs radix-2). That is fine: this measures time,
# not agreement. Agreement is harness/run_parity.sh and the Tier C campaign.
set -euo pipefail
cd "$(dirname "$0")/.."

REPEATS=${REPEATS:-5}
O=harness/out/bench_vs
F=harness/fixtures
mkdir -p "$O" reference/build

FFLAGS=(-cpp -ffixed-line-length-none -O2 -fno-fast-math -w)
FFTW_INC=($(pkg-config --cflags-only-I fftw3f 2>/dev/null) -I/usr/include)

if [ ! -x reference/build/hb_prod ] || \
   [ reference/hb_high_orig.f -nt reference/build/hb_prod ]; then
    echo "building production-equivalent Fortran (-O2, USE_FFTW=ON)..."
    gfortran "${FFLAGS[@]}" "${FFTW_INC[@]}" -DBINMOD -DVERSION1 -DUSE_FFTW \
        -o reference/build/hb_prod reference/hb_high_orig.f -lfftw3f -lm
fi
# Same -O2 Fortran with the radix-2 FFT: identical in every other respect, so the
# difference against hb_prod is FFTW's planning cost plus the FFT implementation.
# This is the APPLES-TO-APPLES comparison for the port, since the Rust build also
# uses radix-2 -- without it the FFTW planning overhead reads as a Rust speedup.
if [ ! -x reference/build/hb_prod_radix2 ] || \
   [ reference/hb_high_orig.f -nt reference/build/hb_prod_radix2 ]; then
    echo "building -O2 Fortran with the radix-2 FFT..."
    gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 \
        -o reference/build/hb_prod_radix2 reference/hb_high_orig.f -lm
fi

cargo build --release --offline -q
RUST=target/release/hb_high

# Best-of-N: we want the achievable time, not the mean of a noisy machine.
#
# Timed with date +%s%N rather than /usr/bin/time: `time` writes to stderr, and so
# does the program under test (the epicentral distance), so separating them means
# either a temp file or losing one of the two. Reading the clock directly avoids
# the problem entirely.
best_of() {
    local exe=$1 deck=$2 best=""
    for _ in $(seq "$REPEATS"); do
        local t0 t1 t
        t0=$(date +%s%N)
        "$exe" < "$deck" >/dev/null 2>/dev/null
        t1=$(date +%s%N)
        t=$(awk "BEGIN{printf \"%.4f\", ($t1 - $t0)/1e9}")
        if [ -z "$best" ] || awk "BEGIN{exit !($t < $best)}"; then best=$t; fi
    done
    echo "$best"
}

printf '%-10s %6s %10s %10s %10s %9s %9s\n' \
    fault subfl fftw_s radix2_s rust_s "vs fftw" "vs r2"
printf '%-10s %6s %10s %10s %10s %9s %9s\n' \
    ---------- ------ ---------- ---------- ---------- --------- ---------

run_fault() {
    local name=$1 stoch=$2 subfaults=$3
    python3 harness/mkdeck.py --stoch "$F/stoch/$stoch.stoch" \
        --velmod "$F/velocity_model" --station-file "$O/s.ll" \
        --output-file "$O/f.bin" --write-station > "$O/df.txt"
    sed 's|f\.bin|r.bin|' "$O/df.txt" > "$O/dr.txt"

    local ft r2 rt s1 s2
    ft=$(best_of reference/build/hb_prod "$O/df.txt")
    r2=$(best_of reference/build/hb_prod_radix2 "$O/df.txt")
    rt=$(best_of "$RUST" "$O/dr.txt")
    s1=$(awk "BEGIN{printf \"%.2f\", $ft/$rt}")
    s2=$(awk "BEGIN{printf \"%.2f\", $r2/$rt}")
    printf '%-10s %6s %10s %10s %10s %8sx %8sx\n' \
        "$name" "$subfaults" "$ft" "$r2" "$rt" "$s1" "$s2"
}

run_fault mini   2012p578973 4
run_fault medium 2013p543824 112
if [ "${SLOW:-0}" = "1" ]; then
    run_fault alpine alpine_base_r1 2827
else
    echo "(skipping alpine; set SLOW=1 -- minutes per run)"
fi

echo
echo "Best of $REPEATS per binary. Startup is well under a millisecond and common to"
echo "all three, so it does not bias the ratios."
echo
echo "'vs r2' is the honest port speedup: same algorithm, both -O2/release."
echo "'vs fftw' is larger only because FFTW_MEASURE planning costs 0.25-0.50 s per"
echo "process and never earns it back at these transform counts -- see PROFILE.md."
