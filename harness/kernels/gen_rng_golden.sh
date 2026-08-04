#!/usr/bin/env bash
# Generate RNG kernel goldens from the Fortran reference.
#
# Output lands in harness/golden/rng/ and is COMMITTED, so `cargo test` works
# on a machine without gfortran. Regenerate only when reference/pcg32.f or
# normal_random_number changes -- and if a golden moves, that is a behaviour
# change that needs justifying, not a routine refresh.
set -euo pipefail
cd "$(dirname "$0")/../.."

FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/rng
mkdir -p "$G" harness/out

python3 reference/make_ref.py >/dev/null

gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/rng_driver \
    harness/kernels/rng_driver.f reference/hb_high_subs.f reference/pcg32.f -lm

run() { # mode seed count outfile
    printf '%s\n%s\n%s\n%s\n' "$1" "$2" "$3" "$4" | ./harness/out/rng_driver
}

# Seeds chosen to cover the sign cases that matter for the line-1366 branch:
# a large positive, a small negative that crosses zero after mutation, a large
# negative that does not, and zero.
for seed in 123456789 -3 -2000000000 0; do
    run 1 "$seed" 4096 "$G/next_u32_$seed.bin"
    run 2 "$seed" 4096 "$G/rand_numb_$seed.bin"
    run 4 "$seed" 1000 "$G/ranu2_$seed.bin"
done

# normal_random_number: odd and even counts exercise the discarded sine
# partner, and 1000/4096 are the sizes the program actually uses (nr=1000 for
# RANU2, np2 for stoc_f).
for n in 1 2 3 15 16 1000 4096; do
    run 3 123456789 "$n" "$G/normal_${n}.bin"
done

# The argument-mutation contract, as text.
: > "$G/seed_mutation.txt"
for seed in 123456789 -3 -2000000000 0 1; do
    run 0 "$seed" 0 unused >> "$G/seed_mutation.txt"
done

echo "goldens written to $G:"
ls -la "$G"
cat "$G/seed_mutation.txt"
