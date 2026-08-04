#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
FFLAGS=(-cpp -ffixed-line-length-none -O0 -fno-fast-math -ffp-contract=off -fwrapv -w)
G=harness/golden/io
mkdir -p "$G" harness/out
python3 reference/make_ref.py >/dev/null
gfortran "${FFLAGS[@]}" -DBINMOD -DVERSION1 -Ireference \
    -o harness/out/io_driver harness/kernels/io_driver.f -lm
F=harness/fixtures
printf '%s\n' "$F/stoch/2012p578973.stoch" > /tmp/io_in.txt
printf '%s\n' "$F/velocity_model" >> /tmp/io_in.txt
printf '%s\n' "$G/stations.ll" >> /tmp/io_in.txt
printf '%s\n' "3" >> /tmp/io_in.txt
printf '%s\n' "$G/mini.bin" >> /tmp/io_in.txt
cat > "$G/stations.ll" <<'ST'
# a comment header
% and another
-179.6826 -37.6329 TEST0001
176.5 -40.25 STATTWO
174.0 -41.3 THIRD
ST
./harness/out/io_driver < /tmp/io_in.txt
printf '  mini.bin  %s bytes\n' "$(stat -c%s "$G/mini.bin")"
sed -i "s|2012p578973|alpine_base_r1|; s|mini.bin|alpine.bin|" /tmp/io_in.txt
./harness/out/io_driver < /tmp/io_in.txt
printf '  alpine.bin %s bytes\n' "$(stat -c%s "$G/alpine.bin")"
