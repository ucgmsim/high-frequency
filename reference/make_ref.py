#!/usr/bin/env python3
"""Generate hb_high_ref.f from the vendored hb_high_orig.f.

Applies exactly two changes, both documented in PROVENANCE.md:

  1. Remove ``init_random_seed`` and ``rand_numb``; ``pcg32.f`` supplies
     replacements. ``normal_random_number`` is KEPT unchanged -- it is the
     Box-Muller + unit-RMS-renormalisation algorithm, not the generator, and
     ``stoc_f``'s amplitude calibration is tuned against it.
  2. Collapse the ``#if defined (USE_FFTW)`` / ``#else`` / ``#endif`` around
     ``FAST`` down to the radix-2 implementation only.

Every excision is guarded by an assertion on the text actually found at those
lines, so if the vendored source is ever re-synced from a different EMOD3D
commit this script fails loudly instead of silently cutting the wrong code.

Line numbers are 1-based and refer to hb_high_orig.f as vendored at EMOD3D
commit 51ed6b5.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "hb_high_orig.f"
DST = HERE / "hb_high_ref.f"

EXPECTED_LINES = 4144

# (first, last) inclusive, 1-based; a substring that must appear on `first`
# and on `last`, as a guard against drift.
CUTS = [
    # FFTW FAST: the #if line, the whole FFTW-backed routine, and the #else.
    (2339, 2394, "#if defined (USE_FFTW)", "#else"),
    # The matching #endif after the radix-2 routine.
    (2425, 2425, "#endif", "#endif"),
    # rand_numb -- replaced by pcg32.f. Cut before init_random_seed so the
    # earlier line numbers stay valid while we work backwards.
    (4136, 4144, "function rand_numb(iflag)", "end"),
    # init_random_seed -- replaced by pcg32.f.
    (4062, 4081, "subroutine init_random_seed(irand)", "end"),
]

NOTE = {
    2339: """cREF ------------------------------------------------------------------
cREF The USE_FFTW variant of FAST has been removed; only the radix-2
cREF implementation below is built. Reason: it planned with FFTW_MEASURE,
cREF which picks a plan by timing it, so that build is not reliably
cREF bit-reproducible even against itself.
cREF
cREF The two variants also used opposite Fourier sign conventions. That was
cREF measured (harness/ab_fft.sh) to cancel: max rel diff 8.1e-07 against
cREF the waveform peak, correlation 0.9999999999996, median 4 ulp.
cREF See reference/PROVENANCE.md.
cREF ------------------------------------------------------------------
""",
    4062: """cREF ------------------------------------------------------------------
cREF init_random_seed and rand_numb have been moved to pcg32.f, which
cREF replaces gfortran's intrinsic generator with PCG32 so the random
cREF stream is reproducible in Rust.
cREF
cREF normal_random_number below is UNCHANGED -- it is the Box-Muller plus
cREF unit-RMS renormalisation algorithm, not the generator.
cREF
cREF pcg32.f pins the seed-word count to 8, matching what gfortran 16.1.1
cREF reports from random_seed(size=n). That matters because the original
cREF init_random_seed incremented its own argument once per seed word, and
cREF hb_high reads that mutated value at line 1366 to gate rupture-time
cREF jitter -- so the word count changes which branch is taken.
cREF See reference/PROVENANCE.md.
cREF ------------------------------------------------------------------
""",
}


def main():
    lines = SRC.read_text().splitlines(keepends=True)
    if len(lines) != EXPECTED_LINES:
        sys.exit(f"{SRC}: expected {EXPECTED_LINES} lines, found {len(lines)}. "
                 "The vendored source has changed; re-verify the cut ranges "
                 "in CUTS before regenerating.")

    for first, last, first_needle, last_needle in CUTS:
        got_first, got_last = lines[first - 1], lines[last - 1]
        if first_needle not in got_first:
            sys.exit(f"guard failed at line {first}: expected {first_needle!r}, "
                     f"got {got_first.strip()!r}")
        if last_needle not in got_last:
            sys.exit(f"guard failed at line {last}: expected {last_needle!r}, "
                     f"got {got_last.strip()!r}")

    # Apply cuts from the bottom up so earlier indices stay valid.
    for first, last, _, _ in sorted(CUTS, reverse=True):
        replacement = [NOTE[first]] if first in NOTE else []
        lines[first - 1:last] = replacement

    DST.write_text("".join(lines))
    print(f"wrote {DST} ({len(DST.read_text().splitlines())} lines, "
          f"from {EXPECTED_LINES})")


if __name__ == "__main__":
    main()
