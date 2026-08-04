#!/usr/bin/env python3
"""Split gcov line coverage into the live and dead subprogram sets.

Usage: cov_report.py <path to hb_high_ref.f.gcov>
"""
import re
import sys

LIVE = {
    "main", "stoc_f", "radfrq_lin", "radv_lin", "highcor_f", "rdatn", "fast",
    "ranu2", "flzero", "even_dist2", "delaz5", "dgamm", "get_sitefacs",
    "siteamp", "gf_amp_tt", "cagcon", "cr", "dtdp", "pnot", "trav", "ttime",
    "geom_terms", "init_random_seed", "normal_random_number", "rand_numb",
    "pcg32_next", "pcg32_skip",
}

hdr = re.compile(r"^\s*(?:[a-zA-Z*0-9 ]*?)(subroutine|function)\s+([a-zA-Z_0-9]+)", re.I)
cur, stats = "main", {}
gcov_path = sys.argv[1] if len(sys.argv) > 1 else "hb_high_ref.f.gcov"
for raw in open(gcov_path, errors="replace"):
    parts = raw.split(":", 2)
    if len(parts) < 3:
        continue
    cnt, src = parts[0].strip(), parts[2]
    m = hdr.match(src)
    if m and not src.lstrip().lower().startswith(("c", "!", "*")):
        cur = m.group(2).lower()
    if cnt == "-":
        continue
    ex, tot = stats.get(cur, (0, 0))
    hit = 0 if cnt.startswith(("#", "=")) else 1
    stats[cur] = (ex + hit, tot + 1)


def show(names, title):
    print(f"\n{title}")
    te = tt = 0
    for n in sorted(names):
        e, t = stats[n]
        te, tt = te + e, tt + t
        print(f"  {n:22} {e:5}/{t:5}  {100*e/t if t else 0:5.1f}%")
    if tt:
        print(f"  {'TOTAL':22} {te:5}/{tt:5}  {100*te/tt:5.1f}%")
    return te, tt


show(LIVE & set(stats), "LIVE subprograms")
de, dt = show(set(stats) - LIVE, "DEAD subprograms (expected 0%)")
if de != 0:
    raise SystemExit(
        f"\nFAIL: {de} lines executed in the dead set. A subprogram believed "
        f"unreachable was reached, so the dead-code analysis is wrong."
    )
print("\nOK: the dead set executed 0 lines, confirming the reachability analysis.")
