# Coverage of the Fortran reference under the deck ladder

Phase 0g. The reference is built with `-fprofile-arcs -ftest-coverage` and driven
through every deck in `harness/run_parity.sh` (tiers 1–3, excluding the slow
alpine case). Reproduce with `harness/cov.sh`.

Whole-file figures from `gcov` are misleading, because
`reference/hb_high_ref.f` still contains the 27 subprograms that are dead in the
`BINMOD`/`VERSION1` configuration. Split by live and dead:

## The dead set is confirmed dead

**0 of 749 lines executed, across all 27 subprograms.**

| | |
| --- | --- |
| `aver_remov`, `bttr2`, `butter`, `cs2`, `distazi`, `dstdps` | 0% |
| `even_dist1`, `famprand`, `filter3d`, `gencof`, `get_sitefacsOLD` | 0% |
| `grandvel`, `highcor`, `INAC`, `prduct`, `RADFRQ`, `RADFRQ_lin5` | 0% |
| `RADTRL`, `RADUNI`, `RANN2`, `recfnct`, `refft`, `revers` | 0% |
| `set`, `stoc`, `tranm`, `zpass` | 0% |

This is the point of measuring. The dead-set classification was derived by
reading call sites and tracing reachability; this is an **independent
confirmation** of it. Had any routine been misclassified — a `grandvel` reached
by some default I misread, a `filter3d` call I thought was guarded — it would
show nonzero coverage here. None does.

## The live set is 89.9% covered

1197 of 1331 lines.

| subprogram | lines | % |
| --- | --- | --- |
| `cagcon`, `even_dist2`, `FAST`, `FLZERO`, `geom_terms`, `get_sitefacs`, `highcor_f`, `RADFRQ_lin`, `RANU2`, `RDATN`, `siteamp`, `stoc_f`, `ttime` | — | **100.0** |
| `pnot` | 32/34 | 94.1 |
| `RADV_lin` | 30/32 | 93.8 |
| `trav` | 62/68 | 91.2 |
| main program | 545/608 | 89.6 |
| `dtdp` | 15/17 | 88.2 |
| `normal_random_number` | 27/31 | 87.1 |
| `cr` | 19/23 | 82.6 |
| `DELAZ5` | 50/62 | 80.6 |
| `gf_amp_tt` | 62/77 | 80.5 |
| `DGAMM` | 21/45 | 46.7 |

## The gaps are covered at the kernel level, by design

Every routine below 100% is one whose uncovered branches are unreachable from a
*realistic deck* but are covered by its own kernel golden, where inputs can be
constructed directly. That division is deliberate: kernel goldens buy branch
coverage, the deck ladder buys integration coverage.

- **`DGAMM` (46.7%)** — the lowest figure and the least concerning. Real decks
  only ever call it with `gsa = 2b+1` for a physical window shape, so one
  argument-reduction path is used. Its golden covers 1000 cases across all four
  reduction paths (`>1.5`, `[0.5,1.5]`, `<0.5`, and negative non-integers) plus
  both error returns.
- **`DELAZ5` (80.6%)** — a single station reaches one separation regime. Its
  golden deliberately constructs near-coincident and near-antipodal geometry,
  since random lat/lon pairs essentially never hit the other two formulations.
  The geocentric-radians path is genuinely unreachable (`even_dist2` always
  passes 0) and is excluded by decision.
- **`cr` (82.6%)** — the branch-cut selection has five regimes; a real ray
  parameter visits some of them. The golden covers all five, including `|Im p|`
  just either side of the `1e-8` threshold.
- **`gf_amp_tt` (80.5%)** — the Moho-multiple loops need `itype > 2`, which
  production never requests (`rayset` is `[1]`). Covered in its golden and in
  the `rayset=1,2` / `rayset=1,3` decks.
- **`pnot` (94.1%)** — the uncovered lines are the immediate-return path, which
  is **structurally unreachable**; see the note in `tests/tier3_golden.rs`.
- **`trav` (91.2%)** — the P and SV mode branches; production is SH only.
  Covered in the tier-1 golden.
- **`normal_random_number` (87.1%)** — the zero-rejection retry loops, which
  need `rand_numb` to return exactly 0.0.

## What is *not* covered anywhere

Worth stating plainly rather than leaving implied:

- `pnot`'s 40-iteration bisection cap. Every observed case exits on the
  `|a| <= 0.01` tolerance instead.
- The zero-rejection retries in `normal_random_number`.
- `DGAMM`'s unit-6 diagnostic writes are exercised by its golden but not by any
  deck.
- The `np2 > mm` abort in the main program.

None of these are reachable by adjusting deck parameters; they would need either
a pathological velocity model or a specific RNG output. They are recorded here so
the 89.9% figure is not mistaken for "everything that matters is covered".
