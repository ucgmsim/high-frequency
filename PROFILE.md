# Where the time goes

Re-measured at Stage 4 §4.6. The previous version of this file was written at §2.3 and
described a program that no longer exists — it quoted `fast` at 48.5% self time for a
function deleted in §2.1, and its whole-program numbers timed a subprocess.

## Whole program, one station

| fault | subfaults | median |
| --- | ---: | ---: |
| mini | 4 | **3.44 ms** |
| medium | 112 | **176.4 ms** |

## Realistic scale, measured

Extrapolating the table above linearly to a 4,070-subfault rupture gives ~7 s per station.
**Measured, it is 26 s** — 3.7× worse — because runtime is not linear in subfault count alone.

| | subfaults | duration | 1 station | 4 stations |
| --- | ---: | ---: | ---: | ---: |
| medium, 40 s | 112 | 40 s | 176 ms | — |
| medium, 60 s, shallow dip | 112 | 60 s | **1.52 s** | 5.97 s |
| realistic, 60 s, shallow dip | 4,070 | 60 s | **26.2 s** | 104.7 s |

Two effects compound. A longer record raises `np2`, so every subfault's FFT costs more; and a
shallow-dipping deep-topped geometry lengthens ray paths, which adds time windows per
subfault. The same 112 subfaults cost **8.6×** more at 60 s with a subduction-like geometry
than at 40 s with a crustal one. **Subfault count alone does not predict runtime** — anyone
sizing a campaign needs to time their own geometry.

For a 1,000-station run that is ~7.2 CPU-hours, so about an hour on this 8-core box or a few
minutes on a large node. A 100-realisation campaign is ~720 CPU-hours, which is genuinely
cluster work.

Four stations cost exactly 4× one (26.16 vs 26.19 s each), confirming the batch loop is
serial by design: the GIL is released so a dask thread pool scales across chunks, and there is
no internal thread pool to compete with it.

## Sizing the transform per subfault, and the ladder it snaps to

Measured on a real production deck — `AlpineF2K_REL33.stoch`, 1,854 subfaults over 7
segments, `dt = 0.005`, a 174.5 s record, `path_duration_model = 11` — at two stations that
bracket the geometry. **far** is the deck's own first station, about 250 km south of the
fault's southern end and 600 km from its northern one; **near** sits on top of segment 5.
Release build, one station per measurement.

| | far | near |
| --- | ---: | ---: |
| before | 39.0 s | 24.3 s |
| per-subfault window + ziggurat, powers of two | 16.6 s | 8.8 s |
| **plus the four-rung ladder** | **12.5 s** | **5.7 s** |
| | **3.12x** | **4.30x** |

Three changes, and the attribution:

* **The transform is sized per subfault**, not per segment. `tmax` was the maximum window
  over every subfault in a segment, and every subfault in it got a buffer that long, so a
  subfault whose envelope had decayed to `η²` of its peak by sample 8,000 still drew 131,072
  normal deviates and transformed all of them. Worth roughly **1.85x** — the residual of the
  2.35x second row after the ziggurat's share below.

* **Ziggurat normals instead of Box-Muller**, `rng::Pcg` against `rng::LegacyPcg`. Directly
  benched at `rng/normal` vs `rng/legacy_normal`: **3.2x** on the fill itself (235 µs against
  691 µs at n = 65536, a 69% reduction). That loop was 31% of a baseline station, so about
  **1.27x** of the whole.

* **A four-rung length ladder** instead of powers of two — `fft::good_length`. Worth
  **1.33x** at the far station and **1.55x** at the near one, measured by rebuilding with
  `LENGTH_MULTIPLIERS = [1]` and changing nothing else. It helps the near station more
  because its windows vary more, so more of them land just above a rung.

### The numbers moved, and the level did not

All three changes alter the draw structure, so `harness/golden/snapshot.txt` moved and was
re-recorded. Its four RMS fields moved by +7.6%, −6.5%, −3.5% and +1.3% — mixed signs.

That is a realisation change rather than a level change, and it was checked rather than
asserted. Over **12 seeds at the far station**, record RMS:

| | mean RMS | sd | sd/mean |
| --- | ---: | ---: | ---: |
| before | 0.29701 | 0.01843 | 6.20% |
| after | 0.29887 | 0.01511 | 5.06% |

A **+0.63%** shift in the mean against a ±2.3% standard error on the difference:
indistinguishable from zero. A single seed had shown record energy dropping 21%, which
looked alarming and is not — energy is RMS², so 21% is a 10% RMS excursion, under two
standard deviations of the scatter above.

**This does not replace the statistical campaign.** It is one station, one seed count and one
statistic — record RMS, not a spectral intensity measure. What it rules out is a gross level
shift, which is the failure that would have made the change obviously wrong. `ENGINEERING_RULES`
§6 has the recipe for the real adjudication.

## Building the simulator once buys almost nothing, and that is the finding

`Simulator::new` hoists everything station-independent out of the station loop: the air
layer, the rupture taper, the path-duration table, the per-segment angles, and
`normalise_source`, which walks every subfault three times and was preceded by a full clone
of the slip model. Before it, a batch redid all of that per station.

**It is worth ~2% on a tiny fault and nothing on a realistic one.** `batch` in
`benches/whole.rs` runs 16 stations two ways — one simulator shared, versus one rebuilt per
station:

| fault | subfaults | shared | rebuilt per station | saving |
| --- | ---: | ---: | ---: | ---: |
| mini | 4 | 56.5 ms | 57.6 ms | 1.9% |
| medium | 112 | 2.891 s | 2.875 s | none, within noise |

The saving *shrinks* as the fault grows, which is the opposite of the intuition that made
this look worth measuring: setup scales with subfault count, so it ought to matter more.
It does scale — but `run` scales with subfaults **times** rays times components times an
FFT each, so the setup's share falls away. At 112 subfaults it is already below the noise
floor on a loaded machine.

So the refactor stands on its structure, not its speed: `run(&self)` is what lets one
simulator be shared across a dask thread pool, and the per-station clone of the slip model
is gone. Nobody should spend further effort optimising the setup path.

## What the batch API removed

Stage 3 measured the mini fault at ~5.8 ms **through the CLI**, against 3.44 ms calling the
library. The difference is process startup, deck generation, text parsing and a file write —
roughly 2.4 ms, which on the smallest fault was about 40% of the total. That is the whole
argument for the batched interface: production runs one station per process, so this overhead
was paid once per station, thousands of times per campaign.

Treat the comparison as indicative rather than exact. The two measurements differ in the
geometry as well as the interface, so they are consistent with a ~2.4 ms saving rather than
proof of one.

## Kernels

| kernel | time |
| --- | ---: |
| `fft` forward/inverse, 16384 | 35.4 / 35.9 µs |
| `fft` forward/inverse, 65536 | 190 / 189 µs |
| `rng::next_f32` | 1.21 ns |
| `fill_normal_deviates`, 65536 | 669 µs |
| `fill_uniform_deviates`, 65536 | 73.3 µs |
| `radiation_pattern` | 25.3 ns |
| `horizontal_radiation_spectrum`, nr=1000 | 50.1 µs |
| `vertical_radiation_spectrum`, nr=1000 | 28.6 µs |
| `cagniard_time` | 192 ns |
| `cagniard_time_derivative` | 224 ns |
| `vertical_slowness` | 10.2 ns |

`fill_normal_deviates` costs **9×** `fill_uniform_deviates` for the same count, which is
Box-Muller's two transcendentals against a shift and a multiply. That ratio is why §3.3's
hoists paid: the expensive per-sample work in this program is transcendental, not arithmetic.

## The LTO experiment was two experiments

Stage 2 tested `lto = "fat"` and `codegen-units = 1` as a single change, found it 1.2%
slower, and rejected both. **They do not behave the same way**, and bundling them threw out
the good half. Re-run at §5.7, six interleaved rounds per build, whole pipeline through the
release snapshot binary:

| | instructions | cycles | IPC | rebuild |
| --- | ---: | ---: | ---: | ---: |
| baseline, `codegen-units = 16` | 3,489,660,418 | 1,699,821,294 | 2.053 | ~10 s |
| **`codegen-units = 1`** | 3,434,677,223 (−1.58%) | **1,575,930,648 (−7.29%)** | **2.180** | 50 s |
| `+ lto = "fat"` | **3,378,573,086 (−3.18%)** | 1,658,985,055 (−2.40%) | 2.037 | 67 s |

LTO removes the **most** instructions and is still the worse option: it hands back most of
the cycle win by dropping IPC from 2.18 to 2.04. That is exactly the mechanism the old note
guessed at — `rustfft`'s AVX kernels are hand-tuned and cross-crate inlining disturbs their
register allocation — so the reasoning was right and only the packaging was wrong.

**The lesson is about the instrument, not about LTO.** Every other entry in this file is
settled on instructions retired, and on instructions alone LTO wins and is wrong. A change
that alters how well code schedules needs cycles too.

`codegen-units = 1` is now on, at ~40 s per release rebuild.

Measured on a box that was **not idle** — six other agent processes, load ~1.3 of 8 cores.
That is fine for these counters and would not have been for wall clock: instructions retired
reproduced to 1.7 parts in 10⁷ *under that load*, and the cycle spread within a build was
~0.5% against gaps of 7.3% and 2.4%. Wall-clock numbers elsewhere in this file were taken on
a quiet machine and were **not** re-measured at §5.7.

**Not isolated:** whether ndarray's arrival in Stage 5 caused this, or whether
`codegen-units = 1` was always worth 7%. That needs a build of the pre-Stage-5 tree, which
has not been done. Do not assume either way.

## Measured and rejected

Do not retry these without re-measuring. Each was tried, measured, and found to make things
worse or to be wrong:

- `lto = "fat"` — **rejected again at §5.7, but for a sharper reason.** See "The LTO
  experiment was two experiments" below; the short version is that it removes the most
  instructions of any option here and is still the wrong choice.
- `overflow-checks = false` — **1.87% more** instructions retired. Genuinely
  counterintuitive; removing the checks perturbs codegen elsewhere by more than it saves.
- `sin` from `sqrt(1 - cos²)` — saved 2.46% and was **wrong in `f32` by 2.4e-4**.
- Shrinking `stoc.rs`'s `as_` buffer from `np2` to `fold_count` — **+5.4M instructions**. At
  np2 = 16384 the `f64` buffer is exactly 128 KB, glibc's `M_MMAP_THRESHOLD`, so `calloc`
  returns pre-zeroed pages for free; at 64 KB it comes off the heap and must be memset.
- A 1-based `Vec` wrapper as a half-measure — +0.62%.

Timing on this box drifts 1–3% between runs, which is the same size as several of those
effects, so anything marginal was settled on **instructions retired** (reproducible to about
1 part in 10⁷) rather than wall clock.
