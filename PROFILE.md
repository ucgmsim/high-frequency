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

## Measured and rejected

Do not retry these without re-measuring. Each was tried, measured, and found to make things
worse or to be wrong:

- `lto = "fat"` + `codegen-units = 1` — **1.2% slower** on the medium fault, plus 29 s per
  rebuild. Plausibly because `rustfft`'s AVX kernels are hand-tuned and cross-crate inlining
  disturbs their register allocation.
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
