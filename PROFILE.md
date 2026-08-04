# Where the time goes

Measured on the current release build, which is bit-identical to the Fortran
oracle. Two independent sources, and they agree:

- **criterion microbenchmarks** — `harness/bench_baseline.csv`, 49 benchmarks,
  reproduce with `cargo bench`
- **`perf record -F 999 --call-graph fp`** on the 112-subfault fault, built with
  `-C force-frame-pointers=yes`

Numbers are the 112-subfault fault (`2013p543824`) at `duration=20`, `dt=0.005`,
which gives `np2 = 16384`. Whole-program time **812 ms**, of which 7.25 ms per
subfault.

---

## The headline

| | |
| --- | --- |
| `hb_high::fft::fast` self time | **50.7%** |
| `libm` (transcendentals, almost all twiddles) | **25.2%** |
| everything else in `hb_high` | 23.1% |

`fast` splits almost exactly evenly between its two callers — 25.15% via
`highcor_f`, 25.26% via `stoc_f` — which is what you would expect from three
forward transforms in `stoc_f` and three inverse in `highcor_f` per subfault.

## Cost model, and it checks out

Built from the microbenchmarks alone, per subfault at `np2 = 16384`:

| component | per subfault | share |
| --- | --- | --- |
| `stoc_f` × 3 | 4.79 ms | 61.5% |
| `highcor_f` × 3 | 2.09 ms | 22.4% |
| `radfrq_lin` × 2 | 0.09 ms | 7.1% |
| `siteamp` × 3 | 0.31 ms | 5.8% |
| `radv_lin` × 1 | 0.03 ms | 2.5% |
| `gf_amp_tt` × 1 | 0.01 ms | 0.7% |
| **model total** | **7.0 ms** | |
| **measured** | **7.25 ms** | |

The model predicts the end-to-end time to within 4%, which means attribution
below is trustworthy rather than plausible.

## Call counts, because they explain the shape

Costs here are driven by *how often* things run, not by any single routine being
slow. Per subfault, per ray:

| | count |
| --- | --- |
| FFTs of length `np2` | 6 (3 forward in `stoc_f`, 3 inverse in `highcor_f`) |
| normal deviates | 3 × `np2` |
| `rdatn` calls | ~3000 (1000 each in two `radfrq_lin` and one `radv_lin`) |
| uniform draws | ~10000 (5 per iteration × 1000 × 2 `radfrq_lin`) |
| `libm` calls for twiddles | 3 × (`np2` − 1) per FFT ≈ **295k per subfault** |

That last row is the whole story of the 25% in libm. `fast` computes
`CEXP(THETA)` once per `(stage, k)` pair, which is `np2 − 1` times per transform,
and each one is three libm calls: `expf`, `cosf`, `sinf`. At `np2 = 16384` over
six transforms that is ~295k libm calls per subfault, ~33M for this fault.

`np2` also **grows with fault size**, so this scales worse than the subfault
count alone suggests. Per-butterfly cost from the FFT benchmarks:

| `np2` | time | ns per element-stage |
| --- | --- | --- |
| 1024 | 19.9 µs | 1.95 |
| 4096 | 85.1 µs | 1.73 |
| 16384 | 611 µs | 2.67 |
| 65536 | 3.66 ms | 3.49 |

An `O(N log N)` algorithm should hold that column flat. It doubles from 4096 to
65536 because 65536 complex `f32` is 512 KB, past L2 — so the alpine fault pays a
cache penalty on top of everything else.

---

## What to do, in order

Each item is tagged with the validation tier it needs (see the plan):
**A** = must stay bit-identical, **B/C** = needs the statistical gates.

### 1. Twiddle table in `fast` — tier A, expect ~1.3×

Precompute `CEXP(THETA)` per `(stage, k)` once per transform length and reuse it
across calls. Removes most of the 25% in libm.

This was already measured on the Fortran side in the earlier EMOD3D profiling
work: **bit-identical** (md5-verified) and worth **1.35–1.43×**. It is exact by
construction — caching a pure function of `(ind, k, kmax)` cannot change its
value, provided the table is built with the *identical* expression, including the
literal `3.141593`. Deriving twiddles by recurrence or half-table symmetry instead
would **not** be bit-identical and would drop to tier C.

### 2. Specialise `Complex::exp` for a pure-imaginary argument — tier A, expect ~8%

`fast` builds `theta` with `re = 0.0` and then calls `Complex32::exp`, which
evaluates `expf(0.0)`, `cosf(im)` and `sinf(im)`. `expf(0.0)` is exactly `1.0`
and `1.0 * x` is exact, so skipping it is bit-identical and removes one of the
three libm calls in the twiddle path.

Worth doing even alongside item 1, since it also helps any future caller.

### 3. Reconsider `hypotf` — tier A if careful, 2.0%

`Complex32::abs` uses `hypot`, which is deliberate: gfortran's `CABS` is a hypot
and `sqrt(re² + im²)` differs in the last bits (`PORTING_RULES.md` §4). The 2.0%
is real but the swap is **not** bit-identical, so it is tier C, not A.

There is a tier-A version: `stoc_f` computes `cabs(ac(i))*cabs(ac(i))`, i.e.
`hypot(re,im)²`. That is not equal to `re²+im²` in floating point, so it cannot be
simplified — but it does call `hypot` twice where once would do. Hoisting to a
single call is exact.

### 4. `stoc_f`'s envelope and spectrum loops — tier C, ~9% self

`stoc_f`'s own 9% self time is two loops over `np2`/`nf` with a transcendental
each: `t.powf(b) * exp(-c*t)` for the envelope, and `powf`/`exp` per frequency bin.

`exp(-c*t)` over evenly spaced `t` is a geometric sequence and could be advanced
by repeated multiplication, but accumulated rounding makes that **not**
bit-identical. Tier C, and modest — do items 1–3 first.

### 5. Cache behaviour at large `np2` — tier C

The per-butterfly cost doubling above 16384 is a blocking problem, and blocking
changes the order of operations. Only relevant to alpine-scale faults.

## What not to do

**Do not strip the `Array1` 1-based wrapper from hot loops.** This was the
obvious candidate and the measurement kills it: an indexed sum over 65536
elements is **65.4 µs against 64.9 µs** for a plain slice sum — a 0.8% difference,
inside the noise. LLVM already elides the bounds checks in these loops. Removing
the wrapper would mean rewriting index arithmetic across every kernel, which
`PORTING_RULES.md` §3 identifies as the single most likely way to introduce a
silent off-by-one, in exchange for nothing measurable.

**Do not reach for a faster FFT algorithm first.** Radix-4 or split-radix would
cut operation counts ~25%, but items 1 and 2 get a comparable win at tier A,
whereas a new algorithm needs the full tier-C campaign.

**Do not parallelise yet.** The subfault loop is embarrassingly parallel, but
production already gets its parallelism by running one process per station
(`hf_sim.py` invokes the binary once per station), so intra-process threading
would contend with that rather than add to it. Revisit only if single-station
latency on a very large fault becomes the constraint.

---

## An aside worth acting on, independent of this port

Timing the Fortran three ways exposes something about the **production build**:

| fault | Fortran `USE_FFTW=ON` | Fortran radix-2 | Rust |
| --- | --- | --- | --- |
| 4 subfaults | 0.520 s | 0.022 s | 0.019 s |
| 112 subfaults | 1.128 s | 0.877 s | 0.765 s |

All three at `-O2` except Rust (release). So:

- **`FFTW_MEASURE` planning costs 0.25–0.50 s per process** and, at these
  transform counts, never earns it back — the FFTW build is *slower* than the
  radix-2 build on both faults.
- Because `hf_sim.py` runs **one process per station**, that cost is paid *per
  station*. On a 1000-station run it is 250–500 s of pure planning.
- Switching the production build to `FFTW_ESTIMATE`, or caching wisdom, would
  recover that immediately — no Rust required.

This also corrects the naive reading of the speedup. Against the FFTW build the
port looks 27× faster on a small fault; against an equal-algorithm Fortran at
`-O2` it is **~1.15×**. The 27× is real for deployment but the mechanism is "the
port does not pay FFTW planning", not "Rust is 27× faster". Quoting it without
that qualification would be misleading.

---

## Reproducing

```bash
cargo bench                                    # microbenchmarks
python3 harness/bench_summary.py               # -> harness/bench_baseline.csv
python3 harness/bench_summary.py --compare harness/bench_baseline.csv
harness/bench_vs_fortran.sh                    # Rust vs production Fortran

RUSTFLAGS="-C force-frame-pointers=yes" cargo build --release
perf record -F 999 --call-graph fp -- ./target/release/hb_high < deck.txt
perf report --stdio --no-children --percent-limit 1.0
perf report --stdio --sort dso --no-children   # the libm share
```
