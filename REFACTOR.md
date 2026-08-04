# Refactor plan: make it small

Bit-identity was the *porting method*, not the goal. It bought a proof that the
port is correct, and that proof is now banked: `harness/run_parity.sh` is green on
22 decks in both profiles, and the Phase 2 campaign found zero refuted endpoints
against production Fortran. From here the goal is a **small, readable
implementation of the physics** — not a transliteration of a 1990s Fortran
program.

This is deliberately two stages, because they have different gates and mixing
them destroys the cheaper one.

| | Stage 1 — organisational | Stage 2 — reduction |
| --- | --- | --- |
| Changes numerics? | **No** | Yes |
| Gate | `run_parity.sh` — exact, instant, free | Tiers B/C/D — statistical, slow |
| Failure tells you | you made a mistake | *maybe* you made a mistake |
| Reviewable by | `cmp` | judgement |

Stage 1 first is not stylistic preference. Under an exact gate, a renaming
mistake or a mis-hoisted loop invariant is caught in 40 seconds by a byte
comparison. After Stage 2 begins, the same mistake produces a slightly shifted
distribution that is indistinguishable from the numerical change you *intended*,
and you are debugging with a statistical test instead of `cmp`. Everything that
can be done bit-identically should be done before anything that cannot.

---

## What the gates mean now

`PORTING_RULES.md` remains authoritative for Stage 1 and becomes largely historical
in Stage 2. Concretely:

- **Tier A** (`run_parity.sh`) is Stage 1's gate and must stay green on every
  commit, both profiles. A debug/release disagreement means the code depends on
  optimisation-level float behaviour, which has already caught one real bug
  (`PORTING_RULES.md` §4b).
- **Tier B** (paired, matched seeds vs the oracle) is Stage 2's workhorse and is
  currently *asleep*: it returns `gm_ratio` exactly 1.000000 with zero half-width,
  because the build is bit-identical. The moment numerics change it becomes an
  extremely sensitive paired test — same RNG stream, same realisations, so the
  only difference is the one you introduced. **Run it first after every Stage 2
  change**; it will localise a regression that Tier C only reports as a wobble.
- **Tier C** (distributional vs production Fortran) is the acceptance gate. n=2500
  per stratum resolves ±2%; see `stats::sample_size_for` for why the obvious
  smaller n is underpowered.
- **Tier D** (inter-frequency correlation) is now Holm-corrected and A/A
  calibrated — see below. Use it as a release gate, not a per-commit gate; it
  costs ~25 minutes.

### Tier D is calibrated, and it found something

The A/A control (`--aa`, one binary split in half, 600v600) returns **0 of 15
flagged at raw p<0.05**, min p = 0.055, where ~0.8 flags are expected. So the
permutation test does **not** over-reject on a max-over-435-pairs statistic. That
removes the benign explanation for the production comparison, which flags **4 of
15** (P(≥4) = 0.0055 under a calibrated null) — and all four fall on the 090/000
components, which consume the live RNG inside the subfault loop, with **zero** on
the vertical, which uses pre-drawn uniforms from `ranu2`.

The effect is **localised, not a global shift**, and this is worth stating
precisely because it constrains the explanation:

| | A/A (same code) | vs production |
| --- | --- | --- |
| mean max\|Δρ\| | 0.1885 | 0.1927 (permutation p = 0.35 — no shift) |
| median | 0.1929 | 0.1776 (*lower*) |
| max | 0.2159 | 0.2735 |
| values above the entire A/A range | — | **4** |

So production is not uniformly less correlated; it is the same on average with a
heavier upper tail in a few (stratum, component) cells. The most likely cause is
**PCG32 vs gfortran's intrinsic generator**, consistent with the component split.
No single test survives Holm, so this is a lead, not a defect. Two consequences
for this plan:

1. **Keep PCG32.** It is portable, auditable, seeded reproducibly, and almost
   certainly the better generator. The difference from production is a documented
   divergence (`PROVENANCE.md` #1), not a bug to chase.
2. Tier D's absolute p-values against *production* have a small non-null floor.
   For Stage 2, compare **Rust-before vs Rust-after** instead, where the RNG is
   shared and an unchanged transform gives Δρ ≡ 0 exactly.

Worth doing when convenient: a prod-vs-prod A/A, which would control for "different
code, same RNG family" and pin the attribution completely.

---

## Stage 1 — organisational refactor

Bit-identical throughout. Every item below is verifiable by `cmp`.

### 1.1 Input restructuring — the single biggest readability win

The current model is a 22-line positional list-directed deck on stdin, and it is
actively dangerous: **list-directed reads span record boundaries**, so one missing
or extra token silently rebinds every field downstream, with no error. This is not
hypothetical. Verified against `workflow/scripts/hf_sim.py:149-158`:

| Fortran variable | Actually receives | Intended |
| --- | --- | --- |
| `ispar_adjust` | a hardcoded `0` | `tect_type` |
| `targ_mag` | `..._fault_area` | `..._target_magnitude` |
| `fault_area` | `..._target_magnitude` | `..._fault_area` |
| — | `tect_type`, **discarded** | `ispar_adjust` |

So `ispar_adjust` is permanently 0, `spar_fac` is permanently 1.0, and **the whole
stress-parameter-adjustment feature is inert in production regardless of
configuration**. It is harmless today only because all three fields default to
`-1`. Report this upstream; it is a production bug, not a port bug.

Target shape — typed, defaulted, no sentinels:

```rust
pub struct HfConfig {
    pub stress_drop: f32,
    pub rayset: Vec<RayType>,
    pub site_amp: bool,
    pub seed: i32,
    pub duration: f32,
    pub dt: f32,
    pub fmax: f32,
    pub kappa: f32,
    pub qfexp: f32,
    pub rupture_velocity: RuptureVelocity,       // { frac, shallow, deep }
    pub czero: f32,
    pub calpha: Option<f32>,                     // None -> 0.1, not -99.0
    pub moment: Option<f32>,                     // None -> derive from slip model
    pub path_duration: PathDurationModel,        // enum, not 0/1/2/11/12
    pub rv_sig1: f32,
    pub stress_param_adjust: StressParamAdjust,  // None | LeonardActive | LeonardStable
}

pub fn simulate(
    config: &HfConfig,
    slip: &StochModel,
    vmod: &VmodIn,
    station: Station,
) -> Result<Simulation, Error>;

pub struct Simulation {
    pub ndata: usize,
    pub dt: f32,
    pub acc: Vec<f32>,               // interleaved 090/000/ver
    pub epicentral_distance_km: f32, // returned, not scraped off stderr
}
```

Four points, each load-bearing:

- **`Option<T>` replaces sentinels.** The current rule is "default applies when the
  value is below −1.0", not below zero — a distinction invisible at the call site.
- **Enums replace magic integers.** `ipdur_model` accepts 0/1/2/11/12 and anything
  else leaves `ndur` undefined in the Fortran. An enum makes that unrepresentable.
- **Return data, don't write files.** The open-without-truncate plus `seek_bytes`
  scheme exists to let many processes stitch one output file. In a library that is
  the caller's problem, and the Python wrapper never wants it.
- **One station per call.** This matches production (`nsite` is always 1) *and* is
  required for correctness: `ranu2` fills `rna`/`rnb` once before the station loop
  and `normal_random_number` draws per station from the same stream, so a
  multi-station run is not a concatenation of single-station runs. Treat `nsite > 1`
  as dead.

**Keep `deck::parse` as a compatibility shim.** Non-negotiable: `run_parity.sh`
drives the binary with generated decks and is the only remaining link to the
Fortran. The binary becomes parse-deck → build-config → `simulate` → write. Delete
the deck path only if you are willing to lose the Fortran gate, which you are not.

### 1.2 Delete dead code

Confirmed by grep, zero non-test callers:

| Item | Status |
| --- | --- |
| `fort::nint64`, `fort::int_trunc64`, `fort::imod` | no references at all |
| `fort::sign` | referenced only by its own 3 test assertions |
| `Array1::{filled, as_mut_slice}`, `Array2::fill`, `ListReader::position` | no non-test callers |

Note `fort::sign` → `f32::copysign` would be a **trap**, not a simplification:
`sign(-3.0, 0.0)` must be `+3.0`, and `copysign` with `-0.0` gives `-3.0`. It is
dead, so delete it rather than rewriting it.

None of the 27 dead Fortran subprograms were ever ported (Phase 0g's analysis was
applied at porting time), so there is nothing to reap there.

Dead *deck fields* under the production config — `nbu`, `flo`, `fhi` (dead at
`ift=0`), `vp_sig`/`vsh_sig`/`rho_sig`/`qs_sig` (all 0.0), `ic_flag`,
`velocity_name`, `nsite`. These should not appear in `HfConfig` at all. Keep
`rv_sig1`: it is 0.1 in production, so the `fgrand` perturbation path is **live**.
Only `famprand` is dead.

Extend the existing `unreachable!()` pattern (`grandvel` at `nl_skip < 0`,
`filter3d` at `ift = 0`) rather than deleting silently — the git log is the grave,
but a guard documents *why* the branch is gone.

### 1.3 Split `main.rs`

`run()` is 730 lines doing deck parsing, model setup, three nested loops and IO.
Extract along the seams the Fortran already has, and which are currently marked
only by comments:

- config parsing — **done** (`c29e40b`: `Deck` + `read_deck`)
- path-duration table construction — **done** (`PathDuration`)
- the rupture-velocity taper — **done** (`RuptureVelocity::factor`)
- per-station model setup
- the **window pass** (`j` outer, `i` inner)
- the **subfault pass** (`i` outer, `j` inner — the opposite order, and it is
  load-bearing because `irandcnt` is consumed in that order)
- accumulation and output

That opposite-order pair is exactly the sort of thing that should be a named
function with the constraint in its doc comment, not a comment 200 lines into a
loop nest.

Note that this step *grows* the file: `c29e40b` took `run()` from 730 to 597 lines
while `main.rs` went 865 → 963, because structs and their doc comments cost more
than the inlining saved. That is the correct trade here — the reduction is Stage 2's
job, and `Deck` is the scaffold §1.1 needs.

### 1.3b Rustify the control flow — iterators, enums, and grouped state

**Done** — see `700399f`, `d3af03a`, `c5b1dcd`. The accumulation-order constraint
held: every conversion stayed bit-identical in both profiles on the first attempt.

Still Stage 1: **tier A gates this, and that is precisely what makes it safe to
attempt.** Converting a Fortran loop to an iterator chain is exactly the kind of
change where a subtle mistake is invisible to inspection and instantly visible to
`cmp`.

**The one hard constraint: preserve accumulation order.** Every float reduction in
this program is a left-to-right fold, and `f32` addition is not associative. So:

- `arr.as_slice().iter().sum::<f32>()` is safe — Rust's `Sum for f32` folds
  left-to-right, matching `do i = 1, n; s = s + x(i)`.
- `.fold(0.0, |a, b| a + b)` is safe for the same reason.
- Anything that reassociates is **not**: `rayon`'s parallel reductions, tree/pairwise
  summation, `chunks().map(sum).sum()`, or reordering a loop to iterate a different
  axis first. If the gate goes red on an iterator conversion, suspect this before
  suspecting an off-by-one.

Worth doing, in rough order of payoff:

- **The subfault grid as two named iterators.** The window pass runs `j` outer /
  `i` inner and the subfault pass runs `i` outer / `j` inner, and that difference is
  load-bearing because `irandcnt` is consumed in the second order. Today it is a
  comment. Give `Segment` two iterator methods — `by_row()` and `by_column()`,
  yielding `(i, j)` — and the constraint becomes a type-level fact with the
  explanation attached to the method that embodies it.
- **Enums for the magic integers.** `irtype` is dispatched as
  `if irtype[ir] == 0 { … } else if irtype[ir] % 2 == 1 { … }`, meaning
  "straight-ray approximation" / "upgoing" / "downgoing". That is an enum with three
  variants and a method, not arithmetic on an `i32`. Same for `ipdur_model`
  (already half-done via `PathDuration`) and `ispar_adjust`.
- **Group the per-segment geometry.** `rlsu`, `phsu`, `thsu`, `dst`, `zet` are
  allocated together, filled together by `even_dist2`, and indexed together at
  `(i, j)`. One `SubfaultGeometry` struct replaces five parallel arrays and five
  arguments.
- **Group the per-component state.** `cs` and `stdd` are already `[_; 3]`; the three
  component blocks that follow are near-identical apart from which radiation routine
  they call. Note they are *not* uniform — 090/000 call `radfrq_lin` and draw from
  the live RNG, `ver` calls `radv_lin` and uses the pre-drawn `rna`/`rnb` (this
  asymmetry is what the Tier D finding turns on). A `match` over a component enum
  models that honestly; a trait would have to smuggle the difference through an
  associated type and would read worse. **Prefer the enum.**
- **`Iterator` over layers for the depth lookup.** The
  `for kk in 1..=j0 { if zdep <= depth0[kk] { k = kk; break } }` pattern appears
  three times and is `position()`. Careful: two of the three fall through with
  `k = j0 + 1` and the Fortran then *indexes* that, so the `None` case is
  load-bearing, not an error.

Where traits genuinely earn their place is narrower than it first looks. This is
one concrete program with one configuration, not a framework; most candidate traits
would have exactly one implementor. Reach for them only if Stage 2's library swaps
want to abstract over a transform (e.g. so `realfft` and the vendored radix-2 can be
compared side by side during the migration) — that is a real use, and a temporary
one.

**Ordering caveat.** Deep rustification wants plain 0-based slices, which is §2.3 in
Stage 2. Read-only reductions can go through `Array1::as_slice().iter()` today, so
do that subset now; leave anything that wants `zip` over two arrays or `chunks_mut`
until after §2.3, and expect a second, smaller 1.3b pass then.

### 1.4 Naming

This is where most of the "hard to read" actually lives. `PORTING_RULES.md` §6
already chose canonical names for the common blocks; extend that to locals.
Current transliteration artifacts: `stdd`, `dfr`, `rdna`, `cs`, `ds`, `amx2`,
`bigc`/`bigc1`/`bigc1b`, `zz`, `aa`, `twin`, `rlsu`/`phsu`/`thsu`/`dst`/`zet`,
`smoe`, `dlm`. Several are assigned five times with only the last surviving
(`bigc`), which a name should say.

Free under Tier A, and it makes every later stage cheaper to review.

### 1.5 `special.rs` — NOT Stage 1 after all; moved to §2.2b

An earlier draft of this plan claimed `special.rs` could be deleted
bit-identically, on the grounds that `dgamm`'s argument is a compile-time
constant. **That was wrong, and the correction is instructive.**

`dgamm` is called once per `stoc_f` with `gsa = 2b+1`, where

```
b = -eps * ln(eta) / (1 + eps*(ln(eps) - 1))
```

On the production path `eps = 0.2` and `eta = 0.05` are hardcoded in `main.rs`, so
the argument really is constant there:

```
b   = 1.2531499
gsa = 3.5062997341156006
dgamm(gsa)      = 3.346549271566832
accurate Γ(gsa) = 3.3465492715668317   (relative difference 1.3e-16)
```

But `eps` and `eta` are *parameters of `stoc_f`*, not constants of the program, and
`harness/kernels/tier4_driver.f:75-76` exercises the kernel with **`eta = 0.2`**,
giving a different `gsa`. Hardcoding one value would break the tier-4 golden, and
worse, would leave `stoc_f` honouring its `eta` argument for the envelope terms `b`
and `c` while silently ignoring it for the gamma normalisation. That is a
correctness landmine dressed as a simplification.

So the constant is a fact about the *call site*, not the function, and exploiting
it would couple `stoc_f` to one caller's configuration. `special.rs` stays intact
through Stage 1 and is replaced in Stage 2 by a real gamma — see §2.2b.

The measurement above is still worth keeping: it says the Stage 2 swap will move
`gm` by ~1 ulp of a double, which then narrows to `f32` in
`aa = sqrt((2c)^(2b+1) / gm)`. So the swap is very likely to be invisible in the
output even though it is not bit-identical by construction.

---

## Stage 2 — the reduction

Numerics change. Gate with Tier B first (paired, will localise), then C, then D.
Ordered by lines-removed per unit of risk.

### 2.1 FFT → `realfft` / `rustfft` — biggest win in the whole plan

`rustfft 6.4.1` and `realfft 3.5.0` are both current on crates.io. This deletes
`fft.rs` (100 lines) **and simultaneously resolves the top three items in
`PROFILE.md`**: `fast` self-time is 50.7% of the program and libm twiddles are a
further 25.2%. rustfft caches twiddles internally and is SIMD-accelerated, which
subsumes PROFILE items 1 (twiddle table), 2 (specialise `Complex::exp`) and 5
(cache behaviour at large `np2`) at a stroke.

`realfft` is the better fit: the input is real, so it halves the work again. The
transform is real→complex forward in `stoc_f` and complex→real inverse in
`highcor_f`, which is exactly `realfft`'s shape.

Two things to pin explicitly, because they are where this goes wrong:

- **Sign convention.** The vendored radix-2 kernel's `IND=-1` is the analysis
  transform, while FFTW (and rustfft) use `FORWARD = -1`. `PROVENANCE.md` covers
  the resulting flip and `harness/ab_fft.sh` measures it. Re-read both before
  wiring this up.
- **Normalisation.** rustfft applies no scaling in either direction; the Fortran
  kernel does. Get this wrong and every amplitude is off by `np2`, which Tier B
  will catch instantly and loudly.

Expect this to be the one Stage 2 item with a *visible* Tier B delta that is
nonetheless correct. Budget time for reading the Tier B output rather than
reacting to it.

### 2.2b `special.rs` → a real gamma

Deletes all 155 lines, plus the `dgamm` golden (1003 cases) that validates an
implementation which no longer exists. `statrs 0.19`, `puruspe 0.4.4` and
`libm::tgamma` all supply gamma; `libm` is the lightest if nothing else needs
`statrs`.

Expect this to be nearly invisible: the measurement in §1.5 shows `gm` moving by
about one double ulp at the production argument, and `aa` narrows to `f32`
afterwards. Verify with Tier B rather than assuming — that is exactly what Tier B
is for.

Gone with it: the `1.0e75` error sentinel, the `x > 57` guard, and the reason
`stoc_f` needed a comment about unchecked error propagation into the spectrum.

### 2.2 `fort::Complex` → `num-complex`

Already in the local registry (0.4.6). Deletes ~130 lines of macro-generated
arithmetic. The three hand-written pieces exist for gfortran bit-compatibility and
that constraint is gone:

- `abs` was `hypot` to match `CABS`; `num_complex::Complex::norm` is also hypot.
- `div` was Smith's algorithm with gfortran's exact branch.
- `mul` was the textbook four-multiply form.

Bonus: `Complex::cis` gives pure-imaginary `exp` directly, which was PROFILE item
2. If 2.1 lands first, that path may already be gone.

### 2.3 `fort::Array1`/`Array2` → 0-based slices

Deletes ~120 lines, plus the mental overhead of 1-based indexing everywhere.
**Do this last, module by module, with Tier B green after each.**

Two reasons for the caution. `PORTING_RULES.md` §3 identifies index-arithmetic
rewriting as the single most likely way to introduce a silent off-by-one. And
`PROFILE.md` measured that removing the wrapper buys **nothing** in performance —
65.4 µs against 64.9 µs for an indexed sum over 65536 elements, inside the noise,
because LLVM already elides the bounds checks.

So this item is purely about size and readability. Under the old goal that made it
a bad trade; under the new goal it is exactly the point. But it is still the
highest-risk item in the plan, so it goes at the end when the gates are trusted.

`Array2`'s column-major layout is load-bearing in one specific place — the
`stdd(0,l)` alias described in 2.5 — so resolve that item first.

### 2.4 Let the `powf` gymnastics go

`PORTING_RULES.md` §4b constrains every constant-exponent `powf` because gfortran
folds some and not others, and LLVM folds differently at `-O0` and `-O2`. There is
currently a dead store left uncomputed purely to avoid an `x**0.5`. All of this
evaporates: write the natural expression. Small line count, large clarity gain,
and it removes a documented footgun that will otherwise outlive the port.

### 2.5 `geom::delaz5` → a geodesy crate

~100 lines of a 1970s distance/azimuth formulation with three separation regimes,
one of which (`geocentric radians`) is unreachable because `even_dist2` always
passes 0. A modern geodesic (`geographiclib-rs`) is both smaller at the call site
and more accurate. Tier C, since it moves distances by metres and distance feeds
the path-duration branch selection.

### 2.6 Decisions that are yours, not mine

Two items are science calls, and I have deliberately not assumed an answer:

- **The `stdd(0, l)` out-of-bounds read.** The original reads one element before
  the array on every subfault's first accumulation sample, which delays each
  subfault's contribution by one sample. The port reproduces it faithfully
  (`main.rs::stdd_at`, `PORTING_RULES.md` §7). It is a **bug in the Fortran**.
  Under "small, not identical", do you keep bug-compatibility or fix it? Fixing it
  changes output for real (Tier C), removes a function and a 15-line comment, and
  means the port no longer matches production on a known defect. My
  recommendation: fix it, in its own commit, with the Tier C delta recorded — but
  it is your call whether production parity on a defect matters more than
  correctness.
- **`nsum` and the dead averaging machinery.** `nsum` is computed from `ratio` and
  then forced to 1 (dated 2004-04-20), which is why the Frankel operator in
  `stoc_f` carries the scaling instead. Similarly `islip_weight_avg` is set to 1
  and then 0. These are frozen switches with live consequences; collapsing them
  removes real code but bakes in a decision someone once left adjustable.

---

## Size budget

Current `crates/hb_high/src` is **4,336 lines**. Rough targets:

| Module | Now | After | Where it goes |
| --- | --- | --- | --- |
| `main.rs` | 865 | ~450 | 1.1 config, 1.3 split, 1.4 naming |
| `ray.rs` | 694 | ~600 | 1.4; `cr` partly to `num-complex` |
| `deck.rs` | 448 | ~200 | 1.1 — shrinks to a compat shim |
| `fort.rs` | 397 | ~30 | 1.2, 2.2, 2.3 |
| `input.rs` | 374 | ~300 | 1.1 |
| `radiation.rs` | 241 | ~230 | 1.4 |
| `geom.rs` | 231 | ~120 | 2.5 |
| `state.rs` | 224 | ~180 | 1.2 |
| `rng.rs` | 199 | 199 | keep — see Tier D finding |
| `stoc.rs` | 180 | ~150 | 2.4 |
| `special.rs` | 155 | **0** | 2.2b |
| `site.rs` | 135 | ~130 | — |
| `fft.rs` | 100 | **0** | 2.1 |
| `highcor.rs` | 66 | ~50 | 2.1 |
| **total** | **4,336** | **~2,650** | **≈ 40% reduction** |

Plus the indirect win: most of `PORTING_RULES.md` §4, §4b and §8 stops applying to
live code, so the *documentation* a new reader must absorb shrinks by more than the
code does.

---

## What not to do

- **Don't parallelise the subfault loop.** Production gets its parallelism from one
  process per station (`hf_sim.py`), so intra-process threading contends rather
  than adds. Revisit only if single-station latency on an alpine-scale fault
  becomes the constraint.
- **Don't delete the deck parser or `run_parity.sh`.** They are Stage 1's gate and
  the only remaining link to the Fortran oracle. Once gone, no future change can
  ever be checked against the original.
- **Don't start Stage 2 before Stage 1 is committed and green.** See the top of
  this document.
- **Don't touch `rng.rs`.** The Tier D result makes it the one numerically
  interesting module, and "small" is not a reason to disturb a generator whose
  stream is baked into every golden.
- **Don't chase the Tier D enrichment as a defect** until a prod-vs-prod A/A has
  ruled out the remaining benign explanation.

---

## Suggested commit sequence

Stage 1, each with `run_parity.sh` green in both profiles:

1. Delete dead items (1.2) — **done**, `d0e8720`, 56 lines removed.
2. Split `main.rs` (1.3) — **done**, `c29e40b` (deck, path-duration table,
   rupture-velocity taper) and `48f1d11` (source normalisation). `run()` 730 → 486.
3. Rustify control flow (1.3b) — **done**, `700399f` (grid iterators, layer lookups
   as `find`), `d3af03a` (`RayKind` enum), `c5b1dcd` (`SubfaultGeometry`).
4. Naming pass (1.4) — **done**, `c012812`.
5. `HfConfig` + `simulate()` + deck shim (1.1) — **remaining**, and it is the whole
   of what is left in Stage 1. Largest commit; split into "add typed config
   alongside the deck" then "move the binary onto it".

Still to extract from `run()` when 1.1 lands: the per-station model setup, and the
station loop body itself, which is now the bulk of the remaining 486 lines.

(`special.rs` was originally item 2 here and has moved to Stage 2 — see §1.5.)

Then re-baseline: `cargo bench`, regenerate `bench_baseline.csv`, and run the full
campaign to confirm Stage 1 changed nothing.

`bench_baseline.csv` is stale in two known ways, both to be fixed by that
regeneration rather than piecemeal:

- it is missing `whole_program/mini/4` and `whole_program/medium/112`, which were
  measured *after* the CSV was written;
- `geom/even_dist2` will step, because `even_dist2` now allocates its five
  `(nq, np)` arrays internally instead of taking them as out-parameters, and the
  bench no longer hoists them out of `b.iter()`. The new number is the honest one —
  the driver allocates per segment too — but it is not comparable to the old.

Stage 2, each with Tier B then C:

6. `special.rs` → real gamma (2.2b). Smallest Stage 2 change, so it is the right
   one to exercise Tier B on first — a near-invisible delta is easier to read than
   a large one.
7. FFT (2.1) — expect a real Tier B delta; verify it is the one you intended.
8. `num-complex` (2.2).
9. `powf` cleanup (2.4).
10. `delaz5` (2.5).
11. The two science decisions (2.6), separately, each with its Tier C delta recorded.
12. `Array1`/`Array2` (2.3), module by module, last.
13. A second, smaller 1.3b pass, now that slices make `zip`/`chunks_mut` available.

Re-run Tier D at the end of Stage 2 as a release gate, comparing Rust-before vs
Rust-after rather than against production.
