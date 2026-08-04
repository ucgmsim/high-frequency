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

### Working cheaply: bit-parity first, bisect on failure

Running the statistical campaign after every change is the wrong default — Tier C is
25 minutes and Tier D another 25. The cheap checks catch nearly everything:

| check | cost | when |
| --- | --- | --- |
| `cargo test` | ~0.1 s release | every edit |
| `harness/run_parity.sh` (release only) | ~40 s | every commit |
| Tier B (50 seeds) | ~1 min | when parity goes red *on purpose* |
| Tier C / Tier D | ~25 min each | batch boundaries only |

**Bit-parity is a cheap test, not just a Stage 1 gate.** Plenty of Stage 2 changes
turn out bit-identical in effect even when they are not so by construction — §2.2b's
gamma swap moved `f64` by three ulps and the output did not change at all, because the
consumer narrows to `f32`. So keep running it, and treat the result as a *classifier*:

- **parity green** — nothing further needed, commit and move on.
- **parity red, and the change was meant to be exact** — a bug. Do not reach for the
  statistical gates; they are far too slow to localise anything. Bisect.
- **parity red, and the change was meant to alter numerics** — now Tier B earns its
  keep, because it is paired and will say *how much* moved.

**Commit small and often.** The value is not tidiness, it is that `git bisect` over
twenty small commits finds a regression in four or five parity runs. Over three large
ones it tells you almost nothing.

**A periodic Tier C is only worth running if the binary's output actually moved.** Prove
it rather than assume it either way: `run_selfparity.sh <commit-of-last-tier-C>` in
bit-exact mode. If it passes, the binary is byte-identical to the one Tier C already
validated and a re-run reproduces the same CSV — 50 minutes for no information. A run of
bit-exact commits, which is what a well-behaved §2.3 looks like, needs no re-validation at
all.

**The drift baseline has to advance past deliberate structural changes.** `run_selfparity`
compares sample `i` to sample `i`, so a change that *translates* the waveform reads as a
difference of order the waveform itself, not of order the change. §2.6's defect-1 fix
shifted every trace one sample earlier, and from that commit onward a 1e-4 drift check
against `pre-loop` fails on all 22 decks — correctly, and uninformatively. Measured either
side of it:

| drift check | result |
| --- | --- |
| vs `pre-loop` (before the shift) | FAIL at 1e-4 |
| vs `fff2abf` (before the shift) | FAIL at 1e-4 |
| vs `262c75f` (the shift itself) | **PASS at 1e-4** |

So compare against the most recent `green-NNN` tag rather than a fixed origin. Anything
else conflates "we have accumulated error" with "we deliberately moved the waveform", and
only the first is a problem.

### How sensitive the gates actually are — measured, and lower than assumed

After §2.1, §2.2, §2.2b, §2.4 and both §2.6 defect fixes, Tier C at 2500 seeds returned
**the same verdict counts as before Stage 2 began**: 324 certified, 0 refuted, 51
undetermined. That is reassuring, but it is weaker evidence than it looks, and the reason
is worth knowing before reading any future "0 REFUTED".

Comparing the endpoint geometric-mean ratios directly against the pre-Stage-2 run:

| | |
| --- | --- |
| endpoints compared | 375 |
| `gm_ratio` **bit-identical** | **370** |
| changed at all | 5, by ~1e-6 |

So the intensity measures barely moved. That is not a property of the gates being lax —
it is a property of what they measure. PGA, PGV, `Ds` and pSA are peaks, integrals and
response-spectrum ordinates, and a perturbation of ~1e-5 in the waveform samples does not
survive into any of them. Tier B reporting exact unity through five numerical changes is
the same fact seen from a different angle.

**The consequence for the workflow.** For rounding-scale changes, the sensitive
instrument is `run_selfparity.sh` — a direct waveform comparison that resolves individual
ulps — and the statistical tiers are *acceptance criteria*, not detectors. Do not read a
green Tier C as evidence that a change was small; read it as evidence that whatever the
change was, it does not move the quantities engineering cares about. Those are different
claims and only the second is supported.

Where the tiers do earn their keep is the failure mode self-parity cannot see: a
**systematic** drift accumulated across many individually-small commits. The number to
watch there is the pooled bias, which after all of Stage 2 is **-0.003%** against an
endpoint-to-endpoint sd of 0.575%. That is the statement worth having.

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

### 1.4b Names: no Fortran abbreviations anywhere in the code

The port inherited its identifiers from the Fortran, so the public API is still
spelled `fast`, `flzero`, `cr`, `dgamm`, `stoc_f`, `rdatn`, `betvs`, `akapp`.
**That convention is the reason the original was unreadable**, so carrying it into
the Rust is a defect rather than fidelity. Resembling the Fortran is explicitly not
a reason to keep a name.

Three rules:

1. No cryptic contractions — `flzero` becomes `remove_quadratic_trend`.
2. Minimum three characters, except coefficients and real coordinates (`x`, `y`, `z`).
3. Tag units where it helps, especially on `pub` arguments. `dt` is exempt; its
   interpretation is unambiguous.

The Fortran name and its `hb_high_ref.f:NNN` line stay in the **doc comment**. That
is documentation, not code, and it is what lets a reader trace a routine back to the
oracle — which the parity harness depends on.

Renames change no arithmetic, so tier A gates the whole pass for free.

| now | becomes |
| --- | --- |
| `fft::flzero(n, dt, a)` | `remove_quadratic_trend(count, dt, acceleration)` |
| `fort::nint` / `int_trunc` | `round_half_away_from_zero` / `truncate_toward_zero` |
| `geom::Delaz5` / `delaz5` | `DistanceAzimuth` / `distance_azimuth(event_lat_deg, …)` |
| `geom::even_dist2` | `subfault_geometry(fault_lon_deg, fault_lat_deg, …)` |
| `highcor::highcor_f` | `apply_radiation_and_invert(fold_count, mirror_count, …)` |
| `radiation::rdatn` | `radiation_pattern(strike_rad, dip_rad, rake_rad, azimuth_rad, takeoff_rad)` |
| `radiation::radfrq_lin` | `horizontal_radiation_spectrum` |
| `radiation::radv_lin` | `vertical_radiation_spectrum` |
| `ray::cr(p, v)` | `vertical_slowness(ray_parameter, velocity_km_s)` |
| `ray::trav` | `build_ray_path` |
| `ray::geom_terms` | `geometric_spreading` |
| `ray::cagcon` / `dtdp` | `cagniard_time` / `cagniard_time_derivative` |
| `ray::pnot` | `stationary_ray_parameter` |
| `ray::ttime` | `travel_time` |
| `ray::GfAmp` / `gf_amp_tt` | `GreenFunction` / `green_function` |
| `rng::ranu2` | `uniform_deviates` |
| `rng::normal_random_number` | `normal_deviates` |
| `Pcg32::rand_numb` | `next_f32` |
| `site::get_sitefacs` | `site_amplification_factors` |
| `site::siteamp` | `apply_site_amplification` |
| `special::dgamm` / `DGAMM_ERROR` | `gamma` / `GAMMA_ERROR` |
| `stoc::stoc_f` | `stochastic_spectrum` |
| `state::Vmod` / `VmodIn` / `Coff` | `VelocityModel` / `VelocityModelInput` / `Coefficients` |

**`fft.rs` is deliberately excluded**, along with `np2` wherever it appears. §2.1
replaces the whole radix-2 implementation with `realfft`, at which point `fast`,
`flzero`'s caller, `ind`, `nfold` and `mfold` either disappear or are renamed by
whatever the library calls them. Renaming them now would be work done twice.
`flzero` itself is in the table because it is a detrend, not a transform, and
survives the swap.

Velocity-model fields gain units: `thic` → `thickness_km`, `depth` → `depth_km`,
`vp` → `vp_km_s`, `vsh` → `vsh_km_s`, `rho` → `density_g_cm3`. The two quality
factors become `attenuation_p` / `attenuation_s` rather than `qp` / `qs`, which are
both under three characters and read as typos next to each other.

`stoc_f`'s nineteen arguments are the worst offenders and get the most from units:
`r` → `distance_km`, `tw` → `window_s`, `betvs` → `shear_velocity_km_s`, `row` →
`density_g_cm3`, `smt` → `subevent_moment`, `fc` → `corner_frequency_hz`, `akapp` →
`kappa_s`, `qfe` → `q_exponent`, `bigc` → `moment_scale`.

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

### 2.3 `fort::Array1`/`Array2` → 0-based storage

`Array1` and `Array2` are ~105 lines of `fort.rs`: 1-based indexing, column-major 2-D,
and bounds assertions. They do look a lot like a hand-rolled `ndarray`, and that is
worth taking seriously rather than reflexively reaching for slices.

**Split the decision by dimensionality**, because the two have different answers:

- **`Array1` → plain `Vec`/slices.** `ndarray` adds nothing to one dimension that a
  slice does not already have, and it would put a bounds-checked wrapper straight back.
  This is also where the churn is: 45 references in `state.rs` alone.
- **`Array2` → also plain storage. Decided.** `ndarray` was the obvious candidate and
  is a reasonable one: there are only ~eleven sites, and column views and `Zip` fit
  them. But it pulls in `matrixmultiply`, `rawpointer`, `num-complex` and `num-traits`
  for what amounts to eleven call sites of 2-D indexing, where `Vec<f32>` plus a
  five-line `index(i, j)` helper does the job. **Slices, for now** — revisit only if a
  later pass wants `Zip`/`azip!` across the spectral loops badly enough to pay for the
  subtree.

> **§2.3's benefit is specific to large hot buffers, and a half-measure gets neither
> half.** Two conversions, measured:
>
> | module | form | instructions |
> | --- | --- | --- |
> | `rng.rs` | 0-based slices, two 262144-element passes | **−5.13%** |
> | `state.rs` | `Vec` indexed 1-based, element 0 unused | **+0.62%**, reverted |
>
> The `rng.rs` win is vectorisation of two long renormalisation passes. `state.rs`'s
> arrays are indexed in short loops over the ~35 *real* layers of a velocity model —
> `NLAYMAX = 500` is a ceiling, not a count — so there is nothing to vectorise, and
> `Array1`'s `assert(i>=1)` then `data[i-1]` evidently folds better than a raw `Vec`
> bounds check on `[i]`.
>
> The clarity argument also runs the wrong way for the half-measure. `Array1` as a *type*
> documents and enforces the 1-based convention; a raw `Vec` indexed 1-based with element
> 0 unused reads as 0-based to anyone who has not read the module header. So it cost
> performance *and* legibility, and was reverted.
>
> **Revised plan for the remaining modules.** Convert only where the loops are long
> enough to vectorise, and convert them **properly to 0-based**, never to a 1-based
> `Vec`. That points at `highcor.rs` and `stoc.rs`, whose loops run over `np2`-sized
> spectra. `state.rs` is a different job: going 0-based there means editing every index
> expression in every kernel that reads a velocity model, `ray.rs` most of all, for no
> measured performance gain. **That needs sign-off, not an unattended pass** — the
> off-by-one risk is real (one was already caught in `rng.rs`) and the payoff is
> readability alone.

> **`Array2` is gone, and the replacement is 1-based on purpose — `d6aa8f3`.** This
> section said to convert "**properly to 0-based**, never to a 1-based `Vec`". That rule
> is right for storage offsets and wrong for the eight subfault grids, and the
> distinction is worth stating because it also decides `state.rs`.
>
> `geom`'s five arrays and `Segment`'s three were never grids: every read of one was at
> the same `(i, j)` as its siblings, so they are **one value per subfault** and became
> `Vec<SubfaultRay>` and `Vec<Subfault>`. That deleted `fort::Array2` outright (`fort.rs`
> 211 → 152 lines) and took `window_s`'s `(NQ, NP)` = 600×100 allocation with it.
>
> The accessors `SubfaultGeometry::at` and `Segment::at` stay **1-based**, because `(i,
> j)` is a subfault *number* rather than a storage offset — the along-strike coordinate of
> subfault `i` is `(i - 0.5) * length`, so the numbering is part of the physics. Going
> 0-based would put a `+ 1` into every such formula. What the change *did* achieve is
> confining the 1-based-ness to two `at()` methods instead of five `Index` impls, and
> making the layout private.
>
> Measured: **−0.01%**, i.e. nothing, exactly as predicted for short branchy loops. The
> payoff was size and readability, and that is the honest accounting.
>
> The order-dependence that had to be preserved was **not** an index question:
> `normalise_source`'s two accumulations sum in depth-major order and floating-point
> addition is not associative. Storing the strike index fastest makes depth-major *be*
> sequential, so both passes now walk a slice and compute no index at all —
> `Segment::depth_rows` names that and its doc says why it is load-bearing.

> **§2.3 has a performance benefit after all, measured.** This section and `PROFILE.md`
> both said it was size and readability only. Converting `rng.rs` alone cut instructions
> retired by **5.13%** for the whole program. The wrapper's `Index` impl is
> bounds-checked and `#[track_caller]`, which blocks vectorisation of the two
> renormalisation passes over `mmv = 262144` elements in `fill_normal_deviates`. The
> `fft.rs` conversion just before it bought exactly zero, so the effect is specific to
> hot loops over large buffers — which is where the remaining modules' loops are too.

**Do `Array2` first, and only after §2.6's defect 1.** The column-major layout is
load-bearing today for exactly one reason: `stdd(0,l)` aliases across columns
(`PORTING_RULES.md` §7). Fixing that defect removes the only observable dependence on
layout, after which the 2-D storage can be whatever `ndarray` defaults to and the
migration stops being delicate. That is a reason to move defect 1 *earlier* in the
sequence, not later.

#### The original framing, still true

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

### 2.4c REJECTED: `sin` from `sqrt(1 - cos^2)` in Box-Muller

`fill_normal_deviates` is the second-hottest routine in the program (15.1% self time)
and needs both the cosine and the sine of one angle per pair. Replacing the `sinf` with
`+/- sqrt(1 - cos^2)` — a sqrt, a multiply and a compare — measured **-2.46% of total
instructions retired**, and it is wrong.

**Why it fails, and why the verification missed it.** The identity was checked over
200,000 angles and gave a worst absolute error of `2.4e-12`. That check was run in
`f64`; the code runs in `f32`, where the same identity gives a worst absolute error of
**`2.4e-4`** — eight orders of magnitude worse. The tier-0 golden caught it at a
deviate of `0.00047` against the Fortran's `0.00038`.

The cause is conditioning, not cancellation that a wider accumulator could fix:

```
d(sin)/d(cos) = -cos/sin
```

Near `|cos| = 1` the sine is tiny, so an `f32`-accurate cosine (error ~`6e-8`) implies a
sine error of `6e-8 / sin`. At `sin = 1e-3` that is `6e-5`. **Widening the arithmetic to
`f64` does not help** — the input cosine is only `f32`-accurate and that is the limiting
term. There is no cheap repair.

**The tempting wrong move was to loosen the golden's tolerance to `1e-4` and move on.**
That would have hidden a genuine accuracy regression concentrated exactly on the
near-zero deviates, two orders of magnitude worse than the ~`1e-6` every other Stage 2
change has produced. Reverted instead.

Rust's `sin_cos` is not an alternative: it is literally `(self.sin(), self.cos())`, so
it saves nothing.

**Lesson worth keeping: verify a float identity at the precision the code uses.** A
`f64` check of an `f32` computation is not a check.

### 2.5 `geom::delaz5` → a geodesy crate — **DONE**

~100 lines of a 1970s distance/azimuth formulation with three separation regimes,
one of which (`geocentric radians`) is unreachable because `even_dist2` always
passes 0. A modern geodesic (`geographiclib-rs`) is both smaller at the call site
and more accurate. Tier C, since it moves distances by metres and distance feeds
the path-duration branch selection.

> **Done. The first change in Stage 2 that moves the waveform at all.**
>
> `distance_azimuth` is now `geographiclib_rs`'s inverse geodesic on WGS84. Gone with it:
> the three separation regimes, the `0.9931177` tangent-scaling stand-in for the ellipsoid,
> the `6371.0` km mean-radius sphere, the sixteen `DOUBLE PRECISION` cosines feeding
> `real*4` trig that `PORTING_RULES.md` §2 used as its worked example, the dead
> `coord_mode` argument, and **four of the seven outputs** — `delt`, `deltdg`, `azse` and
> `azsedg` had no reader outside the tests that checked them.
>
> Measured, on this fixture's real fault geometries and on Canterbury separations of
> 4–409 km:
>
> | | worst |
> | --- | --- |
> | distance | **0.12%** on the fault fixtures, 0.32% at 1 km separation (3 m) |
> | azimuth, under 500 km | **0.045°** |
> | take-off angle | **0.029°** |
> | Tier B pooled bias | **+0.006%**, endpoint sd 0.121% |
>
> `DELAZ5` is systematically **short**, consistently signed — the signature of the
> mean-radius sphere plus the tangent trick, against a true geodesic. The new values are
> the correct ones.
>
> **Self-parity cannot judge this change**, and that is not a failure of the change. A
> distance shift moves arrival times, which *translates* the waveform; self-parity compares
> sample `i` to sample `i`, so it reports 22 of 22 decks differing at any tolerance up to
> 10%. This is the same instrument limitation recorded above for §2.6's one-sample shift.
> Tier B and Tier C are the instruments here.
>
> **Tier B woke up.** It had returned exactly `+0.000%` with zero endpoint spread through
> every previous Stage 2 change, because they were all bit-identical in effect. It now
> reads `+0.006%` with a 0.121% endpoint sd and still certifies 375/375 — which is what
> this section's own description of Tier B promised would happen the moment numerics moved.
>
> **Two goldens changed character rather than being deleted**, and both are more useful for
> it: `delaz5.bin` and `even_dist2.bin` now bound the *size* of the deliberate change
> instead of asserting identity. The `delaz5` bound is **stratified by separation**, because
> the two formulations disagree by 0.045° under 500 km and by **45°** at 20,015 km — where a
> geodesic's azimuth is genuinely ill-conditioned, the endpoints being nearly antipodal.
> A single global tolerance would have had to be 45° and would have said nothing.
> `subfault_geometry`'s `depth_km` is still asserted **bit-exact**, since it comes from the
> dip and the down-dip index and never touches the geodesy — if it moves, something other
> than §2.5 did it.
>
> One property test was wrong and was replaced, not loosened: it asserted the azimuths at
> the two ends of a path differ by 180°, which is a *sphere's* property. On an ellipsoid a
> geodesic's azimuth changes along its length, and over the 13,000 km separations the
> generator was producing, the two ends disagreed by 129° entirely correctly. It now checks
> the azimuth against a flat-Earth bearing for *nearby* stations, which is what actually
> pins our use of the library: the right slot out of a return tuple whose element meanings
> change with its width, degrees not radians, the `[0, 360)` wrap, and the `f32` narrowing.
>
> **A trap worth recording about `geographiclib_rs`.** `InverseGeodesic` is generic over
> the output tuple, and the width changes what the *earlier* slots mean:
>
> ```text
> let x: f64                     = geod.inverse(..);  // s12
> let x: (f64, f64, f64)         = geod.inverse(..);  // (azi1, azi2, a12)   <-- no s12
> let x: (f64, f64, f64, f64)    = geod.inverse(..);  // (s12, azi1, azi2, a12)
> ```
>
> The three-element form has no distance in it at all. Destructuring it as
> `(s12, azi1, azi2)` compiles, runs, and yields an azimuth where a distance is expected —
> which is exactly what happened on the first attempt here, producing negative "distances"
> of a few hundred metres and a nonsense 415 km worst-case error. The type annotation at
> the call site is load-bearing and is commented as such.
>
> Size: `geom.rs` 302 → 267 lines. Instructions +0.013%, i.e. free — the geodesic is a
> longer calculation than DELAZ5 but runs `2 + nx*nw` times per segment, not per sample,
> and `Geodesic::wgs84()` is built once in a `OnceLock` rather than per call.

### 2.6 Defects: both FIXED — `fff2abf` and `262c75f`

> **Done.** Defect 2 (`siteamp` conventions) in `fff2abf`, measured delta 1e-5 of
> waveform peak, isolated to the site path — the one deck unaffected is `siteamp=0`.
> Defect 1 (`stdd(0,l)`) in `262c75f`, verified as a pure one-sample translation: mean
> difference is exactly zero at +1 sample of alignment and non-zero at every other
> offset. Tier B certified 375/375 on both, which for defect 1 is expected rather than
> reassuring — see below.
>
> One correction to record: defect 2's severity was estimated as negligible from the
> kappa attenuation at Nyquist, and the measured delta is 1e-5 of peak rather than the
> ~1e-7 that implied. The estimate looked at `a2` alone and ignored the rest of the
> spectral shaping at that frequency.



Two genuine defects in the original, both faithfully reproduced by the port.

**Decided: fix both.** They are deferred to Stage 2 rather than fixed now for one
reason — fixing either changes output, which would confound the bit-reproducibility
gate that everything in Stage 1 is verified against. Stage 1 needs `cmp` to mean
something; a deliberate output change and an accidental one look identical to it.

So the disposition in `PORTING_RULES.md` §7 — *reproduce, do not fix* — holds through
Stage 1 and is overridden here. These are scheduled work, not open questions.

#### Defect 1 — `stdd(0, l)`: every subfault's contribution is delayed one sample

| | |
| --- | --- |
| Fortran | `hb_high_ref.f:1394-1396` |
| Port | `sim.rs::subfault_acc_at`, which models the read explicitly |
| Register | `PORTING_RULES.md` §7, first row |

The accumulation loop runs `li = k2, kend` and reads `stdd(li-k2, l)`, so its first
iteration reads **index 0** — one element before the column. `apply_radiation_and_invert`
fills only `1..=np2`, so nothing ever writes there.

`stdd` is declared `stdd(mmv,3)` column-major, so `stdd(0,2)` aliases `stdd(mmv,1)`
and `stdd(0,3)` aliases `stdd(mmv,2)`. Both are untouched — the zeroing loop covers
only `1..np2` — and live in `.bss`, hence read as zero. `stdd(0,1)` is genuinely
before the array. The port models all three as zero.

**Effect.** `DS(l,k2)` receives nothing and `DS(l,k2+1)` receives `stdd(1)`, so every
subfault's contribution is shifted one sample later. Because the shift is *uniform*
across subfaults, the result is a whole-trace time offset of one sample — 5 ms at the
production `dt = 0.005` — plus the loss of each contribution's final sample where
`kend` clamps.

**Why the severity is not obvious.** No intensity measure notices: PGA, PGV, duration
and pSA are all invariant under a 5 ms translation. So on the Phase 2 gates this is
invisible, and Tier C would report it as equivalent.

Where it *does* matter is absolute timing. The workflow **sums the high-frequency
synthetic with a low-frequency one**, and a 5 ms offset applied to only one of the two
is a real phase error at high frequency — half a cycle at 100 Hz, a tenth of a cycle
at 10 Hz. That is the question to answer before deciding, and it is answered by
looking at how the two are combined downstream, not by anything in this repository.

**Fix, in its own commit, with the Tier C delta recorded.** Expect Tier C to show no
change, and say so explicitly rather than treating the null result as confirmation the
fix was unnecessary — the gates are blind to this defect by construction, which is
exactly why it needs the downstream-summation argument above rather than a green
dashboard.

Do it *after* §2.1: the FFT swap will move the spectrum slightly, and it is easier to
read one deliberate change at a time than two superimposed.

#### Defect 2 — site amplification applies two different conventions

| | |
| --- | --- |
| Fortran | `siteamp`, `hb_high_ref.f` |
| Port | `site.rs::apply_site_amplification` |
| Found by | writing `tests/properties.rs`; pinned there by `dc_and_nyquist_are_scaled_linearly_not_exponentially` |

The factor table holds **log** amplitudes: every interior bin is scaled by
`exp(interpolated factor)`. But the two end bins are scaled by the factor
**directly**:

```rust
spectrum[1] = spectrum[1] * factors[1];                  // DC      -- linear
...
let fac = (am + (freq - fm) * (ap - am) / (fp - fm)).exp();
spectrum[i] = spectrum[i] * fac;                         // interior -- exponential
...
spectrum[nf] = spectrum[nf] * factors[table_count];      // Nyquist -- linear
```

A table of log-amplitude 0.5 therefore amplifies the interior by `e^0.5 = 1.65` while
*attenuating* the two end bins to 0.5 — they disagree by a factor of 3.3.

**Severity, measured rather than assumed.** An earlier note in this plan called this
"almost certainly a defect" and left the impact unquantified. Quantified, it is much
smaller than the mechanism suggests:

- **The DC half is inert.** `stochastic_spectrum` sets `as_[1] = 0.0`, so the DC bin
  is identically zero on entry. Multiplying zero by the wrong factor is still zero.
- **The Nyquist half is live** — `spectrum[fold_count]` is written with a non-zero
  value — but Nyquist sits at `1/(2*dt) = 100 Hz` in production, where the kappa
  filter `exp(-pi*f*kappa)` with `kappa = 0.045` has already attenuated the spectrum
  by `exp(-14.1) ~ 7e-7`. One bin in 8193 at `np2 = 16384`, six orders of magnitude
  below the passband.

So this is a real inconsistency with negligible numerical consequence. It is worth
fixing because it is *confusing* — two conventions in one routine, with nothing
saying so — not because it moves any waveform.

**Fix**, and delete the property test that pins it. That test asserts the two
conventions visibly disagree, so it fails the moment they are reconciled — which is
the intended trigger, not a regression.

Reconcile *towards the exponential*: the interior convention is the one the table is
built for (`site_amplification_factors` produces log amplitudes) and the one 8191 of
8193 bins already use. Changing the interior to match the ends would be the wrong
direction and would move every waveform.

### 2.6b Size the buffers from the deck, not from a compile-time ceiling

Every large array is currently allocated at `mmv = 262144`, the `params_no_window.h`
constant, regardless of what the deck asks for. That is both a waste and a **capability
limit**: the Fortran aborts with "need to recompile with larger array size" once
`np2 > mm`, so a long enough waveform cannot be run at all without rebuilding. The port
reproduces that abort (`SimError::TransformTooLong`).

Both problems have the same fix — compute the sizes.

What is oversized today, per `simulate` call:

| buffer | allocated | actually needed |
| --- | --- | --- |
| `acc` | `3 × 262144` f32 = 3.0 MB | `3 × ndata` |
| `normal_deviates` | 262144 f32 = 1.0 MB | `np2`, not `mmv` — see below |
| `radv_rand_a`, `radv_rand_b` | 262144 f32 each = 2.0 MB | `nr` = 1000, so **0.4%** of what is reserved |
| `freq`, `radiation` | `mm` = 262144 f32 each | `np2/2 + 1` and `np2` |

At the production `duration = 20`, `dt = 0.005` that is `ndata = 4000` and
`np2 = 16384`, so roughly **7 MB of address space reserved against a few hundred
kilobytes used**.

> **Correction, measured.** An earlier version of this section claimed the pages are
> touched because the arrays are zero-initialised. That is wrong. `Array1::new` is
> `vec![0.0; n]`, which for a large `n` goes through `alloc_zeroed` → `calloc` → an
> `mmap` of zero pages; those are copy-on-write from a shared zero page and never become
> resident until written. Nothing ever wrote to the unused tails, so they were never
> resident. Sizing the buffers correctly moved **peak RSS from 6700 KB to 6584 KB —
> 116 KB, not 7 MB** — and instructions by −0.11%.
>
> So the memory and performance case for this item is nearly nil. **The capability case
> is the whole justification**: removing the recompile-to-go-longer ceiling. That is
> reason enough, but it should not be sold as a memory saving.

Two traps to respect while doing it:

1. **`normal_deviates` is drawn at `mmv`, not `np2`**, and the draw *count* is part of
   the RNG stream — `fill_normal_deviates(rng, MMV, …)` consumes 262144 deviates
   whether or not they are read. Shrinking the allocation is safe; shrinking the
   **draw** changes every waveform. Those are separate decisions and only the first is
   free. See `PORTING_RULES.md` §5.
2. **`np2` is derived, not given.** It comes from `tmax`, which comes from the
   time-window pass over every subfault, so it is not known until after that pass. So
   either allocate the `np2`-sized buffers after it (they are per-segment already), or
   compute an upper bound first.

Once sizes are computed, **delete the `np2 > mm` abort entirely** rather than raising
the ceiling: there is no fixed ceiling left to exceed, and a tool that refuses long
records for a reason the user cannot act on is worse than a slow one. That is a
behaviour improvement over the Fortran rather than a port of it, so it belongs here
with a note in `PROVENANCE.md`.

Ordering: after §2.6's defect fixes (which settle the `stdd` layout question) and
alongside or before §2.3, since both touch the same allocations.

> **§2.3 IS DONE — `Array1` and `Array2` are both deleted.** `fort.rs` is **82 lines**,
> down from 211 at the start of Stage 2 and ~230 at the start of the port. What survives is
> the two rounding intrinsics, which encode genuine Fortran-vs-Rust semantic differences
> rather than transliteration scaffolding.
>
> The velocity-model conversion — the item this section flagged as needing sign-off, which
> Jake gave — measured **−0.014%**, not the ~1% cost predicted here. The prediction came
> from the reverted half-measure (`63eb218`, +0.62%), and the difference is the whole point
> of that revert: a `Vec` indexed 1-based with element 0 unused pays for both conventions
> and gets neither. A proper conversion is free.
>
> Also gone: the `array/indexed_sum` / `array/slice_sum` benchmark pair, which is the one
> that produced `PROFILE.md`'s wrong "the wrapper is free" conclusion by measuring a loop
> that failed to vectorise.
>
> **Final tally of the off-by-one risk this section kept warning about: three, in twelve
> commits.** One caught by `cargo test` with self-parity blind, one by self-parity with
> `cargo test` blind, one by reading alone. Both gates were necessary and neither was
> sufficient — which is the argument for running both on every commit, not for avoiding
> the work.

> **§2.6b is DONE — `504d354`.** No buffer in the program is sized by a compile-time
> constant any more, so `params::NQ`, `NP`, `LV` and `MM` were deleted outright rather
> than left as dead ceilings. Two buffers turned out to be sized by the *wrong* ceiling:
> `siteamp_log_freq` and `siteamp_factors` are frequency tables indexed `0..nsfac = 20`
> but were allocated at `NLAYMAX = 500`, because in the Fortran they shared a common
> block with the velocity model. That is a storage accident, not a bound.
>
> `MMV` survives on purpose, and not as a size: it is the *number of deviates drawn* per
> station, which is part of the RNG stream. Trap 1 above is the whole reason, and it is
> now in the constant's own doc comment where the next reader will hit it. Changing it is
> a §2.7 decision.
>
> The capability goal is met: there is no record length that requires a rebuild.

### 2.8 The de-Fortran-ification pass — **DONE**, and it lands *before* §2.7

Numbered after §2.7 because it was planned later; sequenced before it deliberately.
§2.7's whole risk is that if the stream shifts, every golden, CSV and Tier D attribution
moves at once. Doing the structural work first means anything that moves when §2.7 lands
is unambiguously the RNG rather than a refactor riding along.

Nine commits, `7397c36..3745bb2`. One NUMERIC, the rest bit-exact and gated per commit by
`run_selfparity.sh HEAD~1`.

**The pi commit is the only numeric one.** Nine literals — the Fortran's own truncations,
carried faithfully while bit-identity was the contract — became `std::consts`. Eight were
honest truncations; the ninth, `highcor`'s taper `3.14159625`, was a *typo*, the digits of
`3.14159265` transposed, 1.1e-6 relative and about thirty times worse than the rest. Its
doc comment had said correcting it needed "a written justification, not a quiet cleanup";
that justification is now in the comment.

Three exact goldens became measured-divergence bounds, following the precedent §2.5 set
for `delaz5` — measure the move, bound it by an argument, keep the test sensitive — rather
than regenerating against an oracle that is itself wrong. `vertical_slowness` is where it
bites: on the branch cut the phase is forced to exactly pi, so `cos(phi/2)` should be an
exact zero, and the Fortran's truncation put 5.7e-11 there instead. **That error was the
golden.** Measured: `cr` 2.051e-10 of |eta| against an analytic bound of 2.05e-10 —
agreement to three digits, and 1050 of 1500 cases still bit-exact; `cagcon` 1.082e-10;
`dtdp` 7.012e-10, which amplifies eta's error ~3.4x because it divides *by* eta.

**The campaign, run once at the end of the pass** (`run_science.sh`, all three tiers):

| tier | result | vs. the last recorded run |
| --- | --- | --- |
| B (paired, n=50) | 375 certified, **0 REFUTED**, pooled bias +0.006% | — |
| C (distributional, n=2500) | 324 certified, **0 REFUTED**, 51 undetermined | **identical verdict counts** |
| D (inter-frequency, n=600) | 4 of 15 flagged at raw p, **0 after Holm** | **identical**, P(≥4) = 0.0055 |

The Tier C split is the same 324/0/51 recorded after §2.5, and Tier D's enrichment is the
same 4-of-15 at the same P(≥4) = 0.0055 — with the same four tests, all on 090/000 and
none on `ver`. That is the pattern the RNG attribution predicts, since the horizontals
draw from the live stream and the vertical reads pre-drawn uniforms, and it is unchanged
by a pass that rewrote the code around it. The 51 undetermined are a sample-size limit,
not a failure: worst achieved resolution is ±1.42% against a ±2% band, and certifying them
all would need n ≈ 5074 per stratum.

The one thing that *did* move is the pooled Tier C bias, from **-0.003% to +0.025%**. That
is the pi correction, and it is the only trace of it anywhere above the waveform level —
0.028 percentage points against an endpoint-to-endpoint sd of 0.581%. Tier B, which is
paired at matched seeds and far more sensitive, puts the whole pass at +0.006% with a
worst endpoint ratio of 0.99250.

**What the rest of the pass did.** Clippy did not compile before this — seven deny-level
`approx_constant` errors on those same literals aborted the run before any lint reported,
so 28 warnings had accumulated invisibly, eight of them `needless_range_loop`. A lint
nobody can run is worse than no lint, because it reads like coverage. Now zero, with
`[workspace.lints]` as a floor.

Then: index loops became iterators across seven files; the velocity model became
`Vec<Layer>` (the last parallel-array holdout, 158 access sites); the path-duration table
became `Vec<DurationSegment>`, dropping a 50-element ceiling and an `ndur` that could
disagree with it; `nm`/`it`/`nup` became `WaveMode`/`Interaction`/`Direction` and the
three components became a `Component` enum, deleting two `panic!` arms that existed only
because the operands were integers; six near-copy golden `Reader`s became one.

**The find that mattered most.** `sim.rs`'s `nsum` block looks entirely dead — the loop
runs once, the sub-event offset it computes is unconditionally zeroed two lines later, and
`rise` and the `NINT` shim feed nothing. But `let si = rng.next_f32()` advances the shared
generator once per (subfault, ray). Deleting it as obvious dead code would have moved
every waveform in the program. The arithmetic is gone; the draw stays, named and
explained. This is the sharpest example yet of the §2.6b lesson: in this program, *the
number of draws is data*.

Also worth recording: `Rays::ndeg` looked equally dead — sole writer sets 1, sole reader
tests `< 0` — and is not. `tier1_driver.f` feeds `ndeg = -1` for case `kc == 6`
specifically to exercise the branch that forces `nup = +1`. Verified before deleting
rather than after.

**Size.** `crates/hb_high/src` went 4,899 → 5,237, up 7%. That is the same honest outcome
Stage 1 had and for the same reason: structs, enums and their doc comments cost more lines
than the inlining saves. The tests went the other way — 492 lines deleted for 83 added
plus a 236-line shared module. What improved is not line count: `nm == 3 .or. nm == 4`
appears zero times instead of four, the two `panic!("unreachable")` arms are gone, and the
velocity model can gain a field without seven edit sites and no compiler help.

**What §2.8 did NOT do**, and is still open: `green_function`'s four copy-pasted
descending loops and three Moho scans; the `simulate()` decomposition (still ~500 lines —
`REFACTOR.md` §1.3 named `time_window_pass`/`subfault_pass` specifically and they are
still not extracted); the deck's `read_values` + positional-closure pattern at 13 call
sites; `GeoPoint` for the lat-first/lon-first mismatch between `distance_azimuth` and
`subfault_geometry`; and the two deliberately-last index sites (`ksrc` coming back one
past the model, and the sample accumulate where §2.6 defect 1 lived).

### 2.7 `rng.rs` → `rand` / `rand_pcg` — LAST, and carefully

Queued deliberately at the very end of Stage 2, because it is the one replacement that
can invalidate the harness that validates everything else.

**The good news.** `Pcg32::next_u32` is the standard PCG32 XSH-RR variant — multiplier
`6364136223846793005`, increment `1442695040888963407`, `((old >> 18) ^ old) >> 27`
rotated right by `old >> 59`. That is exactly `rand_pcg::Pcg32` (`Lcg64Xsh32`), so the
*output function* should match bit for bit given the same state.

**The risk, which is why this goes last.** Three things around that core are custom and
load-bearing:

1. **Seeding.** `init_random_seed` folds `irand, irand+1, …` through `SEED_WORDS = 8`
   rounds of `state = state*MULT + irand`, then discards two draws — and it *mutates*
   `irand`, whose final value gates the rupture-time jitter at `:1366`. `rand_pcg`'s
   `Pcg32::new(state, stream)` does its own initialisation and does not expose a raw
   state setter, so matching the stream means constructing state by hand and verifying
   it, not calling a constructor.
2. **`next_f32` takes the top 24 bits** and divides by `2^24`, deliberately: dividing a
   full 32-bit value by `2^32` rounds, and values near 1 round *up* to exactly 1.0,
   breaking the `[0,1)` contract the zero-rejection loops depend on. `rand`'s standard
   float conversion is not necessarily this one.
3. **`fill_normal_deviates` renormalises to unit RMS**, and
   `stochastic_spectrum`'s amplitude calibration is tuned against that. `rand_distr`'s
   `Normal` will not do it, so this routine stays regardless — it is algorithm, not
   generator.

**If the stream does not match exactly, every golden, every recorded CSV, and the
Tier D attribution all move at once** — and Tier D's finding is specifically *about*
the RNG, so disturbing it while that lead is open would destroy the evidence.

**How to do it safely:** before changing anything, write a test that seeds both
generators and asserts the first few thousand `u32` draws are identical. If that test
cannot be made to pass, stop — the win is roughly 25 lines and it is not worth an
unexplained shift in every number. Run Tier D before and after and compare, not just
Tier B.

Honest assessment: `rand` would be the obvious choice writing this fresh. Here it
replaces ~25 lines of standard, property-tested code whose stream is baked into every
fixture. Do it for the dependency hygiene, not for the line count, and only with the
draw-for-draw test in place.

#### Not defects: frozen switches

`nsum` is computed from `ratio` and then forced to 1 (dated 2004-04-20), which is why
the Frankel operator in `stochastic_spectrum` carries the scaling instead.
`islip_weight_avg` is set to 1 and then 0. These are **not** bugs — they are switches
someone deliberately froze. Collapsing them removes real code but bakes in a decision
that was left adjustable, so it needs the same explicit sign-off, and it is listed
here so it does not get done by accident during a tidy.

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
- **Don't touch `rng.rs` until everything else is done** — see §2.7. The Tier D result
  makes it the one numerically interesting module, and "small" is not a reason to
  disturb a generator whose stream is baked into every golden. Replacing it is queued
  as the final step, gated on a draw-for-draw equality test.
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
5. `HfConfig` + `simulate()` + deck shim (1.1) — **done**, `bdf3ce7` (typed config)
   and `3ed9469` (`hb_high::sim::simulate`; `main.rs` 1003 → 341).

**Stage 1 is complete.** Where the code went:

| | before | after |
| --- | --- | --- |
| `main.rs` | 865 | 341 |
| `sim.rs` | — | 796 |
| `config.rs` | — | 356 |
| `crates/hb_high/src` total | 4,336 | ~4,900 |

The crate got **bigger**, by about 13%. That is the honest outcome of Stage 1 and was
the intended trade: structs, enums and doc comments cost more lines than the
inlining saved, and what improved is that `run()` is now a driver you can read in
one sitting, the physics is a library function, and the deck's hazards are described
in one place instead of being latent. The reduction is Stage 2's job, and §2.1 alone
deletes more than Stage 1 added.

One deliberate regression in capability: **`nsite > 1` is refused.** The Fortran's
station loop shares a generator, so stations are not independent and a
one-station-per-call API cannot reproduce a multi-station deck. Production has always
run one process per station, and no parity deck uses anything else.

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
11. The two defect fixes (2.6), separately, each with its Tier C delta recorded:
    the `siteamp` convention split, then the `stdd(0,l)` sample shift. The second of
    these **unblocks** step 12 — see §2.3.
12. `Array1`/`Array2` (2.3) → plain slices, module by module, after step 11 removes
    the layout constraint.
13. Size buffers from the deck (2.6b) and delete the recompile-to-go-longer ceiling.
14. `rng.rs` → `rand_pcg` (2.7). **Very last**, gated on a draw-for-draw equality test
    against the current generator, and with Tier D run before and after.
13. A second, smaller 1.3b pass, now that slices make `zip`/`chunks_mut` available.

Re-run Tier D at the end of Stage 2 as a release gate, comparing Rust-before vs
Rust-after rather than against production.
