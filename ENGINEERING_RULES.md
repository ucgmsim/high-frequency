# Engineering rules

`PORTING_RULES.md` governed the transliteration and is now archaeology: it explains why
`hb_high_ref.f` does what it does, which is still the fastest way to read the original, but it
stopped describing this crate. **This document governs the crate.**

The change in goal is the point. Stages 1–2 optimised for demonstrable fidelity to a
Fortran original. Stage 3 optimises for code someone can read, change and trust, and for
it to be fast.

---

## 1. Fewer lines

Report the net line delta in every substage. Expect it negative.

**The honest tension, stated up front:** Stage 1 grew the crate 13% and §2.8 grew it a
further 15%, and `REFACTOR.md` records both as "the honest outcome" — structs, enums and
doc comments cost more lines than inlined Fortran saved. That was true, and that cost is
now largely paid. Further growth is a smell.

But this is a **signal, not a gate**. A commit that adds twenty lines to delete a hazard
is a good commit. A commit that adds two hundred lines of abstraction nobody asked for is
not, and the delta is how you notice the difference.

## 2. No Fortranisms

Specifically, and these are all things that were in this crate and are not now:

- integers standing in for enums,
- parallel arrays where one array of structs belongs,
- index loops where an iterator fits,
- flag arguments — a parameter whose only job is to switch a function between behaviours,
- out-parameters,
- buffers sized by a compile-time ceiling rather than by the data.

## 3. Clean Rust

**Clippy at zero is a gate**, not an aspiration. It has been at zero since §2.8 and
`[workspace.lints]` keeps it there.

Extend the conventions already in force rather than inventing new ones:

- No cryptic contractions. `flzero` became `remove_quadratic_trend`.
- Three-character minimum, exempting coefficients and real coordinates (`x`, `y`, `z`),
  `new` and `at`. The deck readers that used to be exempt — where the method name *was* the
  type, `f32()`/`i32()` — went with `deck.rs` in §4.3.
- Unit suffixes on public arguments — `_km`, `_s`, `_rad`, `_hz`. `dt` is exempt; its
  interpretation is unambiguous.
- **Keep the provenance line.** Every ported routine's doc comment names its Fortran original
  and `hb_high_ref.f:NNN`. §4.3 deleted `reference/`, which makes these *more* valuable rather
  than less: they are now the only map from this code back to the original, and the original
  is recoverable from git history when a question needs it.
- Lints are `warn`, never `deny`, and a suppression is a targeted `#[allow]` **with the
  reason written next to it**. A crate-root blanket is how 28 warnings accumulated unseen.

## 4. Efficiency beats obedience

Reassociation, vectorisation, hoisting, tabulation, changed draw counts, threading: all
legal. **A measured win needs no fidelity justification** — only a green snapshot, or, if it
moves the snapshot, an argument for why the new numbers are right (see §6).

Two things this specifically unblocks, both previously forbidden:

- **Accumulation order.** `PORTING_RULES.md` §5 required float reductions to fold
  left-to-right because that is what the Fortran did. No longer. The rule is now *choose
  an order deliberately, measure it, record it* — and reassociation may well be **more**
  accurate, since pairwise summation beats a left-to-right `f32` fold over 16384 samples.
- **Draw counts.** The number of random draws was a reproducibility contract with every
  golden and CSV in the repo. Stage 3 broke it deliberately, once, with the long tier
  adjudicating; the long tier is gone, so a future break needs the reasoning written down.

**Measure, do not assume.** This repo has a good record of measuring things that turned
out the opposite way round: `lto = "fat"` is 1.2% *slower*, `overflow-checks = false` is
1.87% *more* instructions, and a `sqrt(1-cos²)` identity that saved 2.46% was wrong in
`f32` by 2.4e-4. `PROFILE.md` also spent a stage claiming an `Array1` wrapper was free
when removing it was worth 5.13%. Numbers in this repo are load-bearing; do not add one
you have not measured.

## 5. What still cannot move

Short, and none of it is fidelity to the Fortran:

- **The component order** of the returned array — 090, 000, vertical. §4.3 deleted the raw
  `f32`-at-a-seek-offset file and the `(1x,f10.4)` distance on stderr along with the CLI
  driver, because a Python caller gets an array and a `PyErr`. What survives is the order the
  three channels come in, which `hf_sim.py` labels and `bb_sim` consumes positionally.
- **Same-version determinism.** One seed, one build, one answer. Free to change *across*
  versions — results are regenerable by pinning the commit — but never within one.
- **`next_f32`'s 24-bit conversion.** Not fidelity: dividing a full `u32` by 2³² rounds
  values near 1 *up* to exactly 1.0, and the zero-rejection loops need `[0, 1)`.
- **The deliberate physics choices**, which are decisions rather than accidents: the ±2%
  equivalence band, and Boore's unit-power spectral calibration.

## 6. How work is gated

| | what | when |
| --- | --- | --- |
| `cargo test --workspace` | properties, kernel goldens, and `snapshot.rs` — the whole pipeline against `harness/golden/snapshot.txt` on a frozen draw source | **every commit**, seconds |
| `pytest tests/` | batch invariants, `station_seeds` properties, stub/dataclass agreement | **every commit**, ~10 s |
| `cargo clippy --workspace` | zero warnings | **every commit**, and CI runs it with `-D warnings` |

**The statistical tiers are gone with the oracle.** §4.3 deleted `crates/validate`,
`crates/im`, `reference/` and the `run_*` scripts, because the port is certified and the
instrument had nothing left to measure. What replaced them is not weaker for the questions
that remain: `snapshot.rs` pins the pipeline exactly rather than distributionally, and
`ENGINEERING_RULES` §6's rules were derived from those campaigns and outlive them.

If a future change needs a statistical adjudication again — a new physics option, a changed
draw structure — the campaign is recoverable from git history at `71d43c3` or earlier, and
`REFACTOR.md` records the results it produced so a new run has something to compare with.

Three rules about gates themselves, all learned the hard way:

- **Every gate asserts its own resolution, not only its verdict.** A tier that cannot
  resolve the band it claims to test has *abstained*, and abstention must not be spelled
  the same way as success. Tier B reported PASS when its stream desynchronised, because
  nothing checked that the sample could decide anything.
- **A gate that cannot fail is worse than no gate**, because it reads as coverage. Delete
  it rather than leave it.
- **Every gate reports its own false-alarm rate, measured against a null run.** This is the
  mirror of the first rule and it cost a whole campaign to learn. Stage 3's LONG returned
  373 of 375 certified on the mean with 0 refuted — and its new quantile gate flagged 14 of
  375, with no way to ask how many it flags when *both sides are the same program*. A count
  with no null beside it is uninterpretable: 14 could be a defect or it could be the gate's
  resting pulse. Tier D was never in that position, because it prints its family-wise
  false-alarm rate (53.7%) next to every verdict, and that number is the only reason its
  lone `p=0.005` reads as expected rather than alarming. **A gate with no null is not a
  gate, it is an opinion.**

Two corollaries, both of which the quantile gate got wrong:

- **A band derived for one statistic does not transfer to another.** The ±2% band is a
  deliberate choice about IM *means*. A tail quantile is a much noisier statistic at the
  same `n`, so the same numeric band is a materially stricter test — which is why
  `--shape-band` is now separate from `--band`.
- **A family of tests needs a multiplicity correction.** Tier D Holm-corrects its 15. The
  quantile gate refutes per-endpoint across **375** and corrects nothing.

The **snapshot** goes red by design when a commit changes the draw structure — count, order,
or generator. That is the gate saying "this one needs adjudicating", which is the correct
answer for exactly those commits. Re-record with `UPDATE_SNAPSHOT=1` and put the
adjudication in the commit message; a re-recorded snapshot with no explanation is
indistinguishable from a silently broken one.

## 7. Reproduced Fortran defects

The port deliberately reproduced several defects in the original. Under Stage 3 these are
**fixed one at a time**, each as its own commit, with the physical argument and the
measured effect recorded. The transliteration contract was the only reason to keep them.

The exception is a **frozen switch** — a parameter someone deliberately fixed, like `nsum`
being forced to 1 in 2004. Collapsing one removes real code but bakes in a decision that
was left adjustable, so it needs explicit sign-off rather than a tidy-up.

Beware the difference between dead code and code that looks dead. `nsum`'s loop body was
entirely dead *except* for one `rng.next_f32()` that advances the shared stream. Deleting
it as obvious dead code would have changed every waveform in the program.
