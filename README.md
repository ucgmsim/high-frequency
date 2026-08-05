# hf-simulation

Stochastic high-frequency seismogram generation: a Rust port of EMOD3D's
`hb_high_v6.0.3` (`BINMOD` + `VERSION1`), exposed to Python as a batched API.

```python
import numpy as np
from hf_simulation import (
    FaultSegment, HfConfig, SlipModel, VelocityModel1D,
    simulate_stations, station_seeds,
)

waveform = simulate_stations(              # (3, n_station, n_time), cm/s²
    SlipModel([segment]), velocity_model, HfConfig(duration_s=40.0),
    latitude_deg=latitudes, longitude_deg=longitudes,
    station_seed=station_seeds(1234, station_names),
)
```

**Status: scientifically equivalent to production Fortran.** Certified at ±2% on
intensity-measure means — **373 of 375 endpoints certified, 0 refuted**, the
remaining two undetermined for want of sample size rather than agreement.
Distribution shape is certified at ±4% on the 5th/50th/95th percentiles, and
inter-frequency correlation passes Holm-corrected at 0 of 15.

Every one of those gates is reported beside its own **null control** — the same
test with both sides drawn from the same program, so its false-alarm rate is
measured rather than assumed. The shape gate flags 11 of 375 on a null at ±2% and
0 of 375 at ±4%, which is where the ±4% comes from. See `ENGINEERING_RULES.md`
§6: a gate with no null is not a gate, it is an opinion.

It is **not** bit-identical to the Fortran and has not been since §2.1. That was
the Stage 1 contract and it was retired on purpose, one change at a time, each
with its measured effect recorded: the FFT (§2.1), the gamma function (§2.2b), a
WGS84 geodesic in place of `DELAZ5` (§2.5), reproduced defects (§2.6, §3.4), the
truncated pi literals (§2.8), and the RNG seeding (§3.1). `REFACTOR.md` is the
log.

## The interface is batched, and that is the point

The Fortran ran **one process per station**, driven by a 22-line text deck on
stdin, and reported the epicentral distance by printing it to stderr. Per-station
seeds had to be forged as `int32(root) ^ hash(name)` because a deck could carry
only one `i32`, which left about half of them negative.

Here a station's seed is a `uint64` derived through
`numpy.random.SeedSequence`, so stations are genuinely independent. Three
properties follow, none of which the Fortran could offer, all of them tested:

- **station order changes no waveform** — so dask rechunking is free;
- **simulating a subset equals slicing the whole batch** — so a failed run is
  resumable;
- **a batch of one equals that station inside a batch.**

The GIL is released for the whole batch, so it parallelises under a dask thread
pool. There is deliberately no internal thread pool: two schedulers competing for
the same cores oversubscribe them.

## Gates

```
cargo test --workspace     properties, kernel goldens, end-to-end snapshot
pytest tests/              batch invariants, seeding properties, the stub
cargo clippy --workspace   zero warnings, enforced in CI
cargo bench                per-fault-size timings (HB_BENCH_SLOW=1 for alpine)
```

`crates/hb_high/tests/snapshot.rs` pins the whole pipeline against
`harness/golden/snapshot.txt` using a frozen draw source. It goes **red by
design** when a commit changes the draw structure — count, order, or generator —
which is the gate asking for a statistical adjudication. Re-record with
`UPDATE_SNAPSHOT=1` and say why in the commit message.

Debug and release must agree with **each other**. Neither is bit-identical to the
Fortran any more, but a disagreement between the two profiles means the port
depends on optimisation-level float behaviour, which has already caught one real
bug. CI runs both.

## Where the Fortran went

`reference/`, `crates/validate`, `crates/im`, the deck reader and the CLI driver
were deleted in §4.3, once `tests/path_equivalence.rs` had proved the array path
reproduced the deck path sample-for-sample on 4, 112 and 2827 subfaults.
Certification transfers: deck path ≡ Fortran (statistically, at n=5100), array
path ≡ deck path (exactly), therefore array path ≡ Fortran.

All of it is recoverable from git history if a question ever needs the oracle
again. `PORTING_RULES.md` is retained as **archaeology** — it explains why
`hb_high_ref.f` does what it does, which is still the fastest way to read the
original, and it is explicitly *not* a description of this crate.
`ENGINEERING_RULES.md` is what governs the code now.

Python owns every file format, reusing `source_modelling.stoch.StochFile` and
`workflow.realisations.HFVelocityModel1D`.

EMOD3D is not modified by this project.
