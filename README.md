# hf-simulation

Stochastic high-frequency seismogram generation: a Rust port of EMOD3D's
`hb_high_v6.0.3` (`BINMOD` + `VERSION1`), exposed to Python as a batched API.

```python
import numpy as np
from hf_simulation import (
    FaultSegment, HfConfig, RecordParameters, SlipModel, Simulator,
    VelocityModel1D, station_seeds,
)

# The four groups mirror the simulation core's own decomposition, so a
# configuration written down elsewhere deserialises straight into this.
config = HfConfig(record=RecordParameters(duration_s=40.0))

# Built once per source: the air layer, the slip-model normalisation and the
# moment scaling do not depend on where the receiver is.
simulator = Simulator(SlipModel([segment]), velocity_model, config)

waveform = simulator.run_stations(         # (3, n_station, n_time), cm/s²
    latitude_deg=latitudes, longitude_deg=longitudes,
    station_seed=station_seeds(1234, station_names),
)
```

## Installation

```
pip install hf-simulation            # core: numpy only
pip install "hf-simulation[xarray]"  # adds hf_simulation.dataset.to_dataset
```

## Status

Statistically equivalent to the production Fortran: intensity-measure means agree within
±2% at 373 of 375 endpoints tested (0 disagree; the remaining two are undetermined for want
of sample size). Distribution shape agrees within ±4% on the 5th/50th/95th percentiles, and
inter-frequency correlation passes Holm-corrected at 0 of 15. Each test is reported beside
a null control — the same test with both sides drawn from the same program — so its
false-alarm rate is measured rather than assumed. On the null, the shape test flags 11 of
375 at ±2% and none at ±4%, which is why the shape band is ±4%.

Output is not bit-identical to the Fortran: the FFT, the gamma function, the geodesic
(WGS84), the pi constants and the RNG seeding all differ, and a few defects in the original
are fixed. Each difference was checked statistically against the production output.

## The batched interface

`Simulator.run_stations` takes arrays of station coordinates and seeds and
returns every station at once. A station's seed is a `uint64` derived by
`station_seeds` through `numpy.random.SeedSequence` from a root seed and the
station name, so stations are independent. Three properties follow, all of them
tested:

- **station order changes no waveform** — so dask rechunking is free;
- **simulating a subset equals slicing the whole batch** — so a failed run is
  resumable;
- **a batch of one equals that station inside a batch.**

The GIL is released for the whole batch, so it parallelises under a dask thread
pool. There is deliberately no internal thread pool: two schedulers competing for
the same cores oversubscribe them.

## Tests

```
pytest tests/              batch invariants, seeding properties, the stub
cargo test --workspace     properties, kernel goldens, end-to-end snapshot
cargo clippy --workspace   zero warnings, enforced in CI
cargo bench                per-fault-size timings (HB_BENCH_SLOW=1 for alpine)
```

`crates/hb_high/tests/snapshot.rs` pins the whole pipeline against
`harness/golden/snapshot.txt` using a frozen draw source. It fails by design
when a change alters the draw structure (count, order or generator), because
such a change needs a statistical check rather than a sample comparison.
Re-record with `UPDATE_SNAPSHOT=1` and say why in the commit message.

CI runs the Rust tests in both debug and release, which must agree with each
other: a disagreement means the output depends on optimisation-level float
behaviour.

## Scope

Python owns every file format; read `.stoch` slip models and velocity models
with `source_modelling.stoch.StochFile` and
`workflow.realisations.HFVelocityModel1D`. EMOD3D itself is not modified by
this project.
