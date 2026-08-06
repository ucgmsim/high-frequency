"""Stochastic high-frequency seismogram generation.

A Rust port of EMOD3D's ``hb_high_v6.0.3`` (``BINMOD``/``VERSION1``), certified
scientifically equivalent to the production Fortran at ±2% on intensity-measure means.

The interface is batched by design. The Fortran ran one process per station, assembled a
22-line text deck per call, and reported the epicentral distance by printing it to stderr;
per-station seeds had to be forged as ``int32(root) ^ hash(name)`` because the deck could
only carry one ``i32``. Here a station's seed is a ``uint64`` argument, so stations are
genuinely independent and a batch can be reordered, sliced or resumed without changing any
waveform.

Examples
--------
>>> import numpy as np
>>> from hf_simulation import FaultSegment, HfConfig, SlipModel, VelocityModel1D
>>> from hf_simulation import simulate_stations, station_seeds
>>> segment = FaultSegment(
...     longitude_deg=173.0, latitude_deg=-43.0,
...     strike_deg=220.0, dip_deg=70.0, rake_deg=160.0,
...     top_depth_km=0.0, subfault_length_km=1.5, subfault_width_km=1.5,
...     hypocentre_along_strike_km=0.0, hypocentre_down_dip_km=1.5,
...     slip=np.full((3, 4), 50.0, np.float32),
...     rise_time_s=np.full((3, 4), 0.5, np.float32),
...     rupture_time_s=np.zeros((3, 4), np.float32),
... )
>>> waveform = simulate_stations(
...     SlipModel([segment]),
...     VelocityModel1D(
...         thickness_km=np.array([1.0, 2.0, 0.0], np.float32),
...         vp_km_s=np.array([3.0, 5.0, 6.0]),
...         vsh_km_s=np.array([1.5, 2.8, 3.5]),
...         density_g_cm3=np.array([2.1, 2.5, 2.7]),
...         quality_factor_p=np.array([100.0, 200.0, 400.0], np.float32),
...         quality_factor_s=np.array([50.0, 100.0, 200.0], np.float32),
...     ),
...     HfConfig(duration_s=10.0),
...     latitude_deg=np.array([-43.4], np.float32),
...     longitude_deg=np.array([172.6], np.float32),
...     station_seed=station_seeds(1234, ["CHCH"]),
... )
>>> waveform.shape[0]
3
"""

import dataclasses
import hashlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import SeedSequence

from hf_simulation._hf_simulation import (
    FaultSegment,
    SlipModel,
    VelocityModel1D,
    _simulate_stations,
)

__all__ = [
    "COMPONENTS",
    "FaultSegment",
    "HfConfig",
    "SlipModel",
    "VelocityModel1D",
    "simulate_stations",
    "station_seeds",
]

# Component order of the returned array's first axis. This is the order the Fortran wrote
# to disk and the order `hf_sim.py` labels its xarray dimension with.
COMPONENTS = ("090", "000", "ver")

# Bytes of BLAKE2b digest used for a station-name hash. Eight gives a 64-bit value with no
# collisions across the largest station lists in use (5000 names measured clean).
_NAME_HASH_BYTES = 8


def _name_hash(name: str) -> int:
    """Hash a station name to an unsigned 64-bit integer, stably across processes.

    Parameters
    ----------
    name : str
        The station name.

    Returns
    -------
    int
        An unsigned 64-bit hash. Unsigned matters: ``SeedSequence`` rejects negative
        entropy outright, and ``workflow.utils.stable_hash`` returns a *signed* int.
    """
    # Same algorithm as workflow.utils.stable_hash, reimplemented rather than imported
    # because `workflow` is this package's consumer -- depending on it would be circular.
    # `signed=False` is the difference that matters; see the docstring.
    return int.from_bytes(
        hashlib.blake2b(name.encode("utf-8"), digest_size=_NAME_HASH_BYTES).digest(),
        "little",
        signed=False,
    )


def station_seeds(
    root_seed: int, station_names: Sequence[str]
) -> npt.NDArray[np.uint64]:
    """Derive per-station seeds that depend on the name, not the position.

    Each name is hashed and mixed with ``root_seed`` through
    :class:`numpy.random.SeedSequence`, so the result is invariant to the order and the
    number of stations passed. Adding a station to a run leaves every other station's
    waveform untouched, and ``station_seeds(root, names[a:b])`` equals
    ``station_seeds(root, names)[a:b]``.

    Parameters
    ----------
    root_seed : int
        The run's root seed. Must be non-negative.
    station_names : Sequence of str
        Station names. Order is irrelevant to the result.

    Returns
    -------
    npt.NDArray[np.uint64]
        One seed per station, in the order the names were given.

    Raises
    ------
    ValueError
        If ``root_seed`` is negative, which ``SeedSequence`` cannot accept as entropy.

    Notes
    -----
    If you are running several realisations of the same station, use ``SeedSequence(root_seed).spawn(n)`` instead.
    """
    if root_seed < 0:
        raise ValueError(
            f"root_seed must be non-negative, got {root_seed}; SeedSequence rejects "
            "negative entropy"
        )
    return np.array(
        [
            SeedSequence(entropy=[root_seed, _name_hash(name)]).generate_state(
                1, dtype=np.uint64
            )[0]
            for name in station_names
        ],
        dtype=np.uint64,
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class HfConfig:
    """Physical configuration for a high-frequency run.

    Every field is keyword-only and named for what it is.

    **These defaults are the single source of truth.** The Rust core takes concrete
    values for all of them and has no defaults of its own, so what is written here is
    what runs -- there is no second copy to drift out of step.
    """

    duration_s: float
    """Record length, seconds."""
    dt: float = 0.005
    """Sample interval, seconds."""
    stress_drop_bars: float = 50.0
    """Average stress drop. Graves and Pitarka use 50 bars."""
    fmax_hz: float = 10.0
    """High-frequency cutoff."""
    kappa_s: float = 0.045
    """Near-surface attenuation, seconds. Anderson and Hough (1984)."""
    q_frequency_exponent: float = 0.6
    """Frequency exponent of Q."""
    rayset: tuple[int, ...] = (1,)
    """Ray types to sum. 1 is the direct ray; 2 adds the Moho reflection."""
    site_amplification: bool = True
    """Apply the Boore-Joyner 1997 site amplification factors."""
    rupture_velocity_fraction: float = 0.8
    """Rupture velocity as a fraction of shear velocity.

    Graves and Pitarka (2010) set the average rupture speed at 80% of the local
    shear-wave velocity.
    """
    rupture_velocity_shallow: float = 0.6
    """Multiplier at the shallow end of the depth taper.

    **Not the published value.** Graves and Pitarka (2010) give 70% for the shallow weak
    zone; 0.6 is the locally calibrated value this pipeline has always run.
    """
    rupture_velocity_deep: float = 0.6
    """Multiplier at the deep end of the depth taper.

    **Not the published value** either: Graves and Pitarka (2015) give a 30% reduction
    for the deep weak zone.
    """
    rupture_velocity_sigma: float = 0.1
    """Log-normal scatter on the rupture-velocity factor. Live in production."""
    corner_frequency_constant: float = 2.0
    """The c0 coefficient of Graves and Pitarka (2010) eq. 13 / (2015) eq. 1.

    **2.0 is the 2015 value; the 2010 paper used 2.1.** This is a version marker: the
    code tracks the later parameterisation. See ``papers/README.md`` finding 5.
    """
    corner_frequency_alpha: float = 0.1
    """The c_alpha coefficient of the dip-and-rake corner-frequency adjustment."""
    path_duration_model: int = 0
    """0 Graves-Pitarka 2010, 1 WUS, 2 ENA, 11 Boore-Thompson 2014, 12 BT 2015."""


def simulate_stations(
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    config: HfConfig,
    *,
    latitude_deg: npt.NDArray[np.float32],
    longitude_deg: npt.NDArray[np.float32],
    station_seed: npt.NDArray[np.uint64],
) -> npt.NDArray[np.float32]:
    """Simulate a batch of stations against one source and one velocity model.

    Parameters
    ----------
    slip_model : SlipModel
        The fault, as one or more :class:`FaultSegment`.
    velocity_model : VelocityModel1D
        The 1-D velocity structure.
    config : HfConfig
        Physical configuration.
    latitude_deg : npt.NDArray[np.float32]
        Station latitudes, one per station.
    longitude_deg : npt.NDArray[np.float32]
        Station longitudes, one per station.
    station_seed : npt.NDArray[np.uint64]
        Per-station seeds, one per station. See :func:`station_seeds`.

    Returns
    -------
    npt.NDArray[np.float32]
        Acceleration in cm/s^2, shaped ``(3, n_station, n_time)`` with the first axis
        ordered as :data:`COMPONENTS`.

    Notes
    -----
    The GIL is released for the whole batch, so this parallelises under a dask thread
    pool. There is deliberately no internal thread pool: with dask on top, two schedulers
    competing for the same cores oversubscribe them.
    """
    return _simulate_stations(
        slip_model,
        velocity_model,
        latitude_deg=np.ascontiguousarray(latitude_deg, dtype=np.float32),
        longitude_deg=np.ascontiguousarray(longitude_deg, dtype=np.float32),
        station_seed=np.ascontiguousarray(station_seed, dtype=np.uint64),
        rayset=list(config.rayset),
        **{
            field.name: getattr(config, field.name)
            for field in dataclasses.fields(config)
            if field.name != "rayset"
        },
    )
