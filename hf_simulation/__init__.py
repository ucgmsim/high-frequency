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
>>> from hf_simulation import Simulator, station_seeds
>>> segment = FaultSegment(
...     longitude_deg=173.0, latitude_deg=-43.0,
...     strike_deg=220.0, dip_deg=70.0, rake_deg=160.0,
...     top_depth_km=0.0, subfault_length_km=1.5, subfault_width_km=1.5,
...     hypocentre_along_strike_km=0.0, hypocentre_down_dip_km=1.5,
...     slip=np.full((3, 4), 50.0, np.float32),
...     rise_time_s=np.full((3, 4), 0.5, np.float32),
...     rupture_time_s=np.zeros((3, 4), np.float32),
... )
>>> simulator = Simulator(
...     SlipModel([segment]),
...     VelocityModel1D(
...         thickness_km=np.array([1.0, 2.0, 0.0], np.float32),
...         vp_km_s=np.array([3.0, 5.0, 6.0]),
...         vsh_km_s=np.array([1.5, 2.8, 3.5]),
...         density_g_cm3=np.array([2.1, 2.5, 2.7]),
...         quality_factor_p=np.array([100.0, 200.0, 400.0], np.float32),
...         quality_factor_s=np.array([50.0, 100.0, 200.0], np.float32),
...         vs_moho_km_s=999.9,
...     ),
...     HfConfig(duration_s=10.0),
... )
>>> waveform = simulator.run_stations(
...     latitude_deg=np.array([-43.4], np.float32),
...     longitude_deg=np.array([172.6], np.float32),
...     station_seed=station_seeds(1234, ["CHCH"]),
... )
>>> waveform.shape[0]
3
"""

import dataclasses
import enum
import hashlib
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from numpy.random import SeedSequence

from hf_simulation._hf_simulation import (
    FaultSegment,
    SlipModel,
    VelocityModel1D,
)
from hf_simulation._hf_simulation import (
    HfConfig as _RustHfConfig,
)
from hf_simulation._hf_simulation import (
    PathParameters as _RustPathParameters,
)
from hf_simulation._hf_simulation import (
    RecordParameters as _RustRecordParameters,
)
from hf_simulation._hf_simulation import (
    Simulator as _RustSimulator,
)
from hf_simulation._hf_simulation import (
    SiteParameters as _RustSiteParameters,
)
from hf_simulation._hf_simulation import (
    SourceParameters as _RustSourceParameters,
)

__all__ = [
    "COMPONENTS",
    "FaultSegment",
    "HfConfig",
    "PathDurationModel",
    "Ray",
    "Simulator",
    "SlipModel",
    "VelocityModel1D",
    "station_seeds",
]


class Ray(enum.IntEnum):
    """A ray path to sum over.

    Only the paths that are meaningful are listed: the Rust tracer decodes any integer by
    parity, but a value outside this set names a Moho multiple nothing has ever run.
    """

    STRAIGHT = 0
    """No ray tracing: a straight-line geometric path."""
    DIRECT = 1
    """The direct upgoing ray. What production uses."""
    MOHO_REFLECTION = 2
    """Down to the Moho and back up."""


class PathDurationModel(enum.IntEnum):
    """How record duration grows with distance.

    The wire values are non-contiguous because every other integer left the duration table
    uninitialised in the original.
    """

    GRAVES_PITARKA_2010 = 0
    """Graves and Pitarka (2010) eq. 17, slope 0.063 s/km."""
    WESTERN_US = 1
    """Slope 0.070 s/km."""
    EASTERN_NORTH_AMERICA = 2
    """Slope 0.100 s/km."""
    BOORE_THOMPSON_2014 = 11
    """Boore and Thompson (2014) Table 1, active crustal regions."""
    BOORE_THOMPSON_2015 = 12
    """Boore and Thompson (2015), stable continental regions."""


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
    rayset: tuple[Ray, ...] = (Ray.DIRECT,)
    """Ray paths to sum over."""
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
    path_duration_model: PathDurationModel = PathDurationModel.GRAVES_PITARKA_2010
    """How record duration grows with distance."""

    def __post_init__(self) -> None:
        """Reject values that would produce a silently wrong record.

        Raises
        ------
        ValueError
            If a duration, sample interval or stress drop is not positive, or ``rayset``
            is empty.
        """
        for name in ("duration_s", "dt", "stress_drop_bars", "fmax_hz"):
            value = getattr(self, name)
            if not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not self.rayset:
            raise ValueError("rayset must name at least one ray path")

    def _to_rust(self) -> _RustHfConfig:
        """Build the Rust configuration objects this dataclass describes.

        Returns
        -------
        _RustHfConfig
            The same values, grouped the way the simulation core wants them.
        """
        return _RustHfConfig(
            source=_RustSourceParameters(
                stress_drop_bars=self.stress_drop_bars,
                corner_frequency_constant=self.corner_frequency_constant,
                corner_frequency_alpha=self.corner_frequency_alpha,
                rupture_velocity_fraction=self.rupture_velocity_fraction,
                rupture_velocity_shallow=self.rupture_velocity_shallow,
                rupture_velocity_deep=self.rupture_velocity_deep,
                rupture_velocity_sigma=self.rupture_velocity_sigma,
            ),
            path=_RustPathParameters(
                rayset=[int(ray) for ray in self.rayset],
                q_frequency_exponent=self.q_frequency_exponent,
                path_duration_model=int(self.path_duration_model),
            ),
            site=_RustSiteParameters(kappa_s=self.kappa_s, fmax_hz=self.fmax_hz),
            record=_RustRecordParameters(duration_s=self.duration_s, dt=self.dt),
        )


class Simulator:
    """A configured simulation, ready to run stations against.

    Build one per source. Everything that does not depend on where the receiver is — the
    air layer, the slip-model normalisation, the moment scaling — is done once here rather
    than once per station.

    Notes
    -----
    :meth:`run_stations` releases the GIL and does not mutate the simulator, so one of
    these can be shared across a **dask thread pool**. It is deliberately not picklable:
    ``dask.distributed`` would have to send it between processes, and re-deriving the
    normalisation on the far side is a failure mode better refused than hidden. There is
    also no internal thread pool by design — with dask on top, two schedulers competing
    for the same cores oversubscribe them.
    """

    def __init__(
        self,
        slip_model: SlipModel,
        velocity_model: VelocityModel1D,
        config: HfConfig,
    ) -> None:
        """Prepare a simulation.

        Parameters
        ----------
        slip_model : SlipModel
            The fault, as one or more :class:`FaultSegment`.
        velocity_model : VelocityModel1D
            The 1-D velocity structure.
        config : HfConfig
            Physical configuration.

        Raises
        ------
        ValueError
            If the slip model's segments disagree on subfault size.
        """
        self._inner = _RustSimulator(config._to_rust(), slip_model, velocity_model)

    def run_stations(
        self,
        *,
        latitude_deg: npt.NDArray[np.float32],
        longitude_deg: npt.NDArray[np.float32],
        station_seed: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.float32]:
        """Simulate a batch of stations.

        Parameters
        ----------
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

        Raises
        ------
        ValueError
            If the three arrays do not have one entry per station, or the batch is empty.
        """
        latitude = np.ascontiguousarray(latitude_deg, dtype=np.float32)
        longitude = np.ascontiguousarray(longitude_deg, dtype=np.float32)
        seeds = np.ascontiguousarray(station_seed, dtype=np.uint64)

        lengths = {
            "latitude_deg": latitude.shape,
            "longitude_deg": longitude.shape,
            "station_seed": seeds.shape,
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                "latitude_deg, longitude_deg and station_seed must have one entry per "
                f"station; got shapes {lengths}"
            )
        if latitude.size == 0:
            raise ValueError("no stations given, so there is nothing to simulate")

        return self._inner.run_stations(
            latitude_deg=latitude,
            longitude_deg=longitude,
            station_seed=seeds,
        )
