"""Type stubs for the ``hf_simulation._hf_simulation`` Rust extension.

Hand-written, and `pyo3-stub-gen` was considered and rejected. It would derive this file
from the annotations, which is worth real money when a stub can drift — but this one cannot:
pyo3 raises ``TypeError`` on an unknown or missing keyword argument, so ``HfConfig._to_rust``
fails on the first test the moment the dataclass and these constructors disagree.
``tests/test_stub.py`` makes that failure explicit and named rather than incidental.
Generating the file would have added a TOML-parser dependency tree to type-check a surface a
test already keeps honest.
"""

import numpy as np

type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float32]]
type FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.float32]]
type FloatArray3D = np.ndarray[tuple[int, int, int], np.dtype[np.float32]]
type DoubleArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type SeedArray = np.ndarray[tuple[int], np.dtype[np.uint64]]

class FaultSegment:
    """One fault segment: geometry plus the three subfault grids."""

    def __init__(
        self,
        *,
        longitude_deg: float,
        latitude_deg: float,
        strike_deg: float,
        dip_deg: float,
        rake_deg: float,
        top_depth_km: float,
        subfault_length_km: float,
        subfault_width_km: float,
        hypocentre_along_strike_km: float,
        hypocentre_down_dip_km: float,
        slip: FloatArray2D,
        rise_time_s: FloatArray2D,
        rupture_time_s: FloatArray2D,
    ) -> None: ...

class SlipModel:
    """A whole slip model: one or more :class:`FaultSegment`."""

    def __init__(self, segments: list[FaultSegment]) -> None: ...
    @property
    def subfault_count(self) -> int: ...

class VelocityModel1D:
    """The 1-D velocity model, truncated at the Moho on construction."""

    def __init__(
        self,
        *,
        thickness_km: FloatArray1D,
        vp_km_s: DoubleArray1D,
        vsh_km_s: DoubleArray1D,
        density_g_cm3: DoubleArray1D,
        quality_factor_p: FloatArray1D,
        quality_factor_s: FloatArray1D,
        vs_moho_km_s: float,
    ) -> None: ...
    @property
    def layer_count(self) -> int: ...

class SourceParameters:
    """The earthquake source: radiation strength and rupture speed."""

    def __init__(
        self,
        *,
        stress_drop_bars: float,
        corner_frequency_constant: float,
        corner_frequency_alpha: float,
        rupture_velocity_fraction: float,
        rupture_velocity_shallow: float,
        rupture_velocity_deep: float,
        rupture_velocity_sigma: float,
    ) -> None: ...

class PathParameters:
    """The path from source to site."""

    def __init__(
        self,
        *,
        rayset: list[int],
        q_frequency_exponent: float,
        path_duration_model: int,
    ) -> None: ...

class SiteParameters:
    """The near-surface."""

    def __init__(self, *, kappa_s: float, fmax_hz: float) -> None: ...

class RecordParameters:
    """The shape of the record to produce."""

    def __init__(self, *, duration_s: float, dt: float) -> None: ...

class HfConfig:
    """Everything needed to simulate."""

    def __init__(
        self,
        *,
        source: SourceParameters,
        path: PathParameters,
        site: SiteParameters,
        record: RecordParameters,
    ) -> None: ...

class Simulator:
    """A configured simulation, ready to run stations against."""

    def __init__(
        self,
        config: HfConfig,
        slip_model: SlipModel,
        velocity_model: VelocityModel1D,
    ) -> None: ...
    def run_stations(
        self,
        *,
        latitude_deg: FloatArray1D,
        longitude_deg: FloatArray1D,
        station_seed: SeedArray,
    ) -> FloatArray3D: ...
