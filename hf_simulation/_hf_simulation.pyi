"""Type stubs for the ``hf_simulation._hf_simulation`` Rust extension.

Hand-written, and `pyo3-stub-gen` was considered and rejected. It would derive this file
from the annotations, which is worth real money when a stub can drift — but this one cannot:
pyo3 raises ``TypeError`` on an unknown or missing keyword argument, so
``simulate_stations``'s ``**dataclasses.asdict`` splat fails on the first test the moment
``HfConfig`` and ``_simulate_stations`` disagree. ``tests/test_stub.py`` makes that failure
explicit and named rather than incidental. Generating the file would have added a TOML-parser
dependency tree to type-check a surface a test already keeps honest.
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

def _simulate_stations(
    slip_model: SlipModel,
    velocity_model: VelocityModel1D,
    *,
    latitude_deg: FloatArray1D,
    longitude_deg: FloatArray1D,
    station_seed: SeedArray,
    duration_s: float,
    dt: float,
    stress_drop_bars: float,
    fmax_hz: float,
    kappa_s: float,
    q_frequency_exponent: float,
    rayset: list[int],
    site_amplification: bool,
    rupture_velocity_fraction: float | None = None,
    rupture_velocity_shallow: float | None = None,
    rupture_velocity_deep: float | None = None,
    rupture_velocity_override: float | None = None,
    corner_frequency_constant: float | None = None,
    corner_frequency_alpha: float | None = None,
    moment: float | None = None,
    fault_area_km2: float | None = None,
    target_magnitude: float | None = None,
    fourier_amplitude_sigma_1: float = 0.0,
    fourier_amplitude_sigma_2: float = 0.0,
    rupture_velocity_sigma: float = 0.0,
    path_duration_model: int = 0,
    stress_adjust_model: int = 0,
) -> FloatArray3D: ...
