"""Wrong dtypes and dimensions are rejected with an error naming the argument.

pyo3-numpy only accepts an array whose dtype and number of dimensions match exactly, and
otherwise raises ``TypeError: 'ndarray' object is not an instance of 'ndarray'`` — which
names neither the argument nor the expected type. These pin the Python-side checks that
run first.
"""

import re

import numpy as np
import pytest

from hf_simulation import (
    FaultSegment,
    HfConfig,
    RecordParameters,
    Simulator,
    SlipModel,
    VelocityModel1D,
    station_seeds,
)
from hf_simulation import _hf_simulation as extension

GRIDS = ("slip", "rise_time_s", "rupture_time_s", "stress_drop_bars")
FLOAT32_LAYERS = ("thickness_km", "quality_factor_p", "quality_factor_s")
FLOAT64_LAYERS = ("vp_km_s", "vsh_km_s", "density_g_cm3")
STATION_ARRAYS = (
    "latitude_deg",
    "longitude_deg",
    "station_seed",
    "kappa_s",
    "q_frequency_exponent",
)


def segment_arguments() -> dict:
    """Keyword arguments for a valid 3-by-4 segment."""
    return {
        "longitude_deg": 173.0,
        "latitude_deg": -43.0,
        "strike_deg": 220.0,
        "dip_deg": 70.0,
        "rake_deg": 160.0,
        "top_depth_km": 0.0,
        "subfault_length_km": 1.5,
        "subfault_width_km": 1.5,
        "hypocentre_along_strike_km": 0.0,
        "hypocentre_down_dip_km": 1.5,
        "slip": np.full((3, 4), 50.0, np.float32),
        "rise_time_s": np.full((3, 4), 0.5, np.float32),
        "rupture_time_s": np.zeros((3, 4), np.float32),
        "stress_drop_bars": np.full((3, 4), 50.0, np.float32),
    }


def velocity_arguments() -> dict:
    """Keyword arguments for a valid three-layer model."""
    return {
        "thickness_km": np.array([1.0, 2.0, 0.0], np.float32),
        "vp_km_s": np.array([3.0, 5.0, 6.0]),
        "vsh_km_s": np.array([1.5, 2.8, 3.5]),
        "density_g_cm3": np.array([2.1, 2.5, 2.7]),
        "quality_factor_p": np.array([100.0, 200.0, 400.0], np.float32),
        "quality_factor_s": np.array([50.0, 100.0, 200.0], np.float32),
        "vs_moho_km_s": 999.9,
    }


def station_arguments() -> dict:
    """Keyword arguments for a valid two-station batch."""
    return {
        "latitude_deg": np.array([-43.4, -43.6], np.float32),
        "longitude_deg": np.array([172.6, 172.4], np.float32),
        "station_seed": station_seeds(1234, ["CHCH", "LINC"]),
        "kappa_s": np.array([0.045, 0.045], np.float32),
        "q_frequency_exponent": np.array([0.6, 0.6], np.float32),
    }


def test_valid_input_still_builds_the_extension_types() -> None:
    """The validated constructors return the Rust objects, and isinstance still holds."""
    segment = FaultSegment(**segment_arguments())
    velocity_model = VelocityModel1D(**velocity_arguments())
    assert isinstance(segment, extension.FaultSegment)
    assert isinstance(segment, FaultSegment)
    assert isinstance(velocity_model, extension.VelocityModel1D)
    assert isinstance(velocity_model, VelocityModel1D)
    assert not isinstance(velocity_model, FaultSegment)
    assert SlipModel([segment]).subfault_count == 12


@pytest.mark.parametrize("name", GRIDS)
def test_a_float64_grid_names_the_argument(name: str) -> None:
    arguments = segment_arguments()
    arguments[name] = arguments[name].astype(np.float64)
    message = (
        f"FaultSegment: '{name}' must be a 2-D float32 array, got float64 with shape "
        "(3, 4); use .astype(np.float32)"
    )
    with pytest.raises(TypeError, match=re.escape(message)):
        FaultSegment(**arguments)


@pytest.mark.parametrize("name", GRIDS)
def test_a_1d_grid_names_the_argument(name: str) -> None:
    arguments = segment_arguments()
    arguments[name] = arguments[name].ravel()
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"FaultSegment: '{name}' must be a 2-D float32 array, got float32 with "
            "shape (12,)"
        ),
    ):
        FaultSegment(**arguments)


def test_a_list_grid_names_the_argument() -> None:
    arguments = segment_arguments()
    arguments["slip"] = [[50.0] * 4] * 3
    with pytest.raises(TypeError, match="FaultSegment: 'slip' must be a 2-D float32"):
        FaultSegment(**arguments)


@pytest.mark.parametrize(
    ("name", "expected", "wrong"),
    [(name, "float32", np.float64) for name in FLOAT32_LAYERS]
    + [(name, "float64", np.float32) for name in FLOAT64_LAYERS],
)
def test_a_wrong_dtype_layer_names_the_argument(
    name: str, expected: str, wrong: type
) -> None:
    arguments = velocity_arguments()
    arguments[name] = arguments[name].astype(wrong)
    message = (
        f"VelocityModel1D: '{name}' must be a 1-D {expected} array, got "
        f"{np.dtype(wrong)} with shape (3,); use .astype(np.{expected})"
    )
    with pytest.raises(TypeError, match=re.escape(message)):
        VelocityModel1D(**arguments)


@pytest.mark.parametrize("name", FLOAT32_LAYERS + FLOAT64_LAYERS)
def test_a_2d_layer_names_the_argument(name: str) -> None:
    arguments = velocity_arguments()
    arguments[name] = arguments[name].reshape(3, 1)
    with pytest.raises(
        ValueError, match=re.escape(f"VelocityModel1D: '{name}' must be a 1-D")
    ):
        VelocityModel1D(**arguments)


def test_the_velocity_model_message_spells_out_the_dtype_split() -> None:
    """A caller who built all six the same way learns which three are which."""
    arguments = velocity_arguments()
    arguments["thickness_km"] = arguments["thickness_km"].astype(np.float64)
    with pytest.raises(TypeError) as error:
        VelocityModel1D(**arguments)
    assert "thickness_km, quality_factor_p and quality_factor_s are float32" in str(
        error.value
    )
    assert "vp_km_s, vsh_km_s and density_g_cm3 are float64" in str(error.value)


@pytest.mark.parametrize("name", STATION_ARRAYS)
def test_2d_station_input_names_the_argument(name: str) -> None:
    simulator = Simulator(
        SlipModel([FaultSegment(**segment_arguments())]),
        VelocityModel1D(**velocity_arguments()),
        HfConfig(record=RecordParameters(duration_s=10.0)),
    )
    arguments = station_arguments()
    arguments[name] = arguments[name].reshape(2, 1)
    with pytest.raises(
        ValueError, match=re.escape(f"'{name}' must be 1-D, got shape (2, 1)")
    ):
        simulator.run_stations(**arguments)
