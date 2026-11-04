"""The xarray assembly, and the fact that it stays optional."""

from typing import Any

import numpy as np
import pytest

from hf_simulation import COMPONENTS
from hf_simulation.dataset import to_dataset

N_STATION, N_TIME, DT = 3, 100, 0.005


def waveform() -> np.ndarray:
    """A distinguishable waveform per component and station.

    Returns
    -------
    np.ndarray
        Shaped ``(3, N_STATION, N_TIME)``.
    """
    return np.arange(len(COMPONENTS) * N_STATION * N_TIME, dtype=np.float32).reshape(
        len(COMPONENTS), N_STATION, N_TIME
    )


def test_dataset_round_trips_the_array() -> None:
    """The array comes back out unchanged, on the axes it went in on."""
    dset = to_dataset(
        waveform(),
        station_names=["AAA", "BBB", "CCC"],
        latitude_deg=np.array([-43.0, -43.1, -43.2], np.float32),
        longitude_deg=np.array([172.0, 172.1, 172.2], np.float32),
        dt=DT,
        start_s=1.5,
    )
    np.testing.assert_array_equal(dset["waveform"].values, waveform())
    assert dset["waveform"].dims == ("component", "station", "time")
    assert list(dset.component.values) == list(COMPONENTS)
    assert dset.attrs == {"dt": DT, "nt": N_TIME, "start_sec": 1.5, "units": "cm/s^2"}
    # The time axis must start at start_s, not at zero: bb_sim aligns LF and HF using it,
    # so an off-by-one-sample origin silently shifts one band against the other.
    assert dset.time.values[0] == pytest.approx(1.5)
    assert dset.time.values[1] - dset.time.values[0] == pytest.approx(DT)


def test_selecting_a_station_by_name_works() -> None:
    """Station is a labelled dimension, which is the point of using xarray at all."""
    dset = to_dataset(
        waveform(),
        station_names=["AAA", "BBB", "CCC"],
        latitude_deg=np.array([-43.0, -43.1, -43.2], np.float32),
        longitude_deg=np.array([172.0, 172.1, 172.2], np.float32),
        dt=DT,
    )
    np.testing.assert_array_equal(
        dset.sel(station="BBB")["waveform"].values, waveform()[:, 1, :]
    )
    assert dset.sel(station="CCC").latitude.item() == pytest.approx(-43.2)


def test_mismatched_metadata_is_rejected() -> None:
    """Fewer names than stations must not silently truncate the dataset."""
    with pytest.raises(ValueError, match="metadata lengths"):
        to_dataset(
            waveform(),
            station_names=["AAA"],
            latitude_deg=np.array([-43.0], np.float32),
            longitude_deg=np.array([172.0], np.float32),
            dt=DT,
        )


def test_a_wrong_component_count_is_rejected() -> None:
    """A transposed array must raise rather than produce a nonsense dataset."""
    with pytest.raises(ValueError, match="n_station, n_time"):
        to_dataset(
            waveform()[0],
            station_names=["AAA", "BBB", "CCC"],
            latitude_deg=np.array([-43.0, -43.1, -43.2], np.float32),
            longitude_deg=np.array([172.0, 172.1, 172.2], np.float32),
            dt=DT,
        )


def test_the_core_package_does_not_need_xarray() -> None:
    """`import hf_simulation` must not pull in xarray; only the dataset module does."""
    import subprocess
    import sys

    probe = (
        "import sys, hf_simulation; "
        "assert 'xarray' not in sys.modules, 'importing hf_simulation dragged in xarray'"
    )
    assert subprocess.run([sys.executable, "-c", probe], check=False).returncode == 0


def _metadata(n_station: int) -> dict[str, Any]:
    """Station metadata for ``n_station`` stations, with distinguishable values.

    Returns
    -------
    dict
        Keyword arguments for :func:`to_dataset` other than the waveform and ``dt``.
    """
    return {
        "station_names": [f"S{i}" for i in range(n_station)],
        "latitude_deg": np.linspace(-43.0, -44.0, n_station, dtype=np.float32),
        "longitude_deg": np.linspace(172.0, 173.0, n_station, dtype=np.float32),
    }


def test_coordinates_sit_on_their_own_dimensions() -> None:
    """Every coordinate is indexed on the dimension bb_sim selects it by."""
    meta = _metadata(N_STATION)
    dset = to_dataset(waveform(), dt=DT, **meta)
    assert dict(dset.sizes) == {
        "component": len(COMPONENTS),
        "station": N_STATION,
        "time": N_TIME,
    }
    assert dset.component.dims == ("component",)
    assert dset.station.dims == ("station",)
    assert dset.time.dims == ("time",)
    assert dset.latitude.dims == ("station",)
    assert dset.longitude.dims == ("station",)
    np.testing.assert_array_equal(dset.longitude.values, meta["longitude_deg"])
    np.testing.assert_array_equal(dset.latitude.values, meta["latitude_deg"])
    # Label-based selection along each axis must hit the matching slice of the array.
    np.testing.assert_array_equal(
        dset.sel(component=COMPONENTS[2])["waveform"].values, waveform()[2]
    )
    np.testing.assert_array_equal(
        dset.sel(time=5 * DT, method="nearest")["waveform"].values, waveform()[:, :, 5]
    )
    assert dset.sel(station="S2").longitude.item() == pytest.approx(173.0)


def test_time_axis_defaults_to_starting_at_zero() -> None:
    """Without ``start_s`` the first sample is at the origin, and attrs say so."""
    dset = to_dataset(waveform(), dt=DT, **_metadata(N_STATION))
    np.testing.assert_allclose(dset.time.values, np.arange(N_TIME) * DT)
    assert dset.attrs["start_sec"] == 0.0


def test_component_count_is_checked_on_the_first_axis() -> None:
    """A ``(n_station, 3, n_time)`` array is transposed and must be rejected.

    Uses a station count other than three so the two axes cannot be confused.
    """
    n_station = 5
    swapped = np.zeros((n_station, len(COMPONENTS), N_TIME), np.float32)
    with pytest.raises(ValueError, match="n_station, n_time"):
        to_dataset(swapped, dt=DT, **_metadata(len(COMPONENTS)))
    # And the correctly ordered array with five stations is accepted.
    good = np.zeros((len(COMPONENTS), n_station, N_TIME), np.float32)
    assert to_dataset(good, dt=DT, **_metadata(n_station)).sizes["station"] == n_station


@pytest.mark.parametrize("bad", ["station_names", "latitude_deg", "longitude_deg"])
def test_mismatch_message_names_every_metadata_length(bad: str) -> None:
    """The error reports each metadata field by its argument name and length."""
    meta = _metadata(N_STATION)
    meta[bad] = meta[bad][:1]
    with pytest.raises(ValueError) as excinfo:
        to_dataset(waveform(), dt=DT, **meta)
    message = str(excinfo.value)
    assert f"waveform has {N_STATION} stations" in message
    for name in ("station_names", "latitude_deg", "longitude_deg"):
        expected = 1 if name == bad else N_STATION
        assert f"'{name}': {expected}" in message
