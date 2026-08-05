"""The xarray assembly, and the fact that it stays optional."""

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
