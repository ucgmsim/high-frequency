"""Assemble simulation output into an xarray Dataset.

Kept in its own module, and ``xarray`` is deliberately **not** a dependency of
``hf_simulation``. ``site_calculation`` ships ``numpy``/``scipy`` and leaves the xarray
plumbing to its callers, and the same split is worth having here: the batch API is usable
from a notebook, a test, or a script that just wants an array, without pulling in xarray,
dask and h5netcdf. Importing this module is the opt-in.

Notes
-----
The module is named ``dataset`` rather than ``xarray`` on purpose. A module named
``xarray`` inside the package reads as though it shadows the real one — it would not, since
Python 3 resolves ``import xarray`` absolutely, but a reader has to know that to be sure.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import xarray as xr

from hf_simulation import COMPONENTS

__all__ = ["to_dataset"]


def to_dataset(
    waveform: npt.NDArray[np.float32],
    *,
    station_names: Sequence[str],
    latitude_deg: npt.NDArray[np.float32],
    longitude_deg: npt.NDArray[np.float32],
    dt: float,
    start_s: float = 0.0,
) -> xr.Dataset:
    """Wrap a waveform array and its station metadata as an xarray Dataset.

    Parameters
    ----------
    waveform : npt.NDArray[np.float32]
        Acceleration in cm/s², shaped ``(3, n_station, n_time)`` as returned by
        :func:`hf_simulation.simulate_stations`.
    station_names : Sequence of str
        One name per station, in the order the waveforms were simulated.
    latitude_deg : npt.NDArray[np.float32]
        Station latitudes, one per station.
    longitude_deg : npt.NDArray[np.float32]
        Station longitudes, one per station.
    dt : float
        Sample interval in seconds.
    start_s : float, optional
        Time of the first sample, seconds relative to origin. Default 0.

    Returns
    -------
    xr.Dataset
        A ``waveform`` variable over ``(component, station, time)``, with ``latitude`` and
        ``longitude`` as station coordinates and ``dt``/``nt``/``start_sec``/``units`` in
        the attributes — the shape ``bb_sim`` already consumes.

    Raises
    ------
    ValueError
        If the array's shape and the metadata lengths disagree.

    Notes
    -----
    The ``component`` coordinate is labelled ``090``/``000``/``ver``, which is what those
    three channels physically are. ``hf_sim.py`` has always labelled them ``x``/``y``/``z``,
    which is not what they are — and nothing downstream depends on the labels, because
    ``bb_sim`` indexes that axis positionally and relabels its own output. Fixing the names
    here costs nothing and stops the next reader having to work it out.
    """
    if waveform.ndim != 3 or waveform.shape[0] != len(COMPONENTS):
        raise ValueError(
            f"waveform must be ({len(COMPONENTS)}, n_station, n_time), got "
            f"{waveform.shape}"
        )
    n_station, n_time = waveform.shape[1], waveform.shape[2]
    lengths = {
        "station_names": len(station_names),
        "latitude_deg": len(latitude_deg),
        "longitude_deg": len(longitude_deg),
    }
    if any(length != n_station for length in lengths.values()):
        raise ValueError(
            f"waveform has {n_station} stations but metadata lengths are {lengths}"
        )

    return xr.Dataset(
        {"waveform": (("component", "station", "time"), waveform)},
        coords={
            "component": ("component", list(COMPONENTS)),
            "station": ("station", list(station_names)),
            "time": ("time", start_s + np.arange(n_time) * dt),
            "latitude": ("station", np.asarray(latitude_deg)),
            "longitude": ("station", np.asarray(longitude_deg)),
        },
        attrs={"dt": dt, "nt": n_time, "start_sec": start_s, "units": "cm/s^2"},
    )
