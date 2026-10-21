#!/usr/bin/env python3
"""Build the golden-comparison fixtures from the topography_runs campaign.

Each event of the campaign is a realisation (``realisation.json``) and the stoch slip model
``generate-stoch`` wrote for it (``stoch.h5``). This copies the slip model and distils the
realisation to what ``hf-sim`` reads from it, in ``case.json``:

* the HF configuration, rupture-velocity taper and 1-D velocity model;
* the record length (``domain.duration``) and ``seeds.hf_seed``;
* a fixed subset of the domain's stations, with their ``kappa_s`` and
  ``q_frequency_exponent`` from the site table.

The full domains hold up to ~100k stations, so a case keeps only the observation stations
(names of four characters or fewer, as ``extract_subset`` does) inside the domain, and of
those at most ``--stations-per-event``, evenly spaced in distance from the fault so near and
far field are both covered.

Usage (needs pandas and pyarrow for the site table, h5py for the slip model):
    uv run --with pandas --with pyarrow --with h5py harness/topography_fixtures.py \\
        --realisations ~/src/topography_runs/events_0p5Hz_squashed \\
        --stoch DIR_OF_<event>/stoch.h5 \\
        --sites ~/src/topography_runs/stations/sites.parquet

The slip models are on Dropbox, beside each event's outputs:
    rclone copy dropbox:/QuakeCoRE/Jake/topography_runs/events_0p5Hz_squashed DIR \\
        --include "/*/stoch.h5"
"""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEFAULT_OUT = Path(__file__).resolve().parent / "fixtures" / "topography_runs"

# Observation stations have three or four character names; the virtual grid's are longer.
OBSERVATION_NAME_LENGTH = 4


def inside(
    polygon: np.ndarray, longitude: np.ndarray, latitude: np.ndarray
) -> np.ndarray:
    """Even-odd point-in-polygon test in longitude/latitude.

    Parameters
    ----------
    polygon : np.ndarray
        Corners as ``(n, 2)`` longitude/latitude, in order.
    longitude, latitude : np.ndarray
        The points to test.

    Returns
    -------
    np.ndarray
        Whether each point is inside.
    """
    result = np.zeros(longitude.shape, bool)
    for (x0, y0), (x1, y1) in zip(polygon, np.roll(polygon, -1, axis=0)):
        crosses = (y0 > latitude) != (y1 > latitude)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = x0 + (latitude - y0) * (x1 - x0) / (y1 - y0)
        result ^= crosses & (longitude < x_at)
    return result


def pick_stations(
    realisation: dict, fault_point: tuple[float, float], sites: pd.DataFrame, count: int
) -> pd.DataFrame:
    """The observation stations in the domain, at most ``count``, spread in distance.

    Parameters
    ----------
    realisation : dict
        The realisation, for its domain corners.
    fault_point : tuple[float, float]
        A point on the fault, longitude/latitude.
    sites : pd.DataFrame
        The site table, indexed by station name.
    count : int
        The most stations to keep.

    Returns
    -------
    pd.DataFrame
        The chosen rows of the site table, nearest first.
    """
    corners = np.array(
        [[c["longitude"], c["latitude"]] for c in realisation["domain"]["domain"]]
    )
    observed = sites[sites.index.str.len() <= OBSERVATION_NAME_LENGTH]
    in_domain = observed[
        inside(corners, observed["longitude"].values, observed["latitude"].values)
    ]
    lon0, lat0 = fault_point
    distance = np.hypot(
        (in_domain["longitude"] - lon0) * np.cos(np.radians(lat0)),
        in_domain["latitude"] - lat0,
    )
    ordered = in_domain.iloc[np.argsort(distance.values, kind="stable")]
    keep = np.unique(
        np.linspace(0, len(ordered) - 1, min(count, len(ordered))).round().astype(int)
    )
    return ordered.iloc[keep]


def case(realisation: dict, stations: pd.DataFrame) -> dict:
    """What ``hf-sim`` reads from a realisation, plus the chosen stations.

    Returns
    -------
    dict
        The ``case.json`` content.
    """
    model = realisation["hf_velocity_model_1d"]
    return {
        "event": realisation["metadata"]["name"],
        "defaults_version": realisation["metadata"]["defaults_version"],
        "magnitude": next(iter(realisation["magnitudes"]["magnitudes"].values())),
        "hf": realisation["hf"],
        "rupture_velocity": realisation["rupture_velocity"],
        "hf_velocity_model_1d": {"model": model["model"], "vs_moho": model["vs_moho"]},
        "duration_s": realisation["domain"]["duration"],
        "hf_seed": realisation["seeds"]["hf_seed"],
        "stations": [
            {
                "name": name,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "kappa_s": row.kappa_s,
                "q_frequency_exponent": row.q_frequency_exponent,
            }
            for name, row in stations.iterrows()
        ],
    }


def main() -> None:
    """Build every fixture whose event has both a realisation and a slip model."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--realisations",
        type=Path,
        required=True,
        help="directory of <event>/R1/realisation.json",
    )
    parser.add_argument(
        "--stoch", type=Path, required=True, help="directory of <event>/stoch.h5"
    )
    parser.add_argument(
        "--sites", type=Path, required=True, help="the campaign's sites.parquet"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stations-per-event", type=int, default=6)
    args = parser.parse_args()

    sites = pd.read_parquet(args.sites)
    built, skipped = [], []
    for realisation_path in sorted(args.realisations.glob("*/R1/realisation.json")):
        event = realisation_path.parent.parent.name
        stoch = args.stoch / event / "stoch.h5"
        if not stoch.exists():
            skipped.append(event)
            continue
        realisation = json.loads(realisation_path.read_text())
        with h5py.File(stoch) as tree:
            plane = tree["plane_0"].attrs
            fault_point = (float(plane["longitude"][0]), float(plane["latitude"][0]))
        stations = pick_stations(
            realisation, fault_point, sites, args.stations_per_event
        )
        if stations.empty:
            skipped.append(event)
            continue
        destination = args.out / event
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stoch, destination / "stoch.h5")
        (destination / "case.json").write_text(
            json.dumps(case(realisation, stations), indent=1) + "\n"
        )
        built.append(event)

    print(f"built {len(built)} fixtures in {args.out}")
    if skipped:
        print(f"skipped (no slip model or no stations): {' '.join(skipped)}")


if __name__ == "__main__":
    main()
