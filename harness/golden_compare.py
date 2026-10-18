#!/usr/bin/env python3
"""Run one fixed suite of simulations against two revisions and report what moved.

A refactor that claims bit-identical output needs a check that runs the same inputs through
both builds and compares every sample. The snapshot test cannot be that check: it compares
summary statistics at a relative tolerance of 1e-5 and covers one configuration. This runs a
suite of cases that sweeps the inputs a change is likely to touch (ray sets, path-duration
models, Moho truncation, multi-segment faults, heterogeneous stress, per-station kappa and Q,
clipped records, the real Alpine slip model), then every event of the topography_runs
campaign as ``hf-sim`` ran it (``harness/fixtures/topography_runs``, built by
``harness/topography_fixtures.py``), and compares the records bit for bit.

It drives the public Python API, so it can run unchanged against any revision whose
``hf_simulation`` API matches this one.

Usage:
    harness/golden_compare.py run --out OUT.npz
        Run the suite with whichever ``hf_simulation`` is importable.

    harness/golden_compare.py compare BASE.npz HEAD.npz [--markdown REPORT.md]
        Compare two runs. Exits 1 if any case moved.

    harness/golden_compare.py ab [--base REF] [--work-dir DIR] [--markdown REPORT.md]
        Build the merge base of HEAD and REF (default ``origin/master``) and the working
        tree, each into its own uv environment, run the suite against both, and compare.

Beyond the package, `run` needs numpy and h5py (for the topography_runs slip models) and
`compare` numpy alone.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPOSITORY_ROOT / "harness" / "fixtures"

# Fraction of the peak that counts as the onset, matching the onset test the HF
# comparisons use: a realisation change moves it, a scaling change does not.
ONSET_FRACTION = 0.05

# The result-column marker for a bit-identical case.
IDENTICAL = "✅ identical"


# --------------------------------------------------------------------------- inputs ---


def read_velocity_model(path: Path) -> dict[str, np.ndarray]:
    """Read the harness velocity-model fixture: a layer count, then one row per layer.

    Parameters
    ----------
    path : Path
        The fixture file.

    Returns
    -------
    dict[str, np.ndarray]
        Thickness, velocities, density and Q, as ``VelocityModel1D`` keywords.
    """
    rows = np.loadtxt(path, skiprows=1, ndmin=2)
    return {
        "thickness_km": rows[:, 0].astype(np.float32),
        "vp_km_s": rows[:, 1].astype(np.float64),
        "vsh_km_s": rows[:, 2].astype(np.float64),
        "density_g_cm3": rows[:, 3].astype(np.float64),
        "quality_factor_p": rows[:, 4].astype(np.float32),
        "quality_factor_s": rows[:, 5].astype(np.float32),
    }


def read_stoch(path: Path) -> list[dict]:
    """Read a ``.stoch`` slip model into ``FaultSegment`` keyword sets.

    The format is one whitespace-delimited token stream: a segment count, then per segment
    ``lon lat nx nw dx dw``, ``strike dip rake top_depth hyp_along hyp_down`` and three
    ``nw x nx`` blocks (slip, rise time, rupture time), down-dip rows outermost.

    Parameters
    ----------
    path : Path
        The ``.stoch`` file.

    Returns
    -------
    list[dict]
        One keyword set per segment, without ``stress_drop_bars``.
    """
    tokens = iter(path.read_text().split())
    segment_count = int(next(tokens))
    segments = []
    for _ in range(segment_count):
        lon, lat = float(next(tokens)), float(next(tokens))
        along, down = int(next(tokens)), int(next(tokens))
        length_km, width_km = float(next(tokens)), float(next(tokens))
        strike, dip, rake, top, hyp_along, hyp_down = (
            float(next(tokens)) for _ in range(6)
        )
        blocks = [
            np.array(
                [float(next(tokens)) for _ in range(along * down)], np.float32
            ).reshape(down, along)
            for _ in range(3)
        ]
        segments.append(
            {
                "longitude_deg": lon,
                "latitude_deg": lat,
                "strike_deg": strike,
                "dip_deg": dip,
                "rake_deg": rake,
                "top_depth_km": top,
                "subfault_length_km": length_km,
                "subfault_width_km": width_km,
                "hypocentre_along_strike_km": hyp_along,
                "hypocentre_down_dip_km": hyp_down,
                "slip": blocks[0],
                "rise_time_s": blocks[1],
                "rupture_time_s": blocks[2],
            }
        )
    return segments


def uniform_segment(
    along: int,
    down: int,
    *,
    longitude_deg: float = 173.0,
    latitude_deg: float = -43.0,
    strike_deg: float = 220.0,
    dip_deg: float = 60.0,
    rake_deg: float = 160.0,
    top_depth_km: float = 1.0,
    hypocentre_down_dip_km: float = 1.5,
    subfault_km: float = 1.5,
) -> dict:
    """A uniform-slip segment, ``along`` by ``down`` subfaults, at 50 bars.

    Returns
    -------
    dict
        ``FaultSegment`` keywords.
    """
    shape = (down, along)
    return {
        "longitude_deg": longitude_deg,
        "latitude_deg": latitude_deg,
        "strike_deg": strike_deg,
        "dip_deg": dip_deg,
        "rake_deg": rake_deg,
        "top_depth_km": top_depth_km,
        "subfault_length_km": subfault_km,
        "subfault_width_km": subfault_km,
        "hypocentre_along_strike_km": 0.0,
        "hypocentre_down_dip_km": hypocentre_down_dip_km,
        "slip": np.full(shape, 50.0, np.float32),
        "rise_time_s": np.full(shape, 0.5, np.float32),
        "rupture_time_s": np.zeros(shape, np.float32),
        "stress_drop_bars": np.full(shape, 50.0, np.float32),
    }


def heterogeneous(segment: dict, seed: int) -> dict:
    """Give a segment rough slip, rise and rupture times and a log-normal stress field.

    Parameters
    ----------
    segment : dict
        ``FaultSegment`` keywords to start from.
    seed : int
        Seed for the perturbation, so the case is reproducible.

    Returns
    -------
    dict
        A copy with every grid perturbed.
    """
    rng = np.random.default_rng(seed)
    shape = segment["slip"].shape
    down, along = shape
    distance_km = np.hypot(
        np.arange(along)[None, :] * segment["subfault_length_km"],
        np.arange(down)[:, None] * segment["subfault_width_km"],
    )
    return segment | {
        # Some subfaults below the radiating threshold, so the moment normalisation's
        # count of radiating subfaults differs from the grid size.
        "slip": np.where(
            rng.random(shape) < 0.1, 0.0, rng.gamma(2.0, 40.0, shape)
        ).astype(np.float32),
        "rise_time_s": rng.uniform(0.3, 2.0, shape).astype(np.float32),
        "rupture_time_s": (distance_km / 2.8 + rng.uniform(0.0, 0.3, shape)).astype(
            np.float32
        ),
        "stress_drop_bars": np.exp(rng.normal(np.log(50.0), 0.6, shape)).astype(
            np.float32
        ),
    }


@dataclasses.dataclass(frozen=True)
class Stations:
    """A batch of stations and their per-station parameters."""

    names: tuple[str, ...]
    latitude_deg: tuple[float, ...]
    longitude_deg: tuple[float, ...]
    kappa_s: tuple[float, ...]
    q_frequency_exponent: tuple[float, ...]


def stations(
    points: list[tuple[str, float, float]],
    kappa_s: float | list[float] = 0.045,
    q_frequency_exponent: float | list[float] = 0.6,
) -> Stations:
    """Build a station batch, broadcasting scalar kappa and Q exponent.

    Returns
    -------
    Stations
        The batch.
    """
    count = len(points)

    def broadcast(value: float | list[float]) -> tuple[float, ...]:
        return tuple(value) if isinstance(value, list) else (value,) * count

    return Stations(
        names=tuple(p[0] for p in points),
        latitude_deg=tuple(p[1] for p in points),
        longitude_deg=tuple(p[2] for p in points),
        kappa_s=broadcast(kappa_s),
        q_frequency_exponent=broadcast(q_frequency_exponent),
    )


@dataclasses.dataclass(frozen=True)
class Case:
    """One simulation: a source, a medium, a configuration and a station batch."""

    name: str
    description: str
    segments: Callable[[], list[dict]]
    velocity_model: Callable[[], dict]
    config: Callable[[object], object]
    stations: Stations
    root_seed: int = 1234


NEAR = [
    ("NEAR", -43.1, 173.2),
    ("MID", -43.4, 172.6),
    ("FAR", -42.0, 175.0),
]


def cases() -> list[Case]:
    """The suite.

    Returns
    -------
    list[Case]
        Every case, in run order.
    """
    harness_vmod = lambda: read_velocity_model(FIXTURES / "velocity_model")

    def small_vmod() -> dict:
        return {
            "thickness_km": np.array([0.5, 1.5, 6.0, 10.0, 0.0], np.float32),
            "vp_km_s": np.array([1.8, 3.2, 5.5, 6.2, 8.0]),
            "vsh_km_s": np.array([0.6, 1.8, 3.2, 3.6, 4.5]),
            "density_g_cm3": np.array([1.9, 2.3, 2.6, 2.75, 3.3]),
            "quality_factor_p": np.array(
                [60.0, 180.0, 320.0, 360.0, 450.0], np.float32
            ),
            "quality_factor_s": np.array([30.0, 90.0, 160.0, 180.0, 225.0], np.float32),
        }

    def two_segments() -> list[dict]:
        first = heterogeneous(uniform_segment(10, 6), seed=1)
        second = heterogeneous(
            uniform_segment(
                8,
                6,
                longitude_deg=173.1,
                latitude_deg=-43.08,
                strike_deg=245.0,
                dip_deg=75.0,
                rake_deg=120.0,
            ),
            seed=2,
        )
        # Rupture reaches the second segment later.
        second["rupture_time_s"] = second["rupture_time_s"] + np.float32(4.0)
        return [first, second]

    def alpine() -> list[dict]:
        return [
            segment
            | {"stress_drop_bars": np.full(segment["slip"].shape, 50.0, np.float32)}
            for segment in read_stoch(FIXTURES / "stoch" / "alpine_base_r1.stoch")
        ]

    def alpine_heterogeneous_stress() -> list[dict]:
        rng = np.random.default_rng(71072)
        return [
            segment
            | {
                "stress_drop_bars": np.exp(
                    rng.normal(np.log(50.0), 0.5, segment["slip"].shape)
                ).astype(np.float32)
            }
            for segment in read_stoch(FIXTURES / "stoch" / "alpine_base_r1.stoch")
        ]

    def config(
        duration_s: float = 40.0,
        dt: float = 0.005,
        rayset: tuple[int, ...] = (1,),
        path_duration_model: int = 0,
        fmax_hz: float = 10.0,
        corner_frequency_constant: float = 2.0,
        sigma: float = 0.1,
    ) -> Callable[[object], object]:
        def build(hf: object) -> object:
            return hf.HfConfig(
                record=hf.RecordParameters(duration_s=duration_s, dt=dt),
                source=hf.SourceParameters(
                    corner_frequency_constant=corner_frequency_constant,
                    rupture_velocity=hf.RuptureVelocity(sigma=sigma),
                ),
                path=hf.PathParameters(
                    rayset=tuple(rayset), path_duration_model=path_duration_model
                ),
                site=hf.SiteParameters(fmax_hz=fmax_hz),
            )

        return build

    small = lambda: [uniform_segment(4, 3)]
    medium = lambda: [uniform_segment(14, 8)]
    production_vmod = lambda: harness_vmod() | {"vs_moho_km_s": 999.9}

    suite = [
        Case(
            "small",
            "4x3 uniform fault, production config",
            small,
            production_vmod,
            config(),
            stations(NEAR),
        ),
        Case(
            "medium",
            "14x8 uniform fault, production config",
            medium,
            production_vmod,
            config(),
            stations(NEAR),
        ),
        Case(
            "overhead",
            "station directly above the fault",
            small,
            production_vmod,
            config(duration_s=20.0),
            stations([("TOP", -43.0, 173.0)]),
        ),
        Case(
            "far_300km",
            "station ~300 km away, long record",
            medium,
            production_vmod,
            config(duration_s=200.0),
            stations([("FAR300", -40.6, 174.6)]),
        ),
    ]
    for code, label in [
        (0, "gp2010"),
        (1, "wus"),
        (2, "ena"),
        (11, "bt2014"),
        (12, "bt2015"),
    ]:
        suite.append(
            Case(
                f"duration_{label}",
                f"path-duration model {code}",
                medium,
                production_vmod,
                config(path_duration_model=code, duration_s=60.0),
                stations(NEAR),
            )
        )
    suite += [
        Case(
            "ray_straight",
            "straight-line ray only",
            medium,
            production_vmod,
            config(rayset=(0,)),
            stations(NEAR),
        ),
        Case(
            "ray_moho",
            "Moho reflection only, truncated Moho",
            medium,
            lambda: harness_vmod() | {"vs_moho_km_s": 4.0},
            config(rayset=(2,), duration_s=60.0),
            stations(NEAR),
        ),
        Case(
            "ray_all",
            "rays 0-4 summed, truncated Moho",
            medium,
            lambda: harness_vmod() | {"vs_moho_km_s": 4.0},
            config(rayset=(0, 1, 2, 3, 4), duration_s=60.0),
            stations(NEAR),
        ),
        Case(
            "ray_straight_far_slow",
            "straight line, far station, slow model (issue #7)",
            medium,
            lambda: small_vmod() | {"vs_moho_km_s": 999.9},
            config(rayset=(0, 1), duration_s=120.0),
            stations([("FAR300", -40.6, 174.6)]),
        ),
        Case(
            "small_vmod",
            "5-layer model with no thin surface layer",
            medium,
            lambda: small_vmod() | {"vs_moho_km_s": 999.9},
            config(),
            stations(NEAR),
        ),
        Case(
            "two_segments",
            "two heterogeneous segments, delayed second rupture",
            two_segments,
            production_vmod,
            config(duration_s=60.0),
            stations(NEAR),
        ),
        Case(
            "deep_hypocentre",
            "hypocentre below 15 km, so the deep taper moves",
            lambda: [
                uniform_segment(
                    10, 12, top_depth_km=12.0, dip_deg=45.0, hypocentre_down_dip_km=10.0
                )
            ],
            production_vmod,
            config(),
            stations(NEAR),
        ),
        Case(
            "kappa_q_sweep",
            "per-station kappa and Q exponent varied",
            medium,
            production_vmod,
            config(),
            stations(
                [(f"KQ{k}", -43.3, 172.9) for k in range(5)],
                kappa_s=[0.0, 0.01, 0.045, 0.08, 0.045],
                q_frequency_exponent=[0.6, 0.6, 0.3, 0.9, 0.0],
            ),
        ),
        Case(
            "no_rupture_sigma",
            "rupture-velocity sigma 0, so no perturbation draws",
            medium,
            production_vmod,
            config(sigma=0.0),
            stations(NEAR),
        ),
        Case(
            "fmax_25_dt_0.01",
            "fmax above the vertical cap, coarser dt",
            medium,
            production_vmod,
            config(fmax_hz=25.0, dt=0.01),
            stations(NEAR),
        ),
        Case(
            "corner_2.1",
            "2010 corner-frequency constant",
            medium,
            production_vmod,
            config(corner_frequency_constant=2.1),
            stations(NEAR),
        ),
        Case(
            "clipped",
            "record far too short for the far station",
            medium,
            production_vmod,
            config(duration_s=5.0),
            stations(NEAR),
        ),
        Case(
            "alpine",
            "Alpine Fault realisation (2827 subfaults), 50 bars",
            alpine,
            production_vmod,
            config(duration_s=100.0, path_duration_model=11),
            stations(
                [("FJDS", -43.389137, 170.18423), ("dJRHSCA", -43.531956, 172.63547)]
            ),
        ),
        Case(
            "alpine_stress",
            "Alpine Fault with a log-normal stress field",
            alpine_heterogeneous_stress,
            production_vmod,
            config(duration_s=100.0, path_duration_model=11),
            stations([("HMCS", -42.716922, 170.96396)]),
        ),
    ]
    return suite + topography_cases()


def read_stoch_h5(path: Path) -> list[dict]:
    """Read a ``generate-stoch`` slip model into ``FaultSegment`` keyword sets.

    The same mapping as ``workflow``'s ``hf-sim``: one segment per ``plane_<n>`` group, in
    numeric order, each grid ``(dip, strike)``.

    Parameters
    ----------
    path : Path
        The ``stoch.h5`` file.

    Returns
    -------
    list[dict]
        One keyword set per plane.
    """
    import h5py  # Only these cases need it, so `compare` runs on numpy alone.

    with h5py.File(path) as tree:
        dx, dy = float(tree.attrs["dx"][0]), float(tree.attrs["dy"][0])
        names = sorted(tree, key=lambda name: int(name.removeprefix("plane_")))
        segments = []
        for name in names:
            plane = tree[name]
            dimensions = [
                tree[ref[0]].name.rsplit("/", 1)[-1]
                for ref in plane["slip"].attrs["DIMENSION_LIST"]
            ]
            assert dimensions == ["dip", "strike"], f"{path}:{name} is {dimensions}"
            attrs = {key: float(value[0]) for key, value in plane.attrs.items()}
            grid = {
                variable: plane[variable][()].astype(np.float32)
                for variable in ("slip", "rise_time", "rupture_time", "stress_drop")
            }
            segments.append(
                {
                    "longitude_deg": attrs["longitude"],
                    "latitude_deg": attrs["latitude"],
                    "strike_deg": attrs["strike"],
                    "dip_deg": attrs["dip"],
                    "rake_deg": attrs["rake"],
                    "top_depth_km": attrs["dtop"],
                    "subfault_length_km": dx,
                    "subfault_width_km": dy,
                    "hypocentre_along_strike_km": attrs["shypo"],
                    "hypocentre_down_dip_km": attrs["dhypo"],
                    "slip": grid["slip"],
                    "rise_time_s": grid["rise_time"],
                    "rupture_time_s": grid["rupture_time"],
                    "stress_drop_bars": grid["stress_drop"],
                }
            )
        return segments


def topography_cases() -> list[Case]:
    """One case per topography_runs event, as ``hf-sim`` would run it.

    The fixtures (``harness/topography_fixtures.py`` builds them) hold each event's
    ``stoch.h5`` and the realisation's HF inputs with a fixed station subset.

    Returns
    -------
    list[Case]
        The cases, in event order.
    """
    suite = []
    for directory in sorted((FIXTURES / "topography_runs").iterdir()):
        spec = json.loads((directory / "case.json").read_text())
        hf_spec, taper = spec["hf"], spec["rupture_velocity"]
        model = spec["hf_velocity_model_1d"]

        def velocity_model(model: dict = model) -> dict:
            layers = model["model"]
            column = lambda key, dtype: np.array(
                [layer[key] for layer in layers], dtype
            )
            return {
                "thickness_km": column("thickness", np.float32),
                "vp_km_s": column("Vp", np.float64),
                "vsh_km_s": column("Vs", np.float64),
                "density_g_cm3": column("rho", np.float64),
                "quality_factor_p": column("Qp", np.float32),
                "quality_factor_s": column("Qs", np.float32),
                "vs_moho_km_s": model["vs_moho"],
            }

        def config(
            hf: object,
            hf_spec: dict = hf_spec,
            taper: dict = taper,
            duration_s: float = spec["duration_s"],
        ) -> object:
            # `stress_drop_bars` is per subfault, from the slip model, as in `hf-sim`.
            source = {
                k: v for k, v in hf_spec["source"].items() if k != "stress_drop_bars"
            }
            return hf.HfConfig(
                source=hf.SourceParameters(
                    **source
                    | {
                        "rupture_velocity": hf.RuptureVelocity(
                            fraction=taper["rvfrac"],
                            shallow=taper["rvfrac_shal"],
                            deep=taper["rvfrac_deep"],
                            **source["rupture_velocity"],
                        ),
                    }
                ),
                path=hf.PathParameters(
                    rayset=tuple(hf.Ray(ray) for ray in hf_spec["path"]["rayset"]),
                    path_duration_model=hf.PathDurationModel(
                        hf_spec["path"]["path_duration_model"]
                    ),
                ),
                site=hf.SiteParameters(**hf_spec["site"]),
                record=hf.RecordParameters(duration_s=duration_s, **hf_spec["record"]),
            )

        rows = spec["stations"]
        suite.append(
            Case(
                f"topo_{spec['event']}",
                f"topography_runs {spec['event']}, Mw {spec['magnitude']:.1f}",
                lambda path=directory / "stoch.h5": read_stoch_h5(path),
                velocity_model,
                config,
                stations(
                    [(r["name"], r["latitude"], r["longitude"]) for r in rows],
                    kappa_s=[r["kappa_s"] for r in rows],
                    q_frequency_exponent=[r["q_frequency_exponent"] for r in rows],
                ),
                root_seed=spec["hf_seed"],
            )
        )
    return suite


# ------------------------------------------------------------------------------ run ---


def run(out: Path) -> None:
    """Run the suite with the importable ``hf_simulation`` and save every record.

    Parameters
    ----------
    out : Path
        Where to write the ``.npz``. A ``.json`` manifest goes alongside it.
    """
    import hf_simulation as hf

    records: dict[str, np.ndarray] = {}
    manifest: dict[str, dict] = {}
    for case in cases():
        start = time.perf_counter()
        try:
            slip = hf.SlipModel([hf.FaultSegment(**s) for s in case.segments()])
            simulator = hf.Simulator(
                slip, hf.VelocityModel1D(**case.velocity_model()), case.config(hf)
            )
            stations = case.stations
            records[case.name] = simulator.run_stations(
                latitude_deg=np.array(stations.latitude_deg, np.float32),
                longitude_deg=np.array(stations.longitude_deg, np.float32),
                station_seed=hf.station_seeds(case.root_seed, list(stations.names)),
                kappa_s=np.array(stations.kappa_s, np.float32),
                q_frequency_exponent=np.array(
                    stations.q_frequency_exponent, np.float32
                ),
            )
            error = None
        # A revision may reject a case the other accepts; that is a result, not a crash.
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
        manifest[case.name] = {
            "description": case.description,
            "stations": list(case.stations.names),
            "seconds": round(time.perf_counter() - start, 2),
            "error": error,
        }
        status = (
            error or f"{records[case.name].shape} in {manifest[case.name]['seconds']} s"
        )
        print(f"{case.name:<24} {status}", file=sys.stderr)

    np.savez_compressed(out, **records)
    out.with_suffix(".json").write_text(
        json.dumps({"package": hf.__file__, "cases": manifest}, indent=2)
    )


# -------------------------------------------------------------------------- compare ---


def onset_sample(trace: np.ndarray) -> int:
    """First sample whose magnitude reaches :data:`ONSET_FRACTION` of the peak.

    Returns
    -------
    int
        The sample index, or -1 for a silent trace.
    """
    magnitude = np.abs(trace)
    peak = magnitude.max()
    return int(np.argmax(magnitude >= ONSET_FRACTION * peak)) if peak > 0 else -1


def compare_case(base: np.ndarray, head: np.ndarray) -> dict:
    """Bitwise and physical differences between two records of one case.

    Parameters
    ----------
    base, head : np.ndarray
        Records shaped ``(3, n_station, n_time)``.

    Returns
    -------
    dict
        Whether they are bit-identical, and if not by how much they differ.
    """
    if base.shape != head.shape:
        return {"identical": False, "shape": f"{base.shape} -> {head.shape}"}
    differs = base.view(np.uint32) != head.view(np.uint32)
    result = {
        "identical": not differs.any(),
        "shape": str(base.shape),
        "samples_differ": int(differs.sum()),
        "samples": int(base.size),
    }
    if result["identical"]:
        return result

    peak = np.abs(base).max(axis=-1)
    head_peak = np.abs(head).max(axis=-1)
    rms = np.sqrt(np.mean(base.astype(np.float64) ** 2, axis=-1))
    head_rms = np.sqrt(np.mean(head.astype(np.float64) ** 2, axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.abs(head - base).max(axis=-1) / peak
        pga_ratio = head_peak / peak
        rms_ratio = head_rms / rms
    onset_shift = [
        onset_sample(h) - onset_sample(b)
        for b, h in zip(
            base.reshape(-1, base.shape[-1]), head.reshape(-1, head.shape[-1])
        )
    ]
    result |= {
        "max_abs_diff": float(np.abs(head - base).max()),
        "max_diff_over_peak": float(np.nanmax(relative)),
        "pga_ratio": (float(np.nanmin(pga_ratio)), float(np.nanmax(pga_ratio))),
        "rms_ratio": (float(np.nanmin(rms_ratio)), float(np.nanmax(rms_ratio))),
        "onset_shift_samples": (min(onset_shift), max(onset_shift)),
    }
    return result


def compare(base_path: Path, head_path: Path, markdown: Path | None) -> bool:
    """Compare two runs, print a table, and optionally write it as markdown.

    Parameters
    ----------
    base_path, head_path : Path
        ``.npz`` files written by :func:`run`.
    markdown : Path or None
        Where to write the report, if anywhere.

    Returns
    -------
    bool
        Whether every case is bit-identical.
    """
    base, head = np.load(base_path), np.load(head_path)
    base_manifest = json.loads(base_path.with_suffix(".json").read_text())["cases"]
    head_manifest = json.loads(head_path.with_suffix(".json").read_text())["cases"]

    rows = []
    all_identical = True
    for name, info in head_manifest.items():
        base_info = base_manifest.get(name, {"error": "missing", "seconds": ""})
        timing = f"{base_info['seconds']} → {info['seconds']}"
        if info["error"] or base_info["error"]:
            all_identical = False
            rows.append(
                (
                    name,
                    info["description"],
                    "—",
                    "⚠️ error",
                    f"base: {base_info['error'] or 'ok'}; head: {info['error'] or 'ok'}",
                    "",
                    "",
                    "",
                    "",
                    timing,
                )
            )
            continue
        result = compare_case(base[name], head[name])
        all_identical &= result["identical"]
        if result["identical"]:
            rows.append(
                (
                    name,
                    info["description"],
                    result["shape"],
                    IDENTICAL,
                    "0",
                    "",
                    "",
                    "",
                    "",
                    timing,
                )
            )
        elif "samples" not in result:
            rows.append(
                (
                    name,
                    info["description"],
                    result["shape"],
                    "❌ shape",
                    "",
                    "",
                    "",
                    "",
                    "",
                    timing,
                )
            )
        else:
            rows.append(
                (
                    name,
                    info["description"],
                    result["shape"],
                    "❌ moved",
                    f"{result['samples_differ']:,} / {result['samples']:,}",
                    f"{result['max_diff_over_peak']:.2e}",
                    "{:.4f}–{:.4f}".format(*result["pga_ratio"]),
                    "{:.4f}–{:.4f}".format(*result["rms_ratio"]),
                    "{}…{}".format(*result["onset_shift_samples"]),
                    timing,
                )
            )

    columns = (
        "case",
        "inputs",
        "shape (comp, sta, time)",
        "result",
        "samples differing",
        "max |Δ| / peak",
        "PGA ratio",
        "RMS ratio",
        "onset shift (samples)",
        "time base → head (s)",
    )

    def table(selected: list[tuple]) -> list[str]:
        return [
            "| " + " | ".join(columns) + " |",
            "|" + "---|" * len(columns),
            *("| " + " | ".join(map(str, row)) + " |" for row in selected),
        ]

    def tally(selected: list[tuple]) -> str:
        return f"{sum(row[3] == IDENTICAL for row in selected)} / {len(selected)}"

    synthetic = [row for row in rows if not row[0].startswith("topo_")]
    topography = [row for row in rows if row[0].startswith("topo_")]
    moved_topography = [row for row in topography if row[3] != IDENTICAL]
    summary = (
        f"**{tally(rows)} cases bit-identical** "
        f"({sum(int(base[n].size) for n in base.files):,} samples compared): "
        f"{tally(synthetic)} synthetic, {tally(topography)} topography_runs events."
    )
    report = [summary, "", "### Synthetic cases", "", *table(synthetic)]
    if topography:
        report += [
            "",
            "### topography_runs events",
            "",
            (
                f"Each event's `stoch.h5` and realisation, run as `hf-sim` would at up to six "
                f"observation stations: {tally(topography)} bit-identical."
            ),
            "",
        ]
        if moved_topography:
            report += [*table(moved_topography), ""]
        report += [
            "<details><summary>Every event</summary>",
            "",
            *table(topography),
            "",
            "</details>",
        ]
    report = "\n".join(report)
    print(report)
    if markdown:
        markdown.write_text(report + "\n")
    return all_identical


# ------------------------------------------------------------------------------- ab ---


def ab(base_ref: str, work_dir: Path, markdown: Path | None) -> bool:
    """Run the suite against the merge base of HEAD and ``base_ref`` and the working tree.

    The merge base is extracted with ``git archive`` and built into its own uv environment;
    the working tree uses its own. This script (from the working tree) drives both, so the
    two runs see exactly the same inputs.

    Parameters
    ----------
    base_ref : str
        The branch HEAD will merge into.
    work_dir : Path
        Scratch directory for the extracted tree and the outputs.
    markdown : Path or None
        Where to write the report, if anywhere.

    Returns
    -------
    bool
        Whether every case is bit-identical.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    merge_base = git("merge-base", "HEAD", base_ref)
    head = git("rev-parse", "HEAD")
    print(
        f"merge base {merge_base[:10]} ({base_ref}), head {head[:10]}", file=sys.stderr
    )

    work_dir.mkdir(parents=True, exist_ok=True)
    base_tree = work_dir / f"base-{merge_base[:10]}"
    if not base_tree.exists():
        base_tree.mkdir()
        archive = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "archive", merge_base],
            check=True,
            capture_output=True,
        ).stdout
        subprocess.run(["tar", "-x", "-C", str(base_tree)], input=archive, check=True)

    outputs = {}
    for label, tree in [("base", base_tree), ("head", REPOSITORY_ROOT)]:
        out = work_dir / f"{label}.npz"
        # An archive has no `.git` for setuptools-scm to read a version from.
        env = os.environ | {"SETUPTOOLS_SCM_PRETEND_VERSION_FOR_HF_SIMULATION": "0.0.0"}
        # `-I`: nothing from the current directory or the environment shadows the build
        # under test.
        subprocess.run(
            [
                "uv",
                "run",
                "--frozen",
                "--with",
                "h5py",
                "--project",
                str(tree),
                "python",
                "-I",
                str(Path(__file__).resolve()),
                "run",
                "--out",
                str(out),
            ],
            check=True,
            env=env,
        )
        outputs[label] = out

    identical = compare(outputs["base"], outputs["head"], markdown)
    if markdown:
        markdown.write_text(
            f"Golden comparison of `{head[:10]}` against merge base `{merge_base[:10]}` "
            f"(`{base_ref}`), run by `harness/golden_compare.py ab`.\n\n"
            + markdown.read_text()
        )
    return identical


def main() -> int:
    """Parse the command line and dispatch.

    Returns
    -------
    int
        The process exit code: 1 if anything moved.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run the suite")
    run_parser.add_argument("--out", type=Path, required=True)
    compare_parser = commands.add_parser("compare", help="compare two runs")
    compare_parser.add_argument("base", type=Path)
    compare_parser.add_argument("head", type=Path)
    compare_parser.add_argument("--markdown", type=Path)
    ab_parser = commands.add_parser(
        "ab", help="build and compare HEAD against its merge base"
    )
    ab_parser.add_argument("--base", default="origin/master")
    ab_parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "hf-golden-compare",
    )
    ab_parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    if args.command == "run":
        run(args.out)
        return 0
    if args.command == "compare":
        return 0 if compare(args.base, args.head, args.markdown) else 1
    return 0 if ab(args.base, args.work_dir, args.markdown) else 1


if __name__ == "__main__":
    sys.exit(main())
