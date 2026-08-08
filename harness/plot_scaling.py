#!/usr/bin/env python3
"""Plot the scaling sweep emitted by `benches/alpine.rs` under HB_STATION_LIST.

Two questions, two panels, and they are not the same question:

* **Fault size.** Does runtime scale with subfault count? It should not, quite —
  a longer rupture also places subfaults further from the station, and distance
  drives the shaping window and therefore the transform length. The straight
  reference line is what makes the departure visible.
* **Station count.** Runtime should be exactly linear: `Simulator::run` takes
  `&self`, each station seeds its own streams, and the batch loop is serial by
  design. The panel is really a test of that claim.

Usage:
    uv run --with pandas,matplotlib harness/plot_scaling.py scaling.csv outdir/
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Reference palette, light mode. Slots 1 and 2, validated as an adjacent pair.
MEASURED = "#2a78d6"
REFERENCE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e0dfda"


def style(ax):
    """Recessive grid and axes; the data is the only thing with weight."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="both", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(INK_2)


def main(csv_path: str, out_dir: str) -> None:
    frame = pd.read_csv(csv_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fault = frame[frame.sweep == "fault"].sort_values("subfaults")
    stations = frame[frame.sweep == "stations"].sort_values("stations")

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    # --- fault size, log-log -------------------------------------------------
    x = fault.subfaults.to_numpy()
    y = fault.wall_s_per_station.to_numpy()
    # Anchored at the smallest measured point, so the line says "if it were
    # linear in subfault count, it would go here".
    linear = y[0] * x / x[0]
    left.plot(x, linear, color=REFERENCE, linewidth=2, linestyle="--",
              label="linear in subfaults", zorder=2)
    left.plot(x, y, color=MEASURED, linewidth=2, marker="o", markersize=8,
              markeredgecolor=SURFACE, markeredgewidth=2, label="measured", zorder=3)
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("subfaults", color=INK_2, fontsize=10)
    left.set_ylabel("seconds per station", color=INK_2, fontsize=10)
    left.set_title("Runtime vs fault size", color=INK, fontsize=12, loc="left", pad=12)
    style(left)
    left.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    exponent = _slope(x, y)
    left.annotate(
        f"slope {exponent:.2f}",
        xy=(x[-1], y[-1]),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=INK_2,
    )

    # --- station count, linear ----------------------------------------------
    sx = stations.stations.to_numpy()
    sy = stations.wall_s.to_numpy()
    # Least-squares through the origin, NOT anchored on the n = 1 point. Anchoring there
    # would conflate "is this linear?" with "is the first station typical?" -- and it is not:
    # the per-station cost varies ~19% across this sample purely by geometry, so a reference
    # pinned to one station makes station mix look like curvature.
    rate = float((sx * sy).sum() / (sx * sx).sum())
    right.plot(sx, rate * sx, color=REFERENCE, linewidth=2, linestyle="--",
               label=f"linear at {rate:.2f} s/station", zorder=2)
    right.plot(sx, sy, color=MEASURED, linewidth=2, marker="o", markersize=8,
               markeredgecolor=SURFACE, markeredgewidth=2, label="measured", zorder=3)
    right.set_xlabel("stations", color=INK_2, fontsize=10)
    right.set_ylabel("seconds", color=INK_2, fontsize=10)
    subfaults = int(stations.subfaults.iloc[0])
    right.set_title(f"Runtime vs station count ({subfaults} subfaults)",
                    color=INK, fontsize=12, loc="left", pad=12)
    style(right)
    right.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")

    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"scaling.{suffix}", dpi=160, facecolor=SURFACE)
    print(f"wrote {out / 'scaling.png'} and {out / 'scaling.svg'}")
    print(f"fault-size log-log slope: {exponent:.3f}")
    per_station = stations.wall_s / stations.stations
    print(f"station sweep, s/station: min {per_station.min():.3f} "
          f"max {per_station.max():.3f} spread "
          f"{100 * (per_station.max() / per_station.min() - 1):.1f}%")


def _slope(x, y) -> float:
    """Least-squares slope in log-log, which is the scaling exponent."""
    import numpy as np

    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
