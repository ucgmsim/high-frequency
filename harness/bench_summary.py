#!/usr/bin/env python3
"""Summarise criterion results into a committable CSV.

Criterion's own HTML reports are good for browsing but useless for review and
diffing: they are not text, and they live under `target/`, which is gitignored.
This walks `target/criterion/**/new/estimates.json` and emits one row per
benchmark so the baseline can be committed and later runs diffed against it.

Usage:
    harness/bench_summary.py [--out harness/bench_baseline.csv]
                             [--criterion-dir target/criterion]
                             [--compare harness/bench_baseline.csv]

With --compare, prints a change report against a previously committed CSV
instead of overwriting it. Criterion's own --baseline mechanism does this too,
but only inside target/, so it does not survive a clean build or a checkout on
another machine.

No third-party dependencies.
"""

import argparse
import csv
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path

# Criterion writes `benchmarks.json` alongside estimates in some versions and not
# others; the directory layout is the reliable source of the group/id split.
ESTIMATES = "estimates.json"


def find_runs(root: Path) -> Iterator[tuple[str, str, str, dict, dict | None]]:
    """Yield (group, benchmark id, parameter, estimates dict, throughput or None).

    Criterion's layout is target/criterion/<group>/<id>/<param>/new/estimates.json
    for parameterised benches and .../<group>/<id>/new/... for plain ones. Rather
    than guess the depth, take everything between the root and `new` as the
    path components.
    """
    for est in sorted(root.rglob(f"*/new/{ESTIMATES}")):
        parts = est.relative_to(root).parts[:-2]  # drop 'new', 'estimates.json'
        if not parts:
            continue
        # `report` holds criterion's own index pages, not measurements.
        if parts[0] == "report":
            continue
        group = parts[0]
        rest = parts[1:]
        # A parameterised bench is <id>/<param>; a plain one is just <id>.
        bench = rest[0] if rest else ""
        param = "/".join(rest[1:]) if len(rest) > 1 else ""
        try:
            with open(est) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"warning: skipping {est}: {e}", file=sys.stderr)
            continue

        thr = None
        bench_json = est.parent / "benchmark.json"
        if bench_json.exists():
            try:
                with open(bench_json) as f:
                    meta = json.load(f)
                thr = meta.get("throughput")
            except (OSError, json.JSONDecodeError):
                pass
        yield group, bench, param, data, thr


def point(data: dict, key: str) -> float | None:
    """Criterion nests each statistic as {'confidence_interval': ..., 'point_estimate': ..., 'standard_error': ...}."""
    node = data.get(key) or {}
    return node.get("point_estimate")


def human_ns(ns: float | None) -> str:
    """Format nanoseconds in the largest unit that keeps the value below 1000."""
    if ns is None:
        return ""
    for unit, scale in (("ns", 1.0), ("us", 1e3), ("ms", 1e6), ("s", 1e9)):
        if ns < 1000 * scale:
            return f"{ns / scale:.3f} {unit}"
    return f"{ns / 1e9:.3f} s"


def throughput_per_sec(thr: dict | None, median_ns: float | None) -> tuple[str, str]:
    """Elements or bytes per second, if the bench declared a throughput."""
    if not thr or not median_ns:
        return "", ""
    for kind in ("Elements", "Bytes"):
        if kind in thr:
            n = thr[kind]
            return kind.lower(), f"{n / (median_ns * 1e-9):.4g}"
    return "", ""


def collect(root: Path) -> list[dict[str, str]]:
    """One CSV row per benchmark found under `root`, sorted by name."""
    rows = []
    for group, bench, param, data, thr in find_runs(root):
        median = point(data, "median")
        mean = point(data, "mean")
        # Criterion reports MAD as 'median_abs_dev'.
        mad = point(data, "median_abs_dev")
        std = point(data, "std_dev")
        unit, per_sec = throughput_per_sec(thr, median)
        rows.append(
            {
                "group": group,
                "benchmark": bench,
                "parameter": param,
                "median_ns": f"{median:.6g}" if median else "",
                "median_human": human_ns(median),
                "mean_ns": f"{mean:.6g}" if mean else "",
                "mad_ns": f"{mad:.6g}" if mad else "",
                "std_dev_ns": f"{std:.6g}" if std else "",
                "rel_mad": f"{mad / median:.4f}" if (mad and median) else "",
                "throughput_unit": unit,
                "throughput_per_sec": per_sec,
            }
        )
    rows.sort(key=lambda r: (r["group"], r["benchmark"], r["parameter"]))
    return rows


FIELDS = [
    "group",
    "benchmark",
    "parameter",
    "median_ns",
    "median_human",
    "mean_ns",
    "mad_ns",
    "std_dev_ns",
    "rel_mad",
    "throughput_unit",
    "throughput_per_sec",
]


def write_csv(rows: list[dict[str, str]], out: Path) -> None:
    """Write `rows` to `out` as CSV."""
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} benchmarks)")


def compare(rows: list[dict[str, str]], baseline: Path) -> int:
    """Print medians that moved more than 5% against `baseline`, and added or removed benches."""
    with open(baseline) as f:
        old = {
            (r["group"], r["benchmark"], r["parameter"]): r for r in csv.DictReader(f)
        }
    new = {(r["group"], r["benchmark"], r["parameter"]): r for r in rows}

    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = []
    for k in sorted(set(new) & set(old)):
        try:
            a = float(old[k]["median_ns"])
            b = float(new[k]["median_ns"])
        except (ValueError, KeyError):
            continue
        if a <= 0:
            continue
        ratio = b / a
        # Report anything outside +/-5%; below that criterion's own noise on this
        # machine is not distinguishable from a real change.
        if abs(math.log(ratio)) > math.log(1.05):
            changed.append((k, a, b, ratio))

    if not (added or removed or changed):
        print(f"no change beyond +/-5% against {baseline}")
        return 0

    if changed:
        print(f"{'benchmark':52} {'baseline':>12} {'now':>12} {'ratio':>8}")
        for (g, bch, p), a, b, ratio in sorted(changed, key=lambda x: -x[3]):
            name = "/".join(x for x in (g, bch, p) if x)
            flag = "SLOWER" if ratio > 1 else "faster"
            print(f"{name:52} {human_ns(a):>12} {human_ns(b):>12} {ratio:7.3f}x {flag}")
    for k in added:
        print(f"added:   {'/'.join(x for x in k if x)}")
    for k in removed:
        print(f"removed: {'/'.join(x for x in k if x)}")
    return 0


def main() -> int:
    """Entry point; see the module docstring for usage."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion-dir", type=Path, default=Path("target/criterion"))
    ap.add_argument("--out", type=Path, default=Path("harness/bench_baseline.csv"))
    ap.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="diff against a committed CSV instead of writing one",
    )
    a = ap.parse_args()

    if not a.criterion_dir.exists():
        sys.exit(f"{a.criterion_dir} does not exist -- run `cargo bench` first")

    rows = collect(a.criterion_dir)
    if not rows:
        sys.exit(f"no criterion estimates found under {a.criterion_dir}")

    if a.compare:
        return compare(rows, a.compare)
    write_csv(rows, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
