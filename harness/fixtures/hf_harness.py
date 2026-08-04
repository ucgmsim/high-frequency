#!/usr/bin/env python3
"""HF profiling harness.

Builds a representative, controllable invocation of the HF binary
(hb_high_binmod_v6.0.3) so we can profile it. The stdin parameter
sequence faithfully replicates workflow.scripts.hf_sim.build_hf_input
(which targets v6.0.3) using the default values from the workflow
defaults.yaml.

Usage examples
--------------
  # Generate the shared velocity_model file (run once):
  python3 hf_harness.py gen-velmod

  # Run HF on one stoch file with a single station, timed:
  python3 hf_harness.py run --stoch <stoch_file> --duration 100 --out <outdir>

  # Just emit the stdin for inspection:
  python3 hf_harness.py stdin --stoch <stoch_file> --duration 100
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

import yaml

HARNESS_DIR = Path(__file__).resolve().parent
DEFAULTS_YAML = Path(
    "/home/arr65/src/workflow/workflow/default_parameters/root/defaults.yaml"
)
BINARY = Path("/home/arr65/src/EMOD3D/tools/hb_high_binmod_v6.0.3")
VELMOD = HARNESS_DIR / "velocity_model"


def load_defaults():
    with open(DEFAULTS_YAML) as f:
        return yaml.safe_load(f)


def gen_velmod(out: Path = VELMOD):
    """Write the 1D velocity model file: Nlayers then 'thic Vp Vs rho Qp Qs'."""
    d = load_defaults()
    model = d["velocity_model_1d"]["model"]
    lines = [str(len(model))]
    for layer in model:
        lines.append(
            f"{layer['thickness']} {layer['Vp']} {layer['Vs']} "
            f"{layer['rho']} {layer['Qp']} {layer['Qs']}"
        )
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(model)} layers)")
    return out


def read_fault_lonlat(stoch: Path):
    """Read fault reference lon/lat from stoch header line 2 (first two floats)."""
    with open(stoch) as f:
        f.readline()  # segment count
        parts = f.readline().split()
    return float(parts[0]), float(parts[1])


def build_stdin(stoch: Path, station_file: Path, output_file: Path, seed: int,
                duration: float, dt: float):
    """Replicate workflow.scripts.hf_sim.build_hf_input for v6.0.3 defaults."""
    d = load_defaults()
    hf = d["hf"]
    rv = d["rupture_velocity"]
    rayset = hf["rayset"]
    siteamp = int(not hf["no_siteamp"])
    mom = hf["mom"] if hf["mom"] else -1
    rupv = hf["rupv"] if hf["rupv"] else -1
    # stress-parameter line: Python `x or -1`, and tect_type default 0 -> -1
    fa = hf["stress_parameter_adjustment_fault_area"] or -1
    tm = hf["stress_parameter_adjustment_target_magnitude"] or -1
    tt = hf["stress_parameter_adjustment_tect_type"] or -1
    lines = [
        "",
        hf["sdrop"],
        str(station_file),
        str(output_file),
        f"{len(rayset)} {' '.join(str(r) for r in rayset)}",
        siteamp,
        f"{hf['nbu']} {hf['ift']} {hf['flo']} {hf['fhi']}",
        seed,
        1,  # nsite (one station in the input)
        f"{duration} {dt} {hf['fmax']} {hf['kappa']} {hf['qfexp']}",
        f"{rv['rvfrac']} {rv['rvfrac_shal']} {rv['rvfrac_deep']} {hf['czero']} {hf['calpha']}",
        f"{mom} {rupv}",
        str(stoch),
        str(VELMOD),
        hf["vs_moho"],
        f"{hf['nl_skip']} {hf['vp_sig']} {hf['vsh_sig']} {hf['rho_sig']} {hf['qs_sig']} {int(hf['ic_flag'])}",
        hf["velocity_name"],
        f"{hf['fa_sig1']} {hf['fa_sig2']} {hf['rv_sig1']}",
        hf["path_dur"],
        0,
        f"{fa} {tm} {tt}",
        0,  # seek bytes
        "",
    ]
    return "\n".join(str(x) for x in lines)


def make_station_file(stoch: Path, out: Path, dlon: float = 0.1, dlat: float = 0.0,
                      name: str = "PROF0001"):
    """Place a single station at a fixed offset from the fault reference point."""
    lon, lat = read_fault_lonlat(stoch)
    out.write_text(f"{lon + dlon} {lat + dlat} {name}\n")
    return out


def run(stoch: Path, outdir: Path, duration: float, dt: float, seed: int,
        binary: Path = BINARY, dlon: float = 0.1, repeats: int = 1):
    outdir.mkdir(parents=True, exist_ok=True)
    station_file = outdir / "station.ll"
    make_station_file(stoch, station_file, dlon=dlon)
    output_file = outdir / "hf_out.bin"
    stdin = build_stdin(stoch, station_file, output_file, seed, duration, dt)
    (outdir / "hf_stdin.txt").write_text(stdin)
    times = []
    for i in range(repeats):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(binary)], input=stdin, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        dt_run = time.perf_counter() - t0
        times.append(dt_run)
    print(f"stoch={stoch.parent.name:18s} rc={proc.returncode} "
          f"t={min(times):.4f}s (best of {repeats})  stderr={proc.stderr.strip()[:60]!r}")
    if proc.returncode != 0:
        print("STDOUT:", proc.stdout[-2000:], file=sys.stderr)
        print("STDERR:", proc.stderr[-2000:], file=sys.stderr)
    return proc, min(times)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gen-velmod")
    sp = sub.add_parser("stdin")
    sp.add_argument("--stoch", required=True)
    sp.add_argument("--duration", type=float, default=100.0)
    sp.add_argument("--dt", type=float, default=0.005)
    sr = sub.add_parser("run")
    sr.add_argument("--stoch", required=True)
    sr.add_argument("--out", required=True)
    sr.add_argument("--duration", type=float, default=100.0)
    sr.add_argument("--dt", type=float, default=0.005)
    sr.add_argument("--seed", type=int, default=12345)
    sr.add_argument("--binary", default=str(BINARY))
    sr.add_argument("--dlon", type=float, default=0.1)
    sr.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "gen-velmod":
        gen_velmod()
    elif a.cmd == "stdin":
        print(build_stdin(Path(a.stoch), Path("STATION"), Path("OUTPUT"), 12345,
                          a.duration, a.dt))
    elif a.cmd == "run":
        run(Path(a.stoch), Path(a.out), a.duration, a.dt, a.seed,
            Path(a.binary), a.dlon, a.repeats)


if __name__ == "__main__":
    main()
