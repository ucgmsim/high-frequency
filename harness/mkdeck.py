#!/usr/bin/env python3
"""Emit an hb_high v6.0.3 stdin parameter deck.

Replicates ``workflow.scripts.hf_sim.build_hf_input`` with the values from
``workflow/default_parameters/root/defaults.yaml``. The defaults are inlined
rather than read from the yaml so that this harness is self-contained and the
goldens cannot silently move when the workflow repo changes.

Two quirks of the production deck are deliberate and must not be "fixed":

* The leading blank line. List-directed input skips blank records, so the first
  ``read(5,*)`` still lands on ``sdrop``.
* The bare ``0`` line before the stress-parameter line. ``read(5,*)`` at source
  line 385 wants *three* items, so it consumes that ``0`` as ``ispar_adjust``
  and then continues into the next record for ``targ_mag`` and ``fault_area``.
  The third value on that line is discarded and ``seek_bytes`` reads the final
  ``0``. Production has always behaved this way; ``tect_type`` never reaches the
  program.
"""

import argparse
from pathlib import Path

# --- defaults.yaml: hf section -------------------------------------------------
SDROP = 50.0
RAYSET = [1]
NO_SITEAMP = False
NBU, IFT, FLO, FHI = 4, 0, 0.02, 19.9
FMAX, KAPPA, QFEXP = 10.0, 0.045, 0.6
CZERO, CALPHA = 2.1, -99.0
MOM, RUPV = None, None
VS_MOHO = 999.9
NL_SKIP = -99
VP_SIG = VSH_SIG = RHO_SIG = QS_SIG = 0.0
IC_FLAG = True
VELOCITY_NAME = "-1"
FA_SIG1, FA_SIG2, RV_SIG1 = 0.0, 0.0, 0.1
PATH_DUR = 11
SPA_FAULT_AREA = None
SPA_TARGET_MAGNITUDE = None
SPA_TECT_TYPE = 0

# --- defaults.yaml: rupture_velocity section ----------------------------------
RVFRAC, RVFRAC_SHAL, RVFRAC_DEEP = 0.8, 0.7, 0.7


def build_deck(stoch, velmod, station_file, output_file, seed, duration, dt,
               *, ipdur=PATH_DUR, siteamp_override=None, rayset=None,
               rupv=None, kappa=None, vs_moho=None, ift=None, fhi=None):
    siteamp = int(not NO_SITEAMP) if siteamp_override is None else siteamp_override
    rayset = RAYSET if rayset is None else rayset
    kappa = KAPPA if kappa is None else kappa
    vs_moho = VS_MOHO if vs_moho is None else vs_moho
    ift = IFT if ift is None else ift
    fhi = FHI if fhi is None else fhi
    mom = MOM or -1
    # rupv > 0 takes the geometric rupture-time branch, which is the only way to
    # reach the irand jitter test at source line 1366. Production passes -1, so
    # that branch is dead there.
    rupv = (RUPV or -1) if rupv is None else rupv
    # Python truthiness, replicated exactly: tect_type 0 is falsy, so `or -1`
    # turns it into -1. hf_sim.py does the same.
    fa = SPA_FAULT_AREA or -1
    tm = SPA_TARGET_MAGNITUDE or -1
    tt = SPA_TECT_TYPE or -1
    lines = [
        "",
        SDROP,
        station_file,
        output_file,
        f"{len(rayset)} {' '.join(str(r) for r in rayset)}",
        siteamp,
        f"{NBU} {ift} {FLO} {fhi}",
        seed,
        1,  # nsite: hf_sim.py invokes the binary once per station
        f"{duration} {dt} {FMAX} {kappa} {QFEXP}",
        f"{RVFRAC} {RVFRAC_SHAL} {RVFRAC_DEEP} {CZERO} {CALPHA}",
        f"{mom} {rupv}",
        stoch,
        velmod,
        vs_moho,
        f"{NL_SKIP} {VP_SIG} {VSH_SIG} {RHO_SIG} {QS_SIG} {int(IC_FLAG)}",
        VELOCITY_NAME,
        f"{FA_SIG1} {FA_SIG2} {RV_SIG1}",
        ipdur,
        0,
        f"{fa} {tm} {tt}",
        0,  # seek_bytes
        "",
    ]
    return "\n".join(str(x) for x in lines)


def fault_ref_lonlat(stoch: Path):
    """First two floats of the stoch header's second record."""
    with open(stoch) as f:
        f.readline()  # nevnt
        parts = f.readline().split()
    return float(parts[0]), float(parts[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stoch", type=Path, required=True)
    ap.add_argument("--velmod", type=Path, required=True)
    ap.add_argument("--station-file", type=Path, required=True)
    ap.add_argument("--output-file", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=123456789)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--dlon", type=float, default=0.1)
    ap.add_argument("--dlat", type=float, default=0.0)
    ap.add_argument("--station-name", default="TEST0001")
    ap.add_argument("--ipdur", type=int, default=PATH_DUR)
    ap.add_argument("--siteamp", type=int, default=None)
    ap.add_argument("--rayset", type=str, default=None,
                    help="comma-separated ray types, e.g. 1,2")
    ap.add_argument("--rupv", type=float, default=None)
    ap.add_argument("--kappa", type=float, default=None)
    ap.add_argument("--vs-moho", type=float, default=None)
    ap.add_argument("--ift", type=int, default=None)
    ap.add_argument("--fhi", type=float, default=None)
    ap.add_argument("--write-station", action="store_true",
                    help="also write the station file, offset from the fault reference point")
    a = ap.parse_args()

    if a.write_station:
        lon, lat = fault_ref_lonlat(a.stoch)
        a.station_file.write_text(
            f"{lon + a.dlon} {lat + a.dlat} {a.station_name}\n"
        )

    rayset = [int(x) for x in a.rayset.split(",")] if a.rayset else None
    print(build_deck(a.stoch, a.velmod, a.station_file, a.output_file,
                     a.seed, a.duration, a.dt,
                     ipdur=a.ipdur, siteamp_override=a.siteamp, rayset=rayset,
                     rupv=a.rupv, kappa=a.kappa, vs_moho=a.vs_moho,
                     ift=a.ift, fhi=a.fhi), end="")


if __name__ == "__main__":
    main()
