#!/usr/bin/env python3
"""Generate a synthetic HF "stoch" slip-model file.

The stoch file is read by hb_high_v6.0.3.f on Fortran unit 10 (opened at
source line ~248, parsed at lines 250-284). This generator reproduces that
exact format so the HF binary accepts and processes the file identically to a
real slip model of the same subfault count. Values (slip / rise time /
rupture time) are filled with simple representative numbers -- physical
realism is NOT the goal; identical processing cost (for runtime-vs-subfault
profiling) IS.

File format (one segment / "event"; nevnt = 1):
  line 1 : nevnt                                   (integer, here always 1)
  line 2 : lon lat nx ny dx dy                     (read into elonq,elatq,nx,nw,dx,dw)
  line 3 : strike dip rake dtop shyp dhyp          (strq,dipq,rakeq,dtop,shyp,dhyp)
  then 3 data blocks, each ny rows x nx values, list-directed (whitespace):
    block 1 : slip          sddp(i,j)  [cm]        (lines 270-272)
    block 2 : rise time     rist(i,j)  [s]         (lines 274-276)
    block 3 : rupture time  rupt(i,j)  [s] (tinit) (lines 278-280)
  Row index j = 1..ny is the DOWN-DIP index (nw); within a row the values run
  i = 1..nx, the ALONG-STRIKE index. (Reads are `(arr(i,j),i=1,nx)` inside
  `do j=1,nw`, so each line is one down-dip row, columns = along-strike.)

Bounds (params.h: nq=600, np=100, lv=1000; arrays sddp/rist/rupt(lv,nq,np)):
  nx <= nq = 600   (along-strike)
  ny <= np = 100   (down-dip)
  nevnt <= lv = 1000
  Total subfaults = nx*ny (nstot); not separately bounded -- it is only a
  counter/normaliser, never a 1-D array index, so the only limits are the
  per-dimension ones above (max single segment = 600*100 = 60000 subfaults).

Note on rupture time: the profiling harness passes rupv=-1 (=> vr<=0), so HF
uses rupt(i,j) straight from this file as the subfault rupture-arrival time
(hb_high_v6.0.3.f line 1349). We fill it with a radial pattern from the
hypocentre divided by a rupture velocity (exactly HF's own vr>0 formula,
lines 1351-1353), which is realistic and bounded.

CLI:
  python3 gen_stoch.py --nx 30 --ny 20 --out model.stoch
"""
import argparse
import math
import sys

# params.h limits (V6.0): parameter (nq=600,np=100,...,lv=1000)
NQ_MAX_NX = 600   # along-strike  (2nd dim of sddp(lv,nq,np))
NP_MAX_NY = 100   # down-dip      (3rd dim)
LV_MAX_SEG = 1000


def build_stoch(nx, ny, dx, dy, lon, lat, strike, dip, rake,
                dtop, shyp, dhyp, slip, rise, vrup):
    """Return the full stoch-file text for a single fault segment."""
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be >= 1")
    if nx > NQ_MAX_NX:
        raise ValueError(f"nx={nx} exceeds HF limit nq={NQ_MAX_NX}")
    if ny > NP_MAX_NY:
        raise ValueError(f"ny={ny} exceeds HF limit np={NP_MAX_NY}")

    # If hypocentre location not supplied, centre it on the fault:
    #   shyp = 0  -> along-strike centre (HF measures shyp from fault centre)
    #   dhyp = 0.5*ny*dy -> down-dip centre (measured from the top edge)
    if shyp is None:
        shyp = 0.0
    if dhyp is None:
        dhyp = 0.5 * ny * dy

    lines = []
    lines.append("1")  # nevnt: one segment
    # line 2: lon lat nx ny dx dy  (nx, ny MUST be plain integers for the
    # list-directed read into integer variables nx(iv), nw(iv))
    lines.append(f" {lon:.4f} {lat:.4f} {nx:d} {ny:d} {dx:.4f} {dy:.4f}")
    # line 3: strike dip rake dtop shyp dhyp  (all read into REAL vars)
    lines.append(f" {strike:.2f} {dip:.2f} {rake:.2f} "
                 f"{dtop:.4f} {shyp:.4f} {dhyp:.4f}")

    def block(value_fn):
        rows = []
        for j in range(1, ny + 1):          # down-dip row
            vals = [value_fn(i, j) for i in range(1, nx + 1)]  # along strike
            rows.append("".join(f"  {v:.5e}" for v in vals))
        return rows

    # Block 1: slip (cm). Constant, well above HF's 0.001 cutoff (line 696)
    # so every subfault is counted/processed.
    lines += block(lambda i, j: slip)
    # Block 2: rise time (s). Constant, non-zero.
    lines += block(lambda i, j: rise)
    # Block 3: rupture time / tinit (s). Radial from the hypocentre,
    # mirroring HF's own vr>0 calculation (lines 1351-1353); always >= 0.
    def rupt(i, j):
        xra = shyp - (i - 0.5 * (nx + 1)) * dx
        yra = dhyp - (j - 0.5) * dy
        return math.sqrt(xra * xra + yra * yra) / vrup
    lines += block(rupt)

    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate a synthetic HF stoch slip-model file.")
    ap.add_argument("--nx", type=int, required=True,
                    help=f"along-strike subfaults (1..{NQ_MAX_NX})")
    ap.add_argument("--ny", type=int, required=True,
                    help=f"down-dip subfaults (1..{NP_MAX_NY})")
    ap.add_argument("--out", required=True, help="output stoch file path")
    ap.add_argument("--dx", type=float, default=2.0, help="subfault size along strike (km)")
    ap.add_argument("--dy", type=float, default=2.0, help="subfault size down dip (km)")
    ap.add_argument("--lon", type=float, default=172.0, help="fault reference longitude")
    ap.add_argument("--lat", type=float, default=-43.0, help="fault reference latitude")
    ap.add_argument("--strike", type=float, default=45.0)
    ap.add_argument("--dip", type=float, default=80.0)
    ap.add_argument("--rake", type=float, default=160.0)
    ap.add_argument("--dtop", type=float, default=0.0, help="depth to top of fault (km)")
    ap.add_argument("--shyp", type=float, default=None,
                    help="along-strike hypocentre offset from centre (km); default 0")
    ap.add_argument("--dhyp", type=float, default=None,
                    help="down-dip hypocentre distance from top (km); default centre")
    ap.add_argument("--slip", type=float, default=50.0, help="constant slip (cm)")
    ap.add_argument("--rise", type=float, default=0.5, help="constant rise time (s)")
    ap.add_argument("--vrup", type=float, default=2.5,
                    help="rupture velocity for synthetic rupture times (km/s)")
    a = ap.parse_args(argv)

    text = build_stoch(a.nx, a.ny, a.dx, a.dy, a.lon, a.lat, a.strike, a.dip,
                       a.rake, a.dtop, a.shyp, a.dhyp, a.slip, a.rise, a.vrup)
    with open(a.out, "w") as f:
        f.write(text)
    print(f"wrote {a.out}: nx={a.nx} ny={a.ny} -> {a.nx * a.ny} subfaults "
          f"(3 data blocks of {a.ny} rows x {a.nx} cols)", file=sys.stderr)


if __name__ == "__main__":
    main()
