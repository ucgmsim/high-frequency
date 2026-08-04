#!/usr/bin/env python3
"""Phase 0c, leg 2 -- isolate the RNG swap.

Compares the unpatched radix-2 build (gfortran's intrinsic generator) against
the patched build (PCG32). Both use the same FFT, so the generator is the only
variable.

These two CANNOT agree sample by sample: a different generator means a
different noise realisation, which is the whole point of the swap. What must
hold is that they are the same *stochastic process* -- so this compares
distributions over many seeds, not waveforms.

Statistics reported, per component:

  * peak ground acceleration
  * Fourier amplitude, geometric-mean averaged in log-spaced frequency bands
  * 5%-damped pseudo-acceleration response spectrum at standard periods,
    via the Nigam-Jennings recursive filter

Acceptance is on the geometric-mean ratio between the two populations. Ratios
should sit near 1 with no systematic trend across frequency; a real error in
the RNG replacement (wrong variance, broken unit-RMS renormalisation, drawing
the wrong number of deviates) shows up as a flat offset or a frequency-dependent
tilt well outside sampling noise.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PERIODS = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 3.0])
BAND_EDGES = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])


def run_binary(binary, stoch, velmod, station, seed, duration, dt, workdir):
    out = workdir / f"out_{binary.name}_{seed}.bin"
    deck = subprocess.run(
        [sys.executable, str(HERE / "mkdeck.py"),
         "--stoch", str(stoch), "--velmod", str(velmod),
         "--station-file", str(station), "--output-file", str(out),
         "--seed", str(seed), "--duration", str(duration), "--dt", str(dt)],
        check=True, capture_output=True, text=True).stdout
    subprocess.run([str(binary)], input=deck, text=True, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32).reshape(-1, 3)


def nigam_jennings(acc, dt, periods, damping=0.05):
    """5%-damped pseudo-acceleration spectrum. Exact for piecewise-linear input,
    which is the standard engineering method (Nigam & Jennings 1969)."""
    psa = np.empty(len(periods))
    for k, T in enumerate(periods):
        wn = 2.0 * np.pi / T
        wd = wn * np.sqrt(1.0 - damping**2)
        e = np.exp(-damping * wn * dt)
        sin_, cos_ = np.sin(wd * dt), np.cos(wd * dt)
        a11 = e * (damping / np.sqrt(1 - damping**2) * sin_ + cos_)
        a12 = e / wd * sin_
        a21 = -wn / np.sqrt(1 - damping**2) * e * sin_
        a22 = e * (cos_ - damping / np.sqrt(1 - damping**2) * sin_)
        d = damping / (wn * dt)
        b11 = e * ((2 * damping**2 - 1) / (wn**2 * dt) + damping / wn) * sin_ / wd \
            + e * (2 * d / wn + 1 / wn**2) * cos_ - 2 * d / wn
        b12 = -e * ((2 * damping**2 - 1) / (wn**2 * dt)) * sin_ / wd \
            - e * (2 * d / wn) * cos_ - 1 / wn**2 + 2 * d / wn
        b21 = e * ((2 * damping**2 - 1) / (wn**2 * dt) + damping / wn) \
            * (cos_ - damping / np.sqrt(1 - damping**2) * sin_) \
            - e * (2 * d / wn + 1 / wn**2) * (wd * sin_ + damping * wn * cos_) \
            + 1 / (wn**2 * dt)
        b22 = -e * ((2 * damping**2 - 1) / (wn**2 * dt)) \
            * (cos_ - damping / np.sqrt(1 - damping**2) * sin_) \
            + e * (2 * d / wn) * (wd * sin_ + damping * wn * cos_) \
            - 1 / (wn**2 * dt)
        x = v = 0.0
        xmax = 0.0
        for i in range(len(acc) - 1):
            xn = a11 * x + a12 * v + b11 * acc[i] + b12 * acc[i + 1]
            vn = a21 * x + a22 * v + b21 * acc[i] + b22 * acc[i + 1]
            x, v = xn, vn
            xmax = max(xmax, abs(x))
        psa[k] = xmax * wn**2
    return psa


def banded_fas(acc, dt, edges):
    n = len(acc)
    amp = np.abs(np.fft.rfft(acc)) * dt
    freq = np.fft.rfftfreq(n, dt)
    out = np.empty(len(edges) - 1)
    for i in range(len(edges) - 1):
        m = (freq >= edges[i]) & (freq < edges[i + 1])
        # geometric mean, guarding the zero bins
        v = amp[m]
        v = v[v > 0]
        out[i] = np.exp(np.mean(np.log(v))) if v.size else np.nan
    return out


def gm_ratio(a, b):
    """Geometric-mean ratio with a lognormal 95% CI on the mean."""
    la, lb = np.log(a), np.log(b)
    d = la - lb
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
    return np.exp(d.mean()), np.exp(d.mean() - 1.96 * se), np.exp(d.mean() + 1.96 * se)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--stoch", type=Path,
                    default=ROOT / "harness/fixtures/stoch/2012p578973.stoch")
    ap.add_argument("--psa", action="store_true",
                    help="also compute response spectra (slow: pure-python oscillator)")
    a = ap.parse_args()

    orig = ROOT / "reference/build/hb_orig_radix2"
    patched = ROOT / "reference/build/hb_ref"
    for b in (orig, patched):
        if not b.exists():
            sys.exit(f"missing {b}; run harness/build_ref.sh first")

    velmod = ROOT / "harness/fixtures/velocity_model"

    with tempfile.TemporaryDirectory() as td:
        wd = Path(td)
        station = wd / "station.ll"
        # Station placement must match mkdeck's --write-station convention.
        subprocess.run(
            [sys.executable, str(HERE / "mkdeck.py"), "--stoch", str(a.stoch),
             "--velmod", str(velmod), "--station-file", str(station),
             "--output-file", str(wd / "unused.bin"), "--write-station"],
            check=True, capture_output=True)

        pk_o, pk_p, fas_o, fas_p, psa_o, psa_p = [], [], [], [], [], []
        for i in range(a.seeds):
            seed = 100003 + i * 7919  # arbitrary but fixed and spread out
            wo = run_binary(orig, a.stoch, velmod, station, seed, a.duration, a.dt, wd)
            wp = run_binary(patched, a.stoch, velmod, station, seed, a.duration, a.dt, wd)
            pk_o.append(np.abs(wo).max(axis=0))
            pk_p.append(np.abs(wp).max(axis=0))
            fas_o.append([banded_fas(wo[:, c], a.dt, BAND_EDGES) for c in range(3)])
            fas_p.append([banded_fas(wp[:, c], a.dt, BAND_EDGES) for c in range(3)])
            if a.psa:
                psa_o.append([nigam_jennings(wo[:, c], a.dt, PERIODS) for c in range(3)])
                psa_p.append([nigam_jennings(wp[:, c], a.dt, PERIODS) for c in range(3)])
            print(f"  seed {seed}: pga_orig={pk_o[-1]}  pga_pcg={pk_p[-1]}", flush=True)

    pk_o, pk_p = np.array(pk_o), np.array(pk_p)
    fas_o, fas_p = np.array(fas_o), np.array(fas_p)
    comps = ["090", "000", "ver"]

    print(f"\n=== RNG swap: gfortran intrinsic vs PCG32, {a.seeds} seeds ===")
    print("\nPeak acceleration, geometric-mean ratio (orig/pcg32), 95% CI:")
    for c in range(3):
        g, lo, hi = gm_ratio(pk_o[:, c], pk_p[:, c])
        flag = "" if lo <= 1.0 <= hi else "   <-- CI excludes 1"
        print(f"  {comps[c]}: {g:6.4f}  [{lo:6.4f}, {hi:6.4f}]{flag}")

    print("\nFourier amplitude, geometric-mean ratio per band:")
    hdr = "  band (Hz)      " + "".join(f"{c:>22}" for c in comps)
    print(hdr)
    for b in range(len(BAND_EDGES) - 1):
        row = f"  {BAND_EDGES[b]:5.2f}-{BAND_EDGES[b+1]:<6.2f} "
        for c in range(3):
            g, lo, hi = gm_ratio(fas_o[:, c, b], fas_p[:, c, b])
            row += f"  {g:6.4f} [{lo:5.3f},{hi:5.3f}]"
        print(row)

    if a.psa:
        psa_o, psa_p = np.array(psa_o), np.array(psa_p)
        print("\n5%-damped PSA, geometric-mean ratio per period:")
        for k, T in enumerate(PERIODS):
            row = f"  T={T:5.2f}s "
            for c in range(3):
                g, lo, hi = gm_ratio(psa_o[:, c, k], psa_p[:, c, k])
                row += f"  {g:6.4f} [{lo:5.3f},{hi:5.3f}]"
            print(row)

    print("\nInterpretation: ratios near 1 with CIs straddling 1, and no trend\n"
          "across frequency, means the two generators drive the same process.\n"
          "A flat offset would indicate wrong variance or a broken unit-RMS\n"
          "renormalisation; a tilt would indicate the wrong number of deviates\n"
          "being consumed somewhere in the spectral loop.")


if __name__ == "__main__":
    main()
