#!/usr/bin/env python3
"""Compare two hb_high binary outputs.

Output format is headerless: ``ndata * 3`` interleaved ``float32``
(component order 090, 000, ver), exactly as ``hf_sim.py`` reads it with
``np.fromfile(...).reshape((-1, 3))``.

Default mode is **bit-identical**, which is the Phase 1/2 gate. Passing
``--max-rel``/``--min-corr`` switches to the tolerance mode used only for the
one-time Phase 0c A/B legs.

Exit status is 0 on pass, 1 on fail, so this can be used directly as a gate.
"""

import argparse
import sys

import numpy as np


def load(path):
    a = np.fromfile(path, dtype=np.float32)
    if a.size % 3:
        sys.exit(f"{path}: {a.size} floats is not a multiple of 3")
    return a.reshape(-1, 3)


def ulp_distance(a, b):
    """Signed-magnitude to biased-integer, so subtraction counts representable
    floats between the two values."""
    ai = a.view(np.int32).astype(np.int64)
    bi = b.view(np.int32).astype(np.int64)
    ai = np.where(ai < 0, np.int64(-(2**31)) - ai, ai)
    bi = np.where(bi < 0, np.int64(-(2**31)) - bi, bi)
    return np.abs(ai - bi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--max-rel", type=float, default=None,
                    help="tolerance mode: max |diff| / waveform peak")
    ap.add_argument("--min-corr", type=float, default=None,
                    help="tolerance mode: minimum per-component correlation")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    if a.shape != b.shape:
        sys.exit(f"FAIL shape mismatch: {a.shape} vs {b.shape}")

    identical = bool((a.view(np.uint32) == b.view(np.uint32)).all())
    peak = float(np.abs(a).max())
    diff = np.abs(a - b)
    max_abs = float(diff.max())
    max_rel = max_abs / peak if peak else 0.0
    ulp = ulp_distance(a, b)
    corr = [float(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(3)]

    if not args.quiet:
        print(f"samples          : {a.shape[0]} x 3")
        print(f"finite           : {np.isfinite(a).all()} / {np.isfinite(b).all()}")
        print(f"peak |a|         : {peak:.6e}")
        print(f"bit-identical    : {identical}")
        print(f"max abs diff     : {max_abs:.6e}")
        print(f"max rel to peak  : {max_rel:.3e}")
        print(f"ulp max/mean/med : {ulp.max()} / {ulp.mean():.2f} / {np.median(ulp):.0f}")
        print(f"differing samples: {(ulp > 0).mean():.4f}")
        print(f"correlation      : {', '.join(f'{c:.15f}' for c in corr)}")

    # Bit-identical mode: the Phase 1/2 gate.
    if args.max_rel is None and args.min_corr is None:
        if identical:
            print("PASS bit-identical")
            return 0
        # Locate the first divergence to make the failure actionable.
        bad = np.argwhere(ulp > 0)
        i, c = bad[0]
        print(f"FAIL not bit-identical; first divergence at sample {i}, "
              f"component {c}: {a[i, c]!r} vs {b[i, c]!r} ({ulp[i, c]} ulp)")
        return 1

    # Tolerance mode: Phase 0c A/B legs only.
    ok = True
    if args.max_rel is not None and max_rel > args.max_rel:
        print(f"FAIL max rel {max_rel:.3e} > {args.max_rel:.3e}")
        ok = False
    if args.min_corr is not None and min(corr) < args.min_corr:
        print(f"FAIL min corr {min(corr):.15f} < {args.min_corr}")
        ok = False
    print("PASS within tolerance" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
