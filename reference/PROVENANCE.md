# Provenance of the vendored Fortran reference

`hb_high_ref.f` is a vendored copy of `HFStoch/V6.0/hb_high_v6.0.3.f` from EMOD3D:

| | |
| --- | --- |
| Upstream repo | `git@github.com:ucgmsim/EMOD3D` (local clone `~/src/EMOD3D`) |
| Commit | `51ed6b570255de75060b84780a38a549c808f30d` |
| Commit date | 2026-07-31 |
| Commit subject | `fail more often` |
| Source path | `HFStoch/V6.0/hb_high_v6.0.3.f` (4144 lines) |
| Headers | `params.h`, `params_no_window.h` (both vendored unchanged) |

EMOD3D itself is **not modified** by this project. This directory is the only
copy of the Fortran we touch.

## Target build configuration

```
-DBINMOD -DVERSION1
```

`VERSION1` corresponds to EMOD3D's `HF_TIME_WINDOW=OFF`, confirmed as the
configuration actually built for production. Note this is *not* the EMOD3D CMake
default (`HFStoch/CMakeLists.txt` defaults `HF_TIME_WINDOW` to `ON`).

Under `VERSION1` the main program includes `params_no_window.h`
(`mm=mmv=262144`) while every subroutine unconditionally includes `params.h`
(`mm=32769, mmv=180000`). This mismatch is benign for the live subprogram set —
those routines use only `nq`/`np`/`nlaymax`/`lv`, which are identical in both
headers — but it determines the alias target of the out-of-bounds `stdd(0,l)`
read at line 1394 of the original.

## Deliberate divergences from the production build

Exactly two, both required to make bit-identity achievable at all. Each is
marked in `hb_high_ref.f` with a `cREF` comment.

### 1. `random_number` → PCG32

The original seeds gfortran's intrinsic generator via
`random_seed(size=n)` / `random_seed(put=...)`. Two problems:

- The generator and the seed-array length `n` are gfortran-version-specific, so
  the random stream is not reproducible in any other language.
- `init_random_seed` **mutates its argument** (`irand = irand + 1`, `n` times),
  and the mutated value is read at line 1366 (`if(irand.gt.0)`) where it gates
  the rupture-time jitter. So `n` — a compiler constant — changes which branch
  the program takes.

PCG32 is hand-written identically in Fortran (`pcg32.f`) and Rust
(`crates/hb_high/src/rng.rs`) so the stream is portable and auditable
line-by-line. `SEED_WORDS` in `pcg32.f` fixes `n`, which makes the line-1366
branch a documented choice rather than a compiler artefact.

### 2. `FAST` forced to the radix-2 implementation

Production builds `USE_FFTW=ON`. That path is dropped here because it plans with
`FFTW_MEASURE`, which selects a plan by timing it — so the production binary is
not reliably bit-reproducible even against itself.

The two `FAST` implementations use **opposite Fourier sign conventions**
(`fftw3.f` has `FFTW_FORWARD=-1`, and the wrapper maps `IND==1` to the forward
plan, whereas the radix-2 kernel's `IND=-1` is the analysis transform). Every
spectral operation between the forward and inverse transform is either
multiplication by a real factor or an explicit conjugate-symmetric mirror, so
the flip is expected to cancel and leave the real part of the output unchanged
to within rounding. `harness/ab_fft.sh` measures this rather than assuming it.

## Reference build flags

```
gfortran -DBINMOD -DVERSION1 -cpp -ffixed-line-length-none \
         -O0 -fno-fast-math -ffp-contract=off
```

`-O0` and `-ffp-contract=off` are part of the contract: production builds `-O2`
with FMA contraction *enabled* (EMOD3D root `CMakeLists.txt:37-68`), which
reassociates float arithmetic. Goldens are defined against the flags above and
nothing else.
