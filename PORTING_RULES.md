# Porting rules

The rulebook for transliterating `reference/hb_high_ref.f` into
`crates/hb_high`. Read this before touching any module.

**When the same class of error shows up in more than one file, fix it here and
regenerate the affected files.** Do not accumulate per-file patches — that is how
a port drifts.

Until the whole-program parity gate is green, the goal is *bit-identity*, not
good Rust. Every rule below exists because breaking it produces a silent
numerical difference rather than a compile error.

---

## 1. Literal constants are copied verbatim

The source uses truncated approximations to π in several places, and they are
load-bearing. `3.141593` differs from π by about 2 `f32` ulps, which is enough
to change the last bits of every FFT output.

| source | value | where |
| --- | --- | --- |
| `FAST` | `3.141593` | radix-2 twiddle argument |
| `stoc_f` | `3.1415926` (`PAI`) | source spectrum |
| `normal_random_number` | `6.2831853` | Box-Muller angle |
| `RADV_lin`, `RADFRQ_lin` | `3.1415926/180` (`PU`) | degrees to radians |
| `even_dist2` | `3.14159265` | geometry |
| **`highcor_f`** | **`3.14159625`** | **taper — a typo, see below** |
| `zpass` (dead) | `3.141592654` | filter prewarping |

`highcor_f:2266` is not a truncation of pi — it is `3.14159625`, with the last
digits of `3.14159265` **transposed**. Every other occurrence in the file is a
correct truncation, so this is a genuine slip, and it makes the raised-cosine
taper fall about 1.1e-6 short of a half cosine, so the final sample is not
exactly zero.

Copy it verbatim anyway. Confirmed with the gate: "fixing" it to `3.14159265`
changes `stdd` in the taper region. If it is ever worth correcting, that is a
Phase 3 re-baseline with a written justification.

**Never** substitute `std::f32::consts::PI`, `TAU`, or a "more accurate" value.
Never fold `3.1415926/180` into a single pre-divided decimal either — the
division is part of the arithmetic and rounds once.

The same applies to every other magic number: `rp=0.63`, `prtitn=0.71`,
`fs=2.0`, `radmin=1.0`, `radvh=0.7`, `range=10.`/`40.`, `1.5707965`. Copy the
digits that are there.

### 1b. An unsuffixed literal in a `real*8` context carries only `f32` precision

A Fortran literal without a `d0`/`_8` suffix is **default real**, i.e. `real*4`,
*even when it is immediately assigned to or combined with a `real*8`*. It is
rounded to `f32` first and then widened. Verified on gfortran 16.1.1:

```fortran
real*8 :: a, b
a = 0.999999      ! -> 3FEFFFFDE0000000   (the f32 value, widened)
b = 0.999999d0    ! -> 3FEFFFFDE7210BE9   (the true f64 value)
```

So in a `real*8` routine, `sini = 0.999999` must be ported as
`0.999999f32 as f64`, **not** `0.999999f64`. Writing the natural Rust literal
silently changes the value in the 30th bit.

This only bites when the decimal is not exactly representable in `f32`.
`0.0`, `1.0`, `2.0`, `0.5`, `0.25` and similar are exact and can be written
plainly. Known affected sites so far:

| routine | literal | context |
| --- | --- | --- |
| `geom_terms` | `0.999999` | `sini` clamp, `real*8` |
| `geom_terms` | `0.001` | `rsum` floor, `real*8` |
| `gf_amp_tt` | `0.02` | `hs_tol`, `real*8` |
| air-layer insert | `0.001` | `vp0(1)`, `rho0(1)`, `real*8` |
| air-layer insert | `0.0005` | `vsh0(1)`, `real*8` |

When porting any `implicit real*8` routine, check every unsuffixed literal
against this rule. The ray cluster (`cagcon`, `dtdp`, `pnot`, `trav`, `ttime`)
mostly uses explicit `d0` suffixes, but do not assume it.

## 2. Types and precision

| Fortran | Rust |
| --- | --- |
| `real*4`, implicit `a-h`/`o-z` without `implicit real*8` | `f32` |
| `real*8`, or implicit under `implicit real*8 (a-h,o-z)` | `f64` |
| `integer`, implicit `i-n` | `i32` |
| `integer*8` | `i64` |
| `complex*8` | `Complex32` (in `fort.rs`) |
| `complex*16` | `Complex64` |
| `character*256` | `String` (trimmed at the first blank — see rule 8) |

Precision is **per expression**, not per function. Annotate every literal and
every intermediate. The traps that actually occur in this program:

- The ray cluster (`gf_amp_tt`, `cagcon`, `cr`, `dtdp`, `pnot`, `trav`, `ttime`,
  `geom_terms`) is `implicit real*8 (a-h,o-z)`, so unannotated locals there are
  `f64` — but `i`–`n` names are still `i32`, because `implicit real*8 (a-h,o-z)`
  does not cover them.
- `/travel/`'s `alp`/`als` are **explicitly `real*4`** inside routines that are
  otherwise `implicit real*8`. That explicit declaration is load-bearing.
- Narrowing seams to preserve exactly, all `f64` → `f32`:
  `gf_amp_tt`'s `rp0 = p0`, `stime = t0`, `rpath = rpd`; `geom_terms`'s `qb`
  accumulator (so the attenuation sum accumulates in single precision, not
  double); `pnot`'s `a = dtdp(...)` and `t0 = t`, which silently take the real
  part of a `complex*16`.
- `/vmod/` is `f64` for `depth,thic,vp,vsh,rho` but `f32` for `qp,qs`.
- `/vmod_in/` is `f32` for `depth0,thic0,qp0` and `f64` for `vp0,vsh0,rho0`.
  Those three are `f32` *only because they are undeclared in both scopes that
  declare the block*. Do not "tidy" them.

## 3. Arrays stay 1-based and column-major

Use `Array1<T>` / `Array2<T>` from `fort.rs`, which index from 1 and store 2-D
data column-major. **Do not rewrite index arithmetic** during transliteration —
not `i-1`, not iterator chains, not slice windows. An off-by-one introduced
while "cleaning up" indexing is the single most likely way to produce a
plausible-looking wrong answer.

`stdd` is declared `stdd(mmv,3)` and is read at index 0 (see rule 7), so its
backing store must be laid out column-major for the aliasing to work.

Assumed-size dummies (`dimension x(1)`, twelve of them) become slices with an
explicit length taken from the call site. Resolve the true length before
porting the routine; do not guess.

## 4. Intrinsic shims

Implemented in `fort.rs`. Each exists because the obvious Rust equivalent is
subtly different:

| Fortran | Rust equivalent | trap |
| --- | --- | --- |
| `nint(x)` | `fort::nint` | rounds half **away from zero**; `f32::round_ties_even` rounds to even, and `f32::round` is correct but returns a float |
| `int(x)` | `fort::int_trunc` | truncates **toward zero**, so negative values truncate up. `k2` at line 1371 relies on this and can go negative |
| `mod(a,b)` | `fort::imod` / `%` | Fortran `mod` takes the sign of `a`, same as Rust `%` — but `modulo` does not. Check which one the source means |
| `sign(a,b)` | `fort::sign` | returns `|a|` with the sign of `b`; **not** `signum` |
| `amax1`/`amin1`/`max`/`min` | `f32::max` / `f32::min` | differ on NaN, which does not arise here but do not rely on that |
| `alog`, `alog10` | `f32::ln`, `f32::log10` | `f32` intrinsics, not `f64` promoted then narrowed |
| `float(i)` | `i as f32` | |
| `cabs(z)` | `fort::cabs` | Fortran computes `hypot`; do **not** use `(re*re+im*im).sqrt()` |
| `cexp(z)` | `fort::cexp` | |
| `conjg(z)` | `Complex32::conj` | |
| `real(z)` / `dimag(z)` | `.re` / `.im` | note `cr`'s local named `pi` is `dimag(p)`, **not** π |

### 4b. Constant exponents: check each one, do not generalise

`**` appears 161 times. Integer exponents (`x**2`, `x**3`) expand to repeated
multiplication and port as `x*x`, `x*x*x`. **Real** constant exponents are the
trap, because gfortran folds some and not others, and Rust's `powf` agrees with
neither folded form reliably. Measured over 200,000 values on gfortran 16.1.1 /
rustc 1.92:

| expression | agrees with | disagreement rate |
| --- | --- | --- |
| gfortran `x**(-1.0)` | `1.0/x` | **0** |
| Rust `powf(x, -1.0)` | `1.0/x` | 126 / 200000 |
| gfortran `x**0.5` | `sqrt(x)` | 108 / 200000 |
| Rust `powf(x, 0.5)` | `sqrt(x)` | 108 / 200000 |

So:

- `x**(-1.0)` → write `1.0 / x`. gfortran folds it to a division; `powf(-1.0)`
  does not match.
- `x**e` with a **variable** exponent → `x.powf(e)`. Safe: LLVM cannot fold a
  non-constant exponent, and the two libms agree — verified over 20,000 cases in
  `tests/intrinsics.rs`.
- `x**0.5` with a **constant** exponent → **cannot be expressed portably in
  Rust.** gfortran emits a real `powf` call, but LLVM rewrites
  `powf(x, 0.5)` into `sqrt(x)` at `-O2` and leaves it alone at `-O0`, so the
  same Rust source compares against different functions in the two profiles.
  If a live site ever needs this, wrap it in an `#[inline(never)]` helper to
  force the libm call, and verify in **both** profiles.

**Every constant-exponent `powf` must be justified, and there are currently
none in the port.** `stoc_f`'s only `x**0.5` fed a dead store, so it is simply
not computed.

Two bugs came out of this, both worth remembering:

1. `stoc_f`'s `a2 = (1.0+(omg/omgm)**1)**(-1.0)` written as `powf(-1.0)` gave a
   one-ulp error in exactly one frequency bin out of 1025.
2. Writing `powf(0.5)` passed in debug and **failed in release** — caught only
   because the suite runs in both profiles. This is the concrete reason that
   requirement exists.

**Never verify a `powf` mapping without `#[inline(never)]`.** An inlined check
silently tests the folded path, passes, and leaves the real libm call
unverified. Three successive attempts to characterise this behaviour gave the
wrong answer for exactly that reason — including one that produced a confident
but false "verified bit-identical" claim.

## 5. Control flow

`goto` becomes labelled `loop`/`break`/`continue`. Catalogue of what appears:

- **Computed `goto (1,2),j`** (`normal_random_number:4091`) → `match j { 1 => .., 2 => .. }`.
- **Arithmetic `IF (expr) L1,L2,L3`** (three in `DELAZ5`) → branch on
  `< 0`, `== 0`, `> 0` in that order. Renumber carefully; these are easy to get
  backwards.
- **Shared `DO` terminators.** `:876/:879` and `:1129/:1130/:1407` end two
  nested `DO`s on one statement, and `even_dist2` has a triple nest on one
  label. `goto 4` at `:1134` lands on the inner loop's terminator, so it means
  `continue` on the **inner** (`j`) loop, not the outer.
- **Backward `goto` as a loop** (the `2776`/`777` power-of-two searches, the
  `964`/`963` zero-rejection retries) → `loop { ... }` with an explicit `break`.

### Iteration order is part of the contract

The two subfault passes iterate in **opposite** order:

- `:974` (`do 893`/`do 894`) is `j` outer, `i` inner.
- `:1129` (`do 4`) is `i` outer, `j` inner.

`irandcnt` (reset at `:1128`, incremented at `:1182`) is consumed in the
`:1129` order. Swapping the loops silently reorders RNG consumption, which
changes every sample downstream while still producing plausible noise.

## 6. State: common blocks become one context struct

The live blocks are `/vmod/`, `/vmod_in/`, `/travel/`, `/rays/`, `/coff/`. They
are positionally consistent across all declarations, but the *names* differ per
routine. Canonical names to use everywhere:

| block | canonical fields | aliases in the source |
| --- | --- | --- |
| `/vmod/` | `depth, thic, vp, vsh, rho, qp, qs` | `dep/dpt`, `th`, `vs/s`, `dn/d/rh`, and `gencof` renames slots 3–5 to `c,s,d` |
| `/vmod_in/` | `depth0, thic0, vp0, vsh0, rho0, qp0, qs0, grand` | `dpt0`, `th0`, `vs0`, `rh0`, `gr` |
| `/travel/` | `alp, als, ndeep, nup` | slot 3 is `ndeep` in its only writer (`trav`) but read as `nd` by `cagcon`/`dtdp` and `ndp` by `pnot` |
| `/rays/` | `nh, nm, ndeg, nd` | consistent |
| `/coff/` | `it, nup1` | consistent |

Two collisions to keep straight:

- `/travel/`'s slot 3 (read as `nd`) is **not** `/rays/nd`. They are different
  quantities declared in the same routines. `/travel/` slot 3 is the deepest
  penetrated layer index; `/rays/nd` is the ray segment count.
- `gencof` (dead) has a dummy argument named `it` and deliberately does not
  declare `/coff/`. Do not merge them.

`/rays/` declares `nh(1,nlaymax)` and `nm(1,nlaymax)` with a degenerate leading
dimension; `ir` is hardwired to 1 everywhere. Keep the `ir` argument during
transliteration so call sites match the Fortran, and drop it in Phase 3.

Do not port `/butt/`, `/larry/`, `/lprint/`, `/rmode/`, `/RANDO/` — all dead.

## 7. Known bugs: reproduce, do not fix

Each of these is live in production. Rust's bounds checks will turn several into
panics, so each needs an explicit decision recorded here rather than an ad-hoc
one at the call site. **Default disposition: reproduce.**

| bug | location | disposition |
| --- | --- | --- |
| `stdd(0,l)` read below the array start; `stdd(0,2)` aliases `stdd(mmv,1)`, `stdd(0,1)` is out of bounds entirely. Shifts the trace one sample | `:1394-1396` | reproduce via an explicitly offset flat buffer; document the alias |
| `k2` can be negative, so `DS(l,li)` writes before the array start | `:1371` | reproduce; needs the same offset buffer treatment |
| `bet` used uninitialised in the `do 893/894` pass | `:974-983` | reproduce: initialise to the same value gfortran leaves, which at `-O0` is whatever the previous iteration left. Pin this with a kernel test before relying on it |
| `trav` zeroes only `alp(1:100)` of 500, leaking stale multipliers | `:3521` | reproduce (benign at the ~34 layers used, but do not silently widen) |
| `get_sitefacs` loop can reach `i = j0+1` | `:3020` | reproduce |
| `ksrc = j0+1` passed as a layer index | `:1145` | reproduce |
| `d10` reset to `10000.` inside the segment loop, so the stderr distance covers only the last segment | `:973` | reproduce |
| `ttime`'s `p1`/`t1` discarded by its only caller | `:3313` | keep the call; it is side-effect-free but keeping it preserves line-by-line comparability |

Fixing any of these is Phase 3 work, done as a deliberate re-baseline with a
written justification. Never silently re-baseline a golden.

## 8. Input parsing

`read(5,*)` is list-directed, which is not `split_whitespace().parse()`:

- Blank records are **skipped**, not treated as end of data. The production deck
  starts with a blank line and the first read still lands on `sdrop`.
- A single read consumes as many records as it needs to fill its item list. The
  stress-parameter read at `:385` wants three items and spans two records,
  taking `ispar_adjust` from one and `targ_mag`/`fault_area` from the next.
- Each new read **starts a fresh record**; leftovers on the previous record are
  discarded. This is why the third value on the `-1 -1 -1` line never reaches
  the program.
- `/` terminates the record, leaving remaining items unchanged.
- `r*value` is a repeat count.
- Comma and blank both separate.

`read(...,'(a256)')` is a fixed-width character read: take the whole record,
pad to 256 with blanks. Filenames are then trimmed at the **first blank**
(`index(name,' ')-1`), not at the last non-blank — so a path containing a space
truncates, exactly as the Fortran does.

## 9. Output

`open(22, form='unformatted', access='stream')` is raw bytes with **no record
markers and no header**. Write `((ds(l,i),l=1,3),i=1,ndata)` — component index
fastest, so the byte stream is `090,000,ver` per time sample, `4*3*ndata` bytes.

`status=` is unspecified, so gfortran opens `unknown`, which does **not**
truncate. Combined with the initial `FSEEK`, this lets several invocations write
disjoint byte ranges of one shared file. Preserve that: open for read-write
without truncation, seek, write.

The stderr line is `write(0,'(1x,f10.4)') c(2)` — one leading space, then
`f10.4`. `hf_sim.py` parses it with `float(stderr.strip())`, so the format must
produce exactly one parseable float per station.

## 10. Verification is not optional

A module is not ported until `cargo test` shows its kernel golden
**bit-identical** to the Fortran. "It compiles and looks right" is not a signal
in this codebase — the output is noise, so wrong answers look fine.

If a kernel cannot be made bit-identical, do **not** relax the gate. Isolate the
discrepancy to a named function, document the ulp bound and the cause, and raise
it. The likeliest causes are libm differences on `f32` transcendentals and `**`
expanding to a different multiply sequence than `powi`/`powf` (161 occurrences —
audit each for integer vs real exponent).
