# The physics

This document exists so that someone with a general geophysics background — attenuation, site
response, a bit of source theory — can read this codebase without having written it. It assumes
undergraduate mathematics and no familiarity with the original Fortran.

Every equation below has been checked against the paper it is attributed to. `papers/README.md`
records the verification, paper by paper, including the two places where the code departs from
the published method.

---

## 1. What this computes

This crate is the **high-frequency (f > 1 Hz) module** of the Graves & Pitarka hybrid broadband
ground-motion method. It takes a finite-fault rupture, a 1-D velocity model and one station, and
returns three components of ground acceleration.

The authors state the lineage themselves (Graves & Pitarka 2010, p. 2100):

> The high-frequency portion of the simulation methodology has its roots in the pioneering work
> of Brune (1970) and Hanks and McGuire (1981), with the formal simulation approach for point
> sources first developed by Boore (1983) and the extension to finite-faults given by Frankel
> (1995), Beresnev and Atkinson (1997), and Hartzell et al. (1999).

So there are two papers to keep open. **Boore (1983)** gives the point-source stochastic method —
that is `stoc.rs`. **Graves & Pitarka (2010)** wraps it in a finite fault — that is `sim.rs`.

The central idea of the stochastic method is worth stating plainly, because it is unusual and it
explains the shape of the whole program:

> High-frequency ground motion looks like *filtered noise*. So rather than solving the wave
> equation, specify the **Fourier amplitude spectrum** from seismology — source, path and site
> as separate multiplicative factors — pair it with a **random phase spectrum**, and inverse
> transform. The result is one plausible accelerogram. Change the random seed and you get
> another, equally plausible one.

That is why the code is full of random number generation and why the *number* of random draws is
treated as sacred throughout: the draws are the phase spectrum.

Each of `N` subfaults contributes a spectrum, summed over `M` ray paths
(Graves & Pitarka 2010, eq. 10):

```
A_i(f) = Σ_j  C_ij · S_i(f) · G_ij(f) · P(f)
         └──┘   └────┘   └────┘  └──┘
        radiation source   path   high-cut
```

`sim.rs` walks the subfaults and rays; `stoc.rs` builds one `A_i(f)` and inverse-transforms it;
`radiation.rs` supplies `C_ij`; `site.rs` adds the site term; `ray.rs` traces the rays that give
`G_ij`.

---

## 2. The source spectrum

### Brune's ω-squared model

Far-field shear-wave displacement from a point source falls off as ω⁻² above a **corner
frequency** `f_c`. In acceleration (two derivatives, so ×ω²) that means a spectrum rising as f²
at low frequency and flat above `f_c` (Boore 1983, eq. 3, "following Aki (1967) and Brune (1970)"):

```
S(ω, ω_c) = ω² / (1 + (ω/ω_c)²)
```

In `stoc.rs` this is `a1`, with the constant of proportionality (Boore 1983, eq. 2):

```
C = R_θφ · FS · PRTITN / (4π ρ β³)
```

- `R_θφ` — the radiation pattern. §5 explains why the 0.63 in the code is a placeholder.
- `FS = 2` — free-surface amplification.
- `PRTITN = 0.71 ≈ 1/√2` — energy split between two horizontal components.
- `ρ`, `β` — density and shear-wave speed **at the subfault**, not at the station.

The corner frequency ties to seismic moment through a stress parameter (Boore 1983, eq. 5):

```
f_c = 4.9×10⁶ · β · (Δσ / M₀)^(1/3)
```

This relation is *not* evaluated in the code — the corner frequency arrives from the rupture
kinematics instead (§7) — but it is the reason a "stress drop" appears in the configuration at
all. Boore's own caution is worth repeating: Δσ "is best thought of here as simply a parameter
controlling the strength of the high-frequency radiation", not as a measured static stress drop.

### The finite-fault correction

A single subfault has too little moment to have the mainshock's corner frequency. Summing `N`
subfault spectra naively gives the wrong low-frequency level. Frankel's factor fixes both
(Graves & Pitarka 2010, eq. 12):

```
S_i(f) = m_i · F · f² · [1 + F(f/f_ci)²]⁻¹     where  F = M₀ / (N σ_p dl³)
```

G&P describe `F` as scaling "the subfault corner frequency to that of the mainshock" while
ensuring the summed moment matches. **It is worth doing the algebra, because it is not obvious
from the code.** With `S = F` and `x = (f/f_ci)²`, `stoc.rs` computes

```
a1    ∝ M₀ f² / (1 + x)                    the subfault's own spectrum
frank  = S(1 + x) / (1 + Sx)                the correction
                                            ↓  (1+x) cancels
a1 · frank ∝ (S·M₀) · f² / (1 + Sx)  =  (S·M₀) · f² / (1 + (f / (f_ci/√S))²)
```

So the product is **again a single-corner Brune spectrum**, with moment `S·M₀` and corner
frequency `f_ci/√S`. The code applies it as two factors, but the model is one spectrum. If you
went looking for a genuine *two-corner* spectrum — one with a sag between `f_a` and `f_b`, as in
Boore, Di Alessandro & Abrahamson (2014) eq. 4 — you would not find one here.

**One departure from the published method.** G&P define `F` with `N`, linear in subfault count.
`sim.rs` uses `√N`. Two other variants (`N` and a 2/3 power) survive as dead code beside it.
G&P (2015) does not revise `F`, so this is a local choice rather than something the literature
licenses — flagged rather than quietly normalised.

---

## 3. The path

Three separate effects, all in `G_ij(f)` (Graves & Pitarka 2010, eq. 14):

```
G_ij(f) = (I_i(f)/r_ij) · exp[ −π f^(1−x) Σ_k t_ijk / q_k ]
```

**Geometric spreading** is `1/r`: body waves on a spherical wavefront. In `stoc.rs` this is the
division by `distance_cm`. Note it is the *ray path length*, not the epicentral distance.

**Anelastic attenuation** is the exponential. The physical content is `exp(−ωR/2Qβ)`
(Boore 1983, eq. 1) — amplitude lost per wavelength travelled, parameterised by the quality
factor `Q`. Two refinements matter for reading the code:

- `Q` is **frequency dependent**, `Q(f) = Q₀ f^x`, which turns `ωR/2Qβ` into `π f^(1−x) · q̄`.
  The code precomputes `f^(1−x)` per frequency bin as `path_exponent`.
- `q̄` is a **travel-time weighted average** over the layers the ray actually crosses
  (Ou & Herrmann 1990), accumulated in `ray.rs`. For the straight-ray case it reduces to
  `q̄ = R/(βQ)` with `Q = 150β` — i.e. G&P eq. 15's `q_k = a + bβ_k` with `a=0, b=150`.

**Near-surface attenuation (κ)** is the last few hundred metres beneath the station, where most
of the high-frequency loss happens. Anderson & Hough (1984) showed the acceleration spectrum
decays as a clean exponential in `f` there, parameterised by a single number `κ`
(Graves & Pitarka 2010, eq. 16):

```
P(f) = exp(−π κ f)
```

Production uses `κ = 0.045 s`. In `stoc.rs` the κ term is folded into the same `exp` as the path
attenuation — one transcendental instead of two, which is an arithmetic identity, not an
approximation.

**The `κ ≤ 0` branch is a different filter.** When κ is non-positive the code instead applies
`1/(1 + f/f_max)`, a single-pole high-cut. That is *not* Boore (1983) eq. 4, which is an
eight-pole form `[1 + (f/f_max)⁸]^(−1/2)`. Production never takes this branch; only the tier-4
golden's negative-κ case exercises it.

A note on `f_max` itself: it is the frequency above which acceleration spectra fall off faster
than attenuation alone explains. Whether it is a *source* property (Papageorgiou & Aki 1983) or
a *site* property (Hanks 1982) was an open question when Boore (1983) was written, and κ is
essentially the site answer. Both parameters survive in the code because both interpretations
survive in practice.

---

## 4. The site

`site.rs` implements **quarter-wavelength site amplification** (Boore & Joyner 1997, which is
what G&P 2010 cite for it).

The idea is a neat piece of physics. A wave of frequency `f` is most sensitive to the material
within about a quarter wavelength of the surface. So: find the depth `z(f)` whose one-way S-wave
travel time equals a quarter period, average velocity and density down to it, and the
amplification is the impedance contrast between the source region and that average:

```
amplification = √( ρ_src β_src / (ρ̄ β̄) )
```

`site_amplification_factors` walks down the velocity model accumulating travel time until it
reaches a quarter period, interpolates within the layer where it stops, and returns
`½·ln(impedance ratio)` — a log amplitude, which is why `apply_site_amplification`
exponentiates. The table is built once per source layer and interpolated in `ln f`.

The layer walk is genuinely sequential: each step's travel time depends on the previous one's.
That is why it is a loop and not an array operation.

---

## 5. The radiation pattern

A double couple does not radiate equally in all directions. `radiation.rs` computes the SH and
SV coefficients for a given strike, dip, rake, azimuth and take-off angle
(Aki & Richards, *Quantitative Seismology* 2nd ed., ch. 4).

But a single subfault's theoretical pattern is too sharp to be realistic — small errors in
geometry produce large errors in amplitude, and real ruptures are not point double couples. So
G&P use a **conically averaged** pattern (2010, eq. 11): perturb strike, dip, rake, azimuth and
take-off randomly and average the result.

```
C_ij = F_s · RP_ij / (4π ρ_i β_i³)
```

where `RP_ij` is "a conically averaged radiation pattern term spanning a range of ±45° in slip
mechanism and take-off angle". `radiation.rs` draws its five perturbations over a 90° full width
— exactly ±45°.

**The non-obvious part, which no amount of reading the code reveals on its own:** `stoc.rs`'s
constant `C` carries `R_θφ = 0.63` and `PRTITN = 0.71`, but `radiate_and_invert` later divides by
`0.63 × 0.71` after multiplying in the conical pattern. **The constants cancel exactly.** Their
only job is to be cancelled, so that the conically averaged pattern stands where Boore's average
constant would have been. If you change one, change the other.

The vertical component is handled separately and draws no random numbers from the shared stream
— it reads a pre-filled table instead. That asymmetry between the horizontals and the vertical
is load-bearing for reproducibility, not an accident.

---

## 6. From spectrum to time series

This is Boore's "third method" (1983, p. 1867), and it is what makes the result an *accelerogram*
rather than a spectrum:

1. Generate band-limited white Gaussian noise.
2. Multiply by a **shaping window** so the signal is a transient of the right duration.
3. Fourier transform, normalise so the noise contributes unit spectrum on average.
4. Multiply by the target amplitude spectrum from §2–§4.
5. Inverse transform.

The window is the Saragoni & Hart (1974) envelope, which Boore (1983) eq. 7 writes as

```
w(t) = a · t^b · e^(−ct) · H(t)
```

with the shape parameters chosen so the envelope peaks at a fraction `ε` of the duration and has
decayed to a fraction `η` by the end (Boore 1983, eq. 8 and 9):

```
b = −ε ln η / [1 + ε(ln ε − 1)]
c = b / (ε T_w)
```

and normalised to unit squared area (eq. 11) — **this is the only reason the gamma function
appears anywhere in this codebase**:

```
a = [ (2c)^(2b+1) / Γ(2b+1) ]^(1/2)
```

The code uses `ε = 0.2`, `η = 0.05`, which are Boore's own values (1983, p. 1869), and G&P (2010)
confirm the same choice.

Step 3 deserves a word because it is easy to misread. Boore specifies unit average *spectral
amplitude*, achieved by choosing the noise variance. The code instead **measures** the realised
spectrum of its own noise sequence and rescales to unit average *power*. The two aim at the same
thing; the power form is what is implemented, and it is self-referential in a useful way — the
calibration is insensitive to any overall scale factor in the random number generator, because
the measurement and the correction carry it identically.

Finally a **raised-cosine taper** over the last tenth of the record, so the transient closes
smoothly rather than being truncated.

---

## 7. Assembling a rupture

`sim.rs` is the finite-fault layer: it loops over subfaults, and for each one over the requested
ray types, and for each of those over the three components.

**Subfault corner frequency** (Graves & Pitarka 2010, eq. 13; 2015, eq. 1):

```
f_ci = c₀ · V_Ri / (α_τ · π · dl)
```

`V_Ri` is the local rupture speed, `dl` the average subfault dimension, and `α_τ` a scale factor
that shortens rise times and raises corner frequencies for shallow-dipping thrusts — a real
observed trend (Somerville 1998; higher dynamic stress drop). G&P (2010) parameterised `α_τ` on
dip alone, piecewise; G&P (2015) made it a continuous function of dip *and* rake, which is what
this code implements. `c₀ = 2.0` here, which is the **2015** value; G&P (2010) used 2.1.

**Window duration** (Graves & Pitarka 2010, eq. 17):

```
T_di = f_ci⁻¹ + c₁ R_i ,   c₁ = 0.063 s/km
```

a source term plus a path term linear in distance. The code generalises the path term to a
piecewise-linear table so that the Boore & Thompson (2014, 2015) duration models can be selected
instead. Two details connect back to earlier sections: the source term is
`√F / f_ci = 1/f_c^effective`, using §2's rescaled corner rather than the raw subfault corner;
and the whole thing is multiplied by ≈2, because Boore (1983, p. 1869) sets the window length to
about twice the duration of strong shaking.

**Placement and summation.** Each subfault's contribution starts at its rupture time plus its
S-wave travel time, and is added into the station record with a weight proportional to slip.
Contributions can start before the record begins or after it ends; both are clipped.

---

## 8. Symbol glossary

The identifiers are inherited from Fortran, where six characters was the limit. This is the
translation table.

### Source

| code | symbol | meaning | units |
|---|---|---|---|
| `subevent_moment` | `σ_p·dl³` | subfault moment scale | dyn·cm |
| `moment_scale` | `F` | Frankel finite-fault factor (§2) | — |
| `stress_drop`, `sdrop` | `Δσ`, `σ_p` | Brune stress parameter | bars |
| `corner_frequency_hz`, `fce` | `f_ci` | subfault corner frequency | Hz |
| `czero` | `c₀` | corner-frequency constant (= 2.0) | — |
| `calpha` | `c_α` | `α_τ` coefficient (= 0.1) | — |
| `avg_subfault_km`, `dlm` | `dl` | average subfault dimension | km |
| `slip` | `d_i` | subfault slip → relative moment weight | cm |
| `rvf` | — | rupture-speed factor, fraction of β | — |

### Path and medium

| code | symbol | meaning | units |
|---|---|---|---|
| `shear_velocity_km_s`, `betvs` | `β` | shear-wave speed at the source | km/s |
| `density_g_cm3`, `row` | `ρ` | density at the source | g/cm³ |
| `distance_km`, `rpath` | `R`, `r_ij` | ray path length (**not** epicentral) | km |
| `qbar` | `q̄` | travel-time weighted `Σt/q` | s |
| `q_exponent`, `qfexp` | `x` | exponent in `Q(f) = Q₀f^x` | — |
| `attenuation_s`, `attenuation_p` | `Q_s`, `Q_p` | per-layer quality factors | — |
| `kappa_s`, `akapp` | `κ` | near-surface decay (= 0.045) | s |
| `fmax_hz`, `fmx` | `f_max` | high-cut corner | Hz |

### Time series

| code | symbol | meaning | units |
|---|---|---|---|
| `window_s`, `tw` | `T_w` | shaping-window length | s |
| `window_eps`, `window_eta` | `ε`, `η` | envelope shape (= 0.2, 0.05) | — |
| `np2` | — | FFT length, a power of two | samples |
| `fold_count`, `nfold` | — | positive-frequency bins, `np2/2 + 1` | — |
| `dt` | `Δt` | sample interval | s |
| `ndata` | — | output samples per component | — |
| `radiation` | `RP_ij` | conically averaged pattern, per bin | — |

---

## 9. Deliberate choices, not accidents

Things a reader might mistake for bugs.

- **The random draw count is part of the answer.** The phase spectrum *is* the random sequence,
  so changing how many numbers are drawn, or in what order, changes every waveform. Several
  loops that look inefficient exist to keep the draw sequence intact — including one bare
  `rng.next_f32()` whose value is discarded but whose *advance* is not.
- **Some float reductions must not be reassociated.** Three left-to-right `f32` sums are marked
  as such in the code. Pairwise or parallel summation would be more accurate and would change
  every waveform, so it needs its own justification rather than being tidied in.
- **±2% is a deliberate equivalence band**, roughly 0.04 of a typical ground-motion-model
  aleatory sigma. It is a scientific judgement about intensity-measure *means*, and does not
  transfer to other statistics.
- **Component order is fixed**: 090, 000, vertical. Downstream code consumes it positionally,
  and the two horizontals draw from the shared random stream while the vertical does not.
- **`nsum` is frozen at 1.** A sub-event loop was fixed to one iteration in 2004. The arithmetic
  is gone; the random draw it consumed is not (see above).
- **The `√N` in `moment_scale` departs from Graves & Pitarka**, who use `N`. Recorded in
  `papers/README.md` finding 4, awaiting a domain judgement.

---

## Reading order

If you are new to this and want the shortest path to understanding:

1. **Boore (1983)** §"The essence of the method" and eq. 1–11 — four pages, and it is the whole
   of `stoc.rs`.
2. **Graves & Pitarka (2010)** §"High-Frequency Simulation", p. 2100 — eq. 10–17, two columns,
   and it is the whole of `sim.rs`.
3. `PHYSICS.md` §2 and §5 of this document, for the two things the code does that the papers do
   not make obvious: the corner-frequency collapse, and the cancelling radiation constants.
4. `papers/README.md` for what has and has not been verified.

Boore (2003) is a good modern review of the same material if you want more context than
Boore (1983) gives; it is not the source of any particular line of this code.
