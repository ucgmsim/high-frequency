# Papers behind the models

The PDFs here are **not committed** (`.gitignore`d — ~11 MB of publisher material does not
belong in git history). This file is committed, and it records which paper substantiates which
claim in the code, and how far each claim has actually been checked.

**The rule: nothing is written into the code as a citation until the equation has been read in
the paper.** Where a claim is unverified it says so. A citation I cannot substantiate is worse
than none, because it looks like authority.

Method note: `pdftotext -layout` makes these greppable, which is how the equation-by-equation
matching below was done rather than by reading page images. Where OCR was ambiguous the
rendered page was read directly — see finding 4, which OCR got wrong in a way that would have
produced a false bug report.

## The lineage, in the authors' own words

Graves & Pitarka (2010, p. 2100) state the provenance of the whole high-frequency method:

> The high-frequency portion of the simulation methodology has its roots in the pioneering work
> of Brune (1970) and Hanks and McGuire (1981), with the formal simulation approach for point
> sources first developed by Boore (1983) and the extension to finite-faults given by Frankel
> (1995), Beresnev and Atkinson (1997), and Hartzell et al. (1999).

That sentence is the spine of `PHYSICS.md`. **This crate is the high-frequency module of
Graves & Pitarka**, and `stoc.rs` is Boore (1983) eq. 1–11 with Frankel's finite-fault factor.

## Held

| file | citation | source | status |
|---|---|---|---|
| `Graves_Pitarka_2010_hybrid_broadband_BSSA100.pdf` | Graves, R. W. & Pitarka, A. (2010). Broadband ground-motion simulation using a hybrid approach. *BSSA* **100**(5A), 2095–2123. doi:10.1785/0120100057 | user (paywalled) | **verified** — eq. 10–17, p. 2100 |
| `Graves_Pitarka_2015_refinements_SRL86.pdf` | Graves, R. & Pitarka, A. (2015). Refinements to the Graves and Pitarka (2010) broadband ground-motion simulation method. *SRL* **86**(1). doi:10.1785/0220140101 | [OSTI](https://www.osti.gov/servlets/purl/1409997) (open, LLNL-JRNL-741227) | **verified** — eq. 1–3 |
| `Boore_1983_stochastic_simulation_BSSA73.pdf` | Boore, D. M. (1983). Stochastic simulation of high-frequency ground motions based on seismological models of the radiated spectra. *BSSA* **73**(6A), 1865–1894. | daveboore.com (open) | **verified** — eq. 1–11, p. 1867–1869 |
| `Anderson_Hough_1984_kappa_BSSA74.pdf` | Anderson, J. G. & Hough, S. E. (1984). A model for the shape of the Fourier amplitude spectrum of acceleration at high frequencies. *BSSA* **74**(5), 1969–1993. | user (paywalled) | **verified** — abstract, p. 1969 |
| `Saragoni_Hart_1974_artificial_earthquakes_EESD2.pdf` | Saragoni, G. R. & Hart, G. C. (1974). Simulation of artificial earthquakes. *Earthq. Eng. Struct. Dyn.* **2**, 249–267. | user (paywalled) | identity verified p. 249; the parameterisation actually implemented is Boore (1983) eq. 7–11 |
| `Boore_2003_stochastic_method_PAG160.pdf` | Boore, D. M. (2003). Simulation of ground motion using the stochastic method. *Pure appl. geophys.* **160**, 635–676. | daveboore.com (open) | identity verified p. 635; useful as the modern review, not as the source of any specific line |
| `Boore_Thompson_2014_path_durations_BSSA104.pdf` | Boore, D. M. & Thompson, E. M. (2014). Path durations for use in the stochastic-method simulation of ground motions. *BSSA* **104**(5), 2541–2552. doi:10.1785/0120140058 | daveboore.com (open) | **verified** — Table 1, p. 2546: all six breakpoints exact, and the paper's linear interpolation is what the code does. Also confirms finding 7 |
| `Boore_DiAlessandro_Abrahamson_2014_double_corner_BSSA104.pdf` | Boore, D. M., Di Alessandro, C. & Abrahamson, N. A. (2014). A generalization of the double-corner-frequency source spectral model… *BSSA* **104**(5), 2387–2398. | daveboore.com (open) | **verified** — eq. 4–7, p. 2388. Used to *rule out* a double-corner reading; see finding 4 |
| `Boore_2015.pdf` | Boore, D. M. & Thompson, E. M. (2015). Revisions to some parameters used in stochastic-method simulations of ground motion. *BSSA* **105**(2A), 1029–1041. doi:10.1785/0120140281 | user | **verified** — Table 3, p. 1034: all eight breakpoints exact, and the 0.111 s/km tail slope agrees with the code to 0.05%. See finding 7 |
| `Brune_1970_tectonic_stress_spectra_JGR75.pdf` | Brune, J. N. (1970). Tectonic stress and the spectra of seismic shear waves from earthquakes. *JGR* **75**(26), 4997–5009. | user (paywalled) | cited via Boore (1983) eq. 3 and 5; **Brune's own equations not read** |

## Not needed after all

**Frankel (1995)**, BSSA 85(4):1144–1160 — the one paper I asked for. No longer required:
Graves & Pitarka (2010) eq. 12 states the factor and attributes it, which is all the code needs
to cite. Left in the record in case someone wants the primary source.

## The verified mapping

`stoc.rs`'s spectral shape is **Boore (1983) eq. (1)**, term for term:

| Boore (1983) | code | notes |
|---|---|---|
| eq. 1 `A(ω)=C·M₀·S(ω,ω_c)·P(ω,ω_m)·e^{−ωR/2Qβ}/R` | the `as_` loop | whole spectrum |
| eq. 2 `C = R_θφ·FS·PRTITN/(4πρβ³)` | `cc` | `rp=0.63`, `fs=2.0`, `prtitn=0.71` (paper: "taken as 1/√2") |
| eq. 3 `S = ω²/(1+(ω/ω_c)²)` | `a1` | "Following Aki (1967) and Brune (1970)" |
| eq. 5 `f_c = 4.9×10⁶ β(Δσ/M₀)^{1/3}` | — | the Brune relation, not computed here |
| eq. 7 `w(t)=a·t^b·e^{−ct}·H(t)` | the envelope | Saragoni & Hart (1974) window |
| eq. 8 `b = −ε·ln η/[1+ε(ln ε−1)]` | `b` | **character-for-character** |
| eq. 9 `c = b/(ε·T_w)` | `c` | **character-for-character** |
| eq. 11 `a = [(2c)^{2b+1}/Γ(2b+1)]^{1/2}` | `aa` | "unit squared area" — **this is why `gamma` exists** |
| ε=0.2, η=0.05 | `tw_eps`, `tw_eta` | Boore's own values, p. 1869 |

and the finite-fault assembly is **Graves & Pitarka (2010) eq. 10–17**:

| G&P (2010) | code | notes |
|---|---|---|
| eq. 10 `A_i(f)=Σ_j C_ij S_i(f) G_ij(f) P(f)` | the `rayset` loop in `sim.rs` | sum over direct/Moho rays |
| eq. 11 `C_ij = F_s·RP_ij/(4πρ_iβ_i³)` | `cc` + `radiation.rs` | see finding 3 |
| eq. 12 `S_i(f)=m_i·F·f²[1+F(f/f_ci)²]^{−1}` | `a1 * frank` | **exact**; see finding 4 |
| eq. 13 `f_ci = c₀V_Ri/(α_τ·π·dl)` | `fce` | see finding 5 |
| eq. 14 `G_ij = (I_i/r_ij)·exp(−πf^{1−x}Σt_ijk/q_k)` | `a3`, `qbar`, `1/R` | frequency-dependent Q |
| eq. 15 `q_k = a + b·β_k` | `qbar = R/(β·150)` | i.e. `a=0`, `b=150` |
| eq. 16 `P(f) = exp(−πκf)` | the `kappa > 0` branch | Anderson & Hough (1984) |
| eq. 17 `T_di = f_ci^{−1} + c₁R_i`, `c₁=0.063` | `PathDurationModel` default | **exact** |

### Boore & Thompson (2014), Table 1 — path-duration model 11

| paper | code | status |
|---|---|---|
| breakpoints `R` = 0, 7, 45, 125, 175, 270 km | `path_duration_table` | **exact** |
| durations `D_P` = 0, 2.4, 8.4, 10.9, 17.4, 34.2 s | same | **exact** |
| "linear interpolation of the tabulated values" | `from_breakpoints` differences consecutive pairs | **exact** |
| "slope of last segment 0.156" s/km | copies the final segment's 0.177 | **DEVIATES** — see finding 7 |

### Boore & Thompson (2015), Table 3 — path-duration model 12

| paper | code | status |
|---|---|---|
| breakpoints `R_PS` = 0, 15, 35, 50, 125, 200, 392, 600 km | `path_duration_table` | **exact** |
| durations `D_P` = 0, 2.6, 17.5, 25.1, 25.1, 28.5, 46.0, 69.1 s | same | **exact** |
| "linear interpolation … not logarithms of these quantities" | `from_breakpoints` | **exact** |
| `D_P(R) = D_P(R_last) + 0.111(R − R_last)` | copies the final segment's 0.111058 | **agrees to 0.05%** |

Model 12 does *not* have model 11's problem: here the paper's tail slope and the last
tabulated segment's slope are the same number, so repeating the segment is right by
coincidence rather than by construction.

## Findings

### 1. `f_max` is Hanks, and the code's `κ ≤ 0` branch is not Boore's filter

Anderson & Hough (1984, p. 1969) credit the second corner frequency to **Hanks (1979, 1982)**;
Boore (1983, p. 1868) adds that Papageorgiou & Aki (1983) attribute it to *source* processes and
Hanks (1982) to *site* attenuation — a genuine open question worth stating, not hiding.

More concretely: Boore (1983) eq. 4 is `P(ω,ω_m) = [1+(ω/ω_m)^{2s}]^{−1/2}` with `s=4`. The
code's `κ ≤ 0` branch is `1/(1 + ω/ω_m)` — **a single-pole high-cut, not Boore's eight-pole
form.** The comment must not imply otherwise. Production uses `κ = 0.045`, so this branch is
exercised only by the tier-4 golden's negative-κ case.

### 2. κ is verified exactly

Anderson & Hough's abstract gives the decay as `e^{−πκf}`; the code computes
`exp(-π·(f·κ + …))`. Direct match, and G&P (2010) eq. 16 uses the same form and citation.

### 3. The 0.63 and 0.71 in `cc` are cancelled later, and that is the point

`cc` carries `rp = 0.63` (average radiation pattern) and `prtitn = 0.71`, per Boore (1983)
eq. 2. But `radiate_and_invert` divides by `RADIATION_NORM * PARTITION_FACTOR = 0.63 * 0.71`
after multiplying by the conically averaged pattern from `radiation.rs`. **The constants cancel
exactly**, so the net radiation term is the conical average — which is precisely G&P (2010)
eq. 11's `RP_ij`, "a conically averaged radiation pattern term spanning a range of ±45° in slip
mechanism and take-off angle".

`radiation.rs` draws its perturbations over `9·range·pu` with `range = 10`, i.e. 90° full width
= **±45°**. Exact match to G&P. No current comment explains the cancellation, and it is the
non-obvious fact a reader most needs.

### 4. The "Frankel two-corner operator" — attribution right, description wrong

**I was wrong to doubt the attribution.** G&P (2010) eq. 12 states
`S_i(f) = m_i F f²[1+F(f/f_ci)²]^{−1}` and says `F = M_o/(Nσ_p dl³)` is "a factor introduced by
**Frankel (1995)**, which scales the subfault corner frequency to that of the mainshock and
ensures the total moment of the summed subfaults is the same as the mainshock moment". Frankel
(1995) is correct.

*(`pdftotext` rendered eq. 12's `[1+F(f/f_ci)²]` as `[1+(Ff/f_ci)²]` — an `F` versus `F²`
difference that would have made me report a spurious factor-of-`F` bug. The rendered page
settled it. Trust page images over OCR for equations.)*

But "two-corner operator" **misdescribes** it. With `S = moment_scale`, `x = (f/f_c)²`:

```
frank = S(1+x)/(1+Sx),   a1 ∝ M₀f²/(1+x)
a1·frank ∝ (S·M₀)·f²/(1 + Sx)        ← the (1+x) cancels exactly
```

so the result is a *single*-corner spectrum of moment `S·M₀` and corner `f_c/√S` — exactly what
G&P describe in words. A reader told "two-corner" will look for the sag between two corners of
Boore, Di Alessandro & Abrahamson (2014) eq. 4 and not find one. The code should say **moment
and corner rescaling after Frankel (1995), via G&P (2010) eq. 12**.

**Open, and needing a domain judgement:** G&P define `F` with `N` (linear in subfault count);
`sim.rs` uses `√N` — and carries two dead alternatives (`by_count`, `by_two_thirds`) alongside
the live `by_sqrt_count`. G&P 2015 does not revise `F`. So the `√N` is a local deviation from
the published method, not something either paper licenses. It is a *physics* choice and I have
left it alone; the code will state the deviation plainly rather than imply G&P sanction it.

Noted and not acted on: the cancellation above is exact in real arithmetic but is not performed
in the code, so simplifying it would move the last bits — a numerical change needing its own
adjudication, not a comment pass.

### 5. The code is GP14.3-era, which explains a puzzle the old comment left hanging

`config.rs` says: *"`CZERO` is written 2.1 in a comment in the original and then 2.0 in code;
the code wins."* The papers explain it. **G&P (2010) uses `c₀ = 2.1`; G&P (2015) changed it to
`c₀ = 2.0`.** The Fortran comment was a stale GP2010 relic and the code had been updated to
GP14.3. Not a mystery — a version marker.

Likewise `alpha_t`. G&P (2010) eq. 9 parameterises `α_τ` on **dip only**, piecewise
(`1` above 60°, `0.82` below 45°). G&P (2015) eq. 3 makes it "a continuous function of fault
rake and dip", `α_T = 1 + F_D F_R c_α`. The code computes `1/(1 + F_D·F_R·c_α)` with
`c_α = 0.1` and divides `c₀` by it — so the code's returned value **is** the paper's `α_τ`
(≤ 1, smaller for shallow-dip thrust), and eq. 13's division is reproduced exactly. Only the
functional form differs from GP2010, in the GP14.3 direction.

*I nearly filed this as an α_τ² bug.* The reciprocal-in-a-badly-named-function plus the
division at the call site cancel; checking the sign of the physical effect (shallow thrusts get
higher corner frequency and shorter rise time — G&P 2010 p. 2099, citing Somerville 1998) is
what showed the code is right.

### 6. Attributions found for two orphaned comments

- `stoc.rs`'s `("Beresnev Northridge")` on a dead Q variant → **Beresnev & Atkinson (1997)**,
  named by G&P (2010, p. 2100) as one of the finite-fault extensions.
- `ray.rs`'s travel-time-weighted `qbar` → **Ou & Herrmann (1990)**, per G&P (2010) eq. 14's
  surrounding text.
- `site.rs`'s quarter-wavelength amplification → **Boore & Joyner (1997)**, which is what G&P
  (2010) cite for it, *not* Boore (2003).

### 7. Path-duration model 11 extrapolates past 270 km on the wrong slope

Boore & Thompson (2014) Table 1 tabulates six `(R, D_P)` breakpoints and then gives, as a
separate row, **"slope of last segment 0.156"** s/km — the rate to use beyond the last
breakpoint. It is not the slope of the final tabulated segment, which is
`(34.2 − 17.4)/(270 − 175) = 0.177` s/km.

The code uses 0.177: `from_breakpoints` repeats the last tabulated slope, reproducing the
original's `dpdr(ndur) = dpdr(ndur-1)`. So the deviation is **in the model as implemented
upstream, not introduced by this port**, and it makes durations about 13% longer than the
paper for ray paths past 270 km — a distance only the largest faults reach.

**How much it matters.** The error is `(0.177 − 0.156)(R − 270) = 0.0208(R − 270)` seconds of
path duration, and the record window is `2.12 ×` that:

| `R` (km) | `D_P` paper | `D_P` code | error | | window |
|---:|---:|---:|---:|---:|---:|
| 300 | 38.9 s | 39.5 s | +0.63 s | +1.6% | +1.3 s |
| 400 | 54.5 s | 57.2 s | +2.71 s | +5.0% | +5.7 s |
| 500 | 70.1 s | 74.9 s | +4.79 s | +6.8% | +10.2 s |
| 600 | 85.7 s | 92.6 s | +6.88 s | +8.0% | +14.6 s |

**This is the operating regime, not an edge case.** The Alpine Fault fixture is a 411 km
rupture striking 057°, and production domains run out to the lower North Island. Slant
distances to the subfaults, and the resulting error:

| station | min `R` | max `R` | subfaults > 270 km | mean `D_P` error | mean window | peak amp. |
|---|---:|---:|---:|---:|---:|---:|
| Hokitika | 30 km | 409 km | 34% | +1.4 s | +3.1 s | −1.6% |
| Christchurch | 128 km | 482 km | 55% | +2.2 s | +4.6 s | −2.1% |
| Nelson | 252 km | 659 km | 96% | +4.0 s | +8.5 s | −3.0% |
| Wellington | 347 km | 756 km | **100%** | +5.8 s | +12.4 s | −3.6% |
| Palmerston North | 465 km | 875 km | **100%** | +8.3 s | +17.6 s | −4.1% |
| Napier | 611 km | 1021 km | **100%** | +11.4 s | +24.1 s | −4.4% |

For anything past Cook Strait **every subfault is in the deviating regime**, so there is no
near-field contribution to dilute it — the whole record is affected roughly uniformly. That is
the opposite of the intuition that distant subfaults matter least, which only holds when the
station is close to part of the rupture.

The amplitude column is an estimate, not a measurement: the Saragoni–Hart envelope is
normalised to unit squared area (Boore 1983 eq. 11), so spreading the same energy over a
`k`-times-longer window scales peaks by about `1/√k`. Total energy is conserved, so **Arias
intensity should be roughly unaffected while PGA and pSA run low and significant duration runs
long.** Worth measuring rather than trusting this arithmetic.

**−4% is twice the ±2% band this project certifies at.** It does not show up in the LONG
campaign because that compares against the *Fortran*, which has the same behaviour — the port
is faithful. The deviation is between the implemented model and Boore & Thompson (2014), and it
is inherited, not introduced here.

Left alone: correcting it would move every long-path waveform, which needs a domain judgement
and a LONG-tier adjudication, not a tidy-up. The six breakpoints themselves are exact, and so
is the linear interpolation between them.

## Not worth a paper hunt

Standard results, cited inline with a canonical reference and no PDF: Box & Muller (1958);
O'Neill (2014) PCG; Steele, Lea & Flood (2014) SplitMix; Holm (1979); Karney (2013) for the
WGS84 geodesic. Aki & Richards, *Quantitative Seismology* (2nd ed.) by chapter for the
double-couple pattern (ch. 4) and generalized rays (ch. 6).

Named by G&P but not implemented here, so not cited in code: Hanks & McGuire (1981),
Hartzell et al. (1999), Joyner & Boore (1986), Somerville et al. (1999), Campbell & Bozorgnia
(2008), Walling et al. (2008) — the last three belong to the low-frequency and site modules,
which this crate does not contain.
