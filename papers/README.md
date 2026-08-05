# Papers behind the models

The PDFs in this directory are **not committed** (`.gitignore`d — 10 MB of BSSA does not
belong in git history). This file is committed, and it is the record of *which paper
substantiates which claim in the code*, and how far each claim has actually been checked.

**The rule: nothing is written into the code as a citation until the equation has been read in
the paper.** "Verified" below means I found the expression the code implements. Where I could
not, it says so — a citation I cannot substantiate is worse than none, because it looks like
authority.

## Held

| file | citation | source | status |
|---|---|---|---|
| `Boore_2003_stochastic_method_PAG160.pdf` | Boore, D. M. (2003). Simulation of ground motion using the stochastic method. *Pure appl. geophys.* **160**, 635–676. | [daveboore.com](http://www.daveboore.com/pubs_online/boore_stochastic_method_pageoph.pdf) (open) | identity verified p. 635 |
| `Boore_1983_stochastic_simulation_BSSA73.pdf` | Boore, D. M. (1983). Stochastic simulation of high-frequency ground motions based on seismological models of the radiated spectra. *BSSA* **73**(6A), 1865–1894. | daveboore.com (open) | **equations not yet read** |
| `Anderson_Hough_1984_kappa_BSSA74.pdf` | Anderson, J. G. & Hough, S. E. (1984). A model for the shape of the Fourier amplitude spectrum of acceleration at high frequencies. *BSSA* **74**(5), 1969–1993. | user-supplied (paywalled) | **verified** |
| `Saragoni_Hart_1974_artificial_earthquakes_EESD2.pdf` | Saragoni, G. R. & Hart, G. C. (1974). Simulation of artificial earthquakes. *Earthq. Eng. Struct. Dyn.* **2**, 249–267. | user-supplied (paywalled) | identity verified p. 249 |
| `Brune_1970_tectonic_stress_spectra_JGR75.pdf` | Brune, J. N. (1970). Tectonic stress and the spectra of seismic shear waves from earthquakes. *JGR* **75**(26), 4997–5009. | user-supplied (paywalled) | **equations not yet read** |
| `Graves_Pitarka_2010_hybrid_broadband_BSSA100.pdf` | Graves, R. W. & Pitarka, A. (2010). Broadband ground-motion simulation using a hybrid approach. *BSSA* **100**(5A), 2095–2123. doi:10.1785/0120100057 | user-supplied (paywalled; no free copy — SCEC #1428 has citation only) | **equations not yet read** |
| `Boore_Thompson_2014_path_durations_BSSA104.pdf` | Boore, D. M. & Thompson, E. M. (2014). Path durations for use in the stochastic-method simulation of ground motions. *BSSA* **104**(5), 2541–2552. | daveboore.com (open) | **equations not yet read** |
| `Boore_DiAlessandro_Abrahamson_2014_double_corner_BSSA104.pdf` | Boore, D. M., Di Alessandro, C. & Abrahamson, N. A. (2014). A generalization of the double-corner-frequency source spectral model and its use in the SCEC BBP validation exercise. *BSSA* **104**(5), 2387–2398. | daveboore.com (open) | **verified** (eq. 4–7, p. 2388) |

## Still wanted (one paper)

1. **Frankel, A. (1995).** Simulating strong motions of large earthquakes using recordings of
   small earthquakes: the Loma Prieta mainshock as a test case. *BSSA* **85**(4), 1144–1160.
   Paywalled at GeoScienceWorld; no free copy found. Needed only to settle the attribution
   question below — see finding 2.

## Findings

### 1. `f_max` is Hanks, not Boore

`stoc.rs` applies a high-cut at `fmax` with no attribution. Anderson & Hough (1984, p. 1969)
credit it explicitly: *"Hanks (1979, 1982) suggests that, in general, the acceleration spectrum
is flat above the corner frequency to a second corner frequency (f_max) above which the
spectrum decays rapidly."* So the citation is **Hanks (1982)**, with Anderson & Hough (1984)
as the κ reference alongside it. Verified: A&H's abstract gives the decay as `e^{-πκf}`, which
is exactly the code's `exp(-π·f·κ)`.

### 2. The "Frankel two-corner operator" is misdescribed — and it algebraically collapses

`stoc.rs` computes

```rust
let frank = moment_scale * (fc2 + fr2) / (fc2 + moment_scale * fr2);
*shape = a1 * a2a3 * frank as f64;
```

with `a1 ∝ M₀·ω² / (1 + (f/f_c)²)` — a single-corner Brune spectrum. Writing `S = moment_scale`
and `x = (f/f_c)²`:

```
frank = S(1 + x) / (1 + Sx)
a1    ∝ M₀ f² / (1 + x)
```

so the product is

```
a1 · frank  ∝  S·M₀·f² / (1 + Sx)  =  (S·M₀)·f² / (1 + (f / (f_c/√S))²)
```

**The `(1 + x)` cancels exactly.** The net effect of `frank` is not a two-corner spectrum at
all: it replaces the subfault's single-corner spectrum with another single-corner spectrum of
moment `S·M₀` and corner frequency `f_c/√S`. Two consequences:

- **The comment is misleading.** Calling it a "two-corner operator" describes the shape of the
  multiplier, not the model. A reader following that description will look for a spectrum with
  a sag between two corners (Boore, Di Alessandro & Abrahamson 2014, eq. 4) and not find one.
- **The attribution is unresolved.** Boore (2003) refers to "the Frankel *et al.* (1996)
  model", i.e. USGS OFR 96-532 (the hazard maps), which is a different work from Frankel
  (1995). Neither is obviously the source of a `√S` corner-frequency rescaling — constant
  stress drop would give `S^{-1/3}`, not `S^{-1/2}`. Since `moment_scale` itself carries a
  `√N` (`sm / (subevent_moment · √subfault_count)` in `sim.rs`), the `√` may come from the
  subfault-summation scheme rather than from a published spectral model.

**This needs a domain judgement, not more reading by me.** The derivation above is exact and
checkable in a couple of minutes by someone who knows the literature. Until it is settled, the
code will say what the term *does* (moment and corner rescaling, with the algebra shown) and
will not name a paper.

A separate note, not acted on: because the cancellation is exact in real arithmetic but is
**not** performed in the code, computing the simplified form would change the last bits. That
would be a numerical change requiring its own adjudication, not part of a comment pass.

## Not worth a paper hunt

Standard results, cited inline with a DOI or canonical reference and no PDF needed:
Box & Muller (1958) for the normal transform; O'Neill (2014) PCG; Steele, Lea & Flood (2014)
SplitMix; Holm (1979) for the multiplicity correction; Karney (2013) for the WGS84 geodesic
(`geographiclib`). Aki & Richards, *Quantitative Seismology* (2nd ed.) is cited by chapter for
the double-couple radiation pattern (ch. 4) and generalized rays (ch. 6) — a textbook, not a
download.
