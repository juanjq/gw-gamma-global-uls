# Physics and methods of `gw-gamma-global-uls`

This document explains what the repository computes and why, stage by stage, with the equations
the code implements and their physical meaning. The [`README`](../README.md) covers installation
and the cluster/local workflow; this file covers the science.

The markdown cells of [`sim_3d_model_CTAO_paper.ipynb`](../notebooks/sim_3d_model_CTAO_paper.ipynb)
cite the sections of this file as "§n". Step numbers such as "Step 4.3" refer to the note
*Simulation guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit*.
The emission model and the viewing angle have their own documents, summarised in §9 and §10;
[`docs/README.md`](README.md) lists them all. Works are cited as author (year) and listed with
links in [§14](#14-references).

**Contents**

0. [The question](#0-the-question)
1. [Schematic of the pipeline](#1-schematic-of-the-pipeline)
2. [Notation and conventions](#2-notation-and-conventions)
3. [A: The GW localisation on the analysis grid](#3-a-the-gw-localisation-on-the-analysis-grid)
4. [B: IACT data, background and the test statistic Λ](#4-b-iact-data-background-and-the-test-statistic-λ)
5. [C: Spectral model and flux ↔ luminosity conversion](#5-c-spectral-model-and-flux--luminosity-conversion)
6. [D: Monte-Carlo realisations](#6-d-monte-carlo-realisations)
7. [E: From Λ distributions to an upper limit](#7-e-from-λ-distributions-to-an-upper-limit)
8. [F: Reporting both limits both ways](#8-f-reporting-both-limits-both-ways)
9. [The emission model](#9-the-emission-model)
10. [The viewing-angle distribution](#10-the-viewing-angle-distribution)
11. [The 3D limit with the emission model](#11-the-3d-limit-with-the-emission-model)
12. [Validation inventory](#12-validation-inventory)
13. [Historical notebooks (removed)](#13-historical-notebooks-removed)
14. [References](#14-references)

---

## 0. The question

An LVK alert localises a compact-binary merger to a region of the sky and a range of
distances. LST-1 (alone, or with MAGIC) observed part of that region and saw no significant
source. **What is the brightest very-high-energy counterpart that is still compatible with
those observations?**

A standard point-source upper limit is not well defined here, because the source position is
unknown. So is its distance, and so is its orientation. The pipeline defines **one global
test statistic** over the GW region, $\Lambda$. It calibrates $\Lambda$ by Monte Carlo, using
the real instrument response and background, and turns it into two upper limits:

| Limit | Fixed per test | Randomised per realisation | Result |
| --- | --- | --- | --- |
| **2D flux UL** | power-law amplitude $\phi_0$ | sky position in the 95% GW region (+ Poisson noise) | $\phi_0^{\rm UL}$, photon/energy flux |
| **3D luminosity UL** | band luminosity $L_0$ | sky position **and** distance from the full GW map (+ Poisson noise) | $L_0^{\rm UL}$ in erg s⁻¹ |
| **3D UL with the emission model** | trial luminosity $L_k$ of a GRB afterglow | sky position, distance **and** viewing angle (+ Poisson noise) | $L_k^{\rm UL}$ in erg s⁻¹ |

Status of each component:

| Component | Code | Status |
| --- | --- | --- |
| 2D flux UL, Γ = 2 power-law injection | `sim_3d_model_CTAO_paper.ipynb`, `gwuls/simulate.py` | working; run on S240615dg ([§8.1](#81-results-for-s240615dg)) |
| 3D band-luminosity UL, Γ = 2 power-law injection | same | working; run on S240615dg |
| 3D UL with the emission model: sky × distance × $\theta_v \mid d$, the model averaged over each run's GTIs with EBL at $z_i$, injected run by run, normalised to $L_k$ (guidelines Steps 3–8) | `sim_3d_model_CTAO_paper.ipynb` (`RUN_MODEL_3D`), `simulate.EmissionModelInjection` | working; run on S240615dg ([§11](#11-the-3d-limit-with-the-emission-model)) |
| Fixed phenomenological jet (`BENCHMARK`), off-axis by the element integral of Abe et al. (2026), 1° grid in $\theta_v$ | `setup_model_phenomenological_fixed.ipynb`, `gwuls/grb_phenomenological.py` | built and validated; **the model the 3D limit injects** ([`phenomenological_model.md`](phenomenological_model.md)) |
| `GRBModel` format, catO5 catalogue files: guidelines Steps 1, 2, 5 and 7 | `setup_model.ipynb`, `gwuls/grb_model.py` | built and validated; any `GRBModelSet` cache can be injected through `model_dir_3d` ([`setup_model.md`](setup_model.md)) |
| Stochastic phenomenological model: a fresh random event per call | `setup_model_phenomenological.ipynb` | exploratory, standalone ([`phenomenological_model.md`](phenomenological_model.md) §8) |
| Viewing-angle draw $p(\theta_v \mid d)$: guidelines Step 4.3 | `setup_angle_distribution.ipynb`, `gwuls/gw_pe.py`, `gwuls/angle_distribution.py` | built and validated; drawn per realisation by the emission-model limit ([`viewing_angle_distribution.md`](viewing_angle_distribution.md)) |

---

## 1. Schematic of the pipeline

```
          LVK alert (skymap .fits)                       LST-1(+MAGIC) DL3 (cluster only)
   PROB, DISTMU, DISTSIGMA, DISTNORM per HEALPix       events, IRFs (aeff, edisp, psf), 3D bkg
                    │                                                │
                    ▼                                                ▼
   [A] GW map on the WCS analysis grid              [B] MapDataset, ring background,
       p_b per bin, 95% mask M95,                       signed Cash TS map, Λ_obs,
       per-bin distance CDF F_b(r)                      containment factor κ
                    │                                                │
                    └──────────────►  input .pkl  ◄──────────────────┘
                     simulation dataset (real IRFs, livetime, pointings, real bkg map)
                     + estimator + p_b + M95 + κ + r_grid + cdf_table
                                          │
   [C] spectral model: PWL, Γ = 2         │
       φ0 ↔ F_ph ↔ F_E ↔ L ───────────►  [D] Monte Carlo, one realisation =
                                              inject source → Poisson-fake counts → Λ
                                              • background only (φ0 = 0): p-value, median Λ
                                              • 2D: φ0 fixed, (RA, Dec) random in M95
                                              • 3D: L0 fixed, (RA, Dec, d) random in the full map
                                              • 3D model: L_k fixed, (RA, Dec, d, θ_v) random,
                                                GRB afterglow + EBL, injected run by run  ◄──┐
                                          │                                                  │
                                         [E] f(x) = P(Λ_x > Λ*), fitted as a probit in log x │
                                              to realisations that each have their own x     │
                                              → x_UL where f = CL = 0.95, ± its MC uncertainty
                                          │                                                  │
                                         [F] both ULs as flux AND as luminosity,             │
                                              over the distance posterior → .json            │
                                                                                             │
   Run once, read by the 3D model limit ─────────────────────────────────────────────────────┘
     setup_model_phenomenological_fixed.ipynb → L(E', t'; θ_v) of the benchmark jet, 1° grid
                                                (data/models/grb_afterglow_phenomenological/benchmark/)
     setup_angle_distribution.ipynb           → p(d, θ_v) and p(θ_v | d) for one alert
                                                (data/gw_input/<alert>_theta_distribution.npz)
```

**Which file does what**

| File | Role |
| --- | --- |
| `notebooks/sim_3d_model_CTAO_paper.ipynb` | Main pipeline. Part 1 prepares data (stages A and B, input `.pkl`, now also the per-run datasets and GTIs); Part 2 runs the simulations (stages D, E and F) for the 2D and 3D limits, and the 3D limit again with the emission model injected (`RUN_MODEL_3D`). |
| `notebooks/setup_model.ipynb` | Run once. Standardises the catO5 emission-model files ([§9](#9-the-emission-model)). |
| `notebooks/setup_model_phenomenological.ipynb` | Exploratory, standalone. Draws random events from the Nava (2020) / Abe et al. (2026) recipe and compares them with the catO5 files. Not read by any other notebook. |
| `notebooks/setup_model_phenomenological_fixed.ipynb` | Run once. Builds the benchmark jet at every angle and writes `data/models/grb_afterglow_phenomenological/benchmark/` (optionally the one-sigma variants). This is the model the 3D limit injects. |
| `notebooks/setup_angle_distribution.ipynb` | Run once per alert. Standardises the PE release and builds $p(d,\theta_v)$ ([§10](#10-the-viewing-angle-distribution)). |
| `gwuls/utils.py` | HEALPix ↔ WCS, credible-region masks, per-bin distance CDFs. |
| `gwuls/simulate.py` | Spectral conversions, input validation, the TS/Λ engine, 2D and 3D samplers, the emission-model injection, the upper-limit fit (and the older bisection), reporting. Also a CLI used by the Slurm grid scan. |
| `gwuls/grb_model.py` | $L(E',t')$ per angle: rest-frame projection, interpolation, $E_{\rm iso}$ rescaling, angle interpolation, EBL. |
| `gwuls/grb_phenomenological.py` | The Nava (2020) / Abe et al. (2026) recipe, the jet structure, the off-axis element integral, `BENCHMARK`. |
| `gwuls/gw_pe.py` | GWTC PE release (GBs) → ~1 MB per-alert file of posterior samples. |
| `gwuls/angle_distribution.py` | Joint $p(d,\theta_v)$, its conditionals and marginals, and the population priors. |
| `gwuls/slurm.py` | Submits one grid point of `simulate.py` as an `sbatch` job. |
| `gwuls/plotting.py`, `gwuls/paths.py` | Figures; every directory path. |

---

## 2. Notation and conventions

| Symbol | Meaning | Units / value |
| --- | --- | --- |
| $b$ | a WCS analysis bin (pixel of the gammapy map) | 0.08° × 0.08° |
| $j$ | a HEALPix pixel of the GW skymap | — |
| $p_b$ | GW probability integrated over bin $b$ | — |
| $\mathcal{M}_{95}$ | bins whose centre lies inside the 95% GW credible contour | — |
| $n_b,\ \mu_b$ | ON counts and background expectation in bin $b$, correlated over $r_c$ and summed over energy | counts |
| $\mathrm{TS}_b$ | signed Cash test statistic in bin $b$ | — |
| $\Lambda$ | $\max_{b\in\mathcal{M}_{95}}(\mathrm{TS}_b + 2\ln p_b)$ | — |
| $\phi_0,\ \Gamma,\ E_0$ | power-law amplitude, photon index, reference energy: $dN/dE = \phi_0 (E/E_0)^{-\Gamma}$ | cm⁻² s⁻¹ TeV⁻¹; Γ = 2 (positive-index convention); $E_0$ = 1 TeV |
| $[E_{\min},E_{\max}]$ | analysis band (reco energy) | 0.6–20 TeV |
| $d,\ z$ | luminosity distance, redshift (Planck18) | Mpc |
| $\theta_{JN},\ \theta_v$ | angle between total angular momentum and line of sight; viewing angle to the nearer jet axis | deg |
| $E', t'$ | rest-frame energy and time since merger | GeV, s |
| CL | confidence level | 0.95 |

Primed quantities are in the source rest frame. A superscript $m$ marks values a model file was
generated at; a subscript $i$ marks values drawn in iteration $i$.

---

## 3. A: The GW localisation on the analysis grid

*Code: `utils.py`; notebook Part 1, "Reading GW data" through "3D GW: distance CDF per WCS bin".*

### 3.1 What the skymap contains

The LVK skymap is a HEALPix map. Each pixel $j$ holds the probability $P_j$ that the source lies in
that pixel. It also holds three numbers describing the distance along that line of sight, using
the ansatz of Singer et al. (2016):

$$
p(d \mid j) = N_j\, d^2\, \frac{1}{\sqrt{2\pi}\,\sigma_j}\exp\!\left[-\frac{(d-\mu_j)^2}{2\sigma_j^2}\right],
\qquad (\mu_j,\sigma_j,N_j) = (\texttt{DISTMU},\texttt{DISTSIGMA},\texttt{DISTNORM}).
$$

The $d^2$ is the volume element. $\mu_j$ and $\sigma_j$ are ansatz parameters, not the mean and
standard deviation.

### 3.2 From HEALPix to the WCS analysis grid

The gammapy analysis uses a WCS map: an AIR projection centred on `source_coord`, 3.575° wide,
with 0.08° bins. Each HEALPix pixel is assigned to the bin that contains it, and the bin probability is
the mean pixel probability scaled to the bin's solid angle:

$$
p_b = \Big(\tfrac{1}{n_b}\sum_{j\in b} P_j\Big)\,\frac{\Omega_b}{\Omega_{\rm pix}}.
$$

This corrects for the whole number of HEALPix pixels that happen to fall in each bin.
$\sum_b p_b$ is the fraction of the GW probability inside the analysis field, printed as
"Geometry covers X% of the GW". Before taking logarithms, $p_b$ is floored at $10^{-20}$.

### 3.3 The 95% credible region

HEALPix pixels are sorted by probability and accumulated until 95% is reached. The probability of
the last pixel added is the contour level. The map is resampled onto a plate-carrée grid and
contoured at that level. A WCS bin belongs to $\mathcal{M}_{95}$ if its centre falls inside the
contour. $\Lambda$ is only searched inside $\mathcal{M}_{95}$.

### 3.4 Distance along each WCS bin

A WCS bin contains many HEALPix pixels, each with its own distance ansatz. The distance
distribution of the bin is their probability-weighted mixture:

$$
F_b(r) = \sum_{j\in b} w_j\, F_j(r), \qquad w_j = \frac{P_j}{\sum_{k\in b}P_k},
$$

where $F_j$ is the CDF of $p(d\mid j)$ (`ligo.skymap.distance.marginal_cdf`). This gives
`cdf_table[b, :]` on `r_grid`: 1000 linear points from 1 to 10 000 Mpc. Pixels with no distance
information ($\mu_j=\infty$) are skipped. `cdf_valid[b]` records whether a bin has a usable CDF.
The table is cached under `data/gw_input/` with a hash of its inputs.

The grid starts at **1 Mpc, not 0**. The 3D injector converts luminosity to flux through
$1/d^2$, so a single draw at $d=0$ would inject infinite flux and dominate every $\Lambda$
distribution.

The sky-marginal distance prior over the field of view is

$$
F(r) = \sum_b \tilde p_b\,F_b(r), \qquad \tilde p_b = \frac{p_b\,\mathbf{1}[\text{valid}_b]}{\sum_{b'} p_{b'}\,\mathbf{1}[\text{valid}_{b'}]}
$$

(`simulate.marginal_distance_pdf`). It is computed once and reused for every flux ↔ luminosity
translation in the report ([§8](#8-f-reporting-both-limits-both-ways)).

### 3.5 Dirac-delta mode

With `USE_DIRAC_DELTA = True`, all the probability is put in the hottest pixel. $\Lambda$ then
reduces to the TS at a single known position, with no trials factor, so the Monte-Carlo limit
should reproduce gammapy's ordinary per-pixel UL. This is the closure test run in the historical
`delta*.ipynb` notebooks ([§13](#13-historical-notebooks-removed)).

---

## 4. B: IACT data, background and the test statistic Λ

*Code: notebook Part 1, "Reading the DL3 data" through "Computing Λ for the real data";
`simulate._TSEngine`.*

### 4.1 Dataset and background

- **Data.** DL3 runs of S240615dg (LST-1 + MAGIC "stereo" production, `obs_ids` 17821–17825).
  Each run carries its IRFs (effective area, energy dispersion, PSF) and a 3D background model
  from `pybkgmodel` or `baccmod`.
- **Geometry.** Reco energy 0.6–20 TeV at 4.5 bins per decade; true energy 0.05–100 TeV; safe mask
  offset < 2.5°.
- **Software.** Datasets, IRF folding and estimators are from Gammapy (Donath et al. 2023).
- **Background.** Ring method (Berge, Funk & Hinton 2007; `RingBackgroundMaker`): for each position, OFF counts are taken from
  a ring with inner radius 0.3° and width 0.2°. The expected background is
  $\mu_{\rm bkg} = \alpha\,n_{\rm off}$, where $\alpha$ is the ratio of ON to OFF acceptance,
  whose spatial shape comes from the 3D background model.

### 4.2 Per-bin test statistic

Counts are summed over energy and correlated with a top-hat of radius $r_c = 0.1°$. This gives,
for each bin, an "ON region" with $n_b$ counts and a background expectation $\mu_b$. With the
Cash (1979) Poisson statistic $C(n,\mu) = 2(\mu - n\ln\mu)$, the likelihood ratio between
"background plus the best-fit excess" (saturated, $\mu=n$) and "background only" ($\mu=\mu_b$) is

$$
\mathrm{TS}_b = \operatorname{sign}(n_b-\mu_b)\cdot\max\!\Big[0,\;C(n_b,\mu_b)-C(n_b,n_b)\Big]
= \operatorname{sign}(n_b-\mu_b)\cdot 2\Big[n_b\ln\frac{n_b}{\mu_b}-(n_b-\mu_b)\Big].
$$

By Wilks' theorem (Wilks 1938), $\sqrt{|\mathrm{TS}_b|}$ is approximately the local significance of an excess
($+$) or deficit ($-$). The background is treated as known ($\mu_b$ fixed), both in the real data
and in the simulations, so the two are computed identically.

### 4.3 GW weighting and the global statistic Λ

$$
\mathrm{TS}'_b = \mathrm{TS}_b + 2\ln p_b, \qquad
\Lambda = \max_{b\in\mathcal{M}_{95}} \mathrm{TS}'_b .
$$

**Physical reading.** $\mathrm{TS}_b = 2\ln(\mathcal{L}_1/\mathcal{L}_0)$, so
$\mathrm{TS}'_b = 2\ln\!\big[(\mathcal{L}_1/\mathcal{L}_0)\cdot p_b\big]$. $\Lambda$ picks the
position where "the gamma-ray data prefer a source" times "the GW data say the source is here"
is largest. A modest excess where the GW probability is high can beat a stronger excess where it
is low. Weighting by the prior tempers the look-elsewhere effect of scanning the whole region.
The price is that an excess in a low-probability bin must overcome a penalty: it wins only if

$$
\mathrm{TS}_b > \Lambda^\star - 2\ln p_b .
$$

Details that matter:

- $\Lambda$ is usually **negative**: $p_b\ll1$, so $2\ln p_b$ is a large negative number.
- $p_b$ depends on the bin size, but that shifts every $\mathrm{TS}'_b$ by the same constant.
  Observed and simulated $\Lambda$ share the grid, so the constant cancels in every comparison.
- Taking a maximum is a *profile* over position. Summing $e^{\mathrm{TS}'_b/2}$ over bins would
  instead be a position-marginalised likelihood ratio (a Bayes factor). The pipeline uses the
  maximum.

### 4.4 Observed significance

$N_{\rm bkg}=5000$ realisations with no injected source ([§6.2](#62-background-only-realisations))
give the null distribution of $\Lambda$:

$$
p = P(\Lambda_{\rm bkg} > \Lambda_{\rm obs}), \qquad S = \Phi^{-1}(1-p), \qquad
\sigma_p = \sqrt{p(1-p)/N_{\rm bkg}} .
$$

Each background realisation takes the same maximum over the same region, so $p$ is already
**post-trials**. If $p$ saturates at 0 or 1, the significance is only bounded, at
$|S| > \Phi^{-1}(1-1/N_{\rm bkg})$.

### 4.5 The simulation dataset

The dataset written to the `.pkl` (`dataset_stacked_simulated`) is built from the **real** runs:
same IRFs, livetimes, pointings and reference times, with its background map set to the **real
ring background** of each run. A simulated observation is therefore

$$
n^{\rm sim}_{\rm pix,E} \sim \operatorname{Poisson}\!\big(\mu^{\rm bkg}_{\rm pix,E} + \mu^{\rm sig}_{\rm pix,E}(\phi_0, \text{position})\big),
$$

with $\mu^{\rm sig}$ the source model folded through exposure, PSF and energy dispersion
(`dataset.fake`). After faking, the model is removed, so the injected source is not treated as
known background, and $\Lambda$ is computed exactly as for the real data.

### 4.6 Containment factor and sky-map ULs

`ExcessMapEstimator` measures flux inside the top-hat of radius $r_c$, which misses part of a
PSF-spread point source. The notebook injects a bright source ($\phi_0=10^{-8}$) at the field
centre 100 times and measures the flux-recovery ratio

$$
\kappa = \operatorname{median}\!\left[F_{\rm est}/F_{\rm true}\right].
$$

Per-pixel sky-map ULs are divided by $\kappa$. These per-pixel ULs (95% CL, `n_sigma_ul = Φ⁻¹(0.95)`)
carry **no trials factor**. They are used as a sanity reference in [§7.7](#77-trials-factor-sanity-check).

---

## 5. C: Spectral model and flux ↔ luminosity conversion

*Code: `simulate.py` Section 1.*

### 5.1 Band integrals

For $dN/dE = \phi_0 (E/E_0)^{-\Gamma}$ and $x = E/E_0$:

$$
F_{\rm ph} = \int_{E_{\min}}^{E_{\max}}\frac{dN}{dE}dE = \phi_0E_0\,I_1,\quad I_1=\int_{x_1}^{x_2}x^{-\Gamma}dx;
\qquad
F_E = \int_{E_{\min}}^{E_{\max}}E\frac{dN}{dE}dE = \phi_0E_0^2\,I_2,\quad I_2=\int_{x_1}^{x_2}x^{1-\Gamma}dx.
$$

**Γ = 2 is the singular case of $I_2$.** The generic form $[x^{2-\Gamma}]/(2-\Gamma)$ gives 0/0
there. The limit is a logarithm, and the code uses it explicitly:

$$
F_E\big|_{\Gamma=2} = \phi_0 E_0^2 \ln\frac{E_{\max}}{E_{\min}}, \qquad
F_{\rm ph}\big|_{\Gamma=2} = \phi_0E_0^2\left(\frac1{E_{\min}}-\frac1{E_{\max}}\right).
$$

A Γ = 2 spectrum carries equal energy per decade, which is why $F_E$ grows only logarithmically
with the band width. The reported SED point is $E^2\,dN/dE$ at the logarithmic band centre
$E_{\rm dec}=\sqrt{E_{\min}E_{\max}}$.

### 5.2 Luminosity

$$
L_{[E_{\min},E_{\max}]} = 4\pi d^2\,F_E\,(1+z)^{\Gamma-2}.
$$

- $L$ is the **isotropic-equivalent luminosity in the analysis band**. It is not bolometric: no
  emission is extrapolated outside the band the data constrain.
- $(1+z)^{\Gamma-2}$ is the power-law **K-correction**. The observed band $[E_{\min},E_{\max}]$
  was emitted in $(1+z)[E_{\min},E_{\max}]$, and for a power law
  $\int E\,dN/dE\,dE$ over a band scaled by $a$ changes by $a^{2-\Gamma}$. At Γ = 2 it is exactly
  1, so band luminosities do not depend on $z$ at this index. It is off by default and only
  matters if the index changes.
- $z(d)$ uses Planck18 through a cached interpolation table.

The inverse, used by the 3D injector, is

$$
\phi_0(L_0,d) = \frac{L_0}{4\pi d^2\,(1+z)^{\Gamma-2}\;E_0^2\,I_2}.
$$

### 5.3 Worked example: 0.6–20 TeV, Γ = 2, $\phi_0 = 10^{-12}$ cm⁻² s⁻¹ TeV⁻¹

Computed with the repository code:

| Quantity | Value |
| --- | --- |
| photon flux $F_{\rm ph}$ | 1.617 × 10⁻¹² cm⁻² s⁻¹ |
| energy flux $F_E$ | 5.618 × 10⁻¹² erg cm⁻² s⁻¹ ($=\phi_0E_0^2\ln 33.3$) |
| $E^2dN/dE$ at $E_{\rm dec}$ = 3.46 TeV | 1.602 × 10⁻¹² erg cm⁻² s⁻¹ |
| $L$ at $d$ = 1123 / 1566 / 1849 Mpc ($z$ = 0.22 / 0.29 / 0.34) | 0.85 / 1.65 / 2.30 × 10⁴⁵ erg s⁻¹ |

The three distances are the 5/50/95% points of the S240615dg PE posterior.

### 5.4 What the luminosity does *not* include: EBL

The injected power law is the spectrum **arriving at Earth**. No extragalactic-background-light
absorption is applied, so $L_0$ is the luminosity of the *observed* (absorbed) spectrum
projected back with $4\pi d^2$, not the intrinsic luminosity. For S240615dg this is a large
effect. At $z\approx0.29$, with the EBL model of Domínguez et al. (2011):

| Energy | 0.3 TeV | 0.6 TeV | 1 TeV | 2 TeV | 5 TeV | 10 TeV |
| --- | --- | --- | --- | --- | --- | --- |
| optical depth $\tau$ | 0.9 | 2.3 | 3.6 | 4.8 | 7.5 | 15.5 |

Only $e^{-2.3}\approx10\%$ of the intrinsic flux survives at the bottom of the band. An intrinsic
luminosity limit needs EBL absorption at the drawn $z_i$ (guidelines Step 7). The emission-model
limit applies it ([§11](#11-the-3d-limit-with-the-emission-model)).

---

## 6. D: Monte-Carlo realisations

*Code: `simulate.py` Sections 3–5.*

### 6.1 One realisation

Common to every mode:

1. Place a point source with spectrum $\phi_0(E/E_0)^{-2}$ at a position (§6.3 or §6.4).
2. Poisson-sample the counts: `dataset.fake(seed)`.
3. Remove the model, compute the $\mathrm{TS}$ map and $\mathrm{TS}'$ map, and record $\Lambda$
   and its position.

### 6.2 Background-only realisations

$\phi_0 = 0$ and $N = 5000$. This gives the null distribution of $\Lambda$, the p-value and
significance ([§4.4](#44-observed-significance)), and the median $\Lambda_{\rm bkg}$ used as the
reference when the data under-fluctuate. A split-half comparison of medians checks that the
distribution is stable.

### 6.3 2D sampler (fixed flux)

- $\phi_0$ is fixed.
- The **position** is a HEALPix pixel $j$ drawn with probability $\propto P_j$, restricted to
  pixels whose nearest WCS bin is in $\mathcal{M}_{95}$ and renormalised. The source is injected
  at the HEALPix pixel centre.

The 2D limit is therefore **conditional on the source lying in the observed part of the 95%
region**. The coverage (`frac_gw_in_mask`, "Mask covers X% of the all-sky GW probability") should
be quoted with it.

### 6.4 3D sampler (fixed luminosity)

- $L_0$ is fixed.
- **Bin:** $b \sim \tilde p_b$ over bins with a valid distance CDF, over the **full** map by
  default. With `restrict_to_mask_3d = True` it is limited to $\mathcal{M}_{95}$.
- **Distance:** $d = F_b^{-1}(u)$ with $u\sim U(0,1)$, by inverse CDF with linear interpolation
  on `r_grid`.
- **Amplitude:** $\phi_0 = \phi_0(L_0, d)$ from §5.2. A round-trip $L\to\phi_0\to L$ on 64 draws is
  checked to machine precision.
- The source is injected at the **bin centre**.

Each realisation is one hypothetical universe: the counterpart is at a GW-allowed place and
distance, has luminosity $L_0$, and shows the flux that implies. The 3D limit is conditional on
the source lying inside the analysis field, since the sampler renormalises over it.

### 6.5 Common random numbers and seeding

Realisation $i$ always uses fake seed `base_seed + i`, and positions or distances come from a
fixed seed (`12345`). Every tested $\phi_0$ or $L_0$ therefore sees **the same** background
fluctuations and the same positions and distances. Only the brightness changes between tests.
The fraction curve $f(x)$ of §7 is then smooth and almost monotonic in $x$, instead of jittering
by Monte-Carlo noise from one bisection step to the next.

Because seeds follow the global realisation index, parallel runs (`N_JOBS`) produce exactly the
same $\Lambda$ sample as sequential ones. The closure tests (§7.6) use their own seeds
(900 000 for the background, 987 654 for sky, distance and angle), so they are independent of
the realisations that set the limit.

The upper-limit fit of §7.3 needs the opposite: its likelihood treats every realisation as an
independent draw, so no two may share a background seed or a position. Realisation $i$ of a
limit uses seed `seed_fit + i` (`seed_fit` = 1 000 000, clear of the background-only block
`0 … n_sim_bkg - 1`, which enters the same fit), and round $r$ draws its positions or
distances with the seed `[12345, r]`.

---

## 7. E: From Λ distributions to an upper limit

*Code: `simulate._fit_ul`, `run_fit_ul`, `run_fit_ul_3d` (default, §7.3); `simulate._bisect_ul`,
`run_iterative_ul`, `run_iterative_ul_3d` (alternative, §7.4).*

### 7.1 Definition

Let $x$ be the injected strength ($\phi_0$ in 2D, $L_0$ in 3D). Define

$$
f(x) = P\big(\Lambda_x > \Lambda^\star\big), \qquad
x_{\rm UL}: \quad f(x_{\rm UL}) = \mathrm{CL} = 0.95,
$$

where the reference is $\Lambda^\star = \Lambda_{\rm obs}$ if $S\ge0$, and
$\Lambda^\star = \operatorname{median}(\Lambda_{\rm bkg})$ if $S<0$.

**Meaning.** A counterpart at $x_{\rm UL}$ would have produced a larger $\Lambda$ than the one
observed in 95% of GW-consistent realisations, so brighter counterparts are excluded at 95% CL.
This is a Neyman-type construction on the global statistic, and it includes the trials factor
of the region automatically.

When the data under-fluctuate ($S<0$), $\Lambda^\star$ is raised to the background median.
The limit then cannot be tighter than the median sensitivity, the same idea as a
power-constrained limit (Cowan et al. 2011). A downward fluctuation of the background cannot produce an
artificially strong constraint.

### 7.2 What sets the value of the limit

**2D.** $f(\phi_0)$ is a probability-weighted average over positions. Positions differ in
exposure (offset from the pointings), background level, and GW penalty $-2\ln p_b$ ([§4.3](#43-gw-weighting-and-the-global-statistic-λ)).
Reaching $f=0.95$ requires detecting the source at almost all GW-probable positions, so the 2D
limit is driven by the least favourable ~5% of the probability: lowest exposure, or highest
penalty.

**3D.** $f_{\rm 3D}(L_0) = \mathbb{E}_{b,d}\big[f_{\rm pos}(\phi_0(L_0,d))\big]$. Suppose the
single-position detection probability were a step at flux $F^\star$. Then
$f_{\rm 3D}(L_0) = P\big(d < \sqrt{L_0/4\pi F^\star}\big)$, and the 95% point is

$$
L_0^{\rm UL}\approx 4\pi\,d_{95}^2\,F^\star .
$$

The 3D limit is governed by the **far tail** of the distance posterior, not its median: the
counterpart must be detectable even when it is at the far end of the allowed distances. A smooth
detection curve pushes the limit up further. The report compares against $L$ at the 5/50/95%
distances, and the 95% one is the natural comparison point.

The notebook printout frames the 2D/3D gap through $\langle 1/d^2\rangle$ and says the nearest
realisations dominate. That is true for the *mean injected flux*, but a 95% exclusion fraction is
limited by the farthest draws.

**Coverage cap (3D, full-map sampling).** A source drawn outside $\mathcal{M}_{95}$ essentially
cannot raise $\Lambda$, which is only searched inside the mask. So

$$
f_{\rm 3D}(L_0)\lesssim P(\mathcal{M}_{95}\mid\text{field}) .
$$

If that probability is below 0.95, $f$ never reaches CL at any $L_0$ and no 95% limit exists.
The bracket check then reports "still below CL at the upper bracket". The notebook prints both
support fractions next to each other for this reason. If it is above 0.95 but below 1, $f$
flattens out below 1 and the limit sits where it is still climbing towards that ceiling. The
fit of §7.3 measures the ceiling as its parameter $c$.

### 7.3 Fit of the detection probability (default, `UL_METHOD = "fit"`)

*Code: `simulate._fit_ul`, `run_fit_ul`, `run_fit_ul_3d`. Figure: `plotting.plot_ul_fit`.*

The bisection simulates $N$ realisations at one $x$, keeps only whether their fraction is above or
below CL, and moves on; every step waits for the previous one. Here every realisation $i$ gets
its **own** injected strength $x_i$, together with its own position (and distance, and viewing
angle), and records $y_i = \mathbf 1[\Lambda_i > \Lambda^\star]$. The $y_i$ are independent Bernoulli
draws with probability $f(x_i)$, modelled as

$$
f(x) = f_0 + (c - f_0)\,\Phi\!\left(z_0 + \frac{\log_{10} x - \log_{10} x_{\rm UL}}{s}\right),
\qquad z_0 = \Phi^{-1}\!\left(\frac{\mathrm{CL} - f_0}{c - f_0}\right).
$$

$f(x_{\rm UL}) = \mathrm{CL}$ holds by construction, so the limit is itself a fit parameter. The other
three are nuisance parameters:

| parameter | meaning | constrained by |
| --- | --- | --- |
| $f_0$ | $f$ with no signal | the background-only sample ($k$ of $N_{\rm bkg}$ above $\Lambda^\star$), in the same likelihood |
| $c$ | $f$ for an arbitrarily bright source: 1, or less under the coverage cap of §7.2 | realisations above the transition |
| $s$ | width of the transition, in dex | realisations across it |

$$
\ln L = \sum_i \Big[y_i \ln f(x_i) + (1-y_i)\ln\big(1-f(x_i)\big)\Big] + k\ln f_0 + (N_{\rm bkg}-k)\ln(1-f_0) .
$$

**Uncertainty.** The profile likelihood of $\log_{10}x_{\rm UL}$, with $s$, $f_0$ and $c$ re-fitted at
every point, gives the 1σ interval where $-2\Delta\ln L = 1$ and the 2σ interval where it is 4.
This is the Monte-Carlo uncertainty of the limit, from the finite number of realisations. It is
not a systematic uncertainty. Every flux quantity scales with $\phi_0$ and every 3D quantity with
$L_0$, so each shares the relative interval of its limit. The intervals go into the summary table,
the figures, and the JSON export (`*_1sigma`, `*_2sigma`).

**Rounds.**

1. *Scout*: `n_scout` = 500 realisations, stratified log-uniformly over the bracket. The fit must
   put the crossing inside the bracket, and the top and bottom 5% of realisations must lie on
   either side of CL; otherwise the run stops with the same kind of bracket error as the
   bisection.
2. *Refine*: up to `n_refine` = 1000 realisations where the linear predictor runs from $-2.5$ to
   $z_0 + 1.5$, i.e. $f$ from just above $f_0$ to 99.9% of the way to $c$, plus 10% just above that
   range to measure $c$. The fit uses every realisation from the lower edge of this window up.
   Below it, $f \simeq f_0$ carries little information, $f_0$ is pinned by the background sample,
   and the curve is least probit-like.
3. *Stop* when the 1σ half-width is at most `precision`, or after `max_sims` = 6000
   realisations or `max_rounds` = 6 rounds. Each round's size follows from the $1/\sqrt N$ scaling
   of the current half-width. The function default is 0.03 dex (7%). The notebook asks for
   `ul_precision` = 0.01 dex, which needs about 9× more realisations than 0.03, so with the
   6000 budget it usually stops on the budget at about ±0.03 dex and reports
   `converged = False` (§8.1). That flag means only that the requested precision was not reached:
   the quoted interval is still the true MC uncertainty.

Three guards cover sparse or unlucky data. A fit to few realisations can collapse into a
near-step, with the misses above it absorbed by $c<1$ and the hits below by $f_0$, and claim a
precision the data cannot give. So a transition narrower than three point spacings never counts
as converged, and the next window is never narrower than ten spacings; it then either resolves
the transition or confirms it is genuinely steep. If a fit puts the crossing outside its window,
it is re-located with every realisation, since the scout spans the whole bracket. Every simulated round is cached (`cache_path`) and reused on a rerun: each realisation is
an independent $(x_i, \Lambda_i)$ pair, so the rounds of an interrupted run remain valid data.

**Goodness of fit.** A Hosmer–Lemeshow test (Hosmer & Lemeshow 1980) compares observed and expected detections in
equal-count groups. The probit is an approximation, and with thousands of realisations the test
can see shape differences that do not bias the limit (see the single-position curve below). A
small p-value is a prompt to look at the pulls near $x_{\rm UL}$ and at the closure test (§7.6),
not a failure by itself.

**Validation.** Synthetic detection curves with a known limit, 200 independent fits each at the
default settings, run through `_fit_ul` with the same rounds, guards and stopping rule. None of
the curves is a probit in $\log x$, so the table also measures the cost of the shape
approximation. Pull = $(\log_{10}x_{\rm UL} - \log_{10}x_{\rm true})$ / 1σ half-width.

| curve $f(x)$ | mimics | 1σ coverage | 2σ coverage | mean pull | pull s.d. | realisations per fit |
| --- | --- | --- | --- | --- | --- | --- |
| $\Phi(x/x_0)$ | one position, significance ∝ flux | 0.75 | 0.99 | −0.16 ± 0.06 | 0.83 | 2600 |
| $\langle\Phi(3xA/x_0)\rangle_A$, $A$ log-normal (σ = 0.5) | 2D: exposure varies over positions | 0.72 | 0.99 | −0.08 ± 0.06 | 0.87 | 3650 |
| $\langle\Phi(x/d^2)\rangle_d$, $d$ log-normal (σ = 0.35) | 3D: distance spread | 0.66 | 0.96 | −0.06 ± 0.07 | 1.05 | 4940 |
| the same, 1% of draws undetectable ($c$ = 0.995) | 3D coverage cap | 0.65 | 0.93 | +0.48 ± 0.07 | 0.99 | 5440 |
| the same, 3% of draws undetectable ($c$ = 0.985) | 3D coverage cap | 0.56 | 0.88 | +0.62 ± 0.15 | 2.18 | 5480 |

Nominal: 0.683 and 0.954 coverage, pull 0 with s.d. 1. For curves that reach 1 the limit is
unbiased to within 0.16σ, and its intervals cover at or slightly above the nominal rate.

**Curves that saturate below 1 are the weak spot.** Their limit sits where the curve is still
climbing towards $c$, which is where the probit's shape matters most. There the limit comes out
about 0.5σ high, which is conservative, and the 1σ interval covers only 56–65%, with occasional
outliers (pull s.d. 2.2 at $c$ = 0.985). The ceiling itself is measured well: the fitted $c$
averages 0.987 for a true 0.985. The run prints a note whenever $c$ is significantly below 1.
In that case treat the interval as a lower bound on the MC uncertainty and rely on the closure
test (§7.6), or set `restrict_to_mask_3d = True`, which removes the cap.

Rejected variants: without the ceiling $c$, the two capped curves came out +1.05σ and +6.3σ high
with 1σ coverage 0.40 and 0.01; a window starting at $\eta = -1.5$ left a −0.28σ bias on the
single-position curve.

**Diagnostics** (`plotting.plot_ul_fit`, six panels): (a) every realisation's $\Lambda$ against its
own $x$, coloured by round, with $\Lambda^\star$; (b) the binned fraction above $\Lambda^\star$ over the
whole scanned range, with the fit, its 1σ/2σ band, $f_0$ and $c$; (c) the same around the limit, with
the limit and its 1σ/2σ intervals; (d) the likelihood profile of $x_{\rm UL}$ against its Gaussian
(Hessian) approximation; (e) the estimate and its 1σ after each round, against the requested
precision; (f) the goodness-of-fit pulls.

### 7.4 Bisection (alternative, `UL_METHOD = "bisect"`)

Bisection works in $\log x$ between `amp_lo, amp_hi` = $10^{-13}, 10^{-10}$ (2D) or
`lum_lo, lum_hi` = $10^{42}, 10^{50}$ erg s⁻¹ (3D):

1. **Bracket check:** evaluate $f$ at both ends. Warn loudly if the crossing is outside.
2. **Iterate:** $x_{\rm mid} = \sqrt{x_{\rm lo}x_{\rm hi}}$. Simulate $N$ realisations and set
   $f = \frac1N\sum\mathbf{1}[\Lambda>\Lambda^\star]$. If $f>\mathrm{CL}$ then $x_{\rm hi}=x_{\rm mid}$,
   otherwise $x_{\rm lo}=x_{\rm mid}$.
3. **Stop** when both conditions hold (or after `max_iter` = 20 steps):

$$
\log_{10}\frac{x_{\rm hi}}{x_{\rm lo}} < \texttt{precision}\ (0.02\ \text{dex}),
\qquad
|f-\mathrm{CL}| < \max\!\big(\texttt{frac\_tol},\;2\sigma_f\big),\quad \sigma_f=\sqrt{\tfrac{f(1-f)}{N}} .
$$

The $2\sigma_f$ term exists because at $N=1000$ and $f=0.95$, $\sigma_f = 0.0069$. A tolerance of
0.005 can never be met by noise alone.

4. **Final value.** The crossing is interpolated linearly in $(\log_{10}x, f)$ between the
   evaluated points that straddle CL, using every point of the bisection:

$$
\log_{10}x_{\rm UL} = \log_{10}x_i + \frac{\mathrm{CL}-f_i}{s},\qquad s=\frac{f_{i+1}-f_i}{\log_{10}x_{i+1}-\log_{10}x_i},
\qquad \sigma_{\log_{10}x_{\rm UL}} = \frac{\bar\sigma_f}{|s|}.
$$

   This gives a 1σ Monte-Carlo uncertainty on the limit.

5. **Diagnostics:** converged or not, limit at a bracket edge, and the number of monotonicity
   violations larger than 3σ in the sorted $f(x)$ curve.

Speed-ups that do not change the statistic: `N_JOBS` parallelises the $N$ realisations of one
step, and `USE_DYNAMIC_N_SIM` uses few realisations while the bracket is wide, growing
geometrically to $N$. Early steps only need the sign of $f-\mathrm{CL}$. Every $\Lambda$ sample is
cached by $x$ on disk.

### 7.5 Grid-scan alternative

With `USE_ITERATIVE_ULS = False`, the notebook submits one Slurm job per amplitude on a 150-point
log grid (`slurm.submit_simulation_job` → `python gwuls/simulate.py ... --mode 2d|3d`). It then
applies the same crossing interpolation to the gridded $f$. Older notebooks fitted a sigmoid
$f(\log x) = b + (L-b)/(1+e^{-k(\log x - x_0)})$ instead (`utils.sigmoid`); that fit is no longer
used for the quoted number.

### 7.6 Closure test (3D)

$x_{\rm UL}$ is read off a fitted curve (or interpolated between bisection steps), so it was
never simulated itself. The notebook runs $N$ fresh realisations exactly at $L_0^{\rm UL}$ with
independent seeds and checks (`simulate.closure_pull`)

$$
\text{pull} = \frac{f_{\rm closure}-\mathrm{CL}}{\sqrt{\sigma_f^2 + \sigma_{\rm UL}^2}},\qquad |\text{pull}|<3 ,
$$

where $\sigma_f$ is the binomial error of the closure sample and $\sigma_{\rm UL}$ the limit's own
MC uncertainty mapped through the fitted curve, $\tfrac12|f(x_{\rm UL}^{+1\sigma}) - f(x_{\rm UL}^{-1\sigma})|$.
At $N = 1000$ the two are comparable, so leaving out $\sigma_{\rm UL}$ would fail good limits
about as often as it passes them. For a bisection result $\sigma_{\rm UL} = 0$.

### 7.7 Trials-factor sanity check

The global 2D limit applies to the *maximum* over the region, so it pays the trials factor. The
per-pixel sky-map ULs do not. The global limit should therefore sit **at or above** the largest
per-pixel UL inside $\mathcal{M}_{95}$. A global limit below it points to a problem with $\kappa$
or with mismatched spectral assumptions.

This check currently **fails** for S240615dg: the global limit is 17% below the largest per-pixel
UL in the 95% region (§8.1). The comparison is not like for like, though. The global limit averages
over GW-weighted positions, whereas the per-pixel maximum is taken at the single least favourable
pixel. So a global limit below the maximum does not necessarily mean a bug. Before the limit is
quoted, check $\kappa$ and compare against a GW-weighted quantile of the per-pixel ULs instead of
their maximum.

---

## 8. F: Reporting both limits both ways

*Code: `simulate.summarize_upper_limits`, `luminosity_ul_from_flux_ul`, `flux_ul_from_luminosity_ul`.*

The 2D method measures a flux and needs a distance to become a luminosity. The 3D method
measures a luminosity and needs a distance to become a flux. For S240615dg the 5% and 95% points
of the distance posterior differ by a factor of about 1.7 (PE and skymap alike), which is about
3 in $L\propto d^2$. Each limit is therefore reported as a **quantile range** over the FoV-marginal
$F(r)$ of §3.4, with $d_q = F^{-1}(q)$ for $q$ = 5, 50, 95%:

$$
\text{2D}\to L:\quad L_q = 4\pi d_q^2\,F_E^{\rm UL}\ \ (\text{also at } d_{\rm rms}=\sqrt{\langle d^2\rangle});
\qquad
\text{3D}\to F:\quad \phi_{0,q} = \phi_0(L_0^{\rm UL}, d_q).
$$

The consistency block prints $L_0^{\rm 3D}/L^{\rm 2D}(d_{50})$ and whether the 3D limit falls inside
the 2D quantile range. Expected reasons for a gap:

1. **Sky support:** full map in 3D versus the mask in 2D (§7.2, coverage cap). Setting
   `restrict_to_mask_3d = True` removes this difference.
2. **Distance marginalisation:** far-tail dominance (§7.2).
3. Mismatched index, band, $\Lambda^\star$ or CL, which would be a bug.

Every quoted number is written to `outputs/results/<source_name>/<stem>_upper_limits.json`. This
includes Λ, the significance, each limit with its 1σ/2σ MC interval, convergence, goodness of fit,
and the settings (for the model limit also the full `EmissionModelInjection` configuration and the
merger time).

### 8.1 Results for S240615dg

These numbers come from the last run of the notebook: LST-1+MAGIC stereo, runs 17821–17825,
`pybkgmodel` background, 0.6–20 TeV, Γ = 2, 95% CL. The GTIs fall 15.4–17.0 h after the merger.

| Quantity | Value |
| --- | --- |
| $\Lambda_{\rm obs}$ / background median | −4.52 / −4.07 |
| p-value, significance | 0.592 ± 0.007, **−0.23σ**: an under-fluctuation, so $\Lambda^\star$ = background median (§7.1) |
| GW probability in the field / in $\mathcal{M}_{95}$ | 98.9% / 95.0% |
| Containment factor $\kappa$ | 0.682 |
| Distance, FoV-marginal skymap posterior (5/50/95%) | 1028 / 1422 / 1817 Mpc |
| **2D flux UL** | $\phi_0 = 2.34\times10^{-12}$ cm⁻² s⁻¹ TeV⁻¹ (−2.9%/+3.2% MC, 1σ); photon flux $3.78\times10^{-12}$ cm⁻² s⁻¹; energy flux $1.31\times10^{-11}$ erg cm⁻² s⁻¹ |
| 2D as a luminosity at 5/50/95% distance | 1.66 / 3.18 / 5.19 × 10⁴⁵ erg s⁻¹ |
| **3D luminosity UL** (power law, no EBL) | $L_0 = 3.85\times10^{45}$ erg s⁻¹ (−5.0%/+5.5%); closure pull +1.36σ, pass |
| 3D / 2D at the median distance | 1.21: the 3D limit is looser and lies inside the 2D range |
| **3D UL with the emission model** (`gti_mean`, Franceschini & Rodighiero 2017 EBL) | $L_k = 1.33\times10^{47}$ erg s⁻¹ (−6.4%/+7.1%); closure pull +0.11σ, pass |
| Model / power-law 3D limit | 34.6 |

All three limits used 5500 realisations and stopped on the budget (`converged = False`, §7.3).
The emission-model limit is 35× above the power-law one. The ratio combines everything the model
changes: EBL absorption at the drawn $z$ (5–95%: 0.20–0.34), the softer index (2.2), the
rest-frame band, and the light curve across the runs (§11). Its fitted
ceiling is $c = 0.995$, slightly below 1, so its 1σ interval may undercover (§7.3).

---

## 9. The emission model

*Code: `grb_model.py`, `grb_phenomenological.py`. Full write-ups:
[`setup_model.md`](setup_model.md) (the format and the catO5 files) and
[`phenomenological_model.md`](phenomenological_model.md) (the jet the limit injects).
Guidelines Steps 1, 2, 5 and 7.*

### 9.1 Why

The power-law limits inject a constant Γ = 2 spectrum. A physical counterpart, such as a GRB
afterglow seen off-axis, has a spectrum and a light curve that depend on energy, time since the
merger, and viewing angle. The emission model provides this in a distance-independent form, so
that each realisation can place it at its own $(d_i, z_i, \theta_i)$.

### 9.2 The format: rest-frame luminosity per viewing angle

A model is a photon-number luminosity $L(E', t';\theta)$ [ph s⁻¹ GeV⁻¹] in the source rest frame,
tabulated per viewing angle (`GRBModel`, one `.npz` per angle in a `GRBModelSet`). It is intrinsic:
no distance and no EBL. A realisation projects it to the observer with

$$
F_i(E,t) = \frac{(1+z_i)^2}{4\pi d_i^2}\;L\big((1+z_i)E,\;t/(1+z_i);\theta_i\big)\;e^{-\tau(E,z_i)},
$$

where $(1+z)^2$ appears because $L$ counts photons, $t$ is the time since the merger, and
$\tau(E,z)$ is the EBL optical depth (Franceschini & Rodighiero 2017 in `sim_3d`, Domínguez et al.
2011 as the `grb_model` default). Cosmology is Planck18 (Planck Collaboration 2020). Between
tabulated angles, $\log L$ is interpolated linearly in θ ([`setup_model.md`](setup_model.md) §4).

### 9.3 The model that is injected: the benchmark phenomenological jet

The 3D limit injects one fixed jet built from the recipe of Nava (2020) and Abe et al. (2026), with
every input at its population median ($E_{\gamma,\mathrm{iso}} = 2\times10^{50}$ erg,
$\theta_\mathrm{core} = 14°$, $\Gamma_\mathrm{core} = 200$, decay index −1.45, photon index 2.2):

- the on-axis light curve is a broken power law peaking at the deceleration time and anchored to
  $L_{\mathrm{TeV}}$(0.3–1 TeV) at $t' = 11$ h through the $L_X$–$E_\mathrm{iso}$ relation of
  Berger (2014);
- the jet is Gaussian in energy and Lorentz factor, and is seen off-axis through the
  equal-arrival-time element integral of Abe et al. (2026) Sec. 3.4, after Lamb & Kobayashi (2017);
- it is tabulated at 0°, 1°, …, 90° in `data/models/grb_afterglow_phenomenological/benchmark/`.

Off-axis, the peak is later and much fainter: at 11 h the flux is $10^{-5}$ of the on-axis value at
50°. The spectrum keeps its shape at every angle. Details, validation (against an independent
on-axis calculation and by shape fits to all five catO5 files) and the population variants are in
[`phenomenological_model.md`](phenomenological_model.md).

### 9.4 The catO5 catalogue files

The five INAF / CTA-GW O5 files (`catO5_*.fits`, at 9.1°, 13.2°, 22.6°, 28.6° and 77.8°) are
the same recipe run on five different events. [`setup_model.md`](setup_model.md) converts them
to the same format. Because they are five events and not one jet seen from five angles, they are
first put on a common $E_\mathrm{iso}$ by the blast-wave scaling
$L\to k\,L(E', t'/k^{1/3})$, $k = E_\mathrm{iso,ref}/E_\mathrm{iso}$ (Blandford & McKee 1976;
Sari, Piran & Narayan 1998; van Eerten & MacFadyen 2012), and then interpolated at fixed
light-curve phase. They can be injected instead of the benchmark by pointing `model_dir_3d` at
their cache. The wide gap between 28.6° and 77.8° makes them a coarse model off-axis.

---

## 10. The viewing-angle distribution

*Code: `gw_pe.py`, `angle_distribution.py`; notebook `setup_angle_distribution.ipynb`. Full
write-up: [`viewing_angle_distribution.md`](viewing_angle_distribution.md). Guidelines Step 4.3.*

### 10.1 Why

Each realisation of the emission-model limit needs a viewing angle $\theta_i$ as well as
$(\mathrm{RA}_i, \mathrm{Dec}_i, d_i)$. An on-axis afterglow is far brighter and earlier than an
off-axis one (§9.3), so the angle distribution decides which light curves the limit is built from.

### 10.2 From $\theta_{JN}$ to the jet viewing angle

GW parameter estimation measures $\theta_{JN}\in[0°,180°]$, the angle between the binary's total
angular momentum and the line of sight. The jet is bipolar along it, so

$$
\theta_v = \min(\theta_{JN},\,180°-\theta_{JN}) \in [0°, 90°].
$$

### 10.3 The distance–inclination degeneracy

The GW amplitude constrains mainly $\Theta(\theta)/d$: a face-on binary far away looks like an
inclined one nearby (Finn & Chernoff 1993). Distance and angle are therefore strongly correlated
(for S240615dg, $\mathrm{corr}(d,\cos\theta_v) = 0.82$), and the angle must be drawn
**conditioned on the realisation's distance**,

$$
p(\theta\mid d_i) = \frac{p(d_i,\theta)}{p(d_i)} .
$$

For S240615dg, the median of $p(\theta_v\mid d)$ runs from 49.5° at 1123 Mpc to 13.6° at 1849 Mpc.

### 10.4 PE data

A GWTC `*-combined_PEDataRelease.hdf5` (GBs) is reduced to
`data/gw_pe/standardized/<superevent>.h5` (~1 MB, tracked in git). It keeps the same number of
samples (5000) from every waveform analysis, so the file is the equal-weight mixture that the
catalogue papers quote.

| alert | GW name | $d_L$ [Mpc], median [90%] | $\theta_v$ [deg], median [90%] | $P(\theta_v<30°)$ |
| --- | --- | --- | --- | --- |
| S240615dg | GW240615_113620 | 1566 [1123, 1849] | 24.4 [6.5, 51.1] | 0.65 |
| S241125n | GW241125_010116 | 4748 [2600, 8611] | 55.8 [15.6, 86.0] | 0.17 |

Both are binary black holes. They exercise the method; no standard GRB afterglow is expected.

### 10.5 The joint $p(d,\theta_v)$

`JointDistanceAngle.from_samples` estimates the joint with a Gaussian KDE in $(d, \cos\theta_v)$,
where an isotropic prior is flat. It uses a full-covariance (Scott's rule) kernel that follows the
correlation, with exact reflection at the boundaries, and maps the result to per-degree density
with the $\sin\theta_v$ Jacobian. Conditionals, marginals and $p(d\mid\theta_0)$ all come from
this one grid.

### 10.6 Without PE

Low-latency skymaps carry no inclination. The fallbacks are: isotropic, $p\propto\sin\theta$
(median 60°); the GW-detected population of Schutz (2011),
$p\propto\sin\theta\,[(1+\cos^2\theta)^2/4+\cos^2\theta]^{3/2}$ (median 36°); a `"selection"` model
with an explicit horizon, which also gives $p(\theta\mid d)$; and fixed placeholders. The
event's PE (median 24.7° for S240615dg) is much more face-on than either prior.

### 10.7 Skymap distance versus PE distance

`sim_3d` draws $d_i$ from the **alert skymap** (§3.4) and then $\theta_i$ from the **PE**
conditional. For S240615dg the two distance posteriors disagree: skymap 1421 [1028, 1815] Mpc,
PE 1566 [1123, 1849] Mpc. Nearer distances pick up the more inclined angles of the near tail,
which shifts the $\theta_v$ median by **+8.2°** (32.6° instead of 24.4°). The preferred route
when PE exists draws a PE sample and takes (RA, Dec, $d$, $\theta_v$) together
(`ThetaDistribution.sample_joint`). It is implemented in `angle_distribution` but not yet wired
into `sim_3d` (§11.4).

---

## 11. The 3D limit with the emission model

*Code: `simulate.EmissionModelInjection`, `run_fit_ul_3d`; notebook section "3D luminosity upper
limit with the emission model" (`RUN_MODEL_3D`).*

### 11.1 One realisation

The construction is the same as for the power-law 3D limit (§6.4), with the same Λ (§4.3), the
same fit (§7.3) and the same sky and distance seeds. Only the injected source changes:

1. draw a sky bin $b$ and a distance $d_i$ from the skymap, exactly as in §6.4;
2. draw a viewing angle $\theta_i \sim p(\theta_v\mid d_i)$ from the alert's cache (§10), on its
   own random stream, so the sky and distance draws do not depend on it;
3. take the benchmark jet at $\theta_i$ (§9.3) and project it to $(d_i, z_i)$ with EBL (§9.2);
4. for each run $j$, average the observer-frame spectrum over that run's GTIs (time since the
   merger) and fold it through run $j$'s own exposure, PSF and energy dispersion; sum the runs;
5. scale the result to the trial luminosity $L_k$ (§11.2), then Poisson-fake the counts and
   compute Λ.

**No temporal template is needed (Step 7).** The analysis is time-integrated, so each run's
counts depend only on its time-averaged spectrum. Averaging per run and injecting run by run is
therefore exact. A fading source automatically weights the early runs more.

### 11.2 What $L_k$ means (`MODEL_NORMALISATION`)

- **`"gti_mean"`** (default, guidelines Step 6): the isotropic-equivalent luminosity in the
  analysis band taken in the rest frame (Step 3), averaged over the GTIs,
  $$
  \text{injected}_i = \frac{L_k}{D_i}\,F_i,\qquad
  D_i=\frac{1}{T_{\rm obs}}\sum_j\int_{t_{1j}}^{t_{2j}}\!\int_{E'_0}^{E'_1}E'\,L\big(E',\,t/(1+z_i);\theta_i\big)\,dE'\,dt .
  $$
  Every realisation has the same $L_k$ during the observation. The model's absolute brightness
  cancels. Its **shape** reaches the limit: the photon index, EBL at $z_i$, the rest-frame band,
  and, through the run-to-run weights, the light curve. This is the direct counterpart of the
  power-law $L_0$.
- **`"anchor"`**: $L_k$ is the jet's own on-axis $L_\mathrm{TeV}$ (0.3–1 TeV) at $t' = 11$ h.
  Every realisation is the same jet scaled by one factor, so its faintness at $\theta_i$ and at
  the observation time also enter. The limit then says how bright the benchmark jet would need
  to be to be excluded, and it is set by the far-off-axis tail of $p(\theta_v)$.

The choice of angle distribution matters far more under `"anchor"`. Between 0° and 45°, the
injected flux per unit $L_k$ changes by a factor of about 3.5 with `"gti_mean"` (three synthetic
20-min runs at 2–4 h, no EBL), but by about six orders of magnitude with `"anchor"`.

### 11.3 Options and the angle distribution

| Notebook option | Default | Meaning |
| --- | --- | --- |
| `model_dir_3d` | `.../grb_afterglow_phenomenological/benchmark` | any `GRBModelSet` cache: a one-sigma variant, or the catO5 set |
| `theta_dist_path` | `data/gw_input/<alert>_theta_distribution.npz` | the alert's $p(\theta_v\mid d)$ |
| `MODEL_NORMALISATION` | `"gti_mean"` | §11.2 |
| `EBL_REFERENCE` | `"franceschini17"` | any gammapy EBL model, as used by Abe et al. (2026); `"dominguez"` for Domínguez et al. (2011). The tables ship in `data/ebl/`, which `gwuls.paths` sets as `$GAMMAPY_DATA` if it is unset |
| `lum_lo_model`, `lum_hi_model` | $10^{42}$, $10^{58}$ erg s⁻¹ | first-batch span, wider than for the power law because EBL removes most of the flux above ~1 TeV |

`EmissionModelInjection.theta_distribution` also accepts the built-in specs `"isotropic"`,
`"schutz"` and `"fixed:<deg>"` (resolved by `simulate.resolve_theta_distribution`). For an
event-specific limit, use the alert's PE conditional: it is the measured inclination, and drawing
it at the same $d_i$ keeps the distance–inclination correlation (§10.3). `"isotropic"` ignores
both the measurement and the face-on selection of GW detectors, so it is a conservative bracket,
not the standard case. `"fixed"` gives the limit at a chosen angle, a curve in $\theta_v$ that
does not depend on the GW inclination. `"schutz"` is the population fallback for alerts without
PE. The notebook's preview cells load `theta_dist_path` as a saved cache, so use the library
call directly for the built-in specs. The simulated rounds are cached in `data/tmp/`. The cache
name includes the model directory, the normalisation, the EBL model, and the modification times
of the input `.pkl` and of the θ cache, so changing any of them forces a fresh run.

### 11.4 Checks and status

Checks in the notebook, before the fit:

1. **Run by run = stacked**, for a constant power law: 252.9 against 254.0 expected counts
   (−0.41%). The small difference comes from the exposure-weighted PSF and energy dispersion of
   the stacked dataset.
2. **Step 6 closure**: over 2000 draws, the rest-frame GTI-mean luminosity rebuilt from the
   injected spectra (EBL divided out) is $1.00003\,L_k$.
3. **What is injected**: the drawn angles (5/50/95%: 9.7° / 33° / 57° for S240615dg), where
   the GTIs fall on the light curve, the run weights, and the EBL transmission at the drawn $z$.

After the fit, the closure test at $L_k^{\rm UL}$ (§7.6) gives a pull of +0.11σ. It also shows
which draws set the limit: the fraction above the target per bin of $\theta_v$ and of $d$.

Still open:

- **Joint PE draw.** Drawing (RA, Dec, $d$, $\theta$) together from PE (§10.7) is not wired in.
  For S240615dg the skymap–PE distance mismatch moves the angles about 8° off-axis.
- **One fixed jet.** The variant caches can be used through `model_dir_3d`, but neither a loop
  over them nor a per-realisation bank member is implemented
  ([`phenomenological_model.md`](phenomenological_model.md) §7).
- **Per-position statistic.** The guidelines' Step 8 alternative,
  $P_{\rm excl}(L_k)=\frac1N\sum_i\mathbf{1}[\mathrm{TS}_i(L_k)>\mathrm{TS}_{\rm obs}(\mathrm{RA}_i,\mathrm{Dec}_i)]$,
  is not implemented. Λ is used for every limit.
- **Source class.** S240615dg is a binary black hole. The limit demonstrates the method, not a
  physical expectation.

---

## 12. Validation inventory

| Check | Where | Guards against | Last recorded result |
| --- | --- | --- | --- |
| Band integrals vs quadrature and gammapy; round trips; Γ = 2 limit from both sides; K = 1 at Γ = 2; $L\propto d^2$ | `simulate.check_spectral_conversions` (`--self-test`) | the Γ = 2 singularity, unit errors | passes (run in notebook) |
| Input `.pkl` validation: shapes, empty mask, κ range, CDF transposed / non-monotonic / truncated | `simulate.validate_simulation_input` | silent failures thousands of iterations later | runs at `.pkl` write and at every load |
| HEALPix → WCS assignment separation < 1.5 bin diagonals | `_map_healpix_to_wcs_bins` | RA-convention errors | printed per run |
| 3D sampler: χ² sky marginal, KS distance marginal, hottest-bin conditional, no $d\le0$, grid-edge probability atoms | `simulate.check_3d_sampling` | the production sampler drifting from the posterior | S240615dg: χ²/dof 855/849 (p = 0.43), KS p = 0.62, 0 draws at $d\le0$: pass |
| Background Λ split-half medians; binomial p-value error | notebook Part 2 | an unstable null distribution | S240615dg: −4.062 / −4.080 |
| Upper-limit fit: bracket check after the scout, goodness of fit, profile-likelihood interval | `_fit_ul`, `plotting.plot_ul_fit` | a limit outside the bracket, a curve the probit misdescribes | per run |
| Upper-limit fit: bias and coverage on synthetic curves of known limit, including non-probit shapes and a ceiling below 1 | §7.3 | a biased limit, an uncertainty that does not cover | 200 fits per curve: unbiased to 0.16σ with nominal coverage when the curve reaches 1; ~0.5σ high and under-covering when it saturates below 1 (§7.3) |
| Bisection: bracket check, MC-aware tolerance, monotonicity, edge flag | `_bisect_ul` | meaningless limits at a bracket edge | per run |
| Closure test at $L_0^{\rm UL}$ and $L_k^{\rm UL}$ with independent seeds | notebook, `closure_pull` | a biased limit | pull < 3σ required; S240615dg: +1.36σ (power law), +0.11σ (model) |
| Global UL ≥ per-pixel sky-map UL | notebook | a missing trials factor, wrong κ | S240615dg: global 17% **below** the 95%-region maximum, open (§7.7) |
| Model round trip Step 2 → Step 7 | `setup_model` | frame or $(1+z)$ power errors | max relative error 0 |
| Peak-aligned interpolation single-peaked; rescale ∘ interpolate commute | `setup_model` | double peaks, $E_{\rm iso}$ inconsistency | 0 of 398 angles multi-peaked; 1.7 × 10⁻¹³ dex |
| Standardised PE vs full raw posterior; fold vs pesummary `viewing_angle` | `setup_angle_distribution` Part 0 | subsampling bias | 5/50/95% quantiles within ~1°; fold exact |
| KDE marginals vs samples; grid $p(\theta\mid d)$ vs grid-free resampling; bandwidth ×0.5 and ×2 | Part 1 | over- or under-smoothing | grid vs resampling medians within 0.5°; bandwidth moves medians by ≤ 1.5° |
| Three sampling routes reproduce the posterior (quantiles, correlation, KS) | Part 1 | sampler bugs | KS ≤ 0.016 |
| Each option's sampler vs its own pdf | Part 2 | sampler bugs | at $1/\sqrt n$ level |
| Schutz vs Monte-Carlo selection model | Part 2 | a wrong closed form | max CDF difference 0.009 |
| `resolve_theta_distribution` for a cache path and every built-in spec; malformed `"fixed:…"` rejected at construction | §11.3, Step 4.3 | a silently wrong angle draw | medians: fixed 20°, isotropic 60.2°, Schutz 36.4°, S240615dg PE given $d$ = 1400 Mpc 33.9° |
| Emission model: run-by-run injection = stacked for a constant source; Step 6 closure of $L_k$ | §11.4 | wrong per-run folding or normalisation | −0.41%; 1.00003 |
| Benchmark jet: on-axis vs 1-D calculation, grid convergence, shape fits to catO5 | `setup_model_phenomenological_fixed` | a wrong off-axis integral | catO5 shapes to 0.09–0.25 dex ([`phenomenological_model.md`](phenomenological_model.md) §6) |

---

## 13. Historical notebooks (removed)

These predated the `gwuls` package, were not runnable as they were, and have been removed from
the tree (retrievable from git history at `notebooks/others/` in earlier commits). Kept here for
the physics context.

| Notebook | What it did | Physics purpose |
| --- | --- | --- |
| `check_1_simulation.ipynb` | Built simulated datasets from the real IRFs and background; compared real minus simulated counts, and source minus null | validate that `fake()` on the real-IRF dataset reproduces the real data |
| `delta.ipynb` / `delta_he.ipynb` | Full 2D pipeline with the GW map collapsed to its hottest pixel (Dirac delta); LST mono 0.15–0.6 TeV (`baccmod`) and LST+MAGIC stereo 0.6–20 TeV (`pybkgmodel`) | closure test: with no trials, the MC ("frequentist") UL must match gammapy's per-pixel UL |
| `results_delta.ipynb` | Hand-entered results of those runs; percentage difference vs significance, pixel size and containment radius | quantified the agreement of the two UL methods |
| `check_energy_dependence.ipynb` | Hand-entered MC UL vs sky-map-maximum UL (95% region and full map) across energy bands, mono vs stereo | size of the trials penalty per band |
| `test_sim_flux.ipynb` | Earliest 2D flux pipeline: Slurm grid, sigmoid or linear crossing | superseded by `simulate.run_iterative_ul` |

---

## 14. References

**Statistics and data analysis**

- W. Cash, *Parameter estimation in astronomy through application of the likelihood ratio*, ApJ
  **228**, 939 (1979), [doi:10.1086/156922](https://doi.org/10.1086/156922): the Poisson
  likelihood statistic of §4.2.
- S. S. Wilks, *The large-sample distribution of the likelihood ratio for testing composite
  hypotheses*, Ann. Math. Stat. **9**, 60 (1938),
  [doi:10.1214/aoms/1177732360](https://doi.org/10.1214/aoms/1177732360): TS as a significance.
- D. Berge, S. Funk and J. Hinton, *Background modelling in very-high-energy γ-ray astronomy*,
  A&A **466**, 1219 (2007), [arXiv:astro-ph/0610959](https://arxiv.org/abs/astro-ph/0610959): the
  ring background (§4.1).
- A. Donath et al., *Gammapy: a Python package for gamma-ray astronomy*, A&A **678**, A157 (2023),
  [arXiv:2308.13584](https://arxiv.org/abs/2308.13584): datasets, IRF folding, estimators.
- G. Cowan, K. Cranmer, E. Gross and O. Vitells, *Power-constrained limits*,
  [arXiv:1105.3166](https://arxiv.org/abs/1105.3166) (2011): the idea behind referencing
  under-fluctuations to the background median (§7.1).
- D. W. Hosmer and S. Lemeshow, *Goodness of fit tests for the multiple logistic regression
  model*, Commun. Stat. Theory Methods **9**, 1035 (1980): the goodness-of-fit test of §7.3.

**Gravitational waves**

- L. P. Singer et al., *Going the distance: mapping host galaxies of LIGO and Virgo sources in
  three dimensions using local cosmography and targeted follow-up*, ApJL **829**, L15 (2016),
  [arXiv:1603.07333](https://arxiv.org/abs/1603.07333): the per-pixel distance ansatz (§3.1).
- L. S. Finn and D. F. Chernoff, *Observing binary inspiral in gravitational radiation: one
  interferometer*, PRD **47**, 2198 (1993), [arXiv:gr-qc/9301003](https://arxiv.org/abs/gr-qc/9301003):
  the orientation factor Θ (§10.3).
- B. F. Schutz, *Networks of gravitational wave detectors and three figures of merit*, CQG **28**,
  125023 (2011), [arXiv:1102.5421](https://arxiv.org/abs/1102.5421): inclination distribution of
  GW-detected binaries (§10.6).
- LVK, GWTC-5.0 parameter-estimation data release (Zenodo): the PE samples of §10.4.

**Emission model and propagation**

- H. Abe et al. (CTAO Consortium), *Chasing gamma-ray signals from binary neutron star
  coalescences with the Cherenkov Telescope Array: prospects and observing strategies*, ApJ
  **1004**, 46 (2026), [arXiv:2604.08748](https://arxiv.org/abs/2604.08748): the phenomenological
  recipe and its off-axis method (§9.3).
- L. Nava, *TeV counterparts of BNS mergers*, internal note (Jan 2020): the original recipe.
- G. P. Lamb and S. Kobayashi, *Low-Γ jets from compact stellar mergers*, MNRAS **472**, 4953
  (2017), [arXiv:1706.03000](https://arxiv.org/abs/1706.03000): the off-axis element integral.
- E. Berger, *Short-duration gamma-ray bursts*, ARA&A **52**, 43 (2014),
  [arXiv:1311.2603](https://arxiv.org/abs/1311.2603): the $L_X$–$E_\mathrm{iso}$ relation.
- R. D. Blandford and C. F. McKee, *Fluid dynamics of relativistic blast waves*, Phys. Fluids
  **19**, 1130 (1976), [doi:10.1063/1.861619](https://doi.org/10.1063/1.861619).
- R. Sari, T. Piran and R. Narayan, *Spectra and light curves of gamma-ray burst afterglows*, ApJL
  **497**, L17 (1998), [arXiv:astro-ph/9712005](https://arxiv.org/abs/astro-ph/9712005).
- H. J. van Eerten and A. I. MacFadyen, *Gamma-ray burst afterglow scaling relations for the full
  blast wave evolution*, ApJL **747**, L30 (2012), [arXiv:1111.3355](https://arxiv.org/abs/1111.3355):
  the $E_\mathrm{iso}$ rescaling of the catO5 files (§9.4).
- A. Franceschini and G. Rodighiero, *The extragalactic background light revisited and the
  cosmic photon-photon opacity*, A&A **603**, A34 (2017),
  [arXiv:1705.10256](https://arxiv.org/abs/1705.10256): the EBL model used by `sim_3d`.
- A. Domínguez et al., *Extragalactic background light inferred from AEGIS galaxy-SED-type
  fractions*, MNRAS **410**, 2556 (2011), [arXiv:1007.1459](https://arxiv.org/abs/1007.1459): the
  EBL model of §5.4 and the `grb_model` default.
- Planck Collaboration, *Planck 2018 results. VI. Cosmological parameters*, A&A **641**, A6 (2020),
  [arXiv:1807.06209](https://arxiv.org/abs/1807.06209): $z(d_L)$.

**Internal**

- *Simulation guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit*,
  internal note (Sep 2026): Steps 1–8 referenced throughout.
