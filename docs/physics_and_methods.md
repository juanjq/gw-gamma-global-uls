# Physics and methods of `gw-gamma-global-uls`

This document explains what the repository computes and why, stage by stage, with the equations
the code implements and their physical meaning. The [`README`](../README.md) covers installation
and the cluster/local workflow; this file covers the science.

Step numbers such as "Step 4.3" refer to the note *Simulation guidelines: from a numerical GRB
model to a GW-marginalised IACT upper limit*. `grb_model.py` and `angle_distribution.py` follow
that note.

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
9. [The emission model (setup_model)](#9-the-emission-model-setup_model)
10. [The viewing-angle distribution (setup_angle_distribution)](#10-the-viewing-angle-distribution-setup_angle_distribution)
11. [What is not wired together yet](#11-what-is-not-wired-together-yet)
12. [Validation inventory](#12-validation-inventory)
13. [Historical notebooks (`notebooks/others/`)](#13-historical-notebooks-notebooksothers)
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

Status of each component:

| Component | Code | Status |
| --- | --- | --- |
| 2D flux UL, Γ = 2 power-law injection | `sim_3d__model_INAF.ipynb`, `gwuls/simulate.py` | working |
| 3D band-luminosity UL, Γ = 2 power-law injection | same | working |
| Emission model $L(E',t';\theta)$: guidelines Steps 1, 2, 5 and 7 | `setup_model.ipynb`, `gwuls/grb_model.py` | built and validated; **not yet used by the simulation** |
| Viewing-angle draw $p(\theta_v \mid d)$: guidelines Step 4.3 | `setup_angle_distribution.ipynb`, `gwuls/gw_pe.py`, `gwuls/angle_distribution.py` | built and validated; **not yet used by the simulation** |
| GTI normalisation, EBL at $z_i$, time-dependent template, event sampling: guidelines Steps 3, 6, 7 and 8 | — | not implemented ([§11](#11-what-is-not-wired-together-yet)) |

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
                                          │
                                         [E] f(x) = P(Λ_x > Λ*),  bisection in log x
                                              → x_UL where f = CL = 0.95
                                          │
                                         [F] both ULs as flux AND as luminosity,
                                              over the distance posterior → .json

   Prepared for the numerical-model 3D UL, not yet read by [D]:
     setup_model.ipynb              → L(E', t'; θ) per viewing angle, rest frame
                                      (data/models/grb_afterglow_inaf/standardized/*.npz)
     setup_angle_distribution.ipynb → p(d, θ_v) and p(θ_v | d) for one alert
                                      (data/gw_input/<alert>_theta_distribution.npz)
```

**Which file does what**

| File | Role |
| --- | --- |
| `notebooks/sim_3d__model_INAF.ipynb` | Main pipeline. Part 1 prepares data (stages A and B, input `.pkl`); Part 2 runs the simulations (stages D, E and F) for both the 2D and 3D limits. Despite its name it does not yet load the INAF model. |
| `notebooks/setup_model.ipynb` | Run once. Standardises the emission-model files ([§9](#9-the-emission-model-setup_model)). |
| `notebooks/setup_angle_distribution.ipynb` | Run once per alert. Standardises the PE release and builds $p(d,\theta_v)$ ([§10](#10-the-viewing-angle-distribution-setup_angle_distribution)). |
| `gwuls/utils.py` | HEALPix ↔ WCS, credible-region masks, per-bin distance CDFs. |
| `gwuls/simulate.py` | Spectral conversions, input validation, the TS/Λ engine, 2D and 3D samplers, bisection, reporting. Also a CLI used by the Slurm grid scan. |
| `gwuls/grb_model.py` | $L(E',t')$ per angle: rest-frame projection, interpolation, $E_{\rm iso}$ rescaling, angle interpolation, EBL. |
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
`delta*.ipynb` notebooks ([§13](#13-historical-notebooks-notebooksothers)).

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
- **Background.** Ring method (`RingBackgroundMaker`): for each position, OFF counts are taken from
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

By Wilks' theorem, $\sqrt{|\mathrm{TS}_b|}$ is approximately the local significance of an excess
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
carry **no trials factor**. They are used as a sanity reference in [§7.6](#76-trials-factor-sanity-check).

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
effect. At $z\approx0.29$ (Domínguez et al. 2011):

| Energy | 0.3 TeV | 0.6 TeV | 1 TeV | 2 TeV | 5 TeV | 10 TeV |
| --- | --- | --- | --- | --- | --- | --- |
| optical depth $\tau$ | 0.9 | 2.3 | 3.6 | 4.8 | 7.5 | 15.5 |

Only $e^{-2.3}\approx10\%$ of the intrinsic flux survives at the bottom of the band. An intrinsic
luminosity limit needs EBL absorption at the drawn $z_i$, which is guidelines Step 7
([§11](#11-what-is-not-wired-together-yet)).

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
same $\Lambda$ sample as sequential ones.

---

## 7. E: From Λ distributions to an upper limit

*Code: `simulate._bisect_ul`, `run_iterative_ul`, `run_iterative_ul_3d`.*

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
power-constrained limit. A downward fluctuation of the background cannot produce an
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
support fractions next to each other for this reason.

### 7.3 Bisection

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

### 7.4 Grid-scan alternative

With `USE_ITERATIVE_ULS = False`, the notebook submits one Slurm job per amplitude on a 150-point
log grid (`slurm.submit_simulation_job` → `python gwuls/simulate.py ... --mode 2d|3d`). It then
applies the same crossing interpolation to the gridded $f$. Older notebooks fitted a sigmoid
$f(\log x) = b + (L-b)/(1+e^{-k(\log x - x_0)})$ instead (`utils.sigmoid`); that fit is no longer
used for the quoted number.

### 7.5 Closure test (3D)

$x_{\rm UL}$ is interpolated, so it was never simulated itself. The notebook runs $N$ fresh
realisations exactly at $L_0^{\rm UL}$ with independent seeds and checks

$$
\text{pull} = \frac{f_{\rm closure}-\mathrm{CL}}{\sigma_f},\qquad |\text{pull}|<3 .
$$

### 7.6 Trials-factor sanity check

The global 2D limit applies to the *maximum* over the region, so it pays the trials factor. The
per-pixel sky-map ULs do not. The global limit should therefore sit **at or above** the largest
per-pixel UL inside $\mathcal{M}_{95}$. A global limit below it points to a problem with $\kappa$
or with mismatched spectral assumptions.

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

Every quoted number is written to `outputs/results/<stem>_upper_limits.json`.

---

## 9. The emission model (setup_model)

*Code: `grb_model.py`; notebook `setup_model.ipynb`. Guidelines Steps 1, 2, 5 and 7.*

### 9.1 Why

The current injection is a constant Γ = 2 power law. A physical counterpart, such as a GRB
afterglow seen off-axis, has a spectrum and a light curve that depend on energy, time since the
merger, and viewing angle. This module prepares such a model in a distance-independent form so
that a later version of the 3D simulation can place it at each drawn $(d_i, z_i, \theta_i)$.

### 9.2 Model options (Step 1)

| Option | What | Frame | Role |
| --- | --- | --- | --- |
| 1 | INAF / CTA-GW O5 BNS afterglow catalogue, `catO5_*.fits`: tabulated $F(E,t)$ at one $(d_L^m,\theta^m)$, no EBL | observer, at the file's distance | baseline |
| 2 | analytic $L = L_0\,f(E')/f(E_{\rm ref})\cdot g(t')$ | rest frame | flexible test |
| 3 | $f=(E'/E_{\rm ref})^{-2}$, $g=1$ | rest frame | closed-form validation anchor |

The five catalogue files in `data/models/grb_afterglow_inaf/raw/`:

| $\theta^m$ [deg] | $d_L^m$ [Mpc] | $z^m$ (Planck18) | $E_{\rm iso}$ [erg] | grid $n_E\times n_t$ after trimming |
| --- | --- | --- | --- | --- |
| 9.075 | 843 | 0.170 | 5.3 × 10⁴⁸ | 40 × 70 |
| 13.167 | 287 | 0.062 | 1.2 × 10⁴⁸ | 40 × 70 |
| 22.631 | 113 | 0.025 | 1.1 × 10⁵⁰ | 40 × 70 |
| 28.554 | 349 | 0.075 | 2.0 × 10⁴⁹ | 40 × 70 |
| 77.841 | 545 | 0.114 | 1.4 × 10⁵⁰ | 38 × 47 |

Each file is a **different simulated event**, with $E_{\rm iso}$ spanning two decades. The files
are not one source seen from five angles (see §9.6).

### 9.3 Observer frame ↔ rest frame (Steps 2 and 7)

Frame relations, with time measured from the merger:

$$
E' = (1+z)E,\qquad t' = \frac{t}{1+z},\qquad dE\,dt = dE'\,dt',\qquad d_M = \frac{d_L}{1+z}.
$$

The $dN$ photons emitted in $dE'\,dt'$ arrive in $dE\,dt$ spread over the sphere $4\pi d_M^2$, so

$$
L(E',t') \equiv \frac{dN}{dE'dt'} = 4\pi d_M^2\,F(E,t) = \frac{4\pi d_L^2}{(1+z)^2}\,F\!\left(\frac{E'}{1+z},\,(1+z)t'\right).
$$

This is a **photon-number** luminosity, in ph s⁻¹ GeV⁻¹, which is why the factor is $(1+z)^2$.
An energy-differential luminosity would carry $(1+z)$ instead.

The observer grid maps **exactly** onto a rest-frame grid ($E'_n=(1+z^m)E_n$,
$t'_n=t_n/(1+z^m)$), so this step needs no interpolation. The inverse (Step 7) places the model
at any drawn distance:

$$
F_i(E,t) = \frac{(1+z_i)^2}{4\pi d_i^2}\;L\big((1+z_i)E,\;t/(1+z_i)\big)\;e^{-\tau(E,z_i)} .
$$

EBL absorption is applied only here, at the observed energy and the drawn $z_i$, and never at
$z^m$ (the files are intrinsic). Round-trip check: Step 2 then Step 7 at $(d_L^m, z^m)$ returns
the file exactly (max relative error 0 for all five files).

### 9.4 Interpolation and cleaning

- $\log L$ is interpolated bilinearly in $(\log E', \log t')$. Spectra and light curves are local
  power laws, for which this is exact. Off the grid, the linear log-log extrapolation continues
  the local power law, and `to_observer` warns with the extrapolated fraction.
- Each file is trimmed to its largest rectangular block of finite, positive cells. This removes
  the all-zero 10 TeV row and the zero-flux early columns of off-axis events.
- Derived quantities:
  - energy light curve $\mathcal{L}(t') = \int E' L\,dE'$ [erg s⁻¹];
  - its peak $(t'_{\rm pk}, \mathcal{L}_{\rm pk})$;
  - band integrals $\int\!\!\int E' L\,dE'dt'$ on fine log grids (trapezoid in $\ln E'$, $\ln t'$),
    which is the $D_i$ that Step 6 will need.

### 9.5 Off-axis afterglow physics

A relativistic jet decelerates in the circum-merger medium. For an adiabatic blast wave,
$\Gamma^2 \propto E/(n r^3)$ with observer time $t\propto r/(c\Gamma^2)$, which gives

$$
\Gamma \propto \left(\frac{E}{n}\right)^{1/8} t^{-3/8}.
$$

Emission is beamed into a cone of half-opening $1/\Gamma$. An observer at angle
$\Delta\theta = \theta_v - \theta_{\rm jet}$ outside the jet sees the flux rise until
$1/\Gamma \sim \Delta\theta$, so

$$
t_{\rm pk}\propto \left(\frac{E}{n}\right)^{1/3}\Delta\theta^{8/3}
$$

for a top-hat jet. Larger viewing angles give **later, much fainter** peaks. After rescaling to a
common $E_{\rm iso}$ (§9.6), the catalogue shows this: $t'_{\rm pk}$ goes from 8 s at 9° to
2.8 × 10⁵ s at 78°, and $\log\mathcal{L}_{\rm pk}$ from 49.6 to 37.2.

### 9.6 One common $E_{\rm iso}$ (blast-wave scale invariance)

To separate angle dependence from event-to-event scatter, each file is rescaled to
$E_{\rm iso,ref} = 10^{50}$ erg:

$$
L(E',t') \;\to\; k^{a}\,L\!\left(E',\,t'/k^{b}\right),\qquad k=\frac{E_{\rm iso,ref}}{E_{\rm iso}},\qquad (a,b)=(1,\tfrac13).
$$

**Physics.** At fixed density and microphysics, the only length scale is the Sedov length
$\ell\propto(E/n)^{1/3}$, so every dynamical time scales as $E^{1/3}$ ($b = 1/3$, consistent with
$t_{\rm pk}$ above). At the same dynamical stage the number of radiating electrons scales as
$E$ while $\Gamma$ and $B'$ are unchanged, so $F_{\nu,\max}\propto E$ and $\nu_m$ is unchanged.
The flux between $\nu_m$ and $\nu_c$ therefore scales as $E$ ($a=1$). Above $\nu_c$ it would scale
as $E^{2/3}$, because $\nu_c\propto t^{-2}\propto E^{-2/3}$.

**Empirical check** (peak of $\mathcal{L}$; is it monotonic in θ?):

| scaling $(a,b)$ | $t'_{\rm pk}$ monotonic | $\mathcal{L}_{\rm pk}$ monotonic |
| --- | --- | --- |
| raw files | no | no |
| (1, 0): plain $L/E_{\rm iso}$ | no | yes |
| (2/3, 1/3) | yes | no |
| **(1, 1/3): default** | **yes** | **yes** |

Density, $\varepsilon_e$, $\varepsilon_B$ and jet structure are not recorded in the headers, so
residual scatter from them remains. The rescaling is a uniform shift in $(\log t',\log L)$, so it
commutes with the angle interpolation below (checked to 1.7 × 10⁻¹³ dex). The cache stores each
file at its **own** $E_{\rm iso}$, and the rescaling is applied after loading.

### 9.7 The model at any viewing angle (Step 5)

Files exist only at $\theta_1<\dots<\theta_5$. For $\theta_a\le\theta<\theta_b$, with
$w=(\theta-\theta_a)/(\theta_b-\theta_a)$:

| Option | Rule | Trade-off |
| --- | --- | --- |
| A, nearest | use the closer file | exact shapes; error set by grid spacing |
| B, stochastic | use $\theta_b$ with probability $w$, otherwise $\theta_a$ | no invented shapes; unbiased on average over iterations |
| C, interpolation (**peak-aligned** by default) | see below | smooth in θ |

Interpolating $\log L$ at fixed $t'$ between two light curves that peak at different times gives
a **double-peaked** curve (58–68 of 398 test angles did). Option C therefore interpolates at fixed
**phase** $x = \log t' - \log t'_{\rm pk}$:

$$
\log L(E',x;\theta) = (1-w)\log L_a(E',x) + w\log L_b(E',x),\qquad
\log t'_{\rm pk}(\theta) = (1-w)\log t'_{{\rm pk},a} + w\log t'_{{\rm pk},b}.
$$

Peak time, peak luminosity and the slope $d\log L/d\log t'$ at every phase are then all
interpolated linearly in θ, and the result has exactly one peak (0 of 398 angles failed). This
relies on two facts. The catalogue peaks are achromatic, so one time shift per file aligns every
energy. And each input is first cut to its single-peaked part, which removes the re-brightening
at the last node of the 77.8° file. Outside $[\theta_1,\theta_5]$ the edge file is used, with a
warning.

**Interpretation (guidelines Step 5).** Once each iteration is normalised to a trial luminosity
(Step 6), θ enters only through the **shape** of the spectrum and light curve, not through its
brightness. The limit is then on the isotropic-equivalent luminosity along our line of sight.

### 9.8 Storage and use

`GRBModelSet.save` writes one `.npz` per angle to
`data/models/grb_afterglow_inaf/standardized/`. Later code reads them with
`GRBModelSet.load(...)`, optionally calls `.rescaled_to_eiso(1e50)`, then
`.get_model(theta_i, method="interp")` and `.to_observer(d_i, z_i, apply_ebl=True)`. The EBL
spot-check (is a file already absorbed?) needs `$GAMMAPY_DATA` and was skipped in the last run.

---

## 10. The viewing-angle distribution (setup_angle_distribution)

*Code: `gw_pe.py`, `angle_distribution.py`; notebook `setup_angle_distribution.ipynb`.
Guidelines Step 4.3.*

### 10.1 Why

With a physical model, each iteration needs a viewing angle $\theta_i$ as well as
$(\mathrm{RA}_i, \mathrm{Dec}_i, d_i)$. An on-axis afterglow is many orders of magnitude brighter and
earlier than an off-axis one (§9.5), so the angle distribution directly sets which model files
dominate the limit.

### 10.2 From $\theta_{JN}$ to the jet viewing angle

GW parameter estimation measures $\theta_{JN}\in[0°,180°]$, the angle between the binary's total
angular momentum $\mathbf{J}$ and the line of sight. A jet is bipolar along $\mathbf{J}$, so the
relevant angle is the one to the **nearer** jet:

$$
\theta_v = \min(\theta_{JN},\,180°-\theta_{JN}) \in [0°, 90°].
$$

### 10.3 The distance–inclination degeneracy

The two GW polarisations scale as

$$
h_+\propto \frac{1+\cos^2\iota}{2}\frac{1}{d},\qquad h_\times\propto \frac{\cos\iota}{d}.
$$

A face-on binary far away and an inclined binary nearby produce similar amplitudes, so $d$ and
$\theta_v$ are strongly correlated in the posterior. For S240615dg,
$\mathrm{corr}(d,\cos\theta_v)=0.82$. The angle drawn in an iteration must therefore be
**conditioned on that iteration's distance**:

$$
p(\theta\mid d_i) = \frac{p(d_i,\theta)}{p(d_i)} .
$$

For S240615dg, the median of $p(\theta_v\mid d)$ runs from 49.5° at $d$ = 1123 Mpc to 13.6° at
1849 Mpc.

### 10.4 PE data (Part 0)

A GWTC `*-combined_PEDataRelease.hdf5` (0.2–7 GB) is reduced to
`data/gw_pe/standardized/<superevent>.h5` (~1 MB, tracked in git), keeping:

- posterior columns: sky position, $d_L$, $z$, $\theta_{JN}$, $\iota$, source-frame masses,
  $\chi_{\rm eff}$, tidal deformabilities when present, network SNR;
- **the same number of samples (5000) from every waveform analysis**, so the concatenation is
  exactly the equal-weight mixture the catalogue papers quote, and each analysis can still be
  inspected alone;
- the PE prior samples of $(d_L,\theta_{JN})$, and each analysis's metadata.

The file is named by the superevent ID (the alert name), looked up in the catalogue's
`PESummaryTable`. `index.csv` is regenerated from the files every time. Current contents:

| alert | GW name | $d_L$ [Mpc], median [90%] | $\theta_v$ [deg], median [90%] | $P(\theta_v<30°)$ | $m_1, m_2$ [M☉] |
| --- | --- | --- | --- | --- | --- |
| S240615dg | GW240615_113620 | 1566 [1123, 1849] | 24.4 [6.5, 51.1] | 0.65 | 34.1, 26.0 |
| S241125n | GW241125_010116 | 4748 [2600, 8611] | 55.8 [15.6, 86.0] | 0.17 | 59.3, 46.1 |

Both events are **binary black holes** ($P(m_2<3\,M_\odot)=0$). The catO5 models describe BNS
afterglows, so for these alerts the machinery is being exercised on events for which no standard
GRB afterglow is expected.

### 10.5 The joint $p(d,\theta_v)$ from PE samples

`JointDistanceAngle.from_samples` builds a Gaussian KDE:

- **In $(d, c=\cos\theta_v)$, not in $\theta_v$.** An isotropic orientation prior is flat in
  $\cos\theta$, and a smooth posterior has finite density at both edges. In $\theta_v$ the density
  per degree must vanish at 0° (the $\sin\theta$ Jacobian), which a direct KDE would violate.
- **Full-covariance kernel** (Scott's rule), so it follows the $d$–$\cos\theta_v$ correlation
  instead of blurring across it:

$$
H = \hat\Sigma\cdot \texttt{bw\_scale}^2\cdot n_{\rm eff}^{-1/3}.
$$

- **Reflection** at $c=0$ and $c=1$, and at $d=0$ if the grid reaches it. Mirrored samples use the
  mirrored (tilt-flipped) kernel, so correlated mass pushed past an edge comes back with the
  correct tilt.
- Binned on a 300 × 400 lattice and convolved by FFT: 20 000 samples take well under a second.
- Mapped to a per-degree density with the Jacobian:

$$
p(d,\theta_v) = p(d,c)\,\sin\theta_v\,\frac{\pi}{180} .
$$

The grid (300 distances × 181 angles, 0.5° steps) is normalised to unit double integral. From it:

$$
p(d)=\int p(d,\theta)\,d\theta,\qquad p(\theta)=\int p(d,\theta)\,dd,\qquad
p(\theta\mid d),\qquad p(d\mid\theta_0)\ \text{or}\ p(d\mid a<\theta<b).
$$

$p(d\mid\theta_0)$ answers "the angle is known", for example from a counterpart: a face-on
angle pulls the distance out (θ = 5° gives 1729 Mpc), an inclined one pulls it in (60° gives
953 Mpc). Sampling uses the inverse CDF of the conditional, row-interpolated linearly in $d$.

### 10.6 Options when no PE is available

Low-latency skymaps carry no inclination. The alternatives, all normalised per degree:

| kind | $p(\theta_v)$ | depends on $d$? | physics |
| --- | --- | --- | --- |
| `isotropic` | $\sin\theta$ | no | random orientations, no detection bias; most pessimistic for on-axis |
| `schutz` | $\propto\sin\theta\,\big[\tfrac{(1+\cos^2\theta)^2}{4}+\cos^2\theta\big]^{3/2}$ | no | orientations of GW-**detected** sources (Schutz 2011); peak ≈ 31°, median ≈ 36° |
| `selection` | $p(d,\theta)\propto d^2\sin\theta\,P_{\rm det}(d/D_h,\theta)$ | yes | explicit detection selection with horizon $D_h$ |
| `fixed`, `two_bin` | δ at θ, or piecewise flat | no | placeholders |

**Where Schutz comes from.** A single detector sees a source at $(d,\theta)$ if its orientation
factor $\Theta/4 > d/D_h$ (Finn & Chernoff 1993), with

$$
\frac{\Theta}{4} = \frac12\sqrt{F_+^2(1+\cos^2\theta)^2+4F_\times^2\cos^2\theta}\ \in[0,1].
$$

$P_{\rm det}$ is the fraction of sky positions and polarisations for which this holds (Monte
Carlo, $10^5$ draws). For sources uniform in volume, integrating $d^2$ up to $D_h\,\Theta/4$ gives
the $d$-marginal $\propto\sin\theta\,\langle(\Theta/4)^3\rangle$. Since
$\langle F_+^2\rangle=\langle F_\times^2\rangle=1/5$, the average
$\langle(\Theta/4)^2\rangle\propto(1+\cos^2\theta)^2/4+\cos^2\theta$, and Schutz's closed form
replaces $\langle(\Theta/4)^3\rangle$ by $\langle(\Theta/4)^2\rangle^{3/2}$. The notebook finds a
maximum CDF difference of 0.009 between the two. The conditional $p(\theta\mid d)$ is what Schutz
discards: near the horizon only face-on sources are detected (median $\theta_v$ = 58° at
$d/D_h=0.1$, 14° at 0.9). The horizon (`HORIZON_MPC` = 3000) is illustrative only.

**Comparison of the options for S240615dg** (PE marginalised over $d$):

| option | median [deg] | $P(<10°)$ | $P(<20°)$ | $P(<30°)$ |
| --- | --- | --- | --- | --- |
| PE S240615dg | 24.7 | 0.10 | 0.36 | 0.64 |
| Schutz | 36.2 | 0.05 | 0.19 | 0.38 |
| selection ($D_h$ = 3000 Mpc) | 36.5 | 0.05 | 0.19 | 0.37 |
| isotropic | 60.0 | 0.015 | 0.06 | 0.13 |

**Where this reaches the limit: which model file each option uses** (nearest-angle, Option A):

| option | 9.1° | 13.2° | 22.6° | 28.6° | 77.8° |
| --- | --- | --- | --- | --- | --- |
| PE S240615dg | 0.12 | 0.18 | 0.23 | 0.44 | 0.04 |
| Schutz | 0.06 | 0.09 | 0.14 | 0.49 | 0.22 |
| isotropic | 0.02 | 0.03 | 0.05 | 0.30 | 0.60 |

Under an isotropic prior, 60% of iterations would use the far off-axis file, whose afterglow is
about twelve orders of magnitude fainter at peak and peaks days after the merger.

### 10.7 Skymap distance versus PE distance

The planned 3D loop draws $d_i$ from the **alert skymap** (§3.4) and then $\theta_i$ from the **PE**
conditional. For S240615dg the two distance posteriors disagree: skymap 1421 [1028, 1815] Mpc
versus PE 1566 [1123, 1849] Mpc. Skymap distances in the near tail pick up the inclined angles of
that tail, which shifts the $\theta_v$ median by **+8.2°**. When PE samples exist, the guidelines'
preferred route avoids this by drawing a PE sample index and taking (RA, Dec, $d$, $\theta_v$)
together (`ThetaDistribution.sample_joint`), which keeps every correlation.

### 10.8 Storage

`save_theta_distribution` writes `data/gw_input/<alert>_theta_distribution.npz`, containing the
kind, the joint grid and the samples. `load_theta_distribution` restores the same object whichever
kind produced it. A round-trip test checks that the reloaded object gives identical draws. For
S240615dg the saved kind is `pe`.

---

## 11. What is not wired together yet

The maintained simulation (§4–§8) still injects a constant Γ = 2 power law without EBL. Moving to
the numerical-model 3D limit of the guidelines needs these pieces, none of which exist in the code
yet:

- **Step 3: fixed rest-frame band $[E'_0,E'_1]$ for $L_k$.** At S240615dg distances,
  $1+z\approx1.2$–$1.34$, which is larger than the IACT energy resolution. The band must be chosen
  to overlap the redshifted analysis band over most of the posterior while staying inside every
  model file's rest-frame grid.
- **Step 6: per-iteration normalisation over the real GTIs.** Map GTIs to the rest frame with
  $z_i$ ($T'_{\rm obs}=T_{\rm obs}/(1+z_i)$), then

$$
L_{\lambda,i} = \frac{L_k\,T'_{\rm obs}}{D_i},\qquad
D_i=\sum_j\int_{t'_{1j}}^{t'_{2j}}\!\!\int_{E'_0}^{E'_1}E'\,L(E',t';\theta_i)\,dE'\,dt'.
$$

  Draws with $D_i\approx0$, where the GTIs fall outside the model's time grid, must be counted.
- **Step 7 inside the loop:** `to_observer(d_i, z_i, apply_ebl=True)`, then an energy-dependent
  `LightCurveTemplateTemporalModel` sampled with `MapDatasetEventSampler`. `dataset.fake()` does
  not support energy-dependent templates.
- **Step 8: the statistic.** Either keep $\Lambda$ (§4.3), or compare a TS at the drawn position
  with the real TS map there:

$$
P_{\rm excl}(L_k)=\frac1N\sum_i\mathbf{1}\big[\mathrm{TS}_i(L_k)>\mathrm{TS}_{\rm obs}(\mathrm{RA}_i,\mathrm{Dec}_i)\big].
$$

  The same coverage cap as §7.2 applies.
- **Draw order:** $(b, d)$ from the skymap as now, or $(\mathrm{RA},\mathrm{Dec},d,\theta)$ jointly
  from PE (§10.7), then $\theta_i\mid d_i$, then the model at $\theta_i$, then $L_{\lambda,i}$, then
  the flux.

---

## 12. Validation inventory

| Check | Where | Guards against | Last recorded result |
| --- | --- | --- | --- |
| Band integrals vs quadrature and gammapy; round trips; Γ = 2 limit from both sides; K = 1 at Γ = 2; $L\propto d^2$ | `simulate.check_spectral_conversions` (`--self-test`) | the Γ = 2 singularity, unit errors | passes (run in notebook) |
| Input `.pkl` validation: shapes, empty mask, κ range, CDF transposed / non-monotonic / truncated | `simulate.validate_simulation_input` | silent failures thousands of iterations later | runs at `.pkl` write and at every load |
| HEALPix → WCS assignment separation < 1.5 bin diagonals | `_map_healpix_to_wcs_bins` | RA-convention errors | printed per run |
| 3D sampler: χ² sky marginal, KS distance marginal, hottest-bin conditional, no $d\le0$, grid-edge probability atoms | `simulate.check_3d_sampling` | the production sampler drifting from the posterior | notebook |
| Background Λ split-half medians; binomial p-value error | notebook Part 2 | an unstable null distribution | notebook |
| Bisection: bracket check, MC-aware tolerance, monotonicity, edge flag | `_bisect_ul` | meaningless limits at a bracket edge | per run |
| Closure test at $L_0^{\rm UL}$ with independent seeds | notebook | interpolation bias | pull < 3σ required |
| Global UL ≥ per-pixel sky-map UL | notebook | a missing trials factor, wrong κ | printed |
| Model round trip Step 2 → Step 7 | `setup_model` | frame or $(1+z)$ power errors | max relative error 0 |
| Peak-aligned interpolation single-peaked; rescale ∘ interpolate commute | `setup_model` | double peaks, $E_{\rm iso}$ inconsistency | 0 of 398 angles multi-peaked; 1.7 × 10⁻¹³ dex |
| Standardised PE vs full raw posterior; fold vs pesummary `viewing_angle` | `setup_angle_distribution` Part 0 | subsampling bias | 5/50/95% quantiles within ~1°; fold exact |
| KDE marginals vs samples; grid $p(\theta\mid d)$ vs grid-free resampling; bandwidth ×0.5 and ×2 | Part 1 | over- or under-smoothing | grid vs resampling medians within 0.5°; bandwidth moves medians by ≤ 1.5° |
| Three sampling routes reproduce the posterior (quantiles, correlation, KS) | Part 1 | sampler bugs | KS ≤ 0.016 |
| Each option's sampler vs its own pdf | Part 2 | sampler bugs | at $1/\sqrt n$ level |
| Schutz vs Monte-Carlo selection model | Part 2 | a wrong closed form | max CDF difference 0.009 |

---

## 13. Historical notebooks (`notebooks/others/`)

These predate the `gwuls` package and are not runnable as they are. Each has a note at the top.

| Notebook | What it did | Physics purpose |
| --- | --- | --- |
| `check_1_simulation.ipynb` | Built simulated datasets from the real IRFs and background; compared real minus simulated counts, and source minus null | validate that `fake()` on the real-IRF dataset reproduces the real data |
| `delta.ipynb` / `delta_he.ipynb` | Full 2D pipeline with the GW map collapsed to its hottest pixel (Dirac delta); LST mono 0.15–0.6 TeV (`baccmod`) and LST+MAGIC stereo 0.6–20 TeV (`pybkgmodel`) | closure test: with no trials, the MC ("frequentist") UL must match gammapy's per-pixel UL |
| `results_delta.ipynb` | Hand-entered results of those runs; percentage difference vs significance, pixel size and containment radius | quantified the agreement of the two UL methods |
| `check_energy_dependence.ipynb` | Hand-entered MC UL vs sky-map-maximum UL (95% region and full map) across energy bands, mono vs stereo | size of the trials penalty per band |
| `test_sim_flux.ipynb` | Earliest 2D flux pipeline: Slurm grid, sigmoid or linear crossing | superseded by `simulate.run_iterative_ul` |

---

## 14. References

- Cash, W. (1979), ApJ 228, 939: Poisson likelihood statistic.
- Singer, L. P. et al. (2016), ApJL 829, L15: the per-pixel distance ansatz of LVK skymaps.
- Schutz, B. F. (2011), CQG 28, 125023: inclination distribution of GW-detected binaries.
- Finn, L. S. & Chernoff, D. F. (1993), PRD 47, 2198: the orientation factor Θ.
- Domínguez, A. et al. (2011), MNRAS 410, 2556: EBL model used for $\tau(E,z)$.
- van Eerten, H. & MacFadyen, A. (2012): blast-wave scale invariance behind the $E_{\rm iso}$
  rescaling, as cited in `grb_model.py`.
- *Simulation guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit*
  (internal note, Sep 2026): Steps 1–8 referenced throughout.
