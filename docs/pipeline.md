# Pipeline: from a GW alert and IACT data to upper limits

What [`3_upper_limits.ipynb`](../notebooks/3_upper_limits.ipynb) computes and why, stage by stage,
with the equations the code implements. Installation and the cluster/local workflow are in the
[`README`](../README.md). The emission model and the viewing angle have their own documents,
[`emission_model.md`](emission_model.md) and [`viewing_angle.md`](viewing_angle.md), summarised in
§9 and §10. Notebook cells cite this file as "§n"; "Step 4.3" and the like refer to the internal note
*Simulation guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit*. Works are
cited as author (year) and listed in [§13](#13-references).

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
13. [References](#13-references)

---

## 0. The question

An LVK alert localises a compact-binary merger to a region of the sky and a range of
distances. LST-1 (alone, or with MAGIC) observed part of that region and saw no significant
source. **What is the brightest very-high-energy counterpart that is still compatible with
those observations?**

A standard point-source upper limit is not well defined here, because the source position is
unknown. So is its distance, and so is its orientation. The pipeline defines **one global
test statistic** over the GW region, $\Lambda$. It calibrates $\Lambda$ by Monte Carlo, using
the real instrument response and background, and turns it into three upper limits:

| Limit | Fixed per test | Randomised per realisation | Result |
| --- | --- | --- | --- |
| **2D flux UL** | power-law amplitude $\phi_0$ | sky position in the 95% GW region (+ Poisson noise) | $\phi_0^{\rm UL}$, photon/energy flux |
| **3D luminosity UL** | band luminosity $L_0$ | sky position **and** distance from the full GW map (+ Poisson noise) | $L_0^{\rm UL}$ in erg s⁻¹ |
| **3D UL with the emission model** | trial luminosity $L_k$ of a GRB afterglow | sky position, distance **and** viewing angle, together from one GW PE sample (+ Poisson noise) | $L_k^{\rm UL}$ in erg s⁻¹ |

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
                                              • 3D model: L_k fixed, (RA, Dec, d, θ_v) from one
                                                PE sample, GRB afterglow + EBL, run by run
                                          │
                                         [E] f(x) = P(Λ_x > Λ*), fitted as a probit in log x
                                              to realisations that each have their own x
                                              → x_UL where f = CL = 0.95, ± its MC uncertainty
                                          │
                                         [F] both ULs as flux AND as luminosity,
                                              over the distance posterior → .json

   Prepared ahead, read by [D] for the emission-model limit (RUN_MODEL_3D):
     1_grb_model.ipynb         → L(E', t'; θ_v) of the benchmark jet, 1° grid
                                 (data/models/grb_afterglow_phenomenological/benchmark/)
     2_gw_viewing_angle.ipynb  → standardized PE samples (RA, Dec, d, θ_v)
                                 (data/gw_pe/standardized/<alert>.h5), and
                                 p(θ_v | d) for the non-PE angle options
```

**Which file does what**

| File | Role |
| --- | --- |
| `notebooks/1_grb_model.ipynb` | Run once. Builds the benchmark jet at every angle and saves it ([§9](#9-the-emission-model)). |
| `notebooks/2_gw_viewing_angle.ipynb` | Run once per alert. Standardises the PE release and builds $p(d,\theta_v)$ ([§10](#10-the-viewing-angle-distribution)). |
| `notebooks/3_upper_limits.ipynb` | Main pipeline. Part 1 prepares the data (stages A and B, the input `.pkl`, per-run datasets and GTIs); Part 2 runs stages D–F for the 2D, 3D and emission-model limits. |
| `notebooks/extra/` | Optional explainers, read by nothing else: the catO5 files, random draws from the jet recipe, how each realisation's source is drawn. |
| `gwuls/utils.py` | HEALPix ↔ WCS, credible-region masks, per-bin distance CDFs. |
| `gwuls/simulate.py` | Spectral conversions, input validation, the TS/Λ engine, 2D and 3D samplers, the emission-model injection, the upper-limit fit, reporting. |
| `gwuls/grb_model.py` | $L(E',t')$ per angle: rest frame, projection to the observer, angle interpolation, EBL. |
| `gwuls/grb_phenomenological.py` | The Nava (2020) / Abe et al. (2026) recipe, the jet structure, the off-axis element integral, `BENCHMARK`. |
| `gwuls/gw_pe.py` | GWTC PE release (GBs) → ~1 MB per-alert file of posterior samples. |
| `gwuls/angle_distribution.py` | Joint $p(d,\theta_v)$, its conditionals and marginals, and the population priors. |
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

*Code: `utils.py`; notebook Part 1, "Reading GW data" to "3D GW: distance CDF per WCS bin".*

### 3.1 What the skymap contains

A HEALPix map: each pixel $j$ holds the probability $P_j$ that the source is there, and three
numbers for the distance along that line of sight (Singer et al. 2016):

$$
p(d \mid j) = N_j\, d^2\, \frac{1}{\sqrt{2\pi}\,\sigma_j}\exp\!\left[-\frac{(d-\mu_j)^2}{2\sigma_j^2}\right],
\qquad (\mu_j,\sigma_j,N_j) = (\texttt{DISTMU},\texttt{DISTSIGMA},\texttt{DISTNORM}).
$$

$d^2$ is the volume element; $\mu_j$ and $\sigma_j$ are ansatz parameters, not the mean and width.

### 3.2 From HEALPix to the WCS analysis grid

The analysis map is WCS (AIR projection around `source_coord`, 3.575° wide, 0.08° bins). Each
HEALPix pixel goes to the bin containing it, and the bin probability is the mean pixel probability
scaled to the bin's solid angle,

$$
p_b = \Big(\tfrac{1}{n_b}\sum_{j\in b} P_j\Big)\,\frac{\Omega_b}{\Omega_{\rm pix}},
$$

which corrects for the whole number of pixels per bin. $\sum_b p_b$ is printed as "Geometry covers
X% of the GW"; $p_b$ is floored at $10^{-20}$ before taking logarithms.

### 3.3 The 95% credible region

Pixels sorted by probability are accumulated to 95%; the last one sets the contour level. A bin is
in $\mathcal{M}_{95}$ if its centre lies inside that contour. $\Lambda$ is searched only there.

### 3.4 Distance along each WCS bin

A bin's distance distribution is the probability-weighted mixture of its pixels',

$$
F_b(r) = \sum_{j\in b} w_j\, F_j(r), \qquad w_j = \frac{P_j}{\sum_{k\in b}P_k},
$$

with $F_j$ the CDF of $p(d\mid j)$ (`ligo.skymap.distance.marginal_cdf`), tabulated as
`cdf_table[b, :]` on `r_grid` (1000 points, 1–10 000 Mpc) and cached under `data/gw_input/`.
Pixels without distance information are skipped; `cdf_valid[b]` marks usable bins. The grid starts
at **1 Mpc, not 0**: the injector converts luminosity to flux through $1/d^2$, and one draw at $d=0$
would dominate every $\Lambda$ distribution.

The sky-marginal distance distribution over the field,
$F(r) = \sum_b \tilde p_b\,F_b(r)$ with $\tilde p_b \propto p_b\,\mathbf{1}[\text{valid}_b]$
(`simulate.marginal_distance_pdf`), is used for every flux ↔ luminosity translation in the report
(§8).

### 3.5 Dirac-delta mode

`USE_DIRAC_DELTA = True` puts all the probability in the hottest pixel. $\Lambda$ then reduces to the
TS at one known position with no trials factor, so the Monte-Carlo limit should reproduce gammapy's
per-pixel UL (checked in early versions, in git history).

---

## 4. B: IACT data, background and the test statistic Λ

*Code: notebook Part 1, "Reading the DL3 data" to "Computing Λ for the real data";
`simulate._TSEngine`.*

### 4.1 Dataset and background

- **Data:** DL3 runs of S240615dg (LST-1 + MAGIC stereo, `obs_ids` 17821–17825), each with its IRFs
  and a 3D background model from `pybkgmodel` or `baccmod`.
- **Geometry:** reco energy 0.6–20 TeV, 4.5 bins per decade; true energy 0.05–100 TeV; offset < 2.5°.
- **Background:** ring method (Berge, Funk & Hinton 2007), inner radius 0.3°, width 0.2°;
  $\mu_{\rm bkg} = \alpha\,n_{\rm off}$ with the acceptance ratio $\alpha$ shaped by the 3D model.
- **Software:** Gammapy (Donath et al. 2023).

### 4.2 Per-bin test statistic

Counts summed over energy and correlated with a top-hat of $r_c = 0.1°$ give, per bin, $n_b$ ON
counts and a background expectation $\mu_b$. With the Cash (1979) statistic,

$$
\mathrm{TS}_b = \operatorname{sign}(n_b-\mu_b)\cdot 2\Big[n_b\ln\frac{n_b}{\mu_b}-(n_b-\mu_b)\Big],
$$

and $\sqrt{|\mathrm{TS}_b|}$ is approximately the local significance (Wilks 1938) of an excess ($+$) or
deficit ($-$). The background is treated as known, in the data and the simulations alike.

### 4.3 GW weighting and the global statistic Λ

$$
\mathrm{TS}'_b = \mathrm{TS}_b + 2\ln p_b, \qquad
\Lambda = \max_{b\in\mathcal{M}_{95}} \mathrm{TS}'_b .
$$

Since $\mathrm{TS}_b = 2\ln(\mathcal{L}_1/\mathcal{L}_0)$, $\Lambda$ picks the position where "the
gamma-ray data prefer a source" times "the GW data say the source is here" is largest. A modest
excess where the GW probability is high can beat a stronger one where it is low, which tempers the
look-elsewhere effect; an excess in bin $b$ wins only if $\mathrm{TS}_b > \Lambda^\star - 2\ln p_b$.

- $\Lambda$ is usually **negative**, since $p_b\ll1$.
- $p_b$ depends on the bin size, but that shifts every $\mathrm{TS}'_b$ by one constant, which
  cancels between observed and simulated $\Lambda$.
- The maximum is a *profile* over position (a sum of $e^{\mathrm{TS}'_b/2}$ would be a Bayes factor).

### 4.4 Observed significance

$N_{\rm bkg}=5000$ background-only realisations (§6.2) give the null distribution:

$$
p = P(\Lambda_{\rm bkg} > \Lambda_{\rm obs}), \qquad S = \Phi^{-1}(1-p), \qquad
\sigma_p = \sqrt{p(1-p)/N_{\rm bkg}} .
$$

Every realisation takes the same maximum over the same region, so $p$ is already **post-trials**.
If $p$ saturates, $|S|$ is only bounded at $\Phi^{-1}(1-1/N_{\rm bkg})$.

### 4.5 The simulation dataset

The simulation dataset in the `.pkl` is built from the **real** runs (IRFs, livetimes, pointings,
reference times) with the **real ring background** of each run, so a simulated observation is

$$
n^{\rm sim}_{\rm pix,E} \sim \operatorname{Poisson}\!\big(\mu^{\rm bkg}_{\rm pix,E} + \mu^{\rm sig}_{\rm pix,E}\big),
$$

with $\mu^{\rm sig}$ the source folded through exposure, PSF and energy dispersion (`dataset.fake`).
The model is then removed and $\Lambda$ computed exactly as for the data.

### 4.6 Containment factor and sky-map ULs

`ExcessMapEstimator` measures flux inside the top-hat, which misses part of a PSF-spread source. A
bright source injected 100 times at the field centre gives
$\kappa = \operatorname{median}[F_{\rm est}/F_{\rm true}]$, and per-pixel sky-map ULs (95% CL) are
divided by $\kappa$. They carry **no trials factor** and serve as a reference in §7.6.

---

## 5. C: Spectral model and flux ↔ luminosity conversion

*Code: `simulate.py` Section 1.*

### 5.1 Band integrals

For $dN/dE = \phi_0 (E/E_0)^{-\Gamma}$ and $x = E/E_0$:

$$
F_{\rm ph} = \phi_0E_0\int_{x_1}^{x_2}x^{-\Gamma}dx,
\qquad
F_E = \phi_0E_0^2\int_{x_1}^{x_2}x^{1-\Gamma}dx .
$$

**Γ = 2 is the singular case** of $F_E$: the generic $[x^{2-\Gamma}]/(2-\Gamma)$ is 0/0 there, and
the code uses the limit explicitly,

$$
F_E\big|_{\Gamma=2} = \phi_0 E_0^2 \ln\frac{E_{\max}}{E_{\min}}, \qquad
F_{\rm ph}\big|_{\Gamma=2} = \phi_0E_0^2\left(\frac1{E_{\min}}-\frac1{E_{\max}}\right).
$$

The reported SED point is $E^2\,dN/dE$ at $E_{\rm dec}=\sqrt{E_{\min}E_{\max}}$.

### 5.2 Luminosity

$$
L_{[E_{\min},E_{\max}]} = 4\pi d^2\,F_E\,(1+z)^{\Gamma-2},
\qquad
\phi_0(L_0,d) = \frac{L_0}{4\pi d^2\,(1+z)^{\Gamma-2}\;E_0^2\,I_2},
$$

with $I_2 = \int_{x_1}^{x_2} x^{1-\Gamma}dx$; the inverse is what the 3D injector uses.

- $L$ is the **isotropic-equivalent luminosity in the analysis band**, not bolometric.
- $(1+z)^{\Gamma-2}$ is the power-law K-correction, exactly 1 at Γ = 2 (off by default).
- $z(d)$ is Planck18, through a cached table.

### 5.3 Worked example: 0.6–20 TeV, Γ = 2, $\phi_0 = 10^{-12}$ cm⁻² s⁻¹ TeV⁻¹

| Quantity | Value |
| --- | --- |
| photon flux $F_{\rm ph}$ | 1.617 × 10⁻¹² cm⁻² s⁻¹ |
| energy flux $F_E$ | 5.618 × 10⁻¹² erg cm⁻² s⁻¹ |
| $E^2dN/dE$ at $E_{\rm dec}$ = 3.46 TeV | 1.602 × 10⁻¹² erg cm⁻² s⁻¹ |
| $L$ at $d$ = 1123 / 1566 / 1849 Mpc (S240615dg PE 5/50/95%) | 0.85 / 1.65 / 2.30 × 10⁴⁵ erg s⁻¹ |

### 5.4 What the luminosity does *not* include: EBL

The power law is the spectrum **arriving at Earth**, so $L_0$ is the observed (absorbed) luminosity,
not the intrinsic one. For S240615dg ($z\approx0.29$, Domínguez et al. 2011) the optical depth is
2.3 at 0.6 TeV, 3.6 at 1 TeV and 7.5 at 5 TeV: only ~10% of the flux survives at the bottom of the
band. The emission-model limit applies EBL at each drawn $z_i$ (§11).

---

## 6. D: Monte-Carlo realisations

*Code: `simulate.py` Sections 3–5.*

### 6.1 One realisation

Place a point source with spectrum $\phi_0(E/E_0)^{-2}$ at a position (§6.3, §6.4), Poisson-sample
the counts (`dataset.fake(seed)`), remove the model, and record $\Lambda$ and its position.

### 6.2 Background-only realisations

$\phi_0 = 0$, $N = 5000$: the null distribution, the p-value (§4.4), and the median
$\Lambda_{\rm bkg}$ used as reference when the data under-fluctuate. A split-half comparison of
medians checks its stability.

### 6.3 2D sampler (fixed flux)

$\phi_0$ is fixed; the position is a HEALPix pixel drawn $\propto P_j$ among pixels whose nearest bin
is in $\mathcal{M}_{95}$, injected at the pixel centre. The 2D limit is therefore **conditional on
the source lying in the observed part of the 95% region**, whose coverage (`frac_gw_in_mask`)
should be quoted with it.

### 6.4 3D sampler (fixed luminosity)

$L_0$ is fixed. A bin $b \sim \tilde p_b$ over the **full** map (or $\mathcal{M}_{95}$ with
`restrict_to_mask_3d = True`), a distance $d = F_b^{-1}(u)$ by inverse CDF, the amplitude
$\phi_0(L_0, d)$ of §5.2 (round trip checked to machine precision), injected at the bin centre.
Each realisation is one hypothetical universe: the counterpart at a GW-allowed place and distance,
with luminosity $L_0$. The limit is conditional on the source lying inside the analysis field.

### 6.5 Common random numbers and seeding

Realisation $i$ uses fake seed `base_seed + i`, so parallel runs (`N_JOBS`) give the same $\Lambda$
sample as sequential ones. The upper-limit fit (§7.3) needs every realisation independent: a limit's
realisation $i$ uses `seed_fit + i` (`seed_fit` = 1 000 000, clear of the background block
`0 … n_sim_bkg - 1`, which enters the same fit), and round $r$ draws positions with `[12345, r]`.
Closure tests use their own seeds (900 000 for the background, 987 654 for sky, distance and angle).

---

## 7. E: From Λ distributions to an upper limit

*Code: `simulate._fit_ul`, `run_fit_ul`, `run_fit_ul_3d`.*

### 7.1 Definition

With $x$ the injected strength ($\phi_0$ in 2D, $L_0$ in 3D),

$$
f(x) = P\big(\Lambda_x > \Lambda^\star\big), \qquad
x_{\rm UL}: \quad f(x_{\rm UL}) = \mathrm{CL} = 0.95,
$$

with $\Lambda^\star = \Lambda_{\rm obs}$ if $S\ge0$ and $\operatorname{median}(\Lambda_{\rm bkg})$ if
$S<0$. A counterpart at $x_{\rm UL}$ would have given a larger $\Lambda$ than observed in 95% of
GW-consistent realisations: a Neyman-type construction on the global statistic, trials factor
included. Raising $\Lambda^\star$ to the background median on an under-fluctuation stops the limit
from beating the median sensitivity, as in power-constrained limits (Cowan et al. 2011).

### 7.2 What sets the value of the limit

**2D.** $f(\phi_0)$ averages over positions with different exposure, background and GW penalty
$-2\ln p_b$. Reaching 0.95 needs detection almost everywhere, so the limit is set by the least
favourable ~5% of the probability.

**3D.** If the single-position detection probability were a step at flux $F^\star$, then
$f_{\rm 3D}(L_0) = P(d < \sqrt{L_0/4\pi F^\star})$ and

$$
L_0^{\rm UL}\approx 4\pi\,d_{95}^2\,F^\star :
$$

the 3D limit is set by the **far tail** of the distance posterior, not its median. (The notebook's
$\langle 1/d^2\rangle$ printout describes the mean injected flux, which near draws dominate; the
exclusion fraction is limited by the far ones.)

**Coverage cap.** A source drawn outside $\mathcal{M}_{95}$ hardly raises $\Lambda$, so with full-map
sampling $f_{\rm 3D}\lesssim P(\mathcal{M}_{95}\mid\text{field})$. Below 0.95 no limit exists (the
bracket check reports it); between 0.95 and 1, $f$ flattens below 1 and the fit of §7.3 measures the
ceiling as $c$.

### 7.3 Fit of the detection probability

*Figure: `plotting.plot_ul_fit`.*

Every realisation $i$ gets its **own** strength $x_i$ (and position, distance, angle) and records
$y_i = \mathbf 1[\Lambda_i > \Lambda^\star]$, a Bernoulli draw with probability

$$
f(x) = f_0 + (c - f_0)\,\Phi\!\left(z_0 + \frac{\log_{10} x - \log_{10} x_{\rm UL}}{s}\right),
\qquad z_0 = \Phi^{-1}\!\left(\frac{\mathrm{CL} - f_0}{c - f_0}\right),
$$

so $f(x_{\rm UL}) = \mathrm{CL}$ by construction and the limit is a fit parameter. Nuisances: $f_0$
(no signal, pinned by the background sample in the same likelihood), $c$ (ceiling, 1 unless capped),
$s$ (transition width in dex):

$$
\ln L = \sum_i \Big[y_i \ln f(x_i) + (1-y_i)\ln\big(1-f(x_i)\big)\Big] + k\ln f_0 + (N_{\rm bkg}-k)\ln(1-f_0) .
$$

**Uncertainty.** The profile likelihood of $\log_{10}x_{\rm UL}$ gives 1σ ($-2\Delta\ln L = 1$) and
2σ (= 4) intervals: the Monte-Carlo uncertainty from the finite number of realisations, not a
systematic. Every derived flux or luminosity shares the relative interval. They go to the table,
figures and JSON (`*_1sigma`, `*_2sigma`).

**Rounds.**

1. *Scout*: `n_scout` = 500 realisations, log-uniform over the bracket. The crossing must fall
   inside, else a bracket error.
2. *Refine*: up to `n_refine` = 1000 realisations where the linear predictor runs from −2.5 to
   $z_0 + 1.5$, plus 10% just above to measure $c$; the fit uses everything from the window's lower
   edge up.
3. *Stop* when the 1σ half-width ≤ `precision`, or at `max_sims` = 6000 or `max_rounds` = 6. The
   notebook asks for 0.01 dex, which needs ~9× more realisations than 0.03, so it usually stops on
   the budget at ~±0.03 dex with `converged = False`: the requested precision was not reached, but
   the quoted interval is still the true MC uncertainty.

Guards: a transition narrower than three point spacings never counts as converged (a sparse fit can
collapse into a near-step), the next window is never narrower than ten spacings, and a crossing
outside the window is re-located with every realisation. Rounds are cached (`cache_path`), so an
interrupted run resumes.

**Goodness of fit.** Hosmer & Lemeshow (1980) on equal-count groups. With thousands of realisations
it can flag shape differences that do not bias the limit; a small p-value is a prompt to check the
pulls near $x_{\rm UL}$ and the closure test, not a failure.

**Validation.** 200 fits per synthetic curve of known limit, none of them a probit, with the
production settings. Pull = $(\log_{10}x_{\rm UL} - \log_{10}x_{\rm true})$ / 1σ half-width:

| curve $f(x)$ | mimics | 1σ coverage | 2σ coverage | mean pull | pull s.d. | realisations per fit |
| --- | --- | --- | --- | --- | --- | --- |
| $\Phi(x/x_0)$ | one position | 0.75 | 0.99 | −0.16 ± 0.06 | 0.83 | 2600 |
| $\langle\Phi(3xA/x_0)\rangle_A$, $A$ log-normal (σ = 0.5) | 2D: exposure spread | 0.72 | 0.99 | −0.08 ± 0.06 | 0.87 | 3650 |
| $\langle\Phi(x/d^2)\rangle_d$, $d$ log-normal (σ = 0.35) | 3D: distance spread | 0.66 | 0.96 | −0.06 ± 0.07 | 1.05 | 4940 |
| the same, $c$ = 0.995 | 3D coverage cap | 0.65 | 0.93 | +0.48 ± 0.07 | 0.99 | 5440 |
| the same, $c$ = 0.985 | 3D coverage cap | 0.56 | 0.88 | +0.62 ± 0.15 | 2.18 | 5480 |

Curves reaching 1: unbiased to 0.16σ, nominal coverage (0.683 / 0.954) or better. **Curves that
saturate below 1** are the weak spot: the limit sits where the curve still climbs towards $c$, comes
out ~0.5σ high (conservative), and the 1σ interval covers only 56–65%. The ceiling itself is measured
well (fitted 0.987 for a true 0.985), and the run prints a note when $c$ is significantly below 1.
Then treat the interval as a lower bound and rely on the closure test, or set
`restrict_to_mask_3d = True`. Without the $c$ parameter the capped curves came out +1.05σ and +6.3σ
high.

**Diagnostics** (`plot_ul_fit`, six panels): every realisation's $\Lambda$ against its $x$; the binned
fraction with the fit and its bands over the whole range and around the limit; the likelihood
profile against its Gaussian approximation; the estimate per round; the pulls.

### 7.4 (removed)

Earlier bisection and Slurm grid-scan versions of the UL search are in the git history before
Nov 2026.

### 7.5 Closure test (3D)

$x_{\rm UL}$ is read off a fitted curve, never simulated itself. $N$ fresh realisations exactly at
$L_0^{\rm UL}$, with independent seeds, must satisfy (`simulate.closure_pull`)

$$
\text{pull} = \frac{f_{\rm closure}-\mathrm{CL}}{\sqrt{\sigma_f^2 + \sigma_{\rm UL}^2}},\qquad |\text{pull}|<3 ,
$$

with $\sigma_f$ the binomial error and $\sigma_{\rm UL} = \tfrac12|f(x_{\rm UL}^{+1\sigma}) - f(x_{\rm UL}^{-1\sigma})|$
the limit's own uncertainty through the fitted curve. At $N = 1000$ the two are comparable.

### 7.6 Trials-factor sanity check

The quoted 2D limit is the **agnostic 95% with GW-weight** value: it is built from the distribution
of $\Lambda = TS + 2\ln p_{\rm GW}$ over $\mathcal{M}_{95}$, so it pays the trials factor. The
per-pixel sky-map ULs do not. The per-pixel references are the **hotspot** UL at the pixel where
$\Lambda$ is largest, the **agnostic 95%** UL, the largest per-pixel UL in $\mathcal{M}_{95}$, and
optionally the **agnostic full map** UL over all pixels. Because the agnostic 95% value is the
maximum over a region that contains the hotspot pixel,
$\mathrm{UL}_{\rm hotspot}\le\mathrm{UL}_{\rm agnostic\,95\%}$ by construction; a hotspot value above
it means the two are not taken from the same map, mask or smoothing. A positive background
fluctuation raises the observed counts and hence the UL at the hotspot pixel, so the hotspot can
coincide with (or approach) the agnostic maximum even though its *expected* sensitivity is good.

The naive expectation is that the GW-weighted limit sits **at or above** the agnostic 95% maximum.
For S240615dg it is **17% below** (§8.1); the hotspot lies below that maximum, at a different pixel.
Depending on the data set the hotspot can also coincide with the agnostic value: in the current mono
run the $\Lambda$-max pixel is also the largest-UL pixel, so both are
$3.58\times10^{-12}$ cm$^{-2}$ s$^{-1}$. Both orderings satisfy
$\mathrm{UL}_{\rm hotspot}\le\mathrm{UL}_{\rm agnostic\,95\%}$. The comparison with the GW-weighted
limit is not like for like — it maximises over a region with GW weights, whereas each sky-map UL is a
single fixed pixel — so this need not be a bug. Before quoting, check $\kappa$ and compare with a
GW-weighted quantile of the per-pixel ULs instead of their maximum.

---

## 8. F: Reporting both limits both ways

*Code: `simulate.summarize_upper_limits`, `luminosity_ul_from_flux_ul`, `flux_ul_from_luminosity_ul`.*

The 2D limit is a flux and needs a distance to become a luminosity; the 3D limit the reverse. The
5–95% distance range of S240615dg spans a factor ~1.7 (~3 in $L\propto d^2$), so each limit is
reported over the quantiles $d_q = F^{-1}(q)$, $q$ = 5, 50, 95%, of the field-marginal $F(r)$ (§3.4):

$$
\text{2D}\to L:\quad L_q = 4\pi d_q^2\,F_E^{\rm UL}\ \ (\text{also at } d_{\rm rms});
\qquad
\text{3D}\to F:\quad \phi_{0,q} = \phi_0(L_0^{\rm UL}, d_q).
$$

A consistency block prints $L_0^{\rm 3D}/L^{\rm 2D}(d_{50})$. Expected reasons for a gap: the sky
support (full map vs mask, §7.2), the far-tail dominance of the distance marginalisation, or, as a
bug, a mismatched index, band, $\Lambda^\star$ or CL.

Every quoted number (Λ, significance, each limit with its 1σ/2σ interval, convergence, goodness of
fit, settings, and for the model limit the full `EmissionModelInjection` configuration and merger
time) goes to `outputs/results/<source_name>/<stem>_upper_limits.json`.

### 8.1 Results for S240615dg

Last run: LST-1+MAGIC stereo, runs 17821–17825, `pybkgmodel` background, 0.6–20 TeV, Γ = 2,
95% CL; GTIs 15.4–17.0 h after the merger.

| Quantity | Value |
| --- | --- |
| $\Lambda_{\rm obs}$ / background median | −4.52 / −4.07 |
| p-value, significance | 0.592 ± 0.007, **−0.23σ**: an under-fluctuation, so $\Lambda^\star$ = background median |
| GW probability in the field / in $\mathcal{M}_{95}$ | 98.9% / 95.0% |
| Containment factor $\kappa$ | 0.682 |
| Distance, field-marginal skymap (5/50/95%) | 1028 / 1422 / 1817 Mpc |
| **2D flux UL** | $\phi_0 = 2.34\times10^{-12}$ cm⁻² s⁻¹ TeV⁻¹ (−2.9%/+3.2% MC, 1σ); photon flux $3.78\times10^{-12}$ cm⁻² s⁻¹; energy flux $1.31\times10^{-11}$ erg cm⁻² s⁻¹ |
| 2D as a luminosity at 5/50/95% distance | 1.66 / 3.18 / 5.19 × 10⁴⁵ erg s⁻¹ |
| **3D luminosity UL** (power law, no EBL) | $L_0 = 3.85\times10^{45}$ erg s⁻¹ (−5.0%/+5.5%); closure +1.36σ |
| 3D / 2D at the median distance | 1.21, inside the 2D range |
| **3D UL with the emission model** (`gti_mean`, Franceschini17 EBL, joint PE draw) | $L_k = 1.99\times10^{47}$ erg s⁻¹ (−7.9%/+8.9%); closure +1.06σ |
| Model / power-law 3D limit | 51.7 |

All three used 5500 realisations and stopped on the budget (`converged = False`). The factor 52
between the model and power-law limits combines everything the model changes: EBL at the drawn $z$
(0.22–0.34), the softer index (2.2), the rest-frame band, the light curve across the runs, and the PE
distances (median 1564 Mpc) against the skymap ones (1422 Mpc). Its fitted ceiling is $c = 0.994$, so
its 1σ interval may undercover (§7.3). The skymap route (distances from the skymap, angles from
$p(\theta_v\mid d)$) gave $1.33\times10^{47}$ erg s⁻¹.

---

## 9. The emission model

*Full write-up: [`emission_model.md`](emission_model.md). Code: `grb_model.py`,
`grb_phenomenological.py`. Guidelines Steps 1, 2, 5 and 7.*

The power-law limits inject a constant Γ = 2 spectrum. A real counterpart, such as a GRB afterglow
seen off-axis, has a spectrum and light curve that depend on energy, time since the merger and
viewing angle. The model is stored as a rest-frame photon luminosity $L(E', t';\theta)$
[ph s⁻¹ GeV⁻¹] per angle, with no distance and no EBL, and each realisation projects it to its own
$(d_i, z_i)$:

$$
F_i(E,t) = \frac{(1+z_i)^2}{4\pi d_i^2}\;L\big((1+z_i)E,\;t/(1+z_i);\theta_i\big)\;e^{-\tau(E,z_i)},
$$

with Franceschini & Rodighiero (2017) EBL and Planck18 cosmology.

The injected model is **one fixed jet**, the benchmark: the recipe of Nava (2020) and Abe et al.
(2026) with every input at its population median ($E_{\gamma,\mathrm{iso}} = 2\times10^{50}$ erg,
$\theta_\mathrm{core} = 14°$, $\Gamma_\mathrm{core} = 200$, decay index −1.45, photon index 2.2),
Gaussian in energy and Lorentz factor, seen off-axis through the element integral of Abe et al.
(2026) Sec. 3.4, tabulated at 0°, 1°, …, 90°, and interpolated in $\log L$ between them. Off-axis
the peak comes later and is much fainter (at 11 h, $10^{-5}$ of on-axis at 50°); the spectrum
keeps its shape. It reproduces the shapes of the five INAF catO5 files, which are the same recipe
run on five events.

---

## 10. The viewing-angle distribution

*Full write-up: [`viewing_angle.md`](viewing_angle.md). Code: `gw_pe.py`, `angle_distribution.py`.
Guidelines Step 4.3.*

GW parameter estimation measures $\theta_{JN}\in[0°,180°]$; the jet is bipolar, so
$\theta_v = \min(\theta_{JN},\,180°-\theta_{JN})$. The GW amplitude constrains mainly
$\Theta(\theta)/d$, so distance and angle are strongly correlated (for S240615dg,
$\mathrm{corr}(d,\cos\theta_v) = 0.82$; the median of $p(\theta_v\mid d)$ runs from 49.5° at 1123 Mpc
to 13.6° at 1849 Mpc) and must be drawn together.

A GWTC PE release (GBs) is reduced to `data/gw_pe/standardized/<superevent>.h5` (~1 MB, tracked):

| alert | GW name | $d_L$ [Mpc], median [90%] | $\theta_v$ [deg], median [90%] | $P(\theta_v<30°)$ |
| --- | --- | --- | --- | --- |
| S240615dg | GW240615_113620 | 1566 [1123, 1849] | 24.4 [6.5, 51.1] | 0.65 |
| S241125n | GW241125_010116 | 4748 [2600, 8611] | 55.8 [15.6, 86.0] | 0.17 |

Both are binary black holes: they exercise the method, no afterglow is expected.

**Skymap vs PE distance.** Drawing $d_i$ from the alert skymap and $\theta_i$ from the PE conditional
mixes two posteriors that disagree for S240615dg (skymap 1421 [1028, 1815] Mpc, PE 1566 [1123, 1849]
Mpc), which moves the $\theta_v$ median by about **+8°**. The emission-model limit therefore takes
(RA, Dec, $d$, $\theta_v$) together from one PE sample by default (§11.1).

**Without PE**, the options are isotropic ($\sin\theta$, median 60°), the GW-detected population of
Schutz (2011) (median 36°), a selection model with an explicit horizon, and fixed placeholders. The
S240615dg PE (median 24.4°) is far more face-on than either prior.

---

## 11. The 3D limit with the emission model

*Code: `simulate.EmissionModelInjection`, `sample_injections_3d`, `run_fit_ul_3d`; notebook section
"3D luminosity upper limit with the emission model" (`RUN_MODEL_3D`).*

### 11.1 One realisation

The construction is the same as for the power-law 3D limit (§6.4), with the same Λ (§4.3) and the
same fit (§7.3). Only the injected source changes:

1. draw a sky position, a distance $d_i$ and a viewing angle $\theta_i$. By default all three come
   from **one PE posterior sample** (`SKY_DISTANCE_MODE = THETA_MODE = "pe"`,
   `simulate.sample_sky_distance_angle_pe`), an exact draw from $p(\Omega, d, \theta_v\mid\mathrm{data})$
   with no density estimate. Samples outside the analysis geometry (or the 95% mask with
   `restrict_to_mask_3d`) are dropped, which conditions the posterior on the sky support as the
   skymap route does by renormalising `prob_gw`. The source is injected at the centre of the bin
   its position falls in. The other options are in §11.3;
2. take the benchmark jet at $\theta_i$ and project it to $(d_i, z_i)$ with EBL (§9);
3. for each run $j$, average the observer-frame spectrum over that run's GTIs (time since the
   merger) and fold it through run $j$'s own exposure, PSF and energy dispersion; sum the runs;
4. scale the result to the trial luminosity $L_k$ (§11.2), then Poisson-fake the counts and
   compute Λ.

**No temporal template is needed (Step 7).** The analysis is time-integrated, so each run's
counts depend only on its time-averaged spectrum. Averaging per run and injecting run by run is
therefore exact. A fading source automatically weights the early runs more.

**The test statistic does not change.** Λ always weights by the skymap $p_{\rm GW}$ the real data
were analysed with; only the injected population follows the PE.

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
20-min runs at 2–4 h, no EBL), but by about six orders of magnitude with `"anchor"`; the isotropic
and event-specific medians then differ by seven orders of magnitude, so the choice must be quoted
with the limit.

### 11.3 Options and the angle distribution

| Notebook option | Default | Meaning |
| --- | --- | --- |
| `model_dir_3d` | `.../grb_afterglow_phenomenological/benchmark` | any `GRBModelSet` cache: a one-sigma variant, or the catO5 set |
| `SKY_DISTANCE_MODE` | `"pe"` | `"pe"`: one PE sample per realisation (`PE_SOURCE`, `data/gw_pe/standardized/<alert>.h5`); `"skymap"`: a skymap bin and its distance ansatz (§3.4), the same draws as the power-law 3D limit for the same seed |
| `THETA_MODE` | `"pe"` | see the table below |
| `theta_dist_path` | `data/gw_input/<alert>_theta_distribution.npz` | the alert's $p(\theta_v\mid d)$, read only for `THETA_MODE = "gw"` |
| `MODEL_NORMALISATION` | `"gti_mean"` | §11.2 |
| `EBL_REFERENCE` | `"franceschini17"` | any gammapy EBL model, as used by Abe et al. (2026); `"dominguez"` for Domínguez et al. (2011). The tables ship in `data/ebl/`, which `gwuls.paths` sets as `$GAMMAPY_DATA` if it is unset |
| `lum_lo_model`, `lum_hi_model` | $10^{42}$, $10^{58}$ erg s⁻¹ | first-batch span, wider than for the power law because EBL removes most of the flux above ~1 TeV |

| `THETA_MODE` | $\theta_i$ | depends on $d_i$? | what the limit means |
| --- | --- | --- | --- |
| `"pe"` | the angle of the same PE sample as (sky, $d_i$); needs `SKY_DISTANCE_MODE = "pe"` | yes, exactly | **standard case**: the limit for this event, marginalised over its joint posterior |
| `"gw"` | $p(\theta_v\mid d_i)$ of the alert (PE KDE, §10) | yes, sky-averaged | cross-check of the KDE; with `"skymap"` distances it carries the bias of §10 |
| `"fixed"` (`THETA_FIXED_DEG`) | that angle | no | the jet at a given angle; quote a few (0°, 14°, 30°, 45°) as a curve against $\theta_v$ |
| `"isotropic"` | $\sin\theta_v$ | no | random orientation, no GW information: a conservative bracket, not the standard case |
| `"schutz"` | Schutz (2011), §10 | no | a typical GW-detected source; fallback without PE |
| `"two_bin"` (`THETA_TWO_BIN`) | uniform below and above a threshold, mass $p$ below | no | a placeholder, no claim about the shape |
| `"selection"` (`THETA_HORIZON_MPC`) | detected population with that horizon, given $d_i$ | yes, population-level | GW selection without event information; fallback without PE |

With `"skymap"` the angle uses its own random stream, so the sky and distance draws do not change
with `THETA_MODE`. Without released PE only `"skymap"` works. Every cache name of the model limit
carries the choice (`_skype_thetape`, `_skyskymap_thetafixed-20`, …) and the modification time of
the PE or $p(\theta_v\mid d)$ file it reads, and the JSON export records `sky_distance_3d_model`,
`theta_mode_3d_model` and `theta_spec_3d_model`.

### 11.4 Checks

Checks in the notebook, before the fit:

1. **Run by run = stacked**, for a constant power law: 252.9 against 254.0 expected counts
   (−0.41%). The small difference comes from the exposure-weighted PSF and energy dispersion of
   the stacked dataset.
2. **Step 6 closure**: over 2000 draws, the rest-frame GTI-mean luminosity rebuilt from the
   injected spectra (EBL divided out) is $1.00003\,L_k$.
3. **What is injected**: the drawn angles, where the GTIs fall on the light curve, the run weights,
   and the EBL transmission at the drawn $z$.
4. **The draws against the posterior**: for S240615dg, $d$ 5/50/95% = 1121/1564/1850 Mpc and
   $\theta_v$ = 6.4°/24.5°/51.1°, KS 0.007 and 0.005 against the PE samples in the same sky support,
   corr$(d,\cos\theta_v)$ 0.824 against 0.823. The skymap route with the PE's $p(\theta_v\mid d)$
   puts the median angle 8.1° further off-axis (§10).

After the fit, the closure test at $L_k^{\rm UL}$ (§7.5) gives a pull of +1.06σ. It also shows
which draws set the limit: the fraction above the target per bin of $\theta_v$ and of $d$.

### 11.5 Still open

- **Convergence.** All three S240615dg limits stopped on the 6000-realisation budget
  (`converged = False`) at a 1σ half-width of 0.013–0.036 dex instead of 0.01. Raise
  `ul_max_sims`, or quote the wider interval.
- **Trials-factor check** (§7.6): the global 2D limit is 17% below the largest per-pixel UL in the
  95% region. Compare against a GW-weighted quantile of the per-pixel ULs before quoting.
- **Power-law 3D limit still uses the skymap.** Only the emission-model limit draws from the PE,
  so the model / power-law ratio also carries the difference between the two distance posteriors.
- **One fixed jet.** The one-sigma variant caches can be used through `model_dir_3d`, but no loop
  over them, nor a per-realisation population draw, is implemented
  ([`emission_model.md`](emission_model.md) §5.3).
- **Per-position statistic.** The guidelines' Step 8 alternative,
  $P_{\rm excl}(L_k)=\frac1N\sum_i\mathbf{1}[\mathrm{TS}_i(L_k)>\mathrm{TS}_{\rm obs}(\mathrm{RA}_i,\mathrm{Dec}_i)]$,
  is not implemented. Λ is used for every limit; the same coverage cap as §7.2 would apply.
- **Source class.** S240615dg is a binary black hole. The limit demonstrates the method, not a
  physical expectation. S241125n (also a BBH) has standardized PE but has not been run.

---

## 12. Validation inventory

| Check | Where | Guards against | Last recorded result |
| --- | --- | --- | --- |
| Band integrals vs quadrature and gammapy; round trips; Γ = 2 limit from both sides; K = 1 at Γ = 2; $L\propto d^2$ | `simulate.check_spectral_conversions` | the Γ = 2 singularity, unit errors | passes (run in notebook) |
| Input `.pkl` validation: shapes, empty mask, κ range, CDF transposed / non-monotonic / truncated | `simulate.validate_simulation_input` | silent failures thousands of iterations later | runs at `.pkl` write and at every load |
| HEALPix → WCS assignment separation < 1.5 bin diagonals | `_map_healpix_to_wcs_bins` | RA-convention errors | printed per run |
| 3D sampler: χ² sky marginal, KS distance marginal, hottest-bin conditional, no $d\le0$, grid-edge probability atoms | `simulate.check_3d_sampling` | the production sampler drifting from the posterior | S240615dg: χ²/dof 855/849 (p = 0.43), KS p = 0.62, 0 draws at $d\le0$: pass |
| Background Λ split-half medians; binomial p-value error | notebook Part 2 | an unstable null distribution | S240615dg: −4.062 / −4.080 |
| Upper-limit fit: bracket check after the scout, goodness of fit, profile-likelihood interval | `_fit_ul`, `plotting.plot_ul_fit` | a limit outside the bracket, a curve the probit misdescribes | per run |
| Upper-limit fit: bias and coverage on synthetic curves of known limit, including non-probit shapes and a ceiling below 1 | §7.3 | a biased limit, an uncertainty that does not cover | 200 fits per curve: unbiased to 0.16σ with nominal coverage when the curve reaches 1; ~0.5σ high and under-covering when it saturates below 1 (§7.3) |
| Closure test at $L_0^{\rm UL}$ and $L_k^{\rm UL}$ with independent seeds | notebook, `closure_pull` | a biased limit | pull < 3σ required; S240615dg: +1.36σ (power law), +1.06σ (model) |
| Global UL ≥ per-pixel sky-map UL | notebook | a missing trials factor, wrong κ | S240615dg: global 17% **below** the 95%-region maximum, open (§7.6) |
| Model round trip Step 2 → Step 7 on the catO5 files | `extra/catO5_models` | frame or $(1+z)$ power errors | max relative error 0 |
| Standardised PE vs full raw posterior; fold vs pesummary `viewing_angle` | `2_gw_viewing_angle` Part 0 | subsampling bias | 5/50/95% quantiles within ~1°; fold exact |
| KDE marginals vs samples; grid $p(\theta\mid d)$ vs grid-free resampling; bandwidth ×0.5 and ×2 | Part 1 | over- or under-smoothing | grid vs resampling medians within 0.5°; bandwidth moves medians by ≤ 1.5° |
| Three sampling routes reproduce the posterior (quantiles, correlation, KS) | Part 1 | sampler bugs | KS ≤ 0.016 |
| Each option's sampler vs its own pdf | Part 2 | sampler bugs | at $1/\sqrt n$ level |
| Schutz vs Monte-Carlo selection model | Part 2 | a wrong closed form | max CDF difference 0.009 |
| `resolve_theta_distribution` for every `THETA_MODE`; malformed `"fixed:…"`, `"two_bin:…"`, `"selection:…"` and `THETA_MODE = "pe"` without PE sky/distance rejected at construction | §11.3, Step 4.3 | a silently wrong angle draw | medians: fixed 20°, isotropic 60.2°, Schutz 36.4°, S240615dg PE given $d$ = 1400 Mpc 33.9° |
| Joint PE draw: bin of every sample vs nearest bin centre; drawn $d$, $\theta_v$ and corr$(d, \cos\theta_v)$ vs the PE samples in the sky support; skymap route alongside | `sample_sky_distance_angle_pe`; notebook crosscheck (4) | a wrong bin, a biased resampling | S240615dg: 98.4% of samples in the geometry; KS 0.007 ($d$), 0.005 ($\theta_v$); corr 0.824 vs 0.823; skymap route +8.1° in the $\theta_v$ median |
| Emission model: run-by-run injection = stacked for a constant source; Step 6 closure of $L_k$ | §11.4 | wrong per-run folding or normalisation | −0.41%; 1.00003 |
| Benchmark jet: on-axis vs 1-D calculation, grid convergence, shape fits to catO5, angle interpolation vs direct build | `1_grb_model` | a wrong off-axis integral or interpolation | 0.0018 dex; catO5 shapes to 0.09–0.25 dex; interp ≤ 0.045 dex ([`emission_model.md`](emission_model.md) §5.4) |

---

## 13. References

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
  the orientation factor Θ (§10).
- B. F. Schutz, *Networks of gravitational wave detectors and three figures of merit*, CQG **28**,
  125023 (2011), [arXiv:1102.5421](https://arxiv.org/abs/1102.5421): inclination distribution of
  GW-detected binaries (§10).
- LVK, GWTC-5.0 parameter-estimation data release (Zenodo): the PE samples of §10.

**Emission model and propagation**

- H. Abe et al. (CTAO Consortium), *Chasing gamma-ray signals from binary neutron star
  coalescences with the Cherenkov Telescope Array: prospects and observing strategies*, ApJ
  **1004**, 46 (2026), [arXiv:2604.08748](https://arxiv.org/abs/2604.08748): the phenomenological
  recipe and its off-axis method (§9).
- L. Nava, *TeV counterparts of BNS mergers*, internal note (Jan 2020): the original recipe.
- G. P. Lamb and S. Kobayashi, *Low-Γ jets from compact stellar mergers*, MNRAS **472**, 4953
  (2017), [arXiv:1706.03000](https://arxiv.org/abs/1706.03000): the off-axis element integral.
- E. Berger, *Short-duration gamma-ray bursts*, ARA&A **52**, 43 (2014),
  [arXiv:1311.2603](https://arxiv.org/abs/1311.2603): the $L_X$–$E_\mathrm{iso}$ relation.
- A. Franceschini and G. Rodighiero, *The extragalactic background light revisited and the
  cosmic photon-photon opacity*, A&A **603**, A34 (2017),
  [arXiv:1705.10256](https://arxiv.org/abs/1705.10256): the EBL model of the emission-model limit.
- A. Domínguez et al., *Extragalactic background light inferred from AEGIS galaxy-SED-type
  fractions*, MNRAS **410**, 2556 (2011), [arXiv:1007.1459](https://arxiv.org/abs/1007.1459): the
  EBL model of §5.4 and the `grb_model` default.
- Planck Collaboration, *Planck 2018 results. VI. Cosmological parameters*, A&A **641**, A6 (2020),
  [arXiv:1807.06209](https://arxiv.org/abs/1807.06209): $z(d_L)$.

**Internal**

- *Simulation guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit*,
  internal note (Sep 2026): Steps 1–8 referenced throughout.
