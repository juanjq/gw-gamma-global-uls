# The emission model: a GRB afterglow $L(E', t'; \theta_v)$

What the emission-model limit of [`3_upper_limits.ipynb`](../notebooks/3_upper_limits.ipynb) injects,
how it is built, and how it was checked. Notebook: [`1_grb_model.ipynb`](../notebooks/1_grb_model.ipynb).
Code: [`gwuls/grb_phenomenological.py`](../gwuls/grb_phenomenological.py) (the jet) and
[`gwuls/grb_model.py`](../gwuls/grb_model.py) (the storage format, projection to the observer, EBL).
Two optional notebooks under [`notebooks/extra/`](../notebooks/extra/) cover the INAF catO5 files (§6)
and random draws from the recipe (§5.3).

**Contents**

1. [Overview](#1-overview)
2. [The on-axis recipe](#2-the-on-axis-recipe)
3. [Jet structure](#3-jet-structure)
4. [Seeing the jet off-axis](#4-seeing-the-jet-off-axis)
5. [The benchmark jet](#5-the-benchmark-jet)
6. [The catO5 files](#6-the-cato5-files)
7. [Format: rest-frame luminosity per angle](#7-format-rest-frame-luminosity-per-angle)
8. [Caveats](#8-caveats)
9. [References](#9-references)

**Notation.** Unprimed $E, t$ are observer-frame energy and time since the merger, primed $E', t'$
rest-frame. $F(E, t)$ is the photon flux [ph cm⁻² s⁻¹ GeV⁻¹], $L(E', t')$ the photon luminosity
[ph s⁻¹ GeV⁻¹]. A superscript $m$ marks where a model file was computed ($d_L^m$, $z^m$), a
subscript $i$ the values drawn in realisation $i$.

---

## 1. Overview

```
Nava (2020) / Abe et al. (2026) recipe, every input at its population median   (§2, §5)
   → Gaussian structured jet                                                    (§3)
   → seen off-axis by the element integral, 0°, 1°, …, 90°                       (§4)
   → rest-frame L(E', t'; θ_v), one .npz per angle                               (§7)
     data/models/grb_afterglow_phenomenological/benchmark/
   → in each realisation: L at θ_i, projected to (d_i, z_i) with EBL, averaged
     over each run's GTIs, normalised to the trial luminosity L_k              (pipeline.md §11)
```

The model is one fixed jet, the **benchmark**. It is **phenomenological**: the TeV luminosity comes
from empirical relations, not a radiation calculation, and no microphysics ($\varepsilon_e$,
$\varepsilon_B$) enters. Every constant in the code is cited to an equation or section of the
sources (the module's `SOURCES` dictionary).

---

## 2. The on-axis recipe

For an observer inside the jet core (Abe et al. 2026 Sec. 3, after Nava 2020):

1. **Prompt energy.** $E_\mathrm{peak}$ from a broken power law (Ghirlanda et al. 2016, Eq. 13); the
   Amati relation (their Eq. 15) gives $E_{\gamma,\mathrm{iso}}$.
2. **X-ray luminosity at 11 h** (Berger 2014):
   $L_{X,11h} = 8.5\times10^{43}\,(E_{\gamma,\mathrm{iso}}/10^{51}\,\mathrm{erg})^{0.83}$ erg s⁻¹,
   0.5 dex scatter.
3. **TeV luminosity at 11 h:** $L_{\mathrm{TeV},11h} = L_{X,11h}\times10^{\,\mathcal N(0,\,0.3)}$ in
   0.3–1 TeV.
4. **Peak time** = deceleration time of the core,
   $$
   R_\mathrm{dec} = \left(\frac{17\,E_k}{16\pi\, m_p c^2\, n\, \Gamma_0^2}\right)^{1/3},
   \qquad t_\mathrm{dec} = \frac{R_\mathrm{dec}}{2c\,\Gamma_0^2},
   $$
   with $E_k = E_{\gamma,\mathrm{iso}}(1-\eta_\gamma)/\eta_\gamma$, $\eta_\gamma = 0.2$ and
   $n = 0.1$ cm⁻³ (Abe et al. Eq. 6; Blandford & McKee 1976).
5. **Light curve:** a broken power law, $t^{\beta_1}$ before $t_\mathrm{dec}$ and $t^{\beta_2}$ after,
   with $\beta_1 = 2.0 \pm 0.05$ and $\beta_2 = -1.45 \pm 0.48$, normalised to pass through
   $L_{\mathrm{TeV},11h}$ at $t' = 11$ h.
6. **Spectrum:** $dN/dE \propto E^{-p}$, $p = 2.2 \pm 0.1$, constant in time.

Timing sets the peak (step 4); the 11 h anchor sets the brightness (steps 2–3).

---

## 3. Jet structure

Gaussian in energy and Lorentz factor (Abe et al. Eqs. 1–2):

$$
\frac{\varepsilon(\theta)}{\varepsilon_\mathrm{core}} = e^{-\theta^2/\theta_\mathrm{core}^2},
\qquad
\Gamma_0(\theta) = (\Gamma_\mathrm{core}-1)\,e^{-\theta^2/2\theta_\mathrm{core}^2} + 1 .
$$

$\theta_\mathrm{core}$ is lognormal around 14° (σ = 0.2 dex, Fong et al. 2015), $\Gamma_\mathrm{core}$
lognormal around 200 (σ = 0.2 dex, close to GRB 170817A, Ghirlanda et al. 2019). The jet is cut at
$\theta_\mathrm{max}$, where $\Gamma_0 = 5$ (40° for the benchmark). Past it,
$t_\mathrm{dec}\propto E^{1/3}\Gamma_0^{-8/3}$ would *decrease* with angle because $\Gamma_0$ is
floored at 1, so the cut is a validity limit as well.

---

## 4. Seeing the jet off-axis

The **element integral** of Abe et al. (2026) Sec. 3.4, after Lamb & Kobayashi (2017)
(`offaxis_luminosity`). Each jet element at angle $\theta$ from the axis is a small blast wave with
its own energy, $\Gamma_0(\theta)$ and $t_\mathrm{dec}(\theta)$. Along its velocity it shows the
recipe's light curve scaled by $\varepsilon(\theta)$; after deceleration it slows as
$\Gamma\propto t^{-3/8}$. An observer at angle $\alpha$ from its velocity sees it Doppler-shifted by
$a = (1-\beta)/(1-\beta\cos\alpha)$:

$$
L_\mathrm{obs}(t_\mathrm{obs}) = \sum_\text{elements}\frac{d\Omega}{\Omega_\mathrm{beam}(\Gamma)}\;a^{2+p}\;h(\theta, t),
\qquad \frac{dt_\mathrm{obs}}{dt} = \frac{1}{a},
\qquad \Omega_\mathrm{beam} = \int a^{2+p}\,d\Omega .
$$

- $a^{2+p}$ multiplies a power law, so **the spectrum keeps its shape off-axis**; only the light
  curve changes.
- With $\Omega_\mathrm{beam}$ as normalisation, a uniform spherical shell reproduces its own light
  curve for every observer.
- One constant normalises every angle, so that the **on-axis** 0.3–1 TeV luminosity at $t' = 11$ h
  is exactly $L_{\mathrm{TeV},11h}$.

**What it gives.** Within the core, the on-axis light curve. Out to $\theta_\mathrm{max}$, a peak from
the material on the line of sight. Beyond it nothing is on the line of sight: the observer waits for
the core to decelerate until its beaming cone ($\sim 1/\Gamma$) reaches it. For the benchmark the
peak is at $1.2\times10^3$ s at 30°, $7\times10^4$ s at 50°, $8\times10^5$ s at 57°, and beyond the
$10^7$ s grid at larger angles. At 11 h the flux is $10^{-5}$ of on-axis at 50° and $10^{-11}$ at 90°.

The simpler *local closure* of Nava (2020) Sec. 5 treats the element on the line of sight as its own
on-axis event. It agrees with the integral on the peak time to ~15% up to 30°, but has nothing to
evaluate past $\theta_\mathrm{max}$ ([`extra/stochastic_jet.ipynb`](../notebooks/extra/stochastic_jet.ipynb)).

---

## 5. The benchmark jet

### 5.1 Inputs

`BENCHMARK` holds the median of every population input:

| Input | Benchmark | Population | Reaches the $L_k$ limit? |
| --- | --- | --- | --- |
| $E_{\gamma,\mathrm{iso}}$ | $2.0\times10^{50}$ erg | $E_\mathrm{peak}$ broken power law + Amati | only through the time scale, $t\propto E^{1/3}$ |
| $\theta_\mathrm{core}$ | 14.1° | lognormal, σ = 0.2 dex | yes |
| $\Gamma_\mathrm{core}$ | 200 | lognormal, σ = 0.2 dex | yes, off-axis |
| $\beta_1$, $\beta_2$ | 2, −1.45 | normal, σ = 0.05, 0.48 | $\beta_2$ yes |
| photon index $p$ | 2.2 | normal, σ = 0.1 | yes |
| $L_{\mathrm{TeV},11h}$ | mean Berger (2014) relation, $L_\mathrm{TeV}/L_X = 1$ | 0.5 and 0.3 dex scatter | no, amplitude only |
| $n$, $\eta_\gamma$ | 0.1 cm⁻³, 0.2 | fixed | — |

### 5.2 Why one fixed jet loses little

Under the default normalisation (`MODEL_NORMALISATION = "gti_mean"`, guidelines Step 6) each
realisation is rescaled to the trial luminosity $L_k$ averaged over the GTIs. Anything that only
multiplies $L$ cancels, including the largest random terms of the recipe (the 0.5 and 0.3 dex
scatters, and the amplitude part of $E_{\gamma,\mathrm{iso}}$). Only inputs that change the **shape**
survive: $\theta_\mathrm{core}$, $\Gamma_\mathrm{core}$, $\beta_2$, the photon index, the time scale.

### 5.3 Population scatter

Random noise on the flux is not a valid substitute: multiplicative noise cancels, and shape noise
means nothing unless it comes from the recipe's own distributions. What exists:

- **Nominal** (what the limit uses): the benchmark, i.e. the population's median short GRB.
- **Systematic band:** `one_sigma_variants` moves one shape input to its 16th or 84th percentile;
  `SAVE_VARIANTS = True` in `1_grb_model` writes the ten caches (`theta_core_lo/`, …). Rerunning the
  limit with `model_dir_3d` pointed at them gives the band; the two $\theta_\mathrm{core}$ variants
  dominate off-axis.
- **Marginalised** (one bank member per realisation): not implemented.

[`extra/stochastic_jet.ipynb`](../notebooks/extra/stochastic_jet.ipynb) draws random events from the
recipe to show the population the benchmark stands for.

### 5.4 Validation

All in `1_grb_model.ipynb`:

| Check | Result |
| --- | --- |
| On-axis element integral vs an independent 1-D calculation | agrees; the arrival-time spread rescales rise and decay by known factors, and the 11 h normalisation removes the decay one |
| Grid convergence (240 α × 128 ψ × 24 per decade in time) vs twice as fine | notebook §2.2 |
| Shape fits to the five catO5 files ($\theta_\mathrm{core}$, $\Gamma_\mathrm{core}$, amplitude free) | four files to 0.09–0.14 dex rms, the 13.2° file to 0.25 dex; fitted $\theta_\mathrm{core}$ (6°–37°) within ~2σ of the population |
| No fit: the benchmark with each file's own $E_\mathrm{iso}$ and index | 0.4–0.9 dex rms for three files, 1.5–1.8 dex for two (a different core) |
| Reload; `get_model(θ, "interp")` between grid nodes vs a direct build | exact reload; within 0.05 dex |

The catO5 files look like ordinary draws from this recipe seen through this off-axis method. The
77.8° file matches in shape but sits 2 dex below the anchor in amplitude, which does not reach the
limit under Step 6.

---

## 6. The catO5 files

Five files from the CTA-GW O5 BNS catalogue (B. Patricelli), computed by L. Nava (Sept 2021) with
the same recipe. Each is **a different event**, not one jet seen from five angles. They are the
reference the benchmark is checked against (§5.4);
[`extra/catO5_models.ipynb`](../notebooks/extra/catO5_models.ipynb) converts them to the format of §7.

| file | $\theta^m$ | $d_L^m$ [Mpc] | $z^m$ | $E_\mathrm{iso}$ [erg] | $t'_\mathrm{pk}$ [s] |
| --- | --- | --- | --- | --- | --- |
| `catO5_301` | 9.1° | 843 | 0.170 | $5.3\times10^{48}$ | 3.0 |
| `catO5_26` | 13.2° | 287 | 0.062 | $1.2\times10^{48}$ | 5.3 |
| `catO5_1378` | 22.6° | 113 | 0.025 | $1.1\times10^{50}$ | 218 |
| `catO5_4` | 28.6° | 349 | 0.075 | $2.0\times10^{49}$ | 131 |
| `catO5_2094` | 77.8° | 545 | 0.114 | $1.4\times10^{50}$ | $3.2\times10^{5}$ |

Each file tabulates the intrinsic (no EBL) $F(E, t)$ at 41 energies (1 GeV–10 TeV) × 70 log time
bins (0.1 s–$10^6$ s), at the bin's geometric centre. Zero cells are removed by dropping whole rows
or columns (the 10 TeV row of every file; for 77.8° also the 6.3 and 7.9 TeV rows and the first
23 time bins, before its emission starts). Every spectrum is a single power law (index 2.07–2.23),
constant in time; every light curve has one achromatic peak.

They can be injected instead of the benchmark (`model_dir_3d` = the standardized catO5 directory),
but only at the tabulated angles (`THETA_MODE = "fixed"` at a file angle). Interpolating between
them would turn event-to-event scatter into angle dependence, and on such a coarse grid an
interpolation at fixed $t'$ blends two distant peaks into one broad, too-faint one.

---

## 7. Format: rest-frame luminosity per angle

### 7.1 Rest frame and back

Photons emitted in $dE'\,dt'$ arrive with $E = E'/(1+z)$, $t = (1+z)\,t'$, spread over
$4\pi d_M^2$ with $d_M = d_L/(1+z)$. Photon number is conserved and the Jacobian is 1, so

$$
F(E, t) = \frac{(1+z)^2}{4\pi d_L^2}\; L\big((1+z)E,\; t/(1+z)\big). \tag{1}
$$

A file computed at $(d_L^m, z^m)$ is moved to its rest frame by inverting Eq. (1) node by node,
an exact regrid with the K-correction built in:

$$
L(E'_n, t'_n) = \frac{4\pi (d_L^m)^2}{(1+z^m)^2}\, F(E_n, t_n),
\qquad E'_n = (1+z^m)\,E_n, \quad t'_n = \frac{t_n}{1+z^m}. \tag{2}
$$

Realisation $i$ projects it back to its own distance with EBL absorption (guidelines Step 7):

$$
F_i(E, t) = \frac{(1+z_i)^2}{4\pi d_i^2}\; L\big((1+z_i)E,\; t/(1+z_i)\big)\; e^{-\tau(E, z_i)}. \tag{3}
$$

EBL is applied only here, at the drawn $z_i$. The limit uses Franceschini & Rodighiero (2017),
the model of Abe et al. (2026) (`EBL_REFERENCE = "franceschini17"`); `to_observer` defaults to
Domínguez et al. (2011). The tables ship in `data/ebl/`, and `gwuls.paths` sets `$GAMMAPY_DATA` to
`data/` if it is unset. Cosmology: Planck18.

### 7.2 Evaluation and angles

`GRBModel.at(E', t')` interpolates $\log L$ bilinearly in $(\log E', \log t')$, exact for local power
laws; outside the grid it extrapolates as a power law from the edge cell (`to_observer` warns).
`GRBModelSet.get_model(θ, method)` gives the model at any angle:

- `"interp"` (default): $\log L$ linear in θ between the two bracketing angles at fixed $(E', t')$,
  on the union of their grids. Accurate to 0.05 dex on the 1° benchmark grid.
- `"nearest"`: the closest tabulated angle. Use it for coarse grids like catO5.

Angles outside the tabulated range are clipped to the edge (with a warning).

### 7.3 Files and use

`GRBModelSet.save` writes one `theta_<θ>deg.npz` per angle with `theta_deg`, `energy_GeV`, `time_s`
(rest frame), `L_ph_s_GeV` (shape $n_E \times n_t$), `E_iso_erg`, `D_L_m_Mpc`, `z_m` and `meta_json`.

```python
from gwuls import grb_model, paths

model_set = grb_model.GRBModelSet.load(paths.GRB_MODEL_PHENOMENOLOGICAL_DIR / "benchmark")
model_i = model_set.get_model(theta_i, method="interp")
E_obs, t_obs, F_i = model_i.to_observer(d_i, z_i, energy_obs=E_grid, time_obs=t_grid,
                                        apply_ebl=True, ebl_reference="franceschini17")
```

In the limit, `simulate.EmissionModelInjection` makes these calls, averages the model over each
run's GTIs and normalises it to $L_k$ ([`pipeline.md`](pipeline.md) §11). `GRBModel.from_analytic`
builds a separable $f(E')\,g(t')$ model in the same format.

---

## 8. Caveats

- **Element normalisation.** $d\Omega/\Omega_\mathrm{beam}$ is the standard point-source choice;
  Lamb & Kobayashi's exact convention is not given in either source. It matters most for late,
  mildly relativistic material, which may explain the amplitude of the 77.8° file.
- **Simple dynamics.** Blandford–McKee deceleration only: no lateral spreading, no non-relativistic
  transition (Γ is floored at 1), no counter-jet. These matter far off-axis at $t' \gtrsim 10^6$ s.
- **Smoothed on-axis peak.** The arrival-time spread lowers the rise (~4×) and rounds the peak.
  Follow-ups starting after about a minute see only the decay, which matches the recipe exactly.
- **Readings of the sources.** "11 h" is read as rest-frame time; the prefactor of
  $t_\mathrm{dec} = R_\mathrm{dec}/2c\Gamma_0^2$ is assumed; the width convention of Eq. 1 (no
  factor 2) is from Abe et al. (2026), whereas Nava (2020) has the factor 2. No Amati scatter.
- **Constant photon index** in time and angle, as in the recipe and the catO5 files.
- **Extrapolation** outside $t' \lesssim 10^7$ s or the 1 GeV–10 TeV band is a power-law continuation.
- **The model is for a BNS.** S240615dg is a binary black hole; it only exercises the method.

---

## 9. References

- H. Abe et al. (CTAO Consortium), *Chasing gamma-ray signals from binary neutron star coalescences
  with the Cherenkov Telescope Array: prospects and observing strategies*, ApJ **1004**, 46 (2026),
  [arXiv:2604.08748](https://arxiv.org/abs/2604.08748): the recipe (Sec. 3), jet structure
  (Eqs. 1–2), off-axis method (Sec. 3.4).
- L. Nava, *TeV counterparts of BNS mergers*, internal note (Jan 2020): the original recipe and the
  local closure (Sec. 5).
- G. P. Lamb and S. Kobayashi, *Low-Γ jets from compact stellar mergers*, MNRAS **472**, 4953 (2017),
  [arXiv:1706.03000](https://arxiv.org/abs/1706.03000): the element integral.
- R. D. Blandford and C. F. McKee, *Fluid dynamics of relativistic blast waves*, Phys. Fluids **19**,
  1130 (1976), [doi:10.1063/1.861619](https://doi.org/10.1063/1.861619): deceleration.
- G. Ghirlanda et al., *Short gamma-ray bursts at the dawn of the gravitational wave era*, A&A
  **594**, A84 (2016), [arXiv:1607.07875](https://arxiv.org/abs/1607.07875): $E_\mathrm{peak}$ and Amati.
- E. Berger, *Short-duration gamma-ray bursts*, ARA&A **52**, 43 (2014),
  [arXiv:1311.2603](https://arxiv.org/abs/1311.2603): $L_{X,11h}$–$E_{\gamma,\mathrm{iso}}$.
- W. Fong et al., *A decade of short-duration gamma-ray burst broadband afterglows*, ApJ **815**, 102
  (2015), [arXiv:1509.02922](https://arxiv.org/abs/1509.02922): $\theta_\mathrm{core}$.
- G. Ghirlanda et al., *Compact radio emission indicates a structured jet was produced by a binary
  neutron star merger*, Science **363**, 968 (2019), [arXiv:1808.00469](https://arxiv.org/abs/1808.00469):
  $\Gamma_\mathrm{core}$.
- A. Franceschini and G. Rodighiero, *The extragalactic background light revisited and the cosmic
  photon-photon opacity*, A&A **603**, A34 (2017), [arXiv:1705.10256](https://arxiv.org/abs/1705.10256).
- A. Domínguez et al., *Extragalactic background light inferred from AEGIS galaxy-SED-type
  fractions*, MNRAS **410**, 2556 (2011), [arXiv:1007.1459](https://arxiv.org/abs/1007.1459).
- Planck Collaboration, *Planck 2018 results. VI*, A&A **641**, A6 (2020),
  [arXiv:1807.06209](https://arxiv.org/abs/1807.06209).
- B. Patricelli et al., CTA-GW O5 BNS catalogue (internal): the events behind `catO5_*.fits`.
