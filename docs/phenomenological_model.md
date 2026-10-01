# The phenomenological GRB afterglow model

This document covers the emission model that the 3D upper limit of
[`sim_3d_model_CTAO_paper.ipynb`](../notebooks/sim_3d_model_CTAO_paper.ipynb) injects: the
structure of the jet, how it is seen off-axis, and why one fixed jet is enough. It covers two
notebooks, and the code for both is in [`gwuls/grb_phenomenological.py`](../gwuls/grb_phenomenological.py):

| Notebook | What it does | Used by `sim_3d`? |
| --- | --- | --- |
| [`setup_model_phenomenological_fixed.ipynb`](../notebooks/setup_model_phenomenological_fixed.ipynb) | Builds **one fixed jet** (the `BENCHMARK`) with every input at its population median, sees it off-axis with the element integral of Abe et al. (2026) Sec. 3.4, and saves it on a 1° grid in $\theta_v$ | **yes**: `model_dir_3d = .../grb_afterglow_phenomenological/benchmark` |
| [`setup_model_phenomenological.ipynb`](../notebooks/setup_model_phenomenological.ipynb) | Draws a fresh **random** event from the same recipe on every call, and compares the population with the five catO5 files | no, exploratory |

The output uses the `GRBModel` / `GRBModelSet` format of [`setup_model.md`](setup_model.md)
(rest-frame $L(E', t')$ per viewing angle). Projecting it to a distance, applying EBL, and
interpolating between angles therefore work exactly as described there.

---

## Contents

1. [Sources](#1-sources)
2. [The on-axis recipe](#2-the-on-axis-recipe)
3. [Jet structure](#3-jet-structure)
4. [Seeing the jet off-axis](#4-seeing-the-jet-off-axis)
5. [The benchmark jet](#5-the-benchmark-jet)
6. [Validation](#6-validation)
7. [Population scatter: variants and banks](#7-population-scatter-variants-and-banks)
8. [The stochastic version](#8-the-stochastic-version)
9. [Output and use in `sim_3d`](#9-output-and-use-in-sim_3d)
10. [Caveats](#10-caveats)
11. [References](#11-references)

---

## 1. Sources

The recipe comes from L. Nava's internal note *TeV counterparts of BNS mergers* (2020). It was
published, slightly refined, in Abe et al. (2026), the CTAO GW–GRB prospects paper. The five
`catO5_*.fits` files of [`setup_model.md`](setup_model.md) are this recipe run on five events of
B. Patricelli's O5 BNS catalogue: their FITS headers read `AUTHOR=Lara`, "Created Sept 2021".
Every constant in the code is cited to an equation or section of these sources (the module's
`SOURCES` dictionary).

The recipe is **phenomenological**. The TeV luminosity is set by empirical relations, not by a
radiation calculation, and no microphysics ($\varepsilon_e$, $\varepsilon_B$) enters.

---

## 2. The on-axis recipe

For an observer inside the jet core, Abe et al. (2026) Sec. 3 build the light curve as follows:

1. **Prompt energy.** $E_\mathrm{peak}$ is drawn from a broken power law (Ghirlanda et al. 2016,
   Eq. 13), and the Amati relation (their Eq. 15) gives $E_{\gamma,\mathrm{iso}}$.
2. **X-ray luminosity at 11 h**, from the short-GRB correlation of Berger (2014):
   $L_{X,11h} = 8.5\times10^{43}\,(E_{\gamma,\mathrm{iso}}/10^{51}\,\mathrm{erg})^{0.83}$ erg s⁻¹,
   with 0.5 dex scatter.
3. **TeV luminosity at 11 h:** $L_{\mathrm{TeV},11h} = L_{X,11h}\times10^{\,\mathcal N(0,\,0.3)}$, in
   0.3–1 TeV.
4. **Peak time** = the deceleration time of the core,
   $$
   R_\mathrm{dec} = \left(\frac{17\,E_k}{16\pi\, m_p c^2\, n\, \Gamma_0^2}\right)^{1/3},
   \qquad t_\mathrm{dec} = \frac{R_\mathrm{dec}}{2c\,\Gamma_0^2},
   $$
   with kinetic energy $E_k = E_{\gamma,\mathrm{iso}}(1-\eta_\gamma)/\eta_\gamma$, efficiency
   $\eta_\gamma = 0.2$, and density $n = 0.1$ cm⁻³ (Abe et al. Eq. 6; Blandford & McKee 1976).
5. **Light curve:** a broken power law $L\propto t^{\beta_1}$ before $t_\mathrm{dec}$ and
   $L\propto t^{\beta_2}$ after it, with $\beta_1 = 2.0 \pm 0.05$ and
   $\beta_2 = -1.45 \pm 0.48$ (from 22 short GRBs). Its normalisation is fixed by passing
   through $L_{\mathrm{TeV},11h}$ at $t' = 11$ h.
6. **Spectrum:** $dN/dE \propto E^{-p}$, with $p = 2.2 \pm 0.1$, constant in time.

The peak is set by timing (step 4) and the brightness by the 11 h anchor (steps 2–3). No
independent blast-wave luminosity is needed.

---

## 3. Jet structure

The jet is Gaussian in energy and in Lorentz factor (Abe et al. Eqs. 1–2):

$$
\frac{\varepsilon(\theta)}{\varepsilon_\mathrm{core}} = e^{-\theta^2/\theta_\mathrm{core}^2},
\qquad
\Gamma_0(\theta) = (\Gamma_\mathrm{core}-1)\,e^{-\theta^2/2\theta_\mathrm{core}^2} + 1 .
$$

The core width $\theta_\mathrm{core}$ is lognormal, centred on 14° with σ = 0.2 dex (Fong et al.
2015). $\Gamma_\mathrm{core}$ is lognormal, centred on 200 with σ = 0.2 dex, close to the
GRB 170817A value (Ghirlanda et al. 2019). Material with $\Gamma_0 < 5$ is cut off: the jet ends at
$\theta_\mathrm{max}$, where $\Gamma_0(\theta_\mathrm{max}) = 5$ (40° for the benchmark).
This cut is also a physical validity limit. Past it, $t_\mathrm{dec}\propto E^{1/3}\Gamma_0^{-8/3}$
would *decrease* with angle, because $\Gamma_0$ is floored at 1.

---

## 4. Seeing the jet off-axis

The fixed model uses the **element integral** of Abe et al. (2026) Sec. 3.4, after Lamb &
Kobayashi (2017) (`offaxis_luminosity`). Each element of the jet at angle $\theta$ from the axis is
a small blast wave with its own energy, $\Gamma_0(\theta)$ and $t_\mathrm{dec}(\theta)$. Along its
own velocity it shows the recipe's light curve, scaled by $\varepsilon(\theta)$, and after
deceleration it slows as $\Gamma\propto t^{-3/8}$. An observer at angle $\alpha$ from the element's
velocity sees it Doppler-shifted by $a = (1-\beta)/(1-\beta\cos\alpha)$:

$$
L_\mathrm{obs}(t_\mathrm{obs}) = \sum_\text{elements}\frac{d\Omega}{\Omega_\mathrm{beam}(\Gamma)}\;a^{2+p}\;h(\theta, t),
\qquad \frac{dt_\mathrm{obs}}{dt} = \frac{1}{a},
\qquad \Omega_\mathrm{beam} = \int a^{2+p}\,d\Omega .
$$

- Because $a^{2+p}$ multiplies a power law, **the spectrum keeps its shape off-axis**. Only the
  light curve changes.
- With $\Omega_\mathrm{beam}$ as the normalisation, a uniform spherical shell reproduces its own
  light curve for every observer.
- Every angle is normalised by the same constant, so that the **on-axis** 0.3–1 TeV luminosity at
  $t' = 11$ h is exactly $L_{\mathrm{TeV},11h}$.

**What it gives.** Within the core the model is the on-axis light curve. Out to
$\theta_\mathrm{max}$ the peak comes from the jet material on the line of sight. Beyond it nothing
is on the line of sight, and the observer waits for the core to decelerate until its beaming cone
($\sim 1/\Gamma$) reaches the line of sight. For the benchmark, the peak is at
$1.2\times10^3$ s at 30°, $7\times10^4$ s at 50° and $8\times10^5$ s at 57°, and beyond the
$10^7$ s grid at larger angles. At 11 h, the flux is $10^{-5}$ of the on-axis value at 50° and
$10^{-11}$ at 90°.

The simpler **local closure** of Nava (2020) Sec. 5, kept in the stochastic module
(`grb_model_at_theta`), instead treats the element on the line of sight as its own on-axis event.
It agrees with the integral on the peak time to about 15% up to 30°. Past $\theta_\mathrm{max}$ it
has nothing to evaluate, so it cannot describe far off-axis views.

---

## 5. The benchmark jet

`BENCHMARK` holds the median of every population input:

| Input | Benchmark | Population | Reaches the $L_k$ limit? |
| --- | --- | --- | --- |
| $E_{\gamma,\mathrm{iso}}$ | $2.0\times10^{50}$ erg | $E_\mathrm{peak}$ broken power law + Amati (Ghirlanda et al. 2016) | only through the time scale, $t\propto E^{1/3}$ |
| $\theta_\mathrm{core}$ | 14.1° | lognormal, σ = 0.2 dex (Fong et al. 2015) | yes |
| $\Gamma_\mathrm{core}$ | 200 | lognormal, σ = 0.2 dex | yes, off-axis |
| $\beta_1$, $\beta_2$ | 2, −1.45 | normal, σ = 0.05, 0.48 | $\beta_2$ yes |
| photon index $p$ | 2.2 | normal, σ = 0.1 | yes (spectrum) |
| $L_{\mathrm{TeV},11h}$ | mean Berger (2014) relation, $L_\mathrm{TeV}/L_X = 1$ | 0.5 and 0.3 dex scatter | no, amplitude only |
| $n$, $\eta_\gamma$ | 0.1 cm⁻³, 0.2 | fixed | — |

**Why one fixed jet loses little.** In the default normalisation of `sim_3d`
(`MODEL_NORMALISATION = "gti_mean"`, guidelines Step 6), each realisation is rescaled to the
trial luminosity $L_k$ averaged over the GTIs. Anything that only multiplies $L$ cancels. That
includes the largest random terms of the recipe: the 0.5 and 0.3 dex scatters, and the amplitude
part of $E_{\gamma,\mathrm{iso}}$. What survives are the inputs that change the **shape** of the
model: $\theta_\mathrm{core}$, $\Gamma_\mathrm{core}$, $\beta_2$, the photon index, and the
time scale.

---

## 6. Validation

All checks are in `setup_model_phenomenological_fixed.ipynb`.

| Check | Result |
| --- | --- |
| On-axis element integral against an independent 1-D calculation (near-uniform jet, exact emission time per angle) | agrees. The arrival-time spread rescales rise and decay by different, analytically known factors; the 11 h normalisation removes the decay factor |
| Grid convergence: default grid (240 α × 128 ψ × 24 per decade in time) against twice as fine | notebook §2.2 |
| Shape fits to the five catO5 files, with $\theta_\mathrm{core}$, $\Gamma_\mathrm{core}$ and the amplitude free | four files to 0.09–0.14 dex rms, the 13.2° file to 0.25 dex; fitted $\theta_\mathrm{core}$ (6°–37°) within ~2σ of the population |
| No fit: the benchmark with each file's own $E_\mathrm{iso}$ and photon index | 0.4–0.9 dex rms for three files, 1.5–1.8 dex for two (their events had a different core) |
| Reload of the cache; `get_model(θ, "interp", align_peaks=False)` between grid nodes against a direct build | exact reload; within 0.05 dex |

The catO5 files therefore look like ordinary draws from this recipe, seen through this off-axis
method. The one exception is the amplitude of the 77.8° file, which is 2 dex below the recipe's
anchor. Its shape matches, and under the Step 6 normalisation only the shape matters.

---

## 7. Population scatter: variants and banks

Adding random noise to the flux is not a valid way to include the population. Any multiplicative
noise cancels under the $L_k$ normalisation, and shape noise has no meaning unless it comes from
the recipe's own distributions. There are three options:

1. **Nominal, the benchmark** (what `sim_3d` uses). The limit is for the population's median
   short GRB.
2. **Systematic band, one-sigma variants.** `one_sigma_variants` moves one shape input to its
   16th or 84th percentile. `SAVE_VARIANTS = True` writes all ten caches, for example
   `theta_core_lo/` and `theta_core_hi/`. Rerunning the limit with `model_dir_3d` pointed at them
   gives the band. Start with the two $\theta_\mathrm{core}$ variants, which dominate at every
   off-axis angle.
3. **Marginalised, a fixed bank.** `draw_jet_params(rng, K)` draws $K$ jets once, with a fixed
   seed, and each gets its own cache. Realisation $i$ would use member $k_i$, drawn from its own
   seed (common random numbers, [`physics_and_methods.md`](physics_and_methods.md) §6.5).
   **Not wired into `sim_3d` yet.**

---

## 8. The stochastic version

`setup_model_phenomenological.ipynb` (`sample_events`, `grb_model_at_theta`) draws independent
events, so it keeps the event-to-event scatter that five fixed files cannot represent. It differs
from the fixed model in two ways. It uses the local off-axis closure (§4), clipped at
$\theta_\mathrm{max}$, and it computes $t_\mathrm{dec}$ from the collimated rather than the
isotropic-equivalent energy, which makes it about 5× earlier for $\theta_\mathrm{core} = 14°$. The
fixed model uses the isotropic energy. The 9.1° catO5 file supports that choice: it peaks at 3 s,
against 4.4 s with the isotropic energy and 1.1 s with the collimated one.

Two further findings. The two routes to $E_{\gamma,\mathrm{iso}}$ ("via $E_\mathrm{peak}$" + Amati,
and Nava's direct broken power law) agree to within a factor of a few, but are not identical,
partly because no Amati scatter is applied. And the local closure cannot reach the late peak times
of the 77.8° file. This notebook is exploratory and nothing else reads its output.

---

## 9. Output and use in `sim_3d`

`model_set(BENCHMARK)` builds the model at 0°, 1°, …, 90° and `GRBModelSet.save` writes one `.npz`
per angle to `data/models/grb_afterglow_phenomenological/benchmark/`. Every angle belongs to the
same jet, so no common-$E_\mathrm{iso}$ rescaling is needed after loading (unlike catO5,
[`setup_model.md`](setup_model.md) §4.2).

```python
from gwuls import grb_model, paths

model_set = grb_model.GRBModelSet.load(paths.GRB_MODEL_PHENOMENOLOGICAL_DIR / "benchmark")
model_i = model_set.get_model(theta_i, method="interp", align_peaks=False)   # 1° grid: <= 0.05 dex
E_obs, t_obs, F_i = model_i.to_observer(d_i, z_i, energy_obs=E_grid, time_obs=t_grid,
                                        apply_ebl=True, ebl_reference="franceschini17")
```

In practice `simulate.EmissionModelInjection` makes these calls. It also averages the model over
each run's GTIs and normalises it to $L_k$
([`physics_and_methods.md`](physics_and_methods.md) §11). On a 1° grid neighbouring light curves are
close enough that interpolation at fixed $t'$ cannot double-peak, so peak alignment is switched off.
Past 57° the peak lies beyond the time grid anyway.

---

## 10. Caveats

- **Element normalisation.** $d\Omega/\Omega_\mathrm{beam}$ is the standard point-source choice.
  Lamb & Kobayashi's exact convention is not given in either source. The difference is largest
  for late, mildly relativistic material, which may explain the amplitude of the 77.8° file.
- **Simple dynamics.** The model has Blandford–McKee deceleration only: no lateral spreading, no
  non-relativistic transition (Γ is only floored at 1), and no counter-jet. These matter far
  off-axis at $t' \gtrsim 10^6$ s.
- **Smoothed on-axis peak.** The arrival-time spread across the beaming cone lowers the rise (by
  about 4×) and rounds the peak. Follow-ups that start after about a minute see only the decay,
  which matches the recipe exactly.
- **Readings of the sources.** The model reads "11 h" as rest-frame time. The prefactor of
  $t_\mathrm{dec} = R_\mathrm{dec}/2c\Gamma_0^2$ is assumed. The width convention of Eq. 1
  ($\theta^2/\theta_\mathrm{core}^2$, no factor 2) comes from PDF text extraction of Abe et al.
  (2026), whereas Nava (2020) has the factor 2. No Amati scatter is applied.
- **Constant photon index** in time and angle, as in the recipe and in the catO5 files.
- **The model is for a BNS.** The worked example S240615dg is a binary black hole, for which no
  jet is expected. It exercises the method only.

---

## 11. References

- H. Abe et al. (CTAO Consortium), *Chasing gamma-ray signals from binary neutron star
  coalescences with the Cherenkov Telescope Array: prospects and observing strategies*, ApJ
  **1004**, 46 (2026), [arXiv:2604.08748](https://arxiv.org/abs/2604.08748): the recipe (Sec. 3),
  the jet structure (Eqs. 1–2), the off-axis method (Sec. 3.4).
- L. Nava, *TeV counterparts of BNS mergers*, internal note (Jan 2020): the original recipe and
  the local off-axis closure (Sec. 5).
- G. P. Lamb and S. Kobayashi, *Low-Γ jets from compact stellar mergers: candidate
  electromagnetic counterparts to gravitational wave sources*, MNRAS **472**, 4953 (2017),
  [arXiv:1706.03000](https://arxiv.org/abs/1706.03000): the multi-element off-axis integral.
- R. D. Blandford and C. F. McKee, *Fluid dynamics of relativistic blast waves*, Phys. Fluids
  **19**, 1130 (1976), [doi:10.1063/1.861619](https://doi.org/10.1063/1.861619): deceleration,
  $\Gamma\propto t^{-3/8}$.
- G. Ghirlanda et al., *Short gamma-ray bursts at the dawn of the gravitational wave era*, A&A
  **594**, A84 (2016), [arXiv:1607.07875](https://arxiv.org/abs/1607.07875): the $E_\mathrm{peak}$
  distribution and the Amati relation (Eqs. 13, 15).
- E. Berger, *Short-duration gamma-ray bursts*, ARA&A **52**, 43 (2014),
  [arXiv:1311.2603](https://arxiv.org/abs/1311.2603): the $L_{X,11h}$–$E_{\gamma,\mathrm{iso}}$
  relation.
- W. Fong et al., *A decade of short-duration gamma-ray burst broadband afterglows: energetics,
  circumburst densities, and jet opening angles*, ApJ **815**, 102 (2015),
  [arXiv:1509.02922](https://arxiv.org/abs/1509.02922): the $\theta_\mathrm{core}$ distribution.
- G. Ghirlanda et al., *Compact radio emission indicates a structured jet was produced by a binary
  neutron star merger*, Science **363**, 968 (2019),
  [arXiv:1808.00469](https://arxiv.org/abs/1808.00469): $\Gamma_\mathrm{core}$ of GRB 170817A.
