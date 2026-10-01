# Documentation

The physics and methods behind `gw-gamma-global-uls`. Installation and the cluster/local workflow
are in the [top-level README](../README.md).

## What to read

| Document | Covers | Notebook | Code |
| --- | --- | --- | --- |
| [`physics_and_methods.md`](physics_and_methods.md) | **Start here.** The whole pipeline: the GW map on the analysis grid, the test statistic Λ, flux ↔ luminosity, the Monte Carlo, the upper-limit fit, the 2D/3D limits, the emission-model limit, results for S240615dg, the validation checks | [`sim_3d_model_CTAO_paper.ipynb`](../notebooks/sim_3d_model_CTAO_paper.ipynb) | `gwuls/simulate.py`, `gwuls/utils.py` |
| [`phenomenological_model.md`](phenomenological_model.md) | The GRB afterglow model that the 3D limit injects: the Nava (2020) / Abe et al. (2026) recipe, the fixed benchmark jet and its off-axis element integral | [`setup_model_phenomenological_fixed.ipynb`](../notebooks/setup_model_phenomenological_fixed.ipynb), [`setup_model_phenomenological.ipynb`](../notebooks/setup_model_phenomenological.ipynb) | `gwuls/grb_phenomenological.py` |
| [`setup_model.md`](setup_model.md) | The `GRBModel` format: rest-frame $L(E',t')$, projection to any distance with EBL, the model at any viewing angle, and the five catO5 catalogue files | [`setup_model.ipynb`](../notebooks/setup_model.ipynb) | `gwuls/grb_model.py` |
| [`viewing_angle_distribution.md`](viewing_angle_distribution.md) | The viewing-angle draw $p(\theta_v \mid d)$ from GW parameter estimation, and the population priors used without it | [`setup_angle_distribution.ipynb`](../notebooks/setup_angle_distribution.ipynb) | `gwuls/gw_pe.py`, `gwuls/angle_distribution.py` |

## How the pieces fit

```
setup_angle_distribution.ipynb ──► data/gw_input/<alert>_theta_distribution.npz ─┐   p(θ_v | d)
setup_model_phenomenological_fixed.ipynb ──► data/models/.../benchmark/*.npz ────┤   L(E', t'; θ_v)
                                                                                 ▼
GW skymap + DL3 ──► sim_3d_model_CTAO_paper.ipynb ──► 2D flux UL, 3D luminosity UL,
                                                      3D UL with the emission model
                                                      ──► outputs/results/<stem>_upper_limits.json
```

The two setup notebooks run once (the angle distribution once per alert). The main notebook only
loads their caches.

## Conventions shared by every document

- **Guidelines steps.** "Step 4.3", "Step 6" and so on refer to the internal note *Simulation
  guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit* (Sep 2026).
- **Section numbers.** The markdown cells of the notebooks cite sections of these documents as
  "§n". Those numbers are kept stable.
- **Frames.** Unprimed $E, t$ are observer-frame, primed $E', t'$ rest-frame; $t$ is the time since
  the merger.
- **Angles.** $\theta_v \in [0°, 90°]$ is the angle between the line of sight and the nearer jet
  axis. Every angle pdf is normalised per degree.
- **Luminosities** are isotropic-equivalent and refer to a stated band. None is bolometric.
- **References** are cited in the text as author (year) and listed with links at the end of
  each document.
