# Documentation

The physics and methods behind `gw-gamma-global-uls`. Installation and the cluster/local workflow
are in the [top-level README](../README.md).

| Document | Covers | Notebook | Code |
| --- | --- | --- | --- |
| [`pipeline.md`](pipeline.md) | **Start here.** The GW map on the analysis grid, the test statistic Λ, flux ↔ luminosity, the Monte Carlo, the upper-limit fit, the three limits, results for S240615dg, validation, what is still open (§11.5) | [`3_upper_limits`](../notebooks/3_upper_limits.ipynb) | `simulate.py`, `utils.py` |
| [`emission_model.md`](emission_model.md) | The GRB afterglow the emission-model limit injects: the Nava (2020) / Abe et al. (2026) recipe, the benchmark jet, its off-axis element integral, the catO5 files, the $L(E', t')$ storage format | [`1_grb_model`](../notebooks/1_grb_model.ipynb) | `grb_phenomenological.py`, `grb_model.py` |
| [`viewing_angle.md`](viewing_angle.md) | The viewing angle from GW parameter estimation (the joint PE draw, $p(\theta_v \mid d)$) and the priors used without PE | [`2_gw_viewing_angle`](../notebooks/2_gw_viewing_angle.ipynb) | `gw_pe.py`, `angle_distribution.py` |

## How the pieces fit

```
1_grb_model.ipynb ────────► data/models/grb_afterglow_phenomenological/benchmark/*.npz ─┐  L(E', t'; θ_v)
2_gw_viewing_angle.ipynb ─► data/gw_pe/standardized/<alert>.h5 ─────────────────────────┤  PE (RA, Dec, d, θ_v)
                         └► data/gw_input/<alert>_theta_distribution.npz ───────────────┤  p(θ_v | d), non-PE options
                                                                                        ▼
GW skymap + DL3 ──► 3_upper_limits.ipynb ──► 2D flux UL, 3D luminosity UL, 3D UL with the emission model
                                             ──► outputs/results/<source>/<stem>_upper_limits.json
```

## Conventions

- **Guidelines steps.** "Step 4.3", "Step 6" and so on refer to the internal note *Simulation
  guidelines: from a numerical GRB model to a GW-marginalised IACT upper limit* (Sep 2026).
- **Section numbers.** Notebook cells cite these documents as "§n"; the numbers are kept stable.
- **Frames.** Unprimed $E, t$ are observer-frame, primed $E', t'$ rest-frame; $t$ is the time since
  the merger.
- **Angles.** $\theta_v \in [0°, 90°]$ is the angle between the line of sight and the nearer jet
  axis. Every angle pdf is per degree.
- **Luminosities** are isotropic-equivalent in a stated band, never bolometric.
