# gw-gamma-global-uls

Monte-Carlo upper-limit pipeline for LST-1 gravitational-wave follow-up. Given a
GW alert and a real (or simulated) gamma-ray dataset, it derives:

- a **2D flux** upper limit: inject a point source at a fixed flux, sampling only
  a sky position from the GW 95% credible region;
- a **3D luminosity** upper limit: inject at a fixed band luminosity, sampling
  both sky position and luminosity distance from the full GW skymap;
- a **3D upper limit with an emission model**: inject a GRB afterglow (fixed
  phenomenological jet, EBL-absorbed) at a fixed GTI-mean luminosity, with sky
  position, distance and viewing angle drawn together from one GW PE sample.

All three use one test statistic, `Lambda = max over the GW region of TS + 2 ln p_GW`.
The limit is the injected strength at which a fraction CL of simulated `Lambda`
values exceeds the observation. Every realisation gets its own injected strength and
the detection probability is fitted with a probit, which also gives the limit's
Monte-Carlo uncertainty.

**Documentation:** [`docs/README.md`](docs/README.md) is the index. Start with
[`docs/pipeline.md`](docs/pipeline.md).

## Status (Nov 2026)

Done for **S240615dg** (LST-1+MAGIC, 95% CL, 0.6–20 TeV; numbers in
[`pipeline.md` §8.1](docs/pipeline.md#81-results-for-s240615dg)):

| Limit | Value | Closure |
| --- | --- | --- |
| 2D flux, Γ = 2 | φ0 = 2.34e-12 cm⁻² s⁻¹ TeV⁻¹ (−2.9%/+3.2%) | — |
| 3D luminosity, Γ = 2 | L0 = 3.85e45 erg/s (−5.0%/+5.5%) | +1.36σ |
| 3D emission model, joint PE draw | L_k = 1.99e47 erg/s (−7.9%/+8.9%) | +1.06σ |

Open (details in [`pipeline.md` §11.5](docs/pipeline.md#115-still-open)):

1. None of the three limits reached the 0.01 dex target before the 6000-realisation budget
   (`converged = False`). Raise `ul_max_sims` or quote the wider interval.
2. The global 2D limit sits 17% *below* the largest per-pixel sky-map UL in the 95% region,
   which a trials factor should not allow. Unexplained (§7.6).
3. The power-law 3D limit still draws (sky, d) from the skymap, while the model limit uses
   the PE, so their ratio also carries the skymap–PE distance difference.
4. One fixed jet only: no loop over the one-sigma model variants.
5. S240615dg is a binary black hole: it demonstrates the method, no counterpart is
   expected. **S241125n** has standardized PE in the repo but has not been run.

## Notebooks, in run order

| Notebook | When | What |
| --- | --- | --- |
| [`1_grb_model.ipynb`](notebooks/1_grb_model.ipynb) | once | builds the benchmark jet at 0–90°, the model the emission-model limit injects |
| [`2_gw_viewing_angle.ipynb`](notebooks/2_gw_viewing_angle.ipynb) | once per alert | standardizes the GWTC PE release, builds p(d, θ_v) and p(θ_v \| d) |
| [`3_upper_limits.ipynb`](notebooks/3_upper_limits.ipynb) | **main** | data prep, then the 2D, 3D and emission-model limits, closure tests, JSON export |

[`notebooks/extra/`](notebooks/extra/) holds optional explainers that nothing else reads:
`catO5_models` (the INAF reference files), `stochastic_jet` (random draws from the jet recipe),
`source_sampling` (how each realisation's source is drawn, for talks).

[`docs/figures/presentation_schemes.ipynb`](docs/figures/presentation_schemes.ipynb) makes the
presentation figures in `docs/figures/presentation/`.

## Layout

```
gwuls/                        # importable library
├── paths.py                  # every directory, in one place
├── simulate.py               # Monte Carlo, samplers, emission-model injection, UL fit, reporting
├── grb_model.py              # emission model L(E', t'; θ_v), projection with EBL
├── grb_phenomenological.py   # Nava (2020) / Abe et al. (2026) jet recipe, benchmark jet
├── gw_pe.py                  # GWTC PE releases -> light per-alert files
├── angle_distribution.py     # joint p(d, θ_v) and the angle draw
├── utils.py                  # HEALPix / GW-map / distance-CDF helpers
└── plotting.py               # diagnostic plots

data/
├── gw_input/                 # GW skymap .fits (tracked) + distance-CDF / θ caches (.npz, ignored)
├── gw_pe/                    # raw/: GWTC PE releases from Zenodo (GBs, ignored)
│                             # standardized/<superevent>.h5 + index.csv (~1 MB/alert, TRACKED)
├── models/grb_afterglow_phenomenological/benchmark/   # benchmark-jet cache (ignored)
├── models/grb_afterglow_inaf/  # raw/ catO5 .fits (tracked), standardized/ cache (ignored)
├── ebl/                      # EBL tables (tracked); used as $GAMMAPY_DATA if unset
└── tmp/                      # per-source dataset .pkl + fit caches (ignored)

outputs/                      # ignored: plots/<source>/ and results/*.json
```

## Setup

```bash
conda env create -f environment.yml
conda activate gw-gamma-global-uls
python -m ipykernel install --user --name gw-gamma-global-uls --display-name "Python (gw-gamma-global-uls)"
```

Use that kernel for the notebooks, not `base`: `ligo.skymap` has a compiled extension and
can crash a mismatched Python instead of raising. Notebooks put the repo root on `sys.path`,
so no `pip install` is needed. Quick library check:
`python -c "from gwuls import simulate; simulate.check_spectral_conversions()"`.

## Cluster vs. local

The real DL3 data lives only on the LST cluster (`gwuls.paths.LST_REAL_DATA_DIR`,
overridable with `$LST_REAL_DATA_DIR`). Part 1 of the main notebook turns it into a small
per-source `.pkl` in `data/tmp/`, and everything after that needs only the `.pkl`:

1. **Cluster:** run Part 1 of `3_upper_limits.ipynb` to build `data/tmp/*.pkl`.
2. **Copy `data/` locally** (code goes through git, never rsync):
   ```bash
   rsync -avz cp02:/fefs/aswg/workspace/juan.jimenez/lstreco/gw-gamma-global-uls/data/ \
     ~/projects/gw-gamma-global-uls/data/
   ```
3. **Either machine:** run Part 2 (the simulations) against that `.pkl`.

`N_JOBS` splits each batch of realisations across processes (`-1`: every CPU of the
allocation). Realisations are seeded by their index, so the result does not depend on
`N_JOBS`. Each worker holds a copy of the dataset: on `BrokenProcessPool`, lower it. Every
simulated round is cached in `data/tmp/`, so an interrupted run resumes.
