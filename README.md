# gw-gamma-global-uls

Monte-Carlo upper-limit pipeline for LST-1 gravitational-wave follow-up. Given a
GW skymap and a real (or simulated) gamma-ray dataset, it derives:

- a **2D flux** upper limit — inject a point source at a fixed flux, sampling only
  a sky position from the GW 95% credible region, and
- a **3D luminosity** upper limit — inject at a fixed band luminosity, sampling
  both sky position and luminosity distance from the full GW posterior,

using a shared test statistic `Lambda = max over the GW region of TS + 2 ln p_GW`,
with the upper limit found by bisecting on the injected amplitude/luminosity until
a target fraction of simulated `Lambda` values exceeds the observation. See the
module docstring in [`gwuls/simulate.py`](gwuls/simulate.py) for the full physics
and the exact flux/luminosity conversions.

## Layout

```
gwuls/                     # importable library
├── paths.py                # single source of truth for every directory (see below)
├── simulate.py              # the Monte-Carlo + bisection engine; also runnable as a CLI
├── slurm.py                  # one function to submit a simulate.py grid point to Slurm
├── utils.py                 # HEALPix / GW-map / distance-CDF helpers
└── plotting.py               # shared diagnostic plots

notebooks/
├── sim_2d.ipynb              # main pipeline: 2D flux upper limit
├── sim_3d.ipynb               # main pipeline: 3D luminosity upper limit (+ 2D comparison)
└── others/                     # historical / one-off notebooks, not part of the maintained
                                 # pipeline (several predate the gwuls package and are not
                                 # runnable as-is — see the note at the top of each)

data/
├── gw_input/                  # small, trackable inputs: GW skymap .fits + generated
│                               # per-source distance-CDF caches (.npz, gitignored)
├── tmp/                        # per-source dataset .pkl + bisection scratch (gitignored)
└── slurm_output/                # sbatch job logs, cluster only (gitignored)

outputs/                      # everything the pipeline generates for human consumption
├── plots/<source_name>/         # figures, one subfolder per GW source
└── results/<source_name>/        # final .npz / .json upper-limit results
```

`outputs/` and the generated parts of `data/` are gitignored — regenerate them by
rerunning the notebooks, don't hand-copy them between machines except as described
below.

## Setup

```bash
conda env create -f environment.yml
conda activate gw-gamma-global-uls
```

Notebooks import the library as `from gwuls import simulate, utils, plotting, paths`
after inserting the repo root on `sys.path` (see the first cell of each notebook) —
no `pip install -e .` needed.

## Cluster vs. local workflow

The real DL3 event data lives only on the LST cluster, under
`/fefs/aswg/workspace/juan.jimenez/data/real` (hardcoded as
`gwuls.paths.LST_REAL_DATA_DIR`, overridable via the `LST_REAL_DATA_DIR` env var) —
far too large to copy locally, so the parts of the notebooks that build a
`MapDataset` from real DL3 data only make sense on the cluster.

What those cells produce, though, is a small per-source `.pkl` in `data/tmp/`
(dataset + GW map + estimators) that the rest of the pipeline — everything in
`gwuls/simulate.py` — consumes on its own, with no further dependency on the real
DL3 data or on `gammapy`'s IRF-heavy machinery. That makes the natural split:

1. **On the cluster**: run the early cells of `sim_2d.ipynb`/`sim_3d.ipynb` against
   the real DL3 data to (re)build `data/gw_input/*.fits` and the per-source
   `data/tmp/*.pkl`.
2. **Copy locally** (once, or whenever the inputs change) — see below.
3. **Locally**: run the simulation/bisection cells against the copied `.pkl`, using
   the same notebooks.

Code changes should flow through git as usual (commit + push to GitHub, pull on
whichever machine you're on next) — the repo already has a GitHub remote
(`git@github.com:juanjq/gw-gamma-global-uls.git`), so **code never needs to be
rsynced from the cluster at all**. The only reason to reach for `rsync` is the
non-git `data/` contents.

### Submitting the grid-scan Slurm jobs

Both notebooks can also sweep a full amplitude/luminosity grid instead of bisecting
(`USE_ITERATIVE_ULS = False`), which submits one Slurm job per grid point through
[`gwuls.slurm.submit_simulation_job`](gwuls/slurm.py) — the single place that builds the
`sbatch` command, so it's identical from either notebook. Check on submitted jobs with
`!squeue -u $USER` (or `gwuls.slurm.print_queue()`); pass `dry_run=True` to preview the
`sbatch` command for one grid point without submitting it.

### First time on a new machine

```bash
git clone git@github.com:juanjq/gw-gamma-global-uls.git ~/projects/gw-gamma-global-uls
```

### Getting the data `sim_2d.ipynb` / `sim_3d.ipynb` need

Both notebooks only ever read/write two directories: `data/gw_input/` (the GW
skymap `.fits` — already tracked in git — plus, for the 3D notebook, the
generated distance-CDF `.npz` cache) and `data/tmp/` (the per-source dataset
`.pkl`, built on the cluster from the real DL3 data, plus bisection/cache
scratch). Nothing else on the cluster is needed to run them locally. From your
local machine:

```bash
rsync -avz --exclude 'slurm_output/' \
  cp02:/fefs/aswg/workspace/juan.jimenez/lstreco/gw-gamma-global-uls/data/ \
  ~/projects/gw-gamma-global-uls/data/
```

Adjust the local destination if you keep the repo somewhere other than
`~/projects/gw-gamma-global-uls`. Re-run this whenever you regenerate a
per-source `.pkl` on the cluster and want the newer one locally.
