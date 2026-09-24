"""Single source of truth for every directory this repo reads from or writes to."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
GW_INPUT_DIR = DATA_DIR / "gw_input"
TMP_DIR = DATA_DIR / "tmp"
SLURM_OUTPUT_DIR = DATA_DIR / "slurm_output"

# Emission models (setup_model.ipynb): raw = as delivered (e.g. INAF catO5_*.fits),
# standardized = gwuls.grb_model's own .npz cache (Steps 1-2 of the simulation
# guidelines), which is what sim_3d actually loads -- it never touches a raw file.
MODELS_DIR = DATA_DIR / "models"
GRB_MODEL_RAW_DIR = MODELS_DIR / "grb_afterglow_inaf" / "raw"
GRB_MODEL_STANDARDIZED_DIR = MODELS_DIR / "grb_afterglow_inaf" / "standardized"

# GW parameter estimation (setup_angle_distribution.ipynb): raw = the full GWTC
# "combined_PEDataRelease.hdf5" files + PESummaryTable as downloaded (gitignored,
# GBs), standardized = gwuls.gw_pe's light per-alert file, <superevent>.h5, plus
# index.csv listing every alert with released PE -- both tracked in git.
GW_PE_DIR = DATA_DIR / "gw_pe"
GW_PE_RAW_DIR = GW_PE_DIR / "raw"
GW_PE_STANDARDIZED_DIR = GW_PE_DIR / "standardized"

OUTPUTS_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Cluster-only real DL3 data root; override with $LST_REAL_DATA_DIR when it lives elsewhere.
LST_REAL_DATA_DIR = Path(os.environ.get(
    "LST_REAL_DATA_DIR", "/fefs/aswg/workspace/juan.jimenez/data/real"
))


def ensure_dirs() -> None:
    """Create every standard directory (input/scratch/output) if missing."""
    for d in (GW_INPUT_DIR, TMP_DIR, SLURM_OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR,
              GRB_MODEL_RAW_DIR, GRB_MODEL_STANDARDIZED_DIR,
              GW_PE_RAW_DIR, GW_PE_STANDARDIZED_DIR):
        d.mkdir(parents=True, exist_ok=True)


def gw_pe_path(superevent_id: str) -> Path:
    """The light, git-tracked PE file for one alert (gwuls.gw_pe.standardize_pe_release
    writes it, gwuls.gw_pe.load_pe reads it). Named by the LVK superevent ID
    (e.g. "S240615dg"), the name the alert went out under, not the GWTC name."""
    return GW_PE_STANDARDIZED_DIR / f"{superevent_id}.h5"


def theta_distribution_path(source_name: str) -> Path:
    """The standardized viewing-angle-distribution cache for one GW source
    (setup_angle_distribution.ipynb writes it, sim_3d reads it) -- see
    gwuls.angle_distribution.save_theta_distribution/load_theta_distribution."""
    return GW_INPUT_DIR / f"{source_name}_theta_distribution.npz"


def plot_path(source_name: str, filename: str) -> Path:
    d = PLOTS_DIR / source_name
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def result_path(source_name: str, filename: str) -> Path:
    d = RESULTS_DIR / source_name
    d.mkdir(parents=True, exist_ok=True)
    return d / filename


def run_stem(source_name, type_obs, e_min, e_max, bkg_type, correlation_radius, binsz,
             use_dirac_delta=False) -> str:
    """
    The filename stem shared by every cache/result file for one run configuration.

    The single place this is built -- every notebook cell that names a per-run
    cache or result file (BKG results, grid results, grid/iterative caches)
    appends its own suffix to this instead of re-deriving the stem by hand,
    which is what let the BKG and grid filename conventions drift apart.
    """
    dirac = "_isdirac" if use_dirac_delta else ""
    return (f"{source_name}_{type_obs}_E{e_min}_{e_max}_bkg{bkg_type}"
            f"_corr{correlation_radius}{dirac}_binsize{binsz}")
