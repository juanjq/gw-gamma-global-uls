"""Single source of truth for every directory this repo reads from or writes to."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
GW_INPUT_DIR = DATA_DIR / "gw_input"
TMP_DIR = DATA_DIR / "tmp"
SLURM_OUTPUT_DIR = DATA_DIR / "slurm_output"

OUTPUTS_DIR = REPO_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"

# Cluster-only real DL3 data root; override with $LST_REAL_DATA_DIR when it lives elsewhere.
LST_REAL_DATA_DIR = Path(os.environ.get(
    "LST_REAL_DATA_DIR", "/fefs/aswg/workspace/juan.jimenez/data/real"
))


def ensure_dirs() -> None:
    """Create every standard directory (input/scratch/output) if missing."""
    for d in (GW_INPUT_DIR, TMP_DIR, SLURM_OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


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
