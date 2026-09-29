"""One consistent way to submit a `simulate.py` grid point to Slurm.

Only the non-iterative grid scan (`USE_ITERATIVE_ULS = False` in the notebooks)
touches Slurm this way -- the iterative bisection (`run_iterative_ul[_3d]` in
`simulate.py`) runs in-process and never calls `sbatch`.
"""

from __future__ import annotations

import getpass
import shlex
import subprocess
import sys

from . import paths

DEFAULT_PARTITION = "short"
DEFAULT_MEM_MB = 20000


def submit_simulation_job(
    n_sim, value, file_input, file_output, *, mode, spectral_index,
    compute_uls=0, seed=None, job_name="simulate_source",
    partition=DEFAULT_PARTITION, mem_mb=DEFAULT_MEM_MB, dry_run=False,
) -> str:
    """
    Submit one `simulate.py` grid point (2D flux or 3D luminosity) as a Slurm job.

    mode : "2d" or "3d"
    value : the PWL amplitude [cm-2 s-1 TeV-1] in 2D mode, or the band
        luminosity [erg/s] in 3D mode -- whatever `simulate.py`'s CLI expects.
    dry_run : print the `sbatch` command instead of running it.

    Returns the `sbatch` command's stdout (typically "Submitted batch job
    <id>"), or the command string itself when `dry_run=True`.
    """
    cmd = [
        sys.executable, str(paths.REPO_ROOT / "gwuls" / "simulate.py"),
        str(n_sim), str(value), str(file_input), str(file_output), str(int(compute_uls)),
        "--mode", mode, "--index", str(spectral_index),
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]

    log_path = paths.SLURM_OUTPUT_DIR / f"job_{job_name}_{mode}_{value:.4e}_N{n_sim}.out"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sbatch_cmd = [
        "sbatch", "-p", partition, "--mem", str(mem_mb),
        "-J", job_name, "-o", str(log_path), "--wrap", shlex.join(cmd),
    ]

    if dry_run:
        printable = " ".join(shlex.quote(c) for c in sbatch_cmd)
        print(printable)
        return printable

    result = subprocess.run(sbatch_cmd, check=True, capture_output=True, text=True)
    print(result.stdout.strip())
    return result.stdout.strip()


def print_queue(user=None):
    """`squeue -u <user>`, defaulting to the current user -- for a quick notebook check."""
    user = user or getpass.getuser()
    subprocess.run(["squeue", "-u", user])
