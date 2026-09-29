"""
gw_pe.py -- GW parameter-estimation (PE) posteriors, reduced to what this
pipeline needs and stored per alert in a light, git-tracked file.

Two layers, mirroring data/models/ (see gwuls.paths):

    data/gw_pe/raw/            the GWTC "*-combined_PEDataRelease.hdf5" files and the
                               catalog's "*-PESummaryTable.hdf5", exactly as downloaded
                               from Zenodo (0.2-7 GB per event -- gitignored)
    data/gw_pe/standardized/   <superevent>.h5, one per alert (~1 MB -- tracked), and
                               index.csv, one row per alert: the running list of alerts
                               whose PE has been released and standardized here

`standardize_pe_release` goes raw -> standardized, `standardize_all` does it
for every raw file present, and `load_pe` reads a standardized file back as a
`PESamples`. Nothing downstream opens a raw file.

What a standardized file keeps, and why:

- Posterior samples for `DEFAULT_COLUMNS` (sky position, distance, redshift,
  both inclination conventions, source-frame masses, chi_eff, tidal
  deformabilities when the analysis has them, network SNR), in float32.
- The *same number* of samples from every waveform analysis (a random
  subset, `n_per_analysis`), tagged by an `analysis` index. The
  concatenation is therefore directly the equal-weight mixture the LVK
  catalog papers quote, and each analysis can still be pulled out on its
  own to check waveform systematics.
- The PE prior samples for (d_L, theta_jn), to show how much the GW data
  actually inform the inclination.
- Each analysis's prior definitions, approximant, reference frequency,
  cosmology and detectors, as HDF5 attributes.

Files are named by the LVK *superevent* ID (e.g. "S240615dg"), which is the
name the alert went out under, the name the skymap in data/gw_input/ has, and
the `source_name` the notebooks use -- not the GWTC name (GW240615_113620).
The mapping comes from the catalog's PESummaryTable when it is next to the
raw file, else from the superevent ID in the analysis run directory recorded
in the file's own config.

Angles in the raw files are in radians; `PESamples.get` also provides the
derived degree-valued columns and the folded jet viewing angle
`viewing_angle_deg = min(theta_JN, 180 - theta_JN)`.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import paths

__all__ = [
    "DEFAULT_COLUMNS",
    "PRIOR_COLUMNS",
    "PESamples",
    "standardize_pe_release",
    "standardize_all",
    "load_pe",
    "list_raw_releases",
    "rebuild_index",
    "read_index",
]

FORMAT_VERSION = 1

#: Posterior columns kept in a standardized file (missing ones are skipped,
#: e.g. lambda_1/lambda_2 exist only for analyses with a neutron-star model).
DEFAULT_COLUMNS = (
    # extrinsic: what a follow-up simulation draws from (Step 4 of the guidelines)
    "ra", "dec", "luminosity_distance", "redshift", "theta_jn", "iota",
    # intrinsic: source class and counterpart physics
    "mass_1_source", "mass_2_source", "chirp_mass_source", "chi_eff",
    "lambda_1", "lambda_2",
    # detection
    "network_matched_filter_snr",
)

#: PE prior samples kept alongside, for posterior-vs-prior comparisons.
PRIOR_COLUMNS = ("luminosity_distance", "theta_jn")

_NON_ANALYSIS_KEYS = {"history", "version"}
_SUPEREVENT_RE = re.compile(r"(?<![A-Za-z0-9])(S\d{6}[a-z]{1,3})(?![A-Za-z0-9])")
_GW_NAME_RE = re.compile(r"(GW\d{6}_\d{6})")
_CATALOG_RE = re.compile(r"IGWN-(GWTC)(\d+)p(\d+)")
_INDEX_NAME = "index.csv"


# ===========================================================================
# The in-memory object
# ===========================================================================

@dataclass
class PESamples:
    """A standardized PE posterior for one alert (see the module docstring).

    `posterior` holds the stored columns, all of equal length, with the
    samples of every analysis concatenated; `analysis_index[k]` says which
    entry of `analyses` sample k came from. `prior`/`prior_analysis_index`
    are the same for the PE prior samples.
    """

    superevent_id: str
    gw_name: str
    catalog: str
    analyses: Tuple[str, ...]
    posterior: Dict[str, np.ndarray]
    analysis_index: np.ndarray
    prior: Dict[str, np.ndarray] = field(default_factory=dict)
    prior_analysis_index: Optional[np.ndarray] = None
    analysis_meta: Dict[str, dict] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.analysis_index.size)

    @property
    def columns(self) -> Tuple[str, ...]:
        return tuple(self.posterior)

    def get(self, name: str, analysis: Optional[str] = None, prior: bool = False) -> np.ndarray:
        """One column (float64), stored or derived, optionally for one analysis.

        Derived names: theta_jn_deg, iota_deg, viewing_angle_deg (the folded
        jet angle), ra_deg, dec_deg, and distance_Mpc (= luminosity_distance).
        """
        table = self.prior if prior else self.posterior
        index = self.prior_analysis_index if prior else self.analysis_index

        def raw(col):
            if col not in table:
                where = "prior" if prior else "posterior"
                raise KeyError(f"{col!r} not in the stored {where} columns {tuple(table)}")
            return np.asarray(table[col], dtype=float)

        derived = {
            "theta_jn_deg": lambda: np.degrees(raw("theta_jn")),
            "iota_deg": lambda: np.degrees(raw("iota")),
            "viewing_angle_deg": lambda: fold_viewing_angle(np.degrees(raw("theta_jn"))),
            "ra_deg": lambda: np.degrees(raw("ra")),
            "dec_deg": lambda: np.degrees(raw("dec")),
            "distance_Mpc": lambda: raw("luminosity_distance"),
        }
        values = derived[name]() if name in derived else raw(name)
        if analysis is None:
            return values
        if analysis not in self.analyses:
            raise KeyError(f"analysis {analysis!r} not in {self.analyses}")
        return values[index == self.analyses.index(analysis)]

    def for_analysis(self, analysis: str) -> "PESamples":
        """The same object restricted to one waveform analysis."""
        k = self.analyses.index(analysis)
        keep = self.analysis_index == k
        prior, prior_index = self.prior, self.prior_analysis_index
        if prior_index is not None:
            pkeep = prior_index == k
            prior = {c: v[pkeep] for c, v in self.prior.items()}
            prior_index = np.zeros(int(pkeep.sum()), dtype=np.uint8)
        return PESamples(
            superevent_id=self.superevent_id, gw_name=self.gw_name, catalog=self.catalog,
            analyses=(analysis,), posterior={c: v[keep] for c, v in self.posterior.items()},
            analysis_index=np.zeros(int(keep.sum()), dtype=np.uint8),
            prior=prior, prior_analysis_index=prior_index,
            analysis_meta={analysis: self.analysis_meta.get(analysis, {})}, meta=dict(self.meta),
        )

    def summary(self, analysis: Optional[str] = None) -> dict:
        """Median and 90% symmetric interval of the quantities the index lists."""
        out = {}
        for name in ("distance_Mpc", "viewing_angle_deg", "theta_jn_deg", "mass_1_source",
                     "mass_2_source", "chirp_mass_source", "network_matched_filter_snr"):
            try:
                x = self.get(name, analysis=analysis)
            except KeyError:
                continue
            q5, q50, q95 = np.percentile(x, [5, 50, 95])
            out[name] = {"median": q50, "q05": q5, "q95": q95}
        if "mass_2_source" in self.posterior:
            out["p_m2_below_3Msun"] = float(np.mean(self.get("mass_2_source", analysis=analysis) < 3.0))
        out["p_viewing_angle_below_30deg"] = float(
            np.mean(self.get("viewing_angle_deg", analysis=analysis) < 30.0))
        return out


def fold_viewing_angle(theta_jn_deg):
    """theta_v = min(theta_JN, 180 - theta_JN): the angle to the *nearer* jet
    axis, since a jet is bipolar and tracks J, not the instantaneous L."""
    theta_jn_deg = np.asarray(theta_jn_deg, dtype=float)
    return np.minimum(theta_jn_deg, 180.0 - theta_jn_deg)


# ===========================================================================
# Raw file introspection
# ===========================================================================

def _decode(value):
    """h5py scalar/1-element dataset -> plain str/float."""
    if isinstance(value, np.ndarray):
        value = value.ravel()
        if value.size != 1:
            return [_decode(v) for v in value]
        value = value[0]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode(errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_optional(group, path: str):
    try:
        return _decode(group[path][()])
    except (KeyError, TypeError, ValueError):
        return None


def _analysis_groups(f) -> List[str]:
    return [k for k in f.keys()
            if k not in _NON_ANALYSIS_KEYS and "posterior_samples" in f[k]]


def _catalog_from_filename(name: str) -> str:
    m = _CATALOG_RE.search(name)
    return f"{m.group(1)}-{m.group(2)}.{m.group(3)}" if m else "unknown"


def _find_summary_table(raw_path: Path) -> Optional[Path]:
    """The catalog's PESummaryTable next to a raw file, same release prefix preferred."""
    prefix = raw_path.name.split("-GW")[0]
    candidates = sorted(raw_path.parent.glob("*PESummaryTable*.hdf5"))
    same = [c for c in candidates if c.name.startswith(prefix)]
    return (same or candidates or [None])[0]


def _superevent_from_summary_table(table_path: Path, gw_name: str) -> Optional[str]:
    import h5py

    with h5py.File(table_path, "r") as f:
        rows = f["summary_info"][()]
    names = np.array([_decode(n) for n in rows["gw_name"]])
    hits = rows["superevent_id"][names == gw_name]
    return _decode(hits[0]) if len(hits) else None


def _superevent_from_config(f, groups: Sequence[str]) -> Optional[str]:
    """Fallback: the superevent ID appears in each analysis's run directory."""
    for g in groups:
        for key in ("webdir", "outdir", "label"):
            text = _read_optional(f[g], f"config_file/config/{key}")
            if isinstance(text, str):
                m = _SUPEREVENT_RE.search(text)
                if m:
                    return m.group(1)
    return None


def _git_revision() -> str:
    try:
        out = subprocess.run(["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def list_raw_releases(raw_dir=None) -> List[dict]:
    """Every raw PE release in `raw_dir`, with its GW name, superevent ID and
    whether it has been standardized yet. Opens each file only to read its
    group names and config, never its samples."""
    import h5py

    raw_dir = Path(raw_dir or paths.GW_PE_RAW_DIR)
    out = []
    for p in sorted(raw_dir.glob("*PEDataRelease*.hdf5")):
        gw = _GW_NAME_RE.search(p.name)
        gw_name = gw.group(1) if gw else p.stem
        table = _find_summary_table(p)
        superevent = _superevent_from_summary_table(table, gw_name) if table else None
        with h5py.File(p, "r") as f:
            groups = _analysis_groups(f)
            superevent = superevent or _superevent_from_config(f, groups)
        out.append({
            "path": p, "gw_name": gw_name, "superevent_id": superevent,
            "catalog": _catalog_from_filename(p.name), "analyses": groups,
            "size_GB": p.stat().st_size / 1e9,
            "standardized": bool(superevent) and paths.gw_pe_path(superevent).exists(),
        })
    return out


# ===========================================================================
# raw -> standardized
# ===========================================================================

def standardize_pe_release(raw_path, superevent_id: Optional[str] = None,
                           n_per_analysis: int = 5000,
                           analyses: Optional[Sequence[str]] = None,
                           columns: Sequence[str] = DEFAULT_COLUMNS,
                           prior_columns: Sequence[str] = PRIOR_COLUMNS,
                           seed: int = 0, out_dir=None, overwrite: bool = False) -> Path:
    """Write the light, per-alert file for one raw GWTC PE release.

    Parameters
    ----------
    raw_path : path to a "*-combined_PEDataRelease.hdf5".
    superevent_id : the alert name to file it under. Default: looked up in
        the PESummaryTable next to `raw_path`, else in the file's config.
    n_per_analysis : samples kept from each analysis. Capped at the smallest
        analysis's size, so every analysis contributes equally.
    analyses : which analysis groups to keep (default: all of them).
    seed : for the random subset -- fixed, so re-running is reproducible.
    overwrite : False (default) leaves an existing standardized file alone.

    Returns the path written (or the existing one, if not overwriting).
    """
    import h5py

    raw_path = Path(raw_path)
    gw = _GW_NAME_RE.search(raw_path.name)
    gw_name = gw.group(1) if gw else raw_path.stem
    catalog = _catalog_from_filename(raw_path.name)
    rng = np.random.default_rng(seed)

    with h5py.File(raw_path, "r") as f:
        available = _analysis_groups(f)
        if not available:
            raise KeyError(f"no analysis group with posterior_samples in {raw_path}")
        chosen = list(analyses) if analyses is not None else available
        missing = [a for a in chosen if a not in available]
        if missing:
            raise KeyError(f"analyses {missing} not in {raw_path.name}; available: {available}")

        if superevent_id is None:
            table = _find_summary_table(raw_path)
            superevent_id = (_superevent_from_summary_table(table, gw_name) if table else None) \
                or _superevent_from_config(f, chosen)
        if superevent_id is None:
            raise ValueError(f"could not determine the superevent ID of {raw_path.name}; "
                             "pass superevent_id=... explicitly")

        out_path = Path(out_dir or paths.GW_PE_STANDARDIZED_DIR) / f"{superevent_id}.h5"
        if out_path.exists() and not overwrite:
            return out_path

        n_available = {a: int(f[a]["posterior_samples"].shape[0]) for a in chosen}
        n_keep = int(min(n_per_analysis, *n_available.values()))

        post_cols: Dict[str, list] = {}
        post_index, prior_cols, prior_index = [], {c: [] for c in prior_columns}, []
        meta_by_analysis = {}
        for k, a in enumerate(chosen):
            ps = f[a]["posterior_samples"]
            names = ps.dtype.names
            take = np.sort(rng.choice(n_available[a], size=n_keep, replace=False))
            for c in columns:
                if names is not None and c in names:
                    post_cols.setdefault(c, []).append((k, np.asarray(ps[c], dtype=float)[take]))
            post_index.append(np.full(n_keep, k, dtype=np.uint8))

            prior_samples = f[a].get("priors/samples")
            n_prior = 0
            if prior_samples is not None and all(c in prior_samples for c in prior_columns):
                n_prior = int(prior_samples[prior_columns[0]].shape[0])
                for c in prior_columns:
                    prior_cols[c].append(np.asarray(prior_samples[c][()], dtype=float))
                prior_index.append(np.full(n_prior, k, dtype=np.uint8))

            meta_by_analysis[a] = {
                "n_available": n_available[a], "n_kept": n_keep, "n_prior": n_prior,
                "approximant": _read_optional(f[a], "meta_data/meta_data/approximant"),
                "f_ref": _read_optional(f[a], "meta_data/meta_data/f_ref"),
                "cosmology": _read_optional(f[a], "meta_data/meta_data/cosmology"),
                "ifos": _read_optional(f[a], "meta_data/meta_data/IFOs"),
                "trigger_time": _read_optional(f[a], "config_file/config/trigger-time"),
                "prior_luminosity_distance": _read_optional(f[a], "priors/analytic/luminosity_distance"),
                "prior_theta_jn": _read_optional(f[a], "priors/analytic/theta_jn"),
            }

    # A column is kept only if *every* analysis has it: the mixture must be
    # the same set of samples for every column.
    kept_columns = [c for c in columns
                    if c in post_cols and len(post_cols[c]) == len(chosen)]
    analysis_index = np.concatenate(post_index)

    trigger_times = [m["trigger_time"] for m in meta_by_analysis.values() if m["trigger_time"]]
    attrs = {
        "format_version": FORMAT_VERSION,
        "superevent_id": superevent_id,
        "gw_name": gw_name,
        "catalog": catalog,
        "source_file": raw_path.name,
        "analyses_json": json.dumps(chosen),
        "columns_json": json.dumps(kept_columns),
        "n_per_analysis": n_keep,
        "seed": seed,
        "trigger_time_gps": float(np.median([float(t) for t in trigger_times])) if trigger_times else np.nan,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "gwuls_git": _git_revision(),
        "note": ("Equal-weight mixture: the same number of posterior samples from each "
                 "analysis in analyses_json; posterior/analysis says which. Angles in rad."),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".h5.tmp")
    ds_kwargs = dict(compression="gzip", compression_opts=9, shuffle=True, track_times=False)
    with h5py.File(tmp_path, "w") as out:
        out.attrs.update(attrs)
        g = out.create_group("posterior")
        for c in kept_columns:
            g.create_dataset(c, data=np.concatenate([v for _, v in post_cols[c]]).astype(np.float32),
                             **ds_kwargs)
        g.create_dataset("analysis", data=analysis_index, **ds_kwargs)
        if prior_index:
            gp = out.create_group("prior")
            for c in prior_columns:
                gp.create_dataset(c, data=np.concatenate(prior_cols[c]).astype(np.float32), **ds_kwargs)
            gp.create_dataset("analysis", data=np.concatenate(prior_index), **ds_kwargs)
        ga = out.create_group("analyses")
        for a, m in meta_by_analysis.items():
            ga.create_group(a).attrs.update({k: ("" if v is None else v) for k, v in m.items()})
    os.replace(tmp_path, out_path)
    return out_path


def standardize_all(raw_dir=None, out_dir=None, overwrite: bool = False,
                    **kwargs) -> List[Path]:
    """`standardize_pe_release` for every raw release in `raw_dir` (skipping
    those already standardized unless `overwrite`), then `rebuild_index`."""
    written = []
    for rel in list_raw_releases(raw_dir):
        written.append(standardize_pe_release(rel["path"], superevent_id=rel["superevent_id"],
                                              out_dir=out_dir, overwrite=overwrite, **kwargs))
    rebuild_index(out_dir)
    return written


# ===========================================================================
# standardized -> memory
# ===========================================================================

def load_pe(superevent_id_or_path) -> PESamples:
    """Read a standardized file, by alert name ("S240615dg") or by path."""
    import h5py

    p = Path(superevent_id_or_path)
    if p.suffix != ".h5":
        p = paths.gw_pe_path(str(superevent_id_or_path))
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Standardize the raw PE release first "
            "(gwuls.gw_pe.standardize_all, or setup_angle_distribution.ipynb Part 0).")

    with h5py.File(p, "r") as f:
        a = dict(f.attrs)
        analyses = tuple(json.loads(a["analyses_json"]))
        posterior = {c: f["posterior"][c][()] for c in json.loads(a["columns_json"])}
        analysis_index = f["posterior"]["analysis"][()]
        prior, prior_index = {}, None
        if "prior" in f:
            prior = {c: f["prior"][c][()] for c in f["prior"] if c != "analysis"}
            prior_index = f["prior"]["analysis"][()]
        analysis_meta = {k: {kk: _decode(vv) for kk, vv in f["analyses"][k].attrs.items()}
                         for k in f["analyses"]}
    meta = {k: _decode(v) for k, v in a.items()
            if k not in ("analyses_json", "columns_json")}
    meta["path"] = str(p)
    return PESamples(
        superevent_id=str(a["superevent_id"]), gw_name=str(a["gw_name"]),
        catalog=str(a["catalog"]), analyses=analyses, posterior=posterior,
        analysis_index=analysis_index, prior=prior, prior_analysis_index=prior_index,
        analysis_meta=analysis_meta, meta=meta,
    )


# ===========================================================================
# index.csv -- the list of alerts with released + standardized PE
# ===========================================================================

_INDEX_FIELDS = (
    "superevent_id", "gw_name", "catalog", "n_analyses", "n_per_analysis",
    "distance_Mpc_median", "distance_Mpc_q05", "distance_Mpc_q95",
    "viewing_angle_deg_median", "viewing_angle_deg_q05", "viewing_angle_deg_q95",
    "p_viewing_angle_below_30deg",
    "mass_1_source_median", "mass_2_source_median", "p_m2_below_3Msun",
    "network_snr_median", "trigger_time_gps", "analyses", "source_file", "created_utc",
)


def rebuild_index(out_dir=None) -> Path:
    """Regenerate index.csv from every standardized file present, so the
    index can never drift from the files it lists."""
    out_dir = Path(out_dir or paths.GW_PE_STANDARDIZED_DIR)
    rows = []
    for p in sorted(out_dir.glob("*.h5")):
        pe = load_pe(p)
        s = pe.summary()

        def q(name, key, fmt="{:.4g}"):
            return fmt.format(s[name][key]) if name in s else ""

        rows.append({
            "superevent_id": pe.superevent_id, "gw_name": pe.gw_name, "catalog": pe.catalog,
            "n_analyses": len(pe.analyses), "n_per_analysis": pe.meta.get("n_per_analysis", ""),
            "distance_Mpc_median": q("distance_Mpc", "median"),
            "distance_Mpc_q05": q("distance_Mpc", "q05"),
            "distance_Mpc_q95": q("distance_Mpc", "q95"),
            "viewing_angle_deg_median": q("viewing_angle_deg", "median", "{:.1f}"),
            "viewing_angle_deg_q05": q("viewing_angle_deg", "q05", "{:.1f}"),
            "viewing_angle_deg_q95": q("viewing_angle_deg", "q95", "{:.1f}"),
            "p_viewing_angle_below_30deg": f"{s['p_viewing_angle_below_30deg']:.3f}",
            "mass_1_source_median": q("mass_1_source", "median", "{:.3g}"),
            "mass_2_source_median": q("mass_2_source", "median", "{:.3g}"),
            "p_m2_below_3Msun": f"{s['p_m2_below_3Msun']:.3f}" if "p_m2_below_3Msun" in s else "",
            "network_snr_median": q("network_matched_filter_snr", "median", "{:.1f}"),
            "trigger_time_gps": f"{pe.meta.get('trigger_time_gps', np.nan):.3f}",
            "analyses": ";".join(pe.analyses),
            "source_file": pe.meta.get("source_file", ""),
            "created_utc": pe.meta.get("created_utc", ""),
        })
    index_path = out_dir / _INDEX_NAME
    with open(index_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return index_path


def read_index(out_dir=None) -> List[dict]:
    """index.csv as a list of dicts (empty if nothing has been standardized)."""
    index_path = Path(out_dir or paths.GW_PE_STANDARDIZED_DIR) / _INDEX_NAME
    if not index_path.exists():
        return []
    with open(index_path, newline="") as fh:
        return list(csv.DictReader(fh))


def known_alerts(out_dir=None) -> List[str]:
    """Superevent IDs with a standardized PE file."""
    return [r["superevent_id"] for r in read_index(out_dir)]
