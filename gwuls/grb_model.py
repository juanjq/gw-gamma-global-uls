"""
grb_model.py -- the emission model F(E, t) / L(E', t'), standardised.

Implements Steps 1, 2 and 5 of "Simulation guidelines: from a numerical GRB
model to a GW-marginalised IACT upper limit" (see the .md in the repo notes):

  * Step 1  -- read a tabulated observer-frame model file (Option 1: the INAF
              catO5_*.fits catalogue), or build one analytically (Option 2:
              an arbitrary rest-frame f(E') x g(t'); Option 3: a constant,
              Gamma=2 power law -- the closed-form validation case).
  * Step 2  -- project observer-frame F(E, t) to the intrinsic, EBL-free,
              rest-frame L(E', t') = 4 pi d_L^2 F / (1+z)^2, by an *exact*
              regrid E'_n = (1+z) E_n, t'_n = t_n/(1+z) -- no interpolation
              needed at this step -- and interpolate/extrapolate log L
              bilinearly in (log E', log t') for any later query.
  * Step 5  -- the model exists only at discrete angles theta_1 < theta_2 <
              ...  (one file per angle). GRBModelSet.get_model implements
              all three options from the notes: nearest angle (A), stochastic
              nearest (B) and log-L interpolation in theta (C) -- the latter
              peak-aligned by default, so it never produces a double peak.
              Each file is a different catalogue event with its own E_iso;
              GRBModelSet.rescaled_to_eiso puts every angle on one common
              E_iso first (blast-wave scale invariance), so what varies
              between angles is the angle and not the event-to-event scatter.

A GRBModel is always the *rest-frame* L(E', t') for one viewing angle -- the
canonical, distance-independent object. `to_observer` (Step 7) projects it to
any (D_L, z), and optionally multiplies in EBL absorption there (never at the
model's own z, which the INAF files do not carry to begin with).

Standardisation: `GRBModel.save`/`.load` and `GRBModelSet.save`/`.load` read
and write one small `.npz` per angle (gwuls.paths.GRB_MODEL_STANDARDIZED_DIR),
so anything downstream (sim_3d) never has to know the model came from a FITS
file, an analytic shape or a power law -- it only ever sees this format.
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

try:
    from .simulate import redshift_from_luminosity_distance
except ImportError:                                     # run as a standalone script
    from simulate import redshift_from_luminosity_distance

__all__ = [
    "LogLogInterpolator2D",
    "GRBModel",
    "GRBModelSet",
    "ebl_transmission",
    "diagnose_ebl",
]

_trapz = getattr(np, "trapezoid", None) or np.trapz
_CM_PER_MPC = u.Mpc.to(u.cm)

# Blast-wave scale invariance (e.g. van Eerten & MacFadyen 2012): at fixed
# density and microphysics, E -> k E stretches every dynamical time scale
# (deceleration, off-axis peak, jet break) by k^(1/3), and at the same stage
# of the evolution multiplies the synchrotron flux by k for nu_m < nu < nu_c
# (F_max ~ number of radiating electrons ~ E), or by k^(2/3) above nu_c.
# (1, 1/3) is also the only one of (1,0), (2/3,1/3), (1,1/3) that makes both
# the peak time and the peak luminosity of the five catO5 files monotonic in
# theta (setup_model.ipynb).
EISO_L_EXPONENT = 1.0
EISO_T_EXPONENT = 1.0 / 3.0


# ===========================================================================
# The "Interpolation?" note: one interpolator, shared by every model
# ===========================================================================

class LogLogInterpolator2D:
    """Interpolate a positive matrix M(E, t) in log10(M) vs (log10(E), log10(t)).

    Both grids here are always log-spaced (energy always; the INAF time grid
    spans 0.1 s to 1e6 s), and spectra/light curves are locally power laws, so
    this is exact for them and avoids the distortion of linear interpolation
    across a decade-wide cell. Grid-edge queries extrapolate as a local power
    law -- exactly what falls out of linear extrapolation in log-log space,
    which is what `RegularGridInterpolator(..., fill_value=None)` does.
    """

    def __init__(self, energy: np.ndarray, time: np.ndarray, values: np.ndarray,
                 floor_fraction: float = 1e-30):
        energy = np.asarray(energy, dtype=float)
        time = np.asarray(time, dtype=float)
        values = np.asarray(values, dtype=float)

        if energy.ndim != 1 or time.ndim != 1:
            raise ValueError("energy and time must be 1D grids")
        if values.shape != (energy.size, time.size):
            raise ValueError(
                f"values shape {values.shape} != (len(energy), len(time)) = "
                f"({energy.size}, {time.size})"
            )
        if np.any(np.diff(energy) <= 0) or np.any(np.diff(time) <= 0):
            raise ValueError("energy and time grids must be strictly ascending")
        if np.any(energy <= 0) or np.any(time <= 0):
            raise ValueError("energy and time grids must be strictly positive (log-log grid)")

        vmax = np.nanmax(values)
        if not np.isfinite(vmax) or vmax <= 0:
            raise ValueError("values must contain at least one finite, positive entry")
        clipped = np.clip(values, vmax * floor_fraction, None)

        self.energy_bounds = (energy[0], energy[-1])
        self.time_bounds = (time[0], time[-1])
        self._log_energy = np.log10(energy)
        self._log_time = np.log10(time)
        self._log_values = np.log10(clipped)
        self._interp = RegularGridInterpolator(
            (self._log_energy, self._log_time), self._log_values,
            method="linear", bounds_error=False, fill_value=None,
        )

    def __call__(self, energy, time, report_out_of_range: bool = False):
        """Evaluate on the outer grid of `energy` x `time`, shape (n_E, n_t)."""
        energy = np.atleast_1d(np.asarray(energy, dtype=float))
        time = np.atleast_1d(np.asarray(time, dtype=float))
        if np.any(energy <= 0) or np.any(time <= 0):
            raise ValueError("query energy and time must be > 0")

        e_grid, t_grid = np.meshgrid(energy, time, indexing="ij")
        pts = np.stack([np.log10(e_grid).ravel(), np.log10(t_grid).ravel()], axis=-1)
        out = np.power(10.0, self._interp(pts).reshape(e_grid.shape))

        if not report_out_of_range:
            return out
        e_lo, e_hi = self.energy_bounds
        t_lo, t_hi = self.time_bounds
        oor = ((e_grid < e_lo) | (e_grid > e_hi) | (t_grid < t_lo) | (t_grid > t_hi))
        return out, float(np.mean(oor))


def _band_integral(values: np.ndarray, x: np.ndarray, axis: int) -> np.ndarray:
    """int y dx = int (y x) d(ln x) on a fine log grid -- Step 2/6's normalisation integral."""
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return np.zeros(np.delete(values.shape, axis))
    shape = [1] * values.ndim
    shape[axis] = x.size
    return _trapz(values * x.reshape(shape), np.log(x), axis=axis)


# ===========================================================================
# EBL -- Step 7, applied only at the *observer* end, never at the model's own z
# ===========================================================================

def ebl_transmission(energy_obs_GeV: np.ndarray, z: float, reference: str = "dominguez",
                     alpha_norm: float = 1.0) -> np.ndarray:
    """exp(-tau(E, z)), the surviving fraction. Needs $GAMMAPY_DATA/ebl/... on disk."""
    from gammapy.modeling.models import EBLAbsorptionNormSpectralModel

    if not os.environ.get("GAMMAPY_DATA"):
        raise RuntimeError(
            "GAMMAPY_DATA is not set, so gammapy cannot find its EBL tables "
            "(Step 7 of the guidelines). Set it to the directory containing "
            "ebl/ebl_dominguez11.fits.gz."
        )
    model = EBLAbsorptionNormSpectralModel.read_builtin(
        reference=reference, redshift=z, alpha_norm=alpha_norm)
    energy = np.atleast_1d(np.asarray(energy_obs_GeV, dtype=float)) * u.GeV
    return np.asarray(model(energy).to_value(u.dimensionless_unscaled))


def diagnose_ebl(energy_obs_GeV: np.ndarray, flux_column: np.ndarray, z: float,
                 reference: str = "dominguez", n_fit: int = 10) -> dict:
    """Step 1's "spot-check": is EBL already baked into this flux column?

    Fits a power law to the lowest `n_fit` energies (where EBL is negligible),
    extrapolates across the band, and compares the observed deficit to
    exp(-tau(E,z)). A matrix carrying EBL tracks the transmission curve; an
    intrinsic one sits near 1 through the energies where EBL would have
    already bitten hard.
    """
    E = np.asarray(energy_obs_GeV, dtype=float)
    col = np.asarray(flux_column, dtype=float)
    good = np.isfinite(col) & (col > 0)
    E, col = E[good], col[good]
    slope, intercept = np.polyfit(np.log(E[:n_fit]), np.log(col[:n_fit]), 1)
    ratio = col / np.exp(intercept + slope * np.log(E))
    trans = ebl_transmission(E, z, reference=reference)
    bitten = trans < 0.7
    absorbed = bool(bitten.any() and np.all(ratio[bitten] < 0.5 * (1 + trans[bitten])))
    return {"energy_GeV": E, "observed_ratio": ratio, "ebl_transmission": trans,
            "absorbed": absorbed, "max_ebl_suppression": float(trans.min()) if trans.size else 1.0}


# ===========================================================================
# One angle, rest frame: Steps 1 (Options 1-3) and 2
# ===========================================================================

@dataclass
class GRBModel:
    """L(E', t') -- rest-frame photon-differential luminosity, ph/(s GeV), for
    one viewing angle. Distance-independent, EBL-free, intrinsic by definition.
    """

    theta_deg: float
    energy: np.ndarray          # GeV, rest-frame E', strictly ascending
    time: np.ndarray            # s, rest-frame t' - t0, strictly ascending
    L: np.ndarray                # ph/(s GeV), shape (n_E, n_t)
    D_L_m: float = float("nan")  # Mpc -- the model file's own distance (informational)
    z_m: float = float("nan")
    E_iso: float = float("nan")  # erg -- prompt E_iso, metadata only (see catO5 docstring)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.energy = np.asarray(self.energy, dtype=float)
        self.time = np.asarray(self.time, dtype=float)
        self.L = np.asarray(self.L, dtype=float)
        if self.L.shape != (self.energy.size, self.time.size):
            raise ValueError(
                f"L shape {self.L.shape} != (len(energy), len(time)) = "
                f"({self.energy.size}, {self.time.size})"
            )
        self._interp = LogLogInterpolator2D(self.energy, self.time, self.L)

    # -- Step 2's interpolation note: query at any (E', t') -----------------
    def at(self, energy_rest, time_rest, report_out_of_range: bool = False):
        return self._interp(energy_rest, time_rest, report_out_of_range=report_out_of_range)

    def band_integral(self, e_lo: float, e_hi: float, t_lo: float, t_hi: float,
                      weight_by_energy: bool = True, n_e: int = 200, n_t: int = 200) -> float:
        """int int [E' *] L(E', t') dE' dt' on a fine log grid -- Step 6's D_i."""
        e_grid = np.geomspace(e_lo, e_hi, n_e)
        t_grid = np.geomspace(t_lo, t_hi, n_t)
        L_grid = self.at(e_grid, t_grid)
        v = L_grid * e_grid[:, None] if weight_by_energy else L_grid
        per_time = _band_integral(v, e_grid, axis=0)
        return float(_band_integral(per_time, t_grid, axis=0))

    def radiated_energy(self) -> float:
        """int int E' L dE' dt' over the whole tabulated grid, in erg (GeV -> erg)."""
        return self.band_integral(self.energy[0], self.energy[-1],
                                  self.time[0], self.time[-1]) * u.GeV.to(u.erg)

    def light_curve(self) -> np.ndarray:
        """int E' L dE' over the whole energy grid, erg/s, at each time node."""
        return _band_integral(self.L * self.energy[:, None], self.energy, axis=0) * u.GeV.to(u.erg)

    def peak(self) -> Tuple[float, float]:
        """(t'_peak [s], L_peak [erg/s]) of `light_curve` -- the time node of its
        maximum, i.e. what Step 5's peak-aligned interpolation lines up."""
        lc = self.light_curve()
        i = int(np.argmax(lc))
        return float(self.time[i]), float(lc[i])

    # -- common-E_iso standardisation ----------------------------------------
    def rescaled_to_eiso(self, E_iso: float, L_exponent: float = EISO_L_EXPONENT,
                         t_exponent: float = EISO_T_EXPONENT) -> "GRBModel":
        """This model as if its event had isotropic energy `E_iso` (erg):
        L -> k^L_exponent L on the time grid t' -> k^t_exponent t', k = E_iso/self.E_iso.

        The energy axis and the spectral shape are left untouched (the catO5
        spectra are time-independent power laws, so there is no spectral
        break for nu_c ~ E^(-2/3) to move). See EISO_L_EXPONENT/EISO_T_EXPONENT
        for the default exponents; (1, 0) is a plain L/E_iso normalisation.
        """
        if not (np.isfinite(self.E_iso) and self.E_iso > 0):
            raise ValueError(f"model at theta={self.theta_deg} deg has no E_iso to rescale from")
        k = float(E_iso) / self.E_iso
        meta = dict(self.meta)
        meta.setdefault("E_iso_original", self.E_iso)
        meta["eiso_scaling"] = {"L_exponent": L_exponent, "t_exponent": t_exponent}
        return GRBModel(
            theta_deg=self.theta_deg, energy=self.energy, time=self.time * k ** t_exponent,
            L=self.L * k ** L_exponent, D_L_m=self.D_L_m, z_m=self.z_m, E_iso=float(E_iso),
            meta=meta,
        )

    # -- Step 7: the one way back to observer flux --------------------------
    def to_observer(self, D_L_Mpc: float, z: float, energy_obs=None, time_obs=None,
                    apply_ebl: bool = False, ebl_reference: str = "dominguez",
                    warn_out_of_range: bool = True):
        """F_i(E,t) = (1+z)^2/(4 pi D_L^2) * L((1+z)E, t/(1+z)) [* EBL].

        With `energy_obs`/`time_obs` left at None they default to this
        model's own grid pulled back -- so re-projecting to the (D_L, z) a
        model came from is an exact round trip (Step 7's cross-check).
        """
        if energy_obs is None:
            energy_obs = self.energy / (1.0 + z)
        if time_obs is None:
            time_obs = self.time * (1.0 + z)
        energy_obs = np.atleast_1d(np.asarray(energy_obs, dtype=float))
        time_obs = np.atleast_1d(np.asarray(time_obs, dtype=float))

        L_vals, frac_oor = self.at((1.0 + z) * energy_obs, time_obs / (1.0 + z),
                                   report_out_of_range=True)
        if warn_out_of_range and frac_oor > 0:
            warnings.warn(
                f"to_observer: {frac_oor:.1%} of query points fell outside the "
                "tabulated rest-frame grid and were extrapolated as a local "
                "power law.", stacklevel=2,
            )

        prefactor = (1.0 + z) ** 2 / (4.0 * np.pi * (D_L_Mpc * _CM_PER_MPC) ** 2)
        F = prefactor * L_vals

        if apply_ebl:
            F = F * ebl_transmission(energy_obs, z, reference=ebl_reference)[:, None]

        return energy_obs, time_obs, F

    # -- persistence: the standardised format --------------------------------
    def save(self, path) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, theta_deg=self.theta_deg, energy_GeV=self.energy, time_s=self.time,
            L_ph_s_GeV=self.L, D_L_m_Mpc=self.D_L_m, z_m=self.z_m, E_iso_erg=self.E_iso,
            meta_json=json.dumps(self.meta, default=str),
        )

    @classmethod
    def load(cls, path) -> "GRBModel":
        d = np.load(path, allow_pickle=False)
        return cls(
            theta_deg=float(d["theta_deg"]), energy=d["energy_GeV"], time=d["time_s"],
            L=d["L_ph_s_GeV"], D_L_m=float(d["D_L_m_Mpc"]), z_m=float(d["z_m"]),
            E_iso=float(d["E_iso_erg"]), meta=json.loads(str(d["meta_json"])),
        )

    # -- Option 1: the INAF / CTA-GW catalogue files -------------------------
    @classmethod
    def from_inaf_fits(cls, path, cosmology=None, trim: bool = True,
                       time_representative: str = "geometric") -> "GRBModel":
        """Step 1 (Option 1) + Step 2: read one catO5_*.fits file and project
        it, by exact regrid, to the rest frame."""
        from astropy.io import fits

        with fits.open(path) as hdul:
            header = hdul["PRIMARY"].header
            energy_obs = np.asarray(hdul["ENERGIES"].data["Energies"], dtype=float)  # GeV
            t_ini = np.asarray(hdul["TIMES"].data["Initial Time"], dtype=float)      # s
            t_fin = np.asarray(hdul["TIMES"].data["Final Time"], dtype=float)
            # SPECTRA is stored (time, energy); this module's convention is (energy, time).
            F_obs = np.asarray(hdul["SPECTRA"].data.tolist(), dtype=float).T  # ph/cm2/s/GeV

        t_obs = {
            "geometric": lambda: np.sqrt(t_ini * t_fin),
            "arithmetic": lambda: 0.5 * (t_ini + t_fin),
            "initial": lambda: t_ini,
        }[time_representative]()

        trimmed = {"energies": [], "times": []}
        if trim:
            keep_E, keep_t = _largest_positive_block(np.isfinite(F_obs) & (F_obs > 0))
            trimmed = {"energies": energy_obs[~keep_E].tolist(), "times": t_obs[~keep_t].tolist()}
            energy_obs, F_obs = energy_obs[keep_E], F_obs[np.ix_(keep_E, keep_t)]
            t_obs = t_obs[keep_t]

        D_L_m_Mpc = (float(header["DISTANCE"]) * u.kpc).to_value(u.Mpc)
        z_m = float(redshift_from_luminosity_distance(D_L_m_Mpc, cosmology=cosmology))
        theta_m = float(header["ANGLE"])
        E_iso = float(header["EISO"])

        # Step 2, exact regrid -- no interpolation at this step.
        prefactor = 4.0 * np.pi * (D_L_m_Mpc * _CM_PER_MPC) ** 2 / (1.0 + z_m) ** 2
        energy_rest = (1.0 + z_m) * energy_obs
        time_rest = t_obs / (1.0 + z_m)
        L_rest = prefactor * F_obs

        return cls(
            theta_deg=theta_m, energy=energy_rest, time=time_rest, L=L_rest,
            D_L_m=D_L_m_Mpc, z_m=z_m, E_iso=E_iso,
            meta={"origin": "from_inaf_fits", "path": str(path),
                 "time_representative": time_representative, "trimmed": trimmed},
        )

    # -- Option 2: an analytical rest-frame shape f(E') x g(t') --------------
    @classmethod
    def from_analytic(cls, spectral_fn: Callable[[np.ndarray], np.ndarray],
                      energy_rest: np.ndarray, time_rest: np.ndarray,
                      temporal_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                      theta_deg: float = 0.0, L0: float = 1.0,
                      E_ref: Optional[float] = None) -> "GRBModel":
        """L(E', t') = L0 * f(E')/f(E_ref) * g(t'). Skips Step 2 -- the
        K-correction is analytic. `spectral_fn`/`temporal_fn` take/return
        plain floats in GeV/s (no units)."""
        energy_rest = np.asarray(energy_rest, dtype=float)
        time_rest = np.asarray(time_rest, dtype=float)
        if E_ref is None:
            E_ref = float(np.sqrt(energy_rest.min() * energy_rest.max()))

        shape_E = np.asarray(spectral_fn(energy_rest), dtype=float) / float(spectral_fn(E_ref))
        shape_t = (np.ones_like(time_rest) if temporal_fn is None
                  else np.asarray(temporal_fn(time_rest), dtype=float))
        L = L0 * shape_E[:, None] * shape_t[None, :]

        return cls(theta_deg=theta_deg, energy=energy_rest, time=time_rest, L=L,
                  meta={"origin": "from_analytic", "E_ref": E_ref, "L0": L0})

    # -- Option 3: constant Gamma power law -- the closed-form test case ----
    @classmethod
    def constant_power_law(cls, energy_rest: np.ndarray, time_rest: np.ndarray,
                           index: float = 2.0, L0: float = 1.0, E_ref: float = 1.0,
                           theta_deg: float = 0.0) -> "GRBModel":
        """f(E') = (E'/E_ref)^-index, g(t') = 1. For index=2, E'_0=E_0, E'_1=E_1,
        the observed energy flux in the band equals L_k / (4 pi d_i^2) for every
        z_i -- the analytic anchor used to validate the whole chain."""
        return cls.from_analytic(
            lambda E: (E / E_ref) ** (-float(index)), energy_rest, time_rest,
            temporal_fn=None, theta_deg=theta_deg, L0=L0, E_ref=E_ref,
        )


def _largest_positive_block(good: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Greedily drop whole energy rows / time columns until every remaining
    cell of `good` (shape (n_E, n_t)) is True; returns the two keep-masks.

    A grid model needs a rectangular block, so one bad cell condemns its
    whole row or column. This is what peels off the INAF files' all-zero
    10 TeV row and the off-axis events' all-zero leading time columns.
    """
    keep_E = np.ones(good.shape[0], bool)
    keep_t = np.ones(good.shape[1], bool)
    while True:
        sub = good[np.ix_(keep_E, keep_t)]
        if sub.size == 0:
            raise ValueError("no finite, positive (energy, time) block remains")
        if sub.all():
            return keep_E, keep_t
        bad = ~sub
        score_E = bad.sum(axis=1) / keep_t.sum()
        score_t = bad.sum(axis=0) / keep_E.sum()
        if score_E.max() >= score_t.max():
            keep_E[np.flatnonzero(keep_E)[int(np.argmax(score_E))]] = False
        else:
            keep_t[np.flatnonzero(keep_t)[int(np.argmax(score_t))]] = False


# ===========================================================================
# Step 5: the model at any theta, discretised as a set of angle files
# ===========================================================================

@dataclass
class GRBModelSet:
    """A dict of GRBModel keyed by viewing angle, plus the three ways of the
    notes to get a model at an angle that isn't on the grid."""

    models: Dict[float, GRBModel]

    @property
    def angles_deg(self) -> np.ndarray:
        return np.array(sorted(self.models))

    def rescaled_to_eiso(self, E_iso: float, L_exponent: float = EISO_L_EXPONENT,
                         t_exponent: float = EISO_T_EXPONENT) -> "GRBModelSet":
        """Every angle rescaled to one common E_iso (see GRBModel.rescaled_to_eiso).

        Each catO5 file is a different catalogue event, with E_iso spanning two
        decades, so the raw set mixes the angle dependence with that event-to-
        event scatter. On a common E_iso, the differences left between angles
        are the angle's (plus whatever parameters the files do not record).
        Because the rescaling is a uniform shift in (log t', log L), it commutes
        with the "interp" option: rescaling the set then interpolating equals
        interpolating then rescaling the result (which carries the
        log-interpolated E_iso of its two brackets).
        """
        return GRBModelSet(models={
            theta: m.rescaled_to_eiso(E_iso, L_exponent=L_exponent, t_exponent=t_exponent)
            for theta, m in self.models.items()
        })

    def get_model(self, theta_deg: float, method: str = "nearest",
                 rng: Optional[np.random.Generator] = None,
                 align_peaks: bool = True) -> GRBModel:
        """Step 5. `method`:

        - "nearest"    (Option A) -- the file with theta^m closest to theta_deg.
        - "stochastic" (Option B) -- for the bracketing theta_a <= theta_deg <
          theta_b, use theta_b with probability (theta_deg-theta_a)/(theta_b-theta_a),
          otherwise theta_a. Correct on average over iterations; needs `rng`.
        - "interp"     (Option C) -- interpolate log L linearly in theta between
          theta_a and theta_b. With `align_peaks` (default) this is done at
          fixed phase log t' - log t'_peak rather than at fixed t', so the
          result has a single peak at the interpolated peak time, with the
          peak luminosity and the rise/decay slopes interpolated consistently.
          With align_peaks=False it is the notes' plain fixed-t' version,
          which blends two peaks at different t' into a double-peaked light
          curve. See _interpolate_in_angle.

        theta_deg outside the tabulated range is clipped to the nearest edge
        angle (with a warning), since the catalogue only spans a handful of
        discrete angles. A theta_deg on a tabulated angle returns that model.
        """
        angles = self.angles_deg
        if angles.size == 0:
            raise ValueError("GRBModelSet is empty")
        if theta_deg <= angles[0]:
            if theta_deg < angles[0]:
                warnings.warn(f"theta={theta_deg:.2f} deg is below the lowest "
                              f"tabulated angle {angles[0]:.2f} deg; clipping.", stacklevel=2)
            return self.models[angles[0]]
        if theta_deg >= angles[-1]:
            if theta_deg > angles[-1]:
                warnings.warn(f"theta={theta_deg:.2f} deg is above the highest "
                              f"tabulated angle {angles[-1]:.2f} deg; clipping.", stacklevel=2)
            return self.models[angles[-1]]
        on_grid = np.isclose(angles, theta_deg, rtol=0.0, atol=1e-9)
        if on_grid.any():
            return self.models[angles[int(np.argmax(on_grid))]]

        i = int(np.searchsorted(angles, theta_deg))
        theta_a, theta_b = angles[i - 1], angles[i]
        model_a, model_b = self.models[theta_a], self.models[theta_b]

        if method == "nearest":
            return model_a if (theta_deg - theta_a) <= (theta_b - theta_deg) else model_b
        if method == "stochastic":
            rng = rng if rng is not None else np.random.default_rng()
            p_b = (theta_deg - theta_a) / (theta_b - theta_a)
            return model_b if rng.random() < p_b else model_a
        if method == "interp":
            return _interpolate_in_angle(model_a, model_b, theta_deg, align_peaks=align_peaks)
        raise ValueError(f"method must be 'nearest', 'stochastic' or 'interp', got {method!r}")

    def evaluate(self, theta_deg: float, energy_rest, time_rest, method: str = "nearest",
                rng: Optional[np.random.Generator] = None, align_peaks: bool = True):
        """Model for any (E', t', theta), even off the tabulated angle grid."""
        model = self.get_model(theta_deg, method=method, rng=rng, align_peaks=align_peaks)
        return model.at(energy_rest, time_rest)

    # -- persistence ----------------------------------------------------------
    def save(self, directory) -> None:
        os.makedirs(directory, exist_ok=True)
        for theta, model in self.models.items():
            model.save(os.path.join(directory, f"theta_{theta:07.3f}deg.npz"))

    @classmethod
    def load(cls, directory) -> "GRBModelSet":
        files = sorted(glob.glob(os.path.join(str(directory), "*.npz")))
        if not files:
            raise FileNotFoundError(f"no standardized model files (*.npz) found in {directory}")
        models = {}
        for f in files:
            m = GRBModel.load(f)
            models[m.theta_deg] = m
        return cls(models=models)

    @classmethod
    def from_inaf_dir(cls, directory, pattern: str = "*.fits", cosmology=None) -> "GRBModelSet":
        """Step 1 (Option 1), for every angle file in `directory` at once."""
        files = sorted(glob.glob(os.path.join(str(directory), pattern)))
        if not files:
            raise FileNotFoundError(f"no model files matching {pattern!r} found in {directory}")
        models = {}
        for f in files:
            m = GRBModel.from_inaf_fits(f, cosmology=cosmology)
            if m.theta_deg in models:
                warnings.warn(f"duplicate angle {m.theta_deg} deg from {f}; overwriting.",
                              stacklevel=2)
            models[m.theta_deg] = m
        return cls(models=models)


def _union_grid(*grids: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Sorted union of 1D grids, merging nodes closer than `tol`."""
    g = np.unique(np.concatenate(grids))
    return g[np.concatenate([[True], np.diff(g) > tol])]


def _single_peak_part(model: GRBModel) -> GRBModel:
    """`model` restricted to the time nodes over which its light curve rises
    monotonically to its peak and decays monotonically after it -- cut at the
    first local minimum on either side. A re-brightening past that point (the
    theta=77.8 deg file's last node) would otherwise be extrapolated, as a
    rising power law, across the whole phase range of the other model."""
    lc = model.light_curve()
    i = int(np.argmax(lc))
    lo, hi = i, i
    while lo > 0 and lc[lo - 1] <= lc[lo]:
        lo -= 1
    while hi < lc.size - 1 and lc[hi + 1] <= lc[hi]:
        hi += 1
    if lo == 0 and hi == lc.size - 1:
        return model
    return GRBModel(
        theta_deg=model.theta_deg, energy=model.energy, time=model.time[lo:hi + 1],
        L=model.L[:, lo:hi + 1], D_L_m=model.D_L_m, z_m=model.z_m, E_iso=model.E_iso,
        meta=model.meta,
    )


def _interpolate_in_angle(model_a: GRBModel, model_b: GRBModel, theta_deg: float,
                          align_peaks: bool = True) -> GRBModel:
    """Step 5, Option C: log L linear in theta, at fixed (E', phase).

    With `align_peaks`, each model's time axis is written as the phase
    x = log t' - log t'_peak (t'_peak from GRBModel.peak -- the catO5 peaks are
    achromatic, so one shift per model aligns every energy), log L is
    interpolated at fixed (E', x), and the result is placed at
    log t'_peak = (1-w) log t'_peak,a + w log t'_peak,b. So the peak time,
    the peak luminosity and the slope d log L / d log t' at every phase are
    each interpolated linearly in theta; and since both inputs peak at x = 0
    (a node of both), the result has exactly one peak -- each input is first
    cut to its single-peaked part (_single_peak_part) to guarantee that.
    Without it, x = log t' and this is the plain fixed-t' interpolation.

    The output grid is the union of both inputs' (log E', x) nodes, so w = 0/1
    reproduce model_a/model_b exactly (bar any nodes _single_peak_part cut);
    where only one input is tabulated the other is extrapolated as a local
    power law (e.g. the theta=77.8 deg file stops 0.3 dex after its peak, so
    its decay beyond that is its last slope, ~t^-2).
    """
    w = (theta_deg - model_a.theta_deg) / (model_b.theta_deg - model_a.theta_deg)
    u_a = u_b = 0.0
    if align_peaks:
        model_a, model_b = _single_peak_part(model_a), _single_peak_part(model_b)
        t_pk = [m.peak()[0] for m in (model_a, model_b)]
        for m, t in zip((model_a, model_b), t_pk):
            if t in (m.time[0], m.time[-1]):
                warnings.warn(
                    f"theta={m.theta_deg} deg: the light-curve peak is at the edge of the "
                    "tabulated time grid, so the peak alignment there is only approximate.",
                    stacklevel=3,
                )
        u_a, u_b = np.log10(t_pk)

    energy = 10.0 ** _union_grid(np.log10(model_a.energy), np.log10(model_b.energy))
    x = _union_grid(np.log10(model_a.time) - u_a, np.log10(model_b.time) - u_b)
    log_L = ((1.0 - w) * np.log10(model_a.at(energy, 10.0 ** (x + u_a)))
             + w * np.log10(model_b.at(energy, 10.0 ** (x + u_b))))

    E_iso = float("nan")
    if model_a.E_iso > 0 and model_b.E_iso > 0:  # False for NaN
        E_iso = 10.0 ** ((1.0 - w) * np.log10(model_a.E_iso) + w * np.log10(model_b.E_iso))

    return GRBModel(
        theta_deg=theta_deg, energy=energy, time=10.0 ** (x + (1.0 - w) * u_a + w * u_b),
        L=10.0 ** log_L, E_iso=E_iso,
        meta={"origin": "interp", "align_peaks": align_peaks,
              "bracket_deg": (model_a.theta_deg, model_b.theta_deg)},
    )
