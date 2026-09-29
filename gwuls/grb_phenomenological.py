"""
grb_phenomenological.py -- Option 2, for real: the stochastic phenomenological
recipe of L. Nava's internal note ("TeV counterparts of BNS mergers", Jan 2020)
and its published, refined form in Abe et al. 2026 (ApJ 1004, 46, the CTAO
Consortium's GW-GRB paper). Every constant below is cited to a section/equation
in one or both sources; see the module-level `SOURCES` dict.

Where this differs from `grb_model.py`'s Option 1 (the five `catO5_*.fits`
files): those files are five *fixed* events, each a snapshot of this same
phenomenological chain run once by L. Nava in Sept 2021 (see their FITS
headers: AUTHOR=Lara, HISTORY="BNS catalog from B. Patricelli. Catalog O5").
This module runs the chain itself, so every call draws a fresh, independent
event -- restoring the event-to-event scatter (in E_iso, theta_core,
Gamma_core, and especially the post-peak decay index, sigma = 0.48 dex) that
picking from only five fixed files cannot represent.

What is (and is not) reproduced here
-------------------------------------
* The on-axis (theta_view ~ 0, within the jet core) light curve is built
  exactly as both sources describe it: E_peak -> Amati -> E_gamma,iso -> the
  Berger (2014) L_X-E_iso relation -> the L_TeV/L_X ratio -> a broken power
  law in time anchored at the two luminosities at t = 11 h, peaking at the
  deceleration time. This part is unambiguous and is implemented faithfully.

* The jet structure (Gaussian in energy and Lorentz factor: Abe et al. 2026
  Eqs. 1-2) is implemented as given.

* The off-axis extension comes in two versions. `grb_model_at_theta` (used
  with `sample_events`) is the simpler prescription of Nava's 2020 note, Sec. 5:
  evaluate the Gaussian structure locally at theta_view and treat that element
  as its own scaled-down on-axis event, clipped at theta_max. The fixed-jet
  section at the end of this module (`JetParams`, `offaxis_luminosity`,
  `model_at_theta`) implements instead the multi-element, equal-arrival-time
  integral that Abe et al. (2026, Sec. 3.4) use, after Lamb & Kobayashi
  (2017). It is validated in `notebooks/setup_model_phenomenological_fixed.ipynb`
  against an independent 1-D calculation on-axis and by shape fits to all five
  catO5 files.

Everything below produces a `grb_model.GRBModel` (theta_deg, rest-frame
L(E', t')), so it composes with the rest of `grb_model.py` unchanged --
`.to_observer()`, `.save()`/`.load()`, `GRBModelSet`, etc. all work on its
output exactly as they do on an INAF file.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import astropy.units as u
from astropy import constants as const

from .grb_model import GRBModel, GRBModelSet
from .utils import trapz as _trapz

__all__ = [
    "SOURCES",
    "draw_epeak_keV",
    "epeak_to_egiso_erg",
    "draw_egiso_direct_erg",
    "draw_theta_core_deg",
    "draw_gamma_core",
    "jet_structure",
    "THETA_MAX_GAMMA0_FLOOR",
    "theta_max_deg",
    "deceleration_time_s",
    "x_ray_luminosity_11h_erg_s",
    "vhe_luminosity_11h_erg_s",
    "draw_light_curve_shape",
    "draw_photon_index",
    "l_peak_from_anchor",
    "sample_events",
    "grb_model_at_theta",
    # fixed (deterministic) jet + the off-axis element integral
    "JetParams",
    "BENCHMARK",
    "VHE_BAND_GEV",
    "DEFAULT_THETA_GRID_DEG",
    "draw_jet_params",
    "one_sigma_variants",
    "offaxis_luminosity",
    "model_at_theta",
    "model_set",
]

SOURCES = {
    "nava2020": 'L. Nava, "TeV counterparts of BNS mergers" (internal note, Jan 2020)',
    "abe2026": "Abe et al. 2026, ApJ 1004, 46 (CTAO Consortium)",
    "ghirlanda2016": "Ghirlanda et al. 2016, A&A 594, A84 (Eqs. 13, 15, model a mode values)",
    "berger2014": "Berger 2014, ARA&A 52, 43 (L_X,11h - E_iso relation)",
    "fong2015": "Fong et al. 2015, ApJ 815, 102 (theta_core distribution)",
    "ghirlanda2019": "Ghirlanda et al. 2019, Science 363, 968 (GRB 170817A Gamma_core)",
    "lamb_kobayashi2017": "Lamb & Kobayashi 2017, MNRAS 472, 4953 "
                          "(the off-axis method Abe et al. 2026 Sec. 3.4 use; "
                          "`offaxis_luminosity` -- see module docstring)",
    "blandford_mckee1976": "Blandford & McKee 1976, Phys. Fluids 19, 1130 "
                           "(Gamma ~ t^-3/8 after deceleration, Abe et al. 2026 Eq. 7)",
}

# ===========================================================================
# Population parameters -- every value cited to its source
# ===========================================================================

# E_peak broken power law (Abe et al. 2026 Eq. 3; Ghirlanda et al. 2016 Eq. 13, model a,
# mode values). phi(Epeak) ~ (Epeak/1.4 MeV)^EPEAK_INDEX_LO for Epeak < 1.4 MeV,
# (Epeak/1.4 MeV)^EPEAK_INDEX_HI above.
EPEAK_BREAK_KEV = 1400.0
EPEAK_INDEX_LO = -0.8
EPEAK_INDEX_HI = -2.6
EPEAK_MIN_KEV = 0.1
EPEAK_MAX_KEV = 1.0e5

# Amati relation (Abe et al. 2026 Eq. 4; Ghirlanda et al. 2016 Eq. 15, model a, mode
# values): log10(E_gamma,iso / 1e51 erg) = AMATI_NORM + AMATI_SLOPE * log10(Epeak / 670 keV).
# Applied deterministically -- no intrinsic scatter term was recoverable from the extracted
# paper text; the real Amati relation normally carries its own scatter (flagged in the
# notebook's caveats).
AMATI_NORM = 0.036
AMATI_SLOPE = 1.1
AMATI_EPEAK_REF_KEV = 670.0

# Nava (2020) README Eq. 1's shortcut: draw E_gamma,iso directly from the broken power law
# she gets by propagating the E_peak distribution above through a *deterministic* Amati
# relation. Kept here as an independent second path -- comparing it against
# `epeak_to_egiso_erg(draw_epeak_keV(...))` is a direct, cheap consistency check between
# the 2020 note and the 2026 paper (see the notebook, Part 1).
EISO_BREAK_ERG = 2.1e51
EISO_INDEX_LO = -0.728
EISO_INDEX_HI = -2.366

# theta_core (Nava 2020 Sec. 5.1 / Abe et al. 2026 Sec. 3.1, both citing Fong et al. 2015):
# lognormal, mean log10(theta_core/deg) = 1.15 (i.e. 14 deg), sigma = 0.2 dex.
LOG10_THETA_CORE_DEG_MEAN = 1.15
LOG10_THETA_CORE_DEG_STD = 0.2

# Gamma_core (Nava 2020 Sec. 5.2 / Abe et al. 2026 Sec. 3.1): lognormal, mean
# log10(Gamma_core) = 2.3 (i.e. 200, similar to the GRB 170817A value of Ghirlanda et al.
# 2019), sigma = 0.2 dex.
LOG10_GAMMA_CORE_MEAN = 2.3
LOG10_GAMMA_CORE_STD = 0.2

ETA_GAMMA = 0.2       # prompt radiative efficiency (Nava Sec. 5.1 / Abe Sec. 3.1)
N_ISM_CM3 = 0.1        # constant circumburst density (Abe Sec. 3.2)

# Light-curve temporal shape (Nava 2020 Sec. 4, "alpha1"/"alpha2" / Abe et al. 2026 Sec. 3.3,
# "beta1"/"beta2" -- identical numbers under a renamed symbol, six years apart).
BETA1_MEAN, BETA1_STD = 2.0, 0.05      # rise before peak: L ~ t^beta1
BETA2_MEAN, BETA2_STD = -1.45, 0.48    # decay after peak: L ~ t^beta2 (from 22 short GRBs)

# L_X,11h - E_iso relation (Berger 2014, via Nava Sec. 4 / Abe Eq. 5, identical):
# L_X,11h = LX11H_NORM * (E_gamma,iso / 1e51 erg)^LX11H_SLOPE erg/s, with LX11H_SCATTER_DEX
# lognormal scatter.
LX11H_NORM = 8.5e43
LX11H_SLOPE = 0.83
LX11H_SCATTER_DEX = 0.5

# L_TeV(11h)/L_X(11h) ratio (Nava Sec. 4 / Abe Sec. 3.2, identical): lognormal, peaked at 1
# (0 dex), scatter 0.3 dex. Abe et al. specify the VHE band as 0.3-1 TeV.
LTEV_LX_RATIO_LOG10_SCATTER = 0.3

# Photon index of dN/dE ~ E^-PHOTON_INDEX. Nava (2020) Sec. 2 gives only "alpha ~ -2"; Abe
# et al. (2026) Sec. 3.2 refines this to a Gaussian peaked at 2.2, sigma 0.1 -- used here as
# the more precise, later value.
PHOTON_INDEX_MEAN = 2.2
PHOTON_INDEX_STD = 0.1

# The observer/rest-frame time at which L_X,11h and L_TeV,11h are defined. Both sources build
# "the light-curve for an observer located along the jet axis" without invoking z at this
# step, so this module treats 11 h as directly the *rest-frame* anchor time t' = 11 h --
# consistent with how GRBModel is otherwise a rest-frame, distance-independent object, but
# an assumption worth flagging (see notebook caveats).
T_ANCHOR_S = 11.0 * 3600.0

_M_P_G = const.m_p.cgs.value
_C_CMS = const.c.cgs.value


# ===========================================================================
# Samplers
# ===========================================================================

def _sample_broken_power_law(rng: np.random.Generator, n: int, x_break: float,
                             idx_lo: float, idx_hi: float, x_min: float, x_max: float
                             ) -> np.ndarray:
    """Inverse-CDF draw from phi(x) = (x/x_break)^idx_lo for x_min<=x<x_break,
    (x/x_break)^idx_hi for x_break<=x<=x_max (continuous at x_break). Neither
    index used in this module is -1, so no log-divergent special case is needed.
    """
    def _pl_integral(a: float, b: float, idx: float) -> float:
        return (b ** (idx + 1.0) - a ** (idx + 1.0)) / (idx + 1.0)

    w_lo = x_break ** (-idx_lo) * _pl_integral(x_min, x_break, idx_lo)
    w_hi = x_break ** (-idx_hi) * _pl_integral(x_break, x_max, idx_hi)
    norm = w_lo + w_hi

    u = rng.uniform(0.0, norm, size=n)
    x = np.empty(n, dtype=float)
    in_lo = u < w_lo

    target_lo = u[in_lo] / x_break ** (-idx_lo)
    x[in_lo] = (target_lo * (idx_lo + 1.0) + x_min ** (idx_lo + 1.0)) ** (1.0 / (idx_lo + 1.0))

    target_hi = (u[~in_lo] - w_lo) / x_break ** (-idx_hi)
    x[~in_lo] = (target_hi * (idx_hi + 1.0) + x_break ** (idx_hi + 1.0)) ** (1.0 / (idx_hi + 1.0))

    return x


def draw_epeak_keV(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """E_peak, keV -- Abe et al. 2026 Eq. 3 / Ghirlanda et al. 2016 Eq. 13."""
    return _sample_broken_power_law(rng, n, EPEAK_BREAK_KEV, EPEAK_INDEX_LO, EPEAK_INDEX_HI,
                                    EPEAK_MIN_KEV, EPEAK_MAX_KEV)


def epeak_to_egiso_erg(epeak_keV: np.ndarray) -> np.ndarray:
    """The Amati relation -- Abe et al. 2026 Eq. 4 / Ghirlanda et al. 2016 Eq. 15
    (model a, mode values), applied deterministically (see module notes)."""
    log10_egiso_51 = AMATI_NORM + AMATI_SLOPE * np.log10(np.asarray(epeak_keV) / AMATI_EPEAK_REF_KEV)
    return (10.0 ** log10_egiso_51) * 1.0e51


# The direct E_iso sampler's bounds are set by pushing the E_peak bounds through the same
# (deterministic) Amati relation, so both sampling paths cover the same physical range.
_EISO_MIN_ERG = float(epeak_to_egiso_erg(np.array([EPEAK_MIN_KEV]))[0])
_EISO_MAX_ERG = float(epeak_to_egiso_erg(np.array([EPEAK_MAX_KEV]))[0])


def draw_egiso_direct_erg(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """E_gamma,iso, erg -- Nava (2020) README Eq. 1's shortcut broken power law
    (E_iso,b = 2.1e51 erg), the *already Amati-propagated* form of `draw_epeak_keV`
    + `epeak_to_egiso_erg`. An independent second path to the same quantity; compare
    the two in the notebook."""
    return _sample_broken_power_law(rng, n, EISO_BREAK_ERG, EISO_INDEX_LO, EISO_INDEX_HI,
                                    _EISO_MIN_ERG, _EISO_MAX_ERG)


def draw_theta_core_deg(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """theta_core, deg -- lognormal (Fong et al. 2015, via Nava Sec. 5.1 / Abe Sec. 3.1)."""
    return 10.0 ** rng.normal(LOG10_THETA_CORE_DEG_MEAN, LOG10_THETA_CORE_DEG_STD, size=n)


def draw_gamma_core(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Gamma_core -- lognormal (Nava Sec. 5.2 / Abe Sec. 3.1)."""
    return 10.0 ** rng.normal(LOG10_GAMMA_CORE_MEAN, LOG10_GAMMA_CORE_STD, size=n)


def draw_light_curve_shape(rng: np.random.Generator, n: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """(beta1, beta2) -- Nava Sec. 4 ("alpha1"/"alpha2") / Abe Sec. 3.3 ("beta1"/"beta2"),
    identical numbers."""
    beta1 = rng.normal(BETA1_MEAN, BETA1_STD, size=n)
    beta2 = rng.normal(BETA2_MEAN, BETA2_STD, size=n)
    return beta1, beta2


def draw_photon_index(rng: np.random.Generator, n: int = 1) -> np.ndarray:
    """Photon index Gamma_phot of dN/dE ~ E^-Gamma_phot -- Abe Sec. 3.2 (refines Nava's
    "alpha ~ -2" to a Gaussian peaked at 2.2, sigma 0.1)."""
    return rng.normal(PHOTON_INDEX_MEAN, PHOTON_INDEX_STD, size=n)


# ===========================================================================
# Jet structure and dynamics
# ===========================================================================

def jet_structure(theta_view_deg, theta_core_deg, gamma_core):
    """(eps_k_fraction, Gamma0_theta): the Gaussian jet-structure profiles of
    Abe et al. 2026 Eqs. 1-2. eps_k_fraction = eps_k(theta)/eps_k,core.

    Convention note: Nava's 2020 README Eqs. 2-3 use exp(-theta^2 / 2 theta_core^2) for
    *both* the energy and the Lorentz-factor profile. Abe et al. (2026) Eq. 1, as extracted
    from the published PDF, instead uses exp(-theta^2/theta_core^2) (no factor of 2) for the
    energy profile, while Eq. 2 keeps the factor of 2 for Gamma0. This module follows the
    2026 (published, citable) convention by default. PDF text extraction can lose exponents,
    so verify this against the typeset paper before relying on it for anything beyond this
    exploratory notebook.

    Abe et al. also impose a cutoff Gamma0(theta_max) = 5 at theta_max = min(90 deg, ...).
    Their text frames it as limiting computation time, but it matters physically too (see
    `theta_max_deg` and `grb_model_at_theta`): Gamma0 is bounded below at 1 by construction
    (it cannot cross into the non-relativistic regime), so past the angle where Gamma0 ~ few,
    the relativistic deceleration formula below is being extrapolated outside where it means
    anything, and eps_k_fraction keeps falling while Gamma0^-8/3 stops growing -- t_dec then
    *decreases* again at large enough theta, which is not physical. `grb_model_at_theta` clips
    to theta_max for this reason.
    """
    theta_view = np.radians(np.asarray(theta_view_deg, dtype=float))
    theta_core = np.radians(np.asarray(theta_core_deg, dtype=float))
    eps_k_fraction = np.exp(-(theta_view ** 2) / (theta_core ** 2))
    gamma0_theta = (np.asarray(gamma_core, dtype=float) - 1.0) * \
        np.exp(-(theta_view ** 2) / (2.0 * theta_core ** 2)) + 1.0
    return eps_k_fraction, gamma0_theta


THETA_MAX_GAMMA0_FLOOR = 5.0   # Abe et al. 2026 Sec. 3.1: theta_max = min(90 deg, Gamma0(theta_max)=5)


def theta_max_deg(theta_core_deg, gamma_core, gamma0_floor: float = THETA_MAX_GAMMA0_FLOOR):
    """theta_max = min(90 deg, the angle where Gamma0(theta) = gamma0_floor) -- Abe et al.
    2026 Sec. 3.1. Solved from Eq. 2: (Gamma_core-1)*exp(-theta^2/2theta_core^2)+1 = floor.
    If Gamma_core <= gamma0_floor already at theta=0 (an unusually low draw), returns 0."""
    theta_core_deg = np.asarray(theta_core_deg, dtype=float)
    gamma_core = np.asarray(gamma_core, dtype=float)
    ratio = (gamma0_floor - 1.0) / (gamma_core - 1.0)
    with np.errstate(invalid="ignore"):
        theta_max = theta_core_deg * np.sqrt(-2.0 * np.log(ratio))
    theta_max = np.where(gamma_core > gamma0_floor, theta_max, 0.0)
    return np.minimum(90.0, np.nan_to_num(theta_max, nan=90.0))


def deceleration_time_s(ek_erg, gamma0, n_cm3: float = N_ISM_CM3):
    """t_dec, s: the deceleration radius (Abe et al. 2026 Eq. 6) converted to an
    observed/rest-frame time via the standard t_dec ~ R_dec / (2 c Gamma0^2)
    relation (e.g. Sari, Piran & Narayan 1998-type afterglow relations). The
    R-to-t prefactor is not spelled out in the extracted paper text, so it is this
    module's assumption -- see the module docstring."""
    ek_erg = np.asarray(ek_erg, dtype=float)
    gamma0 = np.asarray(gamma0, dtype=float)
    r_dec_cm = (17.0 * ek_erg / (16.0 * np.pi * _M_P_G * _C_CMS ** 2 * n_cm3 * gamma0 ** 2)) ** (1.0 / 3.0)
    return r_dec_cm / (2.0 * _C_CMS * gamma0 ** 2)


def x_ray_luminosity_11h_erg_s(e_gamma_iso_erg, rng: np.random.Generator):
    """L_X,11h, erg/s -- Berger (2014), via Nava Sec. 4 / Abe Eq. 5 (identical),
    with LX11H_SCATTER_DEX lognormal scatter about the best-fit relation."""
    e_gamma_iso_51 = np.asarray(e_gamma_iso_erg, dtype=float) / 1.0e51
    mean_log10_lx = np.log10(LX11H_NORM) + LX11H_SLOPE * np.log10(e_gamma_iso_51)
    scatter = rng.normal(0.0, LX11H_SCATTER_DEX, size=np.shape(e_gamma_iso_51))
    return 10.0 ** (mean_log10_lx + scatter)


def vhe_luminosity_11h_erg_s(lx_11h_erg_s, rng: np.random.Generator):
    """L_TeV,11h, erg/s (0.3-1 TeV per Abe Sec. 3.2) -- the L_TeV/L_X ratio of Nava
    Sec. 4 / Abe Sec. 3.2 (identical): lognormal, peaked at 1, scatter 0.3 dex."""
    lx_11h_erg_s = np.asarray(lx_11h_erg_s, dtype=float)
    scatter = rng.normal(0.0, LTEV_LX_RATIO_LOG10_SCATTER, size=np.shape(lx_11h_erg_s))
    return lx_11h_erg_s * 10.0 ** scatter


def l_peak_from_anchor(l_anchor_erg_s, t_anchor_s, t_dec_s, beta1, beta2):
    """L_peak such that the piecewise power law L(t) = L_peak*(t/t_dec)^beta1 (t<t_dec) or
    L_peak*(t/t_dec)^beta2 (t>=t_dec) passes through (t_anchor, l_anchor) exactly. This is
    how both sources tie the empirically-anchored 11h luminosity to a light curve whose peak
    *time* comes from the deceleration-time calculation above, without ever needing an
    independent absolute-normalisation blast-wave luminosity calculation."""
    t_anchor_s = np.asarray(t_anchor_s, dtype=float)
    t_dec_s = np.asarray(t_dec_s, dtype=float)
    beta = np.where(t_anchor_s >= t_dec_s, beta2, beta1)
    return np.asarray(l_anchor_erg_s, dtype=float) * (t_anchor_s / t_dec_s) ** (-beta)


# ===========================================================================
# Assembling a GRBModel
# ===========================================================================

_DEFAULT_ENERGY_GRID_GEV = np.geomspace(1.0, 1.0e4, 41)   # matches the catO5 grid (setup_model.md Sec. 2)
_DEFAULT_TIME_GRID_S = np.geomspace(1.0e-1, 1.0e7, 120)
_E_REF_GEV = 1.0   # matches the Option 2/3 convention already used in setup_model.ipynb


def _with_node(grid: np.ndarray, x: float) -> np.ndarray:
    """`grid` with `x` inserted if not already present (to float tolerance).

    `GRBModel` bakes L onto this grid once at construction and only ever interpolates that
    table afterwards (see `grb_model.LogLogInterpolator2D`); it never re-evaluates
    `temporal_fn` at an arbitrary query time. `temporal_fn` below has a *kink* at t_dec
    (the rise and decay slopes differ), so unless a node sits exactly on t_dec, every query
    near the peak -- including `.at()`, not just the grid-snapped `.peak()` -- interpolates
    across the kink and cuts the corner. Caught empirically in
    `notebooks/setup_model_phenomenological.ipynb` Sec. 2.1: without this, the "continuous"
    query at t_dec was low by a median 9% (max 17%) over 500 random draws, not the ~0 a
    log-log interpolant of a true power law should give.
    """
    if np.any(np.isclose(grid, x, rtol=1e-9)):
        return grid
    return np.sort(np.append(grid, x))


def _build_grb_model(t_dec_s: float, l_peak_erg_s: float, beta1: float, beta2: float,
                     photon_index: float, theta_deg: float, e_iso_meta: float,
                     energy_grid_GeV: np.ndarray, time_grid_s: np.ndarray) -> GRBModel:
    """L(E', t') = L0 * (E'/E_ref)^-photon_index * g(t'), with g the broken power
    law peaking (g=1) at t_dec, and L0 solved in closed form so that the
    energy-integrated light curve at t_dec equals l_peak_erg_s exactly (same
    trapz-in-ln-E convention as `grb_model._band_integral`/`.light_curve()`)."""
    time_grid_s = _with_node(np.asarray(time_grid_s, dtype=float), t_dec_s)

    def temporal_fn(t):
        t = np.asarray(t, dtype=float)
        return np.where(t <= t_dec_s, (t / t_dec_s) ** beta1, (t / t_dec_s) ** beta2)

    def spectral_fn(E):
        return np.asarray(E, dtype=float) ** (-photon_index)

    shape_E = spectral_fn(energy_grid_GeV) / spectral_fn(_E_REF_GEV)
    energy_integral_erg = float(
        _trapz(shape_E * energy_grid_GeV ** 2, np.log(energy_grid_GeV))
    ) * u.GeV.to(u.erg)
    l0 = l_peak_erg_s / energy_integral_erg   # ph/(s GeV) at E_ref, since temporal_fn(t_dec)=1

    model = GRBModel.from_analytic(
        spectral_fn, energy_grid_GeV, time_grid_s, temporal_fn=temporal_fn,
        theta_deg=theta_deg, L0=l0, E_ref=_E_REF_GEV,
    )
    model.E_iso = float(e_iso_meta)
    model.meta.update({
        "origin": "phenomenological", "t_dec_s": float(t_dec_s),
        "L_peak_erg_s": float(l_peak_erg_s), "beta1": float(beta1), "beta2": float(beta2),
        "photon_index": float(photon_index),
    })
    return model


def sample_events(rng: np.random.Generator, n: int, eiso_method: str = "via_epeak"
                  ) -> Dict[str, np.ndarray]:
    """Draw `n` independent core (on-axis) events: Nava (2020) Secs. 3-5 / Abe et al.
    (2026) Sec. 3.1-3.3. Returns a dict of length-n arrays (one entry per drawn
    quantity) -- convenient for population-level histograms and for indexing into
    with `grb_model_at_theta`.

    `eiso_method`: "via_epeak" (default) draws E_peak then applies the Amati relation
    (Abe's published two-step route); "direct" draws E_gamma,iso directly from Nava's
    already-propagated broken power law (her Eq. 1 shortcut). See the notebook for a
    comparison of the two.
    """
    theta_core_deg = draw_theta_core_deg(rng, n)
    gamma_core = draw_gamma_core(rng, n)

    if eiso_method == "via_epeak":
        e_peak_keV = draw_epeak_keV(rng, n)
        e_gamma_iso_erg = epeak_to_egiso_erg(e_peak_keV)
    elif eiso_method == "direct":
        e_peak_keV = np.full(n, np.nan)
        e_gamma_iso_erg = draw_egiso_direct_erg(rng, n)
    else:
        raise ValueError(f"unknown eiso_method {eiso_method!r}, expected 'via_epeak' or 'direct'")

    theta_core_rad = np.radians(theta_core_deg)
    e_gamma_erg = e_gamma_iso_erg * (1.0 - np.cos(theta_core_rad)) / 2.0        # collimation
    ek_core_erg = e_gamma_erg * (1.0 - ETA_GAMMA) / ETA_GAMMA                    # kinetic energy at the core
    eps_k_core_erg_sr = ek_core_erg / (np.pi * theta_core_rad ** 2)              # energy per solid angle

    lx_11h_erg_s = x_ray_luminosity_11h_erg_s(e_gamma_iso_erg, rng)
    ltev_11h_erg_s = vhe_luminosity_11h_erg_s(lx_11h_erg_s, rng)
    beta1, beta2 = draw_light_curve_shape(rng, n)
    photon_index = draw_photon_index(rng, n)

    t_dec_core_s = deceleration_time_s(ek_core_erg, gamma_core)   # jet_structure(0,...) gives Gamma0=gamma_core
    l_peak_core_erg_s = l_peak_from_anchor(ltev_11h_erg_s, T_ANCHOR_S, t_dec_core_s, beta1, beta2)

    return dict(
        theta_core_deg=theta_core_deg, gamma_core=gamma_core, e_peak_keV=e_peak_keV,
        e_gamma_iso_erg=e_gamma_iso_erg, e_gamma_erg=e_gamma_erg, ek_core_erg=ek_core_erg,
        eps_k_core_erg_sr=eps_k_core_erg_sr, lx_11h_erg_s=lx_11h_erg_s,
        ltev_11h_erg_s=ltev_11h_erg_s, beta1=beta1, beta2=beta2, photon_index=photon_index,
        t_dec_core_s=t_dec_core_s, l_peak_core_erg_s=l_peak_core_erg_s,
    )


def grb_model_at_theta(events: Dict[str, np.ndarray], i: int, theta_view_deg: float,
                       energy_grid_GeV: np.ndarray = None, time_grid_s: np.ndarray = None
                       ) -> GRBModel:
    """The phenomenological light curve of event `i` of `events` (from `sample_events`),
    as seen at `theta_view_deg`, as a `grb_model.GRBModel` (rest frame, distance-independent
    -- use `.to_observer()` to project it, exactly as for a catO5 file).

    theta_view_deg <~ theta_core reproduces the on-axis light curve of Nava Sec. 4 / Abe
    Sec. 3.2-3.3 essentially unchanged. Larger theta_view uses this module's off-axis
    approximation -- see the module docstring for what that does and does not capture.

    theta_view_deg is clipped to `theta_max_deg` (Abe et al. 2026 Sec. 3.1: where
    Gamma0(theta) = 5) with a warning, since past that angle Gamma0 is close enough to its
    floor of 1 that the relativistic deceleration formula is being extrapolated outside where
    it is meaningful -- seen directly during development as an unphysical late-time *decrease*
    of t_dec with theta past ~70-80 deg for typical draws.
    """
    if energy_grid_GeV is None:
        energy_grid_GeV = _DEFAULT_ENERGY_GRID_GEV
    if time_grid_s is None:
        time_grid_s = _DEFAULT_TIME_GRID_S

    theta_core = events["theta_core_deg"][i]
    gamma_core = events["gamma_core"][i]
    theta_max = theta_max_deg(theta_core, gamma_core)
    if theta_view_deg > theta_max:
        warnings.warn(
            f"grb_model_at_theta: theta_view={theta_view_deg:.2f} deg exceeds theta_max="
            f"{theta_max:.2f} deg (Gamma0={THETA_MAX_GAMMA0_FLOOR:.0f}) for this event; "
            "clipping to theta_max -- see grb_model_at_theta's docstring.", stacklevel=2,
        )
        theta_view_deg = float(theta_max)

    eps_frac, gamma0_theta = jet_structure(theta_view_deg, theta_core, gamma_core)
    ek_theta_erg = events["ek_core_erg"][i] * eps_frac
    t_dec = deceleration_time_s(ek_theta_erg, gamma0_theta)
    l_peak = events["l_peak_core_erg_s"][i] * eps_frac
    e_iso_theta = events["e_gamma_iso_erg"][i] * eps_frac

    return _build_grb_model(
        t_dec_s=float(t_dec), l_peak_erg_s=float(l_peak), beta1=float(events["beta1"][i]),
        beta2=float(events["beta2"][i]), photon_index=float(events["photon_index"][i]),
        theta_deg=float(theta_view_deg), e_iso_meta=float(e_iso_theta),
        energy_grid_GeV=energy_grid_GeV, time_grid_s=time_grid_s,
    )


# ===========================================================================
# A fixed (deterministic) jet, and the off-axis element integral
# ===========================================================================
#
# Everything above draws a fresh event per call. Below, one jet is a frozen `JetParams`:
# the same inputs always give the same L(E', t'; theta), so a simulation can tabulate it once
# on a theta grid (`model_set`) and reuse it every iteration. `BENCHMARK` is the population's
# central values; population scatter enters only if asked for, through `one_sigma_variants`
# (systematics) or `draw_jet_params` (a fixed, seeded bank of events).
#
# Off-axis, this replaces `grb_model_at_theta`'s local closure (clipped at theta_max) with the
# element integral Abe et al. 2026 Sec. 3.4 describe (after Lamb & Kobayashi 2017): each
# jet element emits its own on-axis light curve, Doppler-corrected to the observer's direction
# and summed at equal arrival time. That is what lets the core "come into view" late for an
# observer outside the jet, with no clipping.

@dataclass(frozen=True)
class JetParams:
    """Every input of the recipe for one jet. Frozen, so hashable and reproducible.

    ltev_11h_erg_s=None means the mean relations with no scatter: L_X,11h from Berger (2014)
    and L_TeV,11h / L_X,11h = 1. Vary one input with `dataclasses.replace`.
    """
    e_gamma_iso_erg: float
    theta_core_deg: float
    gamma_core: float
    beta1: float = BETA1_MEAN
    beta2: float = BETA2_MEAN
    photon_index: float = PHOTON_INDEX_MEAN
    ltev_11h_erg_s: Optional[float] = None
    n_cm3: float = N_ISM_CM3
    eta_gamma: float = ETA_GAMMA

    @property
    def l_tev_11h_erg_s(self) -> float:
        """On-axis L_TeV at t' = 11 h in VHE_BAND_GEV, erg/s."""
        if self.ltev_11h_erg_s is not None:
            return float(self.ltev_11h_erg_s)
        return float(LX11H_NORM * (self.e_gamma_iso_erg / 1.0e51) ** LX11H_SLOPE)

    @property
    def ek_iso_core_erg(self) -> float:
        """Isotropic-equivalent kinetic energy of the core, E_gamma,iso (1-eta)/eta.

        This (4 pi eps_k,core, not the collimated E_k,core) is what sets the local
        deceleration: before it spreads sideways, each part of the jet decelerates as if it
        were a spherical blast wave of its own isotropic-equivalent energy. `sample_events`
        passes the collimated E_k to `deceleration_time_s`, which gives a t_dec
        (2/(1-cos theta_core))^(1/3) ~ 5x earlier for theta_core = 14 deg.
        """
        return self.e_gamma_iso_erg * (1.0 - self.eta_gamma) / self.eta_gamma

    @property
    def t_dec_core_s(self) -> float:
        return float(deceleration_time_s(self.ek_iso_core_erg, self.gamma_core, self.n_cm3))


# The population's central values: the median of every marginal distribution above.
# E_gamma,iso is the median of the "via_epeak" route, 2.0e50 erg. That median moves with
# EPEAK_MIN_KEV (the log-mode of the distribution is ~2e51 erg). Once the model is rescaled to
# the trial luminosity L_k (Step 6), E_gamma,iso reaches the upper limit only through the time
# scale t ~ E^(1/3): one decade in E_gamma,iso is a factor 2.15 in time.
BENCHMARK = JetParams(
    e_gamma_iso_erg=2.0e50,
    theta_core_deg=10.0 ** LOG10_THETA_CORE_DEG_MEAN,
    gamma_core=10.0 ** LOG10_GAMMA_CORE_MEAN,
)

# 16th/84th percentiles of the "via_epeak" E_gamma,iso distribution (400k draws).
_E_GAMMA_ISO_16_84_ERG = (3.5e48, 2.1e51)

VHE_BAND_GEV = (300.0, 1000.0)   # Abe et al. 2026 Sec. 3.3: L_TeV,11h is the 0.3-1 TeV luminosity
DEFAULT_THETA_GRID_DEG = np.arange(0.0, 90.0 + 0.5, 1.0)


def draw_jet_params(rng: np.random.Generator, n: int, eiso_method: str = "via_epeak"
                    ) -> List[JetParams]:
    """`n` jets drawn from the population (`sample_events`), scatter terms included."""
    ev = sample_events(rng, n, eiso_method=eiso_method)
    return [
        JetParams(
            e_gamma_iso_erg=float(ev["e_gamma_iso_erg"][i]),
            theta_core_deg=float(ev["theta_core_deg"][i]),
            gamma_core=float(ev["gamma_core"][i]),
            beta1=float(ev["beta1"][i]), beta2=float(ev["beta2"][i]),
            photon_index=float(ev["photon_index"][i]),
            ltev_11h_erg_s=float(ev["ltev_11h_erg_s"][i]),
        )
        for i in range(n)
    ]


def one_sigma_variants(params: JetParams = BENCHMARK) -> Dict[str, Tuple[JetParams, JetParams]]:
    """{name: (low, high)}: `params` with one input moved to -1 sigma / +1 sigma of its
    population distribution, the others fixed. Only the inputs that change the *shape* of
    L(E', t'; theta) are listed; the L_X and L_TeV/L_X scatter only rescale it, which the
    Step 6 normalisation to L_k removes. beta1 (sigma 0.05) is left out as negligible."""
    th, g = params.theta_core_deg, params.gamma_core
    return {
        "theta_core": (replace(params, theta_core_deg=th * 10 ** -LOG10_THETA_CORE_DEG_STD),
                       replace(params, theta_core_deg=th * 10 ** LOG10_THETA_CORE_DEG_STD)),
        "gamma_core": (replace(params, gamma_core=g * 10 ** -LOG10_GAMMA_CORE_STD),
                       replace(params, gamma_core=g * 10 ** LOG10_GAMMA_CORE_STD)),
        "e_gamma_iso": (replace(params, e_gamma_iso_erg=_E_GAMMA_ISO_16_84_ERG[0]),
                        replace(params, e_gamma_iso_erg=_E_GAMMA_ISO_16_84_ERG[1])),
        "beta2": (replace(params, beta2=params.beta2 - BETA2_STD),
                  replace(params, beta2=params.beta2 + BETA2_STD)),
        "photon_index": (replace(params, photon_index=params.photon_index - PHOTON_INDEX_STD),
                         replace(params, photon_index=params.photon_index + PHOTON_INDEX_STD)),
    }


# --- the element integral ---------------------------------------------------------------
#
# Coordinates are centred on the line of sight: alpha is an element's angle from it, psi its
# azimuth around it (0 = towards the jet axis; psi and -psi are mirror images). Log spacing in
# alpha resolves the beaming cone, alpha ~ 1/Gamma, at every Gamma. The jet structure is
# evaluated at each element's own angle theta from the jet axis. Elements past theta_max
# (Gamma_0 < 5, Abe et al. 2026 Sec. 3.1) are left out, as in the paper.

_N_ALPHA = 240
_N_PSI = 128
_ALPHA_MIN_RAD = 1e-5
_T_ON_S = np.geomspace(1e-4, 1e9, 13 * 24 + 1)    # element on-axis time grid, 24 per decade
_DEFAULT_TIME_GRID_OUT_S = np.geomspace(1e-1, 1e7, 8 * 20 + 1)
_CHUNK = 2048


def _one_minus_beta(gamma):
    """(1 - beta, beta), with 1 - beta computed without cancellation at large Gamma."""
    gamma = np.asarray(gamma, dtype=float)
    beta = np.sqrt(1.0 - 1.0 / gamma ** 2)
    return 1.0 / (gamma ** 2 * (1.0 + beta)), beta


def _doppler_ratio(one_minus_beta, beta, alpha):
    """a = delta(alpha)/delta(0) = (1-beta)/(1-beta cos alpha): the Doppler factor towards an
    observer at alpha from the element's velocity, relative to one along it."""
    return one_minus_beta / (one_minus_beta + beta * 2.0 * np.sin(0.5 * alpha) ** 2)


def _alpha_grid(n_alpha):
    """Cell centres and ring solid angles (full 2 pi in azimuth) of the alpha grid; the
    rings tile the sphere exactly (sum = 4 pi)."""
    edges = np.concatenate([[0.0], np.geomspace(_ALPHA_MIN_RAD, np.pi, n_alpha)])
    lo, hi = edges[:-1], edges[1:]
    centre = np.sqrt(np.maximum(lo, 0.5 * hi[0]) * hi)
    ring = 4.0 * np.pi * np.sin(0.5 * (lo + hi)) * np.sin(0.5 * (hi - lo))
    return centre, ring


def _beaming_table(alpha, ring, q):
    """(Gamma grid, sum over the alpha grid of a^q dOmega): the solid angle a uniform
    spherical shell of Lorentz factor Gamma beams into, on the same grid as the integral.

    Dividing each element by this makes a uniform shell reproduce its on-axis luminosity
    exactly, for any observer, grid and Gamma (so the grid's own discretisation error
    cancels). Analytically it is 2pi/(beta(q-1)) [(1-beta) - (1-beta)^q (1+beta)^(1-q)],
    which tends to 4 pi as Gamma -> 1 and to pi/((q-1) Gamma^2) at Gamma >> 1."""
    gam = np.geomspace(1.0, 1.0e5, 600)
    omb, beta = _one_minus_beta(gam)
    a = _doppler_ratio(omb[:, None], beta[:, None], alpha[None, :])
    return gam, np.sum(a ** q * ring[None, :], axis=1)


def _interp_rows(x, y, x_new):
    """Row-wise linear interpolation of y(x) at the shared points x_new, each row of x
    strictly increasing. Beyond a row's ends it extrapolates its edge segment: exact for
    the element integral's early end, where Gamma is still constant and every element's
    light curve is a pure power law, t^beta1, in both on-axis and observer time."""
    n_rows, n = x.shape
    span = float(max(x.max(), x_new.max()) - min(x.min(), x_new.min())) + 1.0
    offset = span * np.arange(n_rows)[:, None]
    j = np.searchsorted((x + offset).ravel(), (x_new[None, :] + offset).ravel())
    j = np.clip(j.reshape(n_rows, -1) - n * np.arange(n_rows)[:, None], 1, n - 1)
    x0, x1 = np.take_along_axis(x, j - 1, 1), np.take_along_axis(x, j, 1)
    y0, y1 = np.take_along_axis(y, j - 1, 1), np.take_along_axis(y, j, 1)
    return y0 + (y1 - y0) * (x_new[None, :] - x0) / (x1 - x0)


def offaxis_luminosity(params: JetParams, theta_view_deg: float, time_s: np.ndarray,
                       n_alpha: int = _N_ALPHA, n_psi: int = _N_PSI,
                       t_on_s: np.ndarray = _T_ON_S) -> np.ndarray:
    """Band luminosity seen at `theta_view_deg`, at rest-frame times `time_s`, in units of
    the core element's own on-axis luminosity at t' = 11 h.

    Abe et al. 2026 Sec. 3.4 (after Lamb & Kobayashi 2017). Each element, at angle theta
    from the jet axis, carries the recipe's on-axis light curve for its own energy and
    Lorentz factor (Eqs. 1-2): the post-peak decay L ~ E (t'/11 h)^beta2, the same at fixed
    time for every element (blast-wave scaling, grb_model.EISO_L_EXPONENT = 1), and the
    beta1 rise before its own t_dec. Its Lorentz factor is Gamma_0 until t_dec and
    Gamma_0 (t/t_dec)^(-3/8) after (Blandford & McKee 1976, Abe Eq. 7). For an observer at
    alpha from the element, with a = delta(alpha)/delta(0):

      * luminosity at fixed photon energy x a^(2+p) (a power law dN/dE ~ E^-p stays one
        under the Doppler shift, so the spectrum never changes shape off-axis),
      * observer time dt_obs = dt_on / a, integrated since Gamma, and so a, evolve,
      * weight dOmega / (the solid angle a uniform shell beams into, `_beaming_table`).

    The sum over elements at equal observer time is the light curve. Even on-axis it is
    the input light curve smoothed by the arrival-time spread across the beaming cone, which
    rescales the power-law segments (see `model_at_theta`, which normalises that away).
    """
    q = 2.0 + params.photon_index
    time_s = np.asarray(time_s, dtype=float)
    theta_v = np.radians(float(theta_view_deg))
    if theta_v == 0.0:
        n_psi = 1   # every element at a given alpha has theta = alpha

    alpha, ring = _alpha_grid(n_alpha)
    psi = (np.arange(n_psi) + 0.5) * np.pi / n_psi
    gam_tab, beam_tab = _beaming_table(alpha, ring, q)

    cos_theta = (np.cos(alpha)[:, None] * np.cos(theta_v)
                 + np.sin(alpha)[:, None] * np.sin(theta_v) * np.cos(psi)[None, :])
    theta_el_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    alpha_el = np.broadcast_to(alpha[:, None], theta_el_deg.shape)
    d_omega = np.broadcast_to(ring[:, None] / n_psi, theta_el_deg.shape)

    keep = theta_el_deg <= theta_max_deg(params.theta_core_deg, params.gamma_core)
    theta_el_deg, alpha_el, d_omega = theta_el_deg[keep], alpha_el[keep], d_omega[keep]
    eps_frac, gamma0 = jet_structure(theta_el_deg, params.theta_core_deg, params.gamma_core)
    t_dec = deceleration_time_s(params.ek_iso_core_erg * eps_frac, gamma0, params.n_cm3)

    log_t_out = np.log(time_s)
    log_t_on = np.log(t_on_s)
    dt_on = np.diff(t_on_s)
    log_gam_tab, log_beam_tab = np.log(gam_tab), np.log(beam_tab)
    total = np.zeros(time_s.size)
    for s in range(0, theta_el_deg.size, _CHUNK):   # in log space: no pow() per cell
        sl = slice(s, s + _CHUNK)
        log_x = log_t_on[None, :] - np.log(t_dec[sl, None])
        decay = log_x >= 0.0
        log_h = (np.log(eps_frac[sl, None]) + params.beta2 * np.log(t_dec[sl, None] / T_ANCHOR_S)
                 + np.where(decay, params.beta2, params.beta1) * log_x)
        gamma = np.maximum(1.0, gamma0[sl, None] * np.exp(np.where(decay, -0.375 * log_x, 0.0)))
        omb, beta = _one_minus_beta(gamma)
        log_a = np.log(omb) - np.log(omb + beta * 2.0 * np.sin(0.5 * alpha_el[sl, None]) ** 2)
        log_lum = (log_h + q * log_a + np.log(d_omega[sl, None])
                   - np.interp(np.log(gamma), log_gam_tab, log_beam_tab))

        inv_a = np.exp(-log_a)
        t_obs = np.empty_like(inv_a)
        t_obs[:, 0] = t_on_s[0] * inv_a[:, 0]
        t_obs[:, 1:] = t_obs[:, :1] + np.cumsum(0.5 * (inv_a[:, 1:] + inv_a[:, :-1]) * dt_on, axis=1)

        total += np.sum(np.exp(_interp_rows(np.log(t_obs), log_lum, log_t_out)), axis=0)
    return total


@lru_cache(maxsize=256)
def _on_axis_anchor(params: JetParams, n_alpha: int) -> float:
    """offaxis_luminosity at theta_view = 0, t' = 11 h: what `model_at_theta` divides by."""
    return float(offaxis_luminosity(params, 0.0, np.array([T_ANCHOR_S]), n_alpha=n_alpha)[0])


def model_at_theta(params: JetParams, theta_view_deg: float,
                   energy_grid_GeV: Optional[np.ndarray] = None,
                   time_grid_s: Optional[np.ndarray] = None,
                   n_alpha: int = _N_ALPHA, n_psi: int = _N_PSI) -> GRBModel:
    """L(E', t'; theta_view) of the jet `params`, as a `grb_model.GRBModel` (rest frame,
    distance- and EBL-free). Deterministic: the same inputs always give the same model.

    L = N(t') (E'/E_ref)^-p, with N(t') from `offaxis_luminosity`, normalised once per jet so
    that the on-axis (theta_view = 0) VHE_BAND_GEV luminosity at t' = 11 h is exactly
    params.l_tev_11h_erg_s, the recipe's anchor. The same constant applies at every angle,
    so the ratio between angles is the element integral's.
    """
    energy = _DEFAULT_ENERGY_GRID_GEV if energy_grid_GeV is None else np.asarray(energy_grid_GeV, float)
    time = _DEFAULT_TIME_GRID_OUT_S if time_grid_s is None else np.asarray(time_grid_s, float)
    p = params.photon_index

    rel = offaxis_luminosity(params, theta_view_deg, time, n_alpha=n_alpha, n_psi=n_psi)
    band_lum = rel * params.l_tev_11h_erg_s / _on_axis_anchor(params, n_alpha)   # erg/s

    e_lo, e_hi = VHE_BAND_GEV
    band_integral = (_E_REF_GEV ** 2 * np.log(e_hi / e_lo) if np.isclose(p, 2.0) else
                     _E_REF_GEV ** p * (e_hi ** (2.0 - p) - e_lo ** (2.0 - p)) / (2.0 - p))
    n_ref = band_lum / (band_integral * u.GeV.to(u.erg))   # ph/(s GeV) at E_ref

    L = n_ref[None, :] * (energy[:, None] / _E_REF_GEV) ** (-p)
    return GRBModel(
        theta_deg=float(theta_view_deg), energy=energy, time=time, L=L,
        E_iso=float(params.e_gamma_iso_erg),
        meta={"origin": "phenomenological_fixed", "params": asdict(params),
              "E_ref_GeV": _E_REF_GEV, "vhe_band_GeV": list(VHE_BAND_GEV),
              "n_alpha": n_alpha, "n_psi": n_psi},
    )


def model_set(params: JetParams = BENCHMARK, thetas_deg: np.ndarray = DEFAULT_THETA_GRID_DEG,
              **kwargs) -> GRBModelSet:
    """`model_at_theta` on a grid of viewing angles, as a `grb_model.GRBModelSet`: the same
    object (and `.save`/`.load` cache format) setup_model.ipynb builds from the catO5 files,
    so sim_3d consumes either one identically."""
    return GRBModelSet(models={float(th): model_at_theta(params, th, **kwargs) for th in thetas_deg})
