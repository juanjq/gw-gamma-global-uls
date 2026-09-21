"""
simulate.py -- GW follow-up simulations, flux (2D) and luminosity (3D) upper limits.

Overview
--------
Two complementary Monte-Carlo methods share the same test statistic

    Lambda = max over the GW 95% credible region of   TS' = TS + 2 ln p_GW ,

and the same "inject -> fake -> measure Lambda" machinery:

  * 2D (flux):        fix a PWL amplitude phi0 and, per realisation, sample only a
                      sky position inside the 95% GW region -> Lambda distribution.
  * 3D (luminosity):  fix a band luminosity L0 and, per realisation, sample a sky
                      position from the FULL GW map AND a luminosity distance from
                      that pixel's marginal distance CDF, convert (L0, d) -> phi0,
                      then inject -> Lambda distribution marginalised over sky x distance.

In both cases the upper limit is the value of the injected quantity for which a
fraction `cl` of the simulated Lambda values exceeds the observed Lambda (or the
background median Lambda, when the observation under-fluctuates). The crossing is
found by bisection in log space (`run_iterative_ul`, `run_iterative_ul_3d`), both
of which now share a single, instrumented bisection engine (`_bisect_ul`).

Flux <-> luminosity
-------------------
All flux/luminosity conversions are *band-integrated over the assumed power law*
(Section 1). Nothing here is bolometric, and nothing is evaluated "at 1 TeV" as a
stand-in for an integral. For dN/dE = phi0 (E/E0)^-Gamma,

    photon flux   F_ph = phi0 E0 * I1,   I1 = [x^(1-G)]_x1^x2 / (1-G)   (ln x for G=1)
    energy flux   F_E  = phi0 E0^2 * I2, I2 = [x^(2-G)]_x1^x2 / (2-G)   (ln x for G=2)
    luminosity    L    = 4 pi d_L^2 F_E * (1+z)^(Gamma-2)

with x = E/E0. Note that Gamma = 2 -- the index used throughout this analysis -- is
exactly the singular case of I2, so it *must* be handled with the logarithm; a naive
1/(2-Gamma) implementation returns inf/nan there. Gamma = 2 is also exactly the index
for which the power-law K-correction (1+z)^(Gamma-2) is unity, so band luminosities
are redshift-independent at this index (a useful sanity anchor, verified in
`check_spectral_conversions`).

Required contents of the input .pkl
-----------------------------------
2D and 3D:  dataset, excess_estimator, prob_gw, mask_threshold_95,
            containment_factor, energy_edges
2D only:    lvk_prob_hp
3D only:    r_grid (n_r,) [Mpc], cdf_table (ny, nx, n_r), cdf_valid (ny, nx) bool
optional:   ts_estimator, e_min, e_max, source_name, type_obs
"""

from __future__ import annotations

import os
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

try:
    from . import paths
except ImportError:                                     # run as a standalone script
    import paths

# gammapy is required for the simulation machinery, but *not* for Section 1, so
# that the spectral conversions can be imported and unit-tested standalone.
try:
    import healpy as hp
    from gammapy.stats.fit_statistics import cash
    from gammapy.estimators.map.excess import (
        convolved_map_dataset_counts_statistics,
        _get_convolved_maps,
    )
    from gammapy.modeling.models import (
        PointSpatialModel,
        PowerLawSpectralModel,
        ConstantTemporalModel,
        SkyModel,
        Models,
    )
    _HAS_GAMMAPY = True
except ImportError:                                     # pragma: no cover
    _HAS_GAMMAPY = False

try:
    from utils import IndexToDeclRa
except ImportError:                                     # pragma: no cover
    IndexToDeclRa = None
try:
    from plotting import plot_ul_iteration
except ImportError:                                     # pragma: no cover
    plot_ul_iteration = None


__all__ = [
    # Section 1 -- spectral conversions
    "E_REF", "pwl_photon_flux", "pwl_energy_flux", "pwl_e2dnde",
    "amplitude_from_photon_flux", "amplitude_from_energy_flux",
    "luminosity_from_energy_flux", "energy_flux_from_luminosity",
    "amplitude_from_luminosity", "luminosity_from_amplitude",
    "redshift_from_luminosity_distance", "FluxPoint", "flux_point_from_amplitude",
    "check_spectral_conversions",
    # Section 2 -- input handling
    "load_simulation_input", "validate_simulation_input",
    # Sections 4-5 -- simulations
    "perform_n_simulations", "perform_n_simulations_3d",
    "sample_sky_and_distance", "check_3d_sampling",
    # Section 6-7 -- upper limits and reporting
    "run_iterative_ul", "run_iterative_ul_3d",
    "luminosity_ul_from_flux_ul", "summarize_upper_limits", "compare_2d_3d",
]


# ===========================================================================
# 1. Spectral model: band-integrated flux <-> luminosity conversions
# ===========================================================================
#
# Sign convention: `index` (Gamma) is the POSITIVE photon index of
#     dN/dE = phi0 (E / E0)^(-Gamma),
# so the "PWL index -2" spectrum of this analysis is index = 2.0.
# `amplitude` (phi0) is always in cm-2 s-1 TeV-1 at E0 = `e_ref`.

E_REF = 1.0 * u.TeV
AMPLITUDE_UNIT = u.Unit("cm-2 s-1 TeV-1")
_SINGULAR_TOL = 1e-8          # |1-G| or |2-G| below this -> use the log branch


def _as_energy(value, default_unit=u.TeV):
    """Coerce a scalar/Quantity/sequence into a TeV Quantity."""
    q = u.Quantity(value)
    if q.unit.is_unity():
        q = q * default_unit
    return q.to(u.TeV)


def _band(energy_edges):
    """Unpack an energy band into (e_min, e_max) TeV Quantities, validated."""
    edges = _as_energy(np.atleast_1d(u.Quantity(energy_edges)))
    if edges.size < 2:
        raise ValueError("`energy_edges` must contain at least two energies.")
    e_lo, e_hi = edges[0], edges[-1]
    if not np.isfinite(e_lo.value) or not np.isfinite(e_hi.value):
        raise ValueError(f"Non-finite energy edges: {edges}")
    if e_lo.value <= 0:
        raise ValueError(f"Energy edges must be > 0, got e_min = {e_lo}")
    if e_hi <= e_lo:
        raise ValueError(f"Need e_max > e_min, got [{e_lo}, {e_hi}]")
    return e_lo, e_hi


def _power_integral(x_lo, x_hi, exponent):
    """
    Integral of x^exponent dx from x_lo to x_hi, with the exponent = -1
    (logarithmic) branch handled explicitly.

    This is the one place where the Gamma = 2 energy-flux singularity is dealt
    with; every conversion below routes through it.
    """
    p = exponent + 1.0
    if abs(p) < _SINGULAR_TOL:
        return np.log(x_hi / x_lo)
    return (x_hi ** p - x_lo ** p) / p


def pwl_photon_flux(amplitude, energy_edges, index=2.0, e_ref=E_REF):
    """
    Band-integrated PHOTON flux, int_{E1}^{E2} dN/dE dE, in cm-2 s-1.

    amplitude : float or Quantity -- phi0, floats are read as cm-2 s-1 TeV-1.
    """
    amp = u.Quantity(amplitude)
    if amp.unit.is_unity():
        amp = amp * AMPLITUDE_UNIT
    e_lo, e_hi = _band(energy_edges)
    e0 = _as_energy(e_ref)
    x_lo, x_hi = (e_lo / e0).to_value(""), (e_hi / e0).to_value("")
    integral = _power_integral(x_lo, x_hi, -float(index))
    return (amp * e0 * integral).to(u.Unit("cm-2 s-1"))


def pwl_energy_flux(amplitude, energy_edges, index=2.0, e_ref=E_REF):
    """
    Band-integrated ENERGY flux, int_{E1}^{E2} E dN/dE dE, in erg cm-2 s-1.

    For index == 2 this reduces to phi0 * E0^2 * ln(E2/E1); the generic
    1/(2-Gamma) expression is singular there and is never used.
    """
    amp = u.Quantity(amplitude)
    if amp.unit.is_unity():
        amp = amp * AMPLITUDE_UNIT
    e_lo, e_hi = _band(energy_edges)
    e0 = _as_energy(e_ref)
    x_lo, x_hi = (e_lo / e0).to_value(""), (e_hi / e0).to_value("")
    integral = _power_integral(x_lo, x_hi, 1.0 - float(index))
    return (amp * e0 ** 2 * integral).to(u.Unit("erg cm-2 s-1"))


def pwl_e2dnde(amplitude, energy, index=2.0, e_ref=E_REF):
    """Differential SED point E^2 dN/dE at `energy`, in erg cm-2 s-1."""
    amp = u.Quantity(amplitude)
    if amp.unit.is_unity():
        amp = amp * AMPLITUDE_UNIT
    e = _as_energy(energy)
    e0 = _as_energy(e_ref)
    return (amp * e ** 2 * (e / e0) ** (-float(index))).to(u.Unit("erg cm-2 s-1"))


def amplitude_from_photon_flux(photon_flux, energy_edges, index=2.0, e_ref=E_REF):
    """Inverse of `pwl_photon_flux`: returns phi0 in cm-2 s-1 TeV-1 (float)."""
    ref = pwl_photon_flux(1.0, energy_edges, index=index, e_ref=e_ref)
    target = u.Quantity(photon_flux)
    if target.unit.is_unity():
        target = target * u.Unit("cm-2 s-1")
    return float((target / ref).to_value(""))


def amplitude_from_energy_flux(energy_flux, energy_edges, index=2.0, e_ref=E_REF):
    """Inverse of `pwl_energy_flux`: returns phi0 in cm-2 s-1 TeV-1 (float)."""
    ref = pwl_energy_flux(1.0, energy_edges, index=index, e_ref=e_ref)
    target = u.Quantity(energy_flux)
    if target.unit.is_unity():
        target = target * u.Unit("erg cm-2 s-1")
    return float((target / ref).to_value(""))


# --- distance / redshift ---------------------------------------------------

_Z_INTERP = None


def redshift_from_luminosity_distance(distance, cosmology=None):
    """
    Redshift for a luminosity distance, via a cached monotone interpolation
    (fast and vectorised, unlike a per-element `z_at_value` root find).

    Returns a bare float / ndarray. Falls back to the low-z limit z = H0 d / c
    if astropy.cosmology is unavailable.
    """
    global _Z_INTERP
    d = u.Quantity(distance)
    if d.unit.is_unity():
        d = d * u.Mpc
    d_mpc = np.atleast_1d(d.to_value(u.Mpc)).astype(float)

    if cosmology is None:
        try:
            from astropy.cosmology import Planck18 as cosmology
        except ImportError:                              # pragma: no cover
            from astropy.constants import c as _c
            z = (70.0 * d_mpc) / _c.to_value(u.km / u.s)
            return z if np.ndim(distance) else float(z[0])

    if _Z_INTERP is None or _Z_INTERP[2] is not cosmology:
        z_tab = np.concatenate([[0.0], np.logspace(-6, np.log10(5.0), 600)])
        d_tab = cosmology.luminosity_distance(z_tab).to_value(u.Mpc)
        _Z_INTERP = (d_tab, z_tab, cosmology)
    d_tab, z_tab, _ = _Z_INTERP

    z = np.interp(d_mpc, d_tab, z_tab)
    return z if np.ndim(distance) else float(z[0])


def _k_correction(index, redshift):
    """
    Power-law K-correction relating the observed-band energy flux to the
    luminosity in the *same* band in the source rest frame:

        L_[E1,E2],rest = 4 pi d_L^2 * F_E,[E1,E2],obs * (1+z)^(Gamma-2).

    Identically 1 for Gamma = 2, which is the index used here.
    """
    if redshift is None:
        return 1.0
    return (1.0 + np.asarray(redshift, dtype=float)) ** (float(index) - 2.0)


def luminosity_from_energy_flux(energy_flux, distance, index=2.0, redshift=None,
                                apply_k_correction=True):
    """
    Isotropic-equivalent band luminosity [erg/s] from a band energy flux.

        L = 4 pi d_L^2 F_E (1+z)^(Gamma-2)

    distance : Quantity or float (floats read as Mpc).
    redshift : None (no K-correction), "auto" (derive z from d_L), or a value.
    """
    f_e = u.Quantity(energy_flux)
    if f_e.unit.is_unity():
        f_e = f_e * u.Unit("erg cm-2 s-1")
    d = u.Quantity(distance)
    if d.unit.is_unity():
        d = d * u.Mpc

    if not apply_k_correction:
        z = None
    elif isinstance(redshift, str) and redshift == "auto":
        z = redshift_from_luminosity_distance(d)
    else:
        z = redshift

    lum = (4.0 * np.pi * d.to(u.cm) ** 2 * f_e) * _k_correction(index, z)
    return lum.to(u.erg / u.s)


def energy_flux_from_luminosity(luminosity, distance, index=2.0, redshift=None,
                                apply_k_correction=True):
    """Exact inverse of `luminosity_from_energy_flux`; returns erg cm-2 s-1."""
    lum = u.Quantity(luminosity)
    if lum.unit.is_unity():
        lum = lum * u.erg / u.s
    d = u.Quantity(distance)
    if d.unit.is_unity():
        d = d * u.Mpc

    if not apply_k_correction:
        z = None
    elif isinstance(redshift, str) and redshift == "auto":
        z = redshift_from_luminosity_distance(d)
    else:
        z = redshift

    f_e = lum / (4.0 * np.pi * d.to(u.cm) ** 2 * _k_correction(index, z))
    return f_e.to(u.Unit("erg cm-2 s-1"))


def amplitude_from_luminosity(luminosity, distance, energy_edges, index=2.0,
                              e_ref=E_REF, redshift=None, apply_k_correction=True):
    """
    L0 [erg/s] + d_L -> PWL amplitude phi0 [cm-2 s-1 TeV-1 at e_ref].

    L0 is the isotropic-equivalent luminosity *in the band `energy_edges`*
    (source rest frame), NOT a bolometric luminosity. The conversion is the
    exact inverse of the band integral of the assumed power law:

        F_E = L0 / (4 pi d_L^2 (1+z)^(Gamma-2))
        phi0 = F_E / (E0^2 * int_{x1}^{x2} x^(1-Gamma) dx)

    Vectorised over `distance` (and over `luminosity`, if broadcastable).

    Note on the original implementation: it computed the same ratio via
    `PowerLawSpectralModel(amplitude=1).energy_flux(...)`, which is numerically
    equivalent for Gamma != 2 but was documented as a *bolometric* luminosity.
    That framing is what is fixed here -- the number now means "luminosity in
    the analysis band", which is what the injected spectrum can actually
    constrain, and the singular Gamma = 2 branch is explicit.
    """
    f_e = energy_flux_from_luminosity(
        luminosity, distance, index=index, redshift=redshift,
        apply_k_correction=apply_k_correction,
    )
    ref = pwl_energy_flux(1.0, energy_edges, index=index, e_ref=e_ref)
    amp = (f_e / ref).to_value("")
    return float(amp) if np.ndim(amp) == 0 else np.asarray(amp, dtype=float)


def luminosity_from_amplitude(amplitude, distance, energy_edges, index=2.0,
                              e_ref=E_REF, redshift=None, apply_k_correction=True):
    """Exact inverse of `amplitude_from_luminosity`; returns a Quantity [erg/s]."""
    f_e = pwl_energy_flux(amplitude, energy_edges, index=index, e_ref=e_ref)
    return luminosity_from_energy_flux(
        f_e, distance, index=index, redshift=redshift,
        apply_k_correction=apply_k_correction,
    )


@dataclass
class FluxPoint:
    """Every flux-like quantity implied by one PWL amplitude, in one place."""
    amplitude: float                 # phi0 [cm-2 s-1 TeV-1] at e_ref
    index: float
    e_ref: u.Quantity
    e_min: u.Quantity
    e_max: u.Quantity
    photon_flux: u.Quantity          # cm-2 s-1
    energy_flux: u.Quantity          # erg cm-2 s-1
    e_dec: u.Quantity                # sqrt(e_min e_max), the log-centre
    e2dnde: u.Quantity               # erg cm-2 s-1 at e_dec

    def as_dict(self):
        return {
            "amplitude": self.amplitude, "index": self.index,
            "e_ref": self.e_ref, "e_min": self.e_min, "e_max": self.e_max,
            "photon_flux": self.photon_flux, "energy_flux": self.energy_flux,
            "e_dec": self.e_dec, "e2dnde": self.e2dnde,
        }

    def __str__(self):
        return (
            f"phi0        = {self.amplitude:.3e} cm-2 s-1 TeV-1 @ {self.e_ref:.2f}\n"
            f"photon flux = {self.photon_flux.value:.3e} cm-2 s-1        "
            f"[{self.e_min:.2f}-{self.e_max:.2f}]\n"
            f"energy flux = {self.energy_flux.value:.3e} erg cm-2 s-1    "
            f"[{self.e_min:.2f}-{self.e_max:.2f}]\n"
            f"E2 dN/dE    = {self.e2dnde.value:.3e} erg cm-2 s-1 @ {self.e_dec:.2f}"
        )


def flux_point_from_amplitude(amplitude, energy_edges, index=2.0, e_ref=E_REF):
    """Bundle amplitude -> (photon flux, energy flux, SED point) consistently."""
    e_lo, e_hi = _band(energy_edges)
    e_dec = np.sqrt(e_lo * e_hi)
    return FluxPoint(
        amplitude=float(amplitude), index=float(index), e_ref=_as_energy(e_ref),
        e_min=e_lo, e_max=e_hi,
        photon_flux=pwl_photon_flux(amplitude, (e_lo, e_hi), index, e_ref),
        energy_flux=pwl_energy_flux(amplitude, (e_lo, e_hi), index, e_ref),
        e_dec=e_dec, e2dnde=pwl_e2dnde(amplitude, e_dec, index, e_ref),
    )


def check_spectral_conversions(energy_edges=(0.6 * u.TeV, 20 * u.TeV),
                               indices=(1.0, 1.5, 2.0, 2.5, 3.0),
                               amplitude=1e-12, distance=400 * u.Mpc,
                               rtol=1e-9, verbose=True):
    """
    CROSSCHECK -- validates Section 1 against independent implementations.

    Five independent checks, each of which has caught a real class of bug:
      (a) analytic band integrals vs. brute-force numerical quadrature,
      (b) analytic band integrals vs. gammapy's PowerLawSpectralModel (skipped
          if gammapy is absent),
      (c) amplitude -> L -> amplitude round trip (exactness of the inverse),
      (d) the Gamma = 2 logarithmic branch against a Gamma = 2 +/- eps limit,
          i.e. that the singularity is removable and correctly removed,
      (e) K-correction is identically 1 at Gamma = 2, and L scales as d^2.

    Returns True if everything passes; raises AssertionError otherwise.
    """
    from scipy.integrate import quad

    e_lo, e_hi = _band(energy_edges)
    e0 = E_REF
    ok = True

    def _report(name, a, b, tol=rtol):
        nonlocal ok
        rel = abs(a - b) / max(abs(b), 1e-300)
        good = rel < tol
        ok = ok and good
        if verbose:
            print(f"    {'PASS' if good else 'FAIL'}  {name:<46s} "
                  f"{a:.8e} vs {b:.8e}  (rel {rel:.2e})")
        return good

    if verbose:
        print("=" * 78)
        print(f"Spectral conversion crosschecks  |  band "
              f"[{e_lo:.3f}, {e_hi:.3f}]  |  phi0 = {amplitude:.2e}")
        print("=" * 78)

    for g in indices:
        if verbose:
            print(f"\n  Gamma = {g}")

        # (a) numerical quadrature, in TeV
        dnde = lambda e: amplitude * (e / e0.value) ** (-g)
        _qkw = dict(limit=400, epsabs=0.0, epsrel=1e-13)
        num_ph = quad(dnde, e_lo.value, e_hi.value, **_qkw)[0]
        num_en = quad(lambda e: e * dnde(e), e_lo.value, e_hi.value, **_qkw)[0]
        num_en_erg = (num_en * u.TeV).to_value(u.erg)

        ana_ph = pwl_photon_flux(amplitude, (e_lo, e_hi), g).value
        ana_en = pwl_energy_flux(amplitude, (e_lo, e_hi), g).value
        _report("photon flux: analytic vs quad", ana_ph, num_ph, 1e-9)
        _report("energy flux: analytic vs quad", ana_en, num_en_erg, 1e-9)

        # (b) gammapy reference
        if _HAS_GAMMAPY:
            pwl = PowerLawSpectralModel(
                amplitude=amplitude * AMPLITUDE_UNIT, index=g, reference=e0)
            gp_ph = pwl.integral(e_lo, e_hi).to_value("cm-2 s-1")
            gp_en = pwl.energy_flux(e_lo, e_hi).to_value("erg cm-2 s-1")
            _report("photon flux: analytic vs gammapy", ana_ph, gp_ph, 1e-5)
            _report("energy flux: analytic vs gammapy", ana_en, gp_en, 1e-5)
        elif verbose:
            print("    SKIP  gammapy not importable -- library crosscheck skipped")

        # (c) amplitude -> L -> amplitude round trip
        lum = luminosity_from_amplitude(amplitude, distance, (e_lo, e_hi), index=g)
        amp_back = amplitude_from_luminosity(lum, distance, (e_lo, e_hi), index=g)
        _report("round trip phi0 -> L -> phi0", amp_back, amplitude)

        # also through the photon-flux inverse
        amp_ph = amplitude_from_photon_flux(ana_ph, (e_lo, e_hi), g)
        _report("round trip phi0 -> F_ph -> phi0", amp_ph, amplitude)

    # (d) removable singularity at Gamma = 2
    eps = 1e-6
    f2 = pwl_energy_flux(amplitude, (e_lo, e_hi), 2.0).value
    fm = pwl_energy_flux(amplitude, (e_lo, e_hi), 2.0 - eps).value
    fp = pwl_energy_flux(amplitude, (e_lo, e_hi), 2.0 + eps).value
    if verbose:
        print("\n  Gamma = 2 singular branch")
    _report("log branch == limit from both sides", f2, 0.5 * (fm + fp), 1e-8)
    closed = amplitude * (e0.value ** 2) * np.log((e_hi / e_lo).to_value(""))
    _report("log branch == phi0 E0^2 ln(E2/E1)",
            f2, (closed * u.TeV).to_value(u.erg), 1e-10)

    # (e) K-correction and distance scaling
    if verbose:
        print("\n  Luminosity conventions")
    z_test = 0.09
    l_nok = luminosity_from_amplitude(amplitude, distance, (e_lo, e_hi),
                                      index=2.0, apply_k_correction=False).value
    l_k = luminosity_from_amplitude(amplitude, distance, (e_lo, e_hi),
                                    index=2.0, redshift=z_test).value
    _report("K-correction is unity at Gamma = 2", l_k, l_nok, 1e-12)

    l_g25_nok = luminosity_from_amplitude(amplitude, distance, (e_lo, e_hi),
                                          index=2.5, apply_k_correction=False).value
    l_g25_k = luminosity_from_amplitude(amplitude, distance, (e_lo, e_hi),
                                        index=2.5, redshift=z_test).value
    _report("K-correction = (1+z)^(G-2) at Gamma = 2.5",
            l_g25_k / l_g25_nok, (1 + z_test) ** 0.5, 1e-10)

    l_1 = luminosity_from_amplitude(amplitude, 100 * u.Mpc, (e_lo, e_hi)).value
    l_2 = luminosity_from_amplitude(amplitude, 200 * u.Mpc, (e_lo, e_hi)).value
    _report("L scales as d^2", l_2 / l_1, 4.0, 1e-10)

    z_auto = redshift_from_luminosity_distance(distance)
    if verbose:
        print(f"    INFO  z(d_L = {distance}) = {z_auto:.4f} "
              f"(Planck18); at Gamma = 2 this does not affect L")
        print("\n" + "=" * 78)
        print(f"Spectral crosschecks: {'ALL PASSED' if ok else 'FAILURES PRESENT'}")
        print("=" * 78)

    if not ok:
        raise AssertionError("check_spectral_conversions: at least one check failed.")
    return True


# ===========================================================================
# 2. Input handling and validation
# ===========================================================================

_KEYS_COMMON = ("dataset", "excess_estimator", "prob_gw",
                "mask_threshold_95", "containment_factor")
_KEYS_2D = ("lvk_prob_hp",)
_KEYS_3D = ("r_grid", "cdf_table", "cdf_valid", "energy_edges")

_PROB_FLOOR = 1e-20      # guards 2 ln p_GW against log(0)


def validate_simulation_input(data, mode="2d", verbose=True):
    """
    Fail fast, and loudly, on a malformed input .pkl.

    Every check here corresponds to a failure mode that is otherwise silent:
    a missing key surfaces thousands of iterations later, a transposed
    cdf_table produces plausible-but-wrong distances, an all-False mask makes
    `np.nanargmax` raise on an all-NaN slice, a zero containment factor turns
    every flux UL into inf, and prob_gw containing exact zeros makes
    2 ln p_GW = -inf, which propagates into Lambda as nan.
    """
    mode = mode.lower()
    required = _KEYS_COMMON + (_KEYS_3D if mode == "3d" else _KEYS_2D)
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"Input .pkl is missing required key(s) for mode '{mode}': {missing}. "
            f"Present keys: {sorted(data.keys())}"
        )

    prob_gw = np.asarray(data["prob_gw"], dtype=float)
    mask = np.asarray(data["mask_threshold_95"], dtype=bool)
    cf = float(data["containment_factor"])

    if prob_gw.ndim != 2:
        raise ValueError(f"prob_gw must be 2D (ny, nx), got shape {prob_gw.shape}")
    if mask.shape != prob_gw.shape:
        raise ValueError(f"mask_threshold_95 {mask.shape} != prob_gw {prob_gw.shape}")
    if not np.isfinite(prob_gw).all():
        raise ValueError("prob_gw contains non-finite values.")
    if not mask.any():
        raise ValueError("mask_threshold_95 selects zero pixels -- Lambda is undefined.")
    if not (0.0 < cf <= 1.5):
        raise ValueError(f"containment_factor = {cf} is outside a sane range (0, 1.5].")

    geom_shape = data["dataset"].geoms["geom"].to_image().data_shape
    if tuple(geom_shape) != prob_gw.shape:
        raise ValueError(
            f"prob_gw shape {prob_gw.shape} does not match the dataset image geometry "
            f"{tuple(geom_shape)} -- the GW map and the WCS grid are out of sync."
        )

    n_zero = int((prob_gw <= 0).sum())
    info = {
        "mode": mode,
        "shape": prob_gw.shape,
        "n_mask": int(mask.sum()),
        "frac_mask": float(mask.mean()),
        "prob_in_mask": float(prob_gw[mask].sum()),
        "prob_total": float(prob_gw.sum()),
        "containment_factor": cf,
        "n_prob_zero": n_zero,
    }

    if mode == "3d":
        r_grid = np.asarray(data["r_grid"], dtype=float)
        cdf = np.asarray(data["cdf_table"], dtype=float)
        valid = np.asarray(data["cdf_valid"], dtype=bool)

        if cdf.shape[:2] != prob_gw.shape:
            raise ValueError(
                f"cdf_table {cdf.shape} is not (ny, nx, n_r) = "
                f"({prob_gw.shape[0]}, {prob_gw.shape[1]}, n_r) -- likely transposed."
            )
        if cdf.shape[2] != r_grid.size:
            raise ValueError(
                f"cdf_table last axis ({cdf.shape[2]}) != r_grid size ({r_grid.size}).")
        if valid.shape != prob_gw.shape:
            raise ValueError(f"cdf_valid {valid.shape} != prob_gw {prob_gw.shape}")
        if not valid.any():
            raise ValueError("cdf_valid is all False -- no bin has a usable distance CDF.")
        if np.any(np.diff(r_grid) <= 0):
            raise ValueError("r_grid must be strictly increasing.")

        cdf_v = cdf[valid]
        d_mono = np.diff(cdf_v, axis=-1)
        n_nonmono = int((d_mono < -1e-9).sum())
        end_err = float(np.abs(cdf_v[:, -1] - 1.0).max())
        start_err = float(np.abs(cdf_v[:, 0]).max())
        if n_nonmono and verbose:
            print(f"  WARNING: {n_nonmono} non-monotonic steps in cdf_table "
                  f"(valid bins) -- inverse-CDF sampling assumes monotonicity.")
        if end_err > 1e-3 and verbose:
            print(f"  WARNING: cdf_table does not reach 1 in every valid bin "
                  f"(max |CDF(r_max) - 1| = {end_err:.2e}); r_max = {r_grid[-1]:.0f} "
                  f"Mpc truncates the posterior, and the missing mass piles up as a "
                  f"probability atom at r_max when sampling.")
        if start_err > 1e-3 and verbose:
            print(f"  WARNING: cdf_table does not start at 0 in every valid bin "
                  f"(max CDF(r_min) = {start_err:.2e}); r_min = {r_grid[0]:.0f} Mpc "
                  f"truncates the low-distance tail, producing an atom at r_min. "
                  f"Since flux scales as 1/d^2 this biases the nearest -- i.e. "
                  f"brightest -- realisations.")

        # Probability mass reachable by the 3D sampler (sky prob on valid bins)
        p_valid = float(prob_gw[valid].sum() / max(prob_gw.sum(), 1e-300))
        if p_valid < 0.9 and verbose:
            print(f"  WARNING: bins with a valid distance CDF carry only "
                  f"{p_valid*100:.1f}% of the in-FoV GW probability -- the 3D "
                  f"sampler cannot reach the rest.")
        info.update({
            "n_r": int(r_grid.size), "r_min": float(r_grid[0]), "r_max": float(r_grid[-1]),
            "n_cdf_valid": int(valid.sum()), "frac_prob_on_valid": p_valid,
            "cdf_end_err": end_err, "cdf_start_err": start_err,
        })

    if verbose:
        print(f"  Input OK [{mode}]: grid {info['shape']},  "
              f"mask {info['n_mask']} px ({info['frac_mask']*100:.1f}%),  "
              f"GW prob in mask {info['prob_in_mask']*100:.1f}% of FoV total "
              f"{info['prob_total']*100:.1f}%,  containment {cf:.3f}")
        if mode == "3d":
            print(f"             distance grid {info['n_r']} pts "
                  f"[{info['r_min']:.0f}, {info['r_max']:.0f}] Mpc,  "
                  f"{info['n_cdf_valid']} valid bins carrying "
                  f"{info['frac_prob_on_valid']*100:.1f}% of the sky probability")
        if n_zero:
            print(f"             note: {n_zero} pixels have prob_gw <= 0; "
                  f"they are floored at {_PROB_FLOOR:g} for 2 ln p_GW")

    return info


def load_simulation_input(file_input, mode="2d", verbose=True):
    """Load + validate the input .pkl. Returns (data, info)."""
    if not os.path.exists(file_input):
        raise FileNotFoundError(f"Input .pkl not found: {file_input}")
    with open(file_input, "rb") as f:
        data = pickle.load(f)
    info = validate_simulation_input(data, mode=mode, verbose=verbose)
    return data, info


# ===========================================================================
# 3. Shared TS machinery
# ===========================================================================

class _TSEngine:
    """
    Everything that is constant across Monte-Carlo iterations, computed once.

    Previously the kernel, the default mask and the 2 ln p_GW term were
    recomputed (or at least re-derived) inside both `perform_n_simulations`
    and `perform_n_simulations_3d`, in duplicated code that had already drifted
    apart -- the 2D version did not floor prob_gw before taking its log, the 3D
    version did. Centralising this guarantees the two methods compute
    *identically defined* Lambdas, which is the precondition for comparing
    their upper limits at all.
    """

    def __init__(self, dataset, excess_estimator, prob_gw, mask_threshold,
                 containment_factor):
        self.dataset = dataset
        self.excess_estimator = excess_estimator
        self.mask = np.asarray(mask_threshold, dtype=bool)
        self.containment_factor = float(containment_factor)

        self.log_prob_gw = 2.0 * np.log(
            np.clip(np.asarray(prob_gw, dtype=float), _PROB_FLOOR, None))

        self.correlate_off = excess_estimator.correlate_off
        self.kernel = excess_estimator.estimate_kernel(dataset)
        self.mask_default = excess_estimator.estimate_mask_default(dataset)

        geom_centers = dataset.geoms["geom"].get_coord(mode="center")
        self.bin_c_ra = geom_centers.lon[0].value
        self.bin_c_dec = geom_centers.lat[0].value

    def counts_statistics(self):
        return convolved_map_dataset_counts_statistics(
            convolved_maps=_get_convolved_maps(
                dataset=self.dataset, kernel=self.kernel,
                mask=self.mask_default, correlate_off=self.correlate_off,
            ),
            stat_type="cash",
        )

    def ts_maps(self, stats):
        """Signed cash TS map and the GW-weighted TS' = TS + 2 ln p_GW."""
        n_on = stats.n_on.sum(axis=0)
        mu_bkg = stats.mu_bkg.sum(axis=0)

        lik_alt = cash(n_on, n_on)
        lik_null = cash(n_on, mu_bkg)
        ts_sign = np.where((n_on - mu_bkg) >= 0.0, +1.0, -1.0)
        ts = np.where((lik_null - lik_alt) < 0.0, 0.0, lik_null - lik_alt) * ts_sign
        return ts, ts + self.log_prob_gw

    def lambda_from(self, ts, ts2):
        """Masked argmax of TS and TS', with coordinates."""
        ts_masked = np.where(self.mask, ts, np.nan)
        ts2_masked = np.where(self.mask, ts2, np.nan)
        i_ts = np.unravel_index(np.nanargmax(ts_masked), ts_masked.shape)
        i_l = np.unravel_index(np.nanargmax(ts2_masked), ts2_masked.shape)
        return {
            "tsmax": float(ts_masked[i_ts]),
            "tsmax_ra": float(self.bin_c_ra[i_ts]),
            "tsmax_dec": float(self.bin_c_dec[i_ts]),
            "lambda_data": float(ts2_masked[i_l]),
            "lambda_ra": float(self.bin_c_ra[i_l]),
            "lambda_dec": float(self.bin_c_dec[i_l]),
        }

    def flux_ul_map(self):
        """Containment-corrected flux-UL map from the ExcessMapEstimator."""
        maps = self.excess_estimator.run(self.dataset)
        ul = np.asarray(maps["flux_ul"].data[0], dtype=float) / self.containment_factor
        ul_masked = np.where(self.mask, ul, np.nan)
        i = np.unravel_index(np.nanargmax(ul_masked), ul_masked.shape)
        return ul, {
            "ulmax": float(ul_masked[i]),
            "ulmax_ra": float(self.bin_c_ra[i]),
            "ulmax_dec": float(self.bin_c_dec[i]),
        }


def _make_source_model(coord, amplitude, index, e_ref=E_REF, name="model-simulated"):
    return Models([SkyModel(
        spatial_model=PointSpatialModel.from_position(coord),
        spectral_model=PowerLawSpectralModel(
            index=index,
            amplitude=float(amplitude) * AMPLITUDE_UNIT,
            reference=_as_energy(e_ref),
        ),
        temporal_model=ConstantTemporalModel(),
        name=name,
    )])


def _pack_results(results, extra, file_output):
    """Write the .npz, creating the parent directory if needed."""
    parent = os.path.dirname(os.path.abspath(file_output))
    os.makedirs(parent, exist_ok=True)
    out = {}
    for key, value in results.items():
        if not len(value):
            continue
        try:
            out[key] = np.asarray(value)
        except (ValueError, TypeError):
            out[key] = np.asarray(value, dtype=object)
    out.update(extra)
    np.savez(file_output, **out)
    size_mb = os.path.getsize(file_output) / 1024 ** 2
    print(f"\nWriting file --> {file_output}  ({size_mb:.1f} MB)")


# ===========================================================================
# 4. 2D simulation: fixed flux, sampled sky position
# ===========================================================================

def _map_healpix_to_wcs_bins(lvk_prob_hp, bin_c_ra, bin_c_dec, mask_threshold,
                             verbose=True):
    """
    Nearest-WCS-bin lookup for every HEALPix pixel, used to restrict the GW
    sampling distribution to the 95% mask.

    Returns (prob_masked, diagnostics). The RA convention
    `-(((ra + 180) % 360) - 180)` is the one used by the surrounding pipeline
    and is preserved here; the diagnostics include the worst nearest-neighbour
    angular mismatch so a convention error shows up as a large separation
    rather than a silently wrong mask.
    """
    if IndexToDeclRa is None:
        raise ImportError("utils.IndexToDeclRa is required for the 2D sampler.")

    n_pix = len(lvk_prob_hp)
    pix_indices, nside = np.arange(n_pix), hp.npix2nside(n_pix)
    dec_hp, ra_hp = IndexToDeclRa(pix_indices, nside)
    ra_hp_conv = -(((ra_hp + 180) % 360) - 180)

    ra_axis, dec_axis = bin_c_ra[0, :], bin_c_dec[:, 0]
    ra_sorted_idx, dec_sorted_idx = np.argsort(ra_axis), np.argsort(dec_axis)
    ra_sorted, dec_sorted = ra_axis[ra_sorted_idx], dec_axis[dec_sorted_idx]

    def nearest_idx(sorted_arr, values, sorted_idx):
        pos = np.clip(np.searchsorted(sorted_arr, values), 0, len(sorted_arr) - 1)
        pos_left = np.clip(pos - 1, 0, len(sorted_arr) - 1)
        closer_left = (np.abs(values - sorted_arr[pos_left])
                       < np.abs(values - sorted_arr[pos]))
        pos = np.where(closer_left, pos_left, pos)
        return sorted_idx[pos]

    ix = nearest_idx(ra_sorted, ra_hp_conv, ra_sorted_idx)
    iy = nearest_idx(dec_sorted, dec_hp, dec_sorted_idx)
    in_mask = np.asarray(mask_threshold, dtype=bool)[iy, ix]

    prob_masked = np.asarray(lvk_prob_hp, dtype=float).copy()
    prob_masked[~in_mask] = 0.0
    frac_in_mask = prob_masked.sum()
    if frac_in_mask <= 0:
        raise ValueError(
            "No HEALPix probability falls inside mask_threshold_95 -- check the "
            "RA convention in `_map_healpix_to_wcs_bins` and the mask itself.")
    prob_masked /= frac_in_mask

    # CROSSCHECK: how far is each *selected* HEALPix pixel from the WCS bin it
    # was assigned to? Should be <~ one bin diagonal; anything larger means the
    # lookup, not the astronomy, is at fault.
    sel = prob_masked > 0
    sep = SkyCoord(ra=ra_hp_conv[sel] * u.deg, dec=dec_hp[sel] * u.deg).separation(
        SkyCoord(ra=bin_c_ra[iy[sel], ix[sel]] * u.deg,
                 dec=bin_c_dec[iy[sel], ix[sel]] * u.deg)).deg
    binsz = float(np.abs(np.diff(ra_axis)).mean())
    diag = binsz * np.sqrt(2.0)

    diagnostics = {
        "frac_gw_in_mask": float(frac_in_mask),
        "n_hp_in_mask": int(sel.sum()),
        "max_assign_sep_deg": float(sep.max()) if sep.size else 0.0,
        "bin_diagonal_deg": float(diag),
    }
    if verbose:
        print(f"  Mask covers {frac_in_mask*100:.2f}% of the all-sky GW probability "
              f"({sel.sum()} HEALPix pixels)")
        if sep.size and sep.max() > 1.5 * diag:
            print(f"  WARNING: worst HEALPix->WCS assignment is "
                  f"{sep.max():.3f} deg vs a bin diagonal of {diag:.3f} deg -- "
                  f"suspect an RA-convention mismatch in the nearest-bin lookup.")
    return prob_masked, ra_hp_conv, dec_hp, diagnostics


def perform_n_simulations(
    n_sim, amplitude, file_input, file_output, compute_uls=0,
    spectral_index=2.0, e_ref=E_REF, seed=None, position_seed=12345,
    store_ts_maps=True, store_stats=True, verbose=True,
):
    """
    2D method: inject a fixed PWL amplitude at a sky position drawn from the
    masked (95%) GW map, `n_sim` times, and record the Lambda distribution.

    Parameters
    ----------
    n_sim : int
    amplitude : float
        phi0 in cm-2 s-1 TeV-1 at `e_ref`. 0 gives the background-only run.
    file_input, file_output : str
    compute_uls : int/bool
        Also run the (slow) ExcessMapEstimator to get flux-UL maps per iteration.
    spectral_index, e_ref : float, Quantity
        Injected spectrum; must match what the estimators assume.
    seed : int or None
        Base seed for `dataset.fake`. `None` reproduces the historical
        behaviour of seeding realisation i with i, which is *deliberate*: it
        gives common random numbers across amplitudes, so the fraction
        Lambda > target varies smoothly (and near-monotonically) with
        amplitude instead of jittering by Monte-Carlo noise between bisection
        steps. Pass an int to shift the whole block of seeds.
    position_seed : int or None
        Seed for the sky-position draw. Fixed by default so that every
        amplitude tested during a bisection sees the *same* set of injected
        positions -- previously these came from the unseeded global NumPy
        stream, which broke common random numbers and made the bisection
        noisier than it needed to be. Pass None for independent positions.
    store_ts_maps, store_stats : bool
        Keep full per-iteration TS maps / counts-statistics objects. Each is
        O(n_sim x n_pix); turn them off for large runs (the bisection does).
    """
    n_sim = int(n_sim)
    compute_uls = bool(compute_uls)
    if n_sim <= 0:
        raise ValueError(f"n_sim must be positive, got {n_sim}")

    data, _ = load_simulation_input(file_input, mode="2d", verbose=verbose)
    engine = _TSEngine(data["dataset"], data["excess_estimator"], data["prob_gw"],
                       data["mask_threshold_95"], data["containment_factor"])

    print(f"Producing {n_sim} 2D simulations for phi0 = {amplitude:.3e} "
          f"cm-2 s-1 TeV-1 (index {spectral_index})...")

    prob_masked, ra_hp_conv, dec_hp, diag = _map_healpix_to_wcs_bins(
        data["lvk_prob_hp"], engine.bin_c_ra, engine.bin_c_dec,
        engine.mask, verbose=verbose)

    rng = np.random.default_rng(position_seed)
    indices_hp = rng.choice(prob_masked.size, size=n_sim, p=prob_masked)
    ra_sim, dec_sim = ra_hp_conv[indices_hp], dec_hp[indices_hp]
    sim_coords = SkyCoord(ra=ra_sim * u.deg, dec=dec_sim * u.deg, frame="icrs")

    # Report the injected flux in physical units, not just the amplitude.
    if amplitude > 0 and "energy_edges" in data:
        print("  Injected spectrum:")
        for line in str(flux_point_from_amplitude(
                amplitude, data["energy_edges"], spectral_index, e_ref)).splitlines():
            print(f"    {line}")

    results = {k: [] for k in ("lambda_data", "lambda_ra", "lambda_dec", "tsmax",
                               "tsmax_ra", "tsmax_dec")}
    if store_ts_maps:
        results.update({"ts_dist": [], "ts2_dist": []})
    if store_stats:
        results["stats"] = []
    if compute_uls:
        results.update({"ulmax": [], "ulmax_ra": [], "ulmax_dec": [], "ul_dist": []})

    base_seed = 0 if seed is None else int(seed)

    for i in range(n_sim):
        print(f"Computing... {i+1}/{n_sim}", end="\r")

        engine.dataset.models = _make_source_model(
            sim_coords[i], amplitude, spectral_index, e_ref)
        engine.dataset.fake(base_seed + i)
        engine.dataset.models = None          # do not treat the signal as known bkg

        stats = engine.counts_statistics()
        ts, ts2 = engine.ts_maps(stats)
        for key, value in engine.lambda_from(ts, ts2).items():
            results[key].append(value)
        if store_ts_maps:
            results["ts_dist"].append(ts)
            results["ts2_dist"].append(ts2)
        if store_stats:
            results["stats"].append(stats)
        if compute_uls:
            ul_map, ul_info = engine.flux_ul_map()
            for key, value in ul_info.items():
                results[key].append(value)
            results["ul_dist"].append(ul_map)

    _pack_results(results, {
        "f_ra": np.asarray(ra_sim), "f_dec": np.asarray(dec_sim),
        "amplitude": np.float64(amplitude), "spectral_index": np.float64(spectral_index),
        "frac_gw_in_mask": np.float64(diag["frac_gw_in_mask"]),
    }, file_output)


# ===========================================================================
# 5. 3D simulation: fixed band luminosity, sampled sky position and distance
# ===========================================================================

def _sample_bin_indices(prob_gw_2d, cdf_valid, n_sim, rng):
    """Draw n_sim WCS bin indices from the FULL GW map, restricted to bins
    that have a usable distance CDF (and renormalised over those)."""
    weights = np.where(cdf_valid, prob_gw_2d, 0.0).astype(float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("No WCS bin has both GW probability > 0 and a valid distance CDF.")
    flat_idx = rng.choice(weights.size, size=n_sim, p=weights.ravel() / total)
    return np.unravel_index(flat_idx, prob_gw_2d.shape)


def _sample_distances_batch(r_grid, cdf_rows, rng):
    """
    Vectorised inverse-CDF sampling, one distance per row of `cdf_rows`.
    Linear interpolation inside the bracketing grid interval; rows with a flat
    CDF there fall back to the lower edge rather than dividing by zero.
    """
    n_sim, n_r = cdf_rows.shape
    x = rng.uniform(0.0, 1.0, size=n_sim)

    idx = np.clip(np.sum(cdf_rows < x[:, None], axis=1), 1, n_r - 1)
    rows = np.arange(n_sim)
    c_lo, c_hi = cdf_rows[rows, idx - 1], cdf_rows[rows, idx]
    r_lo, r_hi = r_grid[idx - 1], r_grid[idx]

    denom = np.where((c_hi - c_lo) > 0, c_hi - c_lo, 1.0)
    frac = np.clip((x - c_lo) / denom, 0.0, 1.0)
    return r_lo + frac * (r_hi - r_lo)


def sample_sky_and_distance(prob_gw, cdf_valid, cdf_table, r_grid,
                            bin_c_ra, bin_c_dec, n_sim, rng, sky_mask=None):
    """
    One place that defines the 3D draw: bin ~ prob_gw (restricted to valid
    bins, and optionally to `sky_mask`), then d ~ p(d | bin). Exposed publicly
    so the notebook can validate the *actual* sampler rather than a
    re-implementation of it.

    sky_mask : optional boolean map. Pass `mask_threshold_95` to give the 3D
        method the SAME sky support as the 2D method; leave as None to sample
        the full GW map. This choice changes what the luminosity limit means,
        so it is explicit rather than implicit.
    """
    support = np.asarray(cdf_valid, dtype=bool)
    if sky_mask is not None:
        support = support & np.asarray(sky_mask, dtype=bool)
    iy, ix = _sample_bin_indices(prob_gw, support, n_sim, rng)
    d_sim = _sample_distances_batch(r_grid, cdf_table[iy, ix], rng)
    return {
        "iy": iy, "ix": ix, "d_sim": d_sim,
        "ra": bin_c_ra[iy, ix], "dec": bin_c_dec[iy, ix],
    }


def perform_n_simulations_3d(
    n_sim, luminosity, file_input, file_output, compute_uls=0,
    spectral_index=2.0, e_ref=E_REF, seed=None, sampling_seed=12345,
    luminosity_band=None, apply_k_correction=False, restrict_to_mask=False,
    store_ts_maps=False, store_stats=False, verbose=True,
):
    """
    3D method: fix an isotropic-equivalent BAND luminosity L0 [erg/s] and, per
    realisation, draw (sky bin, distance) from the GW posterior, convert to a
    PWL amplitude, inject, and record Lambda.

    luminosity : float
        L0 in erg/s, defined over `luminosity_band` (default: the analysis
        band `energy_edges` from the .pkl). This is NOT bolometric -- see the
        module docstring and `amplitude_from_luminosity`.
    luminosity_band : (e_min, e_max) or None
        Band in which L0 is defined. Defaults to the analysis band, which makes
        the 3D limit directly comparable to the 2D one. Set it to a fixed
        reference band (e.g. 0.1-100 TeV) if you need to quote a limit
        comparable with the literature.
    apply_k_correction : bool
        Include the (1+z)^(Gamma-2) K-correction, with z inferred from d_L.
        Exactly 1 for Gamma = 2, so it is off by default and only matters if
        you change the index.
    restrict_to_mask : bool
        Sample sky positions only inside `mask_threshold_95`, i.e. give the 3D
        method the same sky support as the 2D one. Off by default (the full GW
        map is the point of the 3D method), but turning it on is the cleanest
        way to check that the two limits agree once sky support is matched:
        any remaining difference is then purely the distance marginalisation.
    sampling_seed : int or None
        Seed for the (position, distance) draws. Fixed by default so that all
        L0 values tested during a bisection use the *same* sky/distance
        realisations (common random numbers), which is what makes the fraction
        curve smooth in L0.
    """
    n_sim = int(n_sim)
    compute_uls = bool(compute_uls)
    if n_sim <= 0:
        raise ValueError(f"n_sim must be positive, got {n_sim}")
    if luminosity <= 0:
        raise ValueError(f"luminosity must be > 0, got {luminosity}")

    data, _ = load_simulation_input(file_input, mode="3d", verbose=verbose)
    engine = _TSEngine(data["dataset"], data["excess_estimator"], data["prob_gw"],
                       data["mask_threshold_95"], data["containment_factor"])

    r_grid = np.asarray(data["r_grid"], dtype=float)
    cdf_table = np.asarray(data["cdf_table"], dtype=float)
    cdf_valid = np.asarray(data["cdf_valid"], dtype=bool)
    band = data["energy_edges"] if luminosity_band is None else luminosity_band

    print(f"Producing {n_sim} 3D simulations for L0 = {luminosity:.3e} erg/s "
          f"in [{_band(band)[0]:.2f}, {_band(band)[1]:.2f}] (index {spectral_index})...")

    rng = np.random.default_rng(sampling_seed)
    draw = sample_sky_and_distance(
        np.asarray(data["prob_gw"], dtype=float), cdf_valid, cdf_table, r_grid,
        engine.bin_c_ra, engine.bin_c_dec, n_sim, rng,
        sky_mask=engine.mask if restrict_to_mask else None)
    if verbose and restrict_to_mask:
        print("  Sky support restricted to mask_threshold_95 (matches the 2D method)")

    d_sim = draw["d_sim"]
    if np.any(d_sim <= 0):
        n_bad = int((d_sim <= 0).sum())
        raise ValueError(
            f"{n_bad} sampled distances are <= 0 -- r_grid starts at "
            f"{r_grid[0]:g} Mpc, so the CDF has support at d = 0 and the "
            f"1/d^2 conversion diverges. Start r_grid above 0.")

    z_sim = ("auto" if apply_k_correction else None)
    amp_sim = amplitude_from_luminosity(
        luminosity, d_sim * u.Mpc, band, index=spectral_index, e_ref=e_ref,
        redshift=z_sim, apply_k_correction=apply_k_correction)
    amp_sim = np.atleast_1d(amp_sim).astype(float)

    if verbose:
        print(f"  Sampled d: median {np.median(d_sim):.0f} Mpc, "
              f"90% CI [{np.percentile(d_sim, 5):.0f}, {np.percentile(d_sim, 95):.0f}] Mpc")
        print(f"  Implied phi0: median {np.median(amp_sim):.3e}, "
              f"90% CI [{np.percentile(amp_sim, 5):.3e}, "
              f"{np.percentile(amp_sim, 95):.3e}] cm-2 s-1 TeV-1")
        # CROSSCHECK: invert a few draws and confirm we recover L0 exactly.
        k = min(64, n_sim)
        back = luminosity_from_amplitude(
            amp_sim[:k], d_sim[:k] * u.Mpc, band, index=spectral_index, e_ref=e_ref,
            redshift=z_sim, apply_k_correction=apply_k_correction).to_value(u.erg / u.s)
        worst = float(np.max(np.abs(back / luminosity - 1.0)))
        print(f"  L -> phi0 -> L round trip on {k} draws: max rel. error {worst:.2e}")
        if worst > 1e-9:
            print("  WARNING: luminosity round trip is not exact -- the L <-> phi0 "
                  "conversion is inconsistent with itself.")

    sim_coords = SkyCoord(ra=draw["ra"] * u.deg, dec=draw["dec"] * u.deg, frame="icrs")

    results = {k: [] for k in ("lambda_data", "lambda_ra", "lambda_dec", "tsmax",
                               "tsmax_ra", "tsmax_dec")}
    if store_ts_maps:
        results.update({"ts_dist": [], "ts2_dist": []})
    if store_stats:
        results["stats"] = []
    if compute_uls:
        results.update({"ulmax": [], "ulmax_ra": [], "ulmax_dec": [], "ul_dist": []})

    base_seed = 0 if seed is None else int(seed)

    for i in range(n_sim):
        print(f"Computing... {i+1}/{n_sim}", end="\r")

        engine.dataset.models = _make_source_model(
            sim_coords[i], amp_sim[i], spectral_index, e_ref, name="model-simulated-3d")
        engine.dataset.fake(base_seed + i)
        engine.dataset.models = None

        stats = engine.counts_statistics()
        ts, ts2 = engine.ts_maps(stats)
        for key, value in engine.lambda_from(ts, ts2).items():
            results[key].append(value)
        if store_ts_maps:
            results["ts_dist"].append(ts)
            results["ts2_dist"].append(ts2)
        if store_stats:
            results["stats"].append(stats)
        if compute_uls:
            ul_map, ul_info = engine.flux_ul_map()
            for key, value in ul_info.items():
                results[key].append(value)
            results["ul_dist"].append(ul_map)

    _pack_results(results, {
        "f_ra": np.asarray(draw["ra"]), "f_dec": np.asarray(draw["dec"]),
        "iy": np.asarray(draw["iy"]), "ix": np.asarray(draw["ix"]),
        "d_sim": np.asarray(d_sim), "amp_sim": amp_sim,
        "luminosity": np.float64(luminosity),
        "spectral_index": np.float64(spectral_index),
        "restrict_to_mask": np.bool_(restrict_to_mask),
    }, file_output)


def check_3d_sampling(file_input, n_sim=20000, seed=7, restrict_to_mask=False,
                      verbose=True, make_plot=True):
    """
    CROSSCHECK -- validates the 3D (sky x distance) sampler against the inputs
    it is supposed to reproduce. This calls the *production* sampler
    (`sample_sky_and_distance`), so it tests the code that actually runs.

    Checks:
      1. Sky marginal: sampled per-bin frequencies vs. the renormalised
         prob_gw, via a chi-square on the well-populated bins.
      2. Distance marginal: sampled distances vs. the analytic mixture PDF
         sum_bins w_bin p(d | bin), via a two-sample-style KS statistic against
         the analytic CDF.
      3. Per-bin conditional: for the highest-probability bin, sampled
         distances vs. that bin's own CDF.
      4. Support: no sampled distance at or below 0 (which would blow up 1/d^2),
         and the sampled range sits inside the r_grid.

    Returns a dict of statistics; also prints a verdict and (optionally) draws
    a 3-panel diagnostic figure.
    """
    from scipy.stats import chisquare

    data, _ = load_simulation_input(file_input, mode="3d", verbose=verbose)
    prob_gw = np.asarray(data["prob_gw"], dtype=float)
    cdf_table = np.asarray(data["cdf_table"], dtype=float)
    cdf_valid = np.asarray(data["cdf_valid"], dtype=bool)
    r_grid = np.asarray(data["r_grid"], dtype=float)

    geom_centers = data["dataset"].geoms["geom"].get_coord(mode="center")
    bin_c_ra, bin_c_dec = geom_centers.lon[0].value, geom_centers.lat[0].value

    sky_mask = np.asarray(data["mask_threshold_95"], dtype=bool) if restrict_to_mask else None
    if restrict_to_mask:
        cdf_valid = cdf_valid & sky_mask

    rng = np.random.default_rng(seed)
    draw = sample_sky_and_distance(prob_gw, cdf_valid, cdf_table, r_grid,
                                   bin_c_ra, bin_c_dec, int(n_sim), rng,
                                   sky_mask=sky_mask)
    iy, ix, d_sim = draw["iy"], draw["ix"], draw["d_sim"]

    # --- 1. sky marginal -------------------------------------------------
    w = np.where(cdf_valid, prob_gw, 0.0)
    w_flat = w.ravel() / w.sum()
    counts = np.bincount(np.ravel_multi_index((iy, ix), prob_gw.shape),
                         minlength=w_flat.size).astype(float)
    expected = w_flat * n_sim
    keep = expected >= 5.0                      # chi-square validity condition
    if keep.sum() >= 2:
        obs, exp = counts[keep], expected[keep]
        exp = exp * obs.sum() / exp.sum()
        chi2, p_sky = chisquare(obs, exp)
        dof = keep.sum() - 1
    else:
        chi2, p_sky, dof = np.nan, np.nan, 0

    # --- 2. distance marginal vs analytic mixture ------------------------
    valid_ij = np.argwhere(cdf_valid)
    w_valid = prob_gw[cdf_valid]
    w_valid = w_valid / w_valid.sum()
    cdf_agg = np.zeros_like(r_grid)
    for (i, j), wt in zip(valid_ij, w_valid):
        cdf_agg += wt * cdf_table[i, j]
    cdf_agg = np.clip(cdf_agg / cdf_agg[-1], 0.0, 1.0)

    # If r_grid truncates the posterior, CDF(r_min) > 0 or CDF(r_max) < 1 and
    # the sampler correctly returns probability ATOMS at the grid edges. A
    # naive KS statistic then saturates on those ties and reports a "failure"
    # that is a property of the grid, not of the sampler. So: test the
    # continuous interior against the conditional theory, and report the atoms
    # separately as a truncation diagnostic.
    atom_lo = float(np.mean(d_sim <= r_grid[0] + 1e-9))
    atom_hi = float(np.mean(d_sim >= r_grid[-1] - 1e-9))
    interior = (d_sim > r_grid[0] + 1e-9) & (d_sim < r_grid[-1] - 1e-9)

    c0, c1 = float(cdf_agg[0]), float(cdf_agg[-1])
    d_sorted = np.sort(d_sim[interior])
    n_int = d_sorted.size
    if n_int > 10 and (c1 - c0) > 0:
        ecdf = np.arange(1, n_int + 1) / n_int
        theory = (np.interp(d_sorted, r_grid, cdf_agg) - c0) / (c1 - c0)
        d_plus = np.max(ecdf - theory)
        d_minus = np.max(theory - np.arange(0, n_int) / n_int)
        ks_marg = float(max(d_plus, d_minus))
        lam = (np.sqrt(n_int) + 0.12 + 0.11 / np.sqrt(n_int)) * ks_marg
        p_marg = float(np.clip(
            2.0 * np.sum([(-1) ** (k - 1) * np.exp(-2 * k ** 2 * lam ** 2)
                          for k in range(1, 101)]), 0.0, 1.0))
    else:
        ecdf = np.array([]); ks_marg, p_marg, n_int = np.nan, np.nan, 0

    # --- 3. per-bin conditional in the hottest valid bin -----------------
    hot_flat = np.argmax(np.where(cdf_valid, prob_gw, -np.inf))
    i_hot, j_hot = np.unravel_index(hot_flat, prob_gw.shape)
    sel = (iy == i_hot) & (ix == j_hot)
    n_hot = int(sel.sum())
    if n_hot > 30:
        dh = np.sort(d_sim[sel])
        e_h = np.arange(1, dh.size + 1) / dh.size
        c_h = cdf_table[i_hot, j_hot] / cdf_table[i_hot, j_hot][-1]
        ks_hot = float(np.max(np.abs(e_h - np.interp(dh, r_grid, c_h))))
        ks_hot_crit = 1.36 / np.sqrt(n_hot)
    else:
        ks_hot, ks_hot_crit = np.nan, np.nan

    # --- 4. support ------------------------------------------------------
    n_nonpos = int((d_sim <= 0).sum())
    in_range = bool(d_sim.min() >= r_grid[0] - 1e-9 and d_sim.max() <= r_grid[-1] + 1e-9)

    # KS critical value at 5% for the marginal test (interior sample only)
    ks_crit = 1.36 / np.sqrt(max(n_int, 1))
    verdict = (p_sky > 0.001 or np.isnan(p_sky)) and ks_marg < 2 * ks_crit \
        and n_nonpos == 0 and in_range

    out = {
        "n_sim": int(n_sim), "chi2_sky": float(chi2), "dof_sky": int(dof),
        "p_sky": float(p_sky), "ks_distance_marginal": ks_marg,
        "p_distance_marginal": p_marg, "ks_critical_5pct": float(ks_crit),
        "n_interior": int(n_int),
        "atom_at_r_min": atom_lo, "atom_at_r_max": atom_hi,
        "ks_distance_hotbin": ks_hot, "n_distance_nonpositive": n_nonpos,
        "distance_within_grid": in_range, "hot_bin": (int(i_hot), int(j_hot)),
        "d_median": float(np.median(d_sim)),
        "d_mean": float(np.mean(d_sim)),
        "n_hot_bin": n_hot, "ks_critical_hotbin": float(ks_hot_crit),
        # 1/d^2 is what the flux conversion needs; guard against the d = 0 draws
        # that a distance grid starting at zero produces.
        "inv_d2_mean_Mpc-2": (float(np.mean(1.0 / d_sim[d_sim > 0] ** 2))
                              if np.any(d_sim > 0) else np.nan),
        "passed": bool(verdict),
    }

    if verbose:
        print("\n" + "=" * 78)
        print("3D sampler crosschecks")
        print("=" * 78)
        print(f"  Sky marginal      chi2/dof = {chi2:.1f}/{dof}  p = {p_sky:.3f}  "
              f"({'OK' if p_sky > 0.001 or np.isnan(p_sky) else 'MISMATCH'})")
        print(f"  Distance marginal KS = {ks_marg:.4f}  (5% critical "
              f"{ks_crit:.4f}, p = {p_marg:.3f}, interior N = {n_int})")
        if atom_lo > 1e-3 or atom_hi > 1e-3:
            print(f"  Grid-edge atoms: {atom_lo*100:.2f}% of draws at r_min = "
                  f"{r_grid[0]:.0f} Mpc, {atom_hi*100:.2f}% at r_max = "
                  f"{r_grid[-1]:.0f} Mpc -- the distance grid truncates the "
                  f"posterior; widen r_grid if these are not negligible.")
        print(f"  Hottest bin {out['hot_bin']} conditional KS = {ks_hot:.4f} "
              f"(5% critical {ks_hot_crit:.4f}, N = {n_hot})")
        print(f"  Distances: median {out['d_median']:.0f} Mpc, mean "
              f"{out['d_mean']:.0f} Mpc, <1/d^2> = {out['inv_d2_mean_Mpc-2']:.3e} Mpc^-2")
        print(f"  Support: {n_nonpos} non-positive draws, within r_grid = {in_range}")
        if n_nonpos:
            print(f"  ERROR: {n_nonpos} draws have d <= 0. The flux conversion goes as "
                  f"1/d^2, so these inject an infinite flux and would dominate the "
                  f"Lambda distribution. r_grid starts at {r_grid[0]:g} Mpc -- start it "
                  f"strictly above 0 (e.g. 1 Mpc) and rebuild the CDF table.")
        print(f"  VERDICT: {'PASS' if verdict else 'CHECK THE WARNINGS ABOVE'}")
        print("=" * 78)

    if make_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
        ax[0].hist(d_sim, 60, density=True, color="darkorange", alpha=0.65,
                   label=f"sampler draws (N={n_sim})")
        pdf_agg = np.gradient(cdf_agg, r_grid)
        ax[0].plot(r_grid, pdf_agg, "k--", lw=1.6, label="analytic mixture PDF")
        ax[0].set_xlabel("d [Mpc]"); ax[0].set_ylabel("p(d)")
        ax[0].set_xlim(0, np.percentile(d_sim, 99.8) * 1.2)
        ax[0].legend(fontsize=8, frameon=False); ax[0].set_title("Distance marginal")

        ax[1].plot(d_sorted, c0 + ecdf * (c1 - c0), lw=1.5, label="empirical CDF")
        ax[1].plot(r_grid, cdf_agg, "k--", lw=1.4, label="analytic CDF")
        ax[1].set_xlabel("d [Mpc]"); ax[1].set_ylabel("CDF")
        ax[1].set_xlim(0, np.percentile(d_sim, 99.8) * 1.2)
        ax[1].legend(fontsize=8, frameon=False)
        ax[1].set_title(f"KS = {ks_marg:.4f} (crit {ks_crit:.4f})")

        freq = counts.reshape(prob_gw.shape) / n_sim
        im = ax[2].imshow((freq - w).T, origin="lower", cmap="coolwarm",
                          vmin=-np.abs(freq - w).max(), vmax=np.abs(freq - w).max())
        plt.colorbar(im, ax=ax[2], label="sampled - expected")
        ax[2].set_title(f"Sky residual  (p = {p_sky:.3f})")
        fig.tight_layout(); plt.show()

    return out


# ===========================================================================
# 6. Shared bisection engine for the upper limit
# ===========================================================================

def _binomial_error(frac, n, floor=1e-6):
    """Standard error on a fraction, with a floor so 0 and 1 stay usable."""
    frac = np.clip(np.asarray(frac, dtype=float), 0.0, 1.0)
    n = np.maximum(np.asarray(n, dtype=float), 1.0)
    return np.maximum(np.sqrt(frac * (1.0 - frac) / n), floor)


def _interpolate_crossing(x_vals, frac_vals, n_vals, cl):
    """
    Estimate where frac(x) crosses `cl`, using ALL points evaluated during the
    bisection rather than only the final bracket.

    Bisection alone returns sqrt(lo*hi), whose accuracy is bounded by the
    bracket width; the fraction curve is smooth in log x (especially with
    common random numbers), so a local linear interpolation in
    (log10 x, frac) is both more accurate and gives a statistical uncertainty
    for free by propagating the binomial error through the local slope.

    Returns (x_ul, x_lo, x_hi, slope) -- limits are 1 sigma, NaN if unavailable.
    """
    x = np.asarray(x_vals, dtype=float)
    f = np.asarray(frac_vals, dtype=float)
    n = np.asarray(n_vals, dtype=float)
    order = np.argsort(x)
    x, f, n = x[order], f[order], n[order]
    lx = np.log10(x)

    # Adjacent pairs that straddle the CL. If the curve is noisy there can be
    # several; take the last (largest-x) one, which is the one the bisection
    # actually homed in on.
    straddle = np.where((f[:-1] <= cl) & (f[1:] >= cl))[0]
    if straddle.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    i_lo = int(straddle[-1])
    j_hi = i_lo + 1

    f_lo, f_hi = f[i_lo], f[j_hi]
    if np.isclose(f_hi, f_lo):
        lx_ul, slope = 0.5 * (lx[i_lo] + lx[j_hi]), np.nan
    else:
        slope = (f_hi - f_lo) / (lx[j_hi] - lx[i_lo])       # d frac / d log10 x
        lx_ul = lx[i_lo] + (cl - f_lo) / slope

    if np.isfinite(slope) and abs(slope) > 0:
        err_f = float(np.mean(_binomial_error([f_lo, f_hi], [n[i_lo], n[j_hi]])))
        d_lx = err_f / abs(slope)
        return 10 ** lx_ul, 10 ** (lx_ul - d_lx), 10 ** (lx_ul + d_lx), slope
    return 10 ** lx_ul, np.nan, np.nan, slope


def _bisect_ul(evaluate, x_lo, x_hi, cl, target, precision, frac_tol, max_iter,
               label="x", unit="", lambda_cache=None, cache_path=None,
               check_bracket=True, n_sigma_stop=2.0, on_iteration=None, verbose=True):
    """
    Bisection in log10(x) for the value of x at which
    P(Lambda_sim > target) = cl. Shared by the 2D and 3D upper limits.

    `evaluate(x) -> ndarray of Lambda` must be deterministic and cached by the
    caller (it is, via `lambda_cache`).

    Improvements over the previous per-method loops:
      * the initial bracket is verified to actually contain the crossing
        instead of silently returning an endpoint,
      * the convergence test accounts for Monte-Carlo noise: requiring
        |frac - cl| < frac_tol is impossible when frac_tol is smaller than the
        binomial error sqrt(cl(1-cl)/N) (e.g. frac_tol = 0.005 with N = 1000,
        where the noise alone is 0.0069), so the tolerance is widened to
        n_sigma_stop x that error and the run no longer burns every iteration,
      * the final value comes from interpolating the fraction curve, with a
        1 sigma statistical uncertainty, not just sqrt(lo*hi),
      * monotonicity of the fraction curve is checked and reported.
    """
    lambda_cache = {} if lambda_cache is None else lambda_cache
    history = {"x": [], "frac": [], "err": [], "n": [], "lo": [], "hi": []}

    def _frac_at(x):
        lam = evaluate(x)
        lam = np.asarray(lam, dtype=float)
        lam = lam[np.isfinite(lam)]
        if lam.size == 0:
            raise ValueError(f"All Lambda values are non-finite at {label} = {x:.4e}")
        f = float(np.mean(lam > target))
        return f, lam.size

    def _record(x, f, n, lo, hi):
        history["x"].append(x); history["frac"].append(f)
        history["err"].append(float(_binomial_error(f, n)))
        history["n"].append(n); history["lo"].append(lo); history["hi"].append(hi)

    # --- bracket verification ------------------------------------------
    bracket_warning = None
    if check_bracket:
        f_lo, n_lo = _frac_at(x_lo)
        f_hi, n_hi = _frac_at(x_hi)
        _record(x_lo, f_lo, n_lo, x_lo, x_hi)
        _record(x_hi, f_hi, n_hi, x_lo, x_hi)
        if verbose:
            print(f"\n[bracket] {label}={x_lo:.3e} {unit} -> frac={f_lo:.3f}   |   "
                  f"{label}={x_hi:.3e} {unit} -> frac={f_hi:.3f}   (CL={cl})")
        if f_lo > cl:
            bracket_warning = (
                f"frac({label}={x_lo:.3e}) = {f_lo:.3f} already exceeds CL = {cl}: "
                f"the true limit is BELOW the lower bracket. Lower `{label}_lo`.")
        elif f_hi < cl:
            bracket_warning = (
                f"frac({label}={x_hi:.3e}) = {f_hi:.3f} is still below CL = {cl}: "
                f"the true limit is ABOVE the upper bracket. Raise `{label}_hi`.")
        if bracket_warning:
            print(f"  ERROR: {bracket_warning}")
            print("  The bisection below will converge to a bracket edge and the "
                  "returned limit will be meaningless. Fix the bracket and rerun.")

    # --- bisection -------------------------------------------------------
    converged = False
    for iteration in range(max_iter):
        x_mid = np.sqrt(x_lo * x_hi)
        frac, n_eff = _frac_at(x_mid)
        err = float(_binomial_error(frac, n_eff))

        if frac > cl:
            x_hi = x_mid
        else:
            x_lo = x_mid
        _record(x_mid, frac, n_eff, x_lo, x_hi)

        if on_iteration is not None:
            on_iteration(iteration, x_mid, frac, x_lo, x_hi, history)

        width = np.log10(x_hi / x_lo)
        tol = max(frac_tol, n_sigma_stop * err)
        if verbose:
            print(f"[Iter {iteration+1:2d}] {label} = {x_mid:.4e} {unit}  "
                  f"N = {n_eff}  frac = {frac:.3f} +- {err:.3f}  "
                  f"bracket = [{x_lo:.3e}, {x_hi:.3e}]  width = {width:.4f} dex")

        bracket_ok = width < precision
        frac_ok = abs(frac - cl) < tol
        if bracket_ok and frac_ok:
            converged = True
            if verbose:
                print(f"  Converged after {iteration+1} iterations "
                      f"(bracket {width:.4f} dex, |frac - CL| = {abs(frac-cl):.4f} "
                      f"< {tol:.4f}).")
            break
        if verbose:
            if bracket_ok and not frac_ok:
                print(f"    bracket converged but |frac - CL| = {abs(frac-cl):.4f} "
                      f">= {tol:.4f} (MC noise {err:.4f}) -- continuing")
            elif frac_ok and not bracket_ok:
                print(f"    frac within tolerance but bracket is {width:.3f} dex "
                      f"wide -- continuing")
    else:
        if verbose:
            print(f"  WARNING: reached max_iter = {max_iter} without meeting both "
                  f"convergence criteria.")

    # --- final estimate + diagnostics ------------------------------------
    x_interp, x_1s_lo, x_1s_hi, slope = _interpolate_crossing(
        history["x"], history["frac"], history["n"], cl)
    x_bisect = float(np.sqrt(x_lo * x_hi))
    x_ul = x_interp if np.isfinite(x_interp) else x_bisect

    order = np.argsort(history["x"])
    f_sorted = np.asarray(history["frac"])[order]
    n_violations = int((np.diff(f_sorted) < -3 * np.mean(history["err"])).sum())
    if n_violations and verbose:
        print(f"  NOTE: the fraction curve is non-monotonic at {n_violations} "
              f"place(s) beyond 3x the MC error. Bisection assumes monotonicity; "
              f"consider more simulations per step or a wider bracket.")

    # atol must be 0 here: with the default atol = 1e-8, every amplitude of
    # order 1e-12 is "close" to every other one and the flag fires always.
    edge_flag = bool(
        np.isclose(x_ul, min(history["x"]), rtol=0.02, atol=0.0)
        or np.isclose(x_ul, max(history["x"]), rtol=0.02, atol=0.0)
    )

    return {
        "x_ul": float(x_ul),
        "x_ul_bisect": x_bisect,
        "x_ul_interp": float(x_interp) if np.isfinite(x_interp) else np.nan,
        "x_ul_lo_1sigma": float(x_1s_lo), "x_ul_hi_1sigma": float(x_1s_hi),
        "bracket_final": (float(x_lo), float(x_hi)),
        "bracket_width_dex": float(np.log10(x_hi / x_lo)),
        "slope_dfrac_dlogx": float(slope) if np.isfinite(slope) else np.nan,
        "converged": bool(converged),
        "at_bracket_edge": bool(edge_flag),
        "bracket_warning": bracket_warning,
        "n_monotonicity_violations": n_violations,
        "history": history,
        "lambda_cache": lambda_cache,
        "target": float(target), "cl": float(cl),
    }


def _cached_evaluator(simulate_fn, lambda_cache, cache_path, tag, verbose=True):
    """Wrap a simulation call with the on-disk / in-memory Lambda cache."""
    def evaluate(x):
        key = f"{x:.6e}"
        if key in lambda_cache:
            if verbose:
                print(f"  [cache] {tag} = {x:.4e} -- reusing "
                      f"{len(lambda_cache[key])} existing simulations")
            return lambda_cache[key]
        lambda_cache[key] = simulate_fn(x)
        if cache_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(lambda_cache, f)
            if verbose:
                print(f"  [cache] saved {len(lambda_cache)} entries -> {cache_path}")
        return lambda_cache[key]
    return evaluate


def _load_cache(cache_path, verbose=True):
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if verbose:
            print(f"Loaded Lambda cache with {len(cache)} entries from {cache_path}\n")
        return cache
    return {}


# --- 6a. 2D: upper limit on the flux ---------------------------------------

def run_iterative_ul(
    path_pkl, lambda_real, lambda_bkg, lambda_bkg_m, significance, p_value,
    energy_edges, cl, n_sim=500, precision=0.05, frac_tol=0.01,
    amp_lo=1e-13, amp_hi=1e-10, max_iter=20, cache_path=None,
    spectral_index=2.0, e_ref=E_REF, seed=None, position_seed=12345,
    tmp_dir=str(paths.TMP_DIR), check_bracket=True, make_plots=True, verbose=True,
):
    """
    2D flux upper limit: bisect the PWL amplitude until a fraction `cl` of
    simulated Lambdas exceeds the target (the observed Lambda, or the
    background median when the observation under-fluctuates).

    Returns a dict containing, among the diagnostics, the full set of
    band-integrated flux quantities implied by the limit (`flux_point`).
    """
    target = lambda_bkg_m if significance < 0 else lambda_real
    if verbose:
        print("=" * 78)
        print(f"2D iterative flux UL  |  CL = {cl}  |  target Lambda = {target:.3f} "
              f"({'BKG median (observation under-fluctuates)' if significance < 0 else 'observed'})")
        print("=" * 78)

    lambda_cache = _load_cache(cache_path, verbose)
    os.makedirs(tmp_dir, exist_ok=True)

    def _simulate(amp):
        fname = os.path.join(tmp_dir, f"iterative_ul_{amp:.6e}.npz")
        perform_n_simulations(
            n_sim=n_sim, amplitude=amp, file_input=path_pkl, file_output=fname,
            compute_uls=0, spectral_index=spectral_index, e_ref=e_ref,
            seed=seed, position_seed=position_seed,
            store_ts_maps=False, store_stats=False, verbose=False,
        )
        return np.load(fname)["lambda_data"]

    evaluate = _cached_evaluator(_simulate, lambda_cache, cache_path, "phi0", verbose)

    on_iteration = None
    if make_plots and plot_ul_iteration is not None:
        import matplotlib.pyplot as plt

        def _plot_iteration(it, x_mid, frac, lo, hi, hist):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
            try:
                plot_ul_iteration(
                    ax1, ax2, lambda_bkg, lambda_cache[f"{x_mid:.6e}"], lambda_real,
                    lambda_bkg_m, target, hist["x"], hist["frac"], x_mid, lo, hi,
                    frac, cl, p_value, significance, it, max_iter, energy_edges,
                    precision,
                )
                plt.tight_layout(); plt.show()
            except Exception as exc:                     # plotting must never kill a run
                plt.close(fig)
                print(f"  (per-iteration plot skipped: {exc})")

        on_iteration = _plot_iteration

    res = _bisect_ul(
        evaluate, amp_lo, amp_hi, cl, target, precision, frac_tol, max_iter,
        label="phi0", unit="cm-2 s-1 TeV-1", lambda_cache=lambda_cache,
        cache_path=cache_path, check_bracket=check_bracket,
        on_iteration=on_iteration, verbose=verbose,
    )

    amp_final = res["x_ul"]
    fp = flux_point_from_amplitude(amp_final, energy_edges, spectral_index, e_ref)

    if verbose:
        print(f"\n{'='*78}")
        print("2D FLUX UPPER LIMIT")
        print(f"{'-'*78}")
        print(f"  {str(fp)}")
        if np.isfinite(res['x_ul_lo_1sigma']):
            print(f"  phi0 stat. uncert. (1 sigma, MC): "
                  f"[{res['x_ul_lo_1sigma']:.3e}, {res['x_ul_hi_1sigma']:.3e}]")
        print(f"  bisection value {res['x_ul_bisect']:.3e}  |  interpolated "
              f"{res['x_ul_interp']:.3e}  |  bracket width "
              f"{res['bracket_width_dex']:.4f} dex")
        print(f"  iterations {len(res['history']['x'])}  |  total simulations "
              f"{sum(len(v) for v in lambda_cache.values())}  |  converged "
              f"{res['converged']}")
        if res["at_bracket_edge"]:
            print("  WARNING: the limit sits at the edge of the tested range -- widen it.")
        print(f"{'='*78}")

    out = dict(res)
    out.update({
        # legacy keys, unchanged meaning
        "amp_ul": amp_final,
        "flux_ul": fp.photon_flux,
        "flux_diff_ul": fp.energy_flux,
        "hist_amp": res["history"]["x"],
        "hist_frac": res["history"]["frac"],
        "hist_err": res["history"]["err"],
        # new
        "method": "2d",
        "flux_point": fp,
        "photon_flux_ul": fp.photon_flux,
        "energy_flux_ul": fp.energy_flux,
        "e2dnde_ul": fp.e2dnde,
        "energy_edges": energy_edges,
        "spectral_index": float(spectral_index),
    })
    return out


# --- 6b. 3D: upper limit on the band luminosity ----------------------------

def run_iterative_ul_3d(
    path_pkl, lambda_real, lambda_bkg, lambda_bkg_m, significance, p_value,
    cl, n_sim=500, precision=0.05, frac_tol=0.01, lum_lo=1e45, lum_hi=1e52,
    max_iter=20, cache_path=None, spectral_index=2.0, e_ref=E_REF, seed=None,
    sampling_seed=12345, luminosity_band=None, apply_k_correction=False,
    restrict_to_mask=False, tmp_dir=str(paths.TMP_DIR), check_bracket=True, verbose=True,
):
    """
    3D luminosity upper limit: same bisection, on the isotropic-equivalent
    BAND luminosity L0, with the sky position and the distance marginalised
    over the GW posterior at every realisation.

    Now shares `_bisect_ul` with the 2D version, so both limits are defined by
    exactly the same convergence criteria and the same crossing estimator --
    previously the 3D loop was a hand-copied variant that had already drifted
    (no bracket check, no MC-aware tolerance, no interpolated crossing).
    """
    target = lambda_bkg_m if significance < 0 else lambda_real
    band = luminosity_band
    if band is None:
        with open(path_pkl, "rb") as f:
            band = pickle.load(f)["energy_edges"]

    if verbose:
        e_lo, e_hi = _band(band)
        print("=" * 78)
        print(f"3D iterative luminosity UL  |  CL = {cl}  |  target Lambda = {target:.3f}")
        print(f"L0 is the isotropic-equivalent luminosity in [{e_lo:.2f}, {e_hi:.2f}], "
              f"index {spectral_index} (NOT bolometric)")
        print("=" * 78)

    lambda_cache = _load_cache(cache_path, verbose)
    os.makedirs(tmp_dir, exist_ok=True)

    def _simulate(lum):
        fname = os.path.join(tmp_dir, f"iterative_ul3d_{lum:.6e}.npz")
        perform_n_simulations_3d(
            n_sim=n_sim, luminosity=lum, file_input=path_pkl, file_output=fname,
            compute_uls=0, spectral_index=spectral_index, e_ref=e_ref, seed=seed,
            sampling_seed=sampling_seed, luminosity_band=band,
            apply_k_correction=apply_k_correction, restrict_to_mask=restrict_to_mask,
            store_ts_maps=False, store_stats=False, verbose=False,
        )
        return np.load(fname)["lambda_data"]

    evaluate = _cached_evaluator(_simulate, lambda_cache, cache_path, "L0", verbose)

    res = _bisect_ul(
        evaluate, lum_lo, lum_hi, cl, target, precision, frac_tol, max_iter,
        label="L0", unit="erg/s", lambda_cache=lambda_cache, cache_path=cache_path,
        check_bracket=check_bracket, verbose=verbose,
    )

    lum_final = res["x_ul"]

    if verbose:
        print(f"\n{'='*78}")
        print("3D LUMINOSITY UPPER LIMIT")
        print(f"{'-'*78}")
        e_lo, e_hi = _band(band)
        print(f"  L0 UL = {lum_final:.3e} erg/s   [{e_lo:.2f}-{e_hi:.2f}, "
              f"isotropic-equivalent, index {spectral_index}]")
        if np.isfinite(res["x_ul_lo_1sigma"]):
            print(f"  stat. uncert. (1 sigma, MC): "
                  f"[{res['x_ul_lo_1sigma']:.3e}, {res['x_ul_hi_1sigma']:.3e}] erg/s")
        print(f"  bisection value {res['x_ul_bisect']:.3e}  |  interpolated "
              f"{res['x_ul_interp']:.3e}  |  bracket width "
              f"{res['bracket_width_dex']:.4f} dex")
        print(f"  iterations {len(res['history']['x'])}  |  total simulations "
              f"{sum(len(v) for v in lambda_cache.values())}  |  converged "
              f"{res['converged']}")
        if res["at_bracket_edge"]:
            print("  WARNING: the limit sits at the edge of the tested range -- widen "
                  "`lum_lo`/`lum_hi`.")
        print(f"{'='*78}")

    out = dict(res)
    out.update({
        "lum_ul": lum_final,
        "hist_lum": res["history"]["x"],
        "hist_frac": res["history"]["frac"],
        "hist_err": res["history"]["err"],
        "method": "3d",
        "luminosity_band": band,
        "spectral_index": float(spectral_index),
        "apply_k_correction": bool(apply_k_correction),
        "restrict_to_mask": bool(restrict_to_mask),
    })
    return out


# ===========================================================================
# 7. Reporting: putting the two methods on the same footing
# ===========================================================================

def marginal_distance_pdf(prob_gw, cdf_table, cdf_valid, r_grid):
    """Sky-marginalised distance CDF and PDF over the FoV (the 3D prior)."""
    w = np.where(cdf_valid, np.asarray(prob_gw, dtype=float), 0.0)
    w = w / w.sum()
    cdf = np.tensordot(w, np.asarray(cdf_table, dtype=float), axes=([0, 1], [0, 1]))
    cdf = np.clip(cdf / cdf[-1], 0.0, 1.0)
    pdf = np.gradient(cdf, r_grid)
    pdf = np.clip(pdf, 0.0, None)
    area = np.trapezoid(pdf, r_grid) if hasattr(np, "trapezoid") else np.trapz(pdf, r_grid)
    return cdf, pdf / max(area, 1e-300)


def luminosity_ul_from_flux_ul(amplitude_ul, energy_edges, r_grid, cdf,
                               index=2.0, e_ref=E_REF, percentiles=(5, 50, 95),
                               apply_k_correction=False):
    """
    Convert a 2D flux UL into an equivalent BAND luminosity, marginalised over
    a distance CDF instead of evaluated at one hand-picked pixel distance.

    The previous notebook cell used the median distance of a single WCS bin
    (either the injection pixel or the GW hotspot). That is a point estimate of
    a quantity whose prior spans a factor of a few in distance, hence a factor
    of ~10 in L; quoting the full quantile range makes the comparison with the
    3D limit honest.

    Returns a dict with the energy flux, the L quantiles, and L at <d^2>^(1/2).
    """
    f_e = pwl_energy_flux(amplitude_ul, energy_edges, index=index, e_ref=e_ref)

    d_q = np.interp(np.asarray(percentiles, dtype=float) / 100.0, cdf, r_grid)
    lum_q = luminosity_from_energy_flux(
        f_e, d_q * u.Mpc, index=index,
        redshift="auto" if apply_k_correction else None,
        apply_k_correction=apply_k_correction)

    pdf = np.gradient(cdf, r_grid)
    pdf = np.clip(pdf, 0.0, None)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    norm = max(trapz(pdf, r_grid), 1e-300)
    d2_mean = trapz(r_grid ** 2 * pdf, r_grid) / norm
    lum_d2 = luminosity_from_energy_flux(
        f_e, np.sqrt(d2_mean) * u.Mpc, index=index,
        redshift="auto" if apply_k_correction else None,
        apply_k_correction=apply_k_correction)

    return {
        "amplitude_ul": float(amplitude_ul),
        "energy_flux_ul": f_e,
        "percentiles": tuple(percentiles),
        "distances_Mpc": d_q,
        "luminosities": lum_q,
        "d_rms_Mpc": float(np.sqrt(d2_mean)),
        "luminosity_at_d_rms": lum_d2,
    }


def flux_ul_from_luminosity_ul(luminosity_ul, energy_edges, r_grid, cdf,
                               index=2.0, e_ref=E_REF, percentiles=(5, 50, 95),
                               apply_k_correction=False):
    """Mirror image: the 3D luminosity UL expressed as an equivalent flux,
    quantile by quantile over the same distance prior."""
    d_q = np.interp(np.asarray(percentiles, dtype=float) / 100.0, cdf, r_grid)
    amps = amplitude_from_luminosity(
        luminosity_ul, d_q * u.Mpc, energy_edges, index=index, e_ref=e_ref,
        redshift="auto" if apply_k_correction else None,
        apply_k_correction=apply_k_correction)
    amps = np.atleast_1d(amps)
    return {
        "luminosity_ul": float(luminosity_ul),
        "percentiles": tuple(percentiles),
        "distances_Mpc": d_q,
        "amplitudes": amps,
        "photon_fluxes": u.Quantity(
            [pwl_photon_flux(a, energy_edges, index, e_ref) for a in amps]),
        "energy_fluxes": u.Quantity(
            [pwl_energy_flux(a, energy_edges, index, e_ref) for a in amps]),
    }


def _fmt_q(values, fmt="{:.3e}"):
    """
    Format a (low, median, high) triple as "median [lo, hi]".

    The bracket is sorted, because quantities derived from distance quantiles
    can be either increasing (luminosity) or decreasing (flux) in distance;
    printing the raw order would show [high, low] for the flux-equivalent rows.
    """
    v = np.atleast_1d(u.Quantity(values).value if hasattr(values, "unit")
                      else np.asarray(values))
    if v.size == 3:
        lo, hi = min(v[0], v[2]), max(v[0], v[2])
        return f"{fmt.format(v[1])}  [{fmt.format(lo)}, {fmt.format(hi)}]"
    return "  ".join(fmt.format(x) for x in v)


def summarize_upper_limits(result_2d=None, result_3d=None, energy_edges=None,
                           r_grid=None, distance_cdf=None, index=2.0, e_ref=E_REF,
                           cl=0.95, significance=None, source_name="",
                           percentiles=(5, 50, 95), apply_k_correction=False):
    """
    Print (and return) one table holding BOTH upper limits expressed BOTH ways:
    as a flux and as a luminosity.

    For each method the flux side is reported as the amplitude phi0, the
    band-integrated photon flux, the band-integrated energy flux, and the SED
    point E^2 dN/dE at the logarithmic band centre -- all from the same
    integrated power law, never from a single-energy shortcut. The luminosity
    side is reported as a distance quantile range, because neither limit is a
    single number once the GW distance posterior is taken into account:

      * the 2D method measures a flux and needs a distance to become a
        luminosity, so its L is a quantile spread,
      * the 3D method measures a luminosity and needs a distance to become a
        flux, so its F is a quantile spread.

    `distance_cdf` and `r_grid` come from `marginal_distance_pdf`.
    """
    have_dist = r_grid is not None and distance_cdf is not None
    out = {"cl": cl, "index": index, "energy_edges": energy_edges,
           "source_name": source_name}

    e_lo, e_hi = _band(energy_edges)
    header = (f"UPPER LIMITS   {source_name}   CL = {cl:.0%}   "
              f"band [{e_lo:.2f}, {e_hi:.2f}]   PWL index {index} "
              f"(dN/dE ~ E^-{index:g})")
    print("\n" + "=" * 86)
    print(header)
    if significance is not None:
        ref = ("observed Lambda" if significance >= 0
               else "BKG median Lambda (observation under-fluctuates)")
        print(f"Observation significance {significance:+.2f} sigma  ->  "
              f"limits referenced to the {ref}")
    print("=" * 86)

    if have_dist:
        d_q = np.interp(np.asarray(percentiles) / 100.0, distance_cdf, r_grid)
        print(f"GW distance posterior (FoV, sky-marginalised): "
              f"d = {d_q[1]:.0f} [{d_q[0]:.0f}, {d_q[2]:.0f}] Mpc "
              f"({percentiles[0]}/{percentiles[1]}/{percentiles[2]} percentiles)")
        out["distance_quantiles_Mpc"] = d_q

    # --- 2D --------------------------------------------------------------
    if result_2d is not None:
        amp = result_2d["amp_ul"]
        fp = result_2d.get("flux_point") or flux_point_from_amplitude(
            amp, energy_edges, index, e_ref)
        print("\n-- 2D method (fixed flux, sky position marginalised) " + "-" * 34)
        print(f"   phi0          {amp:.3e} cm-2 s-1 TeV-1 @ {fp.e_ref:.2f}")
        print(f"   photon flux   {fp.photon_flux.value:.3e} cm-2 s-1")
        print(f"   energy flux   {fp.energy_flux.value:.3e} erg cm-2 s-1")
        print(f"   E2 dN/dE      {fp.e2dnde.value:.3e} erg cm-2 s-1 @ {fp.e_dec:.2f}")
        out["2d"] = {"amplitude": amp, "flux_point": fp}
        if np.isfinite(result_2d.get("x_ul_lo_1sigma", np.nan)):
            print(f"   MC 1 sigma    [{result_2d['x_ul_lo_1sigma']:.3e}, "
                  f"{result_2d['x_ul_hi_1sigma']:.3e}] on phi0")
        if have_dist:
            conv = luminosity_ul_from_flux_ul(
                amp, energy_edges, r_grid, distance_cdf, index, e_ref,
                percentiles, apply_k_correction)
            print(f"   -> luminosity {_fmt_q(conv['luminosities'])} erg/s"
                  f"   (at d = {_fmt_q(conv['distances_Mpc'], '{:.0f}')} Mpc)")
            print(f"      at d_rms = {conv['d_rms_Mpc']:.0f} Mpc: "
                  f"{conv['luminosity_at_d_rms'].value:.3e} erg/s")
            out["2d"]["luminosity_equivalent"] = conv

    # --- 3D --------------------------------------------------------------
    if result_3d is not None:
        lum = result_3d["lum_ul"]
        print("\n-- 3D method (fixed luminosity, sky x distance marginalised) " + "-" * 25)
        print(f"   L0            {lum:.3e} erg/s  [isotropic-equivalent, "
              f"{e_lo:.2f}-{e_hi:.2f}]")
        if np.isfinite(result_3d.get("x_ul_lo_1sigma", np.nan)):
            print(f"   MC 1 sigma    [{result_3d['x_ul_lo_1sigma']:.3e}, "
                  f"{result_3d['x_ul_hi_1sigma']:.3e}] erg/s")
        out["3d"] = {"luminosity": lum}
        if have_dist:
            conv = flux_ul_from_luminosity_ul(
                lum, energy_edges, r_grid, distance_cdf, index, e_ref,
                percentiles, apply_k_correction)
            print(f"   -> phi0       {_fmt_q(conv['amplitudes'])} cm-2 s-1 TeV-1")
            print(f"   -> photon flx {_fmt_q(conv['photon_fluxes'])} cm-2 s-1")
            print(f"   -> energy flx {_fmt_q(conv['energy_fluxes'])} erg cm-2 s-1")
            print(f"      (at d = {_fmt_q(conv['distances_Mpc'], '{:.0f}')} Mpc)")
            out["3d"]["flux_equivalent"] = conv

    # --- comparison ------------------------------------------------------
    if result_2d is not None and result_3d is not None and have_dist:
        l2 = out["2d"]["luminosity_equivalent"]["luminosities"].value
        l3 = out["3d"]["luminosity"]
        ratio = l3 / l2[1]
        print("\n-- Consistency " + "-" * 71)
        print(f"   L(3D) / L(2D at median d) = {ratio:.2f}"
              f"   ({'3D tighter' if ratio < 1 else '3D looser'})")
        inside = l2[0] <= l3 <= l2[2]
        print(f"   3D limit {'lies within' if inside else 'lies OUTSIDE'} the 2D "
              f"distance-quantile range [{l2[0]:.2e}, {l2[2]:.2e}] erg/s")
        if not inside:
            print("   -> The gap is larger than the distance posterior alone explains. "
                  "Expected contributors, in rough order of size:")
            print("      * SKY SUPPORT: the 2D injector samples only inside the 95% "
                  "mask, the 3D one samples the FULL map. Probability outside the mask "
                  "lands where TS' is penalised or never searched, so the 3D limit is "
                  "diluted and comes out looser by construction. This is a modelling "
                  "choice, not an error -- but the two numbers are then not the same "
                  "statistic. Restrict the 3D draw to the mask if you want them to be.")
            print("      * <1/d^2> vs 1/<d>^2: the flux depends on 1/d^2, so the nearest "
                  "realisations dominate and the marginalised limit is not the limit at "
                  "the median distance.")
            print("      * Check both used the same index, band, target Lambda and CL.")
        out["comparison"] = {"ratio_3d_over_2d_median": float(ratio),
                             "within_distance_spread": bool(inside)}

    print("=" * 86)
    return out


def compare_2d_3d(result_2d, result_3d, energy_edges, prob_gw, cdf_table,
                  cdf_valid, r_grid, index=2.0, e_ref=E_REF, **kwargs):
    """Convenience wrapper: build the marginal distance CDF, then summarise."""
    cdf, _ = marginal_distance_pdf(prob_gw, cdf_table, cdf_valid, r_grid)
    return summarize_upper_limits(
        result_2d=result_2d, result_3d=result_3d, energy_edges=energy_edges,
        r_grid=r_grid, distance_cdf=cdf, index=index, e_ref=e_ref, **kwargs)


# ===========================================================================
# 8. Entry point
# ===========================================================================

def _main(argv):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GW follow-up simulations (2D flux or 3D luminosity).")
    if "--self-test" in argv:
        check_spectral_conversions()
        return
    parser.add_argument("n_sim", type=int)
    parser.add_argument("value", type=float,
                        help="PWL amplitude [cm-2 s-1 TeV-1] in 2D mode, "
                             "band luminosity [erg/s] in 3D mode.")
    parser.add_argument("file_input")
    parser.add_argument("file_output")
    parser.add_argument("compute_uls", type=int, nargs="?", default=0)
    parser.add_argument("--mode", choices=["2d", "3d"], default="2d")
    parser.add_argument("--index", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-store-maps", action="store_true",
                        help="Do not keep per-iteration TS maps / stats objects.")
    parser.add_argument("--self-test", action="store_true",
                        help="Run the spectral conversion crosschecks and exit.")
    args = parser.parse_args(argv)

    if args.self_test:
        check_spectral_conversions()
        return

    store = not args.no_store_maps
    if args.mode == "2d":
        perform_n_simulations(
            args.n_sim, args.value, args.file_input, args.file_output,
            compute_uls=args.compute_uls, spectral_index=args.index,
            seed=args.seed, store_ts_maps=store, store_stats=store)
    else:
        perform_n_simulations_3d(
            args.n_sim, args.value, args.file_input, args.file_output,
            compute_uls=args.compute_uls, spectral_index=args.index,
            seed=args.seed, store_ts_maps=store, store_stats=store)


if __name__ == "__main__":
    if len(sys.argv) < 5 and "--self-test" not in sys.argv:
        print("Usage: python simulate.py <n_sim> <value> <in.pkl> <out.npz> "
              "[compute_uls] [--mode 2d|3d] [--index 2.0] [--seed N]")
        print("       python simulate.py --self-test")
        sys.exit(1)
    _main(sys.argv[1:])
