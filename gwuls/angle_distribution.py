"""
angle_distribution.py -- the viewing-angle draw of Step 4.3: the joint
p(d, theta_v), and everything derived from it (marginals, and conditionals in
both directions).

Every row of the guidelines' Step 4 table sits behind one standardized
object, `ThetaDistribution`, so `sim_3d` never has to know which case applied
to a given GW alert:

| kind        | information used                          | theta_v depends on d? |
|-------------|-------------------------------------------|-----------------------|
| "pe"        | PE posterior samples (gwuls.gw_pe): joint | yes -- p(theta | d)   |
|             | p(d, theta_v) as a KDE on a grid, plus    | from the joint grid   |
|             | the samples themselves for joint draws    |                       |
| "selection" | population of GW-*detected* sources with  | yes -- nearer the     |
|             | a horizon distance: p(d, theta) ~         | horizon only face-on  |
|             | d^2 sin(theta) P_det(d / D_h, theta)      | sources survive       |
| "schutz"    | Schutz (2011): the d-marginal of the      | no                    |
|             | "selection" joint (approximately)         |                       |
| "isotropic" | geometric sin(theta), no GW selection     | no                    |
| "fixed"     | theta fixed by assumption                 | no                    |
| "two_bin"   | a dummy "on-axis-ish / off-axis-ish" split| no                    |

The joint lives in `JointDistanceAngle`, a density on a regular (d, theta_v)
grid, normalised so that the double integral is 1 (units Mpc^-1 deg^-1).
From it:

- p(theta_v | d): `conditional_theta(d)` -- what sim_3d needs once d_i is drawn;
- p(d | theta_v = theta_0) or p(d | theta_v in [a, b]): `conditional_distance`
  -- "the angle is known" (e.g. from an EM counterpart);
- p(d), p(theta_v): `marginal_distance`, `marginal_theta` -- the angle, or
  the distance, integrated out.

The notes phrase Step 4.3 as "marginalise over d_i"; the operation needed per
iteration is conditioning, p(theta | d_i) = p(d_i, theta) / p(d_i).

Viewing angle convention throughout: theta_v = min(theta_JN, 180 - theta_JN)
in [0, 90] deg, the angle to the nearer jet axis (gwuls.gw_pe.fold_viewing_angle).

All pdfs returned by this module are normalised per degree, so curves from
different kinds can be drawn on one axis and compared directly.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .gw_pe import fold_viewing_angle
from .utils import trapz as _trapz

__all__ = [
    "KINDS",
    "JointDistanceAngle",
    "ThetaDistribution",
    "make_theta_distribution",
    "save_theta_distribution",
    "load_theta_distribution",
    "schutz_pdf",
    "isotropic_pdf",
    "detection_probability",
    "hpd_levels",
    "conditional_theta_by_resampling",
    "sky_regions",
    "sky_dependence_check",
    "fold_viewing_angle",
]

KINDS = ("pe", "selection", "schutz", "isotropic", "fixed", "two_bin")

#: Default viewing-angle grid, 0.5 deg steps.
THETA_GRID_DEG = np.linspace(0.0, 90.0, 181)
_FINE_THETA_DEG = np.linspace(0.0, 90.0, 1801)


def _cumtrapz(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Cumulative trapezoid integral along `axis`, starting at 0."""
    y = np.moveaxis(np.asarray(y, dtype=float), axis, -1)
    dx = np.diff(x)
    inc = 0.5 * (y[..., 1:] + y[..., :-1]) * dx
    out = np.concatenate([np.zeros(y.shape[:-1] + (1,)), np.cumsum(inc, axis=-1)], axis=-1)
    return np.moveaxis(out, -1, axis)


def _inverse_cdf_rows(cum: np.ndarray, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Row-wise inverse of piecewise-linear cumulative curves.

    cum : (n, m) non-decreasing, unnormalised; u : (n,) in [0, 1).
    Returns x at which cum[i] reaches u[i] * cum[i, -1].
    """
    target = u * cum[:, -1]
    k = np.clip((cum < target[:, None]).sum(axis=1) - 1, 0, x.size - 2)
    rows = np.arange(cum.shape[0])
    lo, hi = cum[rows, k], cum[rows, k + 1]
    frac = np.where(hi > lo, (target - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
    return x[k] + np.clip(frac, 0.0, 1.0) * (x[k + 1] - x[k])


def _sample_1d(x: np.ndarray, pdf: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF draws from a tabulated 1D pdf (piecewise-linear CDF). Flat
    stretches of the CDF (zero pdf) have zero probability of being hit."""
    cum = _cumtrapz(pdf, x)
    return np.interp(rng.random(size), cum / cum[-1], x)


# ===========================================================================
# Closed-form, distance-independent p(theta_v)
# ===========================================================================

def isotropic_pdf(theta_deg) -> np.ndarray:
    """sin(theta) on [0, 90] deg, per degree (integrates to 1)."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    return np.where((theta_deg >= 0) & (theta_deg <= 90),
                    np.sin(np.radians(theta_deg)) * np.pi / 180.0, 0.0)


def _schutz_unnormalised(theta_deg):
    c = np.cos(np.radians(theta_deg))
    return np.sin(np.radians(theta_deg)) * ((1.0 + c ** 2) ** 2 / 4.0 + c ** 2) ** 1.5


_SCHUTZ_NORM = float(_trapz(_schutz_unnormalised(np.linspace(0, 90, 20001)),
                            np.linspace(0, 90, 20001)))


def schutz_pdf(theta_deg) -> np.ndarray:
    """Schutz (2011) GW-detected-source inclination distribution, per degree:
    p(theta) ~ sin(theta) [(1+cos^2 theta)^2/4 + cos^2 theta]^(3/2).
    Peaks near 31 deg, median near 36 deg."""
    theta_deg = np.asarray(theta_deg, dtype=float)
    return np.where((theta_deg >= 0) & (theta_deg <= 90),
                    _schutz_unnormalised(theta_deg) / _SCHUTZ_NORM, 0.0)


# ===========================================================================
# GW detection selection: P_det(d / D_h, theta)
# ===========================================================================

def _orientation_factor_samples(theta_deg: np.ndarray, n_mc: int, seed: int) -> np.ndarray:
    """Finn & Chernoff (1993) orientation factor Theta / 4 in [0, 1] of one
    L-shaped detector, for sources at inclination `theta_deg`, isotropic in
    sky position and polarisation. Shape (len(theta_deg), n_mc)."""
    rng = np.random.default_rng(seed)
    cos_sky = rng.uniform(-1.0, 1.0, n_mc)
    phi = rng.uniform(0.0, 2.0 * np.pi, n_mc)
    psi = rng.uniform(0.0, np.pi, n_mc)
    a = 0.5 * (1.0 + cos_sky ** 2) * np.cos(2 * phi)
    b = cos_sky * np.sin(2 * phi)
    f_plus = a * np.cos(2 * psi) - b * np.sin(2 * psi)
    f_cross = a * np.sin(2 * psi) + b * np.cos(2 * psi)
    ci = np.cos(np.radians(np.asarray(theta_deg, dtype=float)))[:, None]
    # Theta = 2 sqrt(F+^2 (1+ci^2)^2 + 4 Fx^2 ci^2), max 4 (overhead, face-on).
    return 0.5 * np.sqrt(f_plus ** 2 * (1 + ci ** 2) ** 2 + 4 * f_cross ** 2 * ci ** 2)


def detection_probability(x, theta_deg=THETA_GRID_DEG, n_mc: int = 100_000,
                          seed: int = 0) -> np.ndarray:
    """Sky- and polarisation-averaged probability that a source at
    x = d / D_h (D_h the horizon: SNR threshold reached overhead, face-on)
    and viewing angle theta is detected, i.e. that its orientation factor
    Theta/4 exceeds x. Shape (len(x), len(theta_deg))."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    w = np.sort(_orientation_factor_samples(np.asarray(theta_deg), n_mc, seed), axis=1)
    # fraction of Theta/4 > x, per theta row
    counts = np.stack([w.shape[1] - np.searchsorted(row, x, side="right") for row in w], axis=1)
    return counts / w.shape[1]


# ===========================================================================
# The joint p(d, theta_v) on a grid
# ===========================================================================

def hpd_levels(density: np.ndarray, cell_area, fractions=(0.5, 0.9)) -> np.ndarray:
    """Density thresholds enclosing the given highest-density fractions of
    the mass (for contour plots). Returned in increasing order of density,
    i.e. matplotlib-ready, for decreasing fractions."""
    dens = density.ravel()
    order = np.argsort(dens)[::-1]
    mass = np.broadcast_to(density * cell_area, density.shape).ravel()[order]
    cum = np.cumsum(mass) / mass.sum()
    levels = [dens[order][min(np.searchsorted(cum, f), dens.size - 1)] for f in fractions]
    return np.sort(np.asarray(levels))


class JointDistanceAngle:
    """p(d, theta_v) on a regular grid.

    `density[i, j]` is the density at (distance_Mpc[i], theta_deg[j]) in
    Mpc^-1 deg^-1, normalised so its double trapezoid integral is 1.
    Build with `from_samples` (KDE of PE posterior samples) or `from_function`.
    """

    def __init__(self, distance_Mpc, theta_deg, density, meta: Optional[dict] = None):
        self.distance_Mpc = np.asarray(distance_Mpc, dtype=float)
        self.theta_deg = np.asarray(theta_deg, dtype=float)
        density = np.clip(np.asarray(density, dtype=float), 0.0, None)
        if density.shape != (self.distance_Mpc.size, self.theta_deg.size):
            raise ValueError(f"density shape {density.shape} != (n_d, n_theta) = "
                             f"({self.distance_Mpc.size}, {self.theta_deg.size})")
        total = _trapz(_trapz(density, self.theta_deg, axis=1), self.distance_Mpc)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("joint density integrates to zero")
        self.density = density / total
        self.meta = dict(meta or {})

        # Precomputed pieces for conditionals and sampling.
        self._cum_theta = _cumtrapz(self.density, self.theta_deg, axis=1)  # (n_d, n_t)
        self._row_mass = self._cum_theta[:, -1]                              # p(d)
        valid = self._row_mass > 1e-6 * self._row_mass.max()
        self._d_valid = (self.distance_Mpc[valid][0], self.distance_Mpc[valid][-1])

    # -- construction -------------------------------------------------------
    @classmethod
    def from_samples(cls, distance_Mpc, theta_deg, weights=None, *, bw_scale: float = 1.0,
                     n_distance: int = 300, n_cos: int = 400,
                     theta_grid_deg: Optional[np.ndarray] = None,
                     distance_range: Optional[Tuple[float, float]] = None,
                     meta: Optional[dict] = None) -> "JointDistanceAngle":
        """Gaussian KDE of (d, theta_v) samples, evaluated on a grid.

        The KDE is done in (d, cos theta_v), where a smooth posterior has a
        finite, smooth density at both edges (theta_v = 0 and 90 deg), with
        reflection at cos theta_v = 0 and 1 (and at d = 0 if the grid reaches
        it), then mapped to theta_v with the Jacobian sin(theta_v). Doing it in
        theta_v directly would smear mass onto theta_v = 0, where any
        orientation posterior has zero density per degree.

        The kernel is a full-covariance Gaussian (Scott's rule, times
        `bw_scale`), so it follows the d-theta correlation instead of blurring
        across it. Binned on an (n_distance x n_cos) lattice and convolved by
        FFT: 20k samples take a fraction of a second.
        """
        from scipy.signal import fftconvolve

        d = np.asarray(distance_Mpc, dtype=float)
        th = np.asarray(theta_deg, dtype=float)
        if d.shape != th.shape:
            raise ValueError("distance and theta samples must have the same shape")
        if th.min() < 0 or th.max() > 90:
            raise ValueError("theta must be the folded viewing angle in [0, 90] deg "
                             "(use gw_pe.fold_viewing_angle on theta_JN)")
        w = np.ones_like(d) if weights is None else np.asarray(weights, dtype=float)
        w = w / w.sum()
        n_eff = 1.0 / np.sum(w ** 2)
        c = np.cos(np.radians(th))

        cov = np.cov(np.vstack([d, c]), aweights=w)
        H = cov * bw_scale ** 2 * n_eff ** (-1.0 / 3.0)  # Scott, 2D: factor n^(-1/6)
        sig_d, sig_c = np.sqrt(np.diag(H))

        if distance_range is None:
            lo = max(0.0, d.min() - 5 * sig_d)
            hi = d.max() + 5 * sig_d
        else:
            lo, hi = distance_range
        d_edges = np.linspace(lo, hi, n_distance + 1)
        c_edges = np.linspace(0.0, 1.0, n_cos + 1)
        dd, dc = d_edges[1] - d_edges[0], c_edges[1] - c_edges[0]
        hist, _, _ = np.histogram2d(d, c, bins=[d_edges, c_edges], weights=w)

        k_d, k_c = int(np.ceil(4 * sig_d / dd)), int(np.ceil(4 * sig_c / dc))
        X, Y = np.meshgrid(np.arange(-k_d, k_d + 1) * dd, np.arange(-k_c, k_c + 1) * dc,
                           indexing="ij")
        Hinv = np.linalg.inv(H)
        kernel = np.exp(-0.5 * (Hinv[0, 0] * X ** 2 + 2 * Hinv[0, 1] * X * Y + Hinv[1, 1] * Y ** 2))
        kernel /= kernel.sum()

        # Reflection at cos theta = 0 and 1 (and at d = 0 if the grid starts
        # there). A mirrored sample is smoothed with the *mirrored* kernel
        # (reversed along that axis, i.e. the d-cos correlation flipped), which
        # makes the estimator the exact reflection f(x) + f(Rx): the mass a
        # correlated kernel pushes past an edge comes back with the right tilt.
        m_c = int(min(n_cos, k_c + 1))
        zc = np.zeros((n_distance, m_c))
        main = np.hstack([zc, hist, zc])
        mirror_c = np.hstack([hist[:, :m_c][:, ::-1], np.zeros_like(hist), hist[:, -m_c:][:, ::-1]])
        parts = [(main, kernel), (mirror_c, kernel[:, ::-1])]
        m_d = int(min(n_distance, k_d + 1)) if lo == 0.0 else 0
        if m_d:
            pad = lambda a: np.vstack([np.zeros((m_d, a.shape[1])), a])  # noqa: E731
            mirror_d = np.vstack([main[:m_d][::-1], np.zeros_like(main)])
            parts = [(pad(main), kernel), (pad(mirror_c), kernel[:, ::-1]),
                     (mirror_d, kernel[::-1, :])]
        smooth = sum(fftconvolve(a, k, mode="same") for a, k in parts)
        smooth = smooth[m_d:m_d + n_distance, m_c:m_c + n_cos]
        dens_c = np.clip(smooth, 0.0, None) / (dd * dc)          # per Mpc per unit cos

        d_centers = 0.5 * (d_edges[1:] + d_edges[:-1])
        c_centers = 0.5 * (c_edges[1:] + c_edges[:-1])
        theta_grid = THETA_GRID_DEG if theta_grid_deg is None else np.asarray(theta_grid_deg)
        c_t = np.cos(np.radians(theta_grid))
        # linear interpolation in cos, shared by every distance row
        j = np.clip(np.searchsorted(c_centers, c_t) - 1, 0, n_cos - 2)
        f = np.clip((c_t - c_centers[j]) / dc, 0.0, 1.0)
        dens_ct = dens_c[:, j] * (1 - f) + dens_c[:, j + 1] * f
        density = dens_ct * np.sin(np.radians(theta_grid)) * np.pi / 180.0  # per Mpc per deg

        info = {"method": "kde", "bw_scale": bw_scale, "n_samples": int(d.size),
                "n_eff": float(n_eff), "kernel_sigma_distance_Mpc": float(sig_d),
                "kernel_sigma_cos_theta": float(sig_c),
                "kernel_correlation": float(H[0, 1] / (sig_d * sig_c))}
        info.update(meta or {})
        return cls(d_centers, theta_grid, density, meta=info)

    @classmethod
    def from_function(cls, distance_Mpc, theta_deg, fn, meta: Optional[dict] = None
                      ) -> "JointDistanceAngle":
        """Tabulate an (unnormalised) fn(D, T) on the grid (D, T from meshgrid 'ij')."""
        D, T = np.meshgrid(distance_Mpc, theta_deg, indexing="ij")
        return cls(distance_Mpc, theta_deg, fn(D, T), meta=meta)

    # -- marginals ----------------------------------------------------------
    def marginal_distance(self) -> Tuple[np.ndarray, np.ndarray]:
        """(d, p(d)) with theta_v integrated out."""
        return self.distance_Mpc, self._row_mass.copy()

    def marginal_theta(self) -> Tuple[np.ndarray, np.ndarray]:
        """(theta_v, p(theta_v)) with d integrated out."""
        return self.theta_deg, _trapz(self.density, self.distance_Mpc, axis=0)

    @property
    def distance_support(self) -> Tuple[float, float]:
        """Distance range where p(d) is above 1e-6 of its peak. Outside it,
        conditionals are taken at the nearest edge of this range."""
        return self._d_valid

    # -- conditionals -------------------------------------------------------
    def _rows_at(self, distance_Mpc):
        d = np.atleast_1d(np.asarray(distance_Mpc, dtype=float))
        outside = (d < self._d_valid[0]) | (d > self._d_valid[1])
        dc = np.clip(d, *self._d_valid)
        j = np.clip(np.searchsorted(self.distance_Mpc, dc) - 1, 0, self.distance_Mpc.size - 2)
        w = np.clip((dc - self.distance_Mpc[j]) /
                    (self.distance_Mpc[j + 1] - self.distance_Mpc[j]), 0.0, 1.0)
        return j, w, outside

    def conditional_theta(self, distance_Mpc: float) -> Tuple[np.ndarray, np.ndarray]:
        """(theta_v, p(theta_v | d)), normalised per degree. d outside
        `distance_support` is taken at the nearest edge of it."""
        j, w, _ = self._rows_at(distance_Mpc)
        row = (1 - w[0]) * self.density[j[0]] + w[0] * self.density[j[0] + 1]
        return self.theta_deg, row / _trapz(row, self.theta_deg)

    def conditional_distance(self, theta_deg: Optional[float] = None,
                             theta_range: Optional[Tuple[float, float]] = None
                             ) -> Tuple[np.ndarray, np.ndarray]:
        """(d, p(d | theta_v)) for a known angle: either exactly
        theta_v = `theta_deg`, or theta_v in `theta_range` = (a, b)."""
        if (theta_deg is None) == (theta_range is None):
            raise ValueError("give exactly one of theta_deg or theta_range")
        if theta_deg is not None:
            col = np.array([np.interp(theta_deg, self.theta_deg, row) for row in self.density])
        else:
            a, b = theta_range
            grid = np.linspace(a, b, max(3, int(np.ceil((b - a) / 0.1)) + 1))
            vals = np.array([np.interp(grid, self.theta_deg, row) for row in self.density])
            col = _trapz(vals, grid, axis=1)
        norm = _trapz(col, self.distance_Mpc)
        if norm <= 0:
            raise ValueError("zero probability for that angle (or angle range)")
        return self.distance_Mpc, col / norm

    def conditional_cdf(self, distance_Mpc, theta_deg) -> np.ndarray:
        """F(theta | d) = P(theta_v <= theta | d), one value per (d, theta)
        pair (vectorised, the joint interpolated linearly in d as in
        `sample_theta_given_distance`). At the samples the joint came from,
        these are the probability-integral-transform values: uniform on [0, 1]
        if the samples follow p(theta | d)."""
        d = np.atleast_1d(np.asarray(distance_Mpc, dtype=float))
        t = np.broadcast_to(np.asarray(theta_deg, dtype=float), d.shape)
        out = np.empty(d.size)
        for s in range(0, d.size, 20_000):
            j, w, _ = self._rows_at(d[s:s + 20_000])
            cum = (1 - w)[:, None] * self._cum_theta[j] + w[:, None] * self._cum_theta[j + 1]
            k = np.clip(np.searchsorted(self.theta_deg, t[s:s + 20_000]) - 1, 0, self.theta_deg.size - 2)
            rows = np.arange(j.size)
            x0, x1 = self.theta_deg[k], self.theta_deg[k + 1]
            frac = np.clip((t[s:s + 20_000] - x0) / (x1 - x0), 0.0, 1.0)
            out[s:s + 20_000] = (cum[rows, k] + frac * (cum[rows, k + 1] - cum[rows, k])) / cum[:, -1]
        return np.clip(out, 0.0, 1.0)

    # -- sampling -----------------------------------------------------------
    def sample_theta_given_distance(self, distance_Mpc, rng: Optional[np.random.Generator] = None,
                                    return_outside: bool = False):
        """One theta_v draw per entry of `distance_Mpc` (vectorised inverse CDF
        on the conditional, with the joint interpolated linearly in d)."""
        rng = rng if rng is not None else np.random.default_rng()
        d = np.atleast_1d(np.asarray(distance_Mpc, dtype=float))
        out = np.empty(d.size)
        outside = np.zeros(d.size, dtype=bool)
        for s in range(0, d.size, 20_000):  # bound the (n, n_theta) temporaries
            j, w, o = self._rows_at(d[s:s + 20_000])
            cum = (1 - w)[:, None] * self._cum_theta[j] + w[:, None] * self._cum_theta[j + 1]
            out[s:s + 20_000] = _inverse_cdf_rows(cum, self.theta_deg, rng.random(j.size))
            outside[s:s + 20_000] = o
        return (out, outside) if return_outside else out

    def sample_distance_given_theta(self, size: int, theta_deg: Optional[float] = None,
                                    theta_range: Optional[Tuple[float, float]] = None,
                                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = rng if rng is not None else np.random.default_rng()
        d, pdf = self.conditional_distance(theta_deg=theta_deg, theta_range=theta_range)
        return _sample_1d(d, pdf, size, rng)

    def sample(self, size: int, rng: Optional[np.random.Generator] = None
               ) -> Tuple[np.ndarray, np.ndarray]:
        """(d, theta_v) pairs from the joint: d from p(d), then theta_v | d."""
        rng = rng if rng is not None else np.random.default_rng()
        d = _sample_1d(self.distance_Mpc, self._row_mass, size, rng)
        return d, self.sample_theta_given_distance(d, rng=rng)

    # -- summaries ----------------------------------------------------------
    def conditional_theta_quantiles(self, quantiles=(0.05, 0.5, 0.95),
                                    distance_Mpc: Optional[np.ndarray] = None) -> np.ndarray:
        """theta_v quantiles of p(theta_v | d) along d, shape (len(d), len(quantiles)).
        The 'ridge' of the d-theta degeneracy."""
        d = self.distance_Mpc if distance_Mpc is None else np.asarray(distance_Mpc, dtype=float)
        out = np.empty((d.size, len(quantiles)))
        for i, di in enumerate(d):
            t, p = self.conditional_theta(di)
            cdf = _cumtrapz(p, t)
            out[i] = np.interp(quantiles, cdf / cdf[-1], t)
        return out

    def hpd_levels(self, fractions=(0.5, 0.9)) -> np.ndarray:
        dd = np.gradient(self.distance_Mpc)[:, None]
        dt = np.gradient(self.theta_deg)[None, :]
        return hpd_levels(self.density, dd * dt, fractions)

    # -- persistence --------------------------------------------------------
    def to_arrays(self, prefix: str = "joint_") -> dict:
        return {f"{prefix}distance_Mpc": self.distance_Mpc, f"{prefix}theta_deg": self.theta_deg,
                f"{prefix}density": self.density,
                f"{prefix}meta_json": json.dumps(self.meta, default=str)}

    @classmethod
    def from_arrays(cls, d, prefix: str = "joint_") -> "JointDistanceAngle":
        return cls(d[f"{prefix}distance_Mpc"], d[f"{prefix}theta_deg"], d[f"{prefix}density"],
                   meta=json.loads(str(d[f"{prefix}meta_json"])))


def selection_joint(horizon_Mpc: float, n_distance: int = 300,
                    theta_grid_deg: Optional[np.ndarray] = None,
                    distance_power: float = 2.0, n_mc: int = 100_000,
                    seed: int = 0) -> JointDistanceAngle:
    """p(d, theta_v) of GW-*detected* sources, uniform in volume (d^2, or
    d^`distance_power`) and isotropic in orientation:

        p(d, theta) ~ d^2 sin(theta) P_det(d / D_h, theta),   0 < d < D_h

    Its d-marginal is sin(theta) <(Theta/4)^3>, which the Schutz (2011)
    formula approximates as sin(theta) <(Theta/4)^2>^(3/2); the notebook
    checks how close the two are. p(theta | d) does not depend on the
    distance prior, only on d / D_h.

    D_h is the horizon for *this* source type and network (SNR threshold
    reached for an optimally located, face-on source) -- a population-level
    number, not something the alert provides.
    """
    theta_grid = THETA_GRID_DEG if theta_grid_deg is None else np.asarray(theta_grid_deg)
    d = np.linspace(0.0, horizon_Mpc, n_distance + 1)[1:]
    pdet = detection_probability(d / horizon_Mpc, theta_grid, n_mc=n_mc, seed=seed)
    density = (d[:, None] ** distance_power) * np.sin(np.radians(theta_grid))[None, :] * pdet
    return JointDistanceAngle(d, theta_grid, density,
                              meta={"method": "selection", "horizon_Mpc": horizon_Mpc,
                                    "distance_power": distance_power, "n_mc": n_mc})


def conditional_theta_by_resampling(theta_deg_samples, distance_Mpc_samples, distance_Mpc: float,
                                    bandwidth_Mpc: Optional[float] = None, size: int = 50_000,
                                    rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Independent cross-check of `JointDistanceAngle.conditional_theta`:
    theta_v draws from the samples themselves, Gaussian-weighted by how close
    their d is to `distance_Mpc` (no grid, no KDE in theta)."""
    rng = rng if rng is not None else np.random.default_rng()
    d = np.asarray(distance_Mpc_samples, dtype=float)
    if bandwidth_Mpc is None:
        bandwidth_Mpc = 0.1 * (np.percentile(d, 84) - np.percentile(d, 16))
    w = np.exp(-0.5 * ((d - distance_Mpc) / bandwidth_Mpc) ** 2)
    if w.sum() <= 0:
        raise ValueError(f"no samples near d={distance_Mpc:.0f} Mpc")
    idx = rng.choice(d.size, size=size, replace=True, p=w / w.sum())
    return np.asarray(theta_deg_samples, dtype=float)[idx]


# ===========================================================================
# Does theta depend on the sky position at fixed d?
# ===========================================================================

def sky_regions(ra_deg, dec_deg, n_regions: int = 3, min_fraction: float = 0.05,
                n_restarts: int = 20, n_iter: int = 100, seed: int = 0) -> np.ndarray:
    """Split posterior samples into `n_regions` contiguous sky regions:
    k-means on the unit sphere (each region is the set of samples closest in
    angle to its centre, so regions are Voronoi cells -- connected patches,
    and separate modes of a multimodal localisation end up apart). The best
    of `n_restarts` k-means++ starts is kept, among those whose smallest
    region holds at least `min_fraction` of the samples (so a handful of
    outlying samples cannot become a region of their own). Returns one
    integer label (0 .. n_regions-1) per sample, largest region first."""
    ra, dec = np.radians(np.asarray(ra_deg, dtype=float)), np.radians(np.asarray(dec_deg, dtype=float))
    v = np.column_stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    rng = np.random.default_rng(seed)
    # seeds only from the 99% of samples nearest the mean direction: k-means++
    # favours far points, and would otherwise seed on isolated outliers
    mean_dir = v.mean(axis=0) / np.linalg.norm(v.mean(axis=0))
    cos_mean = v @ mean_dir
    core = v[cos_mean >= np.quantile(cos_mean, 0.01)]
    best, best_cost = None, np.inf
    for _ in range(n_restarts):
        centres = [core[rng.integers(core.shape[0])]]
        for _ in range(1, n_regions):                   # k-means++ seeding
            dist2 = np.clip(1.0 - np.max(core @ np.array(centres).T, axis=1), 0.0, None)
            centres.append(core[rng.choice(core.shape[0], p=dist2 / dist2.sum())])
        centres = np.array(centres)
        for _ in range(n_iter):
            lab = np.argmax(v @ centres.T, axis=1)
            new = np.array([v[lab == k].sum(axis=0) if np.any(lab == k) else centres[k]
                            for k in range(n_regions)])
            new /= np.linalg.norm(new, axis=1, keepdims=True)
            if np.allclose(new, centres):
                break
            centres = new
        sizes = np.bincount(lab, minlength=n_regions)
        cost = np.sum(1.0 - np.max(v @ centres.T, axis=1))
        if sizes.min() >= min_fraction * v.shape[0] and cost < best_cost:
            best, best_cost = lab, cost
    if best is None:
        raise ValueError(f"no split into {n_regions} sky regions of >= {min_fraction:.0%} "
                         "of the samples each; use fewer regions")
    order = np.argsort(-np.bincount(best, minlength=n_regions))
    return np.argsort(order)[best]


def _weighted_quantile(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(np.interp(q * c[-1], c, x[o]))


def sky_dependence_check(theta_deg, distance_Mpc, regions, joint: Optional[JointDistanceAngle] = None,
                         theta_cut_deg: float = 20.0, n_perm: int = 300, n_distance_bins: int = 40,
                         rng: Optional[np.random.Generator] = None) -> dict:
    """Test whether p(theta_v | d) is the same in every sky region, i.e.
    whether the sky-marginalised conditional p(theta | d) may stand in for the
    per-sky-position p(theta | d, sky) of the strict chain
    sky -> d | sky -> theta | d, sky.

    `regions`: one integer label per posterior sample (e.g. from
    `sky_regions`, or "inside / outside the IACT pointings"). `joint`: the
    pooled KDE joint of the same samples (built here if None).

    The samples are cut into `n_distance_bins` equal-count distance bins.
    Per region r:

    - *exact*: the region's own samples -- draws of p(theta | d, r);
    - *shortcut*: all samples, each weighted by the fraction of its distance
      bin that lies in r -- the pooled p(theta | d) taken at the region's own
      distances, i.e. what drawing theta from the sky-marginalised conditional
      gives there. Grid-free, so the KDE smoothing plays no part;
    - their differences in median theta and in P(theta < theta_cut) are what
      the shortcut costs in that region;
    - formal test: u_k = F_pooled(theta_k | d_k) from the KDE grid; with no
      sky dependence, u is uniform in every region. Statistic: sqrt(n) x the
      KS distance of the region's u from uniform.

    Null distribution (no sky dependence at fixed d): region labels permuted
    within each distance bin, `n_perm` times. It gives the p-values and the
    noise level (sd) of the shifts. Any KDE bias in u affects observed and
    permuted regions alike, so it does not fake a signal.
    Returns {"regions": [one dict per region], "p_value_global", "u",
    "shortcut_weights" (n_regions, n_samples), ...}.
    """
    rng = rng if rng is not None else np.random.default_rng()
    t = np.asarray(theta_deg, dtype=float)
    d = np.asarray(distance_Mpc, dtype=float)
    lab = np.asarray(regions)
    labels = np.unique(lab)
    if joint is None:
        joint = JointDistanceAngle.from_samples(d, t)
    u = joint.conditional_cdf(d, t)
    below = t < theta_cut_deg

    def stats(lab_):
        out = np.empty((labels.size, 3))
        for i, r in enumerate(labels):
            m = lab_ == r
            ur = np.sort(u[m])
            n = ur.size
            ks = max(np.max(np.arange(1, n + 1) / n - ur), np.max(ur - np.arange(n) / n))
            out[i] = (np.sqrt(n) * ks, np.median(t[m]), below[m].mean())
        return out

    dbin = np.searchsorted(np.quantile(d, np.linspace(0, 1, n_distance_bins + 1)[1:-1]), d)
    n_bin = np.bincount(dbin, minlength=n_distance_bins)
    weights = np.stack([np.bincount(dbin[lab == r], minlength=n_distance_bins)[dbin] / n_bin[dbin]
                        for r in labels])
    weights /= weights.sum(axis=1, keepdims=True)

    obs = stats(lab)
    groups = [np.flatnonzero(dbin == b) for b in range(n_distance_bins)]
    null = np.empty((n_perm,) + obs.shape)
    for p in range(n_perm):
        perm = lab.copy()
        for g in groups:
            perm[g] = lab[rng.permutation(g)]
        null[p] = stats(perm)

    rows = []
    for i, r in enumerate(labels):
        m = lab == r
        med_sc = _weighted_quantile(t, weights[i], 0.5)
        p_sc = float(np.sum(weights[i] * below))
        rows.append({
            "region": int(r), "n": int(m.sum()),
            "distance_Mpc_median": float(np.median(d[m])),
            "theta_median_exact": float(obs[i, 1]),
            "theta_median_shortcut": med_sc,
            "median_shift_deg": float(obs[i, 1] - med_sc),
            "median_shift_null_sd": float(np.std(null[:, i, 1])),
            "p_below_cut_exact": float(obs[i, 2]),
            "p_below_cut_shortcut": p_sc,
            "p_below_cut_shift": float(obs[i, 2] - p_sc),
            "p_below_cut_shift_null_sd": float(np.std(null[:, i, 2])),
            "ks_sqrt_n": float(obs[i, 0]),
            "p_value": float((1 + np.sum(null[:, i, 0] >= obs[i, 0])) / (n_perm + 1)),
        })
    glob_null = null[:, :, 0].max(axis=1)
    return {"regions": rows, "labels": lab, "u": u, "shortcut_weights": weights,
            "theta_cut_deg": theta_cut_deg,
            "p_value_global": float((1 + np.sum(glob_null >= obs[:, 0].max())) / (n_perm + 1)),
            "n_perm": n_perm, "n_distance_bins": n_distance_bins}


# ===========================================================================
# The standardized object
# ===========================================================================

@dataclass
class ThetaDistribution:
    """One standardized way to draw/evaluate theta_v, whatever information
    was actually available for this GW alert. Build with
    `make_theta_distribution` rather than the constructor directly."""

    kind: str
    joint: Optional[JointDistanceAngle] = None           # "pe", "selection"
    samples: Optional[Dict[str, np.ndarray]] = None       # "pe": for joint draws
    theta_fixed_deg: Optional[float] = None
    threshold_deg: float = 45.0
    p_below: float = 0.5
    horizon_Mpc: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def depends_on_distance(self) -> bool:
        return self.kind in ("pe", "selection")

    # -- drawing -------------------------------------------------------------
    def sample(self, size: int = 1, distance_Mpc=None,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """theta_v draws in degrees. For kinds whose theta depends on d
        ("pe", "selection") `distance_Mpc` is required: a scalar, or one
        distance per draw (array of length `size`) -- p(theta | d_i). It is
        ignored by the d-independent kinds."""
        rng = rng if rng is not None else np.random.default_rng()
        if self.kind == "fixed":
            return np.full(size, self.theta_fixed_deg, dtype=float)
        if self.kind == "isotropic":
            return np.degrees(np.arccos(1.0 - rng.random(size)))
        if self.kind == "schutz":
            return _sample_1d(_FINE_THETA_DEG, schutz_pdf(_FINE_THETA_DEG), size, rng)
        if self.kind == "two_bin":
            below = rng.random(size) < self.p_below
            theta = np.where(below, rng.uniform(0.0, self.threshold_deg, size),
                             rng.uniform(self.threshold_deg, 90.0, size))
            return theta
        if self.kind in ("pe", "selection"):
            if distance_Mpc is None:
                raise ValueError(f"kind={self.kind!r} draws theta | d: pass distance_Mpc "
                                 "(one per draw, or a scalar)")
            d = np.broadcast_to(np.asarray(distance_Mpc, dtype=float), (size,))
            theta, outside = self.joint.sample_theta_given_distance(d, rng=rng, return_outside=True)
            if outside.any():
                lo, hi = self.joint.distance_support
                warnings.warn(
                    f"{outside.sum()} of {size} distances outside the {self.kind!r} support "
                    f"[{lo:.0f}, {hi:.0f}] Mpc; theta drawn from p(theta | d) at the nearest "
                    "edge of it.", stacklevel=2)
            return theta
        raise ValueError(f"unknown ThetaDistribution.kind={self.kind!r}")

    def sample_joint(self, size: int, rng: Optional[np.random.Generator] = None) -> dict:
        """(ra, dec, d, theta_v) drawn *together* -- the guidelines' best
        option when PE samples exist ("draw one sample index and take all four
        values together"), which keeps every correlation and replaces Steps
        4.1-4.3. For "selection", only d and theta_v (from the grid)."""
        rng = rng if rng is not None else np.random.default_rng()
        if self.kind == "pe":
            n = self.samples["distance_Mpc"].size
            idx = rng.integers(0, n, size)
            return {k: v[idx] for k, v in self.samples.items()}
        if self.kind == "selection":
            d, t = self.joint.sample(size, rng=rng)
            return {"distance_Mpc": d, "theta_deg": t}
        raise ValueError(f"sample_joint needs a joint distribution; kind={self.kind!r} has none")

    # -- evaluating ----------------------------------------------------------
    def pdf_grid(self, distance_Mpc: Optional[float] = None,
                 theta_grid_deg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(theta_grid, p(theta_v)), per degree, integrating to 1. For "pe" and
        "selection": p(theta_v | d) at `distance_Mpc`, or the marginal
        p(theta_v) if `distance_Mpc` is None. "fixed" is drawn as a one-cell
        spike of unit mass."""
        grid = THETA_GRID_DEG if theta_grid_deg is None else np.asarray(theta_grid_deg, dtype=float)
        if self.kind == "fixed":
            pdf = np.zeros_like(grid)
            k = int(np.argmin(np.abs(grid - self.theta_fixed_deg)))
            pdf[k] = 1.0 / np.gradient(grid)[k]
            return grid, pdf
        if self.kind == "isotropic":
            return grid, isotropic_pdf(grid)
        if self.kind == "schutz":
            return grid, schutz_pdf(grid)
        if self.kind == "two_bin":
            pdf = np.where(grid < self.threshold_deg, self.p_below / self.threshold_deg,
                           (1.0 - self.p_below) / (90.0 - self.threshold_deg))
            return grid, pdf
        if self.kind in ("pe", "selection"):
            t, p = (self.joint.marginal_theta() if distance_Mpc is None
                    else self.joint.conditional_theta(distance_Mpc))
            return grid, np.interp(grid, t, p)
        raise ValueError(f"unknown ThetaDistribution.kind={self.kind!r}")

    def cdf_grid(self, distance_Mpc: Optional[float] = None,
                 theta_grid_deg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(theta_grid, P(theta_v <= theta)); exact step for "fixed"."""
        grid = THETA_GRID_DEG if theta_grid_deg is None else np.asarray(theta_grid_deg, dtype=float)
        if self.kind == "fixed":
            return grid, (grid >= self.theta_fixed_deg).astype(float)
        if self.kind == "isotropic":
            return grid, 1.0 - np.cos(np.radians(np.clip(grid, 0, 90)))
        _, pdf = self.pdf_grid(distance_Mpc=distance_Mpc, theta_grid_deg=_FINE_THETA_DEG)
        cdf = _cumtrapz(pdf, _FINE_THETA_DEG)
        return grid, np.interp(grid, _FINE_THETA_DEG, cdf / cdf[-1])

    def prob_below(self, theta_deg: float, distance_Mpc: Optional[float] = None) -> float:
        """P(theta_v < theta_deg), e.g. the chance of seeing the jet within its core."""
        if self.kind == "fixed":  # strict: a jet fixed at 20 deg is not "below 20 deg"
            return float(self.theta_fixed_deg < theta_deg)
        _, cdf = self.cdf_grid(distance_Mpc=distance_Mpc, theta_grid_deg=np.array([theta_deg]))
        return float(cdf[0])


def make_theta_distribution(mode: str, **kwargs) -> ThetaDistribution:
    """Factory for every Step 4.3 case. `mode`:

    - "pe": kwarg `pe` (a gwuls.gw_pe.PESamples; all its analyses are mixed
      with equal weight, as stored), or `theta_deg_samples` +
      `distance_Mpc_samples` (+ optional `ra_deg_samples`, `dec_deg_samples`).
      Optional `bw_scale` (KDE bandwidth multiplier, default 1), `n_distance`.
    - "selection": kwarg `horizon_Mpc` (required), optional `distance_power`.
    - "schutz", "isotropic": no kwargs.
    - "fixed": kwarg `theta_deg`.
    - "two_bin": optional `threshold_deg` (default 45), `p_below` (default 0.5).
    """
    mode = mode.lower()
    meta = dict(kwargs.get("meta", {}))
    if mode == "pe":
        pe = kwargs.get("pe")
        if pe is not None:
            samples = {"distance_Mpc": pe.get("distance_Mpc"),
                       "theta_deg": pe.get("viewing_angle_deg"),
                       "ra_deg": pe.get("ra_deg"), "dec_deg": pe.get("dec_deg")}
            meta.update({"superevent_id": pe.superevent_id, "gw_name": pe.gw_name,
                         "catalog": pe.catalog, "analyses": list(pe.analyses),
                         "pe_file": pe.meta.get("path")})
        else:
            samples = {"distance_Mpc": np.asarray(kwargs["distance_Mpc_samples"], dtype=float),
                       "theta_deg": np.asarray(kwargs["theta_deg_samples"], dtype=float)}
            for k in ("ra_deg", "dec_deg"):
                if f"{k}_samples" in kwargs:
                    samples[k] = np.asarray(kwargs[f"{k}_samples"], dtype=float)
        if samples["distance_Mpc"].shape != samples["theta_deg"].shape:
            raise ValueError("theta and distance samples must have the same shape")
        joint = JointDistanceAngle.from_samples(
            samples["distance_Mpc"], samples["theta_deg"],
            bw_scale=kwargs.get("bw_scale", 1.0), n_distance=kwargs.get("n_distance", 300))
        return ThetaDistribution(kind="pe", joint=joint, samples=samples, meta=meta)
    if mode == "selection":
        horizon = float(kwargs["horizon_Mpc"])
        joint = selection_joint(horizon, distance_power=kwargs.get("distance_power", 2.0))
        return ThetaDistribution(kind="selection", joint=joint, horizon_Mpc=horizon, meta=meta)
    if mode == "schutz":
        return ThetaDistribution(kind="schutz", meta=meta)
    if mode == "isotropic":
        return ThetaDistribution(kind="isotropic", meta=meta)
    if mode == "fixed":
        return ThetaDistribution(kind="fixed", theta_fixed_deg=float(kwargs["theta_deg"]), meta=meta)
    if mode == "two_bin":
        threshold = float(kwargs.get("threshold_deg", 45.0))
        p_below = float(kwargs.get("p_below", 0.5))
        if not (0.0 < threshold < 90.0):
            raise ValueError("threshold_deg must be in (0, 90)")
        if not (0.0 <= p_below <= 1.0):
            raise ValueError("p_below must be in [0, 1]")
        return ThetaDistribution(kind="two_bin", threshold_deg=threshold, p_below=p_below, meta=meta)
    raise ValueError(f"unknown mode {mode!r}; expected one of {KINDS}")


# ===========================================================================
# Persistence -- one small, standardized cache per GW source
# ===========================================================================

def _nan_if_none(x):
    return np.nan if x is None else x


def save_theta_distribution(dist: ThetaDistribution, path) -> None:
    """Write `dist` to `path` (see gwuls.paths.theta_distribution_path)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "kind": dist.kind,
        "threshold_deg": dist.threshold_deg,
        "p_below": dist.p_below,
        "theta_fixed_deg": _nan_if_none(dist.theta_fixed_deg),
        "horizon_Mpc": _nan_if_none(dist.horizon_Mpc),
        "meta_json": json.dumps(dist.meta, default=str),
    }
    if dist.joint is not None:
        payload.update(dist.joint.to_arrays())
    if dist.samples is not None:
        payload.update({f"samples_{k}": v for k, v in dist.samples.items()})
    np.savez_compressed(path, **payload)


def load_theta_distribution(path) -> ThetaDistribution:
    """Inverse of `save_theta_distribution` -- sim_3d's single entry point,
    regardless of which Step 4.3 case produced the cached file."""
    d = np.load(path, allow_pickle=False)
    kind = str(d["kind"])
    if kind not in KINDS:
        raise ValueError(f"{path}: kind {kind!r} is not one of {KINDS} -- written by an "
                         "older version of this module? Re-run setup_angle_distribution.ipynb.")

    def opt(key):
        v = float(d[key]) if key in d else np.nan
        return None if np.isnan(v) else v

    samples = {k[len("samples_"):]: d[k] for k in d.files if k.startswith("samples_")} or None
    return ThetaDistribution(
        kind=kind,
        joint=JointDistanceAngle.from_arrays(d) if "joint_density" in d else None,
        samples=samples,
        theta_fixed_deg=opt("theta_fixed_deg"),
        threshold_deg=float(d["threshold_deg"]),
        p_below=float(d["p_below"]),
        horizon_Mpc=opt("horizon_Mpc"),
        meta=json.loads(str(d["meta_json"])),
    )
