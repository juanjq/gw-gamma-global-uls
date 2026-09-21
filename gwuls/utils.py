"""HEALPix / GW-map / distance-CDF helpers shared by the 2D and 3D notebooks."""

import numpy as np
import glob, os
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import healpy as hp
from scipy.optimize import root_scalar
import json, hashlib
from ligo.skymap.distance import marginal_cdf, marginal_pdf

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def add_bkg(data_store, obs_id, dir_dl3, dim_bkg, bkg_type):
    fname = glob.glob(os.path.join(dir_dl3, f"bkg_{bkg_type}_{dim_bkg}d_{str(obs_id)}.fits"))[0]
    hdul  = fits.open(fname)

    data_store.hdu_table.add_row({
        "OBS_ID"   : obs_id,
        "HDU_TYPE" : "bkg",
        "HDU_CLASS" : f"bkg_{dim_bkg}d",
        "FILE_DIR"  : ".",
        "FILE_NAME" : os.path.basename(fname),
        "HDU_NAME"  : "BACKGROUND",
        "SIZE"      : hdul["BACKGROUND"].size,
    })
    return data_store


# ---------------------------------------------------------------------------
# HEALPix helpers
# ---------------------------------------------------------------------------

def IndexToDeclRa(index, nside):
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.), np.degrees(np.pi * 2. - phi)


def DeclRaToIndex(decl, ra, nside):
    return hp.pixelfunc.ang2pix(
        nside, np.radians(-decl + 90.),
        np.radians(360. - ra)
    )

def healpix2map(healpix_data, ra_bins, dec_bins):
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)
    theta, phi   = np.radians(90 - dec_grid), np.radians(ra_grid)
    nside        = hp.npix2nside(len(healpix_data))
    hp_indices   = hp.ang2pix(nside, theta, phi)
    return healpix_data[hp_indices]


def get_hp_map_thresholds(healpix_data, threshold_percent=[0.9, 0.68]):
    threshold_percent = np.sort(threshold_percent)[::-1]
    sorted_data    = np.sort(healpix_data)[::-1] / np.sum(healpix_data)
    cumulative_sum = np.cumsum(sorted_data)
    indexes_map    = [np.searchsorted(cumulative_sum, t) for t in threshold_percent]
    threshold_maps = [sorted_data[min(idx, len(sorted_data) - 1)] for idx in indexes_map]
    return threshold_maps


def get_2d_map_hotspot(map_data_2d, ra_bins, dec_bins):
    max_prob_index = np.unravel_index(np.argmax(map_data_2d), map_data_2d.shape)
    max_prob_ra    = ra_bins[max_prob_index[1]]
    max_prob_dec   = dec_bins[max_prob_index[0]]
    return SkyCoord(ra=max_prob_ra, dec=max_prob_dec, unit=u.deg, frame="icrs")

def integrate_hp_on_wcs(
    lvk_prob_hp, dec_hp, ra_hp, hp_area, bin_edges_ra, bin_edges_dec, bin_area
):
    """Integrate HEALPix probability map onto WCS bins."""
    mask_hp_wcs = (
        (-(((ra_hp + 180) % 360) - 180) >= bin_edges_ra[:,-1:].min()) &
        (dec_hp >= bin_edges_dec[:1,:].max()) &
        (-(((ra_hp + 180) % 360) - 180) <= bin_edges_ra[:,:1].min()) &
        (dec_hp <= bin_edges_dec[-1:,:].max())
    )
    lvk_prob_hp_wcs = lvk_prob_hp[mask_hp_wcs]
    dec_hp_wcs       = dec_hp[mask_hp_wcs]
    ra_hp_wcs        = -(ra_hp[mask_hp_wcs] + 180) % 360 - 180
    prob       = np.zeros((len(bin_edges_ra)-1, len(bin_edges_dec)-1))
    num_pix    = np.zeros((len(bin_edges_ra)-1, len(bin_edges_dec)-1))
    mask_added = np.zeros(len(ra_hp_wcs), dtype=bool)
    for i in range(len(bin_edges_ra)-1):
        for j in range(len(bin_edges_ra[i])-1):
            mask_bin = (
                (ra_hp_wcs  >= bin_edges_ra[i,j+1])  &
                (dec_hp_wcs >= bin_edges_dec[i,j])    &
                (ra_hp_wcs  <= bin_edges_ra[i,j])     &
                (dec_hp_wcs <= bin_edges_dec[i+1,j+1])
            ) & ~mask_added
            p = np.sum(lvk_prob_hp_wcs[mask_bin]) * (bin_area[i,j].value / (np.sum(mask_bin) * hp_area))
            prob[i,j]    = 0.0 if np.isnan(p) else p
            num_pix[i,j] = np.sum(mask_bin)
            mask_added  |= mask_bin
    return prob, num_pix
    
# ---------------------------------------------------------------------------
# Dirac-delta / WCS integration
# ---------------------------------------------------------------------------

def make_dirac_delta_hp(coord: SkyCoord, nside: int) -> np.ndarray:
    """HEALPix map with probability=1 at the pixel closest to coord."""
    data_hp = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    data_hp[hp.ang2pix(nside, np.pi/2 - coord.dec.rad, coord.ra.rad)] = 1.0
    return data_hp


def make_dirac_delta_2d(coord, ra_bins, dec_bins):
    """2D grid map with probability=1 at the pixel closest to coord."""
    data_2d = np.zeros((len(dec_bins), len(ra_bins)), dtype=np.float64)
    data_2d[
        np.argmin(np.abs(dec_bins - coord.dec.deg)),
        np.argmin(np.abs(ra_bins  - coord.ra.wrap_at('180d').deg))
    ] = 1.0
    return data_2d


def integrate_dirac_delta_on_wcs(lvk_prob_hp, dec_hp, ra_hp, bin_edges_ra, bin_edges_dec):
    """Place Dirac delta probability into the single WCS bin containing the hot pixel."""
    hot_ipix = np.argmax(lvk_prob_hp)
    src_ra   = -(ra_hp[hot_ipix] + 180) % 360 - 180
    src_dec  = dec_hp[hot_ipix]
    prob     = np.zeros((len(bin_edges_ra)-1, len(bin_edges_dec)-1))
    num_pix  = np.zeros((len(bin_edges_ra)-1, len(bin_edges_dec)-1))
    for i in range(len(bin_edges_ra)-1):
        for j in range(len(bin_edges_ra[i])-1):
            if (bin_edges_ra[i,j+1] <= src_ra  <= bin_edges_ra[i,j] and
                bin_edges_dec[i,j]   <= src_dec <= bin_edges_dec[i+1,j+1]):
                prob[i,j], num_pix[i,j] = 1.0, 1
                return prob, num_pix
    raise ValueError(f"Hot pixel ra={src_ra:.4f}° dec={src_dec:.4f}° outside all WCS bins!")

# ---------------------------------------------------------------------------
# 3D GW: distance distribution per WCS bin
# ---------------------------------------------------------------------------

def map_hp_pixels_to_wcs_bins(dec_hp, ra_hp, bin_edges_ra, bin_edges_dec,
                              fill_empty=True, nside=None):
    """
    Assign each HEALPix pixel to a WCS bin, with the SAME conventions/ordering
    as integrate_hp_on_wcs (each pixel used at most once, first bin wins).

    Returns
    -------
    bin_pixels : object array, shape (n_i, n_j)
        bin_pixels[i, j] -> np.ndarray of HEALPix indices falling in that bin.
    """
    ipix_all = np.arange(len(ra_hp))
    mask_hp_wcs = (
        (-(((ra_hp + 180) % 360) - 180) >= bin_edges_ra[:, -1:].min()) &
        (dec_hp >= bin_edges_dec[:1, :].max()) &
        (-(((ra_hp + 180) % 360) - 180) <= bin_edges_ra[:, :1].min()) &
        (dec_hp <= bin_edges_dec[-1:, :].max())
    )
    ipix       = ipix_all[mask_hp_wcs]
    dec_hp_wcs = dec_hp[mask_hp_wcs]
    ra_hp_wcs  = -(ra_hp[mask_hp_wcs] + 180) % 360 - 180

    n_i, n_j   = len(bin_edges_ra) - 1, len(bin_edges_ra[0]) - 1
    bin_pixels = np.empty((n_i, n_j), dtype=object)
    mask_added = np.zeros(len(ipix), dtype=bool)

    for i in range(n_i):
        for j in range(n_j):
            mask_bin = (
                (ra_hp_wcs  >= bin_edges_ra[i, j+1])  &
                (dec_hp_wcs >= bin_edges_dec[i, j])   &
                (ra_hp_wcs  <= bin_edges_ra[i, j])    &
                (dec_hp_wcs <= bin_edges_dec[i+1, j+1])
            ) & ~mask_added
            bin_pixels[i, j] = ipix[mask_bin]
            mask_added |= mask_bin

    # WCS bins smaller than a HEALPix pixel can end up empty -> nearest pixel
    if fill_empty:
        if nside is None:
            raise ValueError("nside required when fill_empty=True")
        empty = [(i, j) for i in range(n_i) for j in range(n_j)
                 if bin_pixels[i, j].size == 0]
        if empty:
            ii  = np.array([e[0] for e in empty])
            jj  = np.array([e[1] for e in empty])
            # bin centres from the edges (same wrapping as above)
            c_ra  = 0.5 * (bin_edges_ra[ii, jj]  + bin_edges_ra[ii, jj+1])
            c_dec = 0.5 * (bin_edges_dec[ii, jj] + bin_edges_dec[ii+1, jj+1])
            ipix_near = hp.ang2pix(nside, np.radians(90 - c_dec),
                                   np.radians(-c_ra % 360))
            for k, (i, j) in enumerate(empty):
                bin_pixels[i, j] = np.array([ipix_near[k]])
    return bin_pixels


def compute_distance_cdf_table(bin_pixels, prob_hp, distmu_hp, distsigma_hp,
                               distnorm_hp, r_grid, normalize=True, dtype=np.float32):
    """
    Marginal (mixture) distance CDF for every WCS bin.

    For each bin, the per-pixel conditional distance PDFs p_i(r) are weighted by
    prob_hp[i] and summed -> a single CDF over r_grid.

    normalize=True  -> CDF of the distance GIVEN the bin (ends at 1).
                       Joint 3D prob = prob_gw_integrated[i,j] * cdf[i,j]
    normalize=False -> CDF scaled by the total prob of the bin's pixels.
    """
    n_i, n_j = bin_pixels.shape
    cdf_table = np.zeros((n_i, n_j, len(r_grid)), dtype=dtype)
    valid     = np.zeros((n_i, n_j), dtype=bool)

    for i in range(n_i):
        for j in range(n_j):
            idx = bin_pixels[i, j]
            if idx.size == 0:
                continue
            p = np.asarray(prob_hp[idx], dtype=np.float64)
            # pixels with no distance information carry distmu = inf
            good = np.isfinite(distmu_hp[idx]) & (p > 0)
            if not good.any():
                continue
            idx, p = idx[good], p[good]
            ptot = p.sum()
            if not np.isfinite(ptot) or ptot <= 0:
                continue
            w = p / ptot if normalize else p
            cdf_table[i, j] = marginal_cdf(r_grid, w, distmu_hp[idx],
                                           distsigma_hp[idx], distnorm_hp[idx])
            valid[i, j] = True
    return cdf_table, valid


# --- cache -----------------------------------------------------------------

def distance_cdf_cache_key(file_lvk, nside, bin_edges_ra, bin_edges_dec, r_grid,
                           normalize=True, dirac=False):
    """Stable hash of everything the table depends on."""
    h = hashlib.sha1()
    for a in (np.asarray(bin_edges_ra, dtype=np.float64),
              np.asarray(bin_edges_dec, dtype=np.float64),
              np.asarray(r_grid, dtype=np.float64)):
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(json.dumps({"file": os.path.basename(file_lvk), "nside": int(nside),
                         "norm": bool(normalize), "dirac": bool(dirac)},
                        sort_keys=True).encode())
    return h.hexdigest()[:12]


def save_distance_cdf_table(path, r_grid, cdf_table, valid, meta):
    np.savez_compressed(path, r_grid=r_grid, cdf_table=cdf_table,
                        valid=valid, meta=json.dumps(meta))


def load_distance_cdf_table(path, expected_key=None):
    """Return (r_grid, cdf_table, valid, meta) or None if missing/stale."""
    if not os.path.isfile(path):
        return None
    d    = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    if expected_key is not None and meta.get("key") != expected_key:
        return None
    return d["r_grid"], d["cdf_table"], d["valid"], meta


# --- using the table -------------------------------------------------------

def distance_quantile(r_grid, cdf_bin, q):
    """Inverse CDF (r_grid and cdf_bin 1D). q scalar or array in [0, 1]."""
    c = np.asarray(cdf_bin, dtype=np.float64)
    c = np.maximum.accumulate(c)                      # enforce monotonicity
    if c[-1] <= 0:
        return np.full(np.shape(q), np.nan)
    return np.interp(q, c / c[-1], r_grid)


def sample_distances(r_grid, cdf_bin, size=1, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    return distance_quantile(r_grid, cdf_bin, rng.random(size))


def distance_map_from_cdf(r_grid, cdf_table, valid, q=0.5):
    """2D map of a given quantile (median by default) for plotting/sanity checks."""
    n_i, n_j = valid.shape
    out = np.full((n_i, n_j), np.nan)
    for i in range(n_i):
        for j in range(n_j):
            if valid[i, j]:
                out[i, j] = distance_quantile(r_grid, cdf_table[i, j], q)
    return out
    
# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def extract_geom_coords(geom):
    """Extract coordinate arrays and derived quantities from a WcsGeom."""
    geom_image    = geom.to_image()
    centers       = geom.get_coord(mode="center")
    edges         = geom.get_coord(mode="edges")
    bin_c_ra,     bin_c_dec     = centers.lon[0].value, centers.lat[0].value
    bin_edges_ra, bin_edges_dec = edges.lon[0].value,   edges.lat[0].value
    bin_area      = geom_image.solid_angle()
    coord_array   = SkyCoord(bin_c_ra, bin_c_dec, unit=u.deg)
    mid           = coord_array.shape[0] // 2
    coord_center  = coord_array[mid, mid]
    separations_map = coord_array.separation(SkyCoord(*geom.center_coord)).deg
    return bin_c_ra, bin_c_dec, bin_edges_ra, bin_edges_dec, bin_area, coord_array, coord_center, separations_map


def compute_threshold_masks(threshold_maps, ra_grid, dec_grid, lvk_prob_2d, bin_c_ra, bin_c_dec, prob_gw_integrated):
    """Compute boolean masks for each GW probability contour level."""
    masks = []
    for level in threshold_maps:
        cs    = plt.contour(np.rad2deg(ra_grid), np.rad2deg(dec_grid),
                            np.flip(lvk_prob_2d, axis=1), levels=[level])
        paths = cs.get_paths()
        plt.close()
        ra_wrapped = -((bin_c_ra + 180) % 360) + 180
        mask = np.array([
            [any(p.contains_point((ra_wrapped[i, j], bin_c_dec[i, j])) for p in paths)
             for j in range(bin_c_ra.shape[1])]
            for i in range(bin_c_ra.shape[0])
        ])
        masks.append(mask)
    return masks


# ---------------------------------------------------------------------------
# Math / statistical helpers
# ---------------------------------------------------------------------------

def sigmoid(x, L, x0, k, b):
    """
    L:  curve's maximum value (upper plateau)
    x0: the x-value of the sigmoid's midpoint
    k:  the steepness of the curve
    b:  the minimum value (lower plateau)
    """
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))


def find_x_at_y(y_target, params):
    a, x0, k, y0 = params
    func   = lambda x: sigmoid(x, a, x0, k, y0) - y_target
    result = root_scalar(func, bracket=[x0 - 10, x0 + 10], method="brentq")
    return result.root if result.converged else np.nan


def find_amplitude_at_cl(x, y, cl=0.95):
    x, y = np.array(x), np.array(y)
    s    = np.argsort(x); x, y = x[s], y[s]
    a    = np.where(y >= cl)[0]
    if len(a) == 0:
        return np.nan
    i = a[0]
    if i == 0:
        return 10**x[0]
    x1, x2, y1, y2 = x[i-1], x[i], y[i-1], y[i]
    return 10**(x1 + (cl - y1) * (x2 - x1) / (y2 - y1))
