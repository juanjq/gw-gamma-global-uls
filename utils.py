import numpy as np
import glob, os
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import healpy as hp
from scipy.optimize import root_scalar

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
    data_ligo_hp, dec_hp, ra_hp, hp_area, bin_edges_ra, bin_edges_dec, bin_area
):
    """Integrate HEALPix probability map onto WCS bins."""
    mask_hp_wcs = (
        (-(((ra_hp + 180) % 360) - 180) >= bin_edges_ra[:,-1:].min()) &
        (dec_hp >= bin_edges_dec[:1,:].max()) &
        (-(((ra_hp + 180) % 360) - 180) <= bin_edges_ra[:,:1].min()) &
        (dec_hp <= bin_edges_dec[-1:,:].max())
    )
    data_ligo_hp_wcs = data_ligo_hp[mask_hp_wcs]
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
            p = np.sum(data_ligo_hp_wcs[mask_bin]) * (bin_area[i,j].value / (np.sum(mask_bin) * hp_area))
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


def integrate_dirac_delta_on_wcs(data_ligo_hp, dec_hp, ra_hp, bin_edges_ra, bin_edges_dec):
    """Place Dirac delta probability into the single WCS bin containing the hot pixel."""
    hot_ipix = np.argmax(data_ligo_hp)
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


def compute_threshold_masks(threshold_maps, ra_grid, dec_grid, data_ligo_2d, bin_c_ra, bin_c_dec, prob_gw_integrated):
    """Compute boolean masks for each GW probability contour level."""
    masks = []
    for level in threshold_maps:
        cs    = plt.contour(np.rad2deg(ra_grid), np.rad2deg(dec_grid),
                            np.flip(data_ligo_2d, axis=1), levels=[level])
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
