import numpy as np
import matplotlib.pyplot as plt
import glob, os, pickle, sys
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle
import healpy as hp
from gammapy.stats.fit_statistics import cash
from gammapy.estimators.map.excess import convolved_map_dataset_counts_statistics

from gammapy.modeling.models import (
    PointSpatialModel, PowerLawSpectralModel, ConstantTemporalModel, SkyModel, Models
)

def IndexToDeclRa(index, nside):
    import healpy as hp
    theta, phi = hp.pixelfunc.pix2ang(nside, index)
    return -np.degrees(theta - np.pi / 2.), np.degrees(np.pi * 2. - phi)

def DeclRaToIndex(decl, ra, nside):
    import healpy as hp
    return hp.pixelfunc.ang2pix(
        nside, np.radians(-decl + 90.),
        np.radians(360. - ra)
    )

def perform_n_simulations(n_sim, flux, file_input, file_output, compute_uls=0):
    '''
    n_sim (int):
        Number of iterations of each simulation
    flux (float): 
        The PWL amplitude parameter in (cm-2 s-1 TeV-1) units.
    file_input (str): 
        Path to the input .pkl file, containing:
            * ExcessMapEstimator
            * dataset_empty
            * GW sky-map in WCS
            * GW sky-map in HealPix
            * Mask 95% of the GW
    file_output (str): 
        Path to the output .npz file
    compute_uls (int): 
        If flux UL distributions need to be computed or not.
        takes some time so only use this to debug. True=1, False=0
    '''
    
    n_sim, compute_uls = int(n_sim), bool(compute_uls)
    print(f"Producing {n_sim} simulations {'without ' if not compute_uls else ''}computing the flux maps.")
    print("Setting up everything...\n")
    # Reading the input parameters
    with open(file_input, "rb") as file:
        dataset, excess_estimator, prob_gw, data_ligo_hp, mask_threshold = pickle.load(file)
    
    # Simulating n_sim source positions following GW sky-map
    pix_indices, nside = np.arange(len(data_ligo_hp)), hp.npix2nside(len(data_ligo_hp))
    dec_hp, ra_hp = IndexToDeclRa(pix_indices, nside)
    indices_hp = np.random.choice(len(data_ligo_hp), size=n_sim, p=data_ligo_hp)
    ra_sim, dec_sim = ra_hp[indices_hp], dec_hp[indices_hp]
    ra_sim = [-(((ra_sim[i] + 180) % 360) - 180) for i in range(len(ra_sim))]
    
    # Getting some details on the geometry
    geom = dataset.geoms["geom"]
    geom_centers = geom.get_coord(mode="center")
    bin_c_ra, bin_c_dec = geom_centers.lon[0].value, geom_centers.lat[0].value

    # Variables to be filled in iterations
    lambda_data, lambda_ra, lambda_dec = [], [], []
    tsmax, tsmax_ra, tsmax_dec = [], [], []
    ts_dist, ts2_dist = [], []
    if compute_uls:
        ulmax, ulmax_ra, ulmax_dec, ul_dist = [], [], [], []
        
    for i in range(n_sim):
        print(f"Computing... {i+1}/{n_sim}") if ((i+1) % (5 if compute_uls else 250) == 0) else None
        
        # The model of the source
        sim_coord = SkyCoord(ra=ra_sim[i], dec=dec_sim[i], unit=u.deg, frame="icrs")
        model_source = Models([SkyModel(
            spatial_model = PointSpatialModel.from_position(sim_coord),
            spectral_model = PowerLawSpectralModel(
                index=2, amplitude=f"{flux} cm-2 s-1 TeV-1", reference="1 TeV"
            ),
            temporal_model = ConstantTemporalModel(const=1),
            name="model-simulated",
        )])
        dataset.models = model_source
        
        # Generating simulated data
        dataset.fake()

        stats = convolved_map_dataset_counts_statistics(
            dataset, 
            excess_estimator.estimate_kernel(dataset), 
            excess_estimator.estimate_mask_default(dataset), 
            excess_estimator.correlate_off
        )

        lik_alt  = cash(stats.n_on.sum(axis=0), stats.n_on.sum(axis=0))
        lik_null = cash(stats.n_on.sum(axis=0), stats.n_bkg.sum(axis=0))
        ts_sign  = np.where((stats.n_on.sum(axis=0) - stats.n_bkg.sum(axis=0)) >= 0.0, +1.0, -1.0)

        # Setting minimum to 0, given some ~1e-6 fluctuations that can be produced
        ts  = np.where((lik_null - lik_alt) < 0.0, 0.0, (lik_null - lik_alt))
        ts  = ts * ts_sign
        ts2 = ts + 2 * np.log(prob_gw)

        # Checking only in 95% area only
        ts_masked  = np.where(mask_threshold, ts, np.nan)
        ts2_masked = np.where(mask_threshold, ts2, np.nan)
        ts_argmax  = np.unravel_index(np.nanargmax(ts_masked),  ts_masked.shape)
        ts2_argmax = np.unravel_index(np.nanargmax(ts2_masked), ts2_masked.shape)
                
        lambda_data.append(np.nanmax(ts2_masked))
        lambda_ra.append(bin_c_ra[ts2_argmax])
        lambda_dec.append(bin_c_dec[ts2_argmax])
        
        tsmax.append(np.nanmax(ts_masked))
        tsmax_ra.append(bin_c_ra[ts_argmax])
        tsmax_dec.append(bin_c_dec[ts_argmax])
        
        ts_dist.append(ts)
        ts2_dist.append(ts2)

        if compute_uls:
            maps = excess_estimator.run(dataset)
            flux_map = maps["flux_ul"].data[0] * mask_threshold
            ul_argmax = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)
            
            ulmax.append(np.nanmax(flux_map))
            ulmax_ra.append(bin_c_ra[ul_argmax])
            ulmax_dec.append(bin_c_dec[ul_argmax])
            
            ul_dist.append(maps["flux_ul"].data[0])        

    print(f"Writting file:\n --> {file_output}")
    data_dict = dict(
        lambda_data = np.array(lambda_data),
        lambda_ra   = np.array(lambda_ra),
        lambda_dec  = np.array(lambda_dec),
        
        tsmax     = np.array(tsmax),
        tsmax_ra  = np.array(tsmax_ra),
        tsmax_dec = np.array(tsmax_dec),
        
        ts_dist  = np.array(ts_dist),
        ts2_dist = np.array(ts2_dist),
        
        f_ra  = np.array(ra_sim),
        f_dec = np.array(dec_sim),
    )
    
    # Extra fields if compute_uls is raised
    if compute_uls:
        data_dict.update(
            ulmax     = np.array(ulmax),
            ulmax_ra  = np.array(ulmax_ra),
            ulmax_dec = np.array(ulmax_dec),
            
            ul_dist   = np.array(ul_dist),
        )

    np.savez(file_output, **data_dict)
    

def read_bkg(data_store, obs_id, dir_dl3, dim_bkg, bkg_type):
    
    fname = glob.glob(os.path.join(dir_dl3, f"bkg_{bkg_type}_{dim_bkg}d_{str(obs_id)}.fits"))[0]
    hdul = fits.open(fname)
    # Adding the acceptance model to the HDU table
    data_store.hdu_table.add_row({
        "OBS_ID" : obs_id,
        "HDU_TYPE" : "bkg",
        "HDU_CLASS" : f"bkg_{dim_bkg}d",
        "FILE_DIR" : ".",
        "FILE_NAME" : os.path.basename(fname),
        "HDU_NAME" : "BACKGROUND",
        "SIZE" : hdul["BACKGROUND"].size,
    })
    return data_store


def summary_geometry(geom, bin_edges_ra, bin_edges_dec, size_fov, data_ligo_2d, threshold_maps):
    fig, ax = plt.subplots(figsize=(2.1, 2), subplot_kw={"projection": geom.wcs})
    trans = ax.get_transform("icrs")

    for i in range(bin_edges_ra.shape[0]):
        ax.plot(bin_edges_ra[i,:], bin_edges_dec[i,:], color="0.7", lw=.5, transform=ax.get_transform("icrs"))
    for j in range(bin_edges_ra.shape[1]):
        ax.plot(bin_edges_ra[:,j], bin_edges_dec[:,j], color="0.7", lw=.5, transform=ax.get_transform("icrs"))

    circle = Circle(
        (bin_edges_ra.mean(), bin_edges_dec.mean()), size_fov.value / 2,
        edgecolor="none", facecolor="r", alpha=0.2, lw=1,
        transform=trans, zorder=10, label=f"{size_fov}"
    )
    ax.add_patch(circle)

    lims = ax.get_xlim(), ax.get_ylim()
    ax.contour(
        data_ligo_2d, levels=threshold_maps, origin="lower", extent=[-180, 180, -90, 90],
        colors="k", linewidths=1, transform=trans
    )
    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])

    ax.plot([], [], lw=1, color="0.7", label="Spatial bins")
    ax.plot([], [], lw=1.5, color="k", label="GW 50, 95%")
    ax.legend(frameon=False, loc=(1.03, 0))
    plt.show()
    
def summary_folded_counts(dataset, bin_c_ra, obs, i, n_obs):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = bin_c_ra.mean(axis=0)

    # Folded counts and background along RA and DEC
    ax.plot(x, dataset.background.data.sum(axis=0).sum(axis=0), 'r--')
    ax.plot(x, dataset.counts.data.sum(axis=0).sum(axis=0), 'r', label="Folded RA")
    ax.plot(x, dataset.background.data.sum(axis=0).sum(axis=1), 'b--')
    ax.plot(x, dataset.counts.data.sum(axis=0).sum(axis=1), 'b', label="Folded DEC")

    # Dummy lines for legend
    ax.plot([], [], 'k-', label="Counts")
    ax.plot([], [], 'k--', label="BKG model")

    # Labels, title, grid, legend
    ax.set(title=f"{i+1}/{n_obs}, bkg norm Run {obs.obs_id}",
           xlabel="RA [deg]", ylabel="BKG Rate")
    ax.grid()
    ax.legend(loc=(1.03, 0), frameon=False)
    plt.show()
    


def summary_maps(dataset, geom, bin_edges_ra, bin_edges_dec,
                 data_ligo_2d, threshold_maps, source_coord, source_name):
    """
    Plot counts, background, exposure, and excess maps for a dataset.
    Adds LIGO contours and source marker to each panel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6), subplot_kw={"projection": geom.wcs})
    axes = axes.ravel()

    data_dict = {
        "Counts":    (dataset.counts.data.sum(axis=0), "viridis", None, None),
        "BKG rate [1 / (MeV s sr)]": (dataset.background.data.sum(axis=0), "viridis", None, None),
        "Exposure [m${}^2$ s]":      (dataset.exposure.data.sum(axis=0), "viridis", None, None),
        "Excess":    (dataset.excess.data.sum(axis=0), "coolwarm",
                      -abs(dataset.excess.data).max(), abs(dataset.excess.data).max()),
    }

    for ax, (label, (data, cmap, vmin, vmax)), c in zip(
        axes, data_dict.items(), ["0.8", "0.8", "0.8", "0.3"]
    ):
        mesh = ax.pcolormesh(bin_edges_ra, bin_edges_dec, data, cmap=cmap,
                             vmin=vmin, vmax=vmax, transform=ax.get_transform("icrs"))
        fig.colorbar(mesh, ax=ax, label=label)

        # LIGO contours and source marker
        lims = ax.get_xlim(), ax.get_ylim()
        ax.contour(data_ligo_2d, levels=threshold_maps, transform=ax.get_transform("icrs"),
                   colors=c, origin="lower", extent=[-180, 180, -90, 90])
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])

        ax.plot(source_coord.ra, source_coord.dec, "xw", label=source_name,
                transform=ax.get_transform("icrs"))
        ax.grid(alpha=0.6)
        ax.set(xlabel="Right Ascension", ylabel="Declination")

    fig.tight_layout()
    plt.show()

def healpix2map(
    healpix_data,
    ra_bins, 
    dec_bins,
):
    
    ra_grid, dec_grid = np.meshgrid(ra_bins, dec_bins)

    # Convert the latitude and longitude to theta and phi
    theta, phi = np.radians(90 - dec_grid), np.radians(ra_grid)
    
    nside = hp.npix2nside(len(healpix_data)) # nside of the grid

    # Convert theta, phi to HEALPix indices and create a 2D map using the HEALPix data
    hp_indices = hp.ang2pix(nside, theta, phi)

    return (healpix_data[hp_indices])

def get_hp_map_thresholds(healpix_data, threshold_percent=[0.9, 0.68],):
    
    # We sort the tresholds itself in descending order
    threshold_percent = np.sort(threshold_percent)[::-1]
    
    # Sort in descending order and normalize
    sorted_data = np.sort(healpix_data)[::-1] / np.sum(healpix_data)
    cumulative_sum = np.cumsum(sorted_data)

    # Find the values corresponding to the thresholds
    indexes_map = [np.searchsorted(cumulative_sum, t) for t in threshold_percent]
    # Then we find the thresholds
    threshold_maps = [sorted_data[min(index, len(sorted_data) - 1)] for index in indexes_map]
    
    return threshold_maps

def get_2d_map_hotspot(map_data_2d, ra_bins, dec_bins):
    
    # Computing coordinate of maximum probability
    max_prob_index = np.unravel_index(np.argmax(map_data_2d), map_data_2d.shape)
    
    max_prob_ra, max_prob_dec = ra_bins[max_prob_index[1]], dec_bins[max_prob_index[0]]
    max_prob_coords = SkyCoord(ra=max_prob_ra, dec=max_prob_dec, unit=u.deg, frame="icrs")
    return max_prob_coords

def add_telescope_location_to_obs(obs_collection, loc):
    for i in range(len(obs_collection)):
        obs_collection[i].obs_info["observatory_earth_location"] = loc
        obs_collection[i].obs_info["GEOLON"] = loc.lon.value
        obs_collection[i].obs_info["GEOLAT"] = loc.lat.value
        obs_collection[i].obs_info["GEOALT"] = loc.height.value
        obs_collection[i]._location = loc
        obs_collection[i].pointing._location = loc
    return obs_collection

def sigmoid(x, L ,x0, k, b):
    """
    L: curve's maximum value (upper plateau)
    x0: the x-value of the sigmoid's midpoint
    k: the steepness of the curve
    b: the minimum value (lower plateau)
    """
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))



if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python script.py <n_sim> <flux> <file_input.pkl> <file_output.npz> <compute_uls>")
        sys.exit(1)

    n_sim = int(sys.argv[1])
    flux = float(sys.argv[2])
    file_input = sys.argv[3]
    file_output = sys.argv[4]
    compute_uls = int(sys.argv[5])
    
    perform_n_simulations(n_sim, flux, file_input, file_output, compute_uls)
    
    