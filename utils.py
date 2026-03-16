import numpy as np
import matplotlib.pyplot as plt
import glob, os, pickle, sys, copy
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle
import healpy as hp
from gammapy.stats.fit_statistics import cash
from gammapy.estimators.map.excess import convolved_map_dataset_counts_statistics, _get_convolved_maps
from gammapy.modeling.models import PointSpatialModel, PowerLawSpectralModel, ConstantTemporalModel, SkyModel, Models

def perform_n_simulations(n_sim, amplitude, file_input, file_output, compute_uls=0):
    """
    n_sim (int):
        Number of iterations of each simulation
    amplitude (float): 
        The PWL amplitude parameter in (cm-2 s-1 TeV-1) units.
    file_input (str): 
        Path to the input .pkl file, containing:
            * dataset_empty
            * Excess estimator (for fast TS computation)
            * TS estimator
            * GW sky-map in WCS
            * GW sky-map in HealPix
            * Mask 95% of the GW
            * Containment factor for flux
            * (optional) precomputed kernel
            * (optional) precomputed mask_default
    file_output (str): 
        Path to the output .npz file
    compute_uls (int): 
        If flux UL distributions need to be computed or not.
        takes some time so only use this to debug. True=1, False=0
    """

    # Reading the input parameters ---
    n_sim = int(n_sim)
    compute_uls = bool(compute_uls)
    with open(file_input, "rb") as file:
        loaded = pickle.load(file)

    dataset, excess_estimator, ts_estimator, prob_gw, data_ligo_hp, mask_threshold, containment_factor, kernel, mask_default = loaded

    print(f"Producing {n_sim} simulations for amplitude {amplitude} cm-2 s-1 TeV-1")
    print("Setting up everything...\n")

    # Precompute log term for ts2 (never changes across iterations) ---
    log_prob_gw = 2 * np.log(prob_gw)

    # Simulating n_sim source positions following GW sky-map ---
    pix_indices, nside = np.arange(len(data_ligo_hp)), hp.npix2nside(len(data_ligo_hp))
    dec_hp, ra_hp = IndexToDeclRa(pix_indices, nside)
    indices_hp = np.random.choice(len(data_ligo_hp), size=n_sim, p=data_ligo_hp)
    ra_sim, dec_sim = ra_hp[indices_hp], dec_hp[indices_hp]
    ra_sim = [-(((ra_sim[i] + 180) % 360) - 180) for i in range(len(ra_sim))]

    # Pre-loading estimator utilities
    correlate_off = excess_estimator.correlate_off
    kernel = excess_estimator.estimate_kernel(dataset)
    mask_default = excess_estimator.estimate_mask_default(dataset)

    # Precompute all SkyCoords at once ---
    sim_coords = [
        SkyCoord(ra=ra_sim[i], dec=dec_sim[i], unit=u.deg, frame="icrs") for i in range(n_sim)
    ]

    # Getting the geometry ---
    geom = dataset.geoms["geom"]
    geom_centers = geom.get_coord(mode="center")
    bin_c_ra  = geom_centers.lon[0].value
    bin_c_dec = geom_centers.lat[0].value

    # Variables to be filled in iterations ---
    results = {
        "lambda_data": [], "lambda_ra": [], "lambda_dec": [], "tsmax": [],
        "tsmax_ra": [], "tsmax_dec": [], "ts_dist": [], "ts2_dist": [], "stats": []
    }
    if compute_uls:
        results.update({"ulmax": [], "ulmax_ra": [], "ulmax_dec": [], "ul_dist": []})

    for i in range(n_sim):
        print(f"Computing... {i+1}/{n_sim}", end="\r")

        # Source model ---
        model_source = Models([SkyModel(
            spatial_model  = PointSpatialModel.from_position(sim_coords[i]),
            spectral_model = PowerLawSpectralModel(
                index=2, amplitude=f"{amplitude} cm-2 s-1 TeV-1", reference="1 TeV"
            ),
            temporal_model = ConstantTemporalModel(),
            name="model-simulated",
        )])

        # Simulate counts, then clear model so it isn't treated as known bkg ---
        dataset.models = model_source
        dataset.fake(i)
        dataset.models = None

        stats = convolved_map_dataset_counts_statistics(
            convolved_maps=_get_convolved_maps(
                dataset=dataset,
                kernel=kernel,
                mask=mask_default,
                correlate_off=correlate_off,
            ),
            stat_type="cash",
        )

        # Computing TS values ---
        n_on_sum   = stats.n_on.sum(axis=0)
        mu_bkg_sum = stats.mu_bkg.sum(axis=0)

        lik_alt  = cash(n_on_sum, n_on_sum)
        lik_null = cash(n_on_sum, mu_bkg_sum)
        ts_sign  = np.where((n_on_sum - mu_bkg_sum) >= 0.0, +1.0, -1.0)
        ts       = np.where((lik_null - lik_alt) < 0.0, 0.0, (lik_null - lik_alt))
        ts       = ts * ts_sign
        ts2      = ts + log_prob_gw

        # Masking with 95% GW ---
        ts_masked  = np.where(mask_threshold, ts,  np.nan)
        ts2_masked = np.where(mask_threshold, ts2, np.nan)
        ts_argmax  = np.unravel_index(np.nanargmax(ts_masked),  ts_masked.shape)
        ts2_argmax = np.unravel_index(np.nanargmax(ts2_masked), ts2_masked.shape)

        # Storing data ---
        results["stats"].append(stats)
        results["lambda_data"].append(np.nanmax(ts2_masked))
        results["lambda_ra"].append(bin_c_ra[ts2_argmax])
        results["lambda_dec"].append(bin_c_dec[ts2_argmax])
        results["tsmax"].append(np.nanmax(ts_masked))
        results["tsmax_ra"].append(bin_c_ra[ts_argmax])
        results["tsmax_dec"].append(bin_c_dec[ts_argmax])
        results["ts_dist"].append(ts)
        results["ts2_dist"].append(ts2)

        # If ULs are needed ---
        if compute_uls:
            maps = excess_estimator.run(dataset)
            flux_map = maps["flux_ul"].data[0] * mask_threshold / containment_factor
            ul_argmax = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)
            results["ulmax"].append(np.nanmax(flux_map))
            results["ulmax_ra"].append(bin_c_ra[ul_argmax])
            results["ulmax_dec"].append(bin_c_dec[ul_argmax])
            results["ul_dist"].append(maps["flux_ul"].data[0])

    # Storing data as .npz ---
    print(f"\nWriting file: --> {file_output}")
    data_dict = {key: np.array(value) for key, value in results.items()}
    data_dict.update({"f_ra":  np.array(ra_sim), "f_dec": np.array(dec_sim)})
    np.savez(file_output, **data_dict)
    

def add_bkg(data_store, obs_id, dir_dl3, dim_bkg, bkg_type):
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

    # Convert the latitude and longitude to theta and phi
    theta, phi = np.radians(90 - dec_grid), np.radians(ra_grid)
    
    nside = hp.npix2nside(len(healpix_data)) # nside of the grid

    # Convert theta, phi to HEALPix indices and create a 2D map using the HEALPix data
    hp_indices = hp.ang2pix(nside, theta, phi)

    return (healpix_data[hp_indices])

def get_hp_map_thresholds(healpix_data, threshold_percent=[0.9, 0.68]):
    
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

def sigmoid(x, L ,x0, k, b):
    """
    L: curve's maximum value (upper plateau)
    x0: the x-value of the sigmoid's midpoint
    k: the steepness of the curve
    b: the minimum value (lower plateau)
    """
    return b + (L - b) / (1 + np.exp(-k * (x - x0)))

def find_x_at_y(y_target, params):
    a, x0, k, y0 = params
    func = lambda x: sigmoid(x, a, x0, k, y0) - y_target
    result = root_scalar(func, bracket=[x0 - 10, x0 + 10], method="brentq")
    return result.root if result.converged else np.nan

def find_amplitude_at_cl(x, y, cl=0.95):
    x, y = np.array(x), np.array(y)
    s = np.argsort(x); x, y = x[s], y[s]
    a = np.where(y >= cl)[0]
    if len(a) == 0: return np.nan
    i = a[0]
    if i == 0: return 10**x[0]
    x1, x2, y1, y2 = x[i-1], x[i], y[i-1], y[i]
    return 10**(x1 + (cl - y1) * (x2 - x1) / (y2 - y1))
    
def summary_geometry(
    geom, bin_edges_ra, bin_edges_dec, size_fov, data_ligo_2d, threshold_maps
):
    fig, ax = plt.subplots(figsize=(2.1, 2), subplot_kw={"projection": geom.wcs})
    trans = ax.get_transform("icrs")
    for i in range(bin_edges_ra.shape[0]):
        ax.plot(bin_edges_ra[i,:], bin_edges_dec[i,:], color="0.7", lw=.5, transform=trans)
    for j in range(bin_edges_ra.shape[1]):
        ax.plot(bin_edges_ra[:,j], bin_edges_dec[:,j], color="0.7", lw=.5, transform=trans)
    circle = Circle(
        (bin_edges_ra.mean(), bin_edges_dec.mean()), size_fov.value / 2,
        edgecolor="none", facecolor="r", alpha=0.2, lw=1,
        transform=trans, zorder=10, label=f"{size_fov}"
    )
    ax.add_patch(circle)
    lims = ax.get_xlim(), ax.get_ylim()

    if threshold_maps is not None:
        ax.contour(
            data_ligo_2d, levels=threshold_maps, origin="lower", extent=[-180, 180, -90, 90],
            colors="k", linewidths=1, transform=trans
        )
        ax.plot([], [], lw=1.5, color="k", label="GW 50, 95%")
    else:
        # Dirac delta: mark the hot pixel directly
        hot_idx = np.unravel_index(np.argmax(data_ligo_2d), data_ligo_2d.shape)
        hot_ra  = np.linspace(-180, 180, data_ligo_2d.shape[1])[hot_idx[1]]
        hot_dec = np.linspace(-90,   90, data_ligo_2d.shape[0])[hot_idx[0]]
        ax.plot(hot_ra, hot_dec, "kx", ms=6, transform=trans, label="GW ($\\delta$)")

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.plot([], [], lw=1, color="0.7", label="Spatial bins")
    ax.legend(frameon=False, loc=(1.03, 0))
    plt.show()
    
def summary_folded_counts(dataset, bin_c_ra, obs, i, n_obs):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(bin_c_ra))

    # Folded counts and background along RA and DEC
    ax.plot(x, dataset.background.data.sum(axis=0).sum(axis=0), "r--")
    ax.plot(x, dataset.counts.data.sum(axis=0).sum(axis=0), "r", label="Folded RA")
    ax.plot(x, dataset.background.data.sum(axis=0).sum(axis=1), "b--")
    ax.plot(x, dataset.counts.data.sum(axis=0).sum(axis=1), "b", label="Folded DEC")

    # Dummy lines for legend
    ax.plot([], [], "k-", label="Counts")
    ax.plot([], [], "k--", label="BKG model")

    # Labels, title, grid, legend
    ax.set(title=f"{i+1}/{n_obs}, bkg norm Run {obs.obs_id}", xlabel="Spatial bins RA / DEC", ylabel="BKG Rate")
    ax.grid(); ax.legend(loc=(1.03, 0), frameon=False)
    plt.show()

def summary_maps(
    dataset, geom, bin_edges_ra, bin_edges_dec, data_ligo_2d, 
    threshold_maps, source_coord, source_name
):
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

def summary_gw_map(geom, bin_edges_ra, bin_edges_dec, prob_gw):
    fig, ax = plt.subplots(figsize=(2.4, 2), subplot_kw={"projection": geom.wcs})
    trans = ax.get_transform("icrs")
    
    pc = ax.pcolormesh(
        bin_edges_ra, bin_edges_dec, prob_gw, cmap="cylon",
        transform=ax.get_transform("icrs")
    ); fig.colorbar(pc, label="GW Prob")
    
    ax.set(title=f"$\\sum P=${np.sum(prob_gw)*100:.2f}%")
    ax.coords[0].set_axislabel_position("b"); ax.coords[0].set_ticklabel_position("b")
    ax.coords[0].set_axislabel("RA [deg]"); ax.coords[1].set_axislabel("DEC [deg]")
    plt.show()

def make_dirac_delta_hp(coord: SkyCoord, nside: int) -> np.ndarray:
    """HEALPix map with probability=1 at the pixel closest to coord."""
    data_hp = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    data_hp[hp.ang2pix(nside, np.pi/2 - coord.dec.rad, coord.ra.rad)] = 1.0
    return data_hp

def make_dirac_delta_2d(coord, ra_bins, dec_bins):
    """2D grid map with probability=1 at the pixel closest to coord."""
    data_2d = np.zeros((len(dec_bins), len(ra_bins)), dtype=np.float64)
    data_2d[np.argmin(np.abs(dec_bins - coord.dec.deg)),
            np.argmin(np.abs(ra_bins  - coord.ra.wrap_at('180d').deg))] = 1.0
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

def integrate_hp_on_wcs(
    data_ligo_hp, dec_hp, ra_hp, hp_area, bin_edges_ra, bin_edges_dec, bin_area
):
    """Integrate HEALPix probability map onto WCS bins."""
    mask_hp_wcs = (
        (-(((ra_hp + 180) % 360) - 180) >= bin_edges_ra[:,-1:].min()) & (dec_hp >= bin_edges_dec[:1,:].max()) &
        (-(((ra_hp + 180) % 360) - 180) <= bin_edges_ra[:,:1].min()) & (dec_hp <= bin_edges_dec[-1:,:].max())
    )
    data_ligo_hp_wcs = data_ligo_hp[mask_hp_wcs]
    dec_hp_wcs       = dec_hp[mask_hp_wcs]
    ra_hp_wcs        = -(ra_hp[mask_hp_wcs] + 180) % 360 - 180
    prob     = np.zeros((len(bin_edges_ra)-1, len(bin_edges_dec)-1))
    num_pix  = np.zeros((len(bin_edges_ra)-1, len(bin_edges_dec)-1))
    mask_added = np.zeros(len(ra_hp_wcs), dtype=bool)
    for i in range(len(bin_edges_ra)-1):
        for j in range(len(bin_edges_ra[i])-1):
            mask_bin = (
                (ra_hp_wcs >= bin_edges_ra[i,j+1]) & (dec_hp_wcs >= bin_edges_dec[i,j]) &
                (ra_hp_wcs <= bin_edges_ra[i,j])   & (dec_hp_wcs <= bin_edges_dec[i+1,j+1])
            ) & ~mask_added
            p = np.sum(data_ligo_hp_wcs[mask_bin]) * (bin_area[i,j].value / (np.sum(mask_bin) * hp_area))
            prob[i,j]    = 0.0 if np.isnan(p) else p
            num_pix[i,j] = np.sum(mask_bin)
            mask_added  |= mask_bin
    return prob, num_pix


def _plot_convergence_ax(
    ax, history_amp, history_frac, amp_lo, amp_hi, amp_final, cl, precision
):
    """Shared convergence plot — fraction vs iteration."""
    ax.plot(range(1, len(history_frac)+1), history_frac, "ko-", ms=5, zorder=5)
    ax.axhline(cl, color="orange", ls="--", lw=1.5, label=f"CL={cl}")
    ax.axhspan(cl - cl*precision, cl + cl*precision, alpha=0.15, color="orange")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fraction > target")
    ax.set_xticks(range(1, len(history_frac)+1))
    # Annotate each point with its amplitude
    for k, (a, f) in enumerate(zip(history_amp, history_frac)):
        ax.annotate(f"{a:.1e}", (k+1, f), textcoords="offset points",
                    xytext=(0, 8), fontsize=7, ha="center")
    ax.set_title(f"Bisection — UL={amp_final:.2e}\nbracket=[{amp_lo:.1e}, {amp_hi:.1e}]")
    ax.legend(frameon=False, fontsize=8)


def plot_ul_iteration(
    ax1, ax2, lambda_bkg, lambda_iter, lambda_real, lambda_bkg_m, target, history_amp, history_frac, 
    amp_mid, amp_lo, amp_hi, frac, cl, p_value, significance, iteration, max_iter, energy_edges, precision
):
    lmin = min(lambda_bkg.min(), lambda_iter.min())
    lmax = max(lambda_bkg.max(), lambda_iter.max())
    bins = np.linspace(lmin - 0.1*abs(lmin), lmax + 0.1*abs(lmax), 60)

    ax1.hist(
        lambda_bkg,  bins, color="royalblue", density=True, histtype="stepfilled", alpha=0.7, 
        label=f"BKG N={len(lambda_bkg)}"
    )
    ax1.hist(
        lambda_iter, bins, color="r", density=True, histtype="step", lw=1.5, 
        label=f"iter {iteration+1}  Amp={amp_mid:.2e}  N={len(lambda_iter)}"
    )
    ax1.axvline(lambda_real,  color="k",        ls="--", label=f"Real $\\Lambda$={lambda_real:.2f}")
    ax1.axvline(lambda_bkg_m, color="darkblue", ls=":",  label=f"BKG med={lambda_bkg_m:.2f}")
    ax1.axvline(target, color="orange", ls="-", lw=2, label="Target")
    ax1.text(0.97, 0.95, f"frac={frac:.3f}  CL={cl}\np-val={p_value:.3f}  sig={significance:.2f}$\\sigma$",
             ha="right", va="top", transform=ax1.transAxes, fontsize=8)
    ax1.set_xlabel("$\\Lambda$"); ax1.set_ylabel("Normalized counts")
    ax1.set_title(f"Iter {iteration+1}/{max_iter} — [{amp_lo:.1e}, {amp_hi:.1e}]")
    ax1.legend(frameon=False, fontsize=7, loc=2)

    _plot_convergence_ax(
        ax2, history_amp, history_frac, amp_lo, amp_hi, np.sqrt(amp_lo*amp_hi), cl, precision
    )


def run_iterative_ul(
    path_pkl, lambda_real, lambda_bkg, lambda_bkg_m, significance, p_value, energy_edges,
    cl, n_sim=500, precision=0.05, frac_tol=0.01, amp_lo=1e-13, amp_hi=1e-10, max_iter=20,
):
    # The lambda we use for comparison
    target = lambda_bkg_m if significance < 0 else lambda_real
    
    history_amp, history_frac, lambda_iter, lambda_cache = [], [], None, {}
    for iteration in range(max_iter):
        amp_mid = np.sqrt(amp_lo * amp_hi)
        key = f"{amp_mid:.6e}"

        # Skip simulation if we already have results for this amplitude
        if key not in lambda_cache:
            fname = f"./data/tmp/iterative_ul_iter{iteration}_{amp_mid:.4e}.npz"
            perform_n_simulations(
                n_sim=n_sim, amplitude=amp_mid,
                file_input=path_pkl, file_output=fname, compute_uls=0,
            )
            lambda_cache[key] = np.load(fname)["lambda_data"]
        else:
            print(f"  [cache] amp={amp_mid:.4e} — reusing {len(lambda_cache[key])} existing sims")

        lambda_iter = lambda_cache[key]
        frac = np.mean(lambda_iter > target)
        history_amp.append(amp_mid); history_frac.append(frac)

        # Update bracket
        if frac > cl:
            amp_hi = amp_mid
        else:
            amp_lo = amp_mid

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
        plot_ul_iteration(
            ax1, ax2, lambda_bkg, lambda_iter, lambda_real, lambda_bkg_m, target, 
            history_amp, history_frac, amp_mid, amp_lo, amp_hi, frac, cl, p_value, 
            significance, iteration, max_iter, energy_edges, precision
        )
        plt.tight_layout()
        plt.show()

        bracket_width = np.log10(amp_hi / amp_lo)
        print(
            f"\n[Iter {iteration+1:2d}]: amp={amp_mid:.4e}  N={len(lambda_iter)}  "
            f"frac={frac:.3f}  bracket=[{amp_lo:.3e}, {amp_hi:.3e}]  width={bracket_width:.3f}"
        )

        # Convergence: bracket narrow enough AND frac close enough to target CL
        bracket_ok = bracket_width < precision
        frac_ok    = abs(frac - cl) < frac_tol
        if bracket_ok and frac_ok:
            print(f"Converged in {iteration+1} iterations — bracket {bracket_width:.4f}")
            break
        elif bracket_ok:
            print(f"  Bracket converged but frac={frac:.3f} still {abs(frac-cl):.3f} from cl={cl} — continuing...")
        elif frac_ok:
            print(f"  Frac converged but bracket still {bracket_width:.3f} width wide — continuing...")

    else:
        print(f"Warning: reached max_iter={max_iter} without full convergence.")

    amp_final = np.sqrt(amp_lo * amp_hi)
    pwl_final = PowerLawSpectralModel(amplitude=amp_final * u.Unit("TeV-1 s-1 cm-2"), index=2)
    flux_final = pwl_final.integral(*energy_edges)
    flux_diff_final = pwl_final.energy_flux(*energy_edges).to(u.erg / (u.s * u.cm**2))

    print(f"\n{'='*50}")
    print(f"  UL amplitude : {amp_final:.4e} TeV-1 s-1 cm-2")
    print(f"  UL flux      : {flux_final:.4e}")
    print(f"  UL diff flux : {flux_diff_final:.4e}")
    print(f"  Final bracket: [{amp_lo:.2e}, {amp_hi:.2e}]")
    print(f"  Iterations   : {len(history_amp)}")
    print(f"  Total sims   : {sum(len(v) for v in lambda_cache.values())}")
    print(f"  Unique amps  : {len(lambda_cache)}")
    print(f"{'='*50}")

    return amp_final, flux_final, flux_diff_final, history_amp, history_frac, lambda_cache
    
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
    