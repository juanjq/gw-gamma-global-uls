import numpy as np
import matplotlib.pyplot as plt
import pickle, sys
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp
from gammapy.stats.fit_statistics import cash
from gammapy.estimators.map.excess import convolved_map_dataset_counts_statistics, _get_convolved_maps
from gammapy.modeling.models import (
    PointSpatialModel, PowerLawSpectralModel, ConstantTemporalModel, SkyModel, Models
)

from utils import IndexToDeclRa
from plotting import plot_ul_iteration


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def perform_n_simulations(n_sim, amplitude, file_input, file_output, compute_uls=0):
    """
    n_sim (int):
        Number of iterations of each simulation.
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
        Path to the output .npz file.
    compute_uls (int):
        If flux UL distributions need to be computed or not.
        Takes some time so only use this to debug. True=1, False=0.
    """
    n_sim       = int(n_sim)
    compute_uls = bool(compute_uls)

    with open(file_input, "rb") as f:
        data = pickle.load(f)
    
    dataset            = data["dataset"]
    excess_estimator   = data["excess_estimator"]
    ts_estimator       = data["ts_estimator"]
    prob_gw            = data["prob_gw"]
    data_ligo_hp       = data["data_ligo_hp"]
    mask_threshold     = data["mask_threshold_95"]
    containment_factor = data["containment_factor"]

    print(f"Producing {n_sim} simulations for amplitude={amplitude:.2e} cm-2 s-1 TeV-1...\n")

    # Precompute log term for ts2 (never changes across iterations)
    log_prob_gw = 2 * np.log(prob_gw)

    # --- Build masked GW sampling distribution ---
    pix_indices, nside = np.arange(len(data_ligo_hp)), hp.npix2nside(len(data_ligo_hp))
    dec_hp, ra_hp      = IndexToDeclRa(pix_indices, nside)
    ra_hp_conv         = -(((ra_hp + 180) % 360) - 180)

    geom          = dataset.geoms["geom"]
    geom_centers  = geom.get_coord(mode="center")
    bin_c_ra      = geom_centers.lon[0].value   # (ny, nx)
    bin_c_dec     = geom_centers.lat[0].value

    # Nearest-pixel lookup: map each HEALPix pixel → WCS bin (memory-efficient)
    ra_sorted_idx  = np.argsort(bin_c_ra[0, :])
    dec_sorted_idx = np.argsort(bin_c_dec[:, 0])
    
    ra_sorted  = bin_c_ra[0, :][ra_sorted_idx]
    dec_sorted = bin_c_dec[:, 0][dec_sorted_idx]
    
    # searchsorted gives insertion point; clamp and pick nearest neighbour
    def nearest_idx(sorted_arr, values, sorted_idx):
        pos = np.searchsorted(sorted_arr, values)
        pos = np.clip(pos, 0, len(sorted_arr) - 1)
        # check left neighbour too
        pos_left = np.clip(pos - 1, 0, len(sorted_arr) - 1)
        closer_left = np.abs(values - sorted_arr[pos_left]) < np.abs(values - sorted_arr[pos])
        pos[closer_left] = pos_left[closer_left]
        return sorted_idx[pos]
    
    ix      = nearest_idx(ra_sorted,  ra_hp_conv, ra_sorted_idx)   # (n_hp,)
    iy      = nearest_idx(dec_sorted, dec_hp,     dec_sorted_idx)  # (n_hp,)
    in_mask = mask_threshold[iy, ix]

    prob_gw_masked        = data_ligo_hp.copy().astype(float)
    prob_gw_masked[~in_mask] = 0.0
    frac_in_mask          = prob_gw_masked.sum()
    prob_gw_masked       /= frac_in_mask
    # print(f"Mask covers {frac_in_mask * 100:.1f}% of total GW probability\n")

    # Sample n_sim positions from masked+renormalized GW map
    indices_hp  = np.random.choice(len(data_ligo_hp), size=n_sim, p=prob_gw_masked)
    ra_sim      = ra_hp_conv[indices_hp]
    dec_sim     = dec_hp[indices_hp]

    # Pre-loading estimator utilities
    correlate_off = excess_estimator.correlate_off
    kernel        = excess_estimator.estimate_kernel(dataset)
    mask_default  = excess_estimator.estimate_mask_default(dataset)

    # Precompute all SkyCoords at once
    sim_coords = [
        SkyCoord(ra=ra_sim[i], dec=dec_sim[i], unit=u.deg, frame="icrs")
        for i in range(n_sim)
    ]
    
    # Getting the geometry
    geom         = dataset.geoms["geom"]
    geom_centers = geom.get_coord(mode="center")
    bin_c_ra     = geom_centers.lon[0].value
    bin_c_dec    = geom_centers.lat[0].value

    results = {
        "lambda_data": [], "lambda_ra": [], "lambda_dec": [], "tsmax": [],
        "tsmax_ra": [], "tsmax_dec": [], "ts_dist": [], "ts2_dist": [], "stats": []
    }
    if compute_uls:
        results.update({"ulmax": [], "ulmax_ra": [], "ulmax_dec": [], "ul_dist": []})

    for i in range(n_sim):
        print(f"Computing... {i+1}/{n_sim}", end="\r")

        # Source model
        model_source = Models([SkyModel(
            spatial_model  = PointSpatialModel.from_position(sim_coords[i]),
            spectral_model = PowerLawSpectralModel(
                index=2, amplitude=f"{amplitude} cm-2 s-1 TeV-1", reference="1 TeV"
            ),
            temporal_model = ConstantTemporalModel(),
            name="model-simulated",
        )])

        # Simulate counts, then clear model so it isn't treated as known bkg
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

        # Computing TS values
        n_on_sum   = stats.n_on.sum(axis=0)
        mu_bkg_sum = stats.mu_bkg.sum(axis=0)

        lik_alt  = cash(n_on_sum, n_on_sum)
        lik_null = cash(n_on_sum, mu_bkg_sum)
        ts_sign  = np.where((n_on_sum - mu_bkg_sum) >= 0.0, +1.0, -1.0)
        ts       = np.where((lik_null - lik_alt) < 0.0, 0.0, (lik_null - lik_alt))
        ts       = ts * ts_sign
        ts2      = ts + log_prob_gw

        # Masking with 95% GW
        ts_masked  = np.where(mask_threshold, ts,  np.nan)
        ts2_masked = np.where(mask_threshold, ts2, np.nan)
        ts_argmax  = np.unravel_index(np.nanargmax(ts_masked),  ts_masked.shape)
        ts2_argmax = np.unravel_index(np.nanargmax(ts2_masked), ts2_masked.shape)

        results["stats"].append(stats)
        results["lambda_data"].append(np.nanmax(ts2_masked))
        results["lambda_ra"].append(bin_c_ra[ts2_argmax])
        results["lambda_dec"].append(bin_c_dec[ts2_argmax])
        results["tsmax"].append(np.nanmax(ts_masked))
        results["tsmax_ra"].append(bin_c_ra[ts_argmax])
        results["tsmax_dec"].append(bin_c_dec[ts_argmax])
        results["ts_dist"].append(ts)
        results["ts2_dist"].append(ts2)

        if compute_uls:
            maps      = excess_estimator.run(dataset)
            flux_map  = maps["flux_ul"].data[0] * mask_threshold / containment_factor
            ul_argmax = np.unravel_index(np.nanargmax(flux_map), flux_map.shape)
            results["ulmax"].append(np.nanmax(flux_map))
            results["ulmax_ra"].append(bin_c_ra[ul_argmax])
            results["ulmax_dec"].append(bin_c_dec[ul_argmax])
            results["ul_dist"].append(maps["flux_ul"].data[0])

    print(f"\nWriting file: --> {file_output}")
    data_dict = {key: np.array(value) for key, value in results.items()}
    data_dict.update({"f_ra": np.array(ra_sim), "f_dec": np.array(dec_sim)})
    np.savez(file_output, **data_dict)


# ---------------------------------------------------------------------------
# Iterative upper-limit algorithm
# ---------------------------------------------------------------------------

def run_iterative_ul(
    path_pkl, lambda_real, lambda_bkg, lambda_bkg_m, significance, p_value, energy_edges,
    cl, n_sim=500, precision=0.05, frac_tol=0.01, amp_lo=1e-13, amp_hi=1e-10, max_iter=20,
    cache_path=None,
):
    target = lambda_bkg_m if significance < 0 else lambda_real

    # --- Load or initialise persistent cache ---
    if cache_path is not None and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            lambda_cache = pickle.load(f)
        print(f"Loaded lambda_cache with {len(lambda_cache)} entries from {cache_path}\n")
    else:
        lambda_cache = {}

    history_amp, history_frac, lambda_iter = [], [], None

    for iteration in range(max_iter):
        amp_mid = np.sqrt(amp_lo * amp_hi)
        key     = f"{amp_mid:.6e}"

        if key not in lambda_cache:
            fname = f"./data/tmp/iterative_ul_iter{iteration}_{amp_mid:.4e}.npz"
            perform_n_simulations(
                n_sim=n_sim, amplitude=amp_mid,
                file_input=path_pkl, file_output=fname, compute_uls=0,
            )
            lambda_cache[key] = np.load(fname)["lambda_data"]

            # Persist cache immediately after every new simulation
            if cache_path is not None:
                with open(cache_path, "wb") as f:
                    pickle.dump(lambda_cache, f)
                print(f"  [cache] saved {len(lambda_cache)} entries → {cache_path}")
        else:
            print(f"  [cache] amp={amp_mid:.4e} - reusing {len(lambda_cache[key])} existing sims")

        lambda_iter = lambda_cache[key]
        frac        = np.mean(lambda_iter > target)
        history_amp.append(amp_mid); history_frac.append(frac)

        if frac > cl:
            amp_hi = amp_mid
        else:
            amp_lo = amp_mid

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3))
        plot_ul_iteration(
            ax1, ax2, lambda_bkg, lambda_iter, lambda_real, lambda_bkg_m, target,
            history_amp, history_frac, amp_mid, amp_lo, amp_hi, frac, cl,
            p_value, significance, iteration, max_iter, energy_edges, precision
        )
        plt.tight_layout()
        plt.show()

        bracket_width = np.log10(amp_hi / amp_lo)
        print(
            f"\n[Iter {iteration+1:2d}]: amp={amp_mid:.4e}  N={len(lambda_iter)}  "
            f"frac={frac:.3f}  bracket=[{amp_lo:.3e}, {amp_hi:.3e}]  width={bracket_width:.3f}"
        )

        bracket_ok = bracket_width < precision
        frac_ok    = abs(frac - cl) < frac_tol
        if bracket_ok and frac_ok:
            print(f"Converged in {iteration+1} iterations - bracket {bracket_width:.4f}")
            break
        elif bracket_ok:
            print(f"  Bracket converged but frac={frac:.3f} still {abs(frac-cl):.3f} from cl={cl} - continuing...")
        elif frac_ok:
            print(f"  Frac converged but bracket still {bracket_width:.3f} width wide - continuing...")
    else:
        print(f"Warning: reached max_iter={max_iter} without full convergence.")

    amp_final       = np.sqrt(amp_lo * amp_hi)
    pwl_final       = PowerLawSpectralModel(amplitude=amp_final * u.Unit("TeV-1 s-1 cm-2"), index=2)
    flux_final      = pwl_final.integral(*energy_edges)
    flux_diff_final = pwl_final.energy_flux(*energy_edges).to(u.erg / (u.s * u.cm**2))

    print(f"\n{'='*50}")
    print(f"  UL amplitude : {amp_final:.3e} TeV-1 s-1 cm-2")
    print(f"  UL flux      : {flux_final:.3e}")
    print(f"  UL diff flux : {flux_diff_final:.3e}")
    print(f"  Ampl uncert  : {amp_hi - amp_lo:.3e}")
    print(f"  Iterations   : {len(history_amp)}")
    print(f"  Total sims   : {sum(len(v) for v in lambda_cache.values())}")
    print(f"{'='*50}")

    return dict(
        amp_ul       = amp_final,
        flux_ul      = flux_final,
        flux_diff_ul = flux_diff_final,
        hist_amp     = history_amp,
        hist_frac    = history_frac,
        lambda_cache = lambda_cache,
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("Usage: python simulate.py <n_sim> <flux> <file_input.pkl> <file_output.npz> <compute_uls>")
        sys.exit(1)

    n_sim       = int(sys.argv[1])
    flux        = float(sys.argv[2])
    file_input  = sys.argv[3]
    file_output = sys.argv[4]
    compute_uls = int(sys.argv[5])

    perform_n_simulations(n_sim, flux, file_input, file_output, compute_uls)