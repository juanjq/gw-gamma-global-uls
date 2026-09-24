"""Shared diagnostic plots for the 2D and 3D notebooks."""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _plot_convergence_ax(
    ax, history_amp, history_frac, amp_lo, amp_hi, amp_final, cl, precision
):
    """Shared convergence plot — fraction vs iteration."""
    ax.plot(range(1, len(history_frac)+1), history_frac, "ko-", ms=5, zorder=5)
    ax.axhline(cl, color="r", ls="--", lw=1.5, label=f"CL={cl}")
    ax.axhspan(cl - cl*precision, cl + cl*precision, alpha=0.15, color="r")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Fraction > target")
    ax.set_xticks(range(1, len(history_frac)+1))
    for k, (a, f) in enumerate(zip(history_amp, history_frac)):
        ax.annotate(
            f"{a:.1e}", (k+1, f), textcoords="offset points", xytext=(0, 8),
            fontsize=6, ha="center"
        )
    ax.set_title(f"UL={amp_final:.2e}")
    ax.legend(frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# UL iteration plot
# ---------------------------------------------------------------------------

def plot_ul_iteration(
    ax1, ax2, lambda_bkg, lambda_iter, lambda_real, lambda_bkg_m, target,
    history_amp, history_frac, amp_mid, amp_lo, amp_hi, frac, cl,
    p_value, significance, iteration, max_iter, energy_edges, precision
):
    lmin = min(lambda_bkg.min(), lambda_iter.min())
    lmax = max(lambda_bkg.max(), lambda_iter.max())
    bins = np.linspace(lmin - 0.1*abs(lmin), lmax + 0.1*abs(lmax), 60)

    ax1.hist(
        lambda_bkg, bins, color="royalblue", density=True, histtype="stepfilled",
        alpha=0.7, label=f"BKG N={len(lambda_bkg)}"
    )
    ax1.hist(
        lambda_iter, bins, color="k", density=True, histtype="step", lw=1.5,
        label=f"iter {iteration+1}  Amp={amp_mid:.2e}  N={len(lambda_iter)}"
    )
    ax1.axvline(target, color="r", ls="-", lw=2, label=f"Target={lambda_real:.2f}")
    ax1.text(
        0.97, 0.95,
        f"frac={frac:.3f}  CL={cl}\np-val={p_value:.3f}  sig={significance:.2f}$\\sigma$",
        ha="right", va="top", transform=ax1.transAxes, fontsize=7
    )
    ax1.set(xlabel="$\\Lambda$", ylabel="Normalized counts")
    ax1.set_title(f"Iter {iteration+1}/{max_iter} - [{amp_lo:.1e}, {amp_hi:.1e}]")
    ax1.legend(frameon=False, fontsize=7, loc=2)

    _plot_convergence_ax(
        ax2, history_amp, history_frac, amp_lo, amp_hi, np.sqrt(amp_lo*amp_hi), cl, precision
    )


# ---------------------------------------------------------------------------
# Summary plots
# ---------------------------------------------------------------------------

def summary_folded_counts(dataset, bin_c_ra, obs, i, n_obs):
    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(bin_c_ra))

    ax.plot(x, dataset.background.data.sum(axis=0).sum(axis=0), "r--")
    ax.plot(x, dataset.counts.data.sum(axis=0).sum(axis=0),     "r", label="Folded RA")
    ax.plot(x, dataset.background.data.sum(axis=0).sum(axis=1), "b--")
    ax.plot(x, dataset.counts.data.sum(axis=0).sum(axis=1),     "b", label="Folded DEC")

    ax.plot([], [], "k-",  label="Counts")
    ax.plot([], [], "k--", label="BKG model")

    ax.set(
        title=f"{i+1}/{n_obs}, bkg norm Run {obs.obs_id}",
        xlabel="Spatial bins RA / DEC",
        ylabel="BKG Rate",
    )
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
        "Exposure [m${}^2$ s]":      (dataset.exposure.data.sum(axis=0),   "viridis", None, None),
        "Excess":    (dataset.excess.data.sum(axis=0), "coolwarm",
                      -abs(dataset.excess.data).max(), abs(dataset.excess.data).max()),
    }

    for ax, (label, (data, cmap, vmin, vmax)), c in zip(
        axes, data_dict.items(), ["0.8", "0.8", "0.8", "0.3"]
    ):
        mesh = ax.pcolormesh(
            bin_edges_ra, bin_edges_dec, data, cmap=cmap,
            vmin=vmin, vmax=vmax, transform=ax.get_transform("icrs")
        )
        fig.colorbar(mesh, ax=ax, label=label)

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
    trans   = ax.get_transform("icrs")

    pc = ax.pcolormesh(
        bin_edges_ra, bin_edges_dec, prob_gw, cmap="cylon",
        transform=ax.get_transform("icrs")
    )
    fig.colorbar(pc, label="GW Prob")

    ax.set(title=f"$\\sum P=${np.sum(prob_gw)*100:.2f}%")
    ax.coords[0].set_axislabel_position("b"); ax.coords[0].set_ticklabel_position("b")
    ax.coords[0].set_axislabel("RA [deg]");   ax.coords[1].set_axislabel("DEC [deg]")
    plt.show()


def summary_geometry(
    geom, bin_edges_ra, bin_edges_dec, size_fov, data_ligo_2d, threshold_maps, correlation_radius
):
    fig, ax = plt.subplots(figsize=(2.1, 2), subplot_kw={"projection": geom.wcs})
    trans   = ax.get_transform("icrs")

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

    frac_pos = (0.15, 0.85)
    circle_corr = Circle(
        ((1-frac_pos[0]) * bin_edges_ra.min() + (frac_pos[0]) * bin_edges_ra.max(), 
         (1-frac_pos[1]) * bin_edges_dec.min() + (frac_pos[1]) * bin_edges_dec.max()), 
        correlation_radius.value, edgecolor="b", facecolor="none", alpha=1.0, lw=1,
        transform=trans, zorder=10, label=f"Corr R {correlation_radius}"
    )
    ax.add_patch(circle_corr)
    
    lims = ax.get_xlim(), ax.get_ylim()

    if threshold_maps is not None:
        ax.contour(
            data_ligo_2d, levels=threshold_maps, origin="lower",
            extent=[-180, 180, -90, 90], colors="k", linewidths=1, transform=trans
        )
        ax.plot([], [], lw=1.5, color="k", label="GW 50, 95%")
    else:
        hot_idx = np.unravel_index(np.argmax(data_ligo_2d), data_ligo_2d.shape)
        hot_ra  = np.linspace(-180, 180, data_ligo_2d.shape[1])[hot_idx[1]]
        hot_dec = np.linspace(-90,   90, data_ligo_2d.shape[0])[hot_idx[0]]
        ax.plot(hot_ra, hot_dec, "kx", ms=6, transform=trans, label="GW ($\\delta$)")

    ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
    ax.plot([], [], lw=1, color="0.7", label="Spatial bins")
    ax.legend(frameon=False, loc=(1.03, 0))
    plt.show()

def summary_ts_maps(
    geom, bin_c_ra, bin_c_dec, ts, log_gw, ts2, data_ligo_2d, threshold_maps, 
    source_coord, lambda_real, lambda_coord_real, axis_energy
):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9, 2.6), subplot_kw={"projection": geom.wcs})

    # --- Colormeshes ---
    tsmax = np.max(np.abs([np.nanmax(ts), np.nanmin(ts)]))
    a1 = ax1.pcolormesh(
        bin_c_ra, bin_c_dec, ts, shading="nearest", transform=ax1.get_transform("icrs"), cmap="seismic", vmin=-tsmax, vmax=tsmax
    )
    a2 = ax2.pcolormesh(
        bin_c_ra, bin_c_dec, log_gw, shading="nearest", transform=ax2.get_transform("icrs"), cmap="cylon",   vmin=-25, vmax=-13
    )
    a3 = ax3.pcolormesh(
        bin_c_ra, bin_c_dec, ts2,    shading="nearest", transform=ax3.get_transform("icrs"), cmap="viridis", vmin=-18
    )

    fig.colorbar(a1, ax=ax1, label=r"TS$=2\log\left(\frac{\mathcal{L}(n|n)}{\mathcal{L}(n|n_{BKG})}\right)$")
    fig.colorbar(a2, ax=ax2, label=r"$2\log(P_{GW})$",    extend="min")
    fig.colorbar(a3, ax=ax3, label=r"TS$+2\log(P_{GW})$", extend="min")

    # --- Per-axis Decorations ---
    for ax in (ax1, ax2, ax3):
        icrs = ax.get_transform("icrs")
        lims = ax.get_xlim(), ax.get_ylim()
        if threshold_maps is not None:
            ax.contour(data_ligo_2d, levels=threshold_maps, transform=icrs,
                       origin="lower", linewidths=1, alpha=0.5,
                       extent=[-180, 180, -90, 90],
                       colors="k" if ax is ax1 else "w")
        delta = 1.0
        ax.set_xlim(lims[0][0] + delta, lims[0][1] - delta)
        ax.set_ylim(lims[1][0] + delta, lims[1][1] - delta)
        ax.plot(source_coord.ra.deg, source_coord.dec.deg, "x", transform=icrs,
                label="Pointing" if ax is ax2 else None, color="k", ls="")
        ax.set_xlabel("RA [deg]")
        ax.grid()
        ax.coords[0].set_axislabel_position("b")
        ax.coords[0].set_ticklabel_position("b")
        ax.coords[0].set_axislabel("RA [deg]")  # force label on WCS axis

    # --- Max Marker & Labels ---
    ax3.plot(
        lambda_coord_real.ra.deg, lambda_coord_real.dec.deg, 
        "x", color="r", ls="", transform=ax3.get_transform("icrs"), label=f"Maximum = {lambda_real:.2f}"
    )

    for dec_coord in (ax2.coords[1], ax3.coords[1]):
        dec_coord.set_ticks_visible(False)
        dec_coord.set_ticklabel_visible(False)

    e_lo, e_hi = axis_energy.edges[0], axis_energy.edges[-1]
    ax1.set(ylabel="DEC [deg]", title=f"TS ({e_lo:.2f}–{e_hi:.2f})")
    ax1.coords[1].set_axislabel("DEC [deg]")  # force label on WCS axis
    ax2.set_title("GW PDF");  ax2.legend(frameon=False, loc=2, fontsize=8)
    ax3.set_title("TS'")
    l3 = ax3.legend(frameon=False, loc=2, fontsize=8)
    [text.set_color("w") for text in l3.get_texts()]

    plt.tight_layout()
    plt.show()

def summary_flux_distributions(flux_uls, fluxs, flux_uls_95, fluxs_95, energy_edges, mask_threshold_95):
    """Plot flux and UL distributions; use vertical lines if only one pixel in 95% region."""
    fig, ax = plt.subplots(figsize=(3, 2))

    h = ax.hist(fluxs,    40, histtype="stepfilled", color="dodgerblue", label="Total flux")
    ax.hist(flux_uls, 40, histtype="stepfilled", color="k", alpha=0.3, label="Total UL")

    single_pixel = np.sum(~np.isnan(flux_uls_95)) <= 1

    if single_pixel:
        val_ul  = flux_uls_95[~np.isnan(flux_uls_95)]
        val_fl  = fluxs_95[~np.isnan(fluxs_95)]
        if len(val_ul): ax.axvline(val_ul[0], color="k",        ls="--", label="UL in 95%")
        if len(val_fl): ax.axvline(val_fl[0], color="darkblue", ls="--", label="Flux in 95%")
    else:
        ax.hist(flux_uls_95, h[1], histtype="step", color="k",        label="ULs in 95%")
        ax.hist(fluxs_95,    h[1], histtype="step", color="darkblue", label="Flux in 95%")

    n_pix = np.sum(mask_threshold_95)
    ax.set(xlabel="Flux [1 / (s cm²)]", ylabel="Counts", yscale="log")
    ax.set_title(f"{energy_edges[0]:.2f}–{energy_edges[1]:.2f} | {n_pix} px in 95%")
    ax.legend(frameon=False, loc=(1.03, 0))
    plt.tight_layout()
    plt.show()

def plot_flux_ul_map_and_distributions(geom, bin_edges_ra, bin_edges_dec, flux_ul_smoothed,
                                       source_coord, source_name, fluxs, fluxs_95,
                                       flux_uls, flux_uls_95, lvk_prob_2d, threshold_maps,
                                       energy_edges):
    """Real-data flux-UL sky map next to the flux/UL distributions, in and out of the 95% mask."""
    fig = plt.figure(figsize=(9.3, 3))
    ax1 = plt.subplot(121, projection=geom.wcs)
    ax2 = plt.subplot(122)

    p_sig = ax1.pcolormesh(bin_edges_ra, bin_edges_dec, flux_ul_smoothed, cmap="viridis",
                           transform=ax1.get_transform("icrs"))
    fig.colorbar(p_sig, ax=ax1, label="Flux UL  [1 / (s cm$^2$)]")
    ax1.plot(source_coord.ra, source_coord.dec, "xw", label=source_name,
             transform=ax1.get_transform("icrs"))

    h = ax2.hist(fluxs, 40, histtype="stepfilled", color="dodgerblue", label="Total flux distribution")
    ax2.hist(fluxs_95, h[1], histtype="step", color="darkblue", label="Flux in 95%")
    h = ax2.hist(flux_uls_95, 40, histtype="step", color="k", label="ULs in 95%")
    ax2.hist(flux_uls, h[1], histtype="stepfilled", color="k", alpha=0.3, label="Total UL distribution")

    lims = ax1.get_xlim(), ax1.get_ylim()
    ax1.contour(lvk_prob_2d, levels=threshold_maps, transform=ax1.get_transform("icrs"),
               colors="0.9", origin="lower", extent=[-180, 180, -90, 90])
    ax1.set_xlim(*lims[0]); ax1.set_ylim(*lims[1]); ax1.grid(alpha=0.7)
    ax2.set(xlabel="Flux UL [1 / (s cm$^2$)]", ylabel="Counts", yscale="log")
    ax2.set_title(f"{energy_edges[0]:.2f}–{energy_edges[1]:.2f}")
    ax1.coords[0].set_axislabel_position("b"); ax1.coords[0].set_ticklabel_position("b")
    ax1.coords[0].set_axislabel("RA [deg]"); ax1.coords[1].set_axislabel("DEC [deg]")
    ax2.legend(loc=(1.01, 0), frameon=False)
    fig.tight_layout()
    plt.show()


def plot_significance_map_and_distribution(geom, bin_edges_ra, bin_edges_dec, sqrt_ts_smoothed,
                                            vmin, vmax, source_coord, source_name, sigs, sigs_95,
                                            lvk_prob_2d, threshold_maps):
    """Real-data significance sky map next to the sqrt(TS) distribution vs. N(0, 1)."""
    mu_fit, std_fit = norm.fit(sigs)
    x_pdf = np.linspace(-7, 7, 100)
    pdf_norm = norm.pdf(x_pdf, 0, 1)

    fig = plt.figure(figsize=(8.5, 3))
    ax1 = plt.subplot(121, projection=geom.wcs)
    ax2 = plt.subplot(122)

    p_sig = ax1.pcolormesh(bin_edges_ra, bin_edges_dec, sqrt_ts_smoothed, cmap="coolwarm",
                           transform=ax1.get_transform("icrs"), vmax=vmax, vmin=vmin)
    fig.colorbar(p_sig, ax=ax1, label="Significance [$\\sigma$]")
    ax1.plot(source_coord.ra, source_coord.dec, "xk", label=source_name,
             transform=ax1.get_transform("icrs"))

    h = ax2.hist(sigs, 40, density=True, color="dodgerblue", lw=0, histtype="stepfilled", label="All bins")
    ax2.hist(sigs_95, h[1], density=True, color="darkblue", lw=1.5, histtype="step", label="95% bins")
    ax2.plot(x_pdf, pdf_norm, lw=1.6, ls="-", color="k", label="$N(0, 1)$")

    lims = ax1.get_xlim(), ax1.get_ylim()
    ax1.contour(lvk_prob_2d, levels=threshold_maps, transform=ax1.get_transform("icrs"),
               colors="0.2", origin="lower", extent=[-180, 180, -90, 90])
    ax1.set_xlim(*lims[0]); ax1.set_ylim(*lims[1]); ax1.grid(alpha=0.7)
    ax2.set(yscale="log", ylim=1e-3, xlim=(-5.5, 5.5),
            xlabel="Significance [$\\sigma$]", ylabel="Norm Counts")
    ax2.set_title(f"$\\mu$ = {mu_fit:.2f}, $\\sigma$ = {std_fit:.2f}, "
                 f"$S^{{MAX}}$ = {np.nanmax(sigs):.2f}$\\sigma$")
    ax1.coords[0].set_axislabel_position("b"); ax1.coords[0].set_ticklabel_position("b")
    ax1.coords[0].set_axislabel("RA [deg]"); ax1.coords[1].set_axislabel("DEC [deg]")
    ax2.legend(loc=(1.01, 0), frameon=False)
    fig.tight_layout()
    plt.show()


def summary_bkg_simulations(geom, bin_edges_ra, bin_edges_dec, data_ligo_2d, threshold_maps,
                             lambda_bkg, lambda_real, lambda_bkg_m, p_value, significance,
                             map_lambda_bkg, map_ts_bkg, map_source_sim, energy_edges, bins_lambda):

    fig = plt.figure(figsize=(7, 6))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[2, 1])
    axt = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0], projection=geom.wcs)
    ax2 = fig.add_subplot(gs[1, 1], projection=geom.wcs)
    ax3 = fig.add_subplot(gs[1, 2], projection=geom.wcs)

    # --- Top: Lambda Distribution ---
    axt.hist(lambda_bkg, bins_lambda, color="royalblue", density=True,
             histtype="stepfilled", label=f"BKG sim\nN={len(lambda_bkg)}")
    axt.axvline(lambda_real,  color="k",        ls="--", label=f"Real data\n$\\Lambda=${lambda_real:.2f}")
    axt.axvline(lambda_bkg_m, color="darkblue", ls=":",  label=f"BKG median\n$\\Lambda=${lambda_bkg_m:.2f}")
    axt.text(0.85, 0.87, f"p-value={p_value:.2f}\nsignificance={significance:.2f} $\\sigma$",
             ha="right", va="center", transform=axt.transAxes)
    axt.set(xlabel=r"$\Lambda$", ylabel="Normalized counts",
            title=f"{energy_edges[0]:.2f}–{energy_edges[1]:.2f}")
    axt.legend(frameon=False, loc=4)

    # --- Bottom: Sky Maps ---
    panels = [
        (ax1, "magma",   map_lambda_bkg,  r"$\Lambda_{BKG}$ ($TS'_{max}$)"),
        (ax2, "magma",   map_ts_bkg,      r"$TS_{max}$"),
        (ax3, "cividis", map_source_sim,  r"Injected source $(F=0)$"),
    ]
    for ax, cmap, mapa, title in panels:
        ax.pcolormesh(bin_edges_ra, bin_edges_dec, mapa.data,
                      cmap=cmap, transform=ax.get_transform("icrs"))
        lims = ax.get_xlim(), ax.get_ylim()
        if threshold_maps is not None:
            ax.contour(data_ligo_2d, levels=threshold_maps, transform=ax.get_transform("icrs"),
                       origin="lower", extent=[-180, 180, -90, 90], colors="w", linewidths=1)
        ax.set_xlim(*lims[0]); ax.set_ylim(*lims[1])
        ax.coords[0].set_axislabel_position("b")
        ax.coords[0].set_ticklabel_position("b")
        ax.coords[0].set_axislabel("RA [deg]")
        ax.set_title(title)

    ax1.coords[1].set_axislabel("DEC [deg]")
    for ax in (ax2, ax3):
        ax.coords[1].set_ticks_visible(False)
        ax.coords[1].set_ticklabel_visible(False)

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Distance-prior diagnostics (3D)
# ---------------------------------------------------------------------------

def plot_distance_prior_summary(r_grid, cdf_table, cdf_valid, prob_gw_integrated,
                                mask_threshold_95, dist_med, dist_lo, dist_hi,
                                n_pix_per_bin, save_path=None, pdf_bins="mask"):
    """
    4-panel (2x2) distance-prior diagnostic: median / 90%-width / HEALPix-density
    maps, plus per-bin distance PDFs (transparency by GW weight) and their
    probability-weighted mixture.

    pdf_bins : {"mask", "all"}
        Which bins' PDFs are drawn in panel 4: only those in the 95% mask, or
        every bin with a usable CDF (then the all-bins mixture is overlaid too).

    Returns the 95%-mask mixture PDF (`pdf_gw_mask`, on `r_grid`) regardless of
    `pdf_bins`, so callers can reuse the exact curve plotted here.
    """
    if pdf_bins not in ("mask", "all"):
        raise ValueError(f"pdf_bins must be 'mask' or 'all', got {pdf_bins!r}")
    in_mask = pdf_bins == "mask"

    dist_med_m = np.where(cdf_valid, dist_med, np.nan)
    dist_lo_m  = np.where(cdf_valid, dist_lo,  np.nan)
    dist_hi_m  = np.where(cdf_valid, dist_hi,  np.nan)
    width90_m  = dist_hi_m - dist_lo_m

    fig, ax = plt.subplots(2, 2, figsize=(7, 6)); ax = ax.ravel()
    im = ax[0].imshow(dist_med_m.T, origin="lower", cmap="viridis")
    plt.colorbar(im, ax=ax[0], label="median distance [Mpc]")
    im = ax[1].imshow(width90_m.T, origin="lower", cmap="magma")
    plt.colorbar(im, ax=ax[1], label="90% width [Mpc]")
    im = ax[2].imshow(n_pix_per_bin.T, origin="lower", cmap="cividis")
    plt.colorbar(im, ax=ax[2], label="HEALPix pixels / WCS bin")

    # Bins in the 95% mask (-> returned mixture) and bins drawn (mask or all valid)
    mask_pdf = mask_threshold_95 & cdf_valid & (prob_gw_integrated > 0)
    show = mask_pdf if in_mask else cdf_valid
    w_mask = np.where(mask_pdf, prob_gw_integrated, 0.0).astype(float); w_mask /= w_mask.sum()
    w_show = np.where(show,     prob_gw_integrated, 0.0).astype(float); w_show /= w_show.sum()
    alpha = 0.04 + 0.56 * w_show / (w_show.max() or 1)

    pdf_mask = np.zeros_like(cdf_table, dtype=float)
    for i, j in np.argwhere(show):
        pdf = np.gradient(cdf_table[i, j].astype(float), r_grid)
        area = np.trapezoid(pdf, r_grid)
        if area > 0 and np.isfinite(area):
            pdf_mask[i, j] = pdf / area
            ax[3].plot(r_grid, pdf_mask[i, j], color="tab:blue", alpha=alpha[i, j], lw=0.8)

    mix = lambda w: np.sum(pdf_mask * w[:, :, None], axis=(0, 1))
    pdf_gw_mask, pdf_show = mix(w_mask), mix(w_show)
    ax[3].plot(r_grid, pdf_gw_mask, "k--", lw=1.5, label="GW-weighted mixture (95% mask)")
    if not in_mask:
        ax[3].plot(r_grid, pdf_show, ":", color="tab:red", lw=1.5, label="GW-weighted mixture (all bins)")
    ax[3].set_xlabel("r [Mpc]"); ax[3].set_ylabel("p(r)")
    ax[3].legend(frameon=False, fontsize=8, loc=0)

    mean = np.trapezoid(r_grid * pdf_show, r_grid)
    std = np.sqrt(np.trapezoid((r_grid - mean) ** 2 * pdf_show, r_grid))
    clipped_bounds = np.clip((mean - 4 * std, mean + 4 * std), r_grid[0], r_grid[-1])
    where = "in 95% mask" if in_mask else "in all valid bins"
    ax[3].set(title=f"Distance PDFs {where} ({show.sum()} bins)", xlim=clipped_bounds)

    for a in ax[:3]:
        try:
            a.contour(prob_gw_integrated.T,
                      levels=np.array([0.1, 0.5]) * np.nanmax(prob_gw_integrated),
                      colors="w", linewidths=0.7, alpha=0.7)
            a.set(xlabel="x-bins", ylabel="y-bins")
        except ValueError:
            pass  # degenerate levels (e.g. a Dirac-delta single-pixel map)

    fig.suptitle(f"{cdf_valid.sum()}/{cdf_valid.size} WCS bins have a usable distance CDF")
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    plt.show()

    # Returned so the BKG-simulation animation (whose injected positions are
    # also drawn from mask_threshold_95) can overlay the SAME mixture PDF
    # instead of silently re-deriving -- or worse, mismatching -- it.
    return pdf_gw_mask


# ---------------------------------------------------------------------------
# BKG simulation animation
# ---------------------------------------------------------------------------

def animate_bkg_simulations(
    geom, geom_image, bin_edges_ra, bin_edges_dec, bin_c_ra, bin_c_dec,
    r_grid, cdf_table, cdf_valid, f_ra_bkg, f_dec_bkg,
    lambda_bkg, lambda_ra_bkg, lambda_dec_bkg, lambda_real, bins_lambda,
    threshold_maps, lvk_prob_2d, energy_edges, path_gif,
    sampling_seed=12345, n_frames=150, n_hold=25, fps=15, pdf_gw_mask=None,
):
    """
    Animated build-up of the BKG Lambda distribution, one realisation at a
    time: the Lambda histogram filling in, injected (F=0) source positions,
    a distance drawn per realisation (purely illustrative -- BKG
    realisations carry no injected source) from its bin's distance CDF, and
    a cumulative map of where Lambda's argmax has landed.

    Returns the array of per-realisation sampled distances.
    """
    from matplotlib.animation import PillowWriter
    from IPython.display import Image, display

    rng_anim = np.random.default_rng(sampling_seed)
    f_ra_a, f_dec_a = np.asarray(f_ra_bkg, float), np.asarray(f_dec_bkg, float)
    n_anim = len(lambda_bkg)

    assert bin_c_ra.shape == cdf_table.shape[:2], (bin_c_ra.shape, cdf_table.shape)
    dra  = (f_ra_a[:, None] - bin_c_ra.ravel()[None, :] + 180) % 360 - 180
    ddec =  f_dec_a[:, None] - bin_c_dec.ravel()[None, :]
    flat = np.argmin((dra * np.cos(np.deg2rad(f_dec_a))[:, None]) ** 2 + ddec ** 2, axis=1)
    src_i, src_j = np.unravel_index(flat, bin_c_ra.shape)

    cdf_src  = np.maximum.accumulate(np.nan_to_num(cdf_table[src_i, src_j].astype(float)), axis=1)
    norm_src = cdf_src[:, -1]
    ok_src   = cdf_valid[src_i, src_j] & (norm_src > 0)
    u_draw   = rng_anim.uniform(size=n_anim)
    dist_anim = np.full(n_anim, np.nan)
    for k in np.flatnonzero(ok_src):
        dist_anim[k] = np.interp(u_draw[k] * norm_src[k], cdf_src[k], r_grid)

    lam_ix, lam_iy = geom_image.coord_to_idx({"lon": np.asarray(lambda_ra_bkg) * u.deg,
                                              "lat": np.asarray(lambda_dec_bkg) * u.deg})
    lam_ix, lam_iy = np.asarray(lam_ix, int), np.asarray(lam_iy, int)
    ny_img, nx_img = geom_image.data_shape
    lam_ok   = (lam_ix >= 0) & (lam_iy >= 0)
    lam_flat = np.where(lam_ok, lam_iy * nx_img + lam_ix, -1)

    print(f"Distances drawn: {np.isfinite(dist_anim).sum()}/{n_anim}  "
          f"(median {np.nanmedian(dist_anim):.0f} Mpc, 90% [{np.nanpercentile(dist_anim, 5):.0f}, "
          f"{np.nanpercentile(dist_anim, 95):.0f}])")
    print(f"Lambda positions inside the map: {lam_ok.sum()}/{n_anim}")

    os.makedirs(os.path.dirname(path_gif), exist_ok=True)
    frames_k = np.unique(np.geomspace(1, n_anim, n_frames).astype(int))   # slow start, fast end
    frames_k = np.concatenate([frames_k, np.full(n_hold, n_anim)])        # hold the last frame

    lam_all   = np.asarray(lambda_bkg, float)
    fin_dist  = np.isfinite(dist_anim)
    bins_dist = np.linspace(*np.nanpercentile(dist_anim, [0.5, 99.5]), 50)
    h_lam_final,  _ = np.histogram(lam_all, bins_lambda)
    h_dist_final, _ = np.histogram(dist_anim[fin_dist], bins_dist)
    map_vmax = max(np.bincount(lam_flat[lam_ok], minlength=ny_img * nx_img).max(), 1)

    fig = plt.figure(figsize=(10, 6.5))
    gs  = GridSpec(2, 3, figure=fig, height_ratios=[2, 1.3])
    axt = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0], projection=geom.wcs)
    ax2 = fig.add_subplot(gs[1, 1])
    ax3 = fig.add_subplot(gs[1, 2], projection=geom.wcs)

    axt.stairs(h_lam_final, bins_lambda, color="royalblue", alpha=0.3, lw=1)
    st_lam = axt.stairs(np.zeros(len(bins_lambda) - 1), bins_lambda, fill=True,
                        color="royalblue", label="BKG sim")
    axt.axvline(lambda_real, color="k", ls="--", label=f"Real data\n$\\Lambda=${lambda_real:.2f}")
    ln_med  = axt.axvline(lam_all[0], color="darkblue", ls=":", label="BKG median")
    txt     = axt.text(0.85, 0.85, "", ha="center", va="center", transform=axt.transAxes)
    counter = axt.text(0.02, 0.9, "", transform=axt.transAxes, fontsize=12, weight="bold")
    x_lo, x_hi = min(bins_lambda[0], lambda_real), max(bins_lambda[-1], lambda_real)
    pad = 0.03 * (x_hi - x_lo)
    axt.set(xlim=(x_lo - pad, x_hi + pad), ylim=(0, 1.15 * h_lam_final.max()),
            xlabel=r"$\Lambda$", ylabel="Counts", title=f"{energy_edges[0]:.2f}–{energy_edges[1]:.2f}")
    axt.legend(frameon=False, loc=4)

    sc_old = ax1.scatter([], [], s=3, c="royalblue", alpha=0.5, lw=0, transform=ax1.get_transform("icrs"))
    sc_new = ax1.scatter([], [], s=18, c="crimson", lw=0, transform=ax1.get_transform("icrs"))

    st_dist = ax2.stairs(np.zeros(len(bins_dist) - 1), bins_dist, fill=True, color="crimson", alpha=0.5)
    ymax_dist = h_dist_final.max()
    if pdf_gw_mask is not None:
        curve = pdf_gw_mask * fin_dist.sum() * np.diff(bins_dist)[0]
        ax2.plot(r_grid, curve, "k--", lw=1)
        ax2.legend(frameon=False, fontsize=7, loc=1)
        ymax_dist = max(ymax_dist, curve[(r_grid >= bins_dist[0]) & (r_grid <= bins_dist[-1])].max())
    ax2.set(xlim=(bins_dist[0], bins_dist[-1]), ylim=(0, 1.2 * ymax_dist),
            xlabel=r"$d_L$ [Mpc]", ylabel="Counts", title=r"Distance $p(r\,|\,\mathrm{pix})$")

    mesh = ax3.pcolormesh(bin_edges_ra, bin_edges_dec, np.zeros((ny_img, nx_img)), cmap="magma",
                          vmin=0, vmax=map_vmax, transform=ax3.get_transform("icrs"))

    for ax, c_cont, title in [(ax1, "0.3", r"Source positions $(F=0)$"),
                              (ax3, "w",   r"$\Lambda_{BKG}$ ($TS'_{max}$)")]:
        if threshold_maps is not None:
            ax.contour(lvk_prob_2d, levels=threshold_maps, transform=ax.get_transform("icrs"),
                       origin="lower", extent=[-180, 180, -90, 90], colors=c_cont, linewidths=1)
        ax.set_xlim(-0.5, nx_img - 0.5); ax.set_ylim(-0.5, ny_img - 0.5)
        ax.coords[0].set_axislabel_position("b"); ax.coords[0].set_ticklabel_position("b")
        ax.coords[0].set_axislabel("RA [deg]")
        ax.set_title(title)
    ax1.coords[1].set_axislabel("DEC [deg]")
    ax3.coords[1].set_ticks_visible(False); ax3.coords[1].set_ticklabel_visible(False)
    fig.tight_layout()

    writer, k_prev = PillowWriter(fps=fps), 0
    with writer.saving(fig, path_gif, dpi=100):
        for n, k in enumerate(frames_k):
            k = int(k)
            lam_k = lam_all[:k]

            st_lam.set_data(np.histogram(lam_k, bins_lambda)[0])
            med_k = np.median(lam_k); ln_med.set_xdata([med_k, med_k])
            p_k = np.mean(lam_k > lambda_real)
            sig_k = (f"> {norm.ppf(1 - 1 / (k + 1)):.2f}" if p_k == 0 else
                     f"< {norm.ppf(1 / (k + 1)):.2f}"     if p_k == 1 else f"{norm.ppf(1 - p_k):.2f}")
            txt.set_text(f"p-value = {p_k:.3f}\nsignificance = {sig_k} $\\sigma$\n"
                         f"median $\\Lambda$ = {med_k:.2f}")
            counter.set_text(f"N = {k}/{n_anim}")

            sc_old.set_offsets(np.column_stack([f_ra_a[:k], f_dec_a[:k]]))
            if k > k_prev:
                sc_new.set_offsets(np.column_stack([f_ra_a[k_prev:k], f_dec_a[k_prev:k]]))
            sc_new.set_visible(k > k_prev)

            d_k = dist_anim[:k]
            st_dist.set_data(np.histogram(d_k[np.isfinite(d_k)], bins_dist)[0])

            mesh.set_array(np.bincount(lam_flat[:k][lam_ok[:k]], minlength=ny_img * nx_img)
                           .reshape(ny_img, nx_img))

            writer.grab_frame()
            k_prev = k
            print(f"Frame {n + 1}/{len(frames_k)}", end="\r")

    plt.close(fig)
    display(Image(filename=path_gif))
    return dist_anim


# ---------------------------------------------------------------------------
# Iterative upper-limit diagnostics (shared by the 2D and 3D bisections)
# ---------------------------------------------------------------------------

def plot_ul_convergence(hist_x, hist_frac, hist_err, x_ul, cl, x_ul_lo_1sigma=None,
                        x_ul_hi_1sigma=None, label="\\phi_0", unit="cm$^{-2}$s$^{-1}$TeV$^{-1}$",
                        method_label="2D", converged=None, bracket_width_dex=None,
                        n_monotonicity_violations=0, at_bracket_edge=False,
                        bracket_warning=None):
    """
    Two-panel bisection diagnostic, shared by the 2D flux and 3D luminosity
    upper limits: the convergence trace (fraction and tested x per
    iteration) and the crossing curve that actually defines the limit.
    """
    hist_x   = np.asarray(hist_x)
    hist_frac = np.asarray(hist_frac)
    hist_err  = np.zeros_like(hist_frac) if hist_err is None else np.asarray(hist_err)
    iters = np.arange(1, len(hist_frac) + 1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 3.6))

    ax2 = axL.twinx()
    ax2.plot(iters, hist_x, color="steelblue", lw=1.4, ls="--", marker="s", ms=4,
             alpha=0.7, label=f"${label}$")
    ax2.set_ylabel(f"Tested ${label}$ [{unit}]", color="steelblue", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="steelblue"); ax2.set_yscale("log")

    axL.errorbar(iters, hist_frac, yerr=hist_err, fmt="ko-", ms=5, lw=1.6,
                 capsize=2, zorder=3, label="Fraction $\\pm$ MC error")
    axL.axhline(cl, color="r", ls="--", lw=1.5, label=f"CL = {cl}")
    axL.set_xlabel("Iteration"); axL.set_ylabel("Fraction > target")
    axL.set_xticks(list(iters))
    axL.set_title(f"{method_label} convergence - ${label}^{{UL}}$ = {x_ul:.3e}")
    axL.set_zorder(ax2.get_zorder() + 1); axL.patch.set_visible(False)
    l1, lab1 = axL.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    axL.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=8)

    order = np.argsort(hist_x)
    axR.errorbar(hist_x[order], hist_frac[order], yerr=hist_err[order], fmt="o-",
                 color="k", ms=4, lw=1.2, capsize=2, label="simulated")
    axR.axhline(cl, color="r", ls="--", lw=1.4, label=f"CL = {cl}")
    axR.axvline(x_ul, color="g", lw=1.6, label=f"UL = {x_ul:.3e}")
    if x_ul_lo_1sigma is not None and np.isfinite(x_ul_lo_1sigma):
        axR.axvspan(x_ul_lo_1sigma, x_ul_hi_1sigma, color="g", alpha=0.15, label="MC 1$\\sigma$")
    axR.set_xscale("log"); axR.set_xlabel(f"${label}$ [{unit}]")
    axR.set_ylabel("Fraction > target"); axR.set_ylim(-0.03, 1.03)
    axR.set_title("Crossing curve"); axR.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout(); plt.show()

    if converged is not None:
        print(f"Converged: {converged}   bracket width {bracket_width_dex:.4f} dex   "
              f"monotonicity violations: {n_monotonicity_violations}")
        if bracket_warning:
            print(f"BRACKET PROBLEM: {bracket_warning}")
        if at_bracket_edge:
            print(f"WARNING: the limit sits at the edge of the tested {label} range.")


def plot_lambda_distributions_by_x(lambda_bkg, lambda_real, lambda_bkg_m,
                                   lambda_cache_by_x, target_lambda, cl,
                                   x_label="L_0", x_unit="erg s$^{-1}$"):
    """
    Lambda distribution for every x (amplitude or luminosity) tested during
    a bisection, colour-coded by x, against the BKG distribution -- plus the
    frac(x) table used to check that the tested values straddle the target
    and that the median Lambda grows monotonically with x.
    """
    xs_tested = np.array(sorted(lambda_cache_by_x.keys()))
    all_l = np.concatenate(
        [lambda_bkg, [lambda_real, lambda_bkg_m]] + [lambda_cache_by_x[x] for x in xs_tested])
    bins_all = np.linspace(np.nanmin(all_l), np.nanmax(all_l), 60)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.hist(lambda_bkg, bins_all, density=True, histtype="stepfilled",
            color="0.85", label=f"BKG sim (N={len(lambda_bkg)})", zorder=-10)

    cmap = plt.cm.plasma
    lognorm_c = plt.Normalize(np.log10(xs_tested.min()), np.log10(xs_tested.max()))
    for x in xs_tested:
        ax.hist(lambda_cache_by_x[x], bins_all, density=True, histtype="step",
                color=cmap(lognorm_c(np.log10(x))), lw=1.3, alpha=0.9)

    ax.axvline(lambda_real,  color="k",        ls="--", label=f"Real data  $\\Lambda$={lambda_real:.2f}")
    ax.axvline(lambda_bkg_m, color="darkblue", ls=":",  label=f"BKG median  $\\Lambda$={lambda_bkg_m:.2f}")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=lognorm_c)
    cbar = fig.colorbar(sm, ax=ax); cbar.set_label(f"$\\log_{{10}}({x_label}$ / {x_unit})")
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("$\\Lambda$"); ax.set_ylabel("Normalized counts")
    ax.set_title(f"$\\Lambda$ distributions for every ${x_label}$ probed ({len(xs_tested)} steps)")
    plt.tight_layout(); plt.show()

    fracs = np.array([np.mean(lambda_cache_by_x[x] > target_lambda) for x in xs_tested])
    meds  = np.array([np.median(lambda_cache_by_x[x]) for x in xs_tested])
    print(f"{x_label} [{x_unit}]        frac > target   median Lambda")
    for x, f, m in zip(xs_tested, fracs, meds):
        print(f"  {x:.4e}      {f:6.3f}        {m:8.3f}")
    print(f"\nTarget Lambda = {target_lambda:.3f}")
    print(f"Tested {x_label} straddle the CL: {fracs.min() <= cl <= fracs.max()}")
    print(f"Median Lambda monotonic in {x_label}: "
          f"{bool(np.all(np.diff(meds) >= -0.05 * np.abs(meds[:-1]).max()))}")


def plot_closure_test(lambda_bkg, lambda_clo, target_lambda, frac_clo, err_clo, cl,
                      d_clo, photon_flux_clo, flux_ul_value, x_label="L_0"):
    """
    3-panel closure-test figure: an independent Lambda sample at the quoted
    UL against BKG, the distances drawn at that UL, and the spread of
    injected fluxes it corresponds to (vs. the 2D flux UL, for scale).
    """
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))

    _b = np.linspace(min(lambda_bkg.min(), lambda_clo.min()),
                     max(lambda_bkg.max(), lambda_clo.max()), 60)
    ax[0].hist(lambda_bkg, _b, density=True, histtype="stepfilled", color="royalblue",
               alpha=0.55, label=f"BKG (N={len(lambda_bkg)})")
    ax[0].hist(lambda_clo, _b, density=True, histtype="step", color="crimson", lw=1.7,
               label=f"injected at ${x_label}^{{UL}}$")
    ax[0].axvline(target_lambda, color="k", ls="--", lw=1.4,
                  label=f"target $\\Lambda$={target_lambda:.2f}")
    ax[0].set_xlabel("$\\Lambda$"); ax[0].set_ylabel("Normalized counts")
    ax[0].set_title(f"frac = {frac_clo:.3f} $\\pm$ {err_clo:.3f}  (CL = {cl})")
    ax[0].legend(frameon=False, fontsize=8)

    ax[1].hist(d_clo, 50, color="darkorange", alpha=0.7)
    ax[1].axvline(np.median(d_clo), color="k", ls="--",
                  label=f"median {np.median(d_clo):.0f} Mpc")
    ax[1].set_xlabel("sampled $d_L$ [Mpc]"); ax[1].set_ylabel("Counts")
    ax[1].set_title("Distances drawn at the limit"); ax[1].legend(frameon=False, fontsize=8)

    log_flux = np.log10(photon_flux_clo)
    ax[2].hist(log_flux, 50, color="seagreen", alpha=0.7)
    ax[2].axvline(np.log10(flux_ul_value), color="crimson", lw=1.8, label="2D flux UL")
    ax[2].axvline(np.median(log_flux), color="k", ls="--", label="median injected")
    ax[2].set_xlabel("$\\log_{10}$ photon flux [cm$^{-2}$s$^{-1}$]"); ax[2].set_ylabel("Counts")
    ax[2].set_title(f"Fluxes injected at ${x_label}^{{UL}}$"); ax[2].legend(frameon=False, fontsize=8)

    fig.tight_layout(); plt.show()


def plot_uls_summary(r_grid, pdf_marg, d_q, lum2, flx3, flx2, lum3, lum3_1sigma,
                     flux_uls_95, result_2d, result_3d, amp_ul, lum_ul,
                     energy_edges, source_name, type_obs, confidence_level,
                     spectral_index):
    """
    The final "both methods, both currencies" comparison: the distance prior
    both conversions share, the flux and luminosity limits from each method
    side by side, and their crossing curves rescaled onto a common x/UL axis.
    """
    fig = plt.figure(figsize=(13.5, 7.2))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(r_grid, pdf_marg, color="darkorange", lw=1.8)
    ax.fill_between(r_grid, 0, pdf_marg, color="darkorange", alpha=0.25)
    for _d, _s in zip(d_q, [":", "--", ":"]):
        ax.axvline(_d, color="k", ls=_s, lw=1.2)
    ax.set_xlim(0, d_q[2] * 2)
    ax.set_xlabel("$d_L$ [Mpc]"); ax.set_ylabel("$p(d)$")
    ax.set_title(f"GW distance posterior (FoV)\nd = {d_q[1]:.0f} [{d_q[0]:.0f}, {d_q[2]:.0f}] Mpc",
                fontsize=10)

    ax = fig.add_subplot(gs[0, 1:])
    ax.errorbar([flx2], [1], xerr=[[flx2 * 0.35]], xuplims=True, fmt="o",
                color="crimson", ms=7, lw=2, label="2D: measured flux UL")
    ax.plot([min(flx3), max(flx3)], [0.55, 0.55], color="seagreen", lw=6, alpha=0.35)
    ax.errorbar([flx3[1]], [0.55], xerr=[[flx3[1] * 0.35]], xuplims=True, fmt="s",
                color="seagreen", ms=7, lw=2,
                label="3D: flux equivalent of $L_0^{UL}$ (5-95% in $d$)")
    # per-pixel sky-map ULs on a twin axis: a different statistic (no trials
    # correction), shown only for scale.
    _axh = ax.twinx()
    _axh.hist(flux_uls_95[~np.isnan(flux_uls_95)], 40, color="0.8", zorder=-5)
    _axh.set_yticks([]); _axh.set_zorder(-1)
    ax.set_zorder(1); ax.patch.set_visible(False)
    ax.plot([], [], color="0.8", lw=6, label="per-pixel sky-map ULs (95% region)")
    ax.set_xscale("log"); ax.set_ylim(-0.05, 1.35); ax.set_yticks([])
    ax.set_xlabel("Photon flux UL [cm$^{-2}$ s$^{-1}$]  "
                 f"({energy_edges[0]:.2f}-{energy_edges[1]:.2f})")
    ax.set_title("FLUX upper limits", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[1, 1:])
    ax.plot([min(lum2), max(lum2)], [1.0, 1.0], color="crimson", lw=6, alpha=0.35)
    ax.errorbar([lum2[1]], [1.0], xerr=[[lum2[1] * 0.35]], xuplims=True, fmt="o",
                color="crimson", ms=7, lw=2,
                label="2D: luminosity equivalent of $\\phi_0^{UL}$ (5-95% in $d$)")
    ax.errorbar([lum3], [0.55], xerr=[[lum3 * 0.35]], xuplims=True, fmt="s",
                color="seagreen", ms=7, lw=2, label="3D: measured $L_0$ UL (marginalised)")
    if lum3_1sigma is not None and np.isfinite(lum3_1sigma[0]):
        ax.plot(lum3_1sigma, [0.55, 0.55], color="seagreen", lw=6, alpha=0.5)
    ax.set_xscale("log"); ax.set_ylim(0.2, 1.45); ax.set_yticks([])
    ax.set_xlabel("Isotropic-equivalent luminosity UL [erg s$^{-1}$]  "
                 f"({energy_edges[0]:.2f}-{energy_edges[1]:.2f})")
    ax.set_title("LUMINOSITY upper limits", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    ax = fig.add_subplot(gs[1, 0])
    _a  = np.asarray(result_2d["hist_amp"]);  _fa = np.asarray(result_2d["hist_frac"])
    _l  = np.asarray(result_3d["hist_lum"]);  _fl = np.asarray(result_3d["hist_frac"])
    ax.plot(np.sort(_a) / amp_ul, _fa[np.argsort(_a)], "o-",
            color="crimson", ms=4, lw=1.2, label="2D ($\\phi_0/\\phi_0^{UL}$)")
    ax.plot(np.sort(_l) / lum_ul, _fl[np.argsort(_l)], "s-",
            color="seagreen", ms=4, lw=1.2, label="3D ($L_0/L_0^{UL}$)")
    ax.axhline(confidence_level, color="k", ls="--", lw=1.2)
    ax.axvline(1.0, color="k", ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("injected / UL"); ax.set_ylabel("fraction > target")
    ax.set_title("Crossing curves, rescaled", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"{source_name} ({type_obs}) - {confidence_level:.0%} CL upper limits, "
                f"PWL index {spectral_index}", fontsize=12)
    plt.show()


# ---------------------------------------------------------------------------
# Grid-scan (USE_ITERATIVE_ULS = False) diagnostics
# ---------------------------------------------------------------------------

def plot_grid_lambda_heatmap(lambda_bins, amplitude_edges, lambda_hist_f,
                             lambda_hist_bkg, energy_edges):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 4), gridspec_kw={"height_ratios": [6, 1]}, sharex=True)
    ax1.pcolormesh(lambda_bins, amplitude_edges, lambda_hist_f, norm=LogNorm(), cmap="plasma")
    ax2.pcolormesh(lambda_bins, [0, 1], [lambda_hist_bkg], norm=LogNorm(), cmap="viridis")
    for ax in (ax1, ax2):
        ax.set_facecolor("k")
    ax1.set_ylim(1e-13, 7e-11); ax1.set_yscale("log"); ax2.set_yticks([])
    ax2.set_ylabel("BKG"); ax2.set_xlabel("$\\Lambda$")
    ax1.set_ylabel("$\\phi_0$ [cm${}^{-2}$s${}^{-1}$ TeV${}^{-1}$]")
    ax1.set_title(f"{energy_edges[0]:.2f}-{energy_edges[1]:.2f}")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def plot_grid_crossing_curve(amplitudes, lambda_f_frac_bkg, lambda_f_frac_real,
                             amplitude95_bkg, amplitude95_real, confidence_level,
                             energy_edges):
    fig, ax = plt.subplots(1, 1, figsize=(5.4, 3))
    ax.plot(amplitudes, lambda_f_frac_bkg,  color="k", label="$\\Lambda>\\Lambda_{bkg}$")
    ax.plot(amplitudes, lambda_f_frac_real, color="r", label="$\\Lambda>\\Lambda_{real}$")
    ax.axvline(amplitude95_bkg,  color="k", ls="--")
    ax.axvline(amplitude95_real, color="r", ls="--")
    ax.axhline(confidence_level, ls=":", color="lightgray", zorder=-2,
              label=f"C.L. = {confidence_level}")
    ax.legend(frameon=False)
    ax.set_xlabel("$\\phi_0$ [cm${}^{-2}$s${}^{-1}$ TeV${}^{-1}$]")
    ax.set_ylabel("Fraction of events")
    ax.set_title(f"{energy_edges[0]:.2f}-{energy_edges[1]:.2f}")
    ax.set_xlim(1e-13, 5e-11); ax.set_xscale("log")
    plt.show()


def plot_grid_threshold_case(lambda_bkg, lambda_bins, lambda_f_at_ul, lambda_bins_ext,
                             lambda_real, lambda_bkg_m, p_value, significance,
                             flux_sim, energy_edges):
    fig, axt = plt.subplots(1, 1, figsize=(7, 3))
    axt.hist(lambda_bkg, lambda_bins, color="royalblue",
             density=True, histtype="stepfilled", label=f"BKG sim\nN={len(lambda_bkg)}")
    axt.hist(lambda_f_at_ul, lambda_bins_ext, color="r", density=True, histtype="step",
             label="Sim of final UL")
    axt.text(0.85, 0.87, f"p-value={p_value:.2f}\nsignificance={significance:.2f} $\\sigma$",
             ha="center", va="center", transform=axt.transAxes)
    axt.axvline(lambda_real, color="k", ls="--", label=f"Real data\n$\\Lambda=${lambda_real:.2f}")
    axt.axvline(lambda_bkg_m, color="darkblue", ls=":", label=f"BKG median\n$\\Lambda=${lambda_bkg_m:.2f}")
    axt.legend(frameon=False, loc=4)
    axt.set_xlabel("$\\Lambda$"); axt.set_ylabel("Normalized counts")
    axt.set_title(f"Sim flux {flux_sim:.2e},   {energy_edges[0]:.2f}-{energy_edges[1]:.2f}")
    fig.tight_layout()
    plt.show()


def plot_uls_vs_real_distribution(flux_uls, flux_uls_95, flux_real, energy_edges,
                                  significance=None, flux_bkg=None):
    """
    Real per-pixel flux-UL distribution with the quoted UL marked. With
    `flux_bkg` given (grid-scan mode), both the sensitivity and the data UL
    are shown, labelled by whether the observation over- or under-fluctuates.
    """
    fig, ax = plt.subplots(figsize=(3, 2))
    mask = ~np.isnan(flux_uls_95)
    single_pixel = np.sum(mask) == 1

    h = ax.hist(flux_uls, 40, histtype="stepfilled", color="lightgray", label="Total distribution")
    if single_pixel:
        ax.axvline(flux_uls_95[mask][0], color="k", lw=1, label="ULs in 95%")
    else:
        ax.hist(flux_uls_95, h[1], histtype="step", color="k", label="ULs in 95%")

    ymid = np.diff(ax.get_ylim())[0] / 2

    if flux_bkg is not None:
        if significance >= 0.0:
            col_sens, col_ul, lab_sens, lab_ul, ls_sens, ls_ul = "0.5", "green", "Sensitivity", "Data UL", "--", "-"
        else:
            col_sens, col_ul, lab_sens, lab_ul, ls_sens, ls_ul = "green", "0.5", "Data UL", "Overestimated UL", "-", "--"
        for val, ls, col, lab in [(flux_bkg, ls_sens, col_sens, lab_sens),
                                  (flux_real, ls_ul,  col_ul,  lab_ul)]:
            ax.axvline(val, ls=ls, color=col)
            ax.errorbar(val, ymid, yerr=0, xerr=val * 0.2, xuplims=True, ls="", color=col, label=lab)
    else:
        ax.axvline(flux_real, ls="-", color="green")
        ax.errorbar(flux_real, ymid, yerr=0, xerr=flux_real * 0.2, xuplims=True, ls="",
                    color="green", label="Flux UL")

    ax.set_xlabel("Flux UL [1 / (s cm2)]"); ax.set_ylabel("Normalized counts")
    ax.set_title(f"{energy_edges[0]:.2f}-{energy_edges[1]:.2f}")
    ax.legend(frameon=False, loc=(1.03, 0))
    plt.show()


def plot_grid_ul_distributions(ulmax_bkg, flux_uls_95, ul_dist, mask_threshold_95,
                               flux_ul_value, energy_edges):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(ulmax_bkg, 25, density=True, histtype="stepfilled", color="royalblue",
            label="Maximum UL in 95%", zorder=-5)
    ax.hist(flux_uls_95, 40, density=True, histtype="step", color="k", label="Real Data 95%", zorder=10)
    for data in ul_dist:
        data_95 = np.where(mask_threshold_95, data, np.nan).ravel()
        ax.hist(data_95, 40, density=True, histtype="step", color="0.5", alpha=0.2, zorder=0)
    ax.axvline(flux_ul_value, ls="-", color="crimson")
    ax.errorbar(flux_ul_value, np.diff(ax.get_ylim())[0] / 2, yerr=0,
                xerr=flux_ul_value * 0.4, xuplims=True, ls="", color="crimson", label="Global UL")
    ax.plot([], [], color="gray", ls="-", marker="", label="Simulation\ndistributions 95%")
    ax.set_title(f"{energy_edges[0]:.2f}-{energy_edges[1]:.2f}")
    ax.legend(loc=(1.03, 0), frameon=False)
    ax.set_xlabel("Flux UL [1 / (s cm2)]"); ax.set_ylabel("Normalized counts")
    plt.show()


def plot_grid_ts_distributions(lambda_bkg, ts2_masked, ts2_dist, mask_threshold_95, energy_edges):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(lambda_bkg, 30, density=True, histtype="stepfilled", color="r",
            label="$\\Lambda$ (max TS')", zorder=-5)
    ax.hist(ts2_masked, 30, density=True, histtype="step", color="k", label="Real Data", zorder=10)
    for data in ts2_dist:
        data = np.where(mask_threshold_95, data, np.nan).ravel()
        ax.hist(data, 30, density=True, histtype="step", color="0.5", alpha=0.2, zorder=0)
    ax.set_title(f"{energy_edges[0]:.2f}-{energy_edges[1]:.2f}")
    ax.plot([], [], color="0.8", label="Simulated data")
    ax.legend(loc=(1.03, 0), frameon=False)
    ax.set_xlabel("TS'"); ax.set_ylabel("Normalized counts")
    plt.show()




# ---------------------------------------------------------------------------
# GW viewing angle: p(d, theta_v) and the Step 4.3 options
# (setup_angle_distribution.ipynb; gwuls.angle_distribution)
# ---------------------------------------------------------------------------

# One colour per entity, the same in every angle figure (validated categorical
# order: slots 1-4 for the distribution kinds, 5-8 for waveform analyses).
ANGLE_KIND_COLORS = {
    "pe": "#2a78d6", "schutz": "#eb6834", "isotropic": "#1baf7a", "selection": "#eda100",
    "two_bin": "#52514e", "fixed": "#52514e",
}
ANGLE_KIND_STYLES = {"two_bin": "--", "fixed": ":"}
ANALYSIS_COLORS = ("#e87ba4", "#008300", "#4a3aa7", "#e34948")
_BLUE_RAMP = ("#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
_TEXT, _TEXT_2, _GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _gray_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("gwuls_gray", ("#ffffff", "#f0efec", "#c3c2b7", "#8a8984"))


def _blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("gwuls_blue", _BLUE_RAMP)


def blue_steps(n):
    """n ordered colours from the sequential blue ramp (light -> dark), for
    ordinal families of curves (e.g. p(theta|d) at increasing d)."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("gwuls_blue_ord", _BLUE_RAMP[2:])
    return [cmap(x) for x in np.linspace(0.0, 1.0, n)]


def _style_axis(ax):
    ax.grid(True, color=_GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_TEXT_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=_TEXT_2, labelcolor=_TEXT)


def _finish(fig, save_path):
    if save_path is not None:
        os.makedirs(os.path.dirname(str(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_distance_angle_corner(joints, colors=None, samples=None, levels=(0.5, 0.9),
                               distance_lim=None, show_isotropic_prior=True, title=None,
                               summary=True, save_path=None):
    """Corner plot of p(d_L, theta_v) from one or more JointDistanceAngle.

    joints : dict label -> JointDistanceAngle. The first one is drawn as a
        filled density with its HPD contours; the others as contour lines, on
        top, for comparison (e.g. per waveform analysis, or bandwidths).
    samples : optional (d, theta) arrays drawn as faint dots under the first
        joint and as step histograms in the marginals -- checks the KDE grid
        against the samples it came from.
    """
    labels = list(joints)
    colors = colors or {lab: c for lab, c in zip(labels, ("#2a78d6",) + ANALYSIS_COLORS)}
    first = joints[labels[0]]

    fig, axes = plt.subplots(2, 2, figsize=(8, 7.5),
                             gridspec_kw={"width_ratios": [1, 1], "height_ratios": [1, 1],
                                          "hspace": 0.06, "wspace": 0.06})
    ax_d, ax_leg, ax_j, ax_t = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
    ax_leg.axis("off")

    if distance_lim is None:
        lo, hi = [], []
        for J in joints.values():
            d, pd = J.marginal_distance()
            c = np.cumsum(pd) / pd.sum()
            lo.append(np.interp(0.001, c, d)); hi.append(np.interp(0.999, c, d))
        distance_lim = (min(lo), max(hi))

    # joint panel: x = d, y = theta_v, so vertical slices are p(theta_v | d)
    # neutral fill, so every entity's coloured contour stays readable on top of it
    ax_j.pcolormesh(first.distance_Mpc, first.theta_deg, first.density.T, cmap=_gray_cmap(),
                    shading="auto", rasterized=True)
    if samples is not None:
        ax_j.plot(samples[0], samples[1], ",", color=_TEXT_2, alpha=0.12, rasterized=True)
    for lab in labels:
        J = joints[lab]
        lv = J.hpd_levels(levels)  # increasing density = decreasing fraction
        ax_j.contour(J.distance_Mpc, J.theta_deg, J.density.T, levels=lv,
                     colors=[colors[lab]], linewidths=[1.8, 1.0][:len(lv)] if lab == labels[0]
                     else [1.3, 0.8][:len(lv)])
    ax_j.set_xlim(distance_lim); ax_j.set_ylim(0, 90)
    ax_j.set_yticks(np.arange(0, 91, 15))
    ax_j.set_xlabel(r"$d_L$ [Mpc]"); ax_j.set_ylabel(r"$\theta_{\rm v}$ [deg]")

    # marginals
    for lab in labels:
        J = joints[lab]
        d, pd = J.marginal_distance()
        t, pt = J.marginal_theta()
        lw = 2.0 if lab == labels[0] else 1.3
        ax_d.plot(d, pd, color=colors[lab], lw=lw, label=lab)
        ax_t.plot(t, pt, color=colors[lab], lw=lw)
    if samples is not None:
        ax_d.hist(samples[0], bins=np.linspace(*distance_lim, 60), density=True, histtype="step",
                  color=_TEXT_2, lw=0.8, label="samples (histogram)")
        ax_t.hist(samples[1], bins=np.linspace(0, 90, 46), density=True, histtype="step",
                  color=_TEXT_2, lw=0.8)
    if show_isotropic_prior:
        t = np.linspace(0, 90, 181)
        ax_t.plot(t, np.sin(np.radians(t)) * np.pi / 180, color=_TEXT_2, lw=1.0, ls=":",
                  label="isotropic prior")
        ax_d.plot([], [], color=_TEXT_2, lw=1.0, ls=":", label=r"isotropic prior ($\theta_{\rm v}$)")
    ax_d.set_xlim(distance_lim); ax_d.set_xticklabels([]); ax_d.set_yticks([])
    ax_d.set_ylabel(r"$p(d_L)$")
    ax_t.set_xlim(0, 90); ax_t.set_xticks(np.arange(0, 91, 15)); ax_t.set_yticks([])
    ax_t.set_xlabel(r"$\theta_{\rm v}$ [deg]"); ax_t.set_ylabel(r"$p(\theta_{\rm v})$")
    for ax in (ax_d, ax_j, ax_t):
        _style_axis(ax)

    handles, hl = ax_d.get_legend_handles_labels()
    ax_leg.legend(handles, hl, loc="upper left", frameon=False, fontsize=9,
                  title=f"contours: {', '.join(f'{int(100 * f)}%' for f in levels)} HPD",
                  title_fontsize=9)
    if summary:
        t, pt = first.marginal_theta()
        d, pd = first.marginal_distance()

        def q(x, p):
            c = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(x))])
            return np.interp([0.05, 0.5, 0.95], c / c[-1], x)

        tq, dq = q(t, pt), q(d, pd)
        rho = first.meta.get("kernel_correlation")
        txt = (f"{labels[0]}\n"
               rf"$\theta_{{\rm v}}$ = {tq[1]:.1f} (+{tq[2]-tq[1]:.1f} / -{tq[1]-tq[0]:.1f}) deg" "\n"
               rf"$d_L$ = {dq[1]:.0f} (+{dq[2]-dq[1]:.0f} / -{dq[1]-dq[0]:.0f}) Mpc" "\n"
               "median, 90% interval")
        if rho is not None:
            txt += "\n" + rf"corr($d_L$, cos$\theta_{{\rm v}}$) = {rho:.2f}"
        ax_leg.text(0.0, 0.0, txt, transform=ax_leg.transAxes, va="bottom", ha="left",
                    fontsize=9, color=_TEXT)
    if title:
        fig.suptitle(title, y=0.93)
    return _finish(fig, save_path)


def plot_conditional_ridge(joint, quantiles=(0.05, 0.5, 0.95), samples=None,
                           distance_lim=None, marks=None, title=None, save_path=None):
    """p(d, theta_v) as a density image with the quantiles of p(theta_v | d)
    along d on top -- the d-theta degeneracy in one picture. `marks`: optional
    distances to draw as vertical lines (e.g. the ones sliced elsewhere)."""
    d, pd = joint.marginal_distance()
    c = np.cumsum(pd) / pd.sum()
    if distance_lim is None:
        distance_lim = (np.interp(0.001, c, d), np.interp(0.999, c, d))
    fig, (ax, ax_m) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    ax.pcolormesh(d, joint.theta_deg, joint.density.T, cmap=_blue_cmap(), shading="auto",
                  rasterized=True)
    if samples is not None:
        ax.plot(samples[0], samples[1], ",", color=_TEXT_2, alpha=0.12, rasterized=True)
    dgrid = np.linspace(*distance_lim, 120)
    qs = joint.conditional_theta_quantiles(quantiles, distance_Mpc=dgrid)
    mid = len(quantiles) // 2
    ax.plot(dgrid, qs[:, mid], color=_TEXT, lw=2.0,
            label=rf"median of $p(\theta_{{\rm v}}\,|\,d_L)$")
    ax.fill_between(dgrid, qs[:, 0], qs[:, -1], color=_TEXT, alpha=0.08, lw=0)
    for k in (0, -1):
        ax.plot(dgrid, qs[:, k], color=_TEXT, lw=1.0)
    ax.plot([], [], color=_TEXT, lw=1.0,
            label=f"{100 * quantiles[0]:.0f}-{100 * quantiles[-1]:.0f}% of "
                  rf"$p(\theta_{{\rm v}}\,|\,d_L)$")
    for m in ([] if marks is None else marks):
        ax.axvline(m, color=_TEXT_2, lw=0.8, alpha=0.6)
        ax_m.axvline(m, color=_TEXT_2, lw=0.8, alpha=0.6)
        ax_m.annotate(f"{m:.0f}", (m, 1.0), xycoords=("data", "axes fraction"), xytext=(2, -2),
                      textcoords="offset points", va="top", fontsize=8, color=_TEXT_2)
    ax.set_ylim(0, 90); ax.set_yticks(np.arange(0, 91, 15)); ax.set_xlim(distance_lim)
    ax.set_ylabel(r"$\theta_{\rm v}$ [deg]")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax_m.plot(d, pd, color=ANGLE_KIND_COLORS["pe"], lw=2.0)
    ax_m.set_yticks([]); ax_m.set_ylabel(r"$p(d_L)$"); ax_m.set_xlabel(r"$d_L$ [Mpc]")
    for a in (ax, ax_m):
        _style_axis(a)
    if title:
        ax.set_title(title)
    return _finish(fig, save_path)


def _model_angle_lines(ax, model_angles, label=True):
    for k, a in enumerate([] if model_angles is None else model_angles):
        ax.axvline(a, color=_GRID, lw=2.5, zorder=0,
                   label="model file angles" if (label and k == 0) else None)


def plot_theta_distribution_comparison(dists, distance_Mpc=None, model_angles=None,
                                       thresholds=(10, 20, 30, 45), title=None, save_path=None):
    """The Step 4.3 options side by side, all normalised per degree so they
    share one axis. dists : dict label -> ThetaDistribution (colour/style by
    kind). Panels: pdf, CDF, pdf per unit cos(theta_v) (where isotropic is
    flat), and P(theta_v < threshold) for a few jet-relevant thresholds."""
    from matplotlib.lines import Line2D

    fig, axs = plt.subplots(2, 2, figsize=(11, 7.5), gridspec_kw={"hspace": 0.32, "wspace": 0.22})
    ax_p, ax_c, ax_cos, ax_b = axs.ravel()
    grid = np.linspace(0, 90, 361)
    handles, cos_max = [], 0.0
    for lab, D in dists.items():
        color = ANGLE_KIND_COLORS.get(D.kind, _TEXT_2)
        ls = ANGLE_KIND_STYLES.get(D.kind, "-")
        dd = distance_Mpc if D.depends_on_distance else None
        if D.kind == "fixed":
            ax_p.axvline(D.theta_fixed_deg, color=color, ls=ls, lw=1.8)
        else:
            _, p = D.pdf_grid(distance_Mpc=dd, theta_grid_deg=grid)
            ax_p.plot(grid, p, color=color, ls=ls, lw=2.0)
        if D.kind not in ANGLE_KIND_STYLES:  # dummies have no meaning per solid angle
            # p(cos theta) = p(theta) / |d cos / d theta| = p(theta) / (sin(theta) pi/180)
            s = np.sin(np.radians(grid)) * np.pi / 180
            ok = grid > 0.5
            pc = p[ok] / s[ok]
            cos_max = max(cos_max, float(np.max(pc)))
            ax_cos.plot(np.cos(np.radians(grid[ok])), pc, color=color, ls=ls, lw=2.0)
        _, cdf = D.cdf_grid(distance_Mpc=dd, theta_grid_deg=grid)
        ax_c.plot(grid, cdf, color=color, ls=ls, lw=2.0)
        handles.append(Line2D([], [], color=color, ls=ls, lw=2.0, label=lab))

    _model_angle_lines(ax_p, model_angles)
    _model_angle_lines(ax_c, model_angles, label=False)
    ax_p.set_xlim(0, 90); ax_p.set_xticks(np.arange(0, 91, 15)); ax_p.set_ylim(bottom=0)
    ax_p.set_xlabel(r"$\theta_{\rm v}$ [deg]"); ax_p.set_ylabel(r"$p(\theta_{\rm v})$ [deg$^{-1}$]")
    ax_p.set_title("pdf, per degree", fontsize=10)
    ax_c.set_xlim(0, 90); ax_c.set_xticks(np.arange(0, 91, 15)); ax_c.set_ylim(0, 1)
    ax_c.set_xlabel(r"$\theta_{\rm v}$ [deg]"); ax_c.set_ylabel(r"$P(<\theta_{\rm v})$")
    ax_c.set_title("CDF", fontsize=10)
    ax_cos.set_xlim(0, 1); ax_cos.set_ylim(0, 1.1 * cos_max if cos_max > 0 else None)
    ax_cos.set_xlabel(r"$\cos\theta_{\rm v}$  (1 = face-on)")
    ax_cos.set_ylabel(r"$p(\cos\theta_{\rm v})$")
    ax_cos.set_title("pdf per unit solid angle: isotropic is flat", fontsize=10)

    labs = [lab for lab in dists if dists[lab].kind != "fixed"]  # a fixed angle is a step: 0 or 1
    width = 0.8 / len(labs)
    x = np.arange(len(thresholds))
    for k, lab in enumerate(labs):
        D = dists[lab]
        dd = distance_Mpc if D.depends_on_distance else None
        vals = [D.prob_below(t, distance_Mpc=dd) for t in thresholds]
        ax_b.bar(x + (k - (len(labs) - 1) / 2) * width, vals, width * 0.9,
                 color=ANGLE_KIND_COLORS.get(D.kind, _TEXT_2),
                 hatch="//" if D.kind in ANGLE_KIND_STYLES else None, edgecolor="white", lw=0)
    ax_b.set_xticks(x, [rf"$\theta_{{\rm v}}<{t}^\circ$" for t in thresholds])
    ax_b.set_ylim(0, 1); ax_b.set_ylabel("probability")
    ax_b.set_title("chance of a near-on-axis view", fontsize=10)
    for a in axs.ravel():
        _style_axis(a)
    ax_b.grid(False, axis="x")
    if model_angles is not None and len(model_angles):
        handles.append(Line2D([], [], color=_GRID, lw=2.5, label="model file angles"))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False,
               bbox_to_anchor=(0.5, 1.0 if not title else 0.97))
    if title:
        fig.suptitle(title, y=1.02)
    return _finish(fig, save_path)


def plot_theta_sampler_check(dists, n=200_000, distance_Mpc=None, seed=0, save_path=None):
    """Small multiples: a histogram of `n` draws from each ThetaDistribution
    against its own tabulated pdf, with the largest CDF difference (KS
    distance) in each title -- checks that `.sample` draws what `.pdf_grid`
    says."""
    rng = np.random.default_rng(seed)
    n_panels = len(dists)
    ncol = min(3, n_panels)
    nrow = int(np.ceil(n_panels / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.8 * nrow), squeeze=False,
                            gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    bins = np.linspace(0, 90, 91)
    for ax, (lab, D) in zip(axs.ravel(), dists.items()):
        dd = distance_Mpc if D.depends_on_distance else None
        draws = D.sample(n, distance_Mpc=dd, rng=rng)
        color = ANGLE_KIND_COLORS.get(D.kind, _TEXT_2)
        ax.hist(draws, bins=bins, density=True, color=color, alpha=0.35, lw=0)
        grid, pdf = D.pdf_grid(distance_Mpc=dd, theta_grid_deg=np.linspace(0, 90, 721))
        g, cdf = D.cdf_grid(distance_Mpc=dd, theta_grid_deg=np.linspace(0, 90, 721))
        emp = np.searchsorted(np.sort(draws), g, side="right") / draws.size
        ks = np.max(np.abs(emp - cdf))
        if D.kind == "fixed":
            ax.axvline(D.theta_fixed_deg, color=_TEXT, lw=1.2)
        else:
            ax.plot(grid, pdf, color=_TEXT, lw=1.2)
        ax.set_title(f"{lab}\nKS distance = {ks:.4f}  (1/sqrt(n) = {1 / np.sqrt(n):.4f})",
                     fontsize=9)
        ax.set_xlim(0, 90); ax.set_xticks(np.arange(0, 91, 15)); ax.set_yticks([])
        ax.set_xlabel(r"$\theta_{\rm v}$ [deg]")
        _style_axis(ax)
    for ax in axs.ravel()[n_panels:]:
        ax.axis("off")
    return _finish(fig, save_path)


def model_angle_weights(dist, model_angles, method="nearest", distance_Mpc=None):
    """Probability that sim_3d ends up using each model file, for a
    ThetaDistribution and GRBModelSet.get_model's `method`:

    - "nearest"    (Option A): the mass of p(theta_v) closest to each angle;
    - "stochastic" (Option B): each theta_v is split between its two
      bracketing angles linearly, so the weights are p(theta_v) integrated
      against hat functions. Outside the tabulated range, everything goes to
      the edge file (as get_model clips).
    """
    angles = np.sort(np.asarray(model_angles, dtype=float))
    dd = distance_Mpc if dist.depends_on_distance else None
    if dist.kind == "fixed":
        theta = np.array([dist.theta_fixed_deg]); w = np.array([1.0])
    else:
        theta = np.linspace(0, 90, 3601)
        _, p = dist.pdf_grid(distance_Mpc=dd, theta_grid_deg=theta)
        w = p * np.gradient(theta)
        w = w / w.sum()
    if method == "nearest":
        k = np.argmin(np.abs(theta[:, None] - angles[None, :]), axis=1)
        return np.bincount(k, weights=w, minlength=angles.size)
    if method == "stochastic":
        out = np.zeros(angles.size)
        for k in range(angles.size):
            e = np.zeros(angles.size); e[k] = 1.0
            out[k] = np.sum(w * np.interp(theta, angles, e))  # np.interp clamps at the edges
        return out
    raise ValueError("method must be 'nearest' or 'stochastic'")


def plot_model_angle_assignment(dists, model_angles, method="nearest", distance_Mpc=None,
                                save_path=None):
    """Grouped bars: for each model file (angle), the fraction of iterations
    that would use it under each Step 4.3 option. This is where the choice of
    angle distribution actually reaches the upper limit."""
    angles = np.sort(np.asarray(model_angles, dtype=float))
    labs = list(dists)
    width = 0.8 / len(labs)
    x = np.arange(angles.size)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    for k, lab in enumerate(labs):
        D = dists[lab]
        w = model_angle_weights(D, angles, method=method, distance_Mpc=distance_Mpc)
        ax.bar(x + (k - (len(labs) - 1) / 2) * width, w, width * 0.9,
               color=ANGLE_KIND_COLORS.get(D.kind, _TEXT_2),
               hatch="//" if D.kind in ANGLE_KIND_STYLES else None, edgecolor="white", lw=0,
               label=lab)
    ax.set_xticks(x, [rf"{a:.1f}$^\circ$" for a in angles])
    ax.set_xlabel("model file used (its viewing angle)")
    ax.set_ylabel("fraction of iterations")
    ax.set_ylim(0, 1)
    ax.set_title(f"Which model file each option sends the draws to ({method})", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _style_axis(ax)
    ax.grid(False, axis="x")
    return _finish(fig, save_path)
