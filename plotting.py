import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.gridspec import GridSpec

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
    ax1.axvline(lambda_bkg_m, color="darkblue", ls=":",  label=f"BKG med={lambda_bkg_m:.2f}")
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
    geom, bin_edges_ra, bin_edges_dec, size_fov, data_ligo_2d, threshold_maps
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

def summary_bkg_simulations(geom, bin_edges_ra, bin_edges_dec, data_ligo_2d, threshold_maps,
                             lambda_bkg, lambda_real, lambda_bkg_m, p_value, significance,
                             map_lambda_bkg, map_ts_bkg, map_source_sim, energy_edges, bins_lambda):

    fig = plt.figure(figsize=(7, 5))
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
             ha="center", va="center", transform=axt.transAxes)
    axt.set(xlabel=r"$\Lambda$", ylabel="Normalized counts",
            title=f"{energy_edges[0]:.2f}–{energy_edges[1]:.2f}")
    axt.legend(frameon=False, loc=4)

    # --- Bottom: Sky Maps ---
    panels = [
        (ax1, "magma",   map_lambda_bkg,  r"$\Lambda_{BKG}$ ($TS'_{max}$)"),
        (ax2, "magma",   map_ts_bkg,      r"$TS_{max}$"),
        (ax3, "cividis", map_source_sim,  r"Source positions $(F=0)$"),
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