"""Regenerate the figures of docs/setup_model.md (and of the setup_model slides)
from the catO5 files in data/models/grb_afterglow_inaf/raw/.

    python docs/figures/make_setup_model_figures.py

Writes PNGs to docs/figures/setup_model/. Sized for a 1920x1080 slide
(1 inch = 100 px there), so the text stays legible when a figure fills a slide.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gwuls import grb_model, paths  # noqa: E402

OUT = Path(__file__).resolve().parent / "setup_model"
E_ISO_REF = 1e50   # erg
E_FIXED = 10.0     # GeV, rest frame
THETA_QUERY = 18.0

BG, INK, SOFT, GRID = "#F7F6F1", "#16213A", "#465267", "#E4E2DA"
BLUE, ORANGE, AQUA, GRAY = "#2A78D6", "#E0662F", "#1BAF7A", "#9AA3AF"
THETA_CMAP = LinearSegmentedColormap.from_list("theta", ["#86b6ef", "#3987e5", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "font.size": 19, "axes.labelsize": 21, "axes.titlesize": 21,
    "xtick.labelsize": 18, "ytick.labelsize": 18, "legend.fontsize": 17,
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": SOFT,
    "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "lines.linewidth": 2.6,
})


def _save(fig, name):
    fig.savefig(OUT / name, dpi=120, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def _theta_color(theta, angles):
    return THETA_CMAP(Normalize(angles[0], angles[-1])(theta))


def fig_catalogue_shapes(ms):
    """What a catO5 model looks like: one power-law spectrum, one peaked light curve."""
    A = ms.angles_deg
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.6, 6.4))
    for th in A:
        m, c = ms.models[th], _theta_color(th, A)
        t_pk, _ = m.peak()
        i_pk = int(np.argmin(np.abs(m.time - t_pk)))
        sed = m.energy ** 2 * m.L[:, i_pk]
        sed /= np.interp(np.log(E_FIXED), np.log(m.energy), sed)
        gamma = -np.polyfit(np.log(m.energy[:20]), np.log(m.L[:20, i_pk]), 1)[0]
        ax1.loglog(m.energy, sed, color=c, label=f"{th:.1f}°  (Γ = {gamma:.2f})")
        L10 = m.at(E_FIXED, m.time)[0]
        ax2.loglog(m.time / t_pk, L10 / L10.max(), color=c, label=f"{th:.1f}°")
    ax1.set_xlabel("E' [GeV]")
    ax1.set_ylabel("E'² L(E') at the peak  (norm. at 10 GeV)")
    ax1.set_title("Spectrum: a single power law, the same at all t'", loc="left")
    ax1.legend(loc="lower left")
    ax2.set_xlabel("t' / t'_peak")
    ax2.set_ylabel(f"L / L_peak  at E' = {E_FIXED:g} GeV")
    ax2.set_title("Light curve: rise ∝ t'², peak, decay", loc="left")
    ax2.set_ylim(1e-6, 3)
    ax2.legend(loc="lower center", ncol=2)
    fig.tight_layout(w_pad=3)
    _save(fig, "catalogue_shapes.png")


def fig_eiso_standardization(ms, std):
    A = ms.angles_deg
    fig, axes = plt.subplots(1, 2, figsize=(16.6, 6.6), sharey=True)
    for ax, s, title in [(axes[0], ms, "Raw files: each at its own E_iso"),
                         (axes[1], std, f"Rescaled to a common E_iso = 10⁵⁰ erg")]:
        for th in A:
            m, c = s.models[th], _theta_color(th, A)
            ax.loglog(m.time, m.light_curve(), color=c, label=f"{th:.1f}°")
            ax.plot(*m.peak(), "o", ms=11, color=c, mec=BG, mew=2)
        ax.set_title(title, loc="left")
        ax.set_xlabel("t' [s]")
    axes[0].set_ylabel("∫ E' L dE'  [erg/s]")
    axes[1].legend(title="θ", loc="upper right", ncol=2)
    fig.tight_layout(w_pad=2)
    _save(fig, "eiso_standardization.png")


def fig_peaks_vs_theta(ms, std):
    A = ms.angles_deg
    raw = np.array([ms.models[th].peak() for th in A])
    res = np.array([std.models[th].peak() for th in A])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.6, 6.0))
    for ax, j, ylabel in [(ax1, 0, "t'_peak [s]"), (ax2, 1, "peak of ∫ E' L dE'  [erg/s]")]:
        ax.semilogy(A, raw[:, j], "o--", color=GRAY, ms=12, mfc=BG, mew=2.5, lw=1.8,
                    label="raw files")
        ax.semilogy(A, res[:, j], "o-", color=BLUE, ms=12, mec=BG, mew=2, lw=1.8,
                    label="common E_iso, (a, b) = (1, 1/3)")
        ax.set_xlabel("θ [deg]")
        ax.set_ylabel(ylabel)
    ax1.set_title("Peak time: ordered in θ only after rescaling", loc="left")
    ax2.set_title("Peak luminosity: ordered in θ after rescaling", loc="left")
    ax1.legend(loc="upper left")
    fig.tight_layout(w_pad=3)
    _save(fig, "peaks_vs_theta.png")


def fig_phase_alignment(std):
    m_al = std.get_model(THETA_QUERY, method="interp")
    m_fx = std.get_model(THETA_QUERY, method="interp", align_peaks=False)
    th_a, th_b = m_al.meta["bracket_deg"]
    a, b = std.models[th_a], std.models[th_b]
    t = np.geomspace(0.2, 5e5, 600)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.6, 6.8))
    for m, c, name in [(a, "#86b6ef", f"{th_a:.1f}° (tabulated)"), (b, "#184f95", f"{th_b:.1f}° (tabulated)")]:
        ax1.loglog(t, m.at(E_FIXED, t)[0], color=c, lw=2.2, label=name)
    ax1.loglog(t, m_fx.at(E_FIXED, t)[0], color=ORANGE, ls="--", lw=3, label=f"fixed t' ({THETA_QUERY:g}°)")
    ax1.loglog(t, m_al.at(E_FIXED, t)[0], color=BLUE, lw=3.2, label=f"peak-aligned ({THETA_QUERY:g}°)")
    ax1.set_xlabel("t' [s]")
    ax1.set_ylabel(f"L(E' = {E_FIXED:g} GeV, t')  [ph/s/GeV]")
    ax1.set_title("In t': the fixed-t' blend loses the peak", loc="left")
    ax1.set_ylim(1e40, 1e47)
    ax1.legend(loc="lower center")

    x = np.linspace(-3, 3.5, 600)
    for m, c, name, lw in [(a, "#86b6ef", f"{th_a:.1f}°", 2.2), (b, "#184f95", f"{th_b:.1f}°", 2.2),
                           (m_al, BLUE, f"peak-aligned {THETA_QUERY:g}°", 3.2)]:
        t_pk = m.peak()[0]
        ax2.semilogy(x, m.at(E_FIXED, t_pk * 10 ** x)[0], color=c, lw=lw, label=name)
    ax2.axvline(0, color=SOFT, lw=1.2, ls=":")
    ax2.set_xlabel("phase  x = log₁₀(t' / t'_peak)")
    ax2.set_ylabel(f"L(E' = {E_FIXED:g} GeV)  [ph/s/GeV]")
    ax2.set_title("In phase x: blend at the same x, one peak", loc="left")
    ax2.set_ylim(1e40, 1e47)
    ax2.legend(loc="lower center")
    fig.tight_layout(w_pad=3)
    _save(fig, "phase_alignment.png")


def fig_theta_sweep(std):
    A = std.angles_deg
    sweep = np.geomspace(A[0], A[-1], 30)
    fine = np.linspace(A[0], A[-1], 300)
    t = np.geomspace(0.1, 1e6, 500)
    fig, axes = plt.subplots(2, 2, figsize=(16.6, 11.5))
    for col, (align, name, color) in enumerate([(True, "peak-aligned", BLUE), (False, "fixed t'", ORANGE)]):
        ax = axes[0, col]
        for th in sweep:
            m = std.get_model(th, method="interp", align_peaks=align)
            ax.loglog(t, m.at(E_FIXED, t)[0], color=_theta_color(th, A), lw=1.6)
        ax.set_title(f"Interpolation, {name}", loc="left")
        ax.set_xlabel("t' [s]")
        ax.set_ylim(1e30, 1e51)
        t_pk = [t[np.argmax(std.get_model(th, method="interp", align_peaks=align).at(E_FIXED, t)[0])]
                for th in fine]
        ax = axes[1, col]
        ax.semilogy(fine, t_pk, color=color, lw=3)
        ax.semilogy(A, [std.models[th].peak()[0] for th in A], "o", ms=12, color="#0d366b",
                    mec=BG, mew=2, label="tabulated angles")
        ax.set_xlabel("θ [deg]")
        ax.set_ylim(3, 1e6)
        ax.legend(loc="upper left")
    axes[0, 0].set_ylabel(f"L(E' = {E_FIXED:g} GeV, t')  [ph/s/GeV]")
    axes[1, 0].set_ylabel(f"t'_peak at {E_FIXED:g} GeV [s]")
    fig.tight_layout(h_pad=2.5, w_pad=3)
    fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(A[0], A[-1]), cmap=THETA_CMAP),
                 ax=axes[0, :], label="θ [deg]", pad=0.015, aspect=18)
    _save(fig, "theta_sweep.png")


def fig_tail_77deg(ms):
    m = ms.models[ms.angles_deg[-1]]
    cut = grb_model._single_peak_part(m)
    t = np.geomspace(3e4, 1e7, 400)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.loglog(t, m.at(E_FIXED, t)[0], color=ORANGE, ls="--", lw=3, label="raw file, extrapolated")
    ax.loglog(t, cut.at(E_FIXED, t)[0], color=BLUE, lw=3, label="single-peaked part, extrapolated")
    sel = m.time > 3e4
    ax.loglog(m.time[sel], m.at(E_FIXED, m.time[sel])[0], "o", ms=11, color=INK, mec=BG, mew=2,
              label="tabulated nodes")
    ax.axvline(m.time[-1], color=SOFT, lw=1.2, ls=":")
    ax.set_xlabel("t' [s]")
    ax.set_ylabel(f"L(E' = {E_FIXED:g} GeV)  [ph/s/GeV]")
    ax.set_title(f"θ = {m.theta_deg:.1f}°: the last bin rises again", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    _save(fig, "tail_77deg.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = grb_model.GRBModelSet.from_inaf_dir(paths.GRB_MODEL_RAW_DIR)
    std = ms.rescaled_to_eiso(E_ISO_REF)
    fig_catalogue_shapes(ms)
    fig_eiso_standardization(ms, std)
    fig_peaks_vs_theta(ms, std)
    fig_phase_alignment(std)
    fig_theta_sweep(std)
    fig_tail_77deg(ms)


if __name__ == "__main__":
    main()
