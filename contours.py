r"""
Fisher \(\Delta\chi^2\) contour experiments: styles, colormaps, and multi-panel layouts.

Uses a **3-parameter** Gaussian Fisher \((f_{\mathrm{NL}}, n_s, A_s)\) with Planck-style priors
on \(n_s\) and \(A_s\). Pairwise plots marginalize the third parameter (full \(3\times3\) cov,
then \(2\times2\) inverse precision in each plane).

For **three** jointly constrained parameters, constant-\(\Delta\chi^2\) regions are **ellipsoids**
\(\{\delta\theta : \delta\theta^\top \mathbf{F}\,\delta\theta = \Delta\chi^2\}\) with thresholds
from \(\chi^2\) with 3 degrees of freedom (not the same numbers as the 2D case). One script
output draws these as 3D surfaces (``contours_3d_ellipsoids.pdf``). Matplotlib does not build
arbitrary volumetric isosurfaces; for smooth “egg” shapes the analytic ellipsoid is appropriate.

Run from this directory::

    python3 contours.py

Save figures only (default; no GUI). Add ``--show`` to open each figure interactively::

    python3 contours.py

To capture printed paths or logs to a text file, redirect stdout/stderr::

    python3 contours.py > contours_run.txt 2>&1

By default, PDFs are written under ``cmbs4/results/contours_pixie/`` (PIXIE) or
``cmbs4/results/contours_specter/`` with ``--specter`` — see ``output_paths.py``.
Override with ``--output-dir``. ``run_specter_contours.py`` runs this module with SPECTER output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fisher_matrix import (
    AS_FID_LEGACY,
    SIGMA_AS_PLANCK2018,
    W_MU_INV_PIXIE,
    W_MU_INV_SPECTER,
    default_ell_grid,
    fisher_muT_general,
)
try:
    from .output_paths import contours_pixie_dir, contours_specter_dir, ensure_dir
    from .plot_params import apply_plot_params
except ImportError:
    from output_paths import contours_pixie_dir, contours_specter_dir, ensure_dir
    from plot_params import apply_plot_params

# Joint 68% / 95% / 99% \Delta\chi^2 for 2 Gaussian parameters
DELTA_CHI2_LEVELS_2D = (2.30, 5.99, 9.21)

# Joint confidence ellipsoid thresholds for 3 Gaussian parameters (\chi^2_3)
# (68%, 95%, 99% enclosed probability for the quadratic form)
DELTA_CHI2_LEVELS_3D = (3.529159, 7.814728, 11.344867)

# --- cosmology / instrument (match contours.py / main_3d conventions) ---
FWHM_DEG = 1.6
NS_FID = 0.965
K_P = 0.002
K_D_I = 1.1e4
K_D_F = 46.0
FNL_FID = 25_000.0
SIGMA_NS_PRIOR = 0.004
DNS_STEP = 5e-5


def marginal_delta_chi2_2d(
    cov_full: np.ndarray,
    i: int,
    j: int,
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
) -> np.ndarray:
    r"""
    \Delta\chi^2 for the marginalized 2D Gaussian of parameters (i, j) about (x0, y0).
    """
    block = cov_full[np.ix_([i, j], [i, j])]
    prec = np.linalg.inv(block)
    X, Y = np.meshgrid(x, y, indexing="xy")
    d0 = X - x0
    d1 = Y - y0
    return (
        prec[0, 0] * d0**2
        + 2.0 * prec[0, 1] * d0 * d1
        + prec[1, 1] * d1**2
    )


def _grid_for_pair(
    cov: np.ndarray,
    i: int,
    j: int,
    c0: float,
    c1: float,
    *,
    n_grid: int,
    sigma_extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sig = np.sqrt(np.diag(cov))
    sx = sigma_extent * sig[i]
    sy = sigma_extent * sig[j]
    gx = np.linspace(c0 - sx, c0 + sx, n_grid)
    gy = np.linspace(c1 - sy, c1 + sy, n_grid)
    chi2 = marginal_delta_chi2_2d(cov, i, j, gx, gy, c0, c1)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    return X, Y, chi2


def _fisher_ellipsoid_mesh(
    P: np.ndarray,
    theta0: np.ndarray,
    k_chi2: float,
    nu: int = 48,
    nv: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Parametric mesh for the surface
    \(\{\theta : (\theta-\theta_0)^\top \mathbf{P}(\theta-\theta_0) = k\}\)
    with \(\mathbf{P}\) the precision matrix and \(k = \Delta\chi^2\) (not \(\sqrt{k}\)).
    """
    w, V = np.linalg.eigh(P)
    w = np.maximum(w, 1e-18)
    u = np.linspace(0, 2 * np.pi, nu, endpoint=False)
    vv = np.linspace(0, np.pi, nv)
    U, VV = np.meshgrid(u, vv)
    wx = np.cos(U) * np.sin(VV)
    wy = np.sin(U) * np.sin(VV)
    wz = np.cos(VV)
    omega = np.stack([wx, wy, wz], axis=0)
    scales = np.sqrt(k_chi2) / np.sqrt(w)
    a = scales.reshape(3, 1, 1) * omega
    d = (V @ a.reshape(3, -1)).reshape(3, nv, nu)
    X = theta0[0] + d[0]
    Y = theta0[1] + d[1]
    Z = theta0[2] + d[2]
    return X, Y, Z


def _precision_in_plot_coords(P_physical: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Map (f_NL, n_s, A_s) to plot axes (f_NL/1e4, n_s, 10^9 A_s) so 3D axes are readable.

    If d_plot = T^{-1} d_phys with T = diag(1e4, 1, 1e-9), then
    d_phys^T P d_phys = d_plot^T (T P T) d_plot.
    """
    T = np.diag([1e4, 1.0, 1e-9])
    P_plot = T @ P_physical @ T
    return P_plot, T


def variant_3d_ellipsoid_surfaces(
    F_total: np.ndarray,
    *,
    fnl_fid: float,
    ns_fid: float,
    as_fid: float,
    outdir: Path,
    show: bool,
) -> Path:
    r"""
    3D **joint** Fisher ellipsoids: constant \(\Delta\chi^2 = \chi^2_{3,\mathrm{CL}}\) surfaces
    in \((f_{\mathrm{NL}}, n_s, A_s)\) space (scaled axes for display).
    """
    import matplotlib.pyplot as plt

    P_plot, _ = _precision_in_plot_coords(F_total)
    theta0_plot = np.array([fnl_fid / 1e4, ns_fid, as_fid * 1e9])

    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    colors = ("#1b4332", "#2d6a4f", "#40916c")
    alphas = (0.35, 0.28, 0.22)
    labels = (
        r"68\% CL ($\Delta\chi^2 \approx 3.53$)",
        r"95\% CL ($\Delta\chi^2 \approx 7.81$)",
        r"99\% CL ($\Delta\chi^2 \approx 11.34$)",
    )

    for k, col, al, lab in zip(DELTA_CHI2_LEVELS_3D, colors, alphas, labels):
        X, Y, Z = _fisher_ellipsoid_mesh(P_plot, theta0_plot, k)
        ax.plot_surface(
            X,
            Y,
            Z,
            color=col,
            alpha=al,
            linewidth=0,
            antialiased=True,
            shade=True,
            rstride=2,
            cstride=2,
        )

    ax.scatter(
        [theta0_plot[0]],
        [theta0_plot[1]],
        [theta0_plot[2]],
        color="crimson",
        s=36,
        depthshade=True,
        label="fiducial",
    )

    ax.set_xlabel(r"$f_{\mathrm{NL}} / 10^4$")
    ax.set_ylabel(r"$n_s$")
    ax.set_zlabel(r"$10^9 A_s$")
    ax.set_title(
        r"Joint 3D Fisher: $\Delta\chi^2$ ellipsoids ($\chi^2_3$ thresholds)"
    )
    ax.view_init(elev=22, azim=55)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=c, edgecolor="0.3", alpha=a, label=lb)
        for c, a, lb in zip(colors, alphas, labels)
    ]
    legend_handles.append(
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson", markersize=8, label="fiducial")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    outp = outdir / "contours_3d_ellipsoids.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def apply_publication_rc() -> None:
    import matplotlib as mpl

    apply_plot_params()
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Helvetica",
                "Arial",
                "sans-serif",
            ],
            "mathtext.fontset": "dejavusans",
        }
    )


def variant_legacy_2param_fnl_ns(
    *,
    ell: np.ndarray,
    outdir: Path,
    show: bool,
    w_mu_inv: float = W_MU_INV_PIXIE,
) -> Path:
    """Original 2-parameter Fisher (no A_s): f_NL vs n_s, discrete bands."""
    import matplotlib.pyplot as plt

    r = fisher_muT_general(
        ell,
        FWHM_DEG,
        FNL_FID,
        NS_FID,
        K_D_I,
        K_D_F,
        K_P,
        w_mu_inv=w_mu_inv,
        dns_step=DNS_STEP,
        sigma_ns_prior=SIGMA_NS_PRIOR,
        sigma_As_prior=None,
        include_As=False,
        use_b_analytic=False,
    )
    F = r.F_total
    fnl_ctr, ns_ctr = FNL_FID, NS_FID
    fnl = np.linspace(
        fnl_ctr - 3 * r.sigma_fnl_marg,
        fnl_ctr + 3 * r.sigma_fnl_marg,
        200,
    )
    ns = np.linspace(
        ns_ctr - 3 * r.sigma_ns_marg,
        ns_ctr + 3 * r.sigma_ns_marg,
        200,
    )
    FN, NS = np.meshgrid(fnl, ns)
    d0 = FN - fnl_ctr
    d1 = NS - ns_ctr
    chi2 = F[0, 0] * d0**2 + 2 * F[0, 1] * d0 * d1 + F[1, 1] * d1**2

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.contour(FN, NS, chi2, levels=list(DELTA_CHI2_LEVELS_2D), colors="k", linewidths=0.9)
    ax.contourf(
        FN,
        NS,
        chi2,
        levels=[0, *DELTA_CHI2_LEVELS_2D, 1e9],
        colors=["#e0ecf4", "#b6d0e8", "#8cbad9", "#6a9ac4"],
        alpha=0.9,
    )
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(r"2-param Fisher: $\Delta\chi^2$ ($f_{\mathrm{NL}}^{\mathrm{fid}}=25000$)")
    ax.plot(fnl_ctr, ns_ctr, "k+", ms=10, mew=1.2)
    fig.tight_layout()
    outp = outdir / "contours_legacy_2param_fnl_ns.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def variant_triple_discrete(
    cov: np.ndarray,
    *,
    fnl_fid: float,
    ns_fid: float,
    as_fid: float,
    outdir: Path,
    show: bool,
    n_grid: int = 200,
    sigma_extent: float = 3.0,
) -> Path:
    """Three pairwise marginals; soft discrete fills + black contours (main_3d-like, styled)."""
    import matplotlib.pyplot as plt

    panels = [
        (0, 1, r"$f_{\mathrm{NL}}$", r"$n_s$", fnl_fid, ns_fid),
        (0, 2, r"$f_{\mathrm{NL}}$", r"$A_s$", fnl_fid, as_fid),
        (1, 2, r"$n_s$", r"$A_s$", ns_fid, as_fid),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.0))
    try:
        cmap = plt.colormaps["Blues"]
    except (AttributeError, KeyError):
        cmap = plt.cm.get_cmap("Blues")
    band_colors = [cmap(0.25 + 0.2 * k) for k in range(4)]

    for ax, (i, j, lx, ly, c0, c1) in zip(axes, panels):
        X, Y, chi2 = _grid_for_pair(
            cov, i, j, c0, c1, n_grid=n_grid, sigma_extent=sigma_extent
        )
        ax.contourf(
            X,
            Y,
            chi2,
            levels=[0, *DELTA_CHI2_LEVELS_2D, 1e9],
            colors=band_colors,
            alpha=0.85,
        )
        ax.contour(
            X,
            Y,
            chi2,
            levels=list(DELTA_CHI2_LEVELS_2D),
            colors="0.15",
            linewidths=0.85,
        )
        ax.plot(c0, c1, "+", color="crimson", ms=10, mew=1.2)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title("marginal $\\Delta\\chi^2$")

    fig.suptitle(
        rf"3D Fisher — pairwise marginals ($f_{{\mathrm{{NL}}}}^{{\mathrm{{fid}}}}={fnl_fid:.0f}$)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    outp = outdir / "contours_triple_discrete.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def variant_triple_cmap(
    cov: np.ndarray,
    *,
    fnl_fid: float,
    ns_fid: float,
    as_fid: float,
    outdir: Path,
    show: bool,
    n_grid: int = 220,
    sigma_extent: float = 3.0,
) -> Path:
    """Three pairwise marginals with continuous ``contourf`` + per-panel colorbars (cividis)."""
    import matplotlib.pyplot as plt

    panels = [
        (0, 1, r"$f_{\mathrm{NL}}$", r"$n_s$", fnl_fid, ns_fid),
        (0, 2, r"$f_{\mathrm{NL}}$", r"$A_s$", fnl_fid, as_fid),
        (1, 2, r"$n_s$", r"$A_s$", ns_fid, as_fid),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), constrained_layout=True)

    vmax_global = 0.0
    grids: list[tuple] = []
    for ax, (i, j, lx, ly, c0, c1) in zip(axes, panels):
        X, Y, chi2 = _grid_for_pair(
            cov, i, j, c0, c1, n_grid=n_grid, sigma_extent=sigma_extent
        )
        vmax_global = max(vmax_global, float(chi2.max()))
        grids.append((ax, X, Y, chi2, lx, ly, c0, c1))

    # Same color scale across panels for comparability
    levels = np.linspace(0.0, vmax_global, 80)
    for ax, X, Y, chi2, lx, ly, c0, c1 in grids:
        cf = ax.contourf(
            X,
            Y,
            chi2,
            levels=levels,
            cmap="cividis",
            vmin=0.0,
            vmax=vmax_global,
            extend="max",
        )
        ax.contour(
            X,
            Y,
            chi2,
            levels=list(DELTA_CHI2_LEVELS_2D),
            colors="white",
            linewidths=0.9,
            alpha=0.95,
        )
        ax.plot(c0, c1, "+", color="orangered", ms=9, mew=1.1)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        fig.colorbar(cf, ax=ax, shrink=0.78, label=r"$\Delta\chi^2$")

    fig.suptitle(
        rf"3D Fisher — gradient fill + contours ($f_{{\mathrm{{NL}}}}^{{\mathrm{{fid}}}}={fnl_fid:.0f}$)",
        fontsize=12,
    )
    outp = outdir / "contours_triple_cmap_cividis.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def variant_fnl_ns_lognorm_colorbar(
    cov: np.ndarray,
    *,
    fnl_fid: float,
    ns_fid: float,
    outdir: Path,
    show: bool,
    n_grid: int = 250,
    sigma_extent: float = 4.5,
) -> Path:
    """
    Single (f_NL, n_s) marginal with log-scaled color mapping (cf. contourf_log gallery).
    Uses LogNorm for positive \\Delta\\chi^2; small floor avoids log(0).
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    X, Y, chi2 = _grid_for_pair(
        cov, 0, 1, fnl_fid, ns_fid, n_grid=n_grid, sigma_extent=sigma_extent
    )
    floor = max(float(chi2.min()), 1e-4)
    chi2_plot = np.clip(chi2, floor, None)

    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    norm = mpl.colors.LogNorm(vmin=floor, vmax=float(chi2_plot.max()))
    levels = np.geomspace(floor, float(chi2_plot.max()), 60)
    cf = ax.contourf(X, Y, chi2_plot, levels=levels, cmap="magma", norm=norm, extend="max")
    ax.contour(
        X,
        Y,
        chi2,
        levels=list(DELTA_CHI2_LEVELS_2D),
        colors="cyan",
        linewidths=0.95,
        linestyles="-",
    )
    ax.plot(fnl_fid, ns_fid, "+", color="lime", ms=11, mew=1.3)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(
        r"Marginal $(f_{\mathrm{NL}}, n_s)$ — LogNorm color scale ($A_s$ integrated out)"
    )
    cb = fig.colorbar(cf, ax=ax, label=r"$\Delta\chi^2$")
    cb.ax.minorticks_on()
    fig.tight_layout()
    outp = outdir / "contours_fnl_ns_lognorm_magma.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def variant_fnl_ns_line_clabel(
    cov: np.ndarray,
    *,
    fnl_fid: float,
    ns_fid: float,
    outdir: Path,
    show: bool,
    n_grid: int = 200,
    sigma_extent: float = 3.0,
) -> Path:
    """Contour-only demo with inline labels (matplotlib contour_demo style)."""
    import matplotlib.pyplot as plt

    X, Y, chi2 = _grid_for_pair(
        cov, 0, 1, fnl_fid, ns_fid, n_grid=n_grid, sigma_extent=sigma_extent
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    cs = ax.contour(
        X,
        Y,
        chi2,
        levels=list(DELTA_CHI2_LEVELS_2D),
        colors=["#1b4332", "#2d6a4f", "#40916c"],
        linewidths=1.2,
    )
    # Δχ² = 2.30, 5.99, 9.21 ↔ joint 68%, 95%, 99% for 2 Gaussian params (~1σ, 2σ, 3σ)
    _sigma_labels = (r"$1\sigma$", r"$2\sigma$", r"$3\sigma$")
    clabel_fmt = {
        lev: lab for lev, lab in zip(DELTA_CHI2_LEVELS_2D, _sigma_labels)
    }
    ax.clabel(cs, inline=True, fontsize=9, fmt=clabel_fmt)
    ax.plot(fnl_fid, ns_fid, "ko", ms=5)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(r"Labeled contours — marginal $(f_{\mathrm{NL}}, n_s)$")
    ax.grid(True, alpha=0.35, linestyle=":")
    fig.tight_layout()
    outp = outdir / "contours_fnl_ns_labeled_lines.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def variant_fnl_ns_hatch(
    cov: np.ndarray,
    *,
    fnl_fid: float,
    ns_fid: float,
    outdir: Path,
    show: bool,
    n_grid: int = 200,
    sigma_extent: float = 3.0,
) -> Path:
    """Filled regions with hatching between confidence levels (contourf_demo-style)."""
    import matplotlib.pyplot as plt

    X, Y, chi2 = _grid_for_pair(
        cov, 0, 1, fnl_fid, ns_fid, n_grid=n_grid, sigma_extent=sigma_extent
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    levels = [0.0, *DELTA_CHI2_LEVELS_2D, 1e9]
    band_colors = ["#f0f4f8", "#d9e6f2", "#b8d4ea", "#90c3e6"]
    hatches = [None, "///", "\\\\\\", "xxx"]
    for lo, hi, col, h in zip(levels[:-1], levels[1:], band_colors, hatches):
        ax.contourf(
            X,
            Y,
            chi2,
            levels=[lo, hi],
            colors=[col],
            hatches=[h],
            alpha=0.75,
        )

    ax.contour(
        X,
        Y,
        chi2,
        levels=list(DELTA_CHI2_LEVELS_2D),
        colors="0.2",
        linewidths=0.9,
    )
    ax.plot(fnl_fid, ns_fid, "k+", ms=10, mew=1.2)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(r"Hatched bands — marginal $(f_{\mathrm{NL}}, n_s)$")
    fig.tight_layout()
    outp = outdir / "contours_fnl_ns_hatched.pdf"
    fig.savefig(outp, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return outp


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Fisher contour style experiments (2D and 3D marginal projections)."
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for PDF outputs (default: cmbs4/results/contours_pixie, "
            "or contours_specter with --specter)."
        ),
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Display each figure interactively (blocks until windows close).",
    )
    p.add_argument(
        "--specter",
        action="store_true",
        help=(
            "Use SPECTER mu-noise scaling (w_mu_inv = (2e-9)^2); "
            "default is PIXIE (W_MU_INV_PIXIE)."
        ),
    )
    args = p.parse_args(argv)

    if args.output_dir is None:
        outdir = contours_specter_dir() if args.specter else contours_pixie_dir()
    else:
        outdir = args.output_dir
    ensure_dir(outdir)

    w_mu_inv = W_MU_INV_SPECTER if args.specter else W_MU_INV_PIXIE

    apply_publication_rc()

    ell = default_ell_grid(FWHM_DEG)
    r3 = fisher_muT_general(
        ell,
        FWHM_DEG,
        FNL_FID,
        NS_FID,
        K_D_I,
        K_D_F,
        K_P,
        w_mu_inv=w_mu_inv,
        dns_step=DNS_STEP,
        sigma_ns_prior=SIGMA_NS_PRIOR,
        sigma_As_prior=SIGMA_AS_PLANCK2018,
        use_b_analytic=False,
    )
    assert r3.param_names == ("fnl", "ns", "As")
    cov = r3.cov_marginal

    saved: list[Path] = []
    saved.append(
        variant_legacy_2param_fnl_ns(
            ell=ell, outdir=outdir, show=args.show, w_mu_inv=w_mu_inv
        )
    )
    saved.append(
        variant_triple_discrete(
            cov,
            fnl_fid=FNL_FID,
            ns_fid=NS_FID,
            as_fid=AS_FID_LEGACY,
            outdir=outdir,
            show=args.show,
        )
    )
    saved.append(
        variant_triple_cmap(
            cov,
            fnl_fid=FNL_FID,
            ns_fid=NS_FID,
            as_fid=AS_FID_LEGACY,
            outdir=outdir,
            show=args.show,
        )
    )
    saved.append(
        variant_fnl_ns_lognorm_colorbar(
            cov,
            fnl_fid=FNL_FID,
            ns_fid=NS_FID,
            outdir=outdir,
            show=args.show,
        )
    )
    saved.append(
        variant_fnl_ns_line_clabel(
            cov,
            fnl_fid=FNL_FID,
            ns_fid=NS_FID,
            outdir=outdir,
            show=args.show,
        )
    )
    saved.append(
        variant_fnl_ns_hatch(
            cov,
            fnl_fid=FNL_FID,
            ns_fid=NS_FID,
            outdir=outdir,
            show=args.show,
        )
    )
    saved.append(
        variant_3d_ellipsoid_surfaces(
            r3.F_total,
            fnl_fid=FNL_FID,
            ns_fid=NS_FID,
            as_fid=AS_FID_LEGACY,
            outdir=outdir,
            show=args.show,
        )
    )

    print("Wrote:", file=sys.stderr)
    print(f"  w_mu_inv = {w_mu_inv} (SPECTER)" if args.specter else f"  w_mu_inv = {w_mu_inv} (PIXIE)", file=sys.stderr)
    for path in saved:
        print(f"  {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
