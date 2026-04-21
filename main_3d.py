'3-parameter Fisher forecast: (f_NL, n_s, A_s) with Planck-style priors on n_s and A_s.'

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from fisher_matrix import (
    AS_FID_LEGACY,
    AS_FID_PLANCK2018,
    FisherMuTResult,
    SIGMA_AS_PLANCK2018,
    W_MU_INV_PIXIE,
    W_MU_INV_SPECTER,
    default_ell_grid,
    fisher_1d_fnl_only,
    fisher_muT_general,
)
from spectra import (
    CL_TT_TXT_FIDUCIAL,
    As_brackets_relative,
    Cl_TT,
    cl_tt_on_ell_grid,
    dCl_TT_dAs_numerical,
    dCl_TT_dns_numerical,
    load_ClTT_planck18,
    ns_brackets_absolute,
)
try:
    from .output_paths import ensure_dir, forecast_tables_dir, paper_results_dir
    from .plot_params import apply_plot_params
except ImportError:
    from output_paths import ensure_dir, forecast_tables_dir, paper_results_dir
    from plot_params import apply_plot_params

# Joint 68% / 95% / 99% \Delta\chi^2 for 2 Gaussian parameters
DELTA_CHI2_LEVELS_2D = (2.30, 5.99, 9.21)

# Fiducial f_NL used for contour plots (same spirit as contours.py)
FNl_FID_FOR_PLOTS = 25_000.0

# Indices in cov_marginal / param_names == ("fnl", "ns", "As")
I_FNL, I_NS, I_AS = 0, 1, 2


def _default_cl_tt_txt_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _cl_tt_camb_files_ready(cl_tt_txt_dir):
    return os.path.isfile(os.path.join(cl_tt_txt_dir, CL_TT_TXT_FIDUCIAL))


def report_camb_cl_tt_numerical_derivatives(
    ell,
    *,
    ns_fid,
    as_fid_planck,
    cl_tt_txt_dir,
):
    """Report finite-difference CAMB Cl_TT derivatives on the selected multipole grid."""
    bundle = load_ClTT_planck18(cl_tt_txt_dir)
    ns_hi, ns_lo = ns_brackets_absolute(ns_fid)
    as_hi, as_lo = As_brackets_relative(as_fid_planck)
    d_cl_dns = dCl_TT_dns_numerical(
        cl_tt_on_ell_grid(bundle["ns_high"], ell),
        cl_tt_on_ell_grid(bundle["ns_low"], ell),
        ns_hi,
        ns_lo,
    )
    d_cl_das = dCl_TT_dAs_numerical(
        cl_tt_on_ell_grid(bundle["As_high"], ell),
        cl_tt_on_ell_grid(bundle["As_low"], ell),
        as_hi,
        as_lo,
    )
    print("CAMB Cl_TT numerical derivatives (from text files in {}):".format(cl_tt_txt_dir))
    print(f"  n_s brackets: {ns_lo:g}, {ns_hi:g}  (d n_s = {ns_hi - ns_lo:g})")
    print(f"  A_s brackets: {as_lo:.6e}, {as_hi:.6e}  (relative step from spectra.As_brackets_relative)")
    print(
        f"  |d Cl_TT/d n_s| on ell grid: min={np.nanmin(np.abs(d_cl_dns)):.3e}, "
        f"max={np.nanmax(np.abs(d_cl_dns)):.3e}"
    )
    print(
        f"  |d Cl_TT/d A_s| on ell grid: min={np.nanmin(np.abs(d_cl_das)):.3e}, "
        f"max={np.nanmax(np.abs(d_cl_das)):.3e} per (Delta_R^2)"
    )
    print()


def report_camb_vs_analytic_cl_tt_average_difference(
    ell,
    *,
    as_fid,
    cl_tt_txt_dir,
):
    """Report average differences between CAMB and analytic Cl_TT on the selected grid."""
    bundle = load_ClTT_planck18(cl_tt_txt_dir)
    cl_tt_camb = cl_tt_on_ell_grid(bundle["fiducial"], ell)
    cl_tt_analytic = Cl_TT(ell, A_s=as_fid)
    diff = cl_tt_camb - cl_tt_analytic
    abs_diff = np.abs(diff)
    rel_abs_diff = abs_diff / np.maximum(np.abs(cl_tt_analytic), 1.0e-300)

    print("CAMB vs analytic Cl_TT average difference on ell grid:")
    print(
        "  CAMB Cl_TT: "
        f"min={np.min(cl_tt_camb):.6e}, max={np.max(cl_tt_camb):.6e}, mean={np.mean(cl_tt_camb):.6e}"
    )
    print(
        "  analytic Cl_TT: "
        f"min={np.min(cl_tt_analytic):.6e}, max={np.max(cl_tt_analytic):.6e}, mean={np.mean(cl_tt_analytic):.6e}"
    )
    print(f"  mean(CAMB - analytic) = {np.mean(diff):.6e}")
    print(f"  mean(|CAMB - analytic|) = {np.mean(abs_diff):.6e}")
    print(f"  mean(|CAMB - analytic| / |analytic|) = {np.mean(rel_abs_diff):.6e}")
    print()


def marginal_corr(cov, i, j):
    """Pearson correlation from covariance block, matching ``corr_fnl_ns`` for (0, 1)."""
    if cov[i, i] <= 0 or cov[j, j] <= 0:
        return 0.0
    return float(cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]))


def marginal_delta_chi2_2d(
    cov_full,
    i,
    j,
    x,
    y,
    x0,
    y0,
):
    r"""Compute marginalized 2D delta-chi-squared for a selected parameter pair."""
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


def plot_pairwise_marginal_contours(
    r,
    *,
    fnl_fid,
    ns_fid,
    as_fid,
    n_grid = 200,
    sigma_extent = 3.0,
    outfile = None,
    show = True,
):
    import matplotlib.pyplot as plt

    cov = r.cov_marginal # later take 2x2 blocks for each pair of params to marginalize over the third param
    sig = np.sqrt(np.diag(cov))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

    panels = [
        (0, 1, r"$f_{\mathrm{NL}}$", r"$n_s$", fnl_fid, ns_fid),
        (0, 2, r"$f_{\mathrm{NL}}$", r"$A_s$", fnl_fid, as_fid),
        (1, 2, r"$n_s$", r"$A_s$", ns_fid, as_fid),
    ]

    for ax, (i, j, lx, ly, c0, c1) in zip(axes, panels):
        sx = sigma_extent * sig[i]
        sy = sigma_extent * sig[j]
        gx = np.linspace(c0 - sx, c0 + sx, n_grid)
        gy = np.linspace(c1 - sy, c1 + sy, n_grid)
        chi2 = marginal_delta_chi2_2d(cov, i, j, gx, gy, c0, c1)
        X, Y = np.meshgrid(gx, gy, indexing="xy")
        ax.contourf(
            X,
            Y,
            chi2,
            levels=[0, *DELTA_CHI2_LEVELS_2D, 1e9],
            alpha=0.25,
        )
        ax.contour(X, Y, chi2, levels=list(DELTA_CHI2_LEVELS_2D), colors="k", linewidths=0.8)
        ax.plot(c0, c1, "k+", ms=10, mew=1.2)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title("marginal $\\Delta\\chi^2$")

    fig.suptitle(
        rf"3D Fisher: pairwise marginal contours ($f_{{\mathrm{{NL}}}}^{{\mathrm{{fid}}}}={fnl_fid:.0f}$)",
        fontsize=11,
    )
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")
        print(f"Saved contour figure to {outfile}", file=sys.stderr)
    if show:
        plt.show()
    else:
        plt.close(fig)


def _grid_for_fnl_ns(
    cov_full,
    fnl_fid,
    ns_fid,
    *,
    n_grid = 200,
    sigma_extent = 3.0,
):
    """Helper: grid and Δχ² for marginalized (f_NL, n_s) only."""
    sig = np.sqrt(np.diag(cov_full))
    i, j = I_FNL, I_NS
    sx = sigma_extent * sig[i]
    sy = sigma_extent * sig[j]
    gx = np.linspace(fnl_fid - sx, fnl_fid + sx, n_grid)
    gy = np.linspace(ns_fid - sy, ns_fid + sy, n_grid)
    chi2 = marginal_delta_chi2_2d(cov_full, i, j, gx, gy, fnl_fid, ns_fid)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    return X, Y, chi2


def plot_fnl_ns_single(
    r,
    *,
    fnl_fid,
    ns_fid,
    color = "#3193A2",
    outfile = None,
    show = True,
    n_grid = 200,
    sigma_extent = 3.0,
):
    """2D marginalized (f_NL, n_s) contours in a single color."""
    import matplotlib.pyplot as plt

    X, Y, chi2 = _grid_for_fnl_ns(r.cov_marginal, fnl_fid, ns_fid, n_grid=n_grid, sigma_extent=sigma_extent)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    # 1σ / 2σ only
    lev1, lev2 = DELTA_CHI2_LEVELS_2D[0], DELTA_CHI2_LEVELS_2D[1]
    # Filled 1σ region
    ax.contourf(
        X,
        Y,
        chi2,
        levels=[0.0, lev1],
        colors=[color],
        alpha=0.25,
    )
    # Solid line for 1σ, dotted for 2σ
    ax.contour(
        X,
        Y,
        chi2,
        levels=[lev1, lev2],
        colors=[color, color],
        linewidths=1.4,
        linestyles=["solid", "dotted"],
    )
    # Fiducial crosshairs
    ax.axvline(fnl_fid, color="0.2", ls="--", lw=1.0, alpha=0.35)
    ax.axhline(ns_fid, color="0.2", ls="--", lw=1.0, alpha=0.35)
    ax.plot(fnl_fid, ns_fid, "k+", ms=10, mew=1.2)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(r"marginal $(f_{\mathrm{NL}}, n_s)$")
    ax.grid(True, alpha=0.35, linestyle=":")
    fig.tight_layout()
    if outfile is not None:
        fig.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_fnl_ns_pixie_vs_specter(
    ell,
    fwhm_deg,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    sigma_ns_prior,
    sigma_As_prior,
    as_fid,
    cl_tt_txt_dir,
    b_override,
    outfile,
    n_grid = 200,
    sigma_extent = 3.0,
    show = True,
):
    """Overlay PIXIE and SPECTER marginalized (f_NL, n_s) contours on one plot."""
    import matplotlib.pyplot as plt
    from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER

    r_pixie = fisher_muT_general(
        ell,
        fwhm_deg,
        fnl_fid,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=W_MU_INV_PIXIE,
        dns_step=5e-5,
        sigma_ns_prior=sigma_ns_prior,
        sigma_As_prior=sigma_As_prior,
        use_b_analytic=False,
        b_override=b_override,
        As_fid=as_fid,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    r_specter = fisher_muT_general(
        ell,
        fwhm_deg,
        fnl_fid,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=W_MU_INV_SPECTER,
        dns_step=5e-5,
        sigma_ns_prior=sigma_ns_prior,
        sigma_As_prior=sigma_As_prior,
        use_b_analytic=False,
        b_override=b_override,
        As_fid=as_fid,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )

    Xp, Yp, chi2_pix = _grid_for_fnl_ns(r_pixie.cov_marginal, fnl_fid, ns_fid, n_grid=n_grid, sigma_extent=sigma_extent)
    Xs, Ys, chi2_spec = _grid_for_fnl_ns(
        r_specter.cov_marginal, fnl_fid, ns_fid, n_grid=n_grid, sigma_extent=sigma_extent
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    lev1, lev2 = DELTA_CHI2_LEVELS_2D[0], DELTA_CHI2_LEVELS_2D[1]
    # Filled 1σ regions
    ax.contourf(
        Xp,
        Yp,
        chi2_pix,
        levels=[0.0, lev1],
        colors=["#3193A2"],
        alpha=0.25,
    )
    ax.contourf(
        Xs,
        Ys,
        chi2_spec,
        levels=[0.0, lev1],
        colors=["#C45A62"],
        alpha=0.25,
    )
    # PIXIE: solid 1σ, dotted 2σ
    ax.contour(
        Xp,
        Yp,
        chi2_pix,
        levels=[lev1, lev2],
        colors=["#3193A2", "#3193A2"],
        linewidths=1.4,
        linestyles=["solid", "dotted"],
    )
    # SPECTER: solid 1σ, dotted 2σ
    ax.contour(
        Xs,
        Ys,
        chi2_spec,
        levels=[lev1, lev2],
        colors=["#C45A62", "#C45A62"],
        linewidths=1.4,
        linestyles=["solid", "dotted"],
    )
    # Fiducial crosshairs
    ax.axvline(fnl_fid, color="0.2", ls="--", lw=1.0, alpha=0.35)
    ax.axhline(ns_fid, color="0.2", ls="--", lw=1.0, alpha=0.35)
    ax.plot(fnl_fid, ns_fid, "k+", ms=10, mew=1.2)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    label_cltt = "CAMB $C_\\ell^{TT}$" if cl_tt_txt_dir is not None else "analytic $C_\\ell^{TT}$"
    ax.set_title(rf"marginal $(f_{{\mathrm{{NL}}}}, n_s)$")
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.legend(
        handles=[
            plt.Line2D([], [], color="#3193A2", lw=1.4, ls="-", label="PIXIE"),
            plt.Line2D([], [], color="#C45A62", lw=1.4, ls="--", label="SPECTER"),
        ],
        frameon=False,
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

def print_forecast_table(
    ell,
    fwhm_deg,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    sigma_ns_planck,
    sigma_as_planck,
    fnl_fiducials,
    *,
    cl_tt_txt_dir = None,
    as_fid = AS_FID_LEGACY,
    w_mu_inv = W_MU_INV_SPECTER,
    mu_noise_label = "SPECTER",
    b_override = None,
):
    print("muT Fisher forecast -- 3 parameters (f_NL, n_s, A_s)")
    print(f"  l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg}^\\circ, n_s = {ns_fid}")
    print(f"  mu autospectrum noise: w_mu^-1 = {w_mu_inv:.6e} ({mu_noise_label}; see beam.N_mu_mu)")
    print(f"  A_s fid (Delta_R^2 at pivot): {as_fid:.6e}")
    if cl_tt_txt_dir is not None:
        print(f"  C_l^TT for noise: CAMB from text files in {cl_tt_txt_dir}")
    else:
        print("  C_l^TT for noise: analytic Sachs-Wolfe template (spectra.Cl_TT)")
    print(f"  Prior sigma(n_s) = {sigma_ns_planck}")
    print(f"  Prior sigma(A_s) = {sigma_as_planck:.6e} (Planck 2018 order-of-magnitude; see spectra.py)")
    if b_override is None:
        print("  b(l, n_s): numerical b_integral, dns_step = 5e-5")
    else:
        print(f"  b(l, n_s): fixed constant b = {b_override:g}")
    print()

    print(
        f"{'f_NL fid':>12}  {'sigma(f_NL)':>14}  {'sigma(n_s)':>14}  {'sigma(A_s)':>14}  "
        f"{'corr(f_NL,n_s)':>14}  {'corr(f_NL,A_s)':>14}"
    )
    print("-" * 98)

    for fnl in fnl_fiducials:
        r = fisher_muT_general(
            ell,
            fwhm_deg,
            fnl,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            w_mu_inv=w_mu_inv,
            dns_step=5e-5,
            sigma_ns_prior=sigma_ns_planck,
            sigma_As_prior=sigma_as_planck,
            use_b_analytic=False,
            b_override=b_override,
            As_fid=as_fid,
            cl_tt_txt_dir=cl_tt_txt_dir,
        )
        assert r.param_names == ("fnl", "ns", "As")
        c = r.cov_marginal
        rho_fnl_as = marginal_corr(c, I_FNL, I_AS)
        print(
            f"{fnl:12.0f}  {r.sigma_fnl_marg:14.6e}  {r.sigma_ns_marg:14.6e}  "
            f"{r.sigma_As_marg:14.6e}  {r.corr_fnl_ns:14.4f}  {rho_fnl_as:14.4f}"
        )

    print()


def write_forecast_table_to_txt(
    path,
    ell,
    fwhm_deg,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    sigma_ns_planck,
    sigma_as_planck,
    fnl_fiducials,
    *,
    cl_tt_txt_dir,
    as_fid,
    w_mu_inv,
    mu_noise_label,
    b_override,
):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("muT Fisher forecast -- 3 parameters (f_NL, n_s, A_s)")
    lines.append(f"  l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg} deg, n_s = {ns_fid}")
    lines.append(f"  mu autospectrum noise: w_mu^-1 = {w_mu_inv:.6e} ({mu_noise_label})")
    lines.append(f"  A_s fid (Delta_R^2 at pivot): {as_fid:.6e}")
    if cl_tt_txt_dir is not None:
        lines.append(f"  C_l^TT for noise: CAMB from text files in {cl_tt_txt_dir}")
    else:
        lines.append("  C_l^TT for noise: analytic Sachs-Wolfe template (spectra.Cl_TT)")
    lines.append(f"  Prior sigma(n_s) = {sigma_ns_planck}")
    lines.append(f"  Prior sigma(A_s) = {sigma_as_planck:.6e}")
    if b_override is None:
        lines.append("  b(l, n_s): numerical b_integral, dns_step = 5e-5")
    else:
        lines.append(f"  b(l, n_s): fixed constant b = {b_override:g}")
    lines.append("")

    header = (
        f"{'f_NL_fid':>12}  {'sigma_fNL':>14}  {'sigma_ns':>14}  {'sigma_As':>14}  "
        f"{'corr_fNL_ns':>14}  {'corr_fNL_As':>14}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for fnl in fnl_fiducials:
        r = fisher_muT_general(
            ell,
            fwhm_deg,
            fnl,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            w_mu_inv=w_mu_inv,
            dns_step=5e-5,
            sigma_ns_prior=sigma_ns_planck,
            sigma_As_prior=sigma_as_planck,
            use_b_analytic=False,
            b_override=b_override,
            As_fid=as_fid,
            cl_tt_txt_dir=cl_tt_txt_dir,
        )
        c = r.cov_marginal
        rho_fnl_as = marginal_corr(c, I_FNL, I_AS)
        lines.append(
            f"{fnl:12.0f}  {r.sigma_fnl_marg:14.6e}  {r.sigma_ns_marg:14.6e}  {r.sigma_As_marg:14.6e}  "
            f"{r.corr_fnl_ns:14.4f}  {rho_fnl_as:14.4f}"
        )

    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def print_forecast_1d(
    ell,
    fwhm_deg,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    cl_tt_txt_dir = None,
    as_fid = AS_FID_LEGACY,
    w_mu_inv = W_MU_INV_SPECTER,
    mu_noise_label = "SPECTER",
    b_override = None,
):
    f_1d = fisher_1d_fnl_only(
        ell,
        fwhm_deg,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=w_mu_inv,
        As_fid=as_fid,
        use_b_analytic=False,
        b_override=b_override,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    sigma_1d = 1.0 / np.sqrt(f_1d)

    print("muT Fisher forecast -- 1 parameter (f_NL only)")
    print(f"  l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg}^\\circ, n_s = {ns_fid}")
    print(f"  mu autospectrum noise: w_mu^-1 = {w_mu_inv:.6e} ({mu_noise_label}; see beam.N_mu_mu)")
    print(f"  A_s fid (Delta_R^2 at pivot): {as_fid:.6e}")
    if cl_tt_txt_dir is not None:
        print(f"  C_l^TT for noise: CAMB from text files in {cl_tt_txt_dir}")
    else:
        print("  C_l^TT for noise: analytic Sachs-Wolfe template (spectra.Cl_TT)")
    if b_override is None:
        print("  b(l, n_s): numerical b_integral")
    else:
        print(f"  b(l, n_s): fixed constant b = {b_override:g}")
    print()
    print(f"  F_1d = {f_1d:.6e}")
    print(f"  sigma(f_NL) = {sigma_1d:.6e}")
    print()


def main(argv = None):
    apply_plot_params()
    p = argparse.ArgumentParser(description="3D muT Fisher table and pairwise marginal contours.")
    p.add_argument(
        "--no-plot",
        action="store_true",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for pairwise 3D marginal PDF (default: cmbs4/results/main_3d_contours.pdf).",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
    )
    p.add_argument(
        "--cl-tt-txt-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Directory containing cl_tt_fiducial.txt (and bracket files) from planck_cosmology.py. "
            "Default: use this script's directory if those files exist; else analytic Cl_TT."
        ),
    )
    p.add_argument(
        "--no-camb-cltt",
        action="store_true",
        help="Force analytic Cl_TT (ignore CAMB text files even if present).",
    )
    p.add_argument(
        "--pixie",
        action="store_true",
        help="Use PIXIE w_mu^-1 (beam.W_MU_INV_PIXIE) instead of default SPECTER.",
    )
    p.add_argument(
        "--1d",
        "--one-d",
        dest="one_d",
        action="store_true",
        help="Run 1D f_NL-only Fisher (fisher_1d_fnl_only) with numerical b_integral.",
    )
    p.add_argument(
        "--fnl-ns-contours",
        action="store_true",
        help=(
            "Generate standalone (f_NL, n_s) contour plots, including a combined PIXIE vs SPECTER "
            "figure, using colors #3193A2 and #C45A62."
        ),
    )
    p.add_argument(
        "--save-forecasts",
        action="store_true",
        help="Write PIXIE and SPECTER 3D Fisher tables to text files (requires CAMB Cl_TT).",
    )
    p.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Directory for --save-forecasts outputs (default: cmbs4/results/forecast_tables).",
    )
    p.add_argument(
        "--b-fixed",
        type=float,
        default=None,
        metavar="BVAL",
        help="Fix b to a constant value at all ell (example: --b-fixed 100).",
    )
    args = p.parse_args(argv)

    if args.output is None:
        args.output = ensure_dir(paper_results_dir()) / "main_3d_contours.pdf"
    if args.save_dir is None:
        args.save_dir = ensure_dir(forecast_tables_dir())

    w_mu_inv = W_MU_INV_PIXIE if args.pixie else W_MU_INV_SPECTER
    mu_noise_label = "PIXIE" if args.pixie else "SPECTER"
    b_override = args.b_fixed

    fwhm_deg = 1.6
    ns_fid = 0.965
    k_p = 0.002
    k_D_i = 1.1e4
    k_D_f = 46.0

    ell = default_ell_grid(fwhm_deg)
    sigma_ns_planck = 0.004
    sigma_as_planck = SIGMA_AS_PLANCK2018
    fnl_fiducials = (0.0, 1.0, 12_500.0, 25_000.0)

    script_dir = _default_cl_tt_txt_dir()
    if args.no_camb_cltt:
        cl_tt_txt_dir = None
    elif args.cl_tt_txt_dir is not None:
        d = os.path.abspath(args.cl_tt_txt_dir)
        if not _cl_tt_camb_files_ready(d):
            raise FileNotFoundError(
                f"CAMB Cl_TT file not found: {os.path.join(d, CL_TT_TXT_FIDUCIAL)}"
            )
        cl_tt_txt_dir = d
    else:
        cl_tt_txt_dir = script_dir if _cl_tt_camb_files_ready(script_dir) else None
        if cl_tt_txt_dir is None:
            print(
                "Note: CAMB Cl_TT text files not found next to main_3d.py; using analytic Cl_TT. "
                "Run `python planck_cosmology.py` in this directory to generate them.",
                file=sys.stderr,
            )

    as_fid = AS_FID_PLANCK2018 if cl_tt_txt_dir is not None else AS_FID_LEGACY

    if cl_tt_txt_dir is not None:
        report_camb_cl_tt_numerical_derivatives(
            ell,
            ns_fid=ns_fid,
            as_fid_planck=AS_FID_PLANCK2018,
            cl_tt_txt_dir=cl_tt_txt_dir,
        )
        report_camb_vs_analytic_cl_tt_average_difference(
            ell,
            as_fid=as_fid,
            cl_tt_txt_dir=cl_tt_txt_dir,
        )

    if args.save_forecasts:
        if cl_tt_txt_dir is None:
            raise RuntimeError(
                "--save-forecasts requires CAMB Cl_TT text files. "
                "Run without --no-camb-cltt (and ensure cl_tt_fiducial.txt exists), or pass --cl-tt-txt-dir."
            )
        outdir = args.save_dir.resolve()
        p_pixie = outdir / "forecast_pixie_camb_cltt.txt"
        p_specter = outdir / "forecast_specter_camb_cltt.txt"
        write_forecast_table_to_txt(
            p_pixie,
            ell,
            fwhm_deg,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            sigma_ns_planck,
            sigma_as_planck,
            fnl_fiducials,
            cl_tt_txt_dir=cl_tt_txt_dir,
            as_fid=as_fid,
            w_mu_inv=W_MU_INV_PIXIE,
            mu_noise_label="PIXIE",
            b_override=b_override,
        )
        write_forecast_table_to_txt(
            p_specter,
            ell,
            fwhm_deg,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            sigma_ns_planck,
            sigma_as_planck,
            fnl_fiducials,
            cl_tt_txt_dir=cl_tt_txt_dir,
            as_fid=as_fid,
            w_mu_inv=W_MU_INV_SPECTER,
            mu_noise_label="SPECTER",
            b_override=b_override,
        )
        print(f"Wrote {p_pixie}", file=sys.stderr)
        print(f"Wrote {p_specter}", file=sys.stderr)
        return

    if args.fnl_ns_contours:
        # Use a small set of illustrative fiducial f_NL values
        fnl_vals = (0.0, 1.0, 12_500.0, 25_000.0)
        base_dir = ensure_dir(paper_results_dir())
        cltt_tag = "camb" if cl_tt_txt_dir is not None else "analytic"
        for fnl_plot in fnl_vals:
            # Single SPECTER-only contour in #3193A2
            r_spec = fisher_muT_general(
                ell,
                fwhm_deg,
                fnl_plot,
                ns_fid,
                k_D_i,
                k_D_f,
                k_p,
                w_mu_inv=w_mu_inv,
                dns_step=5e-5,
                sigma_ns_prior=sigma_ns_planck,
                sigma_As_prior=sigma_as_planck,
                use_b_analytic=False,
                b_override=b_override,
                As_fid=as_fid,
                cl_tt_txt_dir=cl_tt_txt_dir,
            )
            out_single = base_dir / f"fnl_ns_{cltt_tag}_specter_fnl{int(fnl_plot):d}.pdf"
            plot_fnl_ns_single(
                r_spec,
                fnl_fid=fnl_plot,
                ns_fid=ns_fid,
                color="#3193A2",
                outfile=out_single,
                show=False,
            )

            # Combined PIXIE vs SPECTER contours
            out_both = base_dir / f"fnl_ns_pixie_vs_specter_{cltt_tag}_fnl{int(fnl_plot):d}.pdf"
            plot_fnl_ns_pixie_vs_specter(
                ell,
                fwhm_deg,
                fnl_plot,
                ns_fid,
                k_D_i,
                k_D_f,
                k_p,
                sigma_ns_prior=sigma_ns_planck,
                sigma_As_prior=sigma_as_planck,
                as_fid=as_fid,
                cl_tt_txt_dir=cl_tt_txt_dir,
                b_override=b_override,
                outfile=out_both,
                show=False,
            )
        return

    if args.one_d:
        print_forecast_1d(
            ell,
            fwhm_deg,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            cl_tt_txt_dir=cl_tt_txt_dir,
            as_fid=as_fid,
            w_mu_inv=w_mu_inv,
            mu_noise_label=mu_noise_label,
            b_override=b_override,
        )
        return

    print_forecast_table(
        ell,
        fwhm_deg,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        sigma_ns_planck,
        sigma_as_planck,
        fnl_fiducials,
        cl_tt_txt_dir=cl_tt_txt_dir,
        as_fid=as_fid,
        w_mu_inv=w_mu_inv,
        mu_noise_label=mu_noise_label,
        b_override=b_override,
    )

    if args.no_plot:
        return

    r_plot = fisher_muT_general(
        ell,
        fwhm_deg,
        FNl_FID_FOR_PLOTS,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=w_mu_inv,
        dns_step=5e-5,
        sigma_ns_prior=sigma_ns_planck,
        sigma_As_prior=sigma_as_planck,
        use_b_analytic=False,
        b_override=b_override,
        As_fid=as_fid,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    plot_pairwise_marginal_contours(
        r_plot,
        fnl_fid=FNl_FID_FOR_PLOTS,
        ns_fid=ns_fid,
        as_fid=as_fid,
        outfile=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
