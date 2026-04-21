'2D marginalized \\((f_{\\mathrm{NL}}, A_D)\\) and \\((f_{\\mathrm{NL}}, \\alpha_D)\\) for the section-4 dust model.'

from __future__ import annotations

import argparse
import os

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from config_section4 import A_D_CODE, SIGMA_AD_PRIOR, SIGMA_ALPHA_PRIOR, SIGMA_NS_PRIOR
from config_section_common import FWHM_PIXIE, FWHM_SPECTER, K_D_F, K_D_I, K_P, NS_FID
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from fisher_matrix import default_ell_grid
from output_paths import ensure_section_layout
from spectra import AS_FID_PLANCK2018

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

# Joint 68% / 95% for 2 Gaussian parameters (same as main_3d.py / contours.py)
DELTA_CHI2_LEVELS_2D = (2.30, 5.99, 9.21)


def _grid_extent_from_2d_cov(
    cov_2d,
    extent_sigma = 3.0,
):
    r"""Estimate plotting extents from a 2x2 covariance block."""
    evals = np.linalg.eigvals(cov_2d)
    evals = np.abs(evals)  # Ensure non-negative
    if np.any(evals <= 0):
        sig_x, sig_y = np.sqrt(np.diag(cov_2d))
        return extent_sigma * sig_x, extent_sigma * sig_y
    
    sig_x = np.sqrt(evals[0])
    sig_y = np.sqrt(evals[1])
    return extent_sigma * sig_x, extent_sigma * sig_y


def marginal_delta_chi2_2d(
    cov_full,
    i,
    j,
    x,
    y,
    x0,
    y0,
):
    block = cov_full[np.ix_([i, j], [i, j])]
    prec = np.linalg.inv(block)
    X, Y = np.meshgrid(x, y, indexing="xy")
    d0 = X - x0
    d1 = Y - y0
    return prec[0, 0] * d0**2 + 2.0 * prec[0, 1] * d0 * d1 + prec[1, 1] * d1**2


def _chi2_grid(
    cov_full,
    i,
    j,
    x0,
    y0,
    *,
    n_grid = 200,
    sigma_extent = 3.0,
):
    """Build a parameter grid and its marginalized delta-chi-squared values."""
    cov_block = cov_full[np.ix_([i, j], [i, j])]
    sx, sy = _grid_extent_from_2d_cov(cov_block, extent_sigma=sigma_extent)
    gx = np.linspace(x0 - sx, x0 + sx, n_grid)
    gy = np.linspace(y0 - sy, y0 + sy, n_grid)
    chi2 = marginal_delta_chi2_2d(cov_full, i, j, gx, gy, x0, y0)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    return X, Y, chi2


def _plot_marginal_chi2_panel(
    ax,
    cov,
    i,
    j,
    x0,
    y0,
    *,
    xlabel,
    ylabel,
    title,
    color,
    n_grid = 200,
    sigma_extent = 3.0,
):
    X, Y, chi2 = _chi2_grid(cov, i, j, x0, y0, n_grid=n_grid, sigma_extent=sigma_extent)
    lev1, lev2 = DELTA_CHI2_LEVELS_2D[0], DELTA_CHI2_LEVELS_2D[1]
    ax.contourf(X, Y, chi2, levels=[0.0, lev1], colors=[color], alpha=0.25)
    ax.contour(
        X,
        Y,
        chi2,
        levels=[lev1, lev2],
        colors=[color, color],
        linewidths=1.4,
        linestyles=["solid", "dotted"],
    )
    ax.axvline(x0, color="0.2", ls="--", lw=1.0, alpha=0.35)
    ax.axhline(y0, color="0.2", ls="--", lw=1.0, alpha=0.35)
    ax.plot(x0, y0, "k+", ms=10, mew=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.35, linestyle=":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("pixie", "specter"), default="specter")
    ap.add_argument("--cf", type=float, default=1000.0)
    ap.add_argument("--fnl-fid", type=float, default=1.0)
    ap.add_argument(
        "--pipeline",
        type=str,
        default="analytic_cltt_analytic_b",
        choices=("analytic_cltt_analytic_b", "camb_cltt_analytic_b"),
    )
    args = ap.parse_args()

    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if args.experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    ell = default_ell_grid(fwhm)
    camb = args.pipeline == "camb_cltt_analytic_b"
    cl_tt_dir = os.path.dirname(os.path.abspath(__file__)) if camb else None

    r = fisher_muT_fnl_ns_with_dust(
        ell,
        fwhm,
        args.fnl_fid,
        NS_FID,
        K_D_I,
        K_D_F,
        K_P,
        w_mu_inv=w_inv,
        c_f=float(args.cf),
        A_D=A_D_CODE,
        alpha_D=AZZONI_ALPHA_D,
        ell0=ELL0_AZZONI,
        As_fid=AS_FID_PLANCK2018,
        sigma_ns_prior=SIGMA_NS_PRIOR,
        sigma_AD_prior=SIGMA_AD_PRIOR,
        sigma_alpha_prior=SIGMA_ALPHA_PRIOR,
        use_b_analytic=camb,
        cl_tt_txt_dir=cl_tt_dir,
    )
    ic = {n: j for j, n in enumerate(r.param_names)}
    cov = r.cov
    dirs = ensure_section_layout("section4_foregrounds", args.pipeline)
    apply_plot_params()
    import matplotlib.pyplot as plt

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.8))
    col0, col1 = "#3193A2", "#e76f51"

    _plot_marginal_chi2_panel(
        ax0,
        cov,
        ic["fnl"],
        ic["A_D"],
        args.fnl_fid,
        A_D_CODE,
        xlabel=r"$f_{\mathrm{NL}}$",
        ylabel=r"$A_D$",
        title=rf"marg.\ $\Delta\chi^2$ $(f_{{\rm NL}},A_D)$ @ $c_f={args.cf:g}$",
        color=col0,
    )
    _plot_marginal_chi2_panel(
        ax1,
        cov,
        ic["fnl"],
        ic["alpha_D"],
        args.fnl_fid,
        AZZONI_ALPHA_D,
        xlabel=r"$f_{\mathrm{NL}}$",
        ylabel=r"$\alpha_D$",
        title=rf"marg.\ $\Delta\chi^2$ $(f_{{\rm NL}},\alpha_D)$ @ $c_f={args.cf:g}$",
        color=col1,
    )

    fig.suptitle(
        rf"{args.experiment} — {args.pipeline} ($f_{{\rm NL}}^{{\rm fid}}={args.fnl_fid:g}$); "
        r"shaded: $\Delta\chi^2<2.30$ (1$\sigma$); solid/dotted: 1$\sigma$/2$\sigma$ contours"
    )
    fig.tight_layout()
    out = dirs["figures"] / f"fnl_dust_contours_{args.experiment}_cf{int(args.cf)}_fnl{int(args.fnl_fid)}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
