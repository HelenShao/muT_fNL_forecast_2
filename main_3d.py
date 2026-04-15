r"""
3-parameter Fisher forecast: (f_NL, n_s, A_s) with Planck-style priors on n_s and A_s,
plus pairwise marginal \Delta\chi^2 contours (Gaussian Fisher approximation).

Run from this directory:
    python3 main_3d.py

Table only (no matplotlib window):
    python3 main_3d.py --no-plot

To save printed output:
    python3 main_3d.py > forecast_3d.txt

Contour figure is saved as ``main_3d_contours.pdf`` (override with ``--output``). Use
``--no-show`` to save without opening a GUI (e.g. batch runs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fisher_matrix import (
    AS_FID_LEGACY,
    FisherMuTResult,
    SIGMA_AS_PLANCK2018,
    default_ell_grid,
    fisher_muT_general,
)

# Joint 68% / 95% / 99% \Delta\chi^2 for 2 Gaussian parameters
DELTA_CHI2_LEVELS_2D = (2.30, 5.99, 9.21)

# Fiducial f_NL used for contour plots (same spirit as contours.py)
FNl_FID_FOR_PLOTS = 25_000.0

# Indices in cov_marginal / param_names == ("fnl", "ns", "As")
I_FNL, I_NS, I_AS = 0, 1, 2


def marginal_corr(cov: np.ndarray, i: int, j: int) -> float:
    """Pearson correlation from covariance block, matching ``corr_fnl_ns`` for (0, 1)."""
    if cov[i, i] <= 0 or cov[j, j] <= 0:
        return 0.0
    return float(cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]))


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
    \Delta\chi^2 for marginalized 2D Gaussian of parameters (i, j), offset from (x0, y0).

    Uses the 2x2 block of the full covariance (inverse Fisher); equivalent to
    integrating the third parameter out of the Gaussian posterior.
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


def plot_pairwise_marginal_contours(
    r: FisherMuTResult,
    *,
    fnl_fid: float,
    ns_fid: float,
    as_fid: float,
    n_grid: int = 200,
    sigma_extent: float = 3.0,
    outfile: Path | None = None,
    show: bool = True,
) -> None:
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


def print_forecast_table(
    ell: np.ndarray,
    fwhm_deg: float,
    ns_fid: float,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    sigma_ns_planck: float,
    sigma_as_planck: float,
    fnl_fiducials: tuple[float, ...],
) -> None:
    print("muT Fisher forecast -- 3 parameters (f_NL, n_s, A_s)")
    print(f"  l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg}^\\circ, n_s = {ns_fid}")
    print(f"  A_s fid (Delta_R^2 at pivot): {AS_FID_LEGACY:.6e}")
    print(f"  Prior sigma(n_s) = {sigma_ns_planck}")
    print(f"  Prior sigma(A_s) = {sigma_as_planck:.6e} (Planck 2018 order-of-magnitude; see spectra.py)")
    print("  numerical b(l, n_s), dns_step = 5e-5")
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
            dns_step=5e-5,
            sigma_ns_prior=sigma_ns_planck,
            sigma_As_prior=sigma_as_planck,
            use_b_analytic=False,
        )
        assert r.param_names == ("fnl", "ns", "As")
        c = r.cov_marginal
        rho_fnl_as = marginal_corr(c, I_FNL, I_AS)
        print(
            f"{fnl:12.0f}  {r.sigma_fnl_marg:14.6e}  {r.sigma_ns_marg:14.6e}  "
            f"{r.sigma_As_marg:14.6e}  {r.corr_fnl_ns:14.4f}  {rho_fnl_as:14.4f}"
        )

    print()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="3D muT Fisher table and pairwise marginal contours.")
    p.add_argument(
        "--no-plot",
        action="store_true",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "main_3d_contours.pdf",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
    )
    args = p.parse_args(argv)

    fwhm_deg = 1.6
    ns_fid = 0.965
    k_p = 0.002
    k_D_i = 1.1e4
    k_D_f = 46.0

    ell = default_ell_grid(fwhm_deg)
    sigma_ns_planck = 0.004
    sigma_as_planck = SIGMA_AS_PLANCK2018
    fnl_fiducials = (0.0, 1.0, 12_500.0, 25_000.0)

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
        dns_step=5e-5,
        sigma_ns_prior=sigma_ns_planck,
        sigma_As_prior=sigma_as_planck,
        use_b_analytic=False,
    )
    plot_pairwise_marginal_contours(
        r_plot,
        fnl_fid=FNl_FID_FOR_PLOTS,
        ns_fid=ns_fid,
        as_fid=AS_FID_LEGACY,
        outfile=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
