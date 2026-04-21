'2D \\((f_{\\mathrm{NL}}, n_s)\\) Fisher ellipses for several ``c_f`` values (foreground cleaning).'

from __future__ import annotations

import argparse

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from fisher_matrix import default_ell_grid
from output_paths import ensure_section_layout

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

from run_section4 import A_D_CODE, K_D_F, K_D_I, K_P, NS_FID

FWHM_PIXIE = 1.6
FWHM_SPECTER = 1.0

# 68% joint ellipse for 2D Gaussian: (θ - θ_fid)^T Σ^{-1} (θ - θ_fid) = Δχ²
# with Σ the marginalized (f_NL, n_s) covariance. Δχ² ≈ 2.30 for ~68.3% (1/0.434).
CHI2_LEVEL = 1.0 / 0.434


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("pixie", "specter"), default="specter")
    ap.add_argument("--fnl-fid", type=float, default=25000.0, help=r"Fiducial $f_{\rm NL}$ for ellipse center.")
    args = ap.parse_args()
    fnl_fid = float(args.fnl_fid)
    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if args.experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    ell = default_ell_grid(fwhm)
    dirs = ensure_section_layout("section4_foregrounds", "analytic_cltt_analytic_b")

    apply_plot_params()
    import matplotlib.pyplot as plt

    cfs = np.logspace(2.0, 4.0, 6)
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = plt.get_cmap("viridis")

    cov_rows: list[tuple[float, np.ndarray]] = []
    for c_f in cfs:
        r = fisher_muT_fnl_ns_with_dust(
            ell,
            fwhm,
            fnl_fid,
            NS_FID,
            K_D_I,
            K_D_F,
            K_P,
            w_mu_inv=w_inv,
            c_f=float(c_f),
            A_D=A_D_CODE,
            alpha_D=AZZONI_ALPHA_D,
            ell0=ELL0_AZZONI,
            sigma_ns_prior=0.004,
            sigma_AD_prior=1e12,
            sigma_alpha_prior=1e6,
            use_b_analytic=False,
        )
        ic = {n: j for j, n in enumerate(r.param_names)}
        cov2 = r.cov[np.ix_([ic["fnl"], ic["ns"]], [ic["fnl"], ic["ns"]])]
        cov_rows.append((float(c_f), cov2))

    # Axis limits: for x^T Σ^{-1} x = CHI2_LEVEL, semi-axes along eigenvectors v_j of Σ have length
    # sqrt(CHI2_LEVEL * λ_j) with Σ v_j = λ_j v_j. Pad symmetrically about (f_NL^fid, n_s^fid).
    fnl_pad = 0.0
    ns_pad = 0.0
    for _, cov2 in cov_rows:
        w_e, v_e = np.linalg.eigh(cov2)
        w_e = np.maximum(w_e, 1e-30)
        for j in range(2):
            t_ax = float(np.sqrt(CHI2_LEVEL * w_e[j]))
            fnl_pad = max(fnl_pad, abs(t_ax * v_e[0, j]))
            ns_pad = max(ns_pad, abs(t_ax * v_e[1, j]))

    n_grid = 400
    fnl_vals = np.linspace(fnl_fid - 1.2 * fnl_pad, fnl_fid + 1.2 * fnl_pad, n_grid)
    ns_vals = np.linspace(NS_FID - 1.2 * ns_pad, NS_FID + 1.2 * ns_pad, n_grid)
    FN, NS = np.meshgrid(fnl_vals, ns_vals)
    X = FN - fnl_fid
    Y = NS - NS_FID

    print(
        f"\n=== Ellipse coefficients ({args.experiment}) ===\n"
        f"f_NL^fid = {fnl_fid:g},  n_s^fid = {NS_FID:g},  "
        f"Δχ² (68% joint 2D) = {CHI2_LEVEL:.12g}\n"
        f"X = f_NL - f_NL^fid,  Y = n_s - n_s^fid\n"
        f"Contour:  a*X^2 + b*X*Y + c*Y^2 = Δχ²  "
        f"with (a,b,c) = (Σ^-1_11, 2Σ^-1_12, Σ^-1_22)\n"
    )
    for i, (c_f, cov2) in enumerate(cov_rows):
        prec2 = np.linalg.inv(cov2)
        a, b_xy, c = prec2[0, 0], 2.0 * prec2[0, 1], prec2[1, 1]
        print(f"--- c_f = {c_f:g} ---")
        print(f"  Σ (marginal cov fnl,ns):\n{cov2}")
        print(f"  Σ^-1 (precision):\n{prec2}")
        print(
            f"  a = {a:.12e}\n"
            f"  b = {b_xy:.12e}  (coefficient of X*Y; 2*Σ^-1_12)\n"
            f"  c = {c:.12e}\n"
            f"  check: Z = a*X^2 + b*X*Y + c*Y^2  →  level = {CHI2_LEVEL:.12g}"
        )
        Z = prec2[0, 0] * X**2 + 2.0 * prec2[0, 1] * X * Y + prec2[1, 1] * Y**2
        color = cmap(i / max(len(cfs) - 1, 1))
        ax.contour(
            FN,
            NS,
            Z,
            levels=[CHI2_LEVEL],
            colors=[color],
            linewidths=1.5,
        )

    ax.axhline(NS_FID, color="0.5", ls=":")
    ax.axvline(fnl_fid, color="0.5", ls=":")
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    title_exp = "PIXIE" if args.experiment == "pixie" else "SPECTER"
    ax.set_title(title_exp)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=cfs.min(), vmax=cfs.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=r"$c_f$")
    fig.tight_layout()
    out_pdf = dirs["figures"] / f"fnl_ns_cf_gradient_{args.experiment}.pdf"
    out_png = dirs["figures"] / f"fnl_ns_cf_gradient_{args.experiment}.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(out_pdf.resolve())
    print(out_png.resolve())


if __name__ == "__main__":
    main()
