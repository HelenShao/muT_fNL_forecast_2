'Overlay \\((f_{\\mathrm{NL}}, n_s)\\) Gaussian Fisher contours: residual dust marginalized at fixed.'

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


import argparse
import json
import math
import os

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from config_section4 import A_D_CODE, SIGMA_AD_PRIOR, SIGMA_ALPHA_PRIOR, SIGMA_NS_PRIOR
from config_section_common import FWHM_PIXIE, FWHM_SPECTER, K_D_F, K_D_I, K_P, NS_FID
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from fisher_matrix import SIGMA_AS_PLANCK2018, default_ell_grid, fisher_muT_general
from output_paths import ensure_section_layout
from plots.plot_fnl_dust_contours import DELTA_CHI2_LEVELS_2D, marginal_delta_chi2_2d, _grid_extent_from_2d_cov
from spectra import AS_FID_PLANCK2018

from plot_params import apply_plot_params

COL_DUST_MARG = "#5A4FB6"
COL_NO_DUST = "#C7873C"


def _fnl_file_tag(fnl):
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _cf_file_tag(c_f):
    if math.isfinite(c_f) and abs(c_f - round(c_f)) < 1e-6:
        return str(int(round(c_f)))
    return str(c_f).replace(".", "p")


def _fnl_list(args):
    if args.section4_config is not None:
        p = Path(args.section4_config).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(
                f"Config JSON not found: {p}\n"
                "Use the real path to config.json (not '.../config.json')."
            )
        data = json.loads(p.read_text(encoding="utf-8"))
        return [float(x) for x in data["fnl_fiducials"]]
    if args.fnl_fiducials:
        return [float(x.strip()) for x in args.fnl_fiducials.split(",") if x.strip()]
    return [float(args.fnl_fid)]


def _cov_fnl_ns_no_dust(
    *,
    experiment,
    fnl_fid,
    use_b_analytic,
    cl_tt_txt_dir,
):
    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    ell = default_ell_grid(fwhm)
    r = fisher_muT_general(
        ell,
        fwhm,
        float(fnl_fid),
        NS_FID,
        K_D_I,
        K_D_F,
        K_P,
        w_mu_inv=w_inv,
        dns_step=5e-5,
        sigma_ns_prior=0.004,
        sigma_As_prior=SIGMA_AS_PLANCK2018,
        use_b_analytic=use_b_analytic,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    idx = {n: i for i, n in enumerate(r.param_names)}
    i0, i1 = idx["fnl"], idx["ns"]
    return np.asarray(r.cov_marginal[np.ix_([i0, i1], [i0, i1])], dtype=float)


def _cov_fnl_ns_dust_marg(
    *,
    experiment,
    fnl_fid,
    c_f,
    use_b_analytic,
    cl_tt_txt_dir,
):
    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    ell = default_ell_grid(fwhm)
    r = fisher_muT_fnl_ns_with_dust(
        ell,
        fwhm,
        float(fnl_fid),
        NS_FID,
        K_D_I,
        K_D_F,
        K_P,
        w_mu_inv=w_inv,
        c_f=float(c_f),
        A_D=A_D_CODE,
        alpha_D=AZZONI_ALPHA_D,
        ell0=ELL0_AZZONI,
        As_fid=AS_FID_PLANCK2018,
        dns_step=5e-5,
        sigma_ns_prior=SIGMA_NS_PRIOR,
        sigma_AD_prior=SIGMA_AD_PRIOR,
        sigma_alpha_prior=SIGMA_ALPHA_PRIOR,
        use_b_analytic=use_b_analytic,
        cl_tt_txt_dir=cl_tt_txt_dir,
        marginalize_dust=True,
    )
    return np.asarray(r.cov[np.ix_([0, 1], [0, 1])], dtype=float)


def _combined_xy_grid(
    covs,
    x0,
    y0,
    *,
    n_grid,
    sigma_extent,
):
    sx = sy = 0.0
    for c in covs:
        ex, ey = _grid_extent_from_2d_cov(c, extent_sigma=sigma_extent)
        sx = max(sx, ex)
        sy = max(sy, ey)
    gx = np.linspace(x0 - sx, x0 + sx, n_grid)
    gy = np.linspace(y0 - sy, y0 + sy, n_grid)
    X, Y = np.meshgrid(gx, gy, indexing="xy")
    return X, Y, gx, gy


def _plot_overlay_panel(
    ax,
    *,
    cov_no_dust,
    cov_dust_marg,
    fnl_fid,
    ns_fid,
    experiment_label,
    n_grid,
    sigma_extent,
):
    X, Y, gx, gy = _combined_xy_grid(
        [cov_no_dust, cov_dust_marg],
        fnl_fid,
        ns_fid,
        n_grid=n_grid,
        sigma_extent=sigma_extent,
    )
    chi2_nd = marginal_delta_chi2_2d(cov_no_dust, 0, 1, gx, gy, fnl_fid, ns_fid)
    chi2_d = marginal_delta_chi2_2d(cov_dust_marg, 0, 1, gx, gy, fnl_fid, ns_fid)
    lev68, lev95 = DELTA_CHI2_LEVELS_2D[0], DELTA_CHI2_LEVELS_2D[1]

    # Purple (dust) underneath; gold (no dust) on top for fills and 95% outlines.
    ax.contourf(
        X,
        Y,
        chi2_d,
        levels=[0.0, lev68],
        colors=[COL_DUST_MARG],
        alpha=0.7,
        zorder=1,
    )
    ax.contourf(
        X,
        Y,
        chi2_nd,
        levels=[0.0, lev68],
        colors=[COL_NO_DUST],
        alpha=0.7,
        zorder=2,
    )
    ax.contour(
        X,
        Y,
        chi2_d,
        levels=[lev95],
        colors=[COL_DUST_MARG],
        linestyles=["--"],
        linewidths=1.6,
        zorder=3,
    )
    ax.contour(
        X,
        Y,
        chi2_nd,
        levels=[lev95],
        colors=[COL_NO_DUST],
        linestyles=["--"],
        linewidths=1.6,
        zorder=4,
    )

    ax.axvline(fnl_fid, color="0.2", ls=":", lw=1.0, alpha=0.35)
    ax.axhline(ns_fid, color="0.2", ls=":", lw=1.0, alpha=0.35)
    ax.plot(fnl_fid, ns_fid, "k+", ms=10, mew=1.2, zorder=5)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(experiment_label)
    ax.grid(True, alpha=0.35, linestyle=":")


def _run_one_fnl(
    *,
    fnl_fid,
    c_f,
    pipeline,
    n_grid,
    sigma_extent,
):
    camb = pipeline == "camb_cltt_analytic_b"
    cl_tt_txt_dir = str(Path(__file__).resolve().parent.parent) if camb else None

    cov_p_nd = _cov_fnl_ns_no_dust(
        experiment="pixie",
        fnl_fid=fnl_fid,
        use_b_analytic=camb,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    cov_p_d = _cov_fnl_ns_dust_marg(
        experiment="pixie",
        fnl_fid=fnl_fid,
        c_f=c_f,
        use_b_analytic=camb,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    cov_s_nd = _cov_fnl_ns_no_dust(
        experiment="specter",
        fnl_fid=fnl_fid,
        use_b_analytic=camb,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )
    cov_s_d = _cov_fnl_ns_dust_marg(
        experiment="specter",
        fnl_fid=fnl_fid,
        c_f=c_f,
        use_b_analytic=camb,
        cl_tt_txt_dir=cl_tt_txt_dir,
    )

    apply_plot_params()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.0, 5.2))
    _plot_overlay_panel(
        ax0,
        cov_no_dust=cov_p_nd,
        cov_dust_marg=cov_p_d,
        fnl_fid=fnl_fid,
        ns_fid=NS_FID,
        experiment_label="PIXIE",
        n_grid=n_grid,
        sigma_extent=sigma_extent,
    )
    _plot_overlay_panel(
        ax1,
        cov_no_dust=cov_s_nd,
        cov_dust_marg=cov_s_d,
        fnl_fid=fnl_fid,
        ns_fid=NS_FID,
        experiment_label="SPECTER",
        n_grid=n_grid,
        sigma_extent=sigma_extent,
    )

    legend_elems = [
        Patch(facecolor=COL_DUST_MARG, edgecolor=COL_DUST_MARG, alpha=0.7, label=rf"Dust marginalized ($c_f={c_f:g}$)"),
        Patch(facecolor=COL_NO_DUST, edgecolor=COL_NO_DUST, alpha=0.7, label=r"No dust"),
    ]
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=2,
        frameon=True,
        fontsize=16,
        bbox_to_anchor=(0.5, -0.02),
    )

    fnl_tag = _fnl_file_tag(fnl_fid)
    cf_tag = _cf_file_tag(c_f)
    dirs = ensure_section_layout("section4_foregrounds", pipeline)
    out = dirs["figures"] / f"fnl_ns_dust_marg_overlay_{pipeline}_fnl{fnl_tag}_cf{cf_tag}.png"
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(out.resolve(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnl-fid", type=float, default=1.0)
    ap.add_argument(
        "--fnl-fiducials",
        type=str,
        default="",
        help="Comma-separated list (overrides --fnl-fid when set).",
    )
    ap.add_argument("--section4-config", type=str, default=None)
    ap.add_argument("--cf", type=float, default=1000.0, help="Cleaning factor for dust-marginalized model.")
    ap.add_argument(
        "--pipeline",
        type=str,
        default="analytic_cltt_analytic_b",
        choices=("analytic_cltt_analytic_b", "camb_cltt_analytic_b"),
    )
    ap.add_argument("--n-grid", type=int, default=200)
    ap.add_argument("--sigma-extent", type=float, default=3.5, help="Extent in σ for grid (from larger ellipse).")
    args = ap.parse_args()

    for fnl in _fnl_list(args):
        _run_one_fnl(
            fnl_fid=fnl,
            c_f=float(args.cf),
            pipeline=args.pipeline,
            n_grid=int(args.n_grid),
            sigma_extent=float(args.sigma_extent),
        )


if __name__ == "__main__":
    main()
