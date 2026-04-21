'Section 4 extension — CosmicFish **triangle** plots for the dust Fisher.'

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from fisher_matrix import default_ell_grid
from run_section4 import (
    A_D_CODE,
    K_D_F,
    K_D_I,
    K_P,
    NS_FID,
)
from spectra import AS_FID_PLANCK2018

try:
    from .cosmicfish_contours import (
        _disable_cosmicfish_fisher_protection,
        _import_cosmicfish,
        _resolve_cosmicfish_python,
        _symmetrize_fisher,
        finish_mu_t_cosmicfish_plot,
        make_cosmicfish_fisher_object,
        mu_t_cosmicfish_plot_style,
        rescale_fisher_ad_scale,
    )
    from .output_paths import ensure_section_layout
    from .plot_params import apply_plot_params
except ImportError:
    from cosmicfish_contours import (
        _disable_cosmicfish_fisher_protection,
        _import_cosmicfish,
        _resolve_cosmicfish_python,
        _symmetrize_fisher,
        finish_mu_t_cosmicfish_plot,
        make_cosmicfish_fisher_object,
        mu_t_cosmicfish_plot_style,
        rescale_fisher_ad_scale,
    )
    from output_paths import ensure_section_layout
    from plot_params import apply_plot_params

FWHM_PIXIE = 1.6
FWHM_SPECTER = 1.0
PIXIE_COLOR = "#3193A2"
SPECTER_COLOR = "#e76f51"
AD_RESCALE = 1e12


def _module_dir():
    return Path(__file__).resolve().parent


def _fnl_file_tag(fnl):
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _cf_file_tag(c_f):
    if math.isfinite(c_f) and abs(c_f - round(c_f)) < 1e-6:
        return str(int(round(c_f)))
    return str(c_f).replace(".", "p")


def _fnl_fiducials_from_args(args):
    if getattr(args, "section4_config", None) is not None:
        p = Path(args.section4_config).expanduser().resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        if "fnl_fiducials" not in data:
            raise SystemExit(f"{p}: missing fnl_fiducials")
        return [float(x) for x in data["fnl_fiducials"]]
    if getattr(args, "fnl_fiducials", None):
        return [
            float(x.strip())
            for x in str(args.fnl_fiducials).split(",")
            if x.strip()
        ]
    if args.fnl_fid is not None:
        return [float(args.fnl_fid)]
    raise SystemExit("Provide --fnl-fid, --fnl-fiducials, or --section4-config")


def _save_triangle_pdf(plotter, out):
    fig = plotter.figure
    leg = getattr(plotter, "legend", None)
    kw: dict = {"bbox_inches": "tight", "pad_inches": 0.45}
    if leg is not None:
        kw["bbox_extra_artists"] = (leg,)
    fig.savefig(str(out), **kw)


def _foreground_result(
    *,
    experiment,
    fnl_fid,
    c_f,
    pipeline,
):
    fwhm = FWHM_PIXIE if experiment == "pixie" else FWHM_SPECTER
    w_inv = W_MU_INV_PIXIE if experiment == "pixie" else W_MU_INV_SPECTER
    ell = default_ell_grid(fwhm)
    camb = pipeline == "camb_cltt_analytic_b"
    cl_tt_dir = str(_module_dir()) if camb else None
    return fisher_muT_fnl_ns_with_dust(
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
        sigma_ns_prior=0.004,
        sigma_AD_prior=1e12,
        sigma_alpha_prior=1e6,
        use_b_analytic=camb,
        cl_tt_txt_dir=cl_tt_dir,
    )


def _cosmicfish_fish(
    fm,
    *,
    experiment,
    fnl_fid,
    c_f,
    pipeline,
    label,
):
    r = _foreground_result(experiment=experiment, fnl_fid=fnl_fid, c_f=c_f, pipeline=pipeline)
    F = _symmetrize_fisher(r.F_total)
    names = list(r.param_names)
    fid_map = {
        "fnl": float(fnl_fid),
        "ns": NS_FID,
        "A_D": A_D_CODE,
        "alpha_D": AZZONI_ALPHA_D,
    }
    fid = [fid_map[n] for n in names]
    F, fid, names = rescale_fisher_ad_scale(F, fid, names, scale=AD_RESCALE)
    return make_cosmicfish_fisher_object(fm, F, list(names), list(fid), None, label)


def main(argv = None):
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
    apply_plot_params()

    ap = argparse.ArgumentParser(
        description="Section 4 CosmicFish 4-parameter triangle (fnl, ns, A_D, alpha_D)."
    )
    ap.add_argument("--cosmicfish-python", type=Path, default=None)
    ap.add_argument(
        "--fnl-fid",
        type=float,
        default=None,
        help=r"Single fiducial $f_{\rm NL}$ (included in each output filename).",
    )
    ap.add_argument(
        "--fnl-fiducials",
        type=str,
        default=None,
        help="Comma-separated f_NL fiducials (each gets its own PDFs; overrides --fnl-fid).",
    )
    ap.add_argument(
        "--section4-config",
        type=Path,
        default=None,
        help="e.g. .../section4_foregrounds/.../logs/config.json — uses fnl_fiducials list.",
    )
    ap.add_argument("--cf", type=float, default=1000.0, help="Foreground cleaning factor c_f.")
    ap.add_argument(
        "--modes",
        type=str,
        default="specter,pixie,overlay",
        help="Comma-separated: specter, pixie, overlay (all three by default).",
    )
    ap.add_argument(
        "--pipeline",
        type=str,
        default="analytic_cltt_analytic_b",
        choices=("analytic_cltt_analytic_b", "camb_cltt_analytic_b"),
    )
    args = ap.parse_args(argv)
    modes = {m.strip().lower() for m in args.modes.split(",") if m.strip()}
    valid = {"specter", "pixie", "overlay"}
    bad = modes - valid
    if bad:
        raise SystemExit(f"Unknown modes {bad}; use {valid}")

    fnl_list = _fnl_fiducials_from_args(args)

    root = _resolve_cosmicfish_python(args.cosmicfish_python)
    fm, fp, fpa = _import_cosmicfish(root)
    _disable_cosmicfish_fisher_protection(fm)

    dirs = ensure_section_layout("section4_foregrounds", args.pipeline)
    cf_tag = _cf_file_tag(float(args.cf))
    c_f = float(args.cf)
    pipe = args.pipeline

    for fnl in fnl_list:
        fnl_f = float(fnl)
        fnl_tag = _fnl_file_tag(fnl_f)

        if "specter" in modes:
            fish = _cosmicfish_fish(
                fm, experiment="specter", fnl_fid=fnl_f, c_f=c_f, pipeline=pipe, label="SPECTER"
            )
            analysis = fpa.CosmicFish_FisherAnalysis(fisher_list=[fish])
            _st = mu_t_cosmicfish_plot_style(["SPECTER"])
            plotter = fp.CosmicFishPlotter(
                fishers=analysis,
                solid_colors=[SPECTER_COLOR],
                line_colors=[SPECTER_COLOR],
                labels=["SPECTER"],
                legend_ncol=1,
                **_st,
            )
            plotter.new_plot()
            plotter.plot_tri(
                title=rf"Section\ 4\ dust\ Fisher,\ SPECTER,\ c_f={c_f:g},\ f_{{\rm NL}}^{{\rm fid}}={fnl_f:g}",
                tight_layout=False,
            )
            finish_mu_t_cosmicfish_plot(
                plotter, dust_alpha_d=True, nudge_triangle_legend=True
            )
            out = dirs["figures"] / f"cosmicfish_triangle_4d_section4_specter_fnl{fnl_tag}_cf{cf_tag}.pdf"
            _save_triangle_pdf(plotter, out)
            print(out.resolve(), flush=True)

        if "pixie" in modes:
            fish = _cosmicfish_fish(
                fm, experiment="pixie", fnl_fid=fnl_f, c_f=c_f, pipeline=pipe, label="PIXIE"
            )
            analysis = fpa.CosmicFish_FisherAnalysis(fisher_list=[fish])
            _st = mu_t_cosmicfish_plot_style(["PIXIE"])
            plotter = fp.CosmicFishPlotter(
                fishers=analysis,
                solid_colors=[PIXIE_COLOR],
                line_colors=[PIXIE_COLOR],
                labels=["PIXIE"],
                legend_ncol=1,
                **_st,
            )
            plotter.new_plot()
            plotter.plot_tri(
                title=rf"Section\ 4\ dust\ Fisher,\ PIXIE,\ c_f={c_f:g},\ f_{{\rm NL}}^{{\rm fid}}={fnl_f:g}",
                tight_layout=False,
            )
            finish_mu_t_cosmicfish_plot(
                plotter, dust_alpha_d=True, nudge_triangle_legend=True
            )
            out = dirs["figures"] / f"cosmicfish_triangle_4d_section4_pixie_fnl{fnl_tag}_cf{cf_tag}.pdf"
            _save_triangle_pdf(plotter, out)
            print(out.resolve(), flush=True)

        if "overlay" in modes:
            fish_pix = _cosmicfish_fish(
                fm, experiment="pixie", fnl_fid=fnl_f, c_f=c_f, pipeline=pipe, label="PIXIE"
            )
            fish_sp = _cosmicfish_fish(
                fm, experiment="specter", fnl_fid=fnl_f, c_f=c_f, pipeline=pipe, label="SPECTER"
            )
            analysis = fpa.CosmicFish_FisherAnalysis(fisher_list=[fish_pix, fish_sp])
            _st = mu_t_cosmicfish_plot_style(["PIXIE", "SPECTER"])
            plotter = fp.CosmicFishPlotter(
                fishers=analysis,
                solid_colors=[PIXIE_COLOR, SPECTER_COLOR],
                line_colors=[PIXIE_COLOR, SPECTER_COLOR],
                labels=["PIXIE", "SPECTER"],
                legend_ncol=2,
                **_st,
            )
            plotter.new_plot()
            plotter.plot_tri(
                title=rf"Section\ 4\ dust\ Fisher,\ PIXIE\ vs.\ SPECTER,\ c_f={c_f:g},\ f_{{\rm NL}}^{{\rm fid}}={fnl_f:g}",
                tight_layout=False,
            )
            finish_mu_t_cosmicfish_plot(
                plotter, dust_alpha_d=True, nudge_triangle_legend=True
            )
            out = (
                dirs["figures"]
                / f"cosmicfish_triangle_4d_section4_pixie_specter_fnl{fnl_tag}_cf{cf_tag}.pdf"
            )
            _save_triangle_pdf(plotter, out)
            print(out.resolve(), flush=True)


if __name__ == "__main__":
    main()
