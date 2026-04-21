r"""
\(\sigma(f_{\rm NL})\) vs \(\ell_{\max}\) for many foreground cleaning factors \(c_f\), colored by \(c_f\).

Uses ``fisher_muT_fnl_ns_with_dust`` with \(n_s\) marginalized (prior). Each \(c_f\) produces one
curve; line color maps to \(c_f\) (colorbar). Default uses many \(c_f\) samples so the colormap is smooth.

Outputs are **tagged by** \(f_{\rm NL}^{\rm fid}\) so batch runs do not overwrite:
``sigma_fnl_vs_lmax_cf_{experiment}_fnl{tag}.png``, matching ``.csv`` / ``.txt`` / run ``.json``.

Run::

    python3 plot_sigma_fnl_vs_lmax_cf.py --experiment specter --fnl-fid 1

    python3 plot_sigma_fnl_vs_lmax_cf.py --section4-config path/to/logs/config.json --experiment specter
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER, ell_max_from_fwhm_deg
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from output_paths import ensure_section_layout
from run_section4 import A_D_CODE, K_D_F, K_D_I, K_P, NS_FID
from spectra import AS_FID_PLANCK2018

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

FWHM_PIXIE = 1.6
FWHM_SPECTER = 1.0
# Full Cabass-style scans to 1000 are expensive with numerical b(ell); override with --lmax-cap.
LMAX_MAX = 1000
# Upper end of the \(\ell_{\max}\) axis (log scale; data may extend further; axis clips here).
XLIM_MAX = 700
# Fixed \(\sigma(f_{\mathrm{NL}})\) axis (log scale): bottom, top.
YLIM_SIGMA_FNL = (1.0e3, 1.0e5)


def _fnl_file_tag(fnl: float) -> str:
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _lmax_grid(n: int, lmax_cap: int) -> list[int]:
    cap = min(int(lmax_cap), LMAX_MAX)
    pts = np.unique(np.round(np.linspace(2, cap, n)).astype(int))
    return [int(x) for x in pts if x >= 2]


def _fnl_list_from_args(args: argparse.Namespace) -> list[float]:
    if args.section4_config is not None:
        p = Path(args.section4_config).expanduser().resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        if "fnl_fiducials" not in data:
            raise SystemExit(f"{p}: missing fnl_fiducials")
        return [float(x) for x in data["fnl_fiducials"]]
    if args.fnl_fiducials:
        return [
            float(x.strip())
            for x in str(args.fnl_fiducials).split(",")
            if x.strip()
        ]
    return [float(args.fnl_fid)]


def run_one_fnl(
    *,
    experiment: str,
    fnl_fid: float,
    n_cf: int,
    n_lmax: int,
    lmax_cap: int,
    numerical_b: bool,
    pipeline: str,
) -> None:
    tag = _fnl_file_tag(float(fnl_fid))
    camb = "camb" in pipeline
    cl_tt_dir = str(Path(__file__).resolve().parent) if camb else None
    if camb:
        use_b_analytic = True
    else:
        use_b_analytic = not bool(numerical_b)
    as_fid = AS_FID_PLANCK2018

    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    cfs = np.logspace(2.0, 4.0, int(n_cf))
    lmax_list = _lmax_grid(int(n_lmax), int(lmax_cap))

    dirs = ensure_section_layout("section4_foregrounds", pipeline)
    run_meta = {
        "experiment": experiment,
        "fnl_fid": float(fnl_fid),
        "fnl_file_tag": tag,
        "n_cf": len(cfs),
        "lmax_points": len(lmax_list),
        "pipeline": pipeline,
    }
    (dirs["logs"] / f"plot_sigma_fnl_lmax_cf_{experiment}_fnl{tag}.json").write_text(
        json.dumps(run_meta, indent=2),
        encoding="utf-8",
    )

    rows = ["lmax,c_f,sigma_fnl"]
    sigma_grid = np.full((len(cfs), len(lmax_list)), np.nan, dtype=float)
    for j, lmax in enumerate(lmax_list):
        ell = np.arange(2, lmax + 1, dtype=float)
        if ell.size == 0:
            continue
        for i, c_f in enumerate(cfs):
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
                As_fid=as_fid,
                sigma_ns_prior=0.004,
                sigma_AD_prior=1e12,
                sigma_alpha_prior=1e6,
                use_b_analytic=use_b_analytic,
                cl_tt_txt_dir=cl_tt_dir,
            )
            s = r.sigma["fnl"]
            sigma_grid[i, j] = s
            rows.append(f"{lmax},{c_f:.8g},{s:.12e}")

    table_body = "\n".join(rows) + "\n"
    stem = f"sigma_fnl_vs_lmax_cf_{experiment}_fnl{tag}"
    csv_path = dirs["tables"] / f"{stem}.csv"
    txt_path = dirs["tables"] / f"{stem}.txt"
    csv_path.write_text(table_body, encoding="utf-8")
    txt_path.write_text(table_body, encoding="utf-8")

    apply_plot_params()
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    larr = np.asarray(lmax_list, dtype=float)
    segs = []
    cols = []
    for i, c_f in enumerate(cfs):
        y = sigma_grid[i, :]
        ok = np.isfinite(y)
        if not np.any(ok):
            continue
        pts = np.column_stack([larr[ok], y[ok]])
        segs.append(pts)
        cols.append(math.log10(c_f))
    if segs:
        lc = LineCollection(segs, cmap="plasma", norm=plt.Normalize(min(cols), max(cols)))
        lc.set_array(np.asarray(cols))
        lc.set_linewidth(2.2)
        # Keep Fisher curves behind reference lines (see beam line zorder below).
        lc.set_zorder(1)
        ax.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax)
        cbar.set_label(r"$\log_{10}(c_f)$")
        ax.relim()
        ax.autoscale_view()

    ax.set_xlim(float(larr.min()), float(XLIM_MAX))
    ax.set_xscale("log")
    ax.set_ylim(YLIM_SIGMA_FNL[0], YLIM_SIGMA_FNL[1])
    ax.set_yscale("log")
    ax.tick_params(axis="y", which="major", length=7, width=0.9)
    ax.set_xlabel(r"$\ell_{\rm max}$")
    ax.set_ylabel(r"$\sigma(f_{\rm NL})$")
    ax.set_title(r"$f_{\mathrm{NL}}$ forecast: varying foreground cleaning (SPECTER)")
    lbeam = ell_max_from_fwhm_deg(fwhm)
    _y0, _y1 = float(YLIM_SIGMA_FNL[0]), float(YLIM_SIGMA_FNL[1])
    # High zorder + explicit vertical segments so beam lines render above LineCollection.
    _vlw = 2.0
    _vz = 100
    ax.plot(
        [lbeam, lbeam],
        [_y0, _y1],
        color="black",
        ls="-",
        lw=_vlw,
        zorder=_vz,
        clip_on=False,
        solid_capstyle="round",
        label=rf"$\ell_{{\rm beam}}\approx{lbeam:.0f}$",
    )
    _leg = ax.legend(loc="upper right", fontsize=18, frameon=True)
    _leg.set_zorder(250)
    fig.tight_layout()
    out_png = dirs["figures"] / f"{stem}.png"
    out_pdf = dirs["figures"] / f"{stem}.pdf"
    fig.savefig(out_png, bbox_inches="tight", dpi=200)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(csv_path.resolve(), flush=True)
    print(txt_path.resolve(), flush=True)
    print(out_png.resolve(), flush=True)
    print(out_pdf.resolve(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("pixie", "specter"), default="specter")
    ap.add_argument(
        "--fnl-fid",
        type=float,
        default=1.0,
        help="Single f_NL fiducial if not using --fnl-fiducials / --section4-config.",
    )
    ap.add_argument(
        "--fnl-fiducials",
        type=str,
        default=None,
        help="Comma-separated f_NL list (each run writes separate plot/csv/txt/json).",
    )
    ap.add_argument(
        "--section4-config",
        type=Path,
        default=None,
        help="JSON with fnl_fiducials (e.g. section4 .../logs/config.json).",
    )
    ap.add_argument(
        "--n-cf",
        type=int,
        default=64,
        help="Number of c_f values in log_10 space [1e2,1e4] (more = smoother color gradient).",
    )
    ap.add_argument("--n-lmax", type=int, default=36, help="Number of ell_max samples.")
    ap.add_argument(
        "--lmax-cap",
        type=int,
        default=700,
        help="Cap on ell_max for this plot (<=1000). Lower is much faster with numerical b(ell).",
    )
    ap.add_argument(
        "--numerical-b",
        action="store_true",
        help="Use numerical b(ell,n_s) like run_section4 analytic pipeline (slow at large ell_max×n_cf).",
    )
    ap.add_argument(
        "--pipeline",
        type=str,
        default="analytic_cltt_analytic_b",
        help="Section 4 pipeline tag for output paths (analytic_cltt_analytic_b or camb_cltt_analytic_b).",
    )
    args = ap.parse_args()

    nsrc = sum(
        1
        for x in (args.section4_config, args.fnl_fiducials)
        if (x is not None and str(x).strip())
    )
    if nsrc > 1:
        print(
            "Warning: use only one of --section4-config or --fnl-fiducials; "
            "first match in internal order wins.",
            file=sys.stderr,
        )

    fnl_list = _fnl_list_from_args(args)
    for fnl in fnl_list:
        run_one_fnl(
            experiment=args.experiment,
            fnl_fid=float(fnl),
            n_cf=args.n_cf,
            n_lmax=args.n_lmax,
            lmax_cap=args.lmax_cap,
            numerical_b=args.numerical_b,
            pipeline=args.pipeline,
        )


if __name__ == "__main__":
    main()
