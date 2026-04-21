r"""
§4.b — \(\sigma(f_{\rm NL})\) vs foreground cleaning factor \(c_f\) (log-spaced grid).

Uses ``fisher_muT_fnl_ns_with_dust`` on ``default_ell_grid(fwhm)`` (same footprint as other §4 plots).
Writes per-experiment PDF/CSV/TXT tagged by \(f_{\rm NL}^{\rm fid}\).

Run::

    python3 plot_sigma_fnl_vs_cf.py --fnl-fid 25000

    python3 plot_sigma_fnl_vs_cf.py --section4-config .../logs/config.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from fisher_matrix import SIGMA_AS_PLANCK2018, default_ell_grid, fisher_muT_general
from output_paths import ensure_section_layout

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

from run_section4 import A_D_CODE, K_D_F, K_D_I, K_P, NS_FID
from spectra import AS_FID_PLANCK2018

FWHM_PIXIE = 1.6
FWHM_SPECTER = 1.0


def _fnl_file_tag(fnl: float) -> str:
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _fnl_list_from_args(args: argparse.Namespace) -> list[float]:
    if args.section4_config is not None:
        p = Path(args.section4_config).expanduser().resolve()
        if not p.is_file():
            raise SystemExit(
                f"Config JSON not found: {p}\n"
                "Use the real path to config.json under section4_foregrounds/.../logs/ "
                "(not a literal '.../config.json' placeholder)."
            )
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


def _sigma_section3_no_dust(*, experiment: str, fnl_fid: float) -> float:
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
        use_b_analytic=False,
    )
    return float(r.sigma_fnl_marg)


def run_one_fnl(
    *,
    fnl_fid: float,
    n_cf: int,
    pipeline: str,
    compare_section3: bool,
) -> None:
    tag = _fnl_file_tag(float(fnl_fid))
    dirs = ensure_section_layout("section4_foregrounds", pipeline)
    cfs = np.logspace(2.0, 4.0, int(n_cf))

    rows = ["experiment,c_f,sigma_fnl_marg_dust"]
    if compare_section3:
        rows[0] += ",sigma_fnl_section3_no_dust"
    series: dict[str, list[float]] = {"pixie": [], "specter": []}
    s3_cache: dict[str, float] = {}
    for exp in ("pixie", "specter"):
        fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if exp == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
        ell = default_ell_grid(fwhm)
        if compare_section3:
            s3_cache[exp] = _sigma_section3_no_dust(experiment=exp, fnl_fid=float(fnl_fid))
        for c_f in cfs:
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
                sigma_ns_prior=0.004,
                sigma_AD_prior=1e12,
                sigma_alpha_prior=1e6,
                use_b_analytic=False,
                cl_tt_txt_dir=None,
            )
            sig = r.sigma["fnl"]
            series[exp].append(sig)
            if compare_section3:
                rows.append(f"{exp},{c_f:.8g},{sig:.12e},{s3_cache[exp]:.12e}")
            else:
                rows.append(f"{exp},{c_f:.8g},{sig:.12e}")

    table = "\n".join(rows) + "\n"
    stem = f"sigma_fnl_vs_cf_fnl{tag}"
    (dirs["tables"] / f"{stem}.csv").write_text(table, encoding="utf-8")
    (dirs["tables"] / f"{stem}.txt").write_text(table, encoding="utf-8")
    meta = {"fnl_fid": float(fnl_fid), "n_cf": int(n_cf), "pipeline": pipeline}
    (dirs["logs"] / f"plot_sigma_fnl_vs_cf_fnl{tag}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    apply_plot_params()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for exp, color in (
        ("pixie", "#3193A2"),
        ("specter", "#e76f51"),
    ):
        ax.plot(
            cfs,
            series[exp],
            lw=1.5,
            color=color,
            label=exp.upper(),
        )

    if compare_section3:
        y_p = s3_cache["pixie"]
        y_s = s3_cache["specter"]
        ax.axhline(
            y_p,
            color="#3193A2",
            ls=":",
            alpha=0.85,
        )
        ax.axhline(
            y_s,
            color="#e76f51",
            ls=":",
            alpha=0.85,
        )
        # Reference values just above each §3 line (log-y: multiplicative offset), at fixed x.
        _x_ann = 330.0
        ax.text(
            _x_ann,
            y_p * 0.65,
            rf"$\sigma(f_{{\rm NL}}) = {y_p:.4g}$ (PIXIE, no dust)",
            color="#3193A2",
            fontsize=16,
            ha="center",
            va="bottom",
        )
        ax.text(
            _x_ann,
            y_s * 1.06,
            rf"$\sigma(f_{{\rm NL}}) = {y_s:.4g}$ (SPECTER, no dust)",
            color="#e76f51",
            fontsize=16,
            ha="center",
            va="bottom",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$c_f$ (foreground cleaning)", fontsize=20)
    ax.set_ylabel(r"$\sigma(f_{\rm NL})$", fontsize=20)
    ax.set_title(rf"$\sigma(f_{{\rm NL}})$ vs $c_f$, $f_{{\rm NL}}^{{\rm fid}}={float(fnl_fid):g}$", fontsize=20)
    ax.legend(loc="best", fontsize=16)
    fig.tight_layout()
    out = dirs["figures"] / f"{stem}.png"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(out.resolve(), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="§4.b sigma(fnl) vs c_f (PIXIE and SPECTER).")
    ap.add_argument("--fnl-fid", type=float, default=25000.0)
    ap.add_argument("--fnl-fiducials", type=str, default=None)
    ap.add_argument("--section4-config", type=Path, default=None)
    ap.add_argument("--n-cf", type=int, default=24)
    ap.add_argument(
        "--pipeline",
        type=str,
        default="analytic_cltt_analytic_b",
    )
    ap.add_argument(
        "--compare-section3",
        action="store_true",
        help="Add horizontal §3 no-dust reference lines and extra columns in the table.",
    )
    args = ap.parse_args()
    for fnl in _fnl_list_from_args(args):
        run_one_fnl(
            fnl_fid=float(fnl),
            n_cf=args.n_cf,
            pipeline=args.pipeline,
            compare_section3=args.compare_section3,
        )


if __name__ == "__main__":
    main()
