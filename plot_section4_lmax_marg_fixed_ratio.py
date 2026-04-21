'§4.c.v — Ratio \\(\\sigma(f_{\\rm NL})_{\\rm marg\\,dust} / \\sigma(f_{\\rm NL})_{\\rm fixed\\,dust}\\).'

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from config_section4 import A_D_CODE, SIGMA_AD_PRIOR, SIGMA_ALPHA_PRIOR, SIGMA_NS_PRIOR
from config_section_common import FWHM_PIXIE, FWHM_SPECTER, K_D_F, K_D_I, K_P, NS_FID
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from output_paths import ensure_section_layout

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

from spectra import AS_FID_PLANCK2018
LMAX_MAX = 1000


def _fnl_file_tag(fnl):
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _lmax_grid(n, lmax_cap):
    cap = min(int(lmax_cap), LMAX_MAX)
    pts = np.unique(np.round(np.linspace(2, cap, n)).astype(int))
    return [int(x) for x in pts if x >= 2]


def _fnl_list(args):
    if args.section4_config is not None:
        p = Path(args.section4_config).expanduser().resolve()
        data = json.loads(p.read_text(encoding="utf-8"))
        return [float(x) for x in data["fnl_fiducials"]]
    if args.fnl_fiducials:
        return [float(x.strip()) for x in args.fnl_fiducials.split(",") if x.strip()]
    return [float(args.fnl_fid)]


def run_one(
    *,
    fnl_fid,
    c_f,
    experiment,
    n_lmax,
    lmax_cap,
    pipeline,
):
    tag = _fnl_file_tag(float(fnl_fid))
    cf_tag = str(int(c_f)) if abs(c_f - round(c_f)) < 1e-6 else str(c_f).replace(".", "p")
    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    lmax_list = _lmax_grid(int(n_lmax), int(lmax_cap))
    dirs = ensure_section_layout("section4_foregrounds", pipeline)

    ratios = []
    sm = []
    sf = []
    for lmax in lmax_list:
        ell = np.arange(2, lmax + 1, dtype=float)
        if ell.size == 0:
            ratios.append(float("nan"))
            sm.append(float("nan"))
            sf.append(float("nan"))
            continue
        rm = fisher_muT_fnl_ns_with_dust(
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
            use_b_analytic=False,
            marginalize_dust=True,
        )
        ru = fisher_muT_fnl_ns_with_dust(
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
            use_b_analytic=False,
            marginalize_dust=False,
        )
        m, f = rm.sigma["fnl"], ru.sigma["fnl"]
        sm.append(m)
        sf.append(f)
        ratios.append(m / max(f, 1e-99))

    rows = ["lmax,ratio_marg_over_fixed,sigma_marg,sigma_fixed"]
    for i, lmax in enumerate(lmax_list):
        rows.append(f"{lmax},{ratios[i]:.12e},{sm[i]:.12e},{sf[i]:.12e}")
    stem = f"sigma_fnl_lmax_marg_fixed_ratio_{experiment}_fnl{tag}_cf{cf_tag}"
    body = "\n".join(rows) + "\n"
    (dirs["tables"] / f"{stem}.csv").write_text(body, encoding="utf-8")
    (dirs["tables"] / f"{stem}.txt").write_text(body, encoding="utf-8")

    apply_plot_params()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(lmax_list, ratios, "k-", lw=2)
    ax.axhline(1.0, color="0.5", ls=":", label=r"ratio $=1$")
    ax.set_xlabel(r"$\ell_{\rm max}$")
    ax.set_ylabel(
        r"$\sigma(f_{\rm NL})_{\rm marg\,dust} \,/\, \sigma(f_{\rm NL})_{\rm fixed\,dust}$"
    )
    ax.set_title(
        rf"§4.c.v: dust marginalization impact vs $\ell_{{\rm max}}$ ({experiment}), "
        rf"$c_f={c_f:g}$, $f_{{\rm NL}}^{{\rm fid}}={float(fnl_fid):g}$"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = dirs["figures"] / f"{stem}.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    (dirs["logs"] / f"plot_{stem}.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "fnl_fid": float(fnl_fid),
                "c_f": float(c_f),
                "lmax_cap": int(lmax_cap),
                "n_lmax": len(lmax_list),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(out.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnl-fid", type=float, default=25000.0)
    ap.add_argument("--fnl-fiducials", type=str, default=None)
    ap.add_argument("--section4-config", type=Path, default=None)
    ap.add_argument("--cf", type=float, default=1000.0)
    ap.add_argument("--experiment", choices=("pixie", "specter", "both"), default="both")
    ap.add_argument("--n-lmax", type=int, default=28)
    ap.add_argument("--lmax-cap", type=int, default=600)
    ap.add_argument("--pipeline", type=str, default="analytic_cltt_analytic_b")
    args = ap.parse_args()

    exps = ("pixie", "specter") if args.experiment == "both" else (args.experiment,)
    for fnl in _fnl_list(args):
        for exp in exps:
            run_one(
                fnl_fid=float(fnl),
                c_f=float(args.cf),
                experiment=exp,
                n_lmax=args.n_lmax,
                lmax_cap=args.lmax_cap,
                pipeline=args.pipeline,
            )


if __name__ == "__main__":
    main()
