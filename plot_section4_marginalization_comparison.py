'§4.c.ii–iv — Comparison panels: PIXIE vs SPECTER, §3 (no dust) vs §4 (dust), and.'

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


def _sigma_dust(
    *,
    experiment,
    fnl_fid,
    c_f,
    marginalize_dust,
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
        sigma_ns_prior=0.004,
        sigma_AD_prior=1e12,
        sigma_alpha_prior=1e6,
        use_b_analytic=False,
        marginalize_dust=marginalize_dust,
    )
    return float(r.sigma["fnl"])


def _sigma_s3(*, experiment, fnl_fid):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fnl-fid", type=float, default=25000.0, help="Single fiducial if no config.")
    ap.add_argument("--fnl-fiducials", type=str, default=None)
    ap.add_argument("--section4-config", type=Path, default=None)
    ap.add_argument("--cf-ref", type=float, default=1000.0, help=r"$c_f$ for §4 curve in §3 vs §4 panel.")
    ap.add_argument("--n-cf", type=int, default=32)
    ap.add_argument("--pipeline", type=str, default="analytic_cltt_analytic_b")
    args = ap.parse_args()

    fnl_list = _fnl_list(args)
    dirs = ensure_section_layout("section4_foregrounds", args.pipeline)
    cfs = np.logspace(2.0, 4.0, int(args.n_cf))
    cf_tag = _cf_file_tag(float(args.cf_ref))

    s3p = [_sigma_s3(experiment="pixie", fnl_fid=f) for f in fnl_list]
    s3s = [_sigma_s3(experiment="specter", fnl_fid=f) for f in fnl_list]
    s4p = [
        _sigma_dust(
            experiment="pixie",
            fnl_fid=f,
            c_f=args.cf_ref,
            marginalize_dust=True,
        )
        for f in fnl_list
    ]
    s4s = [
        _sigma_dust(
            experiment="specter",
            fnl_fid=f,
            c_f=args.cf_ref,
            marginalize_dust=True,
        )
        for f in fnl_list
    ]
    lines = [
        f"# c_f={float(args.cf_ref):.12g}  (foreground cleaning; used only for sigma_s4_pixie, sigma_s4_specter)",
        "# sigma_s3_* are §3 muT Fisher (no C_l^DD dust); independent of c_f.",
        "fnl_fid,sigma_s3_pixie,sigma_s3_specter,sigma_s4_pixie,sigma_s4_specter",
    ]
    for i, f in enumerate(fnl_list):
        lines.append(
            f"{f:.8g},{s3p[i]:.12e},{s3s[i]:.12e},{s4p[i]:.12e},{s4s[i]:.12e}"
        )
    csv_path = dirs["tables"] / f"section4_marginalization_comparison_s3_vs_s4_grid_cf{cf_tag}.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(csv_path.resolve(), flush=True)

    apply_plot_params()
    import matplotlib.pyplot as plt

    xf = np.asarray(fnl_list, dtype=float)
    for fnl_ref in fnl_list:
        fnl_ref = float(fnl_ref)
        tag = _fnl_file_tag(fnl_ref)
        fig, axes = plt.subplots(3, 1, figsize=(8.0, 11.0), sharex=False)

        sp = np.array(
            [
                _sigma_dust(experiment="specter", fnl_fid=fnl_ref, c_f=c, marginalize_dust=True)
                for c in cfs
            ]
        )
        px = np.array(
            [
                _sigma_dust(experiment="pixie", fnl_fid=fnl_ref, c_f=c, marginalize_dust=True)
                for c in cfs
            ]
        )
        axes[0].plot(cfs, sp / np.maximum(px, 1e-99), color="0.2", lw=2)
        axes[0].axhline(1.0, color="0.5", ls=":")
        axes[0].set_xscale("log")
        axes[0].set_ylabel(
            r"$\sigma(f_{\rm NL})_{\rm SPECTER} / \sigma(f_{\rm NL})_{\rm PIXIE}$"
        )
        axes[0].set_title(
            rf"(ii) Relative sensitivity (marg.\ dust), $f_{{\rm NL}}^{{\rm fid}}={fnl_ref:g}$"
        )

        axes[1].plot(xf, s3p, "o-", color="#3193A2", lw=2, label=r"§3 PIXIE ($A_s$ marg., no dust)")
        axes[1].plot(xf, s3s, "s-", color="#e76f51", lw=2, label=r"§3 SPECTER")
        axes[1].plot(
            xf,
            s4p,
            "o--",
            color="#3193A2",
            alpha=0.85,
            lw=1.5,
            label=rf"§4 PIXIE ($c_f={args.cf_ref:g}$, dust marg.)",
        )
        axes[1].plot(
            xf,
            s4s,
            "s--",
            color="#e76f51",
            alpha=0.85,
            lw=1.5,
            label=rf"§4 SPECTER ($c_f={args.cf_ref:g}$)",
        )
        axes[1].set_xscale("symlog", linthresh=1.0)
        axes[1].set_yscale("log")
        axes[1].set_xlabel(r"$f_{\rm NL}^{\rm fid}$")
        axes[1].set_ylabel(r"$\sigma(f_{\rm NL})$")
        axes[1].set_title(r"(iii) §3 (no dust) vs §4 (dust marginalized)")
        axes[1].legend(loc="best", fontsize=8)

        for exp, color in (("pixie", "#3193A2"), ("specter", "#e76f51")):
            m = np.array(
                [
                    _sigma_dust(experiment=exp, fnl_fid=fnl_ref, c_f=c, marginalize_dust=True)
                    for c in cfs
                ]
            )
            u = np.array(
                [
                    _sigma_dust(experiment=exp, fnl_fid=fnl_ref, c_f=c, marginalize_dust=False)
                    for c in cfs
                ]
            )
            axes[2].plot(cfs, m, "-", color=color, lw=2, label=f"{exp.upper()} marg. dust")
            axes[2].plot(
                cfs, u, ":", color=color, lw=2, alpha=0.9, label=f"{exp.upper()} fixed dust"
            )
        axes[2].set_xscale("log")
        axes[2].set_yscale("log")
        axes[2].set_xlabel(r"$c_f$")
        axes[2].set_ylabel(r"$\sigma(f_{\rm NL})$")
        axes[2].set_title(
            rf"(iv) Marg.\ vs fixed $A_D,\alpha_D$ ($f_{{\rm NL}}^{{\rm fid}}={fnl_ref:g}$)"
        )
        axes[2].legend(loc="best", fontsize=8, ncol=2)

        fig.suptitle(r"§4 foreground marginalization comparisons", fontsize=12, y=1.01)
        fig.tight_layout()
        out = dirs["figures"] / f"section4_marginalization_comparison_fnl{tag}_cf{cf_tag}.pdf"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(out.resolve())


if __name__ == "__main__":
    main()
