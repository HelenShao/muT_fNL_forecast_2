'Section 2 (Cabass-style): \\(\\sigma(f_{\\rm NL})\\) vs \\(\\ell_{\\max}\\) up to 1000.'

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER, ell_max_from_fwhm_deg
from b_integral import b_analytic, b_ell_ns, db_dns_central
from fisher_matrix import (
    VARIANCE_FULL_GAUSSIAN_CV,
    VARIANCE_PZ_INSTRUMENTAL_APPROX,
    fisher_1d_fnl_only,
    fisher_muT_general,
)

FIG_DPI = 200
from output_paths import ensure_section_layout

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

FWHM_PIXIE = 1.6
FWHM_SPECTER = 1.0
NS_FID = 0.965
K_D_I = 1.1e4
K_D_F = 46.0
K_P = 0.002
SIGMA_NS_PRIOR = 0.004
DNS_STEP_NUMERIC_B = 5e-5
LMAX_MAX = 1000
DEFAULT_FNL_FIDUCIALS = (1.0, 10.0, 1000.0, 15000.0, 25000.0)


def _parse_fnl_csv(s):
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _lmax_grid():
    pts = np.unique(np.round(np.linspace(2, LMAX_MAX, 120)).astype(int))
    return [int(x) for x in pts if x >= 2]


def _fnl_file_tag(fnl):
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _crossing_lmax(series):
    for i in range(1, len(series)):
        l0, u0, m0 = series[i - 1]
        l1, u1, m1 = series[i]
        if (u0 - m0) * (u1 - m1) <= 0 and u0 != m0:
            return l1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=("pixie", "specter"), default="pixie")
    ap.add_argument(
        "--fnl-fiducials",
        type=str,
        default=",".join(str(x) for x in DEFAULT_FNL_FIDUCIALS),
        help="Comma-separated f_NL fiducials for marg_ns curves (Cabass often uses 1).",
    )
    ap.add_argument(
        "--include-cv-csv",
        action="store_true",
        help="Also write CV-limited 1D sigma rows to the CSV (mode 'cv'; not plotted by default).",
    )
    ap.add_argument(
        "--cv",
        action="store_true",
        help=(
            "CV Fisher: full Gaussian sigma2, N_mu_mu=N_TT=0, numerical b(l) and db/dn_s, "
            "sigma_ns_prior=None; write sigma_fnl_vs_lmax_cv_fnl<tag>.png (two curves only)."
        ),
    )
    ap.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Figure format for the standard Cabass-style plots (default: png).",
    )
    args = ap.parse_args()
    fnl_fids = _parse_fnl_csv(args.fnl_fiducials)

    if args.experiment == "pixie":
        fwhm, w_inv = FWHM_PIXIE, W_MU_INV_PIXIE
    else:
        fwhm, w_inv = FWHM_SPECTER, W_MU_INV_SPECTER

    b0 = b_analytic(NS_FID, K_D_I, K_D_F, K_P)

    section = f"section2_cabass_{args.experiment}"
    dirs = ensure_section_layout(section, "analytic_cltt_analytic_b")
    (dirs["logs"] / "config.json").write_text(
        json.dumps(
            {
                "experiment": args.experiment,
                "fwhm_deg": fwhm,
                "lmax_max": LMAX_MAX,
                "fnl_fiducials": fnl_fids,
                "ns_fid": NS_FID,
                "sigma_ns_prior": SIGMA_NS_PRIOR,
                "b_mode": "analytic",
                "b_analytic_fid": b0,
                "k_D_i": K_D_I,
                "k_D_f": K_D_F,
                "k_p": K_P,
                "cv_plots": bool(args.cv),
                "figure_format": args.format,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lbeam_pix = ell_max_from_fwhm_deg(FWHM_PIXIE)
    lbeam_sp = ell_max_from_fwhm_deg(FWHM_SPECTER)

    b_db_prec_full: tuple[np.ndarray, np.ndarray] | None = None
    if args.cv:
        ell_prec = np.arange(2, LMAX_MAX + 1, dtype=float)
        li = ell_prec.astype(int)
        b_full = np.array(
            [b_ell_ns(int(l), NS_FID, k_D_i=K_D_I, k_D_f=K_D_F, k_p=K_P) for l in li],
            dtype=float,
        )
        db_full = np.array(
            [
                db_dns_central(
                    int(l),
                    NS_FID,
                    DNS_STEP_NUMERIC_B,
                    k_D_i=K_D_I,
                    k_D_f=K_D_F,
                    k_p=K_P,
                )
                for l in li
            ],
            dtype=float,
        )
        b_db_prec_full = (b_full, db_full)

    rows: list[str] = ["lmax,fnl_fid,mode,sigma_unmarg,sigma_marg"]
    instrumental: list[tuple[int, float, float]] = []
    marg_by_fnl: dict[float, list[tuple[int, float, float]]] = {f: [] for f in fnl_fids}
    marg_by_fnl_cv: dict[float, list[tuple[int, float, float]]] = {f: [] for f in fnl_fids}

    for lmax in _lmax_grid():
        ell = np.arange(2, lmax + 1, dtype=float)
        if ell.size == 0:
            continue

        if args.include_cv_csv:
            f_cv = fisher_1d_fnl_only(
                ell,
                fwhm,
                NS_FID,
                K_D_I,
                K_D_F,
                K_P,
                w_mu_inv=0.0,
                use_b_analytic=True,
                variance_mode=VARIANCE_FULL_GAUSSIAN_CV,
                fnl_fid_for_variance=0.0,
            )
            s_cv = 1.0 / math.sqrt(f_cv) if f_cv > 0 else float("nan")
            rows.append(f"{lmax},nan,cv,{s_cv:.12e},{s_cv:.12e}")

        f_ins = fisher_1d_fnl_only(
            ell,
            fwhm,
            NS_FID,
            K_D_I,
            K_D_F,
            K_P,
            w_mu_inv=w_inv,
            use_b_analytic=True,
            variance_mode=VARIANCE_PZ_INSTRUMENTAL_APPROX,
        )
        s_ins = 1.0 / math.sqrt(f_ins) if f_ins > 0 else float("nan")
        instrumental.append((lmax, s_ins, s_ins))
        rows.append(f"{lmax},nan,instrumental,{s_ins:.12e},{s_ins:.12e}")

        for fnl in fnl_fids:
            r = fisher_muT_general(
                ell,
                fwhm,
                fnl,
                NS_FID,
                K_D_I,
                K_D_F,
                K_P,
                w_mu_inv=w_inv,
                sigma_ns_prior=SIGMA_NS_PRIOR,
                use_b_analytic=True,
                variance_mode=VARIANCE_PZ_INSTRUMENTAL_APPROX,
            )
            marg_by_fnl[fnl].append((lmax, r.sigma_fnl_unmarg, r.sigma_fnl_marg))
            rows.append(
                f"{lmax},{fnl:g},marg_ns,{r.sigma_fnl_unmarg:.12e},{r.sigma_fnl_marg:.12e}"
            )

            if args.cv:
                assert b_db_prec_full is not None
                n_ell = len(ell)
                r_cv = fisher_muT_general(
                    ell,
                    fwhm,
                    fnl,
                    NS_FID,
                    K_D_I,
                    K_D_F,
                    K_P,
                    w_mu_inv=0.0,
                    sigma_ns_prior=None,
                    use_b_analytic=False,
                    variance_mode=VARIANCE_FULL_GAUSSIAN_CV,
                    cl_tt_noise=0.0,
                    dns_step=DNS_STEP_NUMERIC_B,
                    b_db_prec=(b_db_prec_full[0][:n_ell], b_db_prec_full[1][:n_ell]),
                )
                marg_by_fnl_cv[fnl].append(
                    (lmax, r_cv.sigma_fnl_unmarg, r_cv.sigma_fnl_marg)
                )
                rows.append(
                    f"{lmax},{fnl:g},cv_bias,{r_cv.sigma_fnl_unmarg:.12e},{r_cv.sigma_fnl_marg:.12e}"
                )

    csv_path = dirs["tables"] / "sigma_fnl_vs_lmax.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    apply_plot_params()
    import matplotlib.pyplot as plt

    lx_i = [d[0] for d in instrumental]
    sy_i = [d[1] for d in instrumental]

    fig_paths: list[Path] = []
    for fnl in fnl_fids:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.plot(lx_i, sy_i, "-", color="0.2", lw=2, label="instrumental (1D)")
        data = marg_by_fnl[fnl]
        lx = [d[0] for d in data]
        su = [d[1] for d in data]
        sm = [d[2] for d in data]
        ax.plot(lx, su, "--", color="C0", lw=1.5, alpha=0.9, label=r"marg_ns unmarg $f_{\rm NL}$")
        ax.plot(lx, sm, "-", color="C0", lw=2.2, label=r"marg_ns marg $f_{\rm NL}$")
        cross = _crossing_lmax(data)
        if cross is not None:
            print(f"sigma_u ~ sigma_m crossing (marg_ns), f_NL^fid={fnl:g}: ell_max ~ {cross}", flush=True)

        ax.axvline(lbeam_pix, color="0.4", ls=":", lw=1, label=rf"PIXIE $\ell_{{\rm beam}}\approx{lbeam_pix:.0f}$")
        ax.axvline(lbeam_sp, color="0.6", ls=":", lw=1, label=rf"SPECTER $\ell_{{\rm beam}}\approx{lbeam_sp:.0f}$")

        ax.set_xlabel(r"$\ell_{\rm max}$")
        ax.set_ylabel(r"$\sigma(f_{\rm NL})$")
        ax.set_yscale("log")
        ax.legend(fontsize=9, loc="best")
        ax.set_title(
            rf"Cabass-style ({args.experiment.upper()}): $n_s$ marginalization; "
            rf"$f_{{\rm NL}}^{{\rm fid}}={fnl:g}$; analytic $b$, $b_{{\rm fid}}={b0:.3f}$"
        )
        fig.tight_layout()
        tag = _fnl_file_tag(fnl)
        ext = args.format
        fig_path = dirs["figures"] / f"sigma_fnl_vs_lmax_fnl{tag}.{ext}"
        fig.savefig(fig_path, bbox_inches="tight", dpi=FIG_DPI if ext == "png" else None)
        plt.close(fig)
        fig_paths.append(fig_path)

    if args.cv:
        for fnl in fnl_fids:
            fig, ax = plt.subplots(figsize=(8.5, 5.2))
            data_cv = marg_by_fnl_cv[fnl]
            lx = [d[0] for d in data_cv]
            su = [d[1] for d in data_cv]
            sm = [d[2] for d in data_cv]
            ax.plot(
                lx,
                su,
                "--",
                color="C0",
                lw=1.5,
                alpha=0.9,
                label=r"CV unmarg.\ $\sigma(f_{\rm NL})$",
            )
            ax.plot(
                lx,
                sm,
                "-",
                color="C0",
                lw=2.2,
                label=r"CV marg.\ $\sigma(f_{\rm NL})$ (bias $b(\ell)$ via $n_s$, no $n_s$ prior)",
            )
            cross = _crossing_lmax(data_cv)
            if cross is not None:
                print(
                    f"[CV] sigma_u ~ sigma_m crossing (cv_bias), f_NL^fid={fnl:g}: ell_max ~ {cross}",
                    flush=True,
                )

            ax.axvline(lbeam_pix, color="0.4", ls=":", lw=1, label=rf"PIXIE $\ell_{{\rm beam}}\approx{lbeam_pix:.0f}$")
            ax.axvline(lbeam_sp, color="0.6", ls=":", lw=1, label=rf"SPECTER $\ell_{{\rm beam}}\approx{lbeam_sp:.0f}$")

            ax.set_xlabel(r"$\ell_{\rm max}$")
            ax.set_ylabel(r"$\sigma(f_{\rm NL})$")
            ax.set_yscale("log")
            ax.legend(fontsize=9, loc="best")
            ax.set_title(
                rf"Cosmic variance ({args.experiment.upper()}): numerical $b(\ell)$, "
                rf"$\partial b/\partial n_s$; no $n_s$ prior; "
                rf"$N_\ell^{{\mu\mu}}=N_\ell^{{TT}}=0$; "
                rf"$f_{{\rm NL}}^{{\rm fid}}={fnl:g}$"
            )
            fig.tight_layout()
            tag = _fnl_file_tag(fnl)
            fig_path_cv = dirs["figures"] / f"sigma_fnl_vs_lmax_cv_fnl{tag}.png"
            fig.savefig(fig_path_cv, bbox_inches="tight", dpi=FIG_DPI)
            plt.close(fig)
            fig_paths.append(fig_path_cv)

    md_lines = [
        f"# Section 2 — Cabass-style $\\ell_{{\\max}}$ ({args.experiment})",
        "",
        f"Data: `{csv_path.name}`.",
        "",
        r"Unmarginalized vs marginalized **marg_ns** curves should approach each other beyond a moderate "
        r"$\ell_{\max}$ (especially at low $f_{\rm NL}^{\rm fid}$, e.g. Cabass $f_{\rm NL}=1$).",
        "",
        f"Fiducials: $n_s={NS_FID}$, analytic $b$, $(k_{{D,i}},k_{{D,f}},k_p)=({K_D_I:g},{K_D_F:g},{K_P:g})\ \\mathrm{{Mpc}}^{{-1}}$.",
    ]
    (dirs["tables"] / "section2_cabass_analysis.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(csv_path)
    for p in fig_paths:
        print(p)


if __name__ == "__main__":
    main()
