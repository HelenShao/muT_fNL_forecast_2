r"""
4-parameter muT Fisher forecast: (f_{\mathrm{NL}}, n_s, A_s, k_{D,f}) with Planck-style priors
on n_s and A_s, and k_{D,f} included as a nuisance (Tier A; numerical b(\ell,n_s) and
finite-difference \partial b/\partial k_{D,f} unless you switch to analytic b).

Run from this directory:
    python3 fisher_4d.py

Default output file (override with ``--output``):
    cmbs4/results/forecast_tables/SPECTER_fNL_4d_forecasts.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from .fisher_matrix import (
        AS_FID_LEGACY,
        SIGMA_AS_PLANCK2018,
        W_MU_INV_SPECTER,
        default_ell_grid,
        fisher_1d_fnl_only,
        fisher_muT_general,
    )
    from .output_paths import ensure_dir, forecast_tables_dir
except ImportError:
    from fisher_matrix import (
        AS_FID_LEGACY,
        SIGMA_AS_PLANCK2018,
        W_MU_INV_SPECTER,
        default_ell_grid,
        fisher_1d_fnl_only,
        fisher_muT_general,
    )
    from output_paths import ensure_dir, forecast_tables_dir


def marginal_corr(cov: np.ndarray, i: int, j: int) -> float:
    if cov[i, i] <= 0 or cov[j, j] <= 0:
        return 0.0
    return float(cov[i, j] / np.sqrt(cov[i, i] * cov[j, j]))


def run_forecast(
    *,
    outfile: Path,
    fwhm_deg: float,
    w_mu_inv: float,
    dns_step: float,
    dkdf_step: float,
    sigma_ns_planck: float,
    sigma_as_planck: float,
    sigma_k_Df_prior: float | None,
    use_b_analytic: bool,
    fnl_fiducials: tuple[float, ...],
) -> None:
    ns_fid = 0.965
    k_p = 0.002
    k_D_i = 1.1e4
    k_D_f = 46.0

    ell = default_ell_grid(fwhm_deg)
    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("fNL Fisher forecast — 4D (f_NL, n_s, A_s, k_{D,f})")
    emit(f"l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg} deg, n_s = {ns_fid}")
    emit(f"A_s fid (Delta_R^2 at pivot): {AS_FID_LEGACY:.6e}")
    emit(f"k_D,i = {k_D_i:g} Mpc^-1, k_D,f = {k_D_f:g} Mpc^-1, k_p = {k_p:g} Mpc^-1")
    emit(f"mu noise: SPECTER (w_mu^-1 from beam.N_mu_mu)")
    emit(f"Planck prior on n_s: sigma(ns) = {sigma_ns_planck}")
    emit(
        f"Planck-scale prior on A_s (Delta_R^2 at pivot): sigma(A_s) = {sigma_as_planck:.6e} "
        "(see SIGMA_AS_PLANCK2018 in spectra.py)"
    )
    if sigma_k_Df_prior is not None:
        emit(
            f"External prior on k_D,f: sigma(k_D,f) = {sigma_k_Df_prior:g} Mpc^-1 "
            "(regularizes Fisher when f_NL^fid=0, since dC/dn_s and dC/dk_D,f scale as f_NL)"
        )
    else:
        emit(
            "External prior on k_D,f: none — do not use f_NL^fid=0 in the table (Fisher is singular)."
        )
    emit(f"use_b_analytic = {use_b_analytic}; dns_step = {dns_step}; dkdf_step = {dkdf_step} Mpc^-1")
    emit()

    # 1D benchmarks (fix nuisances — same spirit as fisher_ns.py)
    f_1d_a = fisher_1d_fnl_only(
        ell,
        fwhm_deg,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=w_mu_inv,
        b_override=1.0,
    )
    sig_1d_a = 1.0 / np.sqrt(f_1d_a)
    emit(
        f"1D Fisher (analytic b=1, fix nuisances): F = {f_1d_a:.6e},  sigma(f_NL) = {sig_1d_a:.6e}"
    )
    f_1d_n = fisher_1d_fnl_only(
        ell,
        fwhm_deg,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=w_mu_inv,
        use_b_analytic=use_b_analytic,
    )
    sig_1d_n = 1.0 / np.sqrt(f_1d_n)
    emit(
        f"1D Fisher (b(l,n_s) style, fix nuisances): F = {f_1d_n:.6e},  "
        f"sigma(f_NL) = {sig_1d_n:.6e}"
    )
    emit()
    emit(
        "Note: sigma(k_D,f) can sit near the external k_D,f prior when |corr(f_NL,k_D,f)| ~ 1 "
        "(strong degeneracy in the muT template)."
    )
    emit()

    emit(
        f"{'f_NL fid':>12}  {'sigma_unmarg':>14}  {'sigma_fNL_m':>14}  {'sigma_ns_m':>14}  "
        f"{'sigma_As_m':>14}  {'sigma_kDf_m':>14}  {'corr(fNL,ns)':>14}  {'corr(fNL,kDf)':>14}"
    )
    emit("-" * 126)

    for fnl in fnl_fiducials:
        r = fisher_muT_general(
            ell,
            fwhm_deg,
            fnl,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            w_mu_inv=w_mu_inv,
            dns_step=dns_step,
            dkdf_step=dkdf_step,
            sigma_ns_prior=sigma_ns_planck,
            sigma_As_prior=sigma_as_planck,
            sigma_k_Df_prior=sigma_k_Df_prior,
            include_As=True,
            include_k_Df=True,
            use_b_analytic=use_b_analytic,
        )
        assert r.param_names == ("fnl", "ns", "As", "k_Df")
        c = r.cov_marginal
        i_fnl, i_kdf = 0, 3
        rho_fnl_kdf = marginal_corr(c, i_fnl, i_kdf)
        emit(
            f"{fnl:12.0f}  {r.sigma_fnl_unmarg:14.6e}  {r.sigma_fnl_marg:14.6e}  "
            f"{r.sigma_ns_marg:14.6e}  {r.sigma_As_marg:14.6e}  {r.sigma_k_Df_marg:14.6e}  "
            f"{r.corr_fnl_ns:14.4f}  {rho_fnl_kdf:14.4f}"
        )

    emit()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {outfile}", file=sys.stderr)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="4D muT Fisher (f_NL, n_s, A_s, k_D,f) with SPECTER noise; save table to file."
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for the forecast table (default: cmbs4/results/forecast_tables/"
            "SPECTER_fNL_4d_forecasts.txt)."
        ),
    )
    p.add_argument(
        "--fwhm-deg",
        type=float,
        default=1.6,
        help="Beam FWHM in degrees (default: 1.6).",
    )
    p.add_argument(
        "--dns-step",
        type=float,
        default=5e-5,
        help="Step for db/dn_s finite difference (default: 5e-5).",
    )
    p.add_argument(
        "--dkdf-step",
        type=float,
        default=0.5,
        help="Half-step in k_D,f for db/dk_D,f finite difference, Mpc^-1 (default: 0.5).",
    )
    p.add_argument(
        "--sigma-kdf-prior",
        type=float,
        default=10_000.0,
        metavar="SIGMA",
        help=(
            "Gaussian prior sigma on k_D,f (Mpc^-1) on the Fisher diagonal. "
            "Default: 1e4 (very weak; needed so F is invertible when f_NL^fid=0). "
            "Use --no-kdf-prior to turn off (then f_NL=0 rows are skipped)."
        ),
    )
    p.add_argument(
        "--no-kdf-prior",
        action="store_true",
        help="Do not add a k_D,f prior (skips f_NL^fid=0 rows; singular otherwise).",
    )
    p.add_argument(
        "--use-b-analytic",
        action="store_true",
        help="Use analytic b and db/dk_D,f (default: numerical b and finite differences).",
    )
    args = p.parse_args(argv)

    if args.output is None:
        args.output = ensure_dir(forecast_tables_dir()) / "SPECTER_fNL_4d_forecasts.txt"

    fnl_all = (0.0, 1.0, 5.0, 12_500.0, 25_000.0)
    sigma_k_Df: float | None
    if args.no_kdf_prior:
        sigma_k_Df = None
        fnl_fiducials = tuple(f for f in fnl_all if f != 0.0)
        if len(fnl_fiducials) < len(fnl_all):
            print(
                "Note: omitting f_NL^fid=0 (singular 4D Fisher without a k_D,f prior).",
                file=sys.stderr,
            )
    else:
        sigma_k_Df = args.sigma_kdf_prior
        fnl_fiducials = fnl_all

    run_forecast(
        outfile=args.output,
        fwhm_deg=args.fwhm_deg,
        w_mu_inv=W_MU_INV_SPECTER,
        dns_step=args.dns_step,
        dkdf_step=args.dkdf_step,
        sigma_ns_planck=0.004,
        sigma_as_planck=SIGMA_AS_PLANCK2018,
        sigma_k_Df_prior=sigma_k_Df,
        use_b_analytic=args.use_b_analytic,
        fnl_fiducials=fnl_fiducials,
    )


if __name__ == "__main__":
    main()
