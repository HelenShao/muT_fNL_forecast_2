"""
muT Fisher forecast: 1D (f_NL) and 2D (f_NL, n_s) with Planck prior on n_s.

Fiducial f_NL values:
0, 1, 5 (Planck 2018 local-type |f_NL| ~ few), 12_500, 25_000
(default reference f_NL = 1 so dC/dn_s != 0 when using numerical b)
"""

from __future__ import annotations
import numpy as np

try:
    from .fisher_matrix import (
        W_MU_INV_PIXIE,
        W_MU_INV_SPECTER,
        default_ell_grid,
        fisher_1d_fnl_only,
        fisher_muT_general,
    )
except ImportError:
    from fisher_matrix import (
        W_MU_INV_PIXIE,
        W_MU_INV_SPECTER,
        default_ell_grid,
        fisher_1d_fnl_only,
        fisher_muT_general,
    )


def main() -> None:
    # --- same baseline as 1D fisher ---
    fwhm_deg = 1.6 #lmax=84
    # mu autospectrum noise w_mu^{-1} in C_l^{mu mu,N} (beam.N_mu_mu); PIXIE vs SPECTER
    #w_mu_inv = W_MU_INV_PIXIE
    w_mu_inv = W_MU_INV_SPECTER
    ns_fid = 0.965
    k_p = 0.002
    k_D_i = 1.1e4
    k_D_f = 46.0

    ell = default_ell_grid(fwhm_deg)
    sigma_ns_planck = 0.004

    fnl_fiducials = (0.0, 1.0, 5.0, 12_500.0, 25_000.0)

    print("fNL Fisher forecast")
    print(f"l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg} deg, n_s = {ns_fid}")
    print(f"Planck prior on n_s: sigma(ns) = {sigma_ns_planck}")
    print()

    # 1D benchmarks (fix n_s)
    f_1d_a = fisher_1d_fnl_only(
        ell,
        fwhm_deg,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=w_mu_inv,
        b_override=1.0,
        #use_b_analytic=True,
    )
    sig_1d_a = 1.0 / np.sqrt(f_1d_a)
    print(f"1D Fisher (analytic b, fix n_s): F = {f_1d_a:.6e},  sigma(f_NL) = {sig_1d_a:.6e}")
    f_1d_n = fisher_1d_fnl_only(
        ell,
        fwhm_deg,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        w_mu_inv=w_mu_inv,
        use_b_analytic=False,
    )
    sig_1d_n = 1.0 / np.sqrt(f_1d_n)
    print(f"1D Fisher (numerical b(l,n_s), fix n_s): F = {f_1d_n:.6e},  sigma(f_NL) = {sig_1d_n:.6e}")
    print()

    print(
        f"{'f_NL fid':>12}  {'sigma_unmarg':>12}  {'sigma_marg':>12}  {'sigma_ns marg':>12}  {'corr(f_NL,n_s)':>12}"
    )
    print("-" * 70)

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
            dns_step=5e-5,
            sigma_ns_prior=sigma_ns_planck,
            use_b_analytic=False,
        )
        print(
            f"{fnl:12.0f}  {r.sigma_fnl_unmarg:12.6e}  {r.sigma_fnl_marg:12.6e}  "
            f"{r.sigma_ns_marg:12.6e}  {r.corr_fnl_ns:12.4f}"
        )


if __name__ == "__main__":
    main()
