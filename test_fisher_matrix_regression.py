"""Regression tests for fisher_matrix (Tier A) and dC^{muT}/dA_s."""

from __future__ import annotations

import numpy as np

from b_integral import b_ell_ns
from fisher_matrix import (
    AS_FID_LEGACY,
    SIGMA_AS_PLANCK2018,
    VARIANCE_FULL_GAUSSIAN_NOISY,
    VARIANCE_PZ_INSTRUMENTAL_APPROX,
    default_ell_grid,
    fisher_1d_fnl_only,
    fisher_muT_general,
)
from spectra import AS_FID_LEGACY as AS_SPEC
from spectra import Cl_mu_mu_gaussian_PZ, Cl_muT, dCl_muT_dAs, sigma2_muT_hat, sigma2_muT_hat_full
from beam import N_mu_mu, W_MU_INV_SPECTER


def test_As_fid_consistent():
    assert AS_FID_LEGACY == AS_SPEC


def test_sigma2_full_matches_pz_approx_when_signal_and_mut_vanish():
    ell = default_ell_grid(1.6)
    cl_tt = np.full_like(ell, 1e-10)
    n_mumu = N_mu_mu(ell, 1.6, w_mu_inv=W_MU_INV_SPECTER)
    z = np.zeros_like(ell)
    s_full = sigma2_muT_hat_full(
        ell, cl_tt, cl_tt_noise=0.0, cl_mumu_signal=z, cl_mumu_noise=n_mumu, cl_mut=z
    )
    s_pz = sigma2_muT_hat(ell, cl_tt, n_mumu)
    np.testing.assert_allclose(s_full, s_pz, rtol=0, atol=0)


def test_fisher_1d_pz_vs_full_noisy_matches_when_fnl0():
    ell = default_ell_grid(1.6)
    f_pz = fisher_1d_fnl_only(
        ell,
        1.6,
        0.965,
        1.1e4,
        46.0,
        0.002,
        w_mu_inv=W_MU_INV_SPECTER,
        use_b_analytic=False,
        variance_mode=VARIANCE_PZ_INSTRUMENTAL_APPROX,
    )
    f_full = fisher_1d_fnl_only(
        ell,
        1.6,
        0.965,
        1.1e4,
        46.0,
        0.002,
        w_mu_inv=W_MU_INV_SPECTER,
        use_b_analytic=False,
        variance_mode=VARIANCE_FULL_GAUSSIAN_NOISY,
        fnl_fid_for_variance=0.0,
    )
    np.testing.assert_allclose(f_full, f_pz, rtol=1e-9)


def test_fisher_muT_general_has_zero_F_cov_by_default():
    ell = default_ell_grid(1.6)
    r = fisher_muT_general(
        ell,
        1.6,
        25_000.0,
        0.965,
        1.1e4,
        46.0,
        0.002,
        dns_step=5e-5,
        sigma_ns_prior=0.004,
        use_b_analytic=False,
    )
    assert r.F_cov.shape == r.F_data.shape
    np.testing.assert_allclose(r.F_cov, 0.0, atol=0.0)


def test_Cl_mu_mu_gaussian_PZ_scale_invariant_ns1():
    
    ell = np.array([2.0, 10.0, 50.0])
    cl = Cl_mu_mu_gaussian_PZ(ell, k_D_f=46.0, ns=1.0, k_p=0.002, k_s=46.0, r_L_mpc=14000.0)
    assert cl.shape == ell.shape
    np.testing.assert_allclose(cl, cl[0], rtol=0, atol=0)
    # ~ 3.5e-17 * ks/(r_L^2 k_D_f^3) with ks=k_D_f => 3.5e-17/(r_L^2 k_D_f^2)
    expect = 3.5e-17 / (14000.0**2 * 46.0**2)
    np.testing.assert_allclose(cl[0], expect, rtol=1e-6)


def test_2d_sigma_fnl_regression():
    """Match 2d (ns, fNL only) forecast for numerical b, f_NL=25000."""
    ell = default_ell_grid(1.6)
    r = fisher_muT_general(
        ell,
        1.6,
        25_000.0,
        0.965,
        1.1e4,
        46.0,
        0.002,
        dns_step=5e-5,
        sigma_ns_prior=0.004,
        use_b_analytic=False,
    )
    assert r.param_names == ("fnl", "ns")
    np.testing.assert_allclose(r.sigma_fnl_marg, 5395.132446, rtol=1e-6)


def test_dCl_dAs_analytic_vs_finite_diff():
    ell = default_ell_grid(1.6)
    ns = 0.965
    k_D_i, k_D_f, k_p = 1.1e4, 46.0, 0.002
    b = np.array(
        [b_ell_ns(int(l), ns, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p) for l in ell.astype(int)]
    )
    fnl = 25_000.0
    h = 1e-12
    fd = (
        Cl_muT(ell, fnl, b, AS_FID_LEGACY + h, k_D_i, k_D_f)
        - Cl_muT(ell, fnl, b, AS_FID_LEGACY - h, k_D_i, k_D_f)
    ) / (2 * h)
    d = dCl_muT_dAs(ell, fnl, b, AS_FID_LEGACY, k_D_i, k_D_f)
    np.testing.assert_allclose(d, fd, rtol=1e-8, atol=1e-15)


def test_3x3_As_prior_marginal_sigma_matches_prior():
    ell = default_ell_grid(1.6)
    r = fisher_muT_general(
        ell,
        1.6,
        25_000.0,
        0.965,
        1.1e4,
        46.0,
        0.002,
        dns_step=5e-5,
        sigma_ns_prior=0.004,
        sigma_As_prior=SIGMA_AS_PLANCK2018,
        use_b_analytic=False,
    )
    assert r.param_names == ("fnl", "ns", "As")

    # Test that marginalized sigma(A_s) matches Planck18 sigma(ln(10^10 A_s)) = 0.014
    np.testing.assert_allclose(r.sigma_As_marg, SIGMA_AS_PLANCK2018, rtol=1e-6)


if __name__ == "__main__":
    test_As_fid_consistent()
    test_sigma2_full_matches_pz_approx_when_signal_and_mut_vanish()
    test_fisher_1d_pz_vs_full_noisy_matches_when_fnl0()
    test_fisher_muT_general_has_zero_F_cov_by_default()
    test_Cl_mu_mu_gaussian_PZ_scale_invariant_ns1()
    test_2d_sigma_fnl_regression()
    test_dCl_dAs_analytic_vs_finite_diff()
    test_3x3_As_prior_marginal_sigma_matches_prior()
    print("test_fisher_matrix_regression: all passed")
