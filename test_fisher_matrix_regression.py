"""Regression tests for fisher_matrix (Tier A) and dC^{muT}/dA_s."""

from __future__ import annotations

import numpy as np

from b_integral import b_ell_ns
from fisher_matrix import (
    AS_FID_LEGACY,
    SIGMA_AS_PLANCK2018,
    default_ell_grid,
    fisher_muT_general,
)
from spectra import AS_FID_LEGACY as AS_SPEC
from spectra import Cl_muT, dCl_muT_dAs


def test_As_fid_consistent():
    assert AS_FID_LEGACY == AS_SPEC


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
    test_2d_sigma_fnl_regression()
    test_dCl_dAs_analytic_vs_finite_diff()
    test_3x3_As_prior_marginal_sigma_matches_prior()
    print("test_fisher_matrix_regression: all passed")
