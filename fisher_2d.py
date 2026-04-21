"""2-parameter Fisher (f_NL, n_s) for C_l^{muT} with optional Planck prior on n_s."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

try:
    from .beam import N_mu_mu, ell_max_from_fwhm_deg
    from .b_integral import b_analytic, b_ell_ns, db_dns_central
    from .spectra import Cl_TT, T_muT_ell, sigma2_muT_hat
except ImportError: 
    from beam import N_mu_mu, ell_max_from_fwhm_deg
    from b_integral import b_analytic, b_ell_ns, db_dns_central
    from spectra import Cl_TT, T_muT_ell, sigma2_muT_hat


@dataclass
class FisherMuTResult:
    """Fisher matrices and derived uncertainties for (f_NL, n_s)."""

    F_data: np.ndarray
    F_total: np.ndarray
    sigma_fnl_unmarg: float
    sigma_fnl_marg: float
    sigma_ns_unmarg: float
    sigma_ns_marg: float
    corr_fnl_ns: float


def _K_derivatives(
    ell,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    dns_step,
    use_b_analytic,
    b_kw,
):
    """dC^{muT}/df_NL and dC^{muT}/dn_s with C^{muT} = f_NL * b * T_l"""
    ell_i = ell.astype(int)
    T = T_muT_ell(ell)
    if use_b_analytic:
        b0 = b_analytic(ns_fid, k_D_i, k_D_f, k_p)
        db_dns = 0.5 * np.log((k_D_i * k_D_f) / (4.0 * k_p**2))
        b = np.full_like(ell, b0, dtype=float)
    else:
        # numerically compute b(l, ns) and db/dn_s
        b = np.array([b_ell_ns(int(l), ns_fid, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw) for l in ell_i])
        db = np.array(
            [db_dns_central(int(l), ns_fid, dns_step, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw) for l in ell_i]
        )
    K_fnl = T * b
    if use_b_analytic:
        K_ns = fnl_fid * T * db_dns
    else:
        K_ns = fnl_fid * T * db
    return K_fnl, K_ns


def fisher_muT_fnl_ns(
    ell,
    fwhm_deg,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    dns_step = 5e-5,
    sigma_ns_prior = 0.004,
    use_b_analytic = False,
    b_integral_kw = None,
):
    """Compute the Gaussian two-parameter Fisher matrix for (f_NL, n_s)."""
    b_kw = dict(b_integral_kw or {})
    cl_tt = Cl_TT(ell)
    n_mumu = N_mu_mu(ell, fwhm_deg)
    var = sigma2_muT_hat(ell, cl_tt, n_mumu)

    # denominator of Fisher matrix
    inv_var = 1.0 / var

    # Compute derivatives of C_l^{muT} with respect to f_NL and n_s
    K_fnl, K_ns = _K_derivatives(
        ell, fnl_fid, ns_fid, k_D_i, k_D_f, k_p, dns_step, use_b_analytic, b_kw
    )

    # Fisher matrix elements (2x2)
    f00 = float(np.sum(inv_var * K_fnl * K_fnl))
    f01 = float(np.sum(inv_var * K_fnl * K_ns))
    f11 = float(np.sum(inv_var * K_ns * K_ns))
    F_data = np.array([[f00, f01], [f01, f11]], dtype=float)

    F_total = F_data.copy()
    if sigma_ns_prior is not None:
        # add Planck prior on ns
        F_total[1, 1] += 1.0 / (sigma_ns_prior**2)

    # fisher covariance matrix - for marginalized uncertainties, need inverse!
    cov = np.linalg.inv(F_total)

    # unmarginalized uncertainties
    sigma_fnl_u = 1.0 / np.sqrt(F_data[0, 0]) if F_data[0, 0] > 0 else np.inf
    sigma_ns_u = 1.0 / np.sqrt(F_data[1, 1]) if F_data[1, 1] > 0 else np.inf

    # corr. between f_NL and n_s
    corr = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])) if cov[0, 0] > 0 and cov[1, 1] > 0 else 0.0

    # pack results into dataclass
    return FisherMuTResult(
        F_data=F_data,
        F_total=F_total,
        sigma_fnl_unmarg=sigma_fnl_u,
        sigma_fnl_marg=float(np.sqrt(cov[0, 0])),
        sigma_ns_unmarg=sigma_ns_u,
        sigma_ns_marg=float(np.sqrt(cov[1, 1])),
        corr_fnl_ns=corr,
    )


def fisher_1d_fnl_only(
    ell,
    fwhm_deg,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    use_b_analytic = True,
    b_integral_kw = None,
):
    """Compute the single-parameter Fisher information for f_NL."""
    b_kw = dict(b_integral_kw or {})
    cl_tt = Cl_TT(ell)
    n_mumu = N_mu_mu(ell, fwhm_deg)
    var = sigma2_muT_hat(ell, cl_tt, n_mumu)
    ell_i = ell.astype(int)
    T = T_muT_ell(ell)
    if use_b_analytic:
        b0 = b_analytic(ns_fid, k_D_i, k_D_f, k_p)
        b = np.full_like(ell, b0, dtype=float)
    else:
        b = np.array([b_ell_ns(int(l), ns_fid, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw) for l in ell_i])
    K_fnl = T * b
    return float(np.sum(K_fnl**2 / var))


def default_ell_grid(fwhm_deg, ell_min = 2):
    lmax = ell_max_from_fwhm_deg(fwhm_deg)
    return np.arange(ell_min, int(np.ceil(lmax)) + 1, dtype=float)
