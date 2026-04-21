'Foreground residual (dust) contribution to \\(\\mu T\\) band-power variance —.'

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from beam import N_mu_mu
from fisher_matrix import fisher_cov_term_diagonal
from fisher_matrix import _Cl_derivative_matrix, _b_and_db  # noqa: SLF001

try:
    from .spectra import (
        AS_FID_PLANCK2018,
        Cl_muT,
        Cl_mu_mu_gaussian_PZ,
        Cl_TT,
        cl_tt_on_ell_grid,
        load_ClTT_planck18,
        sigma2_muT_hat_full,
    )
except ImportError:
    from spectra import (
        AS_FID_PLANCK2018,
        Cl_muT,
        Cl_mu_mu_gaussian_PZ,
        Cl_TT,
        cl_tt_on_ell_grid,
        load_ClTT_planck18,
        sigma2_muT_hat_full,
    )

# Azzoni-style exponents (B-mode foreground paper; used as angular template for residual dust).
AZZONI_ALPHA_D = -0.16
ELL0_AZZONI = 80.0
# Amplitude quoted in μK²; convert to dimensionless C_ℓ for sigma_ell^2 in Fisher.
AZZONI_A_D_MUK2 = 28.0
T_CMB_MICROK = 2.72548e6
AZZONI_A_D_DIMENSIONLESS = AZZONI_A_D_MUK2 / (T_CMB_MICROK**2)


def dust_cl_azzoni(ell, A_D, alpha_D, ell0, c_f):
    r"""Residual dust proxy added to \(C_\ell^{\mu\mu,\mathrm{eff}}\): \(A_D (\ell/\ell_0)^{\alpha_D}/c_f\)."""
    ell = np.maximum(np.asarray(ell, dtype=float), 2.0)
    return (float(A_D) * (ell / float(ell0)) ** float(alpha_D)) / float(c_f)


@dataclass
class ForegroundFisherResult:
    F_data: np.ndarray
    F_cov: np.ndarray
    F_total: np.ndarray
    param_names: tuple[str, ...]
    cov: np.ndarray
    sigma: dict[str, float]


def fisher_muT_fnl_ns_with_dust(
    ell,
    fwhm_deg,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    w_mu_inv,
    c_f,
    A_D,
    alpha_D = AZZONI_ALPHA_D,
    ell0 = ELL0_AZZONI,
    As_fid = AS_FID_PLANCK2018,
    dns_step = 5e-5,
    sigma_ns_prior = 0.004,
    sigma_AD_prior = None,
    sigma_alpha_prior = None,
    use_b_analytic = False,
    b_integral_kw = None,
    cl_tt_txt_dir = None,
    marginalize_dust = True,
):
    """Compute the foreground-aware Fisher matrix with optional dust-parameter marginalization."""
    b_kw = dict(b_integral_kw or {})
    if cl_tt_txt_dir is not None:
        _bundle = load_ClTT_planck18(cl_tt_txt_dir)
        cl_tt = cl_tt_on_ell_grid(_bundle["fiducial"], ell)
    else:
        cl_tt = Cl_TT(ell, As_fid)
    b_arr, _ = _b_and_db(
        ell, ns_fid, k_D_i, k_D_f, k_p, dns_step, use_b_analytic, b_kw, b_override=None
    )
    cl_mut = Cl_muT(ell, fnl_fid, b_arr, As_fid, k_D_i, k_D_f)
    cl_mumu_sig = Cl_mu_mu_gaussian_PZ(ell, k_D_f=k_D_f, ns=ns_fid, k_p=k_p)
    n_mumu = N_mu_mu(ell, fwhm_deg, w_mu_inv=w_mu_inv)
    dust = dust_cl_azzoni(ell, A_D, alpha_D, ell0, c_f)
    mumu_noise = n_mumu + dust
    var = sigma2_muT_hat_full(
        ell,
        cl_tt,
        cl_tt_noise=0.0,
        cl_mumu_signal=cl_mumu_sig,
        cl_mumu_noise=mumu_noise,
        cl_mut=cl_mut,
    )
    inv_var = 1.0 / var

    K2, names2 = _Cl_derivative_matrix(
        ell,
        fnl_fid,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        dns_step,
        0.5,
        use_b_analytic,
        b_kw,
        As_fid=As_fid,
        include_As=False,
        include_k_Df=False,
        b_override=None,
    )
    assert names2 == ("fnl", "ns")
    n_ell = ell.shape[0]
    R = (ell / ell0) ** alpha_D
    d_dust_dA = R / c_f
    d_dust_dalpha = A_D * R * np.log(ell / ell0) / c_f
    fac = cl_tt / (2.0 * ell + 1.0)
    dcl_mut_dfnl = K2[:, 0]  # T*b
    dsig_fnl = 2.0 * cl_mut * dcl_mut_dfnl / (2.0 * ell + 1.0)
    dsig_AD = fac * d_dust_dA
    dsig_alpha = fac * d_dust_dalpha
    dsig_ns = np.zeros_like(ell)  # optional: propagate ns through b in var

    if not marginalize_dust:
        param_names = ("fnl", "ns")
        F_data = (K2 * inv_var[:, None]).T @ K2
        dsigma2_2 = {"fnl": dsig_fnl, "ns": dsig_ns}
        F_cov = fisher_cov_term_diagonal(var, dsigma2_2, param_names)
        F_total = F_data + F_cov
        idx = {n: i for i, n in enumerate(param_names)}
        if sigma_ns_prior is not None:
            F_total[idx["ns"], idx["ns"]] += 1.0 / sigma_ns_prior**2
        cov = np.linalg.inv(F_total)
        sigma = {n: float(np.sqrt(max(cov[i, i], 0.0))) for n, i in idx.items()}
        return ForegroundFisherResult(
            F_data=F_data,
            F_cov=F_cov,
            F_total=F_total,
            param_names=param_names,
            cov=cov,
            sigma=sigma,
        )

    K = np.zeros((n_ell, 4), dtype=float)
    K[:, 0:2] = K2
    param_names = ("fnl", "ns", "A_D", "alpha_D")

    F_data = (K * inv_var[:, None]).T @ K

    dsigma2 = {
        "fnl": dsig_fnl,
        "ns": dsig_ns,
        "A_D": dsig_AD,
        "alpha_D": dsig_alpha,
    }
    F_cov = fisher_cov_term_diagonal(var, dsigma2, param_names)

    F_total = F_data + F_cov
    idx = {n: i for i, n in enumerate(param_names)}
    if sigma_ns_prior is not None:
        F_total[idx["ns"], idx["ns"]] += 1.0 / sigma_ns_prior**2
    if sigma_AD_prior is not None:
        F_total[idx["A_D"], idx["A_D"]] += 1.0 / sigma_AD_prior**2
    if sigma_alpha_prior is not None:
        F_total[idx["alpha_D"], idx["alpha_D"]] += 1.0 / sigma_alpha_prior**2

    cov = np.linalg.inv(F_total)
    sigma = {n: float(np.sqrt(max(cov[i, i], 0.0))) for n, i in idx.items()}
    return ForegroundFisherResult(
        F_data=F_data,
        F_cov=F_cov,
        F_total=F_total,
        param_names=param_names,
        cov=cov,
        sigma=sigma,
    )


__all__ = [
    "AZZONI_A_D_DIMENSIONLESS",
    "AZZONI_A_D_MUK2",
    "AZZONI_ALPHA_D",
    "ELL0_AZZONI",
    "T_CMB_MICROK",
    "ForegroundFisherResult",
    "dust_cl_azzoni",
    "fisher_muT_fnl_ns_with_dust",
]
