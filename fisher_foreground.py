r"""
Foreground residual (dust) contribution to \(\mu T\) band-power variance — 
\(C_\ell \propto A_D (\ell/\ell_0)^{\alpha_D}\), scaled by cleaning factor \(c_f\).

dust nuisances enter in \(\sigma_\ell^2\)); use
``F_data`` + diagonal ``F_cov`` (parameter-dependent variance).

Fiducial numbers from Azzoni et al.\ (polarized foreground paper): \(A_D=28\,\mu\mathrm{K}^2\),
\(\alpha_D=-0.16\) at \(\ell_0=80\).

**Units:** Convert to **dimensionless** \(C_\ell\)
\(C_\ell^{\mathrm{dimless}} = C_\ell^{(\mu\mathrm{K}^2)} / T_{\mathrm{CMB}}^2\) with
\(T_{\mathrm{CMB}}\) in \(\mu\mathrm{K}\) (FIRAS: \(T_{\mathrm{CMB}} \approx 2.72548\times 10^6\,\mu\mathrm{K}\)).
The dust residual amplitude passed into ``dust_cl_azzoni`` is therefore
``AZZONI_A_D_DIMENSIONLESS`` below, not raw \(\mu\mathrm{K}^2\).
"""

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


def dust_cl_azzoni(ell: np.ndarray, A_D: float, alpha_D: float, ell0: float, c_f: float) -> np.ndarray:
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
    ell: np.ndarray,
    fwhm_deg: float,
    fnl_fid: float,
    ns_fid: float,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    *,
    w_mu_inv: float,
    c_f: float,
    A_D: float,
    alpha_D: float = AZZONI_ALPHA_D,
    ell0: float = ELL0_AZZONI,
    As_fid: float = AS_FID_PLANCK2018,
    dns_step: float = 5e-5,
    sigma_ns_prior: float | None = 0.004,
    sigma_AD_prior: float | None = None,
    sigma_alpha_prior: float | None = None,
    use_b_analytic: bool = False,
    b_integral_kw: dict | None = None,
    cl_tt_txt_dir: str | None = None,
    marginalize_dust: bool = True,
) -> ForegroundFisherResult:
    """
    Fisher for ``(f_NL, n_s)`` plus, if ``marginalize_dust``, nuisance amplitudes ``(A_D, \\alpha_D)``.

    Dust residual enters the band-power **variance** only. With ``marginalize_dust=True`` (default),
    the Fisher is 4×4 and foreground amplitudes are marginalized (with optional priors). With
    ``marginalize_dust=False``, only ``(f_NL, n_s)`` are free: residual dust is fixed at the
    fiducial ``(A_D, \\alpha_D, c_f)`` (no uncertainty on those nuisance parameters).

    ``cl_tt_txt_dir`` :
        If set, load fiducial CAMB ``C_\ell^{TT}`` from ``spectra.load_ClTT_planck18`` in that
        directory (same convention as ``fisher_muT_general``).
    """
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
