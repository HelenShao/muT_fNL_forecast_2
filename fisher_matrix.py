'Fisher matrix for C_l^{\\mu T}: f_NL, n_s, optional A_s.'

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from .beam import N_mu_mu, W_MU_INV_PIXIE, W_MU_INV_SPECTER, ell_max_from_fwhm_deg
    from .b_integral import (
        b_analytic,
        b_ell_ns,
        db_dkdf_analytic,
        db_dkdf_central,
        db_dns_central,
    )
    from .spectra import (
        AS_FID_LEGACY,
        AS_FID_PLANCK2018,
        SIGMA_AS_PLANCK2018,
        Cl_TT,
        Cl_muT,
        Cl_mu_mu_gaussian_PZ,
        T_muT_ell,
        cl_tt_on_ell_grid,
        dT_muT_ell_dkdf,
        load_ClTT_planck18,
        sigma2_muT_hat,
        sigma2_muT_hat_full,
    )
except ImportError:
    from beam import N_mu_mu, W_MU_INV_PIXIE, W_MU_INV_SPECTER, ell_max_from_fwhm_deg
    from b_integral import (
        b_analytic,
        b_ell_ns,
        db_dkdf_analytic,
        db_dkdf_central,
        db_dns_central,
    )
    from spectra import (
        AS_FID_LEGACY,
        AS_FID_PLANCK2018,
        SIGMA_AS_PLANCK2018,
        Cl_TT,
        Cl_muT,
        Cl_mu_mu_gaussian_PZ,
        T_muT_ell,
        cl_tt_on_ell_grid,
        dT_muT_ell_dkdf,
        load_ClTT_planck18,
        sigma2_muT_hat,
        sigma2_muT_hat_full,
    )


# Band-power variance models 
VARIANCE_PZ_INSTRUMENTAL_APPROX = "pz_instrumental_approx"
VARIANCE_FULL_GAUSSIAN_CV = "full_gaussian_cv"
VARIANCE_FULL_GAUSSIAN_NOISY = "full_gaussian_noisy"


@dataclass
class FisherMuTResult:
    """Fisher matrices and marginalized uncertainties."""

    F_data: np.ndarray
    F_cov: np.ndarray
    F_total: np.ndarray
    param_names: tuple[str, ...]
    sigma_fnl_unmarg: float
    sigma_fnl_marg: float
    sigma_ns_unmarg: float
    sigma_ns_marg: float
    corr_fnl_ns: float
    cov_marginal: np.ndarray
    sigma_As_unmarg: float | None = None
    sigma_As_marg: float | None = None
    sigma_k_Df_unmarg: float | None = None
    sigma_k_Df_marg: float | None = None


def fisher_cov_term_diagonal(
    var,
    dsigma2,
    param_names,
):
    n = len(param_names)
    F = np.zeros((n, n), dtype=float)
    inv_v4 = 1.0 / (var**2)
    for i, pi in enumerate(param_names):
        if pi not in dsigma2:
            continue
        di = np.asarray(dsigma2[pi], dtype=float)
        for j, pj in enumerate(param_names):
            if pj not in dsigma2:
                continue
            dj = np.asarray(dsigma2[pj], dtype=float)
            F[i, j] = 0.5 * float(np.sum(inv_v4 * di * dj))
    return F


def _muT_bandpower_variance(
    ell,
    variance_mode,
    *,
    cl_tt,
    fwhm_deg,
    w_mu_inv,
    fnl_fid,
    b_arr,
    As_fid,
    k_D_i,
    k_D_f,
    ns_fid,
    k_p,
    cl_tt_noise,
):
    if variance_mode == VARIANCE_PZ_INSTRUMENTAL_APPROX:
        n_mumu = N_mu_mu(ell, fwhm_deg, w_mu_inv=w_mu_inv)
        return sigma2_muT_hat(ell, cl_tt, n_mumu)
    cl_mut = Cl_muT(ell, fnl_fid, b_arr, As_fid, k_D_i, k_D_f)
    cl_mumu_sig = Cl_mu_mu_gaussian_PZ(ell, k_D_f=k_D_f, ns=ns_fid, k_p=k_p)
    if variance_mode == VARIANCE_FULL_GAUSSIAN_CV:
        n_mumu = np.zeros_like(ell, dtype=float)
    elif variance_mode == VARIANCE_FULL_GAUSSIAN_NOISY:
        n_mumu = N_mu_mu(ell, fwhm_deg, w_mu_inv=w_mu_inv)
    else:
        raise ValueError(
            f"variance_mode must be one of {VARIANCE_PZ_INSTRUMENTAL_APPROX!r}, "
            f"{VARIANCE_FULL_GAUSSIAN_CV!r}, {VARIANCE_FULL_GAUSSIAN_NOISY!r}; got {variance_mode!r}"
        )
    return sigma2_muT_hat_full(
        ell,
        cl_tt,
        cl_tt_noise=cl_tt_noise,
        cl_mumu_signal=cl_mumu_sig,
        cl_mumu_noise=n_mumu,
        cl_mut=cl_mut,
    )


def _b_and_db(
    ell,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    dns_step,
    use_b_analytic,
    b_kw,
    b_override = None,
    *,
    b_db_prec = None,
):
    ell_i = ell.astype(int)
    if b_db_prec is not None:
        b_arr, db_arr = b_db_prec
        if b_arr.shape != ell.shape or db_arr.shape != ell.shape:
            raise ValueError("b_db_prec arrays must match ell shape")
        return b_arr, db_arr
    if b_override is not None:
        b = np.full_like(ell, float(b_override), dtype=float)
        db = np.zeros_like(ell, dtype=float)
        return b, db
    if use_b_analytic:
        b0 = b_analytic(ns_fid, k_D_i, k_D_f, k_p)
        db_dns = 0.5 * np.log((k_D_i * k_D_f) / (4.0 * k_p**2))
        b = np.full_like(ell, b0, dtype=float)
        return b, db_dns
    b = np.array(
        [b_ell_ns(int(l), ns_fid, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw) for l in ell_i]
    )
    db = np.array(
        [
            db_dns_central(int(l), ns_fid, dns_step, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw)
            for l in ell_i
        ]
    )
    return b, db


def _Cl_derivative_matrix(
    ell,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    dns_step,
    dkdf_step,
    use_b_analytic,
    b_kw,
    *,
    As_fid,
    include_As,
    include_k_Df,
    b_override = None,
    b_db_prec = None,
):
    r"""Build the derivative matrix of muT spectra with respect to model parameters."""
    ell_i = ell.astype(int)
    b, db_or_scalar = _b_and_db(
        ell,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        dns_step,
        use_b_analytic,
        b_kw,
        b_override=b_override,
        b_db_prec=b_db_prec,
    )
    T = T_muT_ell(ell, As_fid, k_D_i, k_D_f)

    K_fnl = T * b
    K_ns = fnl_fid * T * db_or_scalar

    cols = [K_fnl, K_ns]
    names: list[str] = ["fnl", "ns"]

    if include_As:
        K_As = K_fnl * (2.0 / As_fid)
        cols.append(K_As)
        names.append("As")

    if include_k_Df:
        dT_dk = dT_muT_ell_dkdf(ell, As_fid, k_D_i, k_D_f)
        if b_override is not None:
            db_dk_arr = np.zeros_like(ell, dtype=float)
        elif use_b_analytic:
            db_dk = db_dkdf_analytic(ns_fid, k_D_f)
            db_dk_arr = np.full_like(ell, db_dk, dtype=float)
        else:
            db_dk_arr = np.array(
                [
                    db_dkdf_central(
                        int(l),
                        ns_fid,
                        dkdf_step,
                        k_D_i=k_D_i,
                        k_D_f=k_D_f,
                        k_p=k_p,
                        **b_kw,
                    )
                    for l in ell_i
                ]
            )
        K_kdf = fnl_fid * (b * dT_dk + T * db_dk_arr)
        cols.append(K_kdf)
        names.append("k_Df")

    K = np.column_stack(cols)
    return K, tuple(names)


def fisher_muT_general(
    ell,
    fwhm_deg,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    w_mu_inv = W_MU_INV_PIXIE,
    dns_step = 5e-5,
    dkdf_step = 0.5,
    sigma_ns_prior = 0.004,
    sigma_As_prior = None,
    sigma_k_Df_prior = None,
    include_As = None,
    include_k_Df = False,
    As_fid = AS_FID_LEGACY,
    use_b_analytic = False,
    b_override = None,
    b_integral_kw = None,
    variance_at_fiducial = True,
    cl_tt_txt_dir = None,
    variance_mode = VARIANCE_PZ_INSTRUMENTAL_APPROX,
    cl_tt_noise = 0.0,
    include_covariance_derivative = False,
    dsigma2_wrt = None,
    b_db_prec = None,
):
    r"""Compute the generalized Gaussian Fisher matrix for the muT band-power model."""
    if not variance_at_fiducial and not include_covariance_derivative:
        warnings.warn(
            "variance_at_fiducial=False without include_covariance_derivative: still using "
            "fiducial sigma_ell^2 only (legacy Tier A).",
            UserWarning,
            stacklevel=2,
        )

    b_kw = dict(b_integral_kw or {})
    if include_As is None:
        include_As = sigma_As_prior is not None

    if cl_tt_txt_dir is not None:
        _bundle = load_ClTT_planck18(cl_tt_txt_dir)
        cl_tt = cl_tt_on_ell_grid(_bundle["fiducial"], ell)
    else:
        cl_tt = Cl_TT(ell, As_fid)

    b_arr, _ = _b_and_db(
        ell,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        dns_step,
        use_b_analytic,
        b_kw,
        b_override=b_override,
        b_db_prec=b_db_prec,
    )
    var = _muT_bandpower_variance(
        ell,
        variance_mode,
        cl_tt=cl_tt,
        fwhm_deg=fwhm_deg,
        w_mu_inv=w_mu_inv,
        fnl_fid=fnl_fid,
        b_arr=b_arr,
        As_fid=As_fid,
        k_D_i=k_D_i,
        k_D_f=k_D_f,
        ns_fid=ns_fid,
        k_p=k_p,
        cl_tt_noise=cl_tt_noise,
    )
    inv_var = 1.0 / var

    K, param_names = _Cl_derivative_matrix(
        ell,
        fnl_fid,
        ns_fid,
        k_D_i,
        k_D_f,
        k_p,
        dns_step,
        dkdf_step,
        use_b_analytic,
        b_kw,
        As_fid=As_fid,
        include_As=include_As,
        include_k_Df=include_k_Df,
        b_override=b_override,
        b_db_prec=b_db_prec,
    )

    F_data = (K * inv_var[:, None]).T @ K

    if include_covariance_derivative and dsigma2_wrt:
        F_cov = fisher_cov_term_diagonal(var, dsigma2_wrt, param_names)
    else:
        F_cov = np.zeros_like(F_data)

    F_total = F_data + F_cov
    idx = {n: i for i, n in enumerate(param_names)}
    if sigma_ns_prior is not None:
        F_total[idx["ns"], idx["ns"]] += 1.0 / (sigma_ns_prior**2)
    if "As" in idx and sigma_As_prior is not None:
        F_total[idx["As"], idx["As"]] += 1.0 / (sigma_As_prior**2)
    if "k_Df" in idx and sigma_k_Df_prior is not None:
        F_total[idx["k_Df"], idx["k_Df"]] += 1.0 / (sigma_k_Df_prior**2)

    try:
        cov = np.linalg.inv(F_total)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(F_total)
        warnings.warn(
            "F_total is singular; using pseudoinverse for marginal uncertainties.",
            UserWarning,
            stacklevel=2,
        )

    def _sigma_u(j):
        return 1.0 / np.sqrt(F_data[j, j]) if F_data[j, j] > 0 else np.inf

    def _sigma_m(j):
        v = float(cov[j, j])
        if not math.isfinite(v) or v <= 0.0:
            return float("inf")
        return float(np.sqrt(v))

    sigma_fnl_u = _sigma_u(idx["fnl"])
    sigma_ns_u = _sigma_u(idx["ns"])

    sigma_As_u: float | None = None
    sigma_As_m: float | None = None
    if "As" in idx:
        j = idx["As"]
        sigma_As_u = _sigma_u(j)
        sigma_As_m = _sigma_m(j)

    sigma_k_Df_u: float | None = None
    sigma_k_Df_m: float | None = None
    if "k_Df" in idx:
        j = idx["k_Df"]
        sigma_k_Df_u = _sigma_u(j)
        sigma_k_Df_m = _sigma_m(j)

    i0, i1 = idx["fnl"], idx["ns"]
    corr = (
        float(cov[i0, i1] / np.sqrt(cov[i0, i0] * cov[i1, i1]))
        if cov[i0, i0] > 0 and cov[i1, i1] > 0
        else 0.0
    )

    # pack results into dataclass
    return FisherMuTResult(
        F_data=F_data,
        F_cov=F_cov,
        F_total=F_total,
        param_names=param_names,
        sigma_fnl_unmarg=sigma_fnl_u,
        sigma_fnl_marg=_sigma_m(idx["fnl"]),
        sigma_ns_unmarg=sigma_ns_u,
        sigma_ns_marg=_sigma_m(idx["ns"]),
        corr_fnl_ns=corr,
        cov_marginal=cov,
        sigma_As_unmarg=sigma_As_u,
        sigma_As_marg=sigma_As_m,
        sigma_k_Df_unmarg=sigma_k_Df_u,
        sigma_k_Df_marg=sigma_k_Df_m,
    )


def fisher_1d_fnl_only(
    ell,
    fwhm_deg,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    w_mu_inv = W_MU_INV_PIXIE,
    As_fid = AS_FID_LEGACY,
    use_b_analytic = True,
    b_integral_kw = None,
    b_override = None,
    cl_tt_txt_dir = None,
    variance_mode = VARIANCE_PZ_INSTRUMENTAL_APPROX,
    fnl_fid_for_variance = 0.0,
    cl_tt_noise = 0.0,
):
    """Compute the single-parameter Fisher information for f_NL under the selected variance model."""
    b_kw = dict(b_integral_kw or {})
    if cl_tt_txt_dir is not None:
        _bundle = load_ClTT_planck18(cl_tt_txt_dir)
        cl_tt = cl_tt_on_ell_grid(_bundle["fiducial"], ell)
    else:
        cl_tt = Cl_TT(ell, As_fid)
    ell_i = ell.astype(int)
    T = T_muT_ell(ell, As_fid, k_D_i, k_D_f)
    if b_override is not None:
        b = np.full_like(ell, float(b_override), dtype=float)
    elif use_b_analytic:
        b0 = b_analytic(ns_fid, k_D_i, k_D_f, k_p)
        b = np.full_like(ell, b0, dtype=float)
    else:
        b = np.array(
            [b_ell_ns(int(l), ns_fid, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw) for l in ell_i]
        )
    var = _muT_bandpower_variance(
        ell,
        variance_mode,
        cl_tt=cl_tt,
        fwhm_deg=fwhm_deg,
        w_mu_inv=w_mu_inv,
        fnl_fid=fnl_fid_for_variance,
        b_arr=b,
        As_fid=As_fid,
        k_D_i=k_D_i,
        k_D_f=k_D_f,
        ns_fid=ns_fid,
        k_p=k_p,
        cl_tt_noise=cl_tt_noise,
    )
    K_fnl = T * b
    return float(np.sum(K_fnl**2 / var))


def default_ell_grid(fwhm_deg, ell_min = 2):
    lmax = ell_max_from_fwhm_deg(fwhm_deg)
    return np.arange(ell_min, int(np.ceil(lmax)) + 1, dtype=float)


# Re-export for callers that want Planck-style default prior width
__all__ = [
    "FisherMuTResult",
    "SIGMA_AS_PLANCK2018",
    "AS_FID_LEGACY",
    "AS_FID_PLANCK2018",
    "W_MU_INV_PIXIE",
    "W_MU_INV_SPECTER",
    "VARIANCE_PZ_INSTRUMENTAL_APPROX",
    "VARIANCE_FULL_GAUSSIAN_CV",
    "VARIANCE_FULL_GAUSSIAN_NOISY",
    "fisher_cov_term_diagonal",
    "fisher_muT_general",
    "fisher_1d_fnl_only",
    "default_ell_grid",
]
