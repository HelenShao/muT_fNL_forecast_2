r"""Fisher matrix for C_l^{\mu T}: f_NL, n_s, optional A_s.

Supports multiple parameter configs (1x1, 2x2, 3x3, ...).

Gaussian Fisher for band powers (Tier A): weights 1/sigma_ell^2 evaluated at fiducial cosmology
(``variance_at_fiducial=True``). Optional Tier B (full Gaussian Fisher including dsigma/d\theta) is
documented in module comments -- diagonal multipoles would add
  F_ij^cov = (1/2) sigma_ell sigma_ell^{-4} (dsigma_ell^2/d\theta_i)(dsigma_ell^2/d\theta_j).
"""

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
    var: np.ndarray,
    dsigma2: dict[str, np.ndarray],
    param_names: tuple[str, ...],
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
    ell: np.ndarray,
    variance_mode: str,
    *,
    cl_tt: np.ndarray,
    fwhm_deg: float,
    w_mu_inv: float,
    fnl_fid: float,
    b_arr: np.ndarray,
    As_fid: float,
    k_D_i: float,
    k_D_f: float,
    ns_fid: float,
    k_p: float,
    cl_tt_noise: np.ndarray | float,
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
    ell: np.ndarray,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    dns_step,
    use_b_analytic: bool,
    b_kw: dict[str, Any],
    b_override: float | None = None,
    *,
    b_db_prec: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray | float]:
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
    ell: np.ndarray,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    dns_step,
    dkdf_step: float,
    use_b_analytic: bool,
    b_kw: dict[str, Any],
    *,
    As_fid,
    include_As: bool,
    include_k_Df: bool,
    b_override: float | None = None,
    b_db_prec: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    r"""
    Rows = multipoles, columns = (dC_l^{\mu T}/d\theta_i).

    C_l^{\mu T} = f_NL * b * T_l(A_s, k_D); evaluated at As_fid for K columns.
    dC/dA_s = 2 * C / A_s at fixed f_NL, b, k_D.

    For k_{D,f}: use analytic `\partial b/\partial k_{D,f}` when ``use_b_analytic``;
    otherwise ``db_dkdf_central``
    """
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
    ell: np.ndarray,
    fwhm_deg,
    fnl_fid,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    w_mu_inv: float = W_MU_INV_PIXIE,
    dns_step: float = 5e-5,
    dkdf_step: float = 0.5,
    sigma_ns_prior: float | None = 0.004,
    sigma_As_prior: float | None = None,
    sigma_k_Df_prior: float | None = None,
    include_As: bool | None = None,
    include_k_Df: bool = False,
    As_fid: float = AS_FID_LEGACY,
    use_b_analytic: bool = False,
    b_override: float | None = None,
    b_integral_kw: dict[str, Any] | None = None,
    variance_at_fiducial: bool = True,
    cl_tt_txt_dir: str | None = None,
    variance_mode: str = VARIANCE_PZ_INSTRUMENTAL_APPROX,
    cl_tt_noise: np.ndarray | float = 0.0,
    include_covariance_derivative: bool = False,
    dsigma2_wrt: dict[str, np.ndarray] | None = None,
    b_db_prec: tuple[np.ndarray, np.ndarray] | None = None,
) -> FisherMuTResult:
    r"""
    Gaussian Fisher for muT band powers.

    **Mean term:** :math:`F^{\rm data}_{ij} = \sum_\ell \sigma_\ell^{-2}
    (\partial C_\ell^{\mu T}/\partial\theta_i)(\partial C_\ell^{\mu T}/\partial\theta_j)` + priors.

    **Variance:** ``variance_mode`` selects PZ instrumental approximation
    (:math:`\sigma_\ell^2 \simeq C_\ell^{TT} C_\ell^{\mu\mu,N}/(2\ell+1)`) or the full Gaussian
    band-power form via ``spectra.sigma2_muT_hat_full`` (CV-limited or with instrumental
    :math:`C_\ell^{\mu\mu,N}`).

    **Covariance derivative term:** if ``include_covariance_derivative`` and ``dsigma2_wrt`` are set,
    adds ``fisher_cov_term_diagonal`` to ``F_total`` (diagonal :math:`C` in :math:`\ell`).

    ----------
    cl_tt_txt_dir :
        If not ``None``, load CAMB ``C_l^{TT}`` from text files in this directory
        (see ``planck_cosmology.save_planck2018_cltt_bundle`` / ``spectra.load_ClTT_planck18``).
        Values are assumed to already be in the same dimensionless normalization used by
        ``spectra.Cl_TT`` and are passed directly to ``sigma2_muT_hat``.

    sigma_As_prior :
        If not None, add ``1/sigma^2`` to the A_s diagonal of ``F_total``. Use
        `SIGMA_AS_PLANCK2018` (Planck18 sigma(ln(10^10 A_s)) = 0.014)
    include_As :
        If True, include A_s as a third parameter (K column dC/dA_s)
    include_k_Df :
        If True, include k_{D,f} (Mpc^{-1}) as a nuisance parameter. Use analytic
        d b / d k_{D,f} when ``use_b_analytic``; otherwise ``b_integral.db_dkdf_central``
        (finite difference in ``k_{D,f}`` with step ``dkdf_step``).
    dkdf_step :
        Half-step (Mpc^{-1}) for ``db_dkdf_central`` when ``use_b_analytic`` is False.
    sigma_k_Df_prior :
        If not None, add ``1/\sigma^2`` on the ``k_Df`` diagonal of ``F_total``.
    As_fid :
        Fiducial primordial amplitude at the pivot used in the PZ template (matches legacy TT scale when default).
    w_mu_inv :
        ``w_\mu^{-1}`` in ``C_l^{\mu\mu,N}`` (see ``beam.N_mu_mu``). Use ``beam.W_MU_INV_PIXIE`` or
        ``beam.W_MU_INV_SPECTER`` for PIXIE vs SPECTER. For ``full_gaussian_cv``, this should be ``0``.
    b_override :
        If not ``None``, fix ``b`` to this constant at all multipoles and set
        ``\partial b / \partial n_s = 0`` (and ``\partial b / \partial k_{D,f} = 0`` if included).
        This is useful for fixed-``b`` sensitivity tests.
    variance_at_fiducial :
    variance_mode :
        ``pz_instrumental_approx`` (default), ``full_gaussian_cv``, or ``full_gaussian_noisy``.
    cl_tt_noise :
        :math:`N_\ell^{TT}` for ``sigma2_muT_hat_full`` (scalar or per-:math:`\ell`).
    dsigma2_wrt :
        Maps parameter name to :math:`\partial\sigma_\ell^2/\partial\theta` (length ``len(ell)``).
    """
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

    def _sigma_u(j: int) -> float:
        return 1.0 / np.sqrt(F_data[j, j]) if F_data[j, j] > 0 else np.inf

    def _sigma_m(j: int) -> float:
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
    ell: np.ndarray,
    fwhm_deg,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    *,
    w_mu_inv: float = W_MU_INV_PIXIE,
    As_fid: float = AS_FID_LEGACY,
    use_b_analytic: bool = True,
    b_integral_kw: dict[str, Any] | None = None,
    b_override: float | None = None,
    cl_tt_txt_dir: str | None = None,
    variance_mode: str = VARIANCE_PZ_INSTRUMENTAL_APPROX,
    fnl_fid_for_variance: float = 0.0,
    cl_tt_noise: np.ndarray | float = 0.0,
) -> float:
    """Single-parameter Fisher F = sum_ell (dC/df_NL)^2/sigma_ell^2.

    ``fnl_fid_for_variance`` sets :math:`C_\ell^{\mu T}` inside :math:`\sigma_\ell^2` when using
    ``full_gaussian_*`` modes (ignored for ``pz_instrumental_approx`` in practice).

    If ``b_override`` is set (e.g. ``1.0`` for a simplified tutorial comparison), use constant
    `b` at every multipole and ignore ``use_b_analytic`` / numerical ``b_ell_ns``.
    """
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


def default_ell_grid(fwhm_deg, ell_min: int = 2) -> np.ndarray:
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
