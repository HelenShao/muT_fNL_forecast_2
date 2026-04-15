r"""Fisher matrix for C_l^{\mu T}: f_NL, n_s, optional A_s.

Supports multiple parameter configs (1x1, 2x2, 3x3, ...).

Gaussian Fisher for band powers (Tier A): weights 1/sigma_ell^2 evaluated at fiducial cosmology
(``variance_at_fiducial=True``). Optional Tier B (full Gaussian Fisher including dsigma/d\theta) is
documented in module comments -- diagonal multipoles would add
  F_ij^cov = (1/2) sigma_ell sigma_ell^{-4} (dsigma_ell^2/d\theta_i)(dsigma_ell^2/d\theta_j).
"""

from __future__ import annotations

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
        SIGMA_AS_PLANCK2018,
        Cl_TT,
        T_muT_ell,
        dT_muT_ell_dkdf,
        sigma2_muT_hat,
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
        SIGMA_AS_PLANCK2018,
        Cl_TT,
        T_muT_ell,
        dT_muT_ell_dkdf,
        sigma2_muT_hat,
    )


@dataclass
class FisherMuTResult:
    """Fisher matrices and marginalized uncertainties."""

    F_data: np.ndarray
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


def _b_and_db(
    ell: np.ndarray,
    ns_fid,
    k_D_i,
    k_D_f,
    k_p,
    dns_step,
    use_b_analytic: bool,
    b_kw: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray | float]:
    ell_i = ell.astype(int)
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
        ell, ns_fid, k_D_i, k_D_f, k_p, dns_step, use_b_analytic, b_kw
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
        if use_b_analytic:
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
    b_integral_kw: dict[str, Any] | None = None,
    variance_at_fiducial: bool = True,
) -> FisherMuTResult:
    r"""
    Gaussian Fisher for muT band powers (mean term only; Tier A variance).

        F_ij = sigma_ell sigma_ell^{-2} (dC_l/d\theta_i)(dC_l/d\theta_j)  +  priors

    with sigma_ell^2 \simeq C_l^{TT} C_l^{\mu\mu,N}/(2l+1). Uses ``Cl_TT(ell, As_fid)`` for sigma_ell^2 when
    ``variance_at_fiducial`` is True (fixed noise at fiducial A_s).

    ----------
    sigma_As_prior :
        If not None, add ``1/sigma^2`` to the A_s diagonal of ``F_total``. Use
        `SIGMA_AS_PLANCK2018` (Planck18 sigma(ln(10^10 A_s)) = 0.014)
    include_As :
        If True, include A_s as a third parameter (K column dC/dA_s)
    include_k_Df :
        If True, include k_{D,f} (Mpc^{-1}) as a nuisance parameter. Use analytic
        `\partial b/\partial k_{D,f}` when ``use_b_analytic``; otherwise
        :func:`b_integral.db_dkdf_central` (finite difference in ``k_{D,f}`` with step ``dkdf_step``).
    dkdf_step :
        Half-step (Mpc^{-1}) for ``db_dkdf_central`` when ``use_b_analytic`` is False.
    sigma_k_Df_prior :
        If not None, add ``1/\sigma^2`` on the ``k_Df`` diagonal of ``F_total``.
    As_fid :
        Fiducial primordial amplitude at the pivot used in the PZ template (matches legacy TT scale when default).
    w_mu_inv :
        ``w_\mu^{-1}`` in ``C_l^{\mu\mu,N}`` (see ``beam.N_mu_mu``). Use ``beam.W_MU_INV_PIXIE`` or
        ``beam.W_MU_INV_SPECTER`` for PIXIE vs SPECTER.
    variance_at_fiducial :
        Tier A (default): evaluate sigma_ell^2 at ``As_fid`` only. Tier B (optional future): add
        (1/2) sigma_ell sigma_ell^{-4} (dsigma_ell^2/d\theta_i)(dsigma_ell^2/d\theta_j) when ``variance_at_fiducial`` is False --
        not implemented yet; throws warning and uses fiducial sigma_ell^2.
    """
    if not variance_at_fiducial:
        warnings.warn(
            "Tier B Fisher (derivative of covariance w.r.t. parameters) is not implemented; "
            "using fiducial sigma_ell^2 as in Tier A.",
            UserWarning,
            stacklevel=2,
        )

    b_kw = dict(b_integral_kw or {})
    if include_As is None:
        include_As = sigma_As_prior is not None

    cl_tt = Cl_TT(ell, As_fid)
    n_mumu = N_mu_mu(ell, fwhm_deg, w_mu_inv=w_mu_inv)
    var = sigma2_muT_hat(ell, cl_tt, n_mumu)
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
    )

    F_data = (K * inv_var[:, None]).T @ K

    F_total = F_data.copy()
    idx = {n: i for i, n in enumerate(param_names)}
    if sigma_ns_prior is not None:
        F_total[idx["ns"], idx["ns"]] += 1.0 / (sigma_ns_prior**2)
    if "As" in idx and sigma_As_prior is not None:
        F_total[idx["As"], idx["As"]] += 1.0 / (sigma_As_prior**2)
    if "k_Df" in idx and sigma_k_Df_prior is not None:
        F_total[idx["k_Df"], idx["k_Df"]] += 1.0 / (sigma_k_Df_prior**2)

    cov = np.linalg.inv(F_total)

    def _sigma_u(j: int) -> float:
        return 1.0 / np.sqrt(F_data[j, j]) if F_data[j, j] > 0 else np.inf

    def _sigma_m(j: int) -> float:
        return float(np.sqrt(cov[j, j]))

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
) -> float:
    """Single-parameter Fisher F = sigma_ell (dC/df_NL)^2/sigma_ell^2; sigma_ell^2 uses ``Cl_TT(As_fid)``.

    If ``b_override`` is set (e.g. ``1.0`` for a simplified tutorial comparison), use constant
    `b` at every multipole and ignore ``use_b_analytic`` / numerical ``b_ell_ns``.
    """
    b_kw = dict(b_integral_kw or {})
    cl_tt = Cl_TT(ell, As_fid)
    n_mumu = N_mu_mu(ell, fwhm_deg, w_mu_inv=w_mu_inv)
    var = sigma2_muT_hat(ell, cl_tt, n_mumu)
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
    print("b=", b*(ell*(ell+1)))
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
    "W_MU_INV_PIXIE",
    "W_MU_INV_SPECTER",
    "fisher_muT_general",
    "fisher_1d_fnl_only",
    "default_ell_grid",
]
