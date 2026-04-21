r"""Fisher for C_l^{\mu T}: f_NL, n_s, optional A_s \equiv \Delta_R^2(k_p); optional Planck priors (see fisher_matrix.py)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import spherical_jn

# ------------------------------------------------------------
# Cross spectra (PZ eq. 163 TT; eq. 186 \mu T)
K_TT_SW = 2.0 * np.pi / 25.0
AS_FID_LEGACY = 6e-10 / K_TT_SW
AS_FID_PLANCK2018 = 2.1e-9
SIGMA_AS_PLANCK2018 = AS_FID_PLANCK2018 * 0.014
PZ_MUT_PREFACTOR = 6.1 * np.pi * (9.0 / 25.0)


def Cl_TT(ell: np.ndarray, A_s: float | None = None) -> np.ndarray:
    r"""C_l^{TT} = (2\pi/25) A_s/(l(l+1)); None -> AS_FID_LEGACY matches legacy 6e-10 scaling."""
    if A_s is None:
        A_s = AS_FID_LEGACY
    return K_TT_SW * A_s / (ell * (ell + 1.0))


def T_muT_ell(ell: np.ndarray, A_s: float, k_D_i: float, k_D_f: float) -> np.ndarray:
    r"""PZ eq. (186): T_l = 6.1\pi (9/25) ln(k_D,i/k_D,f) A_s^2 / (l(l+1))."""
    L = np.log(k_D_i / k_D_f)
    return PZ_MUT_PREFACTOR * L * (A_s**2) / (ell * (ell + 1.0))


def sigma2_muT_hat(ell: np.ndarray, cl_tt: np.ndarray, n_mumu: np.ndarray) -> np.ndarray:
    r"""PZ approx: \mathrm{Var}(\hat{C}_l^{\mu T}) \simeq C_l^{TT} C_l^{\mu\mu,N} / (2l+1)"""
    return (cl_tt * n_mumu) / (2.0 * ell + 1.0)


# ------------------------------------------------------------
# Noise model (deconvolved mu noise)

def ell_max_from_fwhm_deg(fwhm_deg: float) -> float:
    """Gaussian beam: l_max = sqrt(8 ln 2) / FWHM (radians)."""
    return float(np.sqrt(8.0 * np.log(2.0)) / np.deg2rad(fwhm_deg))


def N_mu_mu(ell: np.ndarray, fwhm_deg: float, w_mu_inv: float = 1.3e-15) -> np.ndarray:
    r"""
    deconvolved mu autospectrum noise (dimensionless C_l units).
    C_l^{\mu\mu,N} \simeq w_\mu^{-1} * exp(+l^2/l_max^2).
    ** using PIXIE w_\mu^{-1/2} = 1.3e-15 **
    """
    lmax = ell_max_from_fwhm_deg(fwhm_deg)
    return w_mu_inv * np.exp(ell**2 / lmax**2)


# ------------------------------------------------------------
# b(l, n_s) factorized integrals and discrete deriv
r"""
Factorized b(l, n_s) for squeezed limit, with split damping window:
b/(l(l+1)) \approx (2/L) I_+(l) I_-(ns-1),
with I_+ from the k_+ Bessel integral and I_- from the k_- damping integrals.

Tilt in I_- uses (k_-/(2 k_p))(ns-1).
"""

def log_k_D_ratio(k_D_i: float, k_D_f: float) -> float:
    return float(np.log(k_D_i / k_D_f))


def b_analytic(ns: float, k_D_i: float, k_D_f: float, k_p: float) -> float:
    r"""Leading PZ approximation: b \simeq 1 + (n_s-1)/2 * ln(k_D,i k_D,f / (4 k_p^2))."""
    eps = ns - 1.0
    return 1.0 + 0.5 * eps * np.log((k_D_i * k_D_f) / (4.0 * k_p**2))


def I_plus(ell: int, r_L_mpc: float, k_plus: np.ndarray) -> float:
    r"""
    numerically integrate
    I_+ = \int dk_+ j_l^2(k_+ r_L) (measure from \int d ln k_+ k_+ j_l^2 -> \int dk_+ j_l^2)."""
    jl = spherical_jn(ell, k_plus * r_L_mpc)
    return float(np.trapz(jl**2, k_plus))


def I_minus(
    eps: float,
    k_p: float,
    k_D_i: float,
    k_D_f: float,
    k_minus: np.ndarray,
) -> float:
    """
    numerically integrate:
    I_- = I_i - I_f 
    I^(i) = int_0^infty dk_- (k_-/(2k_p))^(ns-1) exp(-k_-^2/(2 k_D,i^2)),
    and similarly for k_D,f (PZ bracket form)
    """
    t = (k_minus / (2.0 * k_p)) ** eps
    I_i = np.trapz(t * np.exp(-(k_minus**2) / (2.0 * k_D_i**2)), k_minus)
    I_f = np.trapz(t * np.exp(-(k_minus**2) / (2.0 * k_D_f**2)), k_minus)
    return float(I_i - I_f)


def F_b_factor(ell: int, ns: float, *, L: float, k_p: float, k_D_i: float, k_D_f: float, r_L_mpc: float, k_plus: np.ndarray, k_minus: np.ndarray) -> float:
    """F(l, n_s) = l(l+1)(2/L) I_+ * I_- used to get b"""
    eps = ns - 1.0
    Ip = I_plus(ell, r_L_mpc, k_plus)
    Im = I_minus(eps, k_p, k_D_i, k_D_f, k_minus)
    return float(ell * (ell + 1.0) * (2.0 / L) * Ip * Im)


def b_ell_ns(
    ell: int,
    ns: float,
    *,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    r_L_mpc: float = 14000.0,
    k_plus_grid: np.ndarray | None = None,
    k_minus_grid: np.ndarray | None = None,
    ell_ref: int = 50,
    ns_ref: float = 0.965,
) -> float:
    """
    numerically integrate to get b(l, n_s), use factorized integrals:
        b(l, n_s) = b_analytic(n_{s,ref}) * F(l, n_s) / F(l_ref, n_{s,ref})
    """
    if k_plus_grid is None:
        k_plus_grid = np.logspace(-5.0, 0.0, 400)
    if k_minus_grid is None:
        k_minus_grid = np.logspace(-3.0, np.log10(5.0e5), 800)

    L = log_k_D_ratio(k_D_i, k_D_f)
    F = F_b_factor(ell, ns, L=L, k_p=k_p, k_D_i=k_D_i, k_D_f=k_D_f, r_L_mpc=r_L_mpc, k_plus=k_plus_grid, k_minus=k_minus_grid)
    F_ref = F_b_factor(
        ell_ref,
        ns_ref,
        L=L,
        k_p=k_p,
        k_D_i=k_D_i,
        k_D_f=k_D_f,
        r_L_mpc=r_L_mpc,
        k_plus=k_plus_grid,
        k_minus=k_minus_grid,
    )
    if F_ref == 0.0:
        return b_analytic(ns, k_D_i, k_D_f, k_p)
    return float(b_analytic(ns_ref, k_D_i, k_D_f, k_p) * F / F_ref)


def db_dns_central(
    ell: int,
    ns: float,
    h: float,
    *,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    **kwargs,
) -> float:
    """
    db/dn_s: discrete derivative, h small
    """
    return (
        b_ell_ns(ell, ns + h, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **kwargs)
        - b_ell_ns(ell, ns - h, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **kwargs)
    ) / (2.0 * h)

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


def _b_and_db(
    ell: np.ndarray,
    ns_fid: float,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    dns_step: float,
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
    fnl_fid: float,
    ns_fid: float,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    dns_step: float,
    use_b_analytic: bool,
    b_kw: dict[str, Any],
    *,
    As_fid: float,
    include_As: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    b, db_or_scalar = _b_and_db(
        ell, ns_fid, k_D_i, k_D_f, k_p, dns_step, use_b_analytic, b_kw
    )
    T = T_muT_ell(ell, As_fid, k_D_i, k_D_f)
    K_fnl = T * b
    K_ns = fnl_fid * T * db_or_scalar
    cols = [K_fnl, K_ns]
    names: list[str] = ["fnl", "ns"]
    if include_As:
        cols.append(K_fnl * (2.0 / As_fid))
        names.append("As")
    return np.column_stack(cols), tuple(names)


def fisher_muT_general(
    ell: np.ndarray,
    fwhm_deg: float,
    fnl_fid: float,
    ns_fid: float,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    *,
    dns_step: float = 5e-5,
    sigma_ns_prior: float | None = 0.004,
    sigma_As_prior: float | None = None,
    include_As: bool | None = None,
    As_fid: float = AS_FID_LEGACY,
    use_b_analytic: bool = False,
    b_integral_kw: dict[str, Any] | None = None,
    variance_at_fiducial: bool = True,
) -> FisherMuTResult:
    r"""Gaussian Fisher (Tier A: \sigma_l^2 at fiducial A_s). See fisher_matrix module docstring for Tier B."""
    if not variance_at_fiducial:
        warnings.warn(
            "Tier B Fisher not implemented; using fiducial \\sigma_\l^2.",
            UserWarning,
            stacklevel=2,
        )
    b_kw = dict(b_integral_kw or {})
    if include_As is None:
        include_As = sigma_As_prior is not None
    cl_tt = Cl_TT(ell, As_fid)
    n_mumu = N_mu_mu(ell, fwhm_deg)
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
        use_b_analytic,
        b_kw,
        As_fid=As_fid,
        include_As=include_As,
    )
    F_data = (K * inv_var[:, None]).T @ K
    F_total = F_data.copy()
    if sigma_ns_prior is not None:
        F_total[1, 1] += 1.0 / (sigma_ns_prior**2)
    if include_As and sigma_As_prior is not None:
        F_total[2, 2] += 1.0 / (sigma_As_prior**2)
    cov = np.linalg.inv(F_total)
    sigma_fnl_u = 1.0 / np.sqrt(F_data[0, 0]) if F_data[0, 0] > 0 else np.inf
    sigma_ns_u = 1.0 / np.sqrt(F_data[1, 1]) if F_data[1, 1] > 0 else np.inf
    sigma_As_u: float | None = None
    sigma_As_m: float | None = None
    if include_As:
        sigma_As_u = 1.0 / np.sqrt(F_data[2, 2]) if F_data[2, 2] > 0 else np.inf
        sigma_As_m = float(np.sqrt(cov[2, 2]))
    corr = (
        float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        if cov[0, 0] > 0 and cov[1, 1] > 0
        else 0.0
    )
    return FisherMuTResult(
        F_data=F_data,
        F_total=F_total,
        param_names=param_names,
        sigma_fnl_unmarg=sigma_fnl_u,
        sigma_fnl_marg=float(np.sqrt(cov[0, 0])),
        sigma_ns_unmarg=sigma_ns_u,
        sigma_ns_marg=float(np.sqrt(cov[1, 1])),
        corr_fnl_ns=corr,
        cov_marginal=cov,
        sigma_As_unmarg=sigma_As_u,
        sigma_As_marg=sigma_As_m,
    )


def fisher_1d_fnl_only(
    ell: np.ndarray,
    fwhm_deg: float,
    ns_fid: float,
    k_D_i: float,
    k_D_f: float,
    k_p: float,
    *,
    As_fid: float = AS_FID_LEGACY,
    use_b_analytic: bool = True,
    b_integral_kw: dict[str, Any] | None = None,
) -> float:
    r"""F = sum{l (dC/df_NL)^2/\sigma_l^2} with Cl_TT(As_fid)"""
    b_kw = dict(b_integral_kw or {})
    cl_tt = Cl_TT(ell, As_fid)
    n_mumu = N_mu_mu(ell, fwhm_deg)
    var = sigma2_muT_hat(ell, cl_tt, n_mumu)
    ell_i = ell.astype(int)
    T = T_muT_ell(ell, As_fid, k_D_i, k_D_f)
    if use_b_analytic:
        b0 = b_analytic(ns_fid, k_D_i, k_D_f, k_p)
        b = np.full_like(ell, b0, dtype=float)
    else:
        b = np.array(
            [b_ell_ns(int(l), ns_fid, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **b_kw) for l in ell_i]
        )
    K_fnl = T * b
    return float(np.sum(K_fnl**2 / var))


def default_ell_grid(fwhm_deg: float, ell_min: int = 2) -> np.ndarray:
    lmax = ell_max_from_fwhm_deg(fwhm_deg)
    return np.arange(ell_min, int(np.ceil(lmax)) + 1, dtype=float)

# ------------------------------------------------------------
# Main function
"""
Fiducial f_NL values:
0, 1, 12_500, 25_000 (default reference f_NL = 1 so dC/dn_s != 0 when using numerical b)
"""


def main():
    # --- same baseline as 1D fisher ---
    fwhm_deg = 1.6 # PIXIE FWHM, lmax-84
    ns_fid = 0.965
    k_p = 0.002
    k_D_i = 1.1e4
    k_D_f = 46.0

    ell = default_ell_grid(fwhm_deg)
    sigma_ns_planck = 0.004

    fnl_fiducials = (0.0, 1.0, 12_500.0, 25_000.0)

    print("fNL Fisher forecast")
    print(f"l range: {int(ell[0])}...{int(ell[-1])}, FWHM = {fwhm_deg}^\\circ, n_s = {ns_fid}")
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
        use_b_analytic=True,
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

