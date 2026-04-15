r"""
Factorized b(l, n_s) for squeezed limit, with split damping window:
b/(l(l+1)) \approx (2/L) I_+(l) I_-(ns-1),
with I_+ from the k_+ Bessel integral and I_- from the k_- damping integrals.

Tilt in I_- uses (k_-/(2 k_p))(ns-1).

Calibration constant C0 fixes overall normalization so b(l_ref, n_s_ref) matches
the analytic PZ approximation b_analytic(n_s) at reference values.
"""

from __future__ import annotations
import numpy as np
from scipy.special import spherical_jn


def log_k_D_ratio(k_D_i: float, k_D_f: float) -> float:
    return float(np.log(k_D_i / k_D_f))


def b_analytic(ns: float, k_D_i: float, k_D_f: float, k_p: float) -> float:
    r"""Leading PZ approximation: b \simeq 1 + (n_s-1)/2 * ln(k_D,i k_D,f / (4 k_p^2))."""
    eps = ns - 1.0
    #print("b_analytic=", 1.0 + 0.5 * eps * np.log((k_D_i * k_D_f) / (4.0 * k_p**2)))
    return 1.0 + 0.5 * eps * np.log((k_D_i * k_D_f) / (4.0 * k_p**2))


def I_plus(ell: int, r_L_mpc: float, k_plus: np.ndarray) -> float:
    r"""
    numerically integrate
    I_+ = \int dk_+ j_l^2(k_+ r_L) (measure from \int d ln k_+ k_+ j_l^2 \to \int dk_+ j_l^2)."""
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
    #print("F=", F)
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


def db_dkdf_analytic(ns: float, k_D_f: float) -> float:
    r"""Leading PZ :math:`b_{\mathrm{analytic}}`: :math:`\partial b/\partial k_{D,f}=(n_s-1)/(2 k_{D,f})`."""
    return 0.5 * (ns - 1.0) / k_D_f


def db_dkdf_central(
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
    \partial b/\partial k_{D,f} from symmetric finite differences of ``b_ell_ns``.
    Use when ``use_b_analytic=False``; match ``h`` to ``dns_step`` scale (e.g. \(\sim 0.1\!-\!1\) Mpc\(^{-1}\)).
    """
    return (
        b_ell_ns(ell, ns, k_D_i=k_D_i, k_D_f=k_D_f + h, k_p=k_p, **kwargs)
        - b_ell_ns(ell, ns, k_D_i=k_D_i, k_D_f=k_D_f - h, k_p=k_p, **kwargs)
    ) / (2.0 * h)
