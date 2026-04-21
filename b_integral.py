'Factorized b(l, n_s) for squeezed limit, with split damping window.'

from __future__ import annotations
import numpy as np
from scipy.special import spherical_jn


def log_k_D_ratio(k_D_i, k_D_f):
    return float(np.log(k_D_i / k_D_f))


def b_analytic(ns, k_D_i, k_D_f, k_p):
    r"""Leading PZ approximation: b \simeq 1 + (n_s-1)/2 * ln(k_D,i k_D,f / (4 k_p^2))."""
    eps = ns - 1.0
    #print("b_analytic=", 1.0 + 0.5 * eps * np.log((k_D_i * k_D_f) / (4.0 * k_p**2)))
    return 1.0 + 0.5 * eps * np.log((k_D_i * k_D_f) / (4.0 * k_p**2))


def I_plus(ell, r_L_mpc, k_plus):
    r"""Compute the factorized plus integral for a given multipole."""
    jl = spherical_jn(ell, k_plus * r_L_mpc)
    return float(np.trapz(jl**2, k_plus))


def I_minus(
    eps,
    k_p,
    k_D_i,
    k_D_f,
    k_minus,
):
    """Compute the factorized minus integral from the damping-window difference."""
    t = (k_minus / (2.0 * k_p)) ** eps
    I_i = np.trapz(t * np.exp(-(k_minus**2) / (2.0 * k_D_i**2)), k_minus)
    I_f = np.trapz(t * np.exp(-(k_minus**2) / (2.0 * k_D_f**2)), k_minus)
    return float(I_i - I_f)


def F_b_factor(ell, ns, *, L, k_p, k_D_i, k_D_f, r_L_mpc, k_plus, k_minus):
    """F(l, n_s) = l(l+1)(2/L) I_+ * I_- used to get b"""
    eps = ns - 1.0
    Ip = I_plus(ell, r_L_mpc, k_plus)
    Im = I_minus(eps, k_p, k_D_i, k_D_f, k_minus)
    return float(ell * (ell + 1.0) * (2.0 / L) * Ip * Im)


def b_ell_ns(
    ell,
    ns,
    *,
    k_D_i,
    k_D_f,
    k_p,
    r_L_mpc = 14000.0,
    k_plus_grid = None,
    k_minus_grid = None,
    ell_ref = 50,
    ns_ref = 0.965,
):
    """Compute b(ell, n_s) from the normalized factorized integrals."""
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
    ell,
    ns,
    h,
    *,
    k_D_i,
    k_D_f,
    k_p,
    **kwargs,
):
    """Compute the central finite-difference derivative of b with respect to n_s."""
    return (
        b_ell_ns(ell, ns + h, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **kwargs)
        - b_ell_ns(ell, ns - h, k_D_i=k_D_i, k_D_f=k_D_f, k_p=k_p, **kwargs)
    ) / (2.0 * h)


def db_dkdf_analytic(ns, k_D_f):
    r"""Leading PZ b_analytic: d b / d k_{D,f} = (n_s-1)/(2 k_{D,f})."""
    return 0.5 * (ns - 1.0) / k_D_f


def db_dkdf_central(
    ell,
    ns,
    h,
    *,
    k_D_i,
    k_D_f,
    k_p,
    **kwargs,
):
    """Compute the central finite-difference derivative of b with respect to k_D_f."""
    return (
        b_ell_ns(ell, ns, k_D_i=k_D_i, k_D_f=k_D_f + h, k_p=k_p, **kwargs)
        - b_ell_ns(ell, ns, k_D_i=k_D_i, k_D_f=k_D_f - h, k_p=k_p, **kwargs)
    ) / (2.0 * h)
