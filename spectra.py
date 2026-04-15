r"""C_l^{TT} and C_l^{\mu T} templates.

For C_l^{TT} from CAMB, central differences use the bracket helpers below.
Defaults: absolute +-0.002 on n_s (``DNS_CLTT_FD``) and symmetric relative +-1e-3 on A_s
(``DAS_REL_CLTT_FD``)
"""

from __future__ import annotations

import numpy as np

# CMB monopole temperature (thermodynamic); Fixsen 2009 scale for conversions involving q ~ 1/T.
TCMB_K = 2.7255

# Sachs--Wolf / large-scale TT: C_l^TT = K_TT * A_s / (l(l+1)), A_s = \Delta_R^2(k_p) at pivot
K_TT_SW = 2.0 * np.pi / 25.0

# Legacy numeric plateau 6e-10/(l(l+1)) used in earlier code
AS_FID_LEGACY = 6e-10 / K_TT_SW

# Planck 2018 TT,TE,EE+lowE: ln(10^10 A_s) = 3.044 +- 0.014 -> \sigma(A_s) \approx A_s \sigma(Y)
AS_FID_PLANCK2018 = 2.1e-9
SIGMA_AS_PLANCK2018 = AS_FID_PLANCK2018 * 0.014  # ~2.9e-11

#C_l^{\mu T} = 6.1\pi (9/25) f_NL b \Delta_R^4/(l(l+1)) ln(k_D,i/k_D,f).
# With A_s \equiv \Delta_R^2(k_p). This is the \mu T factor multiplying f_NL b (no f_NL,b inside).
PZ_MUT_PREFACTOR = 6.1 * np.pi * (9.0 / 25.0)

# Default symmetric brackets for central d C_l^TT / d(n_s, A_s) from CAMB (Fisher-style).
DNS_CLTT_FD = 0.002  # absolute +/- on n_s (typical probe size; ~0.2% on n_s ~ 0.96)
DAS_REL_CLTT_FD = 1e-3  # symmetric relative A_s * (1 +- DAS_REL_CLTT_FD)


def ns_brackets_absolute(ns_fid: float, dns: float = DNS_CLTT_FD) -> tuple[float, float]:
    """Return (n_s+, n_s-) with n_s+- = n_s_fid +- dns (conventional absolute brackets)."""
    return float(ns_fid + dns), float(ns_fid - dns)


def ns_brackets_relative(ns_fid: float, relative_step: float = 0.002) -> tuple[float, float]:
    """
    Return (n_s+, n_s-) with n_s+- = n_s_fid * (1 +- relative_step).

    Default is a small multiplicative tilt (~0.2% at n_s ~ 1). For Fisher Jacobians,
    ``ns_brackets_absolute`` is usually preferred.
    """
    return float(ns_fid * (1.0 + relative_step)), float(ns_fid * (1.0 - relative_step))


def As_brackets_relative(As_fid: float, relative_half_width: float = DAS_REL_CLTT_FD) -> tuple[float, float]:
    """
    Return (A_s+, A_s-) with A_s+- = A_s_fid * (1 +- h).

    Default h = DAS_REL_CLTT_FD (1e-3): small symmetric relative step for central
    derivatives of C_l^TT with respect to A_s.
    """
    return float(As_fid * (1.0 + relative_half_width)), float(As_fid * (1.0 - relative_half_width))


def dCl_TT_dtheta_numerical(
    cl_tt_high: np.ndarray,
    cl_tt_low: np.ndarray,
    theta_high: float,
    theta_low: float,
) -> np.ndarray:
    """
    Central finite difference d C_l^TT / d theta for a scalar theta.

    (C_l(theta+) - C_l(theta-)) / (theta+ - theta-).
    cl_tt_high and cl_tt_low must be TT spectra on the same multipole grid (same length).

    cl_tt_high, cl_tt_low: C_l^TT from CAMB (or any source) at theta_high and theta_low.
    theta_high, theta_low: bracket values (e.g. from ``ns_brackets_absolute`` or
    ``As_brackets_relative``).
    """
    cl_tt_high = np.asarray(cl_tt_high, dtype=float)
    cl_tt_low = np.asarray(cl_tt_low, dtype=float)
    if cl_tt_high.shape != cl_tt_low.shape:
        raise ValueError("cl_tt_high and cl_tt_low must have the same shape")
    denom = float(theta_high - theta_low)
    if denom == 0.0:
        raise ValueError("theta_high and theta_low must differ")
    return (cl_tt_high - cl_tt_low) / denom


def dCl_TT_dns_numerical(
    cl_tt_ns_high: np.ndarray,
    cl_tt_ns_low: np.ndarray,
    ns_high: float,
    ns_low: float,
) -> np.ndarray:
    """
    d C_l^TT / d n_s from two CAMB runs (other cosmology fixed).

    Pass spectra at n_s = n_s+ and n_s = n_s- (e.g. from ``ns_brackets_absolute``).
    """
    return dCl_TT_dtheta_numerical(cl_tt_ns_high, cl_tt_ns_low, ns_high, ns_low)


def dCl_TT_dAs_numerical(
    cl_tt_As_high: np.ndarray,
    cl_tt_As_low: np.ndarray,
    As_high: float,
    As_low: float,
) -> np.ndarray:
    """
    d C_l^TT / d A_s from two CAMB runs (other cosmology fixed).

    Pass spectra at A_s = A_s+ and A_s = A_s-
    """
    return dCl_TT_dtheta_numerical(cl_tt_As_high, cl_tt_As_low, As_high, As_low)


def Cl_TT(ell: np.ndarray, A_s: float | None = None) -> np.ndarray:
    r"""
    Large-scale C_l^{TT} = (2\pi/25) A_s / (l(l+1)).

    If ``A_s`` is None, uses ``AS_FID_LEGACY`` so the result matches the former fixed
    ``6e-10/(l(l+1))`` scaling.
    """
    if A_s is None:
        A_s = AS_FID_LEGACY
    return K_TT_SW * A_s / (ell * (ell + 1.0))


def T_muT_ell(ell: np.ndarray, A_s: float, k_D_i: float, k_D_f: float) -> np.ndarray:
    r"""
    \mu T template factor (everything in PZ eq. (186) except f_NL and b):

        T_l = 6.1\pi (9/25) ln(k_D,i/k_D,f) A_s^2 / (l(l+1)).

    So C_l^{\mu T} = f_NL * b * T_l with explicit k_D,i, k_D,f and A_s dependence.
    The paper's ``\approx 2.2\times10^{-16}/(l(l+1))`` line is recovered for fiducial
    ``(A_s, k_D,i, k_D,f)`` used in the forecast.
    """
    L = np.log(k_D_i / k_D_f)
    return PZ_MUT_PREFACTOR * L * (A_s**2) / (ell * (ell + 1.0))


def dT_muT_ell_dkdf(
    ell: np.ndarray, A_s: float, k_D_i: float, k_D_f: float
) -> np.ndarray:
    r"""
    d T_l / d k_{D,f} at fixed k_{D,i}, A_s.

    With T_l proportional to L = ln(k_{D,i}/k_{D,f}), dL/d k_{D,f} = -1/k_{D,f}, so
    d T_l / d k_{D,f} = -T_l / (k_{D,f} L).
    """
    L = np.log(k_D_i / k_D_f)
    T = T_muT_ell(ell, A_s, k_D_i, k_D_f)
    return -T / (k_D_f * L)


def Cl_muT(
    ell: np.ndarray,
    f_nl: float,
    b: np.ndarray | float,
    A_s: float,
    k_D_i: float,
    k_D_f: float,
) -> np.ndarray:
    """C_l^{\mu T} = f_NL * b * T_l; T_l from ``T_muT_ell``."""
    T = T_muT_ell(ell, A_s, k_D_i, k_D_f)
    return f_nl * np.asarray(b) * T


def dCl_muT_dAs(
    ell: np.ndarray,
    f_nl: float,
    b: np.ndarray | float,
    A_s: float,
    k_D_i: float,
    k_D_f: float,
) -> np.ndarray:
    r"""\partial C_l^{\mu T}/\partial A_s = 2 C_l^{\mu T} / A_s at fixed f_NL, b, k_D (since T_l ~ A_s^2)."""
    c = Cl_muT(ell, f_nl, b, A_s, k_D_i, k_D_f)
    return 2.0 * c / A_s


def sigma2_muT_hat(ell: np.ndarray, cl_tt: np.ndarray, n_mumu: np.ndarray) -> np.ndarray:
    r"""PZ approx: Var(hat C_l^{mu T}) ~ C_l^{TT} C_l^{mu mu,N} / (2l+1)"""
    return (cl_tt * n_mumu) / (2.0 * ell + 1.0)
