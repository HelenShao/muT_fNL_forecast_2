r"""C_l^{TT} and C_l^{\mu T} templates"""

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
    \(\partial T_\ell/\partial k_{D,f}\) at fixed \(k_{D,i}, A_s\).

    With \(T_\ell \propto L=\ln(k_{D,i}/k_{D,f})\), \(\partial L/\partial k_{D,f}=-1/k_{D,f}\), so
    \(\partial T_\ell/\partial k_{D,f} = -T_\ell/(k_{D,f} L)\).
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
    r"""C_l^{\mu T} = f_NL * b * T_l with :func:`T_muT_ell`."""
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
    r"""PZ approx: \mathrm{Var}(\hat{C}_l^{\mu T}) ~ C_l^{TT} C_l^{\mu\mu,N} / (2l+1)"""
    return (cl_tt * n_mumu) / (2.0 * ell + 1.0)
