"""Beam-related helpers (effective lmax, deconvolved mu noise)."""

from __future__ import annotations
import numpy as np

# w_\mu^{-1} in C_l^{\mu\mu,N} (dimensionless); pick experiment when calling Fisher code.
W_MU_INV_PIXIE = 1.3e-15  # legacy default: PIXIE w_\mu^{-1/2} = 1.3e-15
W_MU_INV_SPECTER = (2e-9) ** 2


def ell_max_from_fwhm_deg(fwhm_deg: float) -> float:
    """Gaussian beam: l_max = sqrt(8 ln 2) / FWHM (radians)."""
    return float(np.sqrt(8.0 * np.log(2.0)) / np.deg2rad(fwhm_deg))


def N_mu_mu(ell: np.ndarray, fwhm_deg: float, w_mu_inv: float = W_MU_INV_PIXIE) -> np.ndarray:
    r"""
    deconvolved mu autospectrum noise (dimensionless C_l units).
    C_l^{\mu\mu,N} \simeq w_\mu^{-1} * exp(+l^2/l_max^2).
    Default ``w_mu_inv`` matches PIXIE (``W_MU_INV_PIXIE``); use ``W_MU_INV_SPECTER`` for SPECTER.
    """
    lmax = ell_max_from_fwhm_deg(fwhm_deg)
    return w_mu_inv * np.exp(ell**2 / lmax**2)
