'C_l^{TT}, C_l^{\\mu T}, and minimal Gaussian C_l^{\\mu\\mu}.'

from __future__ import annotations

import os
import numpy as np

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

# PZ: Gaussian C_l^{\mu\mu} is ~white noise in \ell at low l (many small-scale \mu modes averaged).
PZ_MUMU_GAUSS_PREFACTOR = 3.5e-17
# Comoving distance to last scattering r_L \simeq 14\,{\rm Gpc}
PZ_MUMU_R_L_MPC_DEFAULT = 14000.0

# Default symmetric brackets for central d C_l^TT / d(n_s, A_s) from CAMB (Fisher-style).
DNS_CLTT_FD = 0.002  # absolute +/- on n_s (typical probe size; ~0.2% on n_s ~ 0.96)
DAS_REL_CLTT_FD = 1e-3  # symmetric relative A_s * (1 +- DAS_REL_CLTT_FD)

# File names written by ``planck_cosmology.save_planck2018_cltt_bundle``
CL_TT_TXT_FIDUCIAL = "cl_tt_fiducial.txt"
CL_TT_TXT_NS_HIGH = "cl_tt_ns_high.txt"
CL_TT_TXT_NS_LOW = "cl_tt_ns_low.txt"
CL_TT_TXT_AS_HIGH = "cl_tt_As_high.txt"
CL_TT_TXT_AS_LOW = "cl_tt_As_low.txt"


def load_cl_tt_txt(path):
    """Load Cl_TT values from a text file and return them indexed by multipole."""
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cl_TT text file not found: {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        return np.asarray(data, dtype=float)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got shape {data.shape}")
    return np.asarray(data[:, 1], dtype=float)


def cl_tt_on_ell_grid(full_cl_tt, ell):
    """Slice full ``cl[l]`` to the multipoles in ``ell`` (integer part)."""
    li = ell.astype(int, copy=False)
    lmax = int(li.max())
    if lmax >= full_cl_tt.shape[0]:
        raise ValueError(
            f"multipole ell max={lmax} out of range for Cl_TT length {full_cl_tt.shape[0]}"
        )
    return np.asarray(full_cl_tt[li], dtype=float)


def cl_tt_camb_muK2_to_tcmb_norm(cl_muK2):
    """Return the input Cl_TT array in the project's working normalization."""
    return np.asarray(cl_muK2, dtype=float)


def load_ClTT_planck18(cl_tt_txt_dir = None):
    """Load the default Planck18 Cl_TT fiducial and bracket arrays from disk."""
    if cl_tt_txt_dir is None:
        cl_tt_txt_dir = os.path.dirname(os.path.abspath(__file__))
    cl_tt_txt_dir = os.path.abspath(cl_tt_txt_dir)
    keys = {
        "fiducial": CL_TT_TXT_FIDUCIAL,
        "ns_high": CL_TT_TXT_NS_HIGH,
        "ns_low": CL_TT_TXT_NS_LOW,
        "As_high": CL_TT_TXT_AS_HIGH,
        "As_low": CL_TT_TXT_AS_LOW,
    }
    return {k: load_cl_tt_txt(os.path.join(cl_tt_txt_dir, name)) for k, name in keys.items()}


def ns_brackets_absolute(ns_fid, dns = DNS_CLTT_FD):
    """Return (n_s+, n_s-) with n_s+- = n_s_fid +- dns (conventional absolute brackets)."""
    return float(ns_fid + dns), float(ns_fid - dns)


def ns_brackets_relative(ns_fid, relative_step = 0.002):
    """Return multiplicative upper and lower n_s brackets around the fiducial value."""
    return float(ns_fid * (1.0 + relative_step)), float(ns_fid * (1.0 - relative_step))


def As_brackets_relative(As_fid, relative_half_width = DAS_REL_CLTT_FD):
    """Return symmetric relative upper and lower A_s brackets around the fiducial value."""
    return float(As_fid * (1.0 + relative_half_width)), float(As_fid * (1.0 - relative_half_width))


def dCl_TT_dtheta_numerical(
    cl_tt_high,
    cl_tt_low,
    theta_high,
    theta_low,
):
    """Compute a central finite-difference derivative of Cl_TT with respect to a scalar parameter."""
    cl_tt_high = np.asarray(cl_tt_high, dtype=float)
    cl_tt_low = np.asarray(cl_tt_low, dtype=float)
    if cl_tt_high.shape != cl_tt_low.shape:
        raise ValueError("cl_tt_high and cl_tt_low must have the same shape")
    denom = float(theta_high - theta_low)
    if denom == 0.0:
        raise ValueError("theta_high and theta_low must differ")
    return (cl_tt_high - cl_tt_low) / denom


def dCl_TT_dns_numerical(
    cl_tt_ns_high,
    cl_tt_ns_low,
    ns_high,
    ns_low,
):
    """Compute the central finite-difference derivative of Cl_TT with respect to n_s."""
    return dCl_TT_dtheta_numerical(cl_tt_ns_high, cl_tt_ns_low, ns_high, ns_low)


def dCl_TT_dAs_numerical(
    cl_tt_As_high,
    cl_tt_As_low,
    As_high,
    As_low,
):
    """Compute the central finite-difference derivative of Cl_TT with respect to A_s."""
    return dCl_TT_dtheta_numerical(cl_tt_As_high, cl_tt_As_low, As_high, As_low)


def Cl_TT(ell, A_s = None):
    r"""Return the large-scale Sachs-Wolfe Cl_TT template for the chosen A_s value."""
    if A_s is None:
        A_s = AS_FID_LEGACY
    return K_TT_SW * A_s / (ell * (ell + 1.0))


def T_muT_ell(ell, A_s, k_D_i, k_D_f):
    r"""Return the muT template factor as a function of multipole and damping scales."""
    L = np.log(k_D_i / k_D_f)
    return PZ_MUT_PREFACTOR * L * (A_s**2) / (ell * (ell + 1.0))


def dT_muT_ell_dkdf(
    ell, A_s, k_D_i, k_D_f
):
    r"""Return the derivative of the muT template factor with respect to k_D_f."""
    L = np.log(k_D_i / k_D_f)
    T = T_muT_ell(ell, A_s, k_D_i, k_D_f)
    return -T / (k_D_f * L)


def Cl_mu_mu_gaussian_PZ(
    ell,
    *,
    k_D_f,
    ns = 0.965,
    k_p = 0.002,
    k_s = None,
    r_L_mpc = PZ_MUMU_R_L_MPC_DEFAULT,
):
    r"""Return the Gaussian mu-mu template spectrum on the input multipole grid."""
    ell = np.asarray(ell, dtype=float)
    k_df = float(k_D_f)
    kp = float(k_p)
    ks = float(k_s if k_s is not None else k_D_f)
    delta4_ratio = (k_df / kp) ** (2.0 * (float(ns) - 1.0))
    cl = (
        PZ_MUMU_GAUSS_PREFACTOR
        * delta4_ratio
        * (ks / (float(r_L_mpc) ** 2))
        / (k_df**3)
    )
    return np.full_like(ell, cl, dtype=float)


def Cl_muT(
    ell,
    f_nl,
    b,
    A_s,
    k_D_i,
    k_D_f,
):
    """C_l^{\mu T} = f_NL * b * T_l; T_l from ``T_muT_ell``."""
    T = T_muT_ell(ell, A_s, k_D_i, k_D_f)
    return f_nl * np.asarray(b) * T


def dCl_muT_dAs(
    ell,
    f_nl,
    b,
    A_s,
    k_D_i,
    k_D_f,
):
    r"""\partial C_l^{\mu T}/\partial A_s = 2 C_l^{\mu T} / A_s at fixed f_NL, b, k_D (since T_l ~ A_s^2)."""
    c = Cl_muT(ell, f_nl, b, A_s, k_D_i, k_D_f)
    return 2.0 * c / A_s


def sigma2_muT_hat(ell, cl_tt, n_mumu):
    r"""PZ approx: Var(hat C_l^{mu T}) ~ C_l^{TT} C_l^{mu mu,N} / (2l+1)"""
    return (cl_tt * n_mumu) / (2.0 * ell + 1.0)


def sigma2_muT_hat_full(
    ell,
    cl_tt,
    *,
    cl_tt_noise = 0.0,
    cl_mumu_signal,
    cl_mumu_noise,
    cl_mut,
):
    r"""Return the full Gaussian variance of the muT band-power estimator."""
    ell = np.asarray(ell, dtype=float)
    cl_tt = np.asarray(cl_tt, dtype=float)
    tt_n = np.asarray(cl_tt_noise, dtype=float)
    if tt_n.shape == ():
        tt_tot = cl_tt + float(tt_n)
    else:
        tt_tot = cl_tt + tt_n
    mumu_tot = np.asarray(cl_mumu_signal, dtype=float) + np.asarray(cl_mumu_noise, dtype=float)
    cm = np.asarray(cl_mut, dtype=float)
    inner = tt_tot * mumu_tot + cm**2
    return inner / (2.0 * ell + 1.0)
