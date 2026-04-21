r"""C_l^{TT}, C_l^{\mu T}, and minimal Gaussian C_l^{\mu\mu}

For C_l^{TT} from CAMB, central differences use the bracket helpers below.
Defaults: absolute +-0.002 on n_s (``DNS_CLTT_FD``) and symmetric relative +-1e-3 on A_s
(``DAS_REL_CLTT_FD``)
"""

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


def load_cl_tt_txt(path: str) -> np.ndarray:
    """
    Load CAMB ``C_l^{TT}`` (second column: ell, Cl in muK^2) from a text file; skip ``#`` lines.

    Returns ``cl[l]`` indexed by integer multipole ``l`` (row order 0..lmax as in planck_cosmology output).
    """
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cl_TT text file not found: {path}")
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        return np.asarray(data, dtype=float)
    if data.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}, got shape {data.shape}")
    return np.asarray(data[:, 1], dtype=float)


def cl_tt_on_ell_grid(full_cl_tt: np.ndarray, ell: np.ndarray) -> np.ndarray:
    """Slice full ``cl[l]`` to the multipoles in ``ell`` (integer part)."""
    li = ell.astype(int, copy=False)
    lmax = int(li.max())
    if lmax >= full_cl_tt.shape[0]:
        raise ValueError(
            f"multipole ell max={lmax} out of range for Cl_TT length {full_cl_tt.shape[0]}"
        )
    return np.asarray(full_cl_tt[li], dtype=float)


def cl_tt_camb_muK2_to_tcmb_norm(cl_muK2: np.ndarray) -> np.ndarray:
    """
    Legacy compatibility helper.

    CAMB text files in this project are now generated directly in the Fisher code's
    working dimensionless ``C_l^{TT}`` normalization, so no conversion is applied.
    """
    return np.asarray(cl_muK2, dtype=float)


def load_ClTT_planck18(cl_tt_txt_dir: str | None = None) -> dict[str, np.ndarray]:
    """
    Load the five default ``C_l^{TT}`` vectors (fiducial and n_s / A_s brackets).

    ``cl_tt_txt_dir`` defaults to the directory containing this module.
    """
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


def Cl_mu_mu_gaussian_PZ(
    ell: np.ndarray,
    *,
    k_D_f: float,
    ns: float = 0.965,
    k_p: float = 0.002,
    k_s: float | None = None,
    r_L_mpc: float = PZ_MUMU_R_L_MPC_DEFAULT,
) -> np.ndarray:
    r"""
    C_{\ell,\mathrm{Gauss}}^{\mu\mu} \sim 3.5\times 10^{-17}\,
    \frac{\Delta_{\mathcal{R}}^{4}(k_{D,f})}{\Delta_{\mathcal{R}}^{4}(k_{p})}\, \frac{\ks\, r_{L}^{-2}}{k_{D,f}^{3}}\,.

    With `\Delta_{\mathcal{R}}^{2}(k)=A_s\,(k/k_p)^{n_s-1}`, the ratio is
    `(k_{D,f}/k_p)^{2(n_s-1)}` (no explicit `A_s` in this ratio).

    Parameters
    ----------
    ell :
        Multipoles; the returned spectrum is **constant** on this grid (broadcast).
    k_D_f :
        Final dissipation wavenumber `k_{D,f}` (Mpc\ :sup:`-1`), same convention as ``b_integral``.
    ns, k_p :
        Spectral index and pivot for the `\Delta_{\mathcal{R}}^{4}` ratio.
    k_s :
        Short-scale cutoff `\ks`; default ``None`` uses PZ upper bound `\ks=k_{D,f}`.
    r_L_mpc :
        `r_L` in Mpc (default `\sim 14\,\mathrm{Gpc}`).
    """
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


def sigma2_muT_hat_full(
    ell: np.ndarray,
    cl_tt: np.ndarray,
    *,
    cl_tt_noise: np.ndarray | float = 0.0,
    cl_mumu_signal: np.ndarray,
    cl_mumu_noise: np.ndarray,
    cl_mut: np.ndarray,
) -> np.ndarray:
    r"""
    Full-sky Gaussian variance for the :math:`\mu T` band-power estimator (PZ Fisher section,
    commented line before their instrumental-noise approximation):

    .. math::

        \sigma_\ell^2 = \frac{1}{2\ell+1}\Bigl[
            (C_\ell^{TT}+N_\ell^{TT})(C_\ell^{\mu\mu,\mathrm{sig}}+C_\ell^{\mu\mu,N})
            + (C_\ell^{\mu T})^2 \Bigr].

    **PZ instrumental limit:** if :math:`C_\ell^{\mu\mu,\mathrm{sig}}\ll C_\ell^{\mu\mu,N}` and
    :math:`(C_\ell^{\mu T})^2 \ll C_\ell^{TT} C_\ell^{\mu\mu,N}`, this reduces to
    ``sigma2_muT_hat`` (up to the dropped :math:`(C^{\mu T})^2` term in the strict limit).

    Parameters
    ----------
    cl_tt :
        :math:`C_\ell^{TT}` at the fiducial (same units as elsewhere in this module).
    cl_tt_noise :
        :math:`N_\ell^{TT}`; scalar or per-\ell array (default 0).
    cl_mumu_signal, cl_mumu_noise :
        Gaussian signal :math:`C_\ell^{\mu\mu}` (e.g. ``Cl_mu_mu_gaussian_PZ``) and instrumental
        :math:`C_\ell^{\mu\mu,N}` (e.g. ``beam.N_mu_mu``; use zeros for CV-limited :math:`\mu` noise).
    cl_mut :
        :math:`C_\ell^{\mu T}` at the fiducial used in the :math:`(C^{\mu T})^2` term.
    """
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
