"""Planck 2018 best-fit LCDM unlensed C_l^TT from CAMB, plus bracket spectra for d C_l^TT / d n_s and d C_l^TT / d A_s.

Importing this module runs CAMB five times (fiducial + four brackets). Use with
``spectra.dCl_TT_dns_numerical(cl_tt_ns_high, cl_tt_ns_low, NS_HIGH, NS_LOW)`` and the
analogous ``dCl_TT_dAs_numerical`` for ``AS_HIGH`` / ``AS_LOW``.

Run as ``python planck_cosmology.py`` to write ``cl_tt_fiducial.txt`` and the four bracket
files (two-column ``ell``, ``Cl_TT`` in muK^2) next to this script; see ``save_planck2018_cltt_bundle``.
"""

from __future__ import annotations

import os

import camb
import matplotlib.pyplot as plt
import numpy as np

try:
    from .spectra import (
        AS_FID_PLANCK2018,
        As_brackets_relative,
        ns_brackets_absolute,
    )
    from .plot_params import apply_plot_params
except ImportError:
    from spectra import (
        AS_FID_PLANCK2018,
        As_brackets_relative,
        ns_brackets_absolute,
    )
    from plot_params import apply_plot_params

# Planck 2018 TT,TE,EE+lowE style LCDM (fixed here to match your fiducial runs)
H0_PLANCK2018 = 67.36
OMBH2_PLANCK2018 = 0.02237
OMCH2_PLANCK2018 = 0.1200
TAU_PLANCK2018 = 0.0544

NS_FID = 0.965
AS_FID = AS_FID_PLANCK2018  # 2.1e-9


def _lcdm_params() -> camb.CAMBparams:
    p = camb.CAMBparams()
    p.set_cosmology(
        H0=H0_PLANCK2018,
        ombh2=OMBH2_PLANCK2018,
        omch2=OMCH2_PLANCK2018,
        tau=TAU_PLANCK2018,
    )
    return p


def cltt_unlensed_muK(*, ns: float, as_: float) -> np.ndarray:
    """Return CAMB unlensed *scalar* C_l^TT (muK^2) for ``as_`` (= A_s) and ``ns``.

    Uses ``unlensed_scalar`` (not ``unlensed_total``); they match when tensor B-modes are negligible.
    """
    pars = _lcdm_params()
    pars.InitPower.set_params(As=as_, ns=ns)
    results = camb.get_results(pars)
    cls = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    return np.asarray(cls["unlensed_scalar"][:, 0], dtype=float)


# Fiducial (Planck best-fit primordial + LCDM above)
cltt = cltt_unlensed_muK(ns=NS_FID, as_=AS_FID)

# Brackets for central derivatives (same steps as spectra.DNS_CLTT_FD / DAS_REL_CLTT_FD)
NS_HIGH, NS_LOW = ns_brackets_absolute(NS_FID)
AS_HIGH, AS_LOW = As_brackets_relative(AS_FID)

cl_tt_ns_high = cltt_unlensed_muK(ns=NS_HIGH, as_=AS_FID)
cl_tt_ns_low = cltt_unlensed_muK(ns=NS_LOW, as_=AS_FID)
cl_tt_As_high = cltt_unlensed_muK(ns=NS_FID, as_=AS_HIGH)
cl_tt_As_low = cltt_unlensed_muK(ns=NS_FID, as_=AS_LOW)


def save_cltt_to_txt(
    path: str,
    cl_tt: np.ndarray,
    *,
    description: str = "",
) -> str:
    """
    Write ``C_l^{TT}`` (muK^2) as two columns: multipole ell (row index 0..lmax) and value.

    Lines starting with ``#`` are comments (CAMB row ``i`` is multipole ``i``).
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    ell = np.arange(cl_tt.shape[0], dtype=int)
    with open(path, "w", encoding="utf-8") as f:
        if description:
            f.write(f"# {description}\n")
        f.write("# ell  Cl_TT_muK2\n")
        for i, v in zip(ell, np.asarray(cl_tt, dtype=float)):
            f.write(f"{int(i):d}\t{v:.14e}\n")
    return path


def save_planck2018_cltt_bundle(out_dir: str | None = None) -> list[str]:
    """Save fiducial and bracket ``C_l^{TT}`` arrays to text files under ``out_dir`` (default: this directory)."""
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.abspath(out_dir)
    specs: list[tuple[str, np.ndarray, str]] = [
        ("cl_tt_fiducial.txt", cltt, f"fiducial ns={NS_FID} As={AS_FID}"),
        ("cl_tt_ns_high.txt", cl_tt_ns_high, f"n_s={NS_HIGH} As={AS_FID}"),
        ("cl_tt_ns_low.txt", cl_tt_ns_low, f"n_s={NS_LOW} As={AS_FID}"),
        ("cl_tt_As_high.txt", cl_tt_As_high, f"n_s={NS_FID} As={AS_HIGH}"),
        ("cl_tt_As_low.txt", cl_tt_As_low, f"n_s={NS_FID} As={AS_LOW}"),
    ]
    written: list[str] = []
    for name, arr, desc in specs:
        written.append(save_cltt_to_txt(os.path.join(out_dir, name), arr, description=desc))
    return written


def plot_cltt_ell_prefactor(
    cl_tt: np.ndarray,
    out_path: str,
    *,
    title: str = "Planck 2018 unlensed CAMB TT",
) -> str:
    """
    Plot ``ell(ell+1) C_l^{TT} / (2pi)`` from ``ell>=2`` over the full available range.

    ``cl_tt`` is assumed to be CAMB TT in ``muK^2`` indexed by row ``ell``.
    """
    cl_tt = np.asarray(cl_tt, dtype=float)
    ell = np.arange(cl_tt.shape[0], dtype=float)
    mask = ell >= 2.0
    ell_plot = ell[mask]
    prefactor = 1 #ell_plot * (ell_plot + 1.0) / (2.0 * np.pi)
    dl_tt = cl_tt[mask] * prefactor

    out_path = os.path.abspath(out_path)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    apply_plot_params()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(ell_plot, dl_tt, lw=1.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ell")
    ax.set_ylabel("ell(ell+1) C_ell^TT / (2pi) [muK^2]")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    out_paths = save_planck2018_cltt_bundle()
    plot_path = plot_cltt_ell_prefactor(
        cltt,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "cl_tt_fiducial_ell_prefactor.png"),
    )
    for p in out_paths:
        print(p)
    print(plot_path)
