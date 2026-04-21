r"""
Plot \(C_\ell^{\mu\mu,\mathrm{sig}}\), \((C_\ell^{\mu T})^2\), \(C_\ell^{TT}\), and \(C_\ell^{\mu\mu,N}\) vs \(\ell\)
for visual comparison (Phase 1 diagnostic).

Run from ``muT_fNL_forecast_2``::

    python3 plot_muT_variance_terms.py --output figures/muT_variance_terms.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from beam import N_mu_mu, W_MU_INV_PIXIE, W_MU_INV_SPECTER
from spectra import AS_FID_LEGACY, Cl_muT, Cl_mu_mu_gaussian_PZ, Cl_TT

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fwhm-deg", type=float, default=1.6)
    p.add_argument("--fnl-fid", type=float, default=25_000.0)
    p.add_argument("--ns-fid", type=float, default=0.965)
    p.add_argument("--k-d-i", type=float, default=1.1e4)
    p.add_argument("--k-d-f", type=float, default=46.0)
    p.add_argument("--k-p", type=float, default=0.002)
    p.add_argument("--specter-noise", action="store_true", help="Use SPECTER w_mu^-1 instead of PIXIE")
    p.add_argument("--output", type=str, default="muT_variance_terms.pdf")
    args = p.parse_args()

    apply_plot_params()
    import matplotlib.pyplot as plt

    ell = np.arange(2, 201, dtype=float)
    pre_factor = (ell * (ell + 1)) / (2 * np.pi)

    cl_tt = Cl_TT(ell, AS_FID_LEGACY)
    w_inv = W_MU_INV_SPECTER if args.specter_noise else W_MU_INV_PIXIE
    cl_mumu_n = N_mu_mu(ell, args.fwhm_deg, w_mu_inv=w_inv)
    cl_mumu_s = Cl_mu_mu_gaussian_PZ(ell, k_D_f=args.k_d_f, ns=args.ns_fid, k_p=args.k_p)
    b0 = 1.0  # order-unity for illustration
    cl_mut = Cl_muT(ell, args.fnl_fid, b0, AS_FID_LEGACY, args.k_d_i, args.k_d_f)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ell, np.abs(cl_mumu_s), label=r"$|C_\ell^{\mu\mu,\mathrm{sig}}|$ (PZ Gaussian)")
    ax.plot(ell, pre_factor * cl_mut, label=r"$(C_\ell^{\mu T})$")
    ax.plot(ell, pre_factor * cl_tt, label=r"$C_\ell^{TT}$ (analytic SW)")
    ax.plot(ell, pre_factor * cl_mumu_n, label=r"$C_\ell^{\mu\mu,N}$")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\frac{\ell(\ell+1)}{2\pi}\, C_\ell$")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=11)
    ax.set_title(r"$\mu T$ variance ingredients ($f_{\rm NL}^{\rm fid}$=" + f"{args.fnl_fid:g})")
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
