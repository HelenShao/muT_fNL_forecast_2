r"""
Section 5a: Pajer--Zaldarriaga order-of-magnitude S/N for \(\mu T\) (Eq.\ around their \eqref{ref}).

.. math:: S/N \simeq 0.7\times 10^{-3}\, b\, f_{\rm NL}\, \left(\frac{\sqrt{4\pi}\times 10^{-8}}{w_\mu^{-1/2}}\right).

Uses PIXIE-class ``w_\mu^{-1/2} \simeq \sqrt{4\pi}\times 10^{-8}`` as in PZ text.

Run::

    python3 pz_sn_ratio.py > results/sn_table.txt
"""

from __future__ import annotations

import math

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER

W_MU_INV_SQRT_PIXIE = math.sqrt(W_MU_INV_PIXIE)  # ~ w_mu^{-1/2} in PZ convention
PZ_NUM = 0.7e-3
SQRT_4PI_1E8 = math.sqrt(4.0 * math.pi) * 1.0e-8


def pz_sn_over_fnl(*, b: float, w_mu_inv: float) -> float:
    """Return (S/N) / f_NL for given b and w_mu^{-1} (dimensionless, as in ``beam``)."""
    w_half = math.sqrt(w_mu_inv)
    return PZ_NUM * b * (SQRT_4PI_1E8 / w_half)


def main() -> None:
    print("# PZ-style (S/N) scaling per unit f_NL (order of magnitude)")
    print("# b\tPIXIE\tSPECTER")
    for b in (1.0, 10.0, 100.0):
        sp = pz_sn_over_fnl(b=b, w_mu_inv=W_MU_INV_PIXIE)
        ss = pz_sn_over_fnl(b=b, w_mu_inv=W_MU_INV_SPECTER)
        print(f"{b:g}\t{sp:.6e}\t{ss:.6e}")


if __name__ == "__main__":
    main()
