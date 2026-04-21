'Section 5a: Pajer--Zaldarriaga order-of-magnitude S/N for \\(\\mu T\\) (Eq.\\ around their \\eqref{ref}).'

from __future__ import annotations

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


import math

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER

W_MU_INV_SQRT_PIXIE = math.sqrt(W_MU_INV_PIXIE)  # ~ w_mu^{-1/2} in PZ convention
PZ_NUM = 0.7e-3
SQRT_4PI_1E8 = math.sqrt(4.0 * math.pi) * 1.0e-8


def pz_sn_over_fnl(*, b, w_mu_inv):
    """Return (S/N) / f_NL for given b and w_mu^{-1} (dimensionless, as in ``beam``)."""
    w_half = math.sqrt(w_mu_inv)
    return PZ_NUM * b * (SQRT_4PI_1E8 / w_half)


def main():
    print("# Order-of-magnitude (S/N) scaling per unit f_NL")
    print("# b\tPIXIE\tSPECTER")
    for b in (1.0, 10.0, 100.0):
        sp = pz_sn_over_fnl(b=b, w_mu_inv=W_MU_INV_PIXIE)
        ss = pz_sn_over_fnl(b=b, w_mu_inv=W_MU_INV_SPECTER)
        print(f"{b:g}\t{sp:.6e}\t{ss:.6e}")


if __name__ == "__main__":
    main()
