r"""
Section 4: marginalize over Azzoni-style dust residual (amplitude + tilt) with cleaning factor ``c_f``.

Writes CSVs under ``muT_fNL_runs/section4_foregrounds/{pipeline}/tables/`` for multiple
``f_{\rm NL}^{\rm fid}`` values and optionally **CAMB** ``C_\ell^{TT}`` (Planck18 bracket files).

Run::

    python3 run_section4.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from fisher_foreground import (
    AZZONI_A_D_DIMENSIONLESS,
    AZZONI_ALPHA_D,
    ELL0_AZZONI,
    fisher_muT_fnl_ns_with_dust,
)
from fisher_matrix import default_ell_grid
from output_paths import ensure_section_layout
from spectra import AS_FID_PLANCK2018

FWHM_PIXIE = 1.6
FWHM_SPECTER = 1.0
NS_FID = 0.965
K_D_I = 1.1e4
K_D_F = 46.0
K_P = 0.002
# Grid of f_NL fiducials for foreground forecasts (match Cabass-style section 2 grid).
FNL_FIDUCIALS = (1.0, 10.0, 1000.0, 15000.0, 25000.0)
# Dimensionless dust amplitude: Azzoni A_D (μK²) / T_CMB² (μK²); see ``fisher_foreground``.
A_D_CODE = AZZONI_A_D_DIMENSIONLESS


def _module_dir() -> Path:
    return Path(__file__).resolve().parent


def _run_cf_grid(
    *,
    experiment: str,
    fnl_fid: float,
    cl_tt_txt_dir: str | None,
    use_b_analytic: bool,
    as_fid: float,
) -> list[str]:
    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    ell = default_ell_grid(fwhm)
    lines: list[str] = [
        "fnl_fid,c_f,sigma_fnl,sigma_ns,sigma_AD,sigma_alpha_D",
    ]
    for c_f in np.logspace(2.0, 4.0, 9):
        r = fisher_muT_fnl_ns_with_dust(
            ell,
            fwhm,
            float(fnl_fid),
            NS_FID,
            K_D_I,
            K_D_F,
            K_P,
            w_mu_inv=w_inv,
            c_f=float(c_f),
            A_D=A_D_CODE,
            alpha_D=AZZONI_ALPHA_D,
            ell0=ELL0_AZZONI,
            As_fid=as_fid,
            dns_step=5e-5,
            sigma_ns_prior=0.004,
            sigma_AD_prior=1e12,
            sigma_alpha_prior=1e6,
            use_b_analytic=use_b_analytic,
            cl_tt_txt_dir=cl_tt_txt_dir,
        )
        lines.append(
            f"{fnl_fid:g},{c_f:.6g},{r.sigma['fnl']:.8e},{r.sigma['ns']:.8e},"
            f"{r.sigma['A_D']:.8e},{r.sigma['alpha_D']:.8e}"
        )
    return lines


def _run_pipeline(*, pipeline: str, cl_tt_txt_dir: str | None, use_b_analytic: bool, as_fid: float) -> None:
    dirs = ensure_section_layout("section4_foregrounds", pipeline)
    cfg = {
        "A_D_code": A_D_CODE,
        "alpha_D": AZZONI_ALPHA_D,
        "ell0": ELL0_AZZONI,
        "fnl_fiducials": FNL_FIDUCIALS,
        "cl_tt_txt_dir": cl_tt_txt_dir,
        "use_b_analytic": use_b_analytic,
        "As_fid": as_fid,
    }
    (dirs["logs"] / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    for exp in ("pixie", "specter"):
        all_lines = ["fnl_fid,c_f,sigma_fnl,sigma_ns,sigma_AD,sigma_alpha_D"]
        for fnl in FNL_FIDUCIALS:
            all_lines.extend(_run_cf_grid(
                experiment=exp,
                fnl_fid=fnl,
                cl_tt_txt_dir=cl_tt_txt_dir,
                use_b_analytic=use_b_analytic,
                as_fid=as_fid,
            )[1:])
        (dirs["tables"] / f"foreground_cf_sweep_{exp}.csv").write_text("\n".join(all_lines) + "\n", encoding="utf-8")


def main() -> None:
    camb_dir = str(_module_dir())
    _run_pipeline(
        pipeline="analytic_cltt_analytic_b",
        cl_tt_txt_dir=None,
        use_b_analytic=False,
        as_fid=AS_FID_PLANCK2018,
    )
    _run_pipeline(
        pipeline="camb_cltt_analytic_b",
        cl_tt_txt_dir=camb_dir,
        use_b_analytic=True,
        as_fid=AS_FID_PLANCK2018,
    )


if __name__ == "__main__":
    main()
