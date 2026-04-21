'Reusable helpers for section 4 foreground sweep table generation.'

import json

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from config_section4 import (
    A_D_CODE,
    FNL_FIDUCIALS,
    SIGMA_AD_PRIOR,
    SIGMA_ALPHA_PRIOR,
    SIGMA_NS_PRIOR,
)
from config_section_common import FWHM_PIXIE, FWHM_SPECTER, K_D_F, K_D_I, K_P, NS_FID
from fisher_foreground import AZZONI_ALPHA_D, ELL0_AZZONI, fisher_muT_fnl_ns_with_dust
from fisher_matrix import default_ell_grid
from output_paths import ensure_section_layout


def run_cf_grid(
    *,
    experiment,
    fnl_fid,
    cl_tt_txt_dir,
    use_b_analytic,
    as_fid,
):
    fwhm, w_inv = (FWHM_PIXIE, W_MU_INV_PIXIE) if experiment == "pixie" else (FWHM_SPECTER, W_MU_INV_SPECTER)
    ell = default_ell_grid(fwhm)
    lines = [
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
            sigma_ns_prior=SIGMA_NS_PRIOR,
            sigma_AD_prior=SIGMA_AD_PRIOR,
            sigma_alpha_prior=SIGMA_ALPHA_PRIOR,
            use_b_analytic=use_b_analytic,
            cl_tt_txt_dir=cl_tt_txt_dir,
        )
        lines.append(
            f"{fnl_fid:g},{c_f:.6g},{r.sigma['fnl']:.8e},{r.sigma['ns']:.8e},"
            f"{r.sigma['A_D']:.8e},{r.sigma['alpha_D']:.8e}"
        )
    return lines


def run_pipeline(*, pipeline, cl_tt_txt_dir, use_b_analytic, as_fid):
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
            all_lines.extend(
                run_cf_grid(
                    experiment=exp,
                    fnl_fid=fnl,
                    cl_tt_txt_dir=cl_tt_txt_dir,
                    use_b_analytic=use_b_analytic,
                    as_fid=as_fid,
                )[1:]
            )
        (dirs["tables"] / f"foreground_cf_sweep_{exp}.csv").write_text("\n".join(all_lines) + "\n", encoding="utf-8")
