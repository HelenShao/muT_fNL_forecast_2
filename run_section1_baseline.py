'Section 1 baseline (PZ reproduction): dual pipelines, CV + PIXIE + SPECTER, ``b_override`` 10/100.'

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
from fisher_matrix import (
    VARIANCE_FULL_GAUSSIAN_CV,
    VARIANCE_PZ_INSTRUMENTAL_APPROX,
    fisher_1d_fnl_only,
)
from output_paths import ensure_section_layout, forecast_tables_dir

try:
    from .spectra import AS_FID_PLANCK2018
except ImportError:
    from spectra import AS_FID_PLANCK2018

from fisher_matrix import default_ell_grid

# Reference beams (see figure_plans clarifications)
FWHM_PIXIE_DEG = 1.6
FWHM_SPECTER_REF_DEG = 1.0  # ~150 GHz proxy

NS_FID = 0.965
K_D_I = 1.1e4
K_D_F = 46.0
K_P = 0.002
SCRIPT_DIR = Path(__file__).resolve().parent


def _ell_for(fwhm_deg, lmax_cap = None):
    ell = default_ell_grid(fwhm_deg)
    if lmax_cap is not None and ell.size:
        ell = ell[ell <= lmax_cap]
    return ell


def _run_1d_cases(
    *,
    pipeline_name,
    use_camb_cltt,
    use_numerical_b,
    cl_tt_txt_dir,
    lines,
):
    def emit(msg):
        print(msg)
        lines.append(msg)

    emit(f"=== pipeline={pipeline_name} camb_cltt={use_camb_cltt} numerical_b={use_numerical_b} ===")

    for label, fwhm, w_inv, exp in [
        ("pixie", FWHM_PIXIE_DEG, W_MU_INV_PIXIE, "PIXIE"),
        ("specter_ref1deg", FWHM_SPECTER_REF_DEG, W_MU_INV_SPECTER, "SPECTER"),
    ]:
        ell = _ell_for(fwhm)
        if ell.size == 0:
            continue

        # 1a CV: full Gaussian, no mu noise
        f_cv = fisher_1d_fnl_only(
            ell,
            fwhm,
            NS_FID,
            K_D_I,
            K_D_F,
            K_P,
            w_mu_inv=0.0,
            As_fid=AS_FID_PLANCK2018,
            use_b_analytic=not use_numerical_b,
            cl_tt_txt_dir=cl_tt_txt_dir,
            variance_mode=VARIANCE_FULL_GAUSSIAN_CV,
            fnl_fid_for_variance=0.0,
        )
        s_cv = 1.0 / math.sqrt(f_cv) if f_cv > 0 else float("nan")
        emit(f"  [{exp}] 1a_CV_full_gaussian sigma(fNL)_1d = {s_cv:.6e}  F={f_cv:.6e}")

        # 1b/1c instrumental (PZ approx — default)
        f_ins = fisher_1d_fnl_only(
            ell,
            fwhm,
            NS_FID,
            K_D_I,
            K_D_F,
            K_P,
            w_mu_inv=w_inv,
            As_fid=AS_FID_PLANCK2018,
            use_b_analytic=not use_numerical_b,
            cl_tt_txt_dir=cl_tt_txt_dir,
            variance_mode=VARIANCE_PZ_INSTRUMENTAL_APPROX,
        )
        s_ins = 1.0 / math.sqrt(f_ins) if f_ins > 0 else float("nan")
        emit(f"  [{exp}] 1b/c_instr_PZ_approx sigma(fNL)_1d = {s_ins:.6e}  F={f_ins:.6e}")

        # 1e large b (PIXIE only in plan); still run specter for completeness
        for b_ov in (10.0, 100.0):
            f_b = fisher_1d_fnl_only(
                ell,
                fwhm,
                NS_FID,
                K_D_I,
                K_D_F,
                K_P,
                w_mu_inv=w_inv,
                As_fid=AS_FID_PLANCK2018,
                use_b_analytic=not use_numerical_b,
                b_override=b_ov,
                cl_tt_txt_dir=cl_tt_txt_dir,
                variance_mode=VARIANCE_PZ_INSTRUMENTAL_APPROX,
            )
            s_b = 1.0 / math.sqrt(f_b) if f_b > 0 else float("nan")
            emit(f"  [{exp}] 1e_b_override={b_ov:g} sigma(fNL)_1d = {s_b:.6e}  F={f_b:.6e}")


def main():
    camb_dir = str(SCRIPT_DIR)
    has_camb = (SCRIPT_DIR / "cl_tt_fiducial.txt").is_file()

    for pipeline, use_camb, use_num_b in [
        ("analytic_cltt_analytic_b", False, False),
        ("camb_cltt_numerical_b", True, True),
    ]:
        if use_camb and not has_camb:
            print(f"skip pipeline {pipeline}: cl_tt_fiducial.txt not in {camb_dir}")
            continue
        dirs = ensure_section_layout("section1_baseline", pipeline)
        lines: list[str] = []
        cfg = {
            "section": "section1_baseline",
            "pipeline": pipeline,
            "use_camb_cltt": use_camb,
            "use_numerical_b": use_num_b,
            "camb_dir": camb_dir if use_camb else None,
        }
        (dirs["logs"] / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        cl_dir = camb_dir if use_camb else None
        _run_1d_cases(
            pipeline_name=pipeline,
            use_camb_cltt=use_camb,
            use_numerical_b=use_num_b,
            cl_tt_txt_dir=cl_dir,
            lines=lines,
        )

        out_txt = dirs["tables"] / "baseline_results.txt"
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out_txt}")

    agg = forecast_tables_dir()
    agg.mkdir(parents=True, exist_ok=True)
    print(f"aggregate tables dir: {agg}")


if __name__ == "__main__":
    main()
