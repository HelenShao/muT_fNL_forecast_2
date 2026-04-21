'Section 4 runner that writes foreground sweep tables for both pipelines.'

from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))



from config_section4 import A_D_CODE, FNL_FIDUCIALS, SIGMA_AD_PRIOR, SIGMA_ALPHA_PRIOR, SIGMA_NS_PRIOR
from config_section_common import FWHM_PIXIE, FWHM_SPECTER, K_D_F, K_D_I, K_P, NS_FID
from spectra import AS_FID_PLANCK2018
from workflow_section4 import run_pipeline

# Re-export section constants for strict backward compatibility with old imports.
__all__ = [
    "FWHM_PIXIE",
    "FWHM_SPECTER",
    "NS_FID",
    "K_D_I",
    "K_D_F",
    "K_P",
    "FNL_FIDUCIALS",
    "A_D_CODE",
    "SIGMA_NS_PRIOR",
    "SIGMA_AD_PRIOR",
    "SIGMA_ALPHA_PRIOR",
    "main",
]


def _module_dir():
    return Path(__file__).resolve().parent.parent


def main():
    camb_dir = str(_module_dir())
    run_pipeline(
        pipeline="analytic_cltt_analytic_b",
        cl_tt_txt_dir=None,
        use_b_analytic=False,
        as_fid=AS_FID_PLANCK2018,
    )
    run_pipeline(
        pipeline="camb_cltt_analytic_b",
        cl_tt_txt_dir=camb_dir,
        use_b_analytic=True,
        as_fid=AS_FID_PLANCK2018,
    )


if __name__ == "__main__":
    main()
