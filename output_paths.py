r"""
Paths for paper figures under ``cmbs4/results/`` (next to ``main.tex``).

All contour and related figure scripts default into this tree so regenerating plots updates
what LaTeX includes without editing paths.

Layout::

    cmbs4/results/
        fnl_ns_*.pdf                    # main_3d.py --fnl-ns-contours
        main_3d_contours.pdf            # main_3d.py default pairwise 3D plot
        contours_pixie/                 # contours.py (default PIXIE)
        contours_specter/               # contours.py --specter / run_specter_contours.py
        cosmicfish_plots/               # cosmicfish_contours.py
        forecast_tables/                # main_3d.py --save-forecasts, fisher_4d text outputs
        muT_fNL_runs/                   # figure_plans drivers: section*/pipeline/{figures,tables,logs}
"""

from __future__ import annotations

from pathlib import Path


def _cmb_spectral_distortions_root() -> Path:
    # muT_fNL_forecast_2 -> pajer-zaldarriaga -> cmb-spectral-distortions
    return Path(__file__).resolve().parent.parent.parent


def paper_results_dir() -> Path:
    """``.../Forecasts-on-fNL-through-CMB-Spectral-Distortions/cmbs4/results``."""
    return (
        _cmb_spectral_distortions_root()
        / "Forecasts-on-fNL-through-CMB-Spectral-Distortions"
        / "cmbs4"
        / "results"
    )


def contours_pixie_dir() -> Path:
    """PIXIE ``w_mu_inv`` outputs from ``contours.py`` (same filenames as legacy ``contours_variations/``)."""
    return paper_results_dir() / "contours_pixie"


def contours_specter_dir() -> Path:
    """SPECTER ``w_mu_inv`` outputs; avoids overwriting PIXIE PDFs."""
    return paper_results_dir() / "contours_specter"


def cosmicfish_plots_dir() -> Path:
    return paper_results_dir() / "cosmicfish_plots"


def forecast_tables_dir() -> Path:
    """Text tables from ``--save-forecasts``, ``fisher_4d.py``, etc."""
    return paper_results_dir() / "forecast_tables"


def muT_fNL_runs_dir() -> Path:
    """Root for organized ``figure_plans`` runs: ``cmbs4/results/muT_fNL_runs``."""
    return paper_results_dir() / "muT_fNL_runs"


def section_run_dir(
    section: str,
    *,
    pipeline: str,
) -> Path:
    """
    ``section``: e.g. ``section1_baseline``, ``section2_cabass``.

    ``pipeline``: ``analytic_cltt_analytic_b`` or ``camb_cltt_numerical_b``.
    """
    return muT_fNL_runs_dir() / section / pipeline


def section_subdir(section: str, pipeline: str, name: str) -> Path:
    """``name`` in ``figures``, ``tables``, ``logs``."""
    return section_run_dir(section, pipeline=pipeline) / name


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_section_layout(section: str, pipeline: str) -> dict[str, Path]:
    """Create ``figures``, ``tables``, ``logs`` under a section/pipeline and return paths."""
    base = section_run_dir(section, pipeline=pipeline)
    out = {
        "base": ensure_dir(base),
        "figures": ensure_dir(base / "figures"),
        "tables": ensure_dir(base / "tables"),
        "logs": ensure_dir(base / "logs"),
    }
    return out
