'Paths for paper figures under ``cmbs4/results/`` (next to ``main.tex``).'

from __future__ import annotations

from pathlib import Path


def _cmb_spectral_distortions_root():
    # muT_fNL_forecast_2 -> pajer-zaldarriaga -> cmb-spectral-distortions
    return Path(__file__).resolve().parent.parent.parent


def paper_results_dir():
    """``.../Forecasts-on-fNL-through-CMB-Spectral-Distortions/cmbs4/results``."""
    return (
        _cmb_spectral_distortions_root()
        / "Forecasts-on-fNL-through-CMB-Spectral-Distortions"
        / "cmbs4"
        / "results"
    )


def contours_pixie_dir():
    """PIXIE ``w_mu_inv`` outputs from ``contours.py`` (same filenames as legacy ``contours_variations/``)."""
    return paper_results_dir() / "contours_pixie"


def contours_specter_dir():
    """SPECTER ``w_mu_inv`` outputs; avoids overwriting PIXIE PDFs."""
    return paper_results_dir() / "contours_specter"


def cosmicfish_plots_dir():
    return paper_results_dir() / "cosmicfish_plots"


def forecast_tables_dir():
    """Text tables from ``--save-forecasts``, ``fisher_4d.py``, etc."""
    return paper_results_dir() / "forecast_tables"


def muT_fNL_runs_dir():
    """Root for organized ``figure_plans`` runs: ``cmbs4/results/muT_fNL_runs``."""
    return paper_results_dir() / "muT_fNL_runs"


def section_run_dir(
    section,
    *,
    pipeline,
):
    """Return the base output directory for a section and pipeline combination."""
    return muT_fNL_runs_dir() / section / pipeline


def section_subdir(section, pipeline, name):
    """``name`` in ``figures``, ``tables``, ``logs``."""
    return section_run_dir(section, pipeline=pipeline) / name


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_section_layout(section, pipeline):
    """Create ``figures``, ``tables``, ``logs`` under a section/pipeline and return paths."""
    base = section_run_dir(section, pipeline=pipeline)
    out = {
        "base": ensure_dir(base),
        "figures": ensure_dir(base / "figures"),
        "tables": ensure_dir(base / "tables"),
        "logs": ensure_dir(base / "logs"),
    }
    return out
