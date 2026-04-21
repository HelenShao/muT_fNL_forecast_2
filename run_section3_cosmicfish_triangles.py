r"""
Section 3 — CosmicFish **triangle** plots: **PIXIE** and **SPECTER** 3D Fisher
\((f_{\rm NL}, n_s, A_s)\) overlaid on the same figure (one PDF per \(f_{\rm NL}^{\rm fid}\)).

Uses the same Fisher construction as ``cosmicfish_contours.build_muT_fisher`` (numerical \(b(\ell)\),
Planck priors on \(n_s\) and \(A_s\), \(10^9 A_s\) axis scaling for CosmicFish).

Colors: PIXIE ``#3193A2``, SPECTER ``#e76f51``.

Outputs under ``cmbs4/results/muT_fNL_runs/section3_extension/analytic_cltt_analytic_b/figures/``::

    cosmicfish_triangle_pixie_specter_fnl<tag>.pdf

Run::

    export COSMICFISH_PYTHON=/path/to/CosmicFish/python
    python3 run_section3_cosmicfish_triangles.py

    python3 run_section3_cosmicfish_triangles.py --fnl-fiducials 1,25000 \\
        --cosmicfish-python /path/to/CosmicFish/python
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

try:
    from .cosmicfish_contours import (
        _disable_cosmicfish_fisher_protection,
        _import_cosmicfish,
        _resolve_cosmicfish_python,
        build_muT_fisher,
        finish_mu_t_cosmicfish_plot,
        make_cosmicfish_fisher_object,
        mu_t_cosmicfish_plot_style,
        rescale_fisher_As_to_1e9,
    )
    from .output_paths import ensure_section_layout
    from .plot_params import apply_plot_params
    from .run_section3 import FNL_FIDUCIALS
except ImportError:
    from cosmicfish_contours import (
        _disable_cosmicfish_fisher_protection,
        _import_cosmicfish,
        _resolve_cosmicfish_python,
        build_muT_fisher,
        finish_mu_t_cosmicfish_plot,
        make_cosmicfish_fisher_object,
        mu_t_cosmicfish_plot_style,
        rescale_fisher_As_to_1e9,
    )
    from output_paths import ensure_section_layout
    from plot_params import apply_plot_params
    from run_section3 import FNL_FIDUCIALS

PIXIE_COLOR = "#3193A2"
SPECTER_COLOR = "#e76f51"
AS_SCALE = 1e9


def _fnl_file_tag(fnl: float) -> str:
    if math.isfinite(fnl) and abs(fnl - round(fnl)) < 1e-9:
        return str(int(round(fnl)))
    return str(fnl).replace(".", "p")


def _parse_fnl_csv(s: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def _save_triangle_pdf(plotter, out: Path) -> None:
    """
    CosmicFish ``export`` is a thin ``plt.savefig`` wrapper; ``bbox_inches='tight'`` alone often
    drops the **figure** legend. Include the legend in the bbox and add padding.
    """
    fig = plotter.figure
    leg = getattr(plotter, "legend", None)
    kw: dict = {"bbox_inches": "tight", "pad_inches": 0.45}
    if leg is not None:
        kw["bbox_extra_artists"] = (leg,)
    fig.savefig(str(out), **kw)


def main(argv: list[str] | None = None) -> None:
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
    apply_plot_params()

    ap = argparse.ArgumentParser(description="Section 3 CosmicFish triangle: PIXIE vs SPECTER overlay.")
    ap.add_argument(
        "--cosmicfish-python",
        type=Path,
        default=None,
        help="Path to CosmicFish repo's python/ (contains cosmicfish_pylib). Or set COSMICFISH_PYTHON.",
    )
    ap.add_argument(
        "--fnl-fiducials",
        type=str,
        default=",".join(str(x) for x in FNL_FIDUCIALS),
        help="Comma-separated f_NL fiducials (default: same grid as run_section3).",
    )
    args = ap.parse_args(argv)
    fnl_fids = _parse_fnl_csv(args.fnl_fiducials)

    root = _resolve_cosmicfish_python(args.cosmicfish_python)
    fm, fp, fpa = _import_cosmicfish(root)
    _disable_cosmicfish_fisher_protection(fm)

    dirs = ensure_section_layout("section3_extension", "analytic_cltt_analytic_b")
    written: list[Path] = []

    for fnl in fnl_fids:
        F_pix, names_p, fid_p = build_muT_fisher(
            three_params=True, w_mu_label="pixie", fnl_fid=float(fnl)
        )
        F_sp, names_s, fid_s = build_muT_fisher(
            three_params=True, w_mu_label="specter", fnl_fid=float(fnl)
        )
        F_pix, fid_pr, names_r = rescale_fisher_As_to_1e9(
            F_pix, list(fid_p), names_p, scale=AS_SCALE
        )
        F_sp, fid_sr, names_r2 = rescale_fisher_As_to_1e9(
            F_sp, list(fid_s), names_s, scale=AS_SCALE
        )
        if names_r != names_r2 or fid_pr != fid_sr:
            raise RuntimeError("PIXIE/SPECTER Fisher metadata mismatch after A_s rescaling.")

        fish_pix = make_cosmicfish_fisher_object(
            fm, F_pix, list(names_r), list(fid_pr), None, "PIXIE"
        )
        fish_sp = make_cosmicfish_fisher_object(
            fm, F_sp, list(names_r2), list(fid_sr), None, "SPECTER"
        )
        analysis = fpa.CosmicFish_FisherAnalysis(fisher_list=[fish_pix, fish_sp])
        _st = mu_t_cosmicfish_plot_style(["PIXIE", "SPECTER"])
        plotter = fp.CosmicFishPlotter(
            fishers=analysis,
            solid_colors=[PIXIE_COLOR, SPECTER_COLOR],
            line_colors=[PIXIE_COLOR, SPECTER_COLOR],
            labels=["PIXIE", "SPECTER"],
            legend_ncol=2,
            **_st,
        )
        plotter.new_plot()
        # CosmicFish wraps title as $\mathrm{...}$; do not use raw $ inside (see fisher_plot.set_title).
        # tight_layout on triangle plots shrinks the grid and can clip the figure legend; keep CosmicFish layout only.
        plotter.plot_tri(
            title=rf"PIXIE\ vs.\ SPECTER,\ \mu T\ 3D\ Fisher,\ f_{{\rm NL}}^{{\rm fid}}={fnl:g}",
            tight_layout=False,
        )
        finish_mu_t_cosmicfish_plot(
            plotter, dust_alpha_d=False, nudge_triangle_legend=True
        )
        tag = _fnl_file_tag(fnl)
        out = dirs["figures"] / f"cosmicfish_triangle_pixie_specter_fnl{tag}.pdf"
        _save_triangle_pdf(plotter, out)
        written.append(out)
        print(out.resolve(), flush=True)


if __name__ == "__main__":
    main()
