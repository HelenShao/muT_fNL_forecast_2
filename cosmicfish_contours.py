r"""
Contour / triangle plots from a Fisher matrix using the **CosmicFish** Python library
(`cosmicfish_pylib`), which expects a precomputed Fisher matrix — **no MCMC chains**.

Documentation:
  - https://cosmicfish.github.io/documentation/CosmicFishPyLib/introduction.html
  - https://cosmicfish.github.io/documentation/CosmicFishPyLib/example.html.

Setup::

    git clone https://github.com/CosmicFish/CosmicFish.git
    export COSMICFISH_PYTHON=/tmp/CosmicFish/python   # export is required

Or one shot: ``COSMICFISH_PYTHON=... python3 cosmicfish_contours.py`` (variable prefix on the same line as the command).

Run with the bundled muT Fisher (same spirit as ``contours.py`` / ``main_3d.py``)::

    python3 cosmicfish_contours.py

**Noise / degeneracy:** The mu-noise scale ``w_mu_inv`` strongly affects the
``f_{\mathrm{NL}}``–``n_s`` correlation. For SPECTER, run with ``--w-mu specter``
(here) and ``--specter`` in ``contours.py``. Defaults are PIXIE on both.

**2 vs 3 parameters:** A **3×3** Fisher (default here, with ``A_s`` + priors) gives
**marginal** ``f_{\mathrm{NL}}``–``n_s`` contours that match the triple-panel figures
in ``contours.py``. Use ``--two-param-only`` only when you want the legacy
**2-parameter** Fisher (no ``A_s``), as in ``contours_legacy_2param_fnl_ns.pdf``.

Use pre-computed symmetric Fisher matrix (text file, whitespace-separated)::

    python3 cosmicfish_contours.py --fisher-file my_F.dat \\
        --param-names fnl,ns,As --fiducial 25000,0.965,<A_s fiducial>

**CosmicFish vs raw Fisher:** The library’s ``fisher_matrix`` runs ``protect_degenerate()``,
which **rewrites** ``F`` when its condition number exceeds an internal threshold. That can
erase the physical ``f_{\mathrm{NL}}``–``n_s`` degeneracy. This script **disables** that
step by default so ``get_fisher_inverse()`` agrees with ``numpy.linalg.inv(F)`` and with
``contours.py``. Use ``--allow-cosmicfish-protect`` only if you want stock CosmicFish
behavior.

**``A_s`` axis / triangle plots:** Marginal \(\sigma(A_s)\) is \(\sim 10^{-11}\) while
\(\sigma(f_{\mathrm{NL}})\) is \(\sim 10^{3}\). CosmicFish’s automatic axis limits then
collapse the \(A_s\) direction so \((f_{\mathrm{NL}},A_s)\) and \((n_s,A_s)\) panels look
like flat lines. By default this script **reparameterizes** the third parameter to
\(10^{9}A_s\) (Jacobian applied to ``F`` and fiducial). Use ``--no-as-scale`` for raw
\(A_s\) (not recommended for CosmicFish triangle plots).

**Note:** The unrelated PyPI package named ``cosmicfish`` is not this library; need the
GitHub repo ``python/`` directory on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _resolve_cosmicfish_python(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
    else:
        env = os.environ.get("COSMICFISH_PYTHON")
        if env:
            root = Path(env).resolve()
        else:
            print(
                "Error: CosmicFish PyLib path not set.\n"
                "  Clone: git clone https://github.com/CosmicFish/CosmicFish.git\n"
                "  Then either:\n"
                "    export COSMICFISH_PYTHON=/path/to/CosmicFish/python\n"
                "    (you must use export — a plain VAR=value line is not visible to python3)\n"
                "  Or pass:\n"
                "    --cosmicfish-python /path/to/CosmicFish/python\n"
                "Docs: https://cosmicfish.github.io/documentation/CosmicFishPyLib/introduction.html",
                file=sys.stderr,
            )
            sys.exit(1)
    if not (root / "cosmicfish_pylib").is_dir():
        print(
            f"Error: expected package dir {root / 'cosmicfish_pylib'}",
            file=sys.stderr,
        )
        sys.exit(1)
    return root


def _import_cosmicfish(root: Path):
    sys.path.insert(0, str(root))
    import cosmicfish_pylib.fisher_matrix as fm
    import cosmicfish_pylib.fisher_plot as fp
    import cosmicfish_pylib.fisher_plot_analysis as fpa

    return fm, fp, fpa


def _disable_cosmicfish_fisher_protection(fm_module) -> None:
    """
    CosmicFish ``protect_degenerate()`` rebuilds F when cond(F) is large (e.g. muT Fisher
    with very different scales on parameters). That is **not** the same as ``inv(F)`` and
    flattens real correlations. Replace with a no-op so ellipses use the input Fisher matrix.
    """

    def _protect_noop(self, cache=True):
        return None

    fm_module.fisher_matrix.protect_degenerate = _protect_noop


def _symmetrize_fisher(F: np.ndarray) -> np.ndarray:
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] != F.shape[1]:
        raise ValueError("Fisher matrix must be square 2D")
    return 0.5 * (F + F.T)


def _load_fisher_file(path: Path) -> np.ndarray:
    return _symmetrize_fisher(np.loadtxt(path))


def _parse_csv_strings(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _default_latex(names: list[str]) -> list[str]:
    """Reasonable axis labels for standard muT parameter names."""
    m = {
        "fnl": r"f_{\mathrm{NL}}",
        "ns": r"n_s",
        "As": r"A_s",
        "As_1e9": r"10^{9} A_{s}",
    }
    return [m.get(n, n) for n in names]


def rescale_fisher_As_to_1e9(
    F: np.ndarray,
    fid: list[float],
    names: list[str],
    *,
    scale: float,
) -> tuple[np.ndarray, list[float], list[str]]:
    r"""
    If the third parameter is ``As``, use \(\theta_3' = s \theta_3\) with \(s=10^9\).

    Then \(F' = D^{-1} F D^{-1}\) with \(D=\mathrm{diag}(1,1,s)\) so
    \(\Delta\chi^2 = \delta\theta^\top F \delta\theta\) is preserved.
    """
    if len(names) != 3 or names[2] != "As":
        return F, fid, names
    s = float(scale)
    if s == 1.0:
        return F, fid, names
    d = np.array([1.0, 1.0, 1.0 / s])
    Fp = d[:, None] * F * d[None, :]
    fidp = [fid[0], fid[1], fid[2] * s]
    namesp = [names[0], names[1], "As_1e9"]
    return Fp, fidp, namesp


def _rho_fnl_ns_marginal(F: np.ndarray) -> float | None:
    """Pearson corr(f_NL, n_s) from cov = F^{-1} (upper-left 2×2 block)."""
    cov = np.linalg.inv(F)
    sub = cov[np.ix_([0, 1], [0, 1])]
    den = np.sqrt(sub[0, 0] * sub[1, 1])
    if den <= 0:
        return None
    return float(sub[0, 1] / den)


def build_muT_fisher(
    *,
    three_params: bool,
    w_mu_label: str,
) -> tuple[np.ndarray, list[str], list[float]]:
    from beam import W_MU_INV_PIXIE, W_MU_INV_SPECTER
    from fisher_matrix import (
        AS_FID_LEGACY,
        SIGMA_AS_PLANCK2018,
        default_ell_grid,
        fisher_muT_general,
    )

    w_mu_inv = W_MU_INV_SPECTER if w_mu_label.lower() == "specter" else W_MU_INV_PIXIE

    fwhm_deg = 1.6
    fnl_fid = 25_000.0
    ns_fid = 0.965
    k_p = 0.002
    k_D_i = 1.1e4
    k_D_f = 46.0
    ell = default_ell_grid(fwhm_deg)

    if three_params:
        r = fisher_muT_general(
            ell,
            fwhm_deg,
            fnl_fid,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            w_mu_inv=w_mu_inv,
            dns_step=5e-5,
            sigma_ns_prior=0.004,
            sigma_As_prior=SIGMA_AS_PLANCK2018,
            use_b_analytic=False,
        )
        names = ["fnl", "ns", "As"]
        fid = [fnl_fid, ns_fid, AS_FID_LEGACY]
    else:
        r = fisher_muT_general(
            ell,
            fwhm_deg,
            fnl_fid,
            ns_fid,
            k_D_i,
            k_D_f,
            k_p,
            w_mu_inv=w_mu_inv,
            dns_step=5e-5,
            sigma_ns_prior=0.004,
            sigma_As_prior=None,
            include_As=False,
            use_b_analytic=False,
        )
        names = ["fnl", "ns"]
        fid = [fnl_fid, ns_fid]

    F = _symmetrize_fisher(r.F_total)
    return F, names, fid


def _axis_label_is_ns(label: str) -> bool:
    """Detect CosmicFish/matplotlib axis label for :math:`n_s` (string may include ``$...$``)."""
    if not label:
        return False
    s = label.strip()
    return "n_{s}" in s or "n_s" in s


def format_ns_ticks_3f(fig) -> None:
    """
    replace major tick labels with %.3f on n_s axes
    """
    for ax in fig.get_axes():
        if _axis_label_is_ns(ax.get_xlabel()):
            ticks = ax.get_xticks()
            labs = ax.get_xticklabels()
            fs = labs[0].get_fontsize() if labs else "medium"
            ax.set_xticklabels([f"${x:.3f}$" for x in ticks], fontsize=fs)
            ax.xaxis.get_offset_text().set_visible(False)
        if _axis_label_is_ns(ax.get_ylabel()):
            ticks = ax.get_yticks()
            labs = ax.get_yticklabels()
            fs = labs[0].get_fontsize() if labs else "medium"
            ax.set_yticklabels([f"${y:.3f}$" for y in ticks], fontsize=fs)
            ax.yaxis.get_offset_text().set_visible(False)


def make_cosmicfish_fisher_object(
    fm,
    F: np.ndarray,
    param_names: list[str],
    fiducial: list[float],
    param_names_latex: list[str] | None,
    name: str,
):
    latex = param_names_latex if param_names_latex is not None else _default_latex(param_names)
    if len(latex) != len(param_names):
        raise ValueError("param_names_latex length must match param_names")
    fish = fm.fisher_matrix(
        fisher_matrix=F,
        param_names=param_names,
        param_names_latex=latex,
        fiducial=fiducial,
    )
    fish.name = name
    return fish


def run_plots(
    fm,
    fp,
    fpa,
    fish,
    *,
    outdir: Path,
    prefix: str,
    triangle: bool,
    plot_1d: bool,
    show: bool,
) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    analysis = fpa.CosmicFish_FisherAnalysis(fisher_list=fish)
    plotter = fp.CosmicFishPlotter(fishers=analysis, legend_ncol=1)
    saved: list[Path] = []

    # 2D: all unique pairs
    plotter.new_plot()
    plotter.plot2D(title=rf"{fish.name}: 2D marginal contours")
    p2 = outdir / f"{prefix}_2d.pdf"
    format_ns_ticks_3f(plotter.figure)
    plotter.export(str(p2), bbox_inches="tight")
    saved.append(p2)

    if plot_1d:
        plotter.new_plot()
        plotter.plot1D(title=rf"{fish.name}: 1D marginalized")
        p1 = outdir / f"{prefix}_1d.pdf"
        format_ns_ticks_3f(plotter.figure)
        plotter.export(str(p1), bbox_inches="tight")
        saved.append(p1)

    if triangle:
        plotter.new_plot()
        plotter.plot_tri(title=rf"{fish.name}: triangle plot")
        pt = outdir / f"{prefix}_triangle.pdf"
        format_ns_ticks_3f(plotter.figure)
        plotter.export(str(pt), bbox_inches="tight")
        saved.append(pt)

    if show:
        # CosmicFish forces Agg; interactive display may not work. Kept for API parity.
        try:
            import matplotlib.pyplot as plt

            plt.show()
        except Exception:
            pass
    return saved


def main(argv: list[str] | None = None) -> None:
    _script_dir = Path(__file__).resolve().parent
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))

    p = argparse.ArgumentParser(
        description="CosmicFish contour/triangle plots from a Fisher matrix (no MCMC)."
    )
    p.add_argument(
        "--cosmicfish-python",
        type=Path,
        default=None,
        help="Path to the directory that contains the cosmicfish_pylib package "
        "(e.g. .../CosmicFish/python). Overrides COSMICFISH_PYTHON.",
    )
    p.add_argument(
        "--fisher-file",
        type=Path,
        default=None,
        help="Load Fisher matrix from a text file (numeric square matrix).",
    )
    p.add_argument(
        "--param-names",
        type=str,
        default=None,
        help="Comma-separated parameter names (required with --fisher-file).",
    )
    p.add_argument(
        "--fiducial",
        type=str,
        default=None,
        help="Comma-separated fiducial values matching --param-names (required with --fisher-file).",
    )
    p.add_argument(
        "--param-names-latex",
        type=str,
        default=None,
        help="Optional comma-separated LaTeX labels for axes (same order as param-names).",
    )
    p.add_argument(
        "--fisher-name",
        type=str,
        default="muT_Fisher",
        help="Name tag for this Fisher matrix inside CosmicFish (legend).",
    )
    p.add_argument(
        "--two-param-only",
        action="store_true",
        help="Use (f_NL, n_s) Fisher only (no A_s), from fisher_muT_general.",
    )
    p.add_argument(
        "--w-mu",
        choices=("pixie", "specter"),
        default="pixie",
        help="mu-noise preset: pixie (default) or specter (match contours.py --specter).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cosmicfish_plots",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="cosmicfish",
        help="Filename prefix for exported PDFs.",
    )
    p.add_argument(
        "--no-triangle",
        action="store_true",
        help="Skip triangle plot.",
    )
    p.add_argument(
        "--plot-1d",
        action="store_true",
        help="Also write 1D marginalized PDFs.",
    )
    p.add_argument(
        "--save-matrix",
        type=Path,
        default=None,
        help="Save a copy of the Fisher matrix in whitespace-separated .dat form.",
    )
    p.add_argument(
        "--show",
        action="store_true",
        help="Call plt.show() after plotting (CosmicFish uses Agg; may have no effect).",
    )
    p.add_argument(
        "--allow-cosmicfish-protect",
        action="store_true",
        help=(
            "Use CosmicFish's default protect_degenerate() (may change F when ill-conditioned). "
            "Default is to disable it so contours match inv(F) and contours.py."
        ),
    )
    p.add_argument(
        "--as-scale",
        type=float,
        default=1e9,
        metavar="S",
        help=(
            "If the third parameter is named As, plot using S*A_s (Fisher transformed "
            "accordingly). Default 1e9 fixes CosmicFish axis collapse on A_s panels. "
            "Use 1 to disable rescaling."
        ),
    )
    p.add_argument(
        "--no-as-scale",
        action="store_true",
        help="Same as --as-scale 1 (use raw A_s in plots; triangle A_s panels may look flat).",
    )
    args = p.parse_args(argv)
    if args.no_as_scale:
        args.as_scale = 1.0

    root = _resolve_cosmicfish_python(args.cosmicfish_python)
    fm, fp, fpa = _import_cosmicfish(root)
    if not args.allow_cosmicfish_protect:
        _disable_cosmicfish_fisher_protection(fm)
        print(
            "CosmicFish: using input Fisher matrix as-is (protect_degenerate disabled; "
            "see module docstring).",
            file=sys.stderr,
        )

    if args.fisher_file is not None:
        F = _load_fisher_file(args.fisher_file)
        if args.param_names is None or args.fiducial is None:
            p.error("--fisher-file requires --param-names and --fiducial")
        names = _parse_csv_strings(args.param_names)
        fid = _parse_csv_floats(args.fiducial)
        if len(names) != F.shape[0] or len(fid) != len(names):
            raise SystemExit("param-names / fiducial length must match Fisher matrix size")
        latex_arg = (
            _parse_csv_strings(args.param_names_latex)
            if args.param_names_latex
            else None
        )
    else:
        F, names, fid = build_muT_fisher(
            three_params=not args.two_param_only,
            w_mu_label=args.w_mu,
        )
        latex_arg = None

    as_scale = float(args.as_scale)
    if as_scale <= 0:
        raise SystemExit("--as-scale must be positive")
    F, fid, names = rescale_fisher_As_to_1e9(F, fid, names, scale=as_scale)
    if len(names) == 3 and names[2] == "As_1e9" and as_scale != 1.0:
        print(
            f"Reparameterized A_s -> {as_scale:g} * A_s for CosmicFish axes (fnl–As and ns–As panels).",
            file=sys.stderr,
        )

    # Degeneracy diagnostic (matches contours.py conventions for computed Fisher)
    if args.fisher_file is None:
        rho = _rho_fnl_ns_marginal(F)
        if rho is not None:
            print(
                f"Gaussian marginal corr(f_NL, n_s) from inv(F_total) = {rho:.6f} "
                f"({'2-param' if args.two_param_only else '3-param marginal'})",
                file=sys.stderr,
            )
        w_tag = "SPECTER" if args.w_mu == "specter" else "PIXIE"
        print(
            f"muT noise: --w-mu {args.w_mu} ({w_tag}); "
            f"match contours.py noise by using --specter there when using specter here.",
            file=sys.stderr,
        )

    if args.save_matrix is not None:
        np.savetxt(args.save_matrix, F)
        print(f"Saved Fisher matrix to {args.save_matrix}", file=sys.stderr)

    fish = make_cosmicfish_fisher_object(
        fm,
        F,
        names,
        fid,
        latex_arg,
        name=args.fisher_name,
    )

    saved = run_plots(
        fm,
        fp,
        fpa,
        fish,
        outdir=args.output_dir,
        prefix=args.prefix,
        triangle=not args.no_triangle,
        plot_1d=args.plot_1d,
        show=args.show,
    )
    print("Wrote:", file=sys.stderr)
    for path in saved:
        print(f"  {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
