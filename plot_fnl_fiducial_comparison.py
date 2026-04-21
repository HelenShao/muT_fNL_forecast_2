'Wide comparison plot: $\\sigma(f_{\\rm NL})$ vs $f_{\\rm NL}^{\\rm fid}$ for PIXIE and SPECTER (3D Fisher).'

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from .plot_params import apply_plot_params
except ImportError:
    from plot_params import apply_plot_params

from output_paths import section_subdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--grid",
        type=str,
        default="",
        help="Path to forecast_3d_grid.txt (optional; recomputes if empty)",
    )
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    if args.grid and Path(args.grid).is_file():
        data = Path(args.grid).read_text(encoding="utf-8")
        fnl_p, s_p, fnl_s, s_s = [], [], [], []
        for line in data.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            exp, fnl, sig = parts[0], float(parts[1]), float(parts[2])
            if exp == "pixie":
                fnl_p.append(fnl)
                s_p.append(sig)
            elif exp == "specter":
                fnl_s.append(fnl)
                s_s.append(sig)
    else:
        from run_section3 import _run_grid

        rp = _run_grid(experiment="pixie")
        rs = _run_grid(experiment="specter")
        fnl_p = [r["fnl_fid"] for r in rp]
        s_p = [r["sigma_fnl_marg"] for r in rp]
        fnl_s = [r["fnl_fid"] for r in rs]
        s_s = [r["sigma_fnl_marg"] for r in rs]

    apply_plot_params()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, MaxNLocator

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fnl_p, s_p, "o-", label="PIXIE (3D marg.)", lw=2)
    ax.plot(fnl_s, s_s, "s-", label="SPECTER (3D marg.)", lw=2)
    ax.set_xlabel(r"$f_{\mathrm{NL}}^{\mathrm{fid}}$")
    ax.set_ylabel(r"$\sigma(f_{\mathrm{NL}})$ (marginalized)")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10, dtype=float)))
    ax.tick_params(axis="y", which="major", length=6, width=1.0, labelsize=10)
    ax.tick_params(axis="y", which="minor", length=3, width=0.8)
    ax.grid(True, which="major", axis="y", ls="-", alpha=0.35)
    ax.grid(True, which="minor", axis="y", ls=":", alpha=0.2)
    ax.legend()
    ax.set_title(r"Fiducial sensitivity comparison ($n_s$, $A_s$ marginalized)")
    fig.tight_layout()

    if args.output:
        out = Path(args.output)
    else:
        out = section_subdir("section3_extension", "analytic_cltt_analytic_b", "figures") / "fnl_fiducial_sigma_comparison.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
