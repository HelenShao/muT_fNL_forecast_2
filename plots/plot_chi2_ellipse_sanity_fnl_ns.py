'Sanity check: draw Δχ² = k curves for the 2D Gaussian.'
from __future__ import annotations

from pathlib import Path

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    _ROOT = Path(__file__).resolve().parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))



import matplotlib.pyplot as plt
import numpy as np

from plot_params import apply_plot_params

# From forecast_2d_fnl_ns_grid.txt — f_NL^fid = 1.0, CAMB Cl_TT, numerical b
NS_FID = 0.965
FNL_FID = 1.0

ROWS = {
    "pixie": dict(sigma_fnl=7.23473553e3, sigma_ns=4.0e-3, rho=-8.0e-6),
    "specter": dict(sigma_fnl=2.98624566e2, sigma_ns=4.0e-3, rho=-1.9e-4),
}

DELTA_CHI2 = (2.30, 5.99)


def marginal_cov_2d(sigma_fnl, sigma_ns, rho):
    c12 = rho * sigma_fnl * sigma_ns
    return np.array(
        [
            [sigma_fnl**2, c12],
            [c12, sigma_ns**2],
        ],
        dtype=float,
    )


def ellipse_delta_chi2_from_G(
    G,
    k,
    *,
    n_points = 400,
):
    """Return a parametric ellipse for a fixed delta-chi-squared level."""
    lam, V = np.linalg.eigh(G)
    if np.any(lam <= 0):
        raise ValueError(f"G not positive definite: eigenvalues {lam}")
    t = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    w1 = np.sqrt(k / lam[0]) * np.cos(t)
    w2 = np.sqrt(k / lam[1]) * np.sin(t)
    W = np.stack([w1, w2], axis=0)
    delta = V @ W
    return delta[0], delta[1]


def main():
    apply_plot_params()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    styles = {
        "pixie": dict(color="#3193A2", ls=("-", ":")),
        "specter": dict(color="#C45A62", ls=("-", ":")),
    }

    for name, row in ROWS.items():
        C = marginal_cov_2d(row["sigma_fnl"], row["sigma_ns"], row["rho"])
        G = np.linalg.inv(C)
        lam_G, _ = np.linalg.eigh(G)
        # Ascending order: λ_min ~ 1/σ_fnl², λ_max ~ 1/σ_ns² when |ρ| ≪ 1
        print(f"{name}: eigenvalues of G (marginal precision), ascending: {lam_G}")

        for k, ls in zip(DELTA_CHI2, styles[name]["ls"]):
            dfnl, dns = ellipse_delta_chi2_from_G(G, k)
            fnl = FNL_FID + dfnl
            ns = NS_FID + dns
            ax.plot(
                fnl,
                ns,
                ls=ls,
                color=styles[name]["color"],
                lw=1.8,
                label=rf"{name}: $\Delta\chi^2={k:g}$",
            )

    ax.axhline(NS_FID, color="0.5", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(FNL_FID, color="0.5", lw=0.8, ls="--", alpha=0.5)
    ax.plot(FNL_FID, NS_FID, "k+", ms=10, mew=1.2)
    ax.set_xlabel(r"$f_{\mathrm{NL}}$")
    ax.set_ylabel(r"$n_s$")
    ax.set_title(
        r"Sanity: $\delta\mathbf{p}^{\mathsf T} G\,\delta\mathbf{p} = k$; "
        r"parametric curves (no meshgrid); CAMB 2D table, "
        r"$f_{\mathrm{NL}}^{\mathrm{fid}}=1$"
    )
    ax.grid(True, alpha=0.35, linestyle=":")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()

    out = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "Forecasts-on-fNL-through-CMB-Spectral-Distortions"
        / "cmbs4"
        / "results"
        / "muT_fNL_runs"
        / "section3_extension"
        / "camb_cltt_numerical_b"
        / "figures"
        / "chi2_ellipse_sanity_fnl_ns_fnl1.pdf"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out.resolve())


if __name__ == "__main__":
    main()
