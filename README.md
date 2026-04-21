# Fisher Forecasts on fNL through mu-T correlations 

Forecast and plotting pipeline for local-type primordial non-Gaussianity constraints from CMB spectral distortions (\(\mu T\) and related Fisher analyses), including:
- baseline 1D forecasts,
- \(\sigma(f_{\rm NL})\) vs \(\ell_{\max}\),
- 3-parameter \((f_{\rm NL}, n_s, A_s)\) forecasts,
- foreground-marginalized 4-parameter \((f_{\rm NL}, n_s, A_D, \alpha_D)\) forecasts.

## 1) Quick setup

From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy matplotlib seaborn cycler
```

Optional (only needed to regenerate CAMB spectra with `planck_cosmology.py`):

```bash
python3 -m pip install camb
```

## 2) Reproduce key paper results

Run all commands from `muT_fNL_forecast_2/`.

### Section 1 baseline

```bash
python3 scripts/run_section1_baseline.py
# Backward-compatible top-level entrypoint also works:
# python3 run_section1_baseline.py
```

Key output tables:
- `.../cmbs4/results/muT_fNL_runs/section1_baseline/analytic_cltt_analytic_b/tables/baseline_results.txt`
- `.../cmbs4/results/muT_fNL_runs/section1_baseline/camb_cltt_numerical_b/tables/baseline_results.txt`

### Section 2 \(\sigma(f_{\rm NL})\) vs \(\ell_{\max}\)

```bash
python3 scripts/sigma_fnl_vs_lmax.py --experiment pixie
python3 scripts/sigma_fnl_vs_lmax.py --experiment specter
# Backward-compatible top-level entrypoint also works:
# python3 sigma_fnl_vs_lmax.py --experiment pixie
```

Key outputs:
- `.../section2_cabass_pixie/analytic_cltt_analytic_b/tables/sigma_fnl_vs_lmax.csv`
- `.../section2_cabass_specter/analytic_cltt_analytic_b/tables/sigma_fnl_vs_lmax.csv`
- corresponding figures under each `figures/` directory.

### Section 3 extension: \((f_{\rm NL}, n_s, A_s)\)

```bash
python3 scripts/run_section3.py
# Backward-compatible top-level entrypoint also works:
# python3 run_section3.py
```

Key outputs:
- `.../section3_extension/analytic_cltt_analytic_b/tables/forecast_3d_grid.txt`
- `.../section3_extension/analytic_cltt_analytic_b/tables/ns_prior_sweep.csv`
- `.../cmbs4/results/section3_fnl_ns_as_table.tex` (helper snippet for paper table updates)

### Section 4 foreground extension: \((f_{\rm NL}, n_s, A_D, \alpha_D)\)

1) Produce main foreground sweep tables:

```bash
python3 scripts/run_section4.py
# Backward-compatible top-level entrypoint also works:
# python3 run_section4.py
```

Key outputs:
- `.../section4_foregrounds/analytic_cltt_analytic_b/tables/foreground_cf_sweep_pixie.csv`
- `.../section4_foregrounds/analytic_cltt_analytic_b/tables/foreground_cf_sweep_specter.csv`
- `.../section4_foregrounds/camb_cltt_analytic_b/tables/foreground_cf_sweep_pixie.csv`
- `.../section4_foregrounds/camb_cltt_analytic_b/tables/foreground_cf_sweep_specter.csv`

2) Regenerate key foreground figures used in the paper:

```bash
python3 plots/plot_sigma_fnl_vs_cf.py --fnl-fid 1 --compare-section3 --pipeline analytic_cltt_analytic_b
python3 plots/plot_fnl_ns_dust_marg_overlay.py --fnl-fid 1 --cf 1000 --pipeline analytic_cltt_analytic_b
python3 plots/plot_fnl_ns_cf.py --experiment pixie --fnl-fid 25000
python3 plots/plot_fnl_ns_cf.py --experiment specter --fnl-fid 25000
python3 plots/plot_sigma_fnl_vs_lmax_cf.py --experiment specter --fnl-fid 1 --pipeline analytic_cltt_analytic_b
# Backward-compatible top-level plot entrypoints also work.
```

## 3) Optional: CosmicFish triangle plots

If CosmicFish is installed:

```bash
export COSMICFISH_PYTHON=/path/to/CosmicFish/python
python3 scripts/run_section3_cosmicfish_triangles.py
python3 scripts/run_section4_cosmicfish_triangles.py --fnl-fid 1 --cf 1000 --modes specter,pixie,overlay --pipeline analytic_cltt_analytic_b
# Backward-compatible top-level entrypoints also work.
```

## 4) Output organization

All section runs use this structure:

`.../cmbs4/results/muT_fNL_runs/<section>/<pipeline>/{figures,tables,logs}/`

with pipeline tags such as:
- `analytic_cltt_analytic_b`
- `camb_cltt_numerical_b`
- `camb_cltt_analytic_b`

## 5) Minimal consistency check

```bash
python3 test_fisher_matrix_regression.py
```

## 6) Notes and troubleshooting

- If CAMB text files (`cl_tt_fiducial.txt`, etc.) are missing, CAMB-based pipelines may skip/fail; analytic pipelines still run.
