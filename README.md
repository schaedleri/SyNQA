# SyNQA -- Synergistic Network QUBO Analysis

SyNQA formulates microbiome biomarker discovery as a quadratic
unconstrained binary optimization (QUBO) problem that jointly optimizes:

- **F**: standardized single-taxon effect size (Welch's t-statistic on
  CLR-transformed abundances between two disease groups)
- **J**: pairwise correlation divergence (Fisher z-transformed
  difference in Pearson correlation between the two groups)

Simulated annealing selects the taxon subset that minimizes the
resulting energy. Selection is validated through bootstrap resampling,
label-permutation testing, and comparison against established
feature-selection and network-inference methods.

## How it works

1. **Preprocessing**: raw counts are filtered by prevalence, CLR-transformed
   with a data-driven pseudo-count, and optionally adjusted for covariates
   (e.g. age, sex, BMI) via per-taxon OLS regression.
2. **F/J construction**: single-taxon effect sizes (F) and pairwise
   correlation divergence (J) are computed from the CLR-transformed data.
   Edge existence uses a fixed threshold (`|Z| >= edge_z_thresh`); the
   centering point used to drive the optimizer is determined by a
   data-driven Gaussian-mixture-model (GMM) procedure (Ashman's D
   separation statistic with a BIC gate), falling back to the empirical
   median when no statistically supported bimodal structure is present.
3. **Selection**: simulated annealing minimizes `E(s) = -F.s - s.J.s`
   over the binary selection vector `s`, run from multiple random seeds
   to check convergence.
4. **Validation**: bootstrap resampling estimates selection stability;
   label-permutation testing assesses whether the number of selected
   edges exceeds chance; knockout (KO) analysis measures each selected
   taxon's contribution to the optimized energy.

Selection (F_energy/J_energy, centered) and interpretation are kept
separate throughout: every reported/interpreted quantity (taxon role,
network centrality, clustering) uses the original signed, degree-
normalized F/J values, not the centered energies used to drive the
optimizer.

## Requirements

```
numpy
pandas
scipy
scikit-learn
networkx
matplotlib
```

Install with:

```bash
pip install -r requirements.txt
```

## Input data

The pipeline expects, per cohort:

- a metadata TSV with sample IDs, a disease/control group column, and
  (optionally) covariate columns (age, BMI, sex) and a read-count column
- a taxonomic abundance TSV (samples x taxa, or taxa x samples) with
  rank-prefixed taxon names (e.g. `s__Escherichia_coli`)

Edit the `DATA_ROOT` and `COHORTS` dictionary near the top of
`synqa_pipeline.py` to point at your own data paths.

## Usage

```bash
python synqa_pipeline.py \
    --regress_covariates age,BMI,sex \
    --cohort all --rank all \
    --run_ko --run_perm \
    --n_seeds 30 --n_boot 1000 --n_perm 2000 \
    --centering_mode gmm --center_source resample \
    --out_dir results/
```

### Key arguments

| Argument | Description |
|---|---|
| `--cohort`, `--rank` | Which cohort(s)/taxonomic rank(s) to run (`s`/`g`/`f`/`all`) |
| `--n_seeds` | Number of simulated-annealing random seeds (for convergence assessment) |
| `--n_boot` | Number of bootstrap resamples (`0` skips bootstrap for a lightweight run) |
| `--run_ko` | Run knockout analysis on the selected taxa |
| `--run_perm` | Run label-permutation testing (`--n_perm`, `--perm_n_seeds`) |
| `--regress_covariates` | Comma-separated covariates to regress out of CLR values (e.g. `age,BMI,sex`) |
| `--centering_mode` | `median` (fixed 50/50 split) or `gmm` (data-driven, falls back to median) |
| `--center_source` | `resample` (recompute the centering point per bootstrap/permutation draw) or `observed` (freeze it at the value computed from the real-label full dataset) |
| `--ashman_d_threshold` | Minimum Ashman's D for the GMM crossing point to be used instead of the median fallback |
| `--edge_z_thresh` | Minimum `|Z|` for a taxon pair to be considered an edge (default 1.96) |
| `--min_prev`, `--group_mode`, `--min_info_prev`, `--noise_floor_pct`, `--info_combine` | Prevalence/noise-floor filtering parameters |

Run `python synqa_pipeline.py --help` for the full list.

## Output

For each cohort/rank combination (tagged `{cohort}_{rank}`), the pipeline
writes to `--out_dir`:

- `taxa_{tag}.tsv` -- selected taxa with effect sizes, bootstrap
  probability, stability, network role, and cluster assignment
- `summary_{tag}.tsv` -- landscape summary (K, convergence diagnostics,
  GMM centering diagnostics, Ashman's D, BIC gap)
- `rank_{tag}_Dual.tsv` -- taxa ranked by causal importance score
- `ko_{tag}_Dual.tsv` -- knockout analysis results (if `--run_ko`)
- `permtest_{tag}.tsv` -- permutation-test results (if `--run_perm`)

Combined across all cohorts/ranks run in one invocation:

- `all_taxa.tsv`, `all_ranked.tsv`, `all_permtest.tsv`
- `gmm_centering_bic_diagnostics.tsv` -- GMM threshold diagnostics
  (method used, Ashman's D, BIC gap) for every cohort/rank pattern

## Citation

If you use this code, please cite:

[citation to be added on publication]

## License

[license to be added]
