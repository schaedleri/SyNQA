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

The workflow has two stages: `synqa_pipeline.py` runs selection and
validation from raw data; `dualmodel_rescore.py` then re-ranks the
selected taxa and runs the cross-cohort meta-analysis on top of those
results, without re-running simulated annealing, bootstrap, or knockout
analysis.

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
5. **Re-scoring & meta-analysis** (`dualmodel_rescore.py`, post-hoc): taxa
   are re-ranked by a composite `score_causal` (energy contribution x
   bootstrap frequency x sign consistency x stability), a GMM threshold
   splits each cohort's ranking into a high-importance tier, and cohorts
   are combined with a Stouffer meta-analysis. A fairness-check variant
   (`score_causal_strict`) tests whether taxa with no surviving network
   edge are getting an unearned advantage in the ranking.

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

(`dualmodel_rescore.py` uses the same dependencies -- no extra packages needed.)

## Input data

The pipeline expects, per cohort:

- a metadata TSV with sample IDs, a disease/control group column, and
  (optionally) covariate columns (age, BMI, sex) and a read-count column
- a taxonomic abundance TSV (samples x taxa, or taxa x samples) with
  rank-prefixed taxon names (e.g. `s__Escherichia_coli`)

Edit the `DATA_ROOT` and `COHORTS` dictionary near the top of
`synqa_pipeline.py` to point at your own data paths.

## Usage

### Stage 1: selection and validation (`synqa_pipeline.py`)

```bash
python synqa_pipeline.py \
    --regress_covariates age,BMI,sex \
    --cohort all --rank all \
    --run_ko --run_perm \
    --n_seeds 30 --n_boot 1000 --n_perm 2000 \
    --centering_mode gmm --center_source resample \
    --out_dir results/
```

### Stage 2: re-scoring and meta-analysis (`dualmodel_rescore.py`)

Run on the `results/` directory produced by stage 1:

```bash
python dualmodel_rescore.py \
    --results_dir results/ \
    --out_dir results/ \
    --top_n 10 --min_boot_prob 0.5 --min_stability_cross 0.5
```

This does not re-run simulated annealing, bootstrap, or KO -- it only
reads `taxa_{tag}.tsv` and `all_permtest.tsv` from `--results_dir` and
writes its own re-ranked/meta-analysis tables alongside them (see
Output below). Use `--skip_rescore` or `--skip_meta` to run only one of
the two steps.

### Key arguments

**`synqa_pipeline.py`**

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

**`dualmodel_rescore.py`**

| Argument | Description |
|---|---|
| `--results_dir` | Directory containing `synqa_pipeline.py`'s output (`taxa_{tag}.tsv`, `all_permtest.tsv`) |
| `--out_dir` | Where to write outputs (default: same as `--results_dir`) |
| `--top_n` | How many top taxa to print per tag / cross-cohort table |
| `--min_boot_prob` | Only print taxa with `boot_prob >=` this in the per-tag rescore tables |
| `--min_stability_cross` | Stability threshold for the cross-cohort stable-taxa summary |
| `--weight_by_n` | Weight each cohort's z-score by `sqrt(sample size)` instead of equal weighting (meta-analysis only) |
| `--skip_rescore` | Skip the re-ranking step, only run the meta-analysis |
| `--skip_meta` | Skip the meta-analysis step, only run the re-ranking |

Run `python dualmodel_rescore.py --help` for the full list.

## Output

For each cohort/rank combination (tagged `{cohort}_{rank}`), `synqa_pipeline.py`
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

`dualmodel_rescore.py` then adds, per cohort/rank tag:

- `rank_{tag}_Dual_rescored.tsv` -- taxa re-ranked by `score_causal`,
  with the GMM high-importance flag and fairness-check columns
- `bias_check_{tag}.tsv` -- per-taxon F/J score fairness check
  (`score_causal` vs the `score_causal_strict` sensitivity variant)
- `gmm_plots/gmm_threshold_{tag}.jpg` -- histogram of the K=2 GMM fit
  used to set the high-importance threshold

And combined across all tags:

- `all_ranked_rescored.tsv` -- combined re-ranked table
- `bias_check_summary.tsv` -- fairness-check summary across patterns
- `gmm_plots/gmm_threshold_ALL_PATTERNS.jpg` -- combined GMM overview
- `tail_vs_top_networkness.tsv`, `turnover_top_vs_tail.tsv`,
  `top_vs_tail_summary.tsv` -- top-tier vs. tail comparisons (network
  dominance, cross-cohort reproducibility)
- `filter_and_selection_summary.tsv` -- filtering/selection stats per tag
- `cross_cohort_{rank}_rescored.tsv`, `confirmed_candidates_{rank}.tsv` --
  taxa stable across all cohorts, and the strict cross-cohort-stable +
  GMM-high-importance-in-every-cohort subset
- `permutation_test_summary.tsv`, `meta_analysis_summary.tsv` -- the
  reformatted permutation-test table and the Stouffer meta-analysis
  across cohorts

## Citation

If you use this code, please cite:

[citation to be added on publication]

## License

MIT License

Copyright (c) 2026 schaedleri

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
