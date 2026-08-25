"""dualmodel_rescore.py - Post-hoc re-analysis of DualModel results, WITHOUT
re-running SA, bootstrap, or KO analysis."""
import os
import re
import io
import glob
import argparse
import contextlib
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_importance(df_layer):
    """Identical logic to dual_run.py's compute_importance (post-fix)."""
    df = df_layer.copy(); eps = 1e-10
    has_dE = "delta_E" in df.columns and df["delta_E"].notna().any()
    abs_dE = df["delta_E"].abs() if has_dE else pd.Series(np.ones(len(df)), index=df.index)
    boot   = df["boot_prob"].fillna(0)
    str_raw  = df["strength"].fillna(0)
    str_norm = str_raw / (str_raw.max() + eps)

    def _cv_weight(col):
        if col in df.columns:
            return (1.0 / (1.0 + df[col])).fillna(1.0)
        return pd.Series(1.0, index=df.index)

    def _prob_or_neutral(col):
        if col in df.columns:
            return df[col].fillna(1.0)
        return pd.Series(1.0, index=df.index)

    stability_weight_F = _cv_weight("F_cv_boot")
    stability_weight_J = _cv_weight("J_cv_score")
    stability_weight = np.sqrt(stability_weight_F * stability_weight_J)
    df["stability_weight_F"] = stability_weight_F.round(4)
    df["stability_weight_J"] = stability_weight_J.round(4)
    df["stability_weight"]   = stability_weight.round(4)

    F_sign_prob = _prob_or_neutral("F_sign_prob")
    J_sign_prob = _prob_or_neutral("J_sign_prob")
    sign_confidence = np.sqrt(F_sign_prob * J_sign_prob)
    df["sign_confidence"] = sign_confidence.round(4)

    df["score_causal_freq_only"] = (abs_dE * boot).round(6)
    df["score_causal"]  = (abs_dE * boot * sign_confidence * stability_weight).round(6)
    df["score_network"] = (abs_dE * str_norm).round(6)
    df["score_full"]    = (abs_dE * boot * str_norm).round(6)

    for col in ["score_causal", "score_causal_freq_only", "score_network", "score_full"]:
        mx = df[col].max()
        df[f"{col}_norm"] = (df[col] / (mx + eps)).round(4) if mx > 0 else 0.0

    return df.sort_values("score_causal", ascending=False)


def add_taxon_networkness(df):
    """Per-taxon (single-tag) F-vs-J dominance classification, independent
    of the cross-cohort aggregation in cross_cohort_summary(). A taxon
    with degree==0 (no surviving edge to another selected taxon) is
    "isolated": it has no J-side information at all, and is trivially
    F-dominant by construction (I_val==0). This function adds:
      isolated          : bool, degree==0
      networkness       : log10(2*|I|) - log10(|F|), NaN for isolated taxa
      dominant_side_taxon: "F (single-taxon)" / "J (network)" / "F (isolated)"
    """
    df = df.copy()
    eps = 1e-12
    degree = df["degree"] if "degree" in df.columns else (df["I_val"].abs() > eps).astype(int)
    isolated = (degree == 0) if "degree" in df.columns else (df["I_val"].abs() <= eps)
    df["isolated"] = isolated
    f_abs = df["F_val"].abs()
    i_abs = df["I_val"].abs()
    valid = (~isolated) & (f_abs > eps)
    nw = pd.Series(np.nan, index=df.index)
    nw[valid] = np.log10(2 * i_abs[valid]) - np.log10(f_abs[valid])
    df["networkness"] = nw.round(4)
    dom = pd.Series("F (isolated)", index=df.index)
    dom[valid & (nw > 0)] = "J (network)"
    dom[valid & (nw <= 0)] = "F (single-taxon)"
    df["dominant_side_taxon"] = dom
    return df


def compute_score_causal_strict(df):
    """Sensitivity-check variant of score_causal: for taxa with NO J-side
    information (isolated, degree==0), the standard compute_importance()
    fills the missing J_cv_score / J_sign_prob with NEUTRAL-BEST values
    (via .fillna(1.0)), which is equivalent to treating "no J information"
    as "perfectly stable, perfectly sign-consistent J" -- i.e. a free
    pass, rather than a penalty. Since isolated taxa are ALWAYS
    F-dominant by construction, this creates a structural, mechanical
    advantage for F-dominant taxa in score_causal that has nothing to do
    with the strength of their F signal.

    This function instead fills missing J_cv_score / J_sign_prob for
    isolated taxa with the MEDIAN observed value among non-isolated taxa
    IN THE SAME TAG -- i.e., "no information" is treated as "a typical
    (average) J-side penalty", not "the best possible J-side outcome".
    Everything else in the score_causal formula is left unchanged. This
    is one reasonable sensitivity bound, not a claim that this is the
    "correct" formula -- comparing score_causal vs score_causal_strict
    shows how much of the ranking depends on this single modeling choice.
    """
    df = df.copy(); eps = 1e-10
    isolated = df["isolated"] if "isolated" in df.columns else (df["degree"] == 0)
    non_iso = ~isolated
    med_Jcv = df.loc[non_iso, "J_cv_score"].median() if non_iso.any() else np.nan
    med_Jsp = df.loc[non_iso, "J_sign_prob"].median() if non_iso.any() else np.nan

    Jcv_filled = df["J_cv_score"].copy()
    if np.isfinite(med_Jcv):
        Jcv_filled[isolated] = med_Jcv
    stability_weight_J_strict = (1.0 / (1.0 + Jcv_filled)).fillna(1.0)

    Jsp_filled = df["J_sign_prob"].copy()
    if np.isfinite(med_Jsp):
        Jsp_filled[isolated] = med_Jsp
    J_sign_prob_strict = Jsp_filled.fillna(1.0)

    stability_weight_F = (1.0 / (1.0 + df["F_cv_boot"])).fillna(1.0)
    F_sign_prob = df["F_sign_prob"].fillna(1.0) if "F_sign_prob" in df.columns else 1.0

    stability_weight_strict = np.sqrt(stability_weight_F * stability_weight_J_strict)
    sign_confidence_strict = np.sqrt(F_sign_prob * J_sign_prob_strict)

    abs_dE = df["delta_E"].abs() if "delta_E" in df.columns else 1.0
    boot = df["boot_prob"].fillna(0)
    df["score_causal_strict"] = (abs_dE * boot * sign_confidence_strict
                                 * stability_weight_strict).round(6)
    return df


def fj_score_fairness_check(df, tag):
    """Diagnostic for whether score_causal's ranking advantage for
    F-dominant taxa is a real signal or a structural artifact of how
    isolated (no-J-information) taxa are scored.

    Returns a dict summary and a per-taxon DataFrame with the added
    columns from add_taxon_networkness() / compute_score_causal_strict().
    """
    df = add_taxon_networkness(df)
    df = compute_score_causal_strict(df)

    isolated = df["isolated"]
    top = df["gmm_high_importance"] if "gmm_high_importance" in df.columns else pd.Series(False, index=df.index)
    tail = ~top

    frac_iso_top = float(isolated[top].mean()) if top.any() else float("nan")
    frac_iso_tail = float(isolated[tail].mean()) if tail.any() else float("nan")

    non_iso = df[~isolated]
    f_scores = non_iso.loc[non_iso["dominant_side_taxon"] == "F (single-taxon)", "score_causal"]
    j_scores = non_iso.loc[non_iso["dominant_side_taxon"] == "J (network)", "score_causal"]
    if len(f_scores) >= 1 and len(j_scores) >= 1:
        try:
            u_stat, mw_p = stats.mannwhitneyu(f_scores, j_scores, alternative="two-sided")
        except Exception:
            u_stat, mw_p = float("nan"), float("nan")
    else:
        u_stat, mw_p = float("nan"), float("nan")

    valid_ranks = df["score_causal"].notna() & df["score_causal_strict"].notna()
    if valid_ranks.sum() >= 3:
        rank_corr = float(df.loc[valid_ranks, "score_causal"]
                          .corr(df.loc[valid_ranks, "score_causal_strict"], method="spearman"))
    else:
        rank_corr = float("nan")

    rank_orig = df["score_causal"].rank(ascending=False)
    rank_strict = df["score_causal_strict"].rank(ascending=False)
    rank_shift = (rank_strict - rank_orig)
    mean_rank_shift_isolated = float(rank_shift[isolated].mean()) if isolated.any() else float("nan")
    mean_rank_shift_noniso = float(rank_shift[~isolated].mean()) if (~isolated).any() else float("nan")

    n_confirmed_orig = int(top.sum())
    if "score_causal_strict" in df.columns:
        gmm_res = gmm_threshold(df["score_causal_strict"].values)
        n_confirmed_strict = int(gmm_res["n_above"]) if gmm_res is not None else float("nan")
    else:
        n_confirmed_strict = float("nan")

    summary = {
        "tag": tag,
        "n_isolated": int(isolated.sum()), "n_total": int(len(df)),
        "frac_isolated_top": round(frac_iso_top, 4) if not np.isnan(frac_iso_top) else float("nan"),
        "frac_isolated_tail": round(frac_iso_tail, 4) if not np.isnan(frac_iso_tail) else float("nan"),
        "n_F_dominant_noniso": len(f_scores), "n_J_dominant_noniso": len(j_scores),
        "mannwhitney_U": round(float(u_stat), 2) if not np.isnan(u_stat) else float("nan"),
        "mannwhitney_p": round(float(mw_p), 4) if not np.isnan(mw_p) else float("nan"),
        "spearman_corr_orig_vs_strict": round(rank_corr, 4) if not np.isnan(rank_corr) else float("nan"),
        "mean_rank_shift_isolated": round(mean_rank_shift_isolated, 1) if not np.isnan(mean_rank_shift_isolated) else float("nan"),
        "mean_rank_shift_noniso": round(mean_rank_shift_noniso, 1) if not np.isnan(mean_rank_shift_noniso) else float("nan"),
        "n_high_importance_orig": n_confirmed_orig,
        "n_high_importance_strict": n_confirmed_strict,
    }
    return summary, df


def gmm_threshold(scores, k_max=4, n_init=10, seed=0):
    """Fit a 2-component 1-D Gaussian mixture to `scores` (e.g. score_causal).
    Separation between the two components is quantified using Ashman's D
    (Ashman, Bird & Zepf 1994), the standard bimodality-separation
    statistic for a 2-component Gaussian mixture:

        D = sqrt(2) * |mu1 - mu2| / sqrt(sigma1^2 + sigma2^2)

    D > 2 is the conventional cutoff in the mixture-model literature for
    a mixture that is visually/practically bimodal (well-separated)
    given roughly comparable component sizes; this replaces an earlier,
    non-standard "gap in pooled SDs" statistic ((mu1-mu2)/((sigma1+sigma2)/2))
    that is closely related but not equivalent to any commonly-cited
    bimodality criterion.
    """
    from sklearn.mixture import GaussianMixture
    x = np.asarray(scores, dtype=float).reshape(-1, 1)
    x = x[np.isfinite(x[:, 0])]
    n = len(x)
    if n < 6:
        return None
    k_max = min(k_max, max(1, n // 3))

    def top_vs_rest(gm, k):
        means = gm.means_.ravel()
        stds = np.sqrt(gm.covariances_.ravel())
        weights = gm.weights_.ravel()
        top_idx = int(np.argmax(means))
        top_mean, top_std = means[top_idx], stds[top_idx]
        if k == 1:
            return None
        other_idx = 1 - top_idx
        post = gm.predict_proba(x)[:, top_idx]
        rest_mask = post < 0.5
        rest_mean = float(x[rest_mask].mean()) if rest_mask.any() else float(x.min())
        rest_std = float(x[rest_mask].std()) if rest_mask.sum() > 1 else 1e-6
        grid = np.linspace(rest_mean, top_mean, 2000).reshape(-1, 1)
        grid_post = gm.predict_proba(grid)[:, top_idx]
        crossing = float(grid[np.argmin(np.abs(grid_post - 0.5))][0])
        n_above = int((x[:, 0] >= crossing).sum())
        ashman_d = (np.sqrt(2.0) * abs(top_mean - rest_mean)
                    / np.sqrt(top_std ** 2 + rest_std ** 2 + 1e-20))
        separation = float(ashman_d)
        return {
            "threshold": crossing, "top_mean": top_mean, "rest_mean": rest_mean,
            "n_above": n_above, "posterior_high": post, "separation": separation,
            "fitted_top_mean": float(top_mean), "fitted_top_std": float(top_std),
            "fitted_top_weight": float(weights[top_idx]),
            "fitted_rest_mean": float(means[other_idx]), "fitted_rest_std": float(stds[other_idx]),
            "fitted_rest_weight": float(weights[other_idx]),
        }

    gm2 = GaussianMixture(n_components=2, n_init=n_init, random_state=seed).fit(x)
    primary = top_vs_rest(gm2, 2)
    sep = primary["separation"]
    if sep >= 2.0:
        bimodality_note = f"well-separated (Ashman's D={sep:.2f} > 2, K=2 forced)"
    elif sep >= 1.0:
        bimodality_note = f"moderately separated (Ashman's D={sep:.2f}, K=2 forced) -- interpret with some caution"
    else:
        bimodality_note = f"poorly separated (Ashman's D={sep:.2f} < 1, K=2 forced) -- distribution may be unimodal; threshold not strongly justified"

    bic_by_k = {1: float(GaussianMixture(n_components=1, n_init=n_init,
                                          random_state=seed).fit(x).bic(x)),
                2: float(gm2.bic(x))}
    fits = {2: gm2}
    for k in range(3, k_max + 1):
        gm_k = GaussianMixture(n_components=k, n_init=n_init, random_state=seed).fit(x)
        bic_by_k[k] = float(gm_k.bic(x))
        fits[k] = gm_k
    bic_best_k = min(bic_by_k, key=bic_by_k.get)
    bic_agrees_with_k2 = (bic_best_k == 2)

    # Diagnostic: if BIC prefers K>2, does the extra structure fragment the
    # TOP (shrinking the high-mean component -- too strict) or the TAIL
    # (leaving the top component roughly K=2-sized -- harmless/refining)?
    bic_top_n = None
    bic_top_frac_of_k2 = None
    if bic_best_k > 2:
        gm_bic = fits[bic_best_k]
        means_bic = gm_bic.means_.ravel()
        top_idx_bic = int(np.argmax(means_bic))
        labels_bic = gm_bic.predict(x)
        bic_top_n = int((labels_bic == top_idx_bic).sum())
        if primary["n_above"] > 0:
            bic_top_frac_of_k2 = round(bic_top_n / primary["n_above"], 3)

    out = {
        "threshold": primary["threshold"],
        "high_component_mean": primary["top_mean"], "rest_mean": primary["rest_mean"],
        "n_above": primary["n_above"], "n_total": n,
        "separation_sd": float(sep),  # legacy key name, same value as ashman_d
        "ashman_d": float(sep),
        "bimodality_note": bimodality_note,
        "posterior_high": primary["posterior_high"],
        "bic_by_k": bic_by_k, "bic_best_k": bic_best_k,
        "bic_agrees_with_k2": bic_agrees_with_k2,
        "bic_top_n": bic_top_n, "bic_top_frac_of_k2": bic_top_frac_of_k2,
        "fitted_top_mean": primary["fitted_top_mean"], "fitted_top_std": primary["fitted_top_std"],
        "fitted_top_weight": primary["fitted_top_weight"],
        "fitted_rest_mean": primary["fitted_rest_mean"], "fitted_rest_std": primary["fitted_rest_std"],
        "fitted_rest_weight": primary["fitted_rest_weight"],
    }
    return out


def tail_vs_top_networkness(df):
    """df: a single tag's rescored per-taxon table (already has gmm_high_importance)."""
    if "gmm_high_importance" not in df.columns or not {"F_val", "I_val"}.issubset(df.columns):
        return None

    def _summarize(sub):
        f_abs = sub["F_val"].abs()
        i_abs = sub["I_val"].abs()
        isolated = i_abs <= 1e-12
        n_iso = int(isolated.sum())
        valid = (~isolated) & (f_abs > 1e-12)
        nw = np.log10(2 * i_abs[valid]) - np.log10(f_abs[valid])
        frac_j = float((nw > 0).mean()) if len(nw) > 0 else float("nan")
        mean_nw = float(nw.mean()) if len(nw) > 0 else float("nan")
        frac_healthy = (float((sub["role"] == "Healthy_enriched").mean())
                        if "role" in sub.columns and len(sub) > 0 else float("nan"))
        return {"n": len(sub), "n_isolated": n_iso,
                "mean_networkness": mean_nw, "frac_J_dominant": frac_j,
                "frac_healthy_enriched": frac_healthy}

    top = df[df["gmm_high_importance"]]
    tail = df[~df["gmm_high_importance"]]
    return {"top": _summarize(top), "tail": _summarize(tail)}


def plot_gmm_threshold(scores, taxon_names, gmm_result, tag, out_path):
    """Histogram of score_causal with the fitted K=2 GMM component densities overlaid."""
    x = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(x, bins=min(30, max(8, len(x) // 3)), density=True,
            color="lightgray", edgecolor="white", label="score_causal (all taxa)")

    top_m = gmm_result["high_component_mean"]
    rest_m = gmm_result["rest_mean"]
    grid = np.linspace(x.min(), x.max(), 500)
    top_std = gmm_result["fitted_top_std"]
    top_w = gmm_result["fitted_top_weight"]
    rest_std = gmm_result["fitted_rest_std"]
    rest_w = gmm_result["fitted_rest_weight"]
    top_density = top_w * stats.norm.pdf(grid, loc=gmm_result["fitted_top_mean"], scale=top_std)
    rest_density = rest_w * stats.norm.pdf(grid, loc=gmm_result["fitted_rest_mean"], scale=rest_std)
    ax.plot(grid, rest_density, color="steelblue", lw=2, label="rest component (fitted)")
    ax.plot(grid, top_density, color="crimson", lw=2, label="high component (fitted)")
    ax.plot(grid, rest_density + top_density, color="black", lw=1.5, ls="-.",
            label="mixture (sum)")
    ax.axvline(gmm_result["threshold"], color="black", ls="--", lw=1.5,
               label=f"threshold={gmm_result['threshold']:.3f}")
    ax.axvline(top_m, color="crimson", lw=1, ls=":", label=f"high component mean={top_m:.2f}")
    ax.axvline(rest_m, color="steelblue", lw=1, ls=":", label=f"rest mean={rest_m:.2f}")

    bic_str = ("BIC agrees this is the best split (K=2)" if gmm_result["bic_agrees_with_k2"]
              else f"BIC diagnostic: would prefer K={gmm_result['bic_best_k']} "
                   f"(reported threshold still uses K=2; interpret with that in mind)")
    ax.set_title(f"K=2 GMM threshold on score_causal [{tag}]\n"
                 f"{gmm_result['n_above']}/{gmm_result['n_total']} taxa above threshold  "
                 f"({gmm_result['bimodality_note']})\n{bic_str}",
                 fontsize=9, fontweight="bold")
    ax.set_xlabel("score_causal")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def plot_gmm_threshold_grid(gmm_records, out_path):
    """A single combined figure with one small panel per cohort/rank tag."""
    n = len(gmm_records)
    if n == 0:
        return None
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax, (tag, scores, gmm_result) in zip(axes, gmm_records):
        x = np.asarray(scores, dtype=float)
        ax.hist(x, bins=min(25, max(6, len(x) // 3)), density=True,
                color="lightgray", edgecolor="white")
        grid = np.linspace(x.min(), x.max(), 500)
        top_density = (gmm_result["fitted_top_weight"]
                      * stats.norm.pdf(grid, loc=gmm_result["fitted_top_mean"],
                                       scale=gmm_result["fitted_top_std"]))
        rest_density = (gmm_result["fitted_rest_weight"]
                        * stats.norm.pdf(grid, loc=gmm_result["fitted_rest_mean"],
                                         scale=gmm_result["fitted_rest_std"]))
        ax.plot(grid, rest_density, color="steelblue", lw=1.5)
        ax.plot(grid, top_density, color="crimson", lw=1.5)
        ax.plot(grid, rest_density + top_density, color="black", lw=1, ls="-.")
        ax.axvline(gmm_result["threshold"], color="black", ls="--", lw=1.2)
        ax.set_title(f"{tag}\n{gmm_result['n_above']}/{gmm_result['n_total']} above "
                     f"threshold={gmm_result['threshold']:.2f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[len(gmm_records):]:
        ax.axis("off")
    fig.suptitle("K=2 GMM threshold on score_causal, all cohort/rank patterns",
                fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def split_tag(tag):
    """tag = "{cohort_name}_{rank_code}", e.g. "WirbelJ_2019_s"."""
    cohort, rank = tag.rsplit("_", 1)
    return cohort, rank


def try_get_filter_stats(cohort_name, rank_code, **load_data_kwargs):
    """Best-effort, no SA/bootstrap/KO rerun needed: recover p_before_filter,
    p_after_filter, and n_edges_full_network (edge count across ALL
    filtered taxa, matching dual_run.py's own "edges with |Z|>=... (of
    N possible)" console line -- NOT the same quantity as edges among
    only the SA-selected K taxa, which taxa_*.tsv's "degree" column
    cannot recover since it counts each selected taxon's edges to ALL
    filtered taxa, not just to other selected ones). Returns
    (p_before, p_after, n_edges_full_network), any of which may be None
    if dual_run.py / the cohort's data files aren't importable/found
    from the current working directory."""
    try:
        import dual_run as dr
    except ImportError:
        return None, None, None
    paths = dr.COHORTS.get(cohort_name)
    if paths is None or not os.path.exists(paths.get("meta", "")):
        return None, None, None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            X, d, taxa = dr.load_data(paths["meta"], paths["X"], rank_code=rank_code,
                                      **load_data_kwargs)
            dr.build_dual_layer(X, d, verbose=True)
    except Exception:
        return None, None, None
    log = buf.getvalue()
    m_filt = re.search(r"kept=(\d+)/(\d+)", log)
    m_edge = re.search(r"edges with \|Z\|>=[\d.]+: (\d+) \(of (\d+) possible\)", log)
    p_before = p_after = n_edges = None
    if m_filt:
        p_after, p_before = int(m_filt.group(1)), int(m_filt.group(2))
    if m_edge:
        n_edges = int(m_edge.group(1))
    return p_before, p_after, n_edges


def cross_cohort_summary(combined, min_stability=0.5):
    """Re-derive the "taxa stable across all cohorts, for a given rank" table."""
    has_gmm = "gmm_high_importance" in combined.columns
    has_fj = all(c in combined.columns for c in ["F_val", "I_val"])
    results = {}
    for rank in sorted(combined["rank"].unique()):
        sub = combined[(combined["rank"] == rank) & (combined["stability"] >= min_stability)]
        if len(sub) == 0:
            continue
        n_c = sub["cohort"].nunique()
        freq = sub.groupby("taxon").size()
        cross = freq[freq == n_c].index.tolist()
        if not cross:
            continue
        agg_dict = {
            "score_causal": ("score_causal", "mean"),
            "stability": ("stability", "mean"),
            "delta_E": ("delta_E", "mean"),
            "role": ("role", lambda x: x.mode()[0]),
        }
        if has_gmm:
            agg_dict["n_gmm_high"] = ("gmm_high_importance", lambda x: int(x.sum()))
        if has_fj:
            agg_dict["F_val"] = ("F_val", lambda x: x.abs().mean())
            agg_dict["I_val"] = ("I_val", lambda x: x.abs().mean())
            agg_dict["strength"] = ("strength", "mean")
        grp = (sub[sub["taxon"].isin(cross)]
               .groupby("taxon")
               .agg(**agg_dict)
               .reset_index()
               .sort_values("score_causal", ascending=False))
        if has_gmm:
            grp["n_cohorts"] = n_c
            grp["all_gmm_high"] = grp["n_gmm_high"] == n_c
        if has_fj:
            def _mean_networkness_corrected(taxon):
                rows = sub.loc[sub["taxon"] == taxon, ["F_val", "I_val"]]
                f_abs = rows["F_val"].abs()
                i_abs = rows["I_val"].abs()
                isolated = i_abs <= 1e-12
                n_isolated = int(isolated.sum())
                valid = (~isolated) & (f_abs > 1e-12)
                if valid.sum() == 0:
                    return float("nan"), n_isolated
                nw = np.log10(2 * i_abs[valid]) - np.log10(f_abs[valid])
                return float(nw.mean()), n_isolated
            nw_results = grp["taxon"].apply(_mean_networkness_corrected)
            grp["networkness"] = [r[0] for r in nw_results]
            grp["n_isolated_cohorts"] = [r[1] for r in nw_results]
            grp["dominant_side"] = np.where(
                grp["networkness"].isna(), "n/a",
                np.where(grp["networkness"] > 0, "J (network)", "F (single-taxon)"))
        results[rank] = (n_c, grp)
    return results


def confirmed_candidates(cross_results):
    """The strictest, two-criteria "confirmed candidate" list: taxa that are
    cross-cohort stable AND GMM-high-importance in EVERY cohort."""
    out = {}
    for rank, (n_c, grp) in cross_results.items():
        if "all_gmm_high" not in grp.columns:
            out[rank] = pd.DataFrame()
            continue
        out[rank] = grp[grp["all_gmm_high"]].reset_index(drop=True)
    return out


def integrated_cross_cohort_ranking(combined, min_stability=0.5, weight_by_n=False,
                                    n_map=None):
    """Alternative to the headcount-based confirmed-candidate list
    (n_gmm_high == n_cohorts). Rather than requiring a taxon to clear the
    per-cohort GMM high-importance threshold in EVERY cohort (a discrete
    vote that treats cohorts as equally powered/equally-sized, which they
    are not -- e.g. Yachida_2019 n=571 vs Zeller_2014 n=114), this
    standardizes each cohort's score_causal to a within-cohort z-score
    (so cohorts with different score_causal scales are comparable), then
    combines the per-cohort z-scores with Stouffer's method -- the SAME
    combination function already used for the permutation-test
    meta-analysis (stouffer_combine()), just applied here to
    score_causal instead of the permutation null-distribution z-scores.

    This does NOT replace or modify confirmed_candidates() / the
    headcount-based cross-cohort summary -- it is an additional,
    independent ranking saved to its own output file, so a taxon that
    reproduces strongly in 2 of 3 cohorts (e.g. with a much bigger
    effect size in the 2 it does clear) is not automatically excluded
    the way the strict 3/3 headcount rule would exclude it.

    Only taxa with stability >= min_stability in ALL cohorts they occur
    in are considered (same base filter as cross_cohort_summary), and
    only taxa present in at least 2 cohorts are scored (a 1-cohort
    "reproducibility" claim is not meaningful).
    """
    n_map = n_map or DEFAULT_N_MAP
    results = {}
    for rank in sorted(combined["rank"].unique()):
        sub = combined[(combined["rank"] == rank) & (combined["stability"] >= min_stability)].copy()
        if len(sub) == 0:
            continue
        # Within-cohort z-score of score_causal, so cohorts on different scales are comparable.
        sub["z_score_causal"] = (
            sub.groupby("cohort")["score_causal"]
               .transform(lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-10))
        )
        rows = []
        for taxon, g in sub.groupby("taxon"):
            k_c = g["cohort"].nunique()
            if k_c < 2:
                continue
            cohorts = g["cohort"].tolist()
            z_vals = g["z_score_causal"].values
            if weight_by_n:
                w = np.array([np.sqrt(n_map.get(c, 1.0)) for c in cohorts])
            else:
                w = np.ones(len(cohorts))
            combo = stouffer_combine(z_vals, weights=w)
            role_mode = g["role"].mode()
            rows.append({
                "taxon": taxon,
                "n_cohorts_present": k_c,
                "cohorts": ",".join(cohorts),
                "z_per_cohort": ",".join(f"{z:.2f}" for z in z_vals),
                "mean_score_causal": round(float(g["score_causal"].mean()), 4),
                "mean_stability": round(float(g["stability"].mean()), 4),
                "role": role_mode.iloc[0] if len(role_mode) > 0 else "n/a",
                "Z_combined": round(float(combo["Z_combined"]), 4),
                "p_combined": round(float(combo["p_combined"]), 6),
                "I2_pct": round(float(combo["I2_pct"]), 2),
            })
        if not rows:
            continue
        rank_df = (pd.DataFrame(rows)
                  .sort_values("Z_combined", ascending=False)
                  .reset_index(drop=True))
        results[rank] = rank_df
    return results


def rescore_one(taxa_tsv_path, out_dir):
    tag = os.path.basename(taxa_tsv_path)
    tag = tag[len("taxa_"):] if tag.startswith("taxa_") else tag
    tag = tag[:-len(".tsv")] if tag.endswith(".tsv") else tag
    df = pd.read_csv(taxa_tsv_path, sep="\t")
    if len(df) == 0:
        return tag, pd.DataFrame()
    ranked = compute_importance(df)
    out_path = os.path.join(out_dir, f"rank_{tag}_Dual_rescored.tsv")
    ranked.to_csv(out_path, sep="\t", index=False)
    return tag, ranked


def run_rescore(results_dir, out_dir, top_n, min_boot_prob, min_stability_cross):
    taxa_files = sorted(glob.glob(os.path.join(results_dir, "taxa_*.tsv")))
    if not taxa_files:
        print(f"[rescore] No taxa_*.tsv files found in {results_dir}/ -- skipping re-ranking.")
        return None

    print(f"[rescore] Found {len(taxa_files)} taxa_*.tsv file(s) in {results_dir}/")
    print("[rescore] Re-ranking by score_causal = |delta_E| x boot_prob "
          "(F and J weighted equally, per the model's joint optimization).\n")

    all_ranked = []
    gmm_records = []
    tail_vs_top_records = []
    filter_selection_records = []
    bias_check_records = []

    for f in taxa_files:
        tag, ranked = rescore_one(f, out_dir)
        if len(ranked) == 0:
            print(f"  [{tag}] empty, skipped")
            continue
        ranked = ranked.copy()
        ranked["tag"] = tag
        all_ranked.append(ranked)

        K_this = len(ranked)
        n_edges_this = (int(ranked["degree"].sum()) // 2
                        if "degree" in ranked.columns else None)
        rec = {"tag": tag, "K": K_this, "n_edges": n_edges_this}
        # Pull E_cv/Jaccard/K_range from summary_{tag}.tsv when present;
        # older files lack these columns, left as "n/a" rather than
        # re-running SA (which would defeat the point of a cheap post-hoc step).
        summary_path = os.path.join(results_dir, f"summary_{tag}.tsv")
        if os.path.exists(summary_path):
            try:
                summ_df = pd.read_csv(summary_path, sep="\t")
                if len(summ_df) > 0:
                    srow = summ_df.iloc[0]
                    for col in ["E_cv_pct", "sa_jaccard", "K_min_across_seeds", "K_max_across_seeds"]:
                        rec[col] = srow[col] if col in srow.index else "n/a"
            except Exception:
                pass
        for col in ["E_cv_pct", "sa_jaccard", "K_min_across_seeds", "K_max_across_seeds"]:
            rec.setdefault(col, "n/a")
        filter_selection_records.append(rec)

        top = ranked[ranked["boot_prob"] >= min_boot_prob].head(top_n)
        print(f"  [{tag}]  K={len(ranked)}  (rescored file: "
              f"rank_{tag}_Dual_rescored.tsv)")
        if len(top) > 0:
            print(f"    {'score_causal':>12}  {'freq_only(old)':>15}  {'sign_conf':>9}  {'stab_wt':>8}  "
                  f"{'(F)':>6}  {'(J)':>6}  "
                  f"{'|dE|':>7}  {'boot':>5}  {'role':17s}  Taxon")
            print("    " + "-" * 100)
            for _, r in top.iterrows():
                name = (r["taxon"].replace("s__", "").replace("g__", "")
                        .replace("f__", ""))[:30]
                de = f"{r['delta_E']:.4f}" if not np.isnan(r["delta_E"]) else "  -   "
                print(f"    {r['score_causal']:>12.4f}  {r['score_causal_freq_only']:>15.4f}  "
                      f"{r['sign_confidence']:>9.3f}  {r['stability_weight']:>8.3f}  "
                      f"{r['stability_weight_F']:>6.3f}  {r['stability_weight_J']:>6.3f}  "
                      f"{de:>7}  {r['boot_prob']:>5.2f}  {r['role']:17s}  {name}")

        gmm_res = gmm_threshold(ranked["score_causal"].values)
        if gmm_res is not None:
            ranked["gmm_high_importance"] = (
                ranked["score_causal"].values >= gmm_res["threshold"])
            bic_str = ", ".join(f"K={k}:{v:.1f}" for k, v in gmm_res["bic_by_k"].items())
            print(f"    [GMM threshold, K=2] score_causal >= {gmm_res['threshold']:.4f}  "
                  f"({gmm_res['n_above']}/{gmm_res['n_total']} taxa)  "
                  f"high_mean={gmm_res['high_component_mean']:.3f}  "
                  f"rest_mean={gmm_res['rest_mean']:.3f}  "
                  f"{gmm_res['bimodality_note']}")
            print(f"    [GMM threshold, BIC diagnostic] BIC by K: {bic_str}  "
                  f"(best K={gmm_res['bic_best_k']})")
            if not gmm_res["bic_agrees_with_k2"]:
                bic_top_n = gmm_res.get("bic_top_n")
                bic_frac = gmm_res.get("bic_top_frac_of_k2")
                if bic_top_n is not None:
                    if bic_frac is not None and bic_frac <= 0.5:
                        interp = ("fragmenting the TOP: adopting BIC's K would shrink "
                                 "the top group substantially -- keep K=2")
                    else:
                        interp = ("fragmenting the TAIL: the top (highest-mean) component "
                                 "is a similar size to the K=2 top group -- adopting BIC's K "
                                 "would not meaningfully change, and could refine, the split")
                    print(f"    [GMM threshold] NOTE: BIC would prefer K={gmm_res['bic_best_k']} "
                          f"over K=2. Under K={gmm_res['bic_best_k']}, the highest-mean "
                          f"component contains {bic_top_n}/{gmm_res['n_total']} taxa "
                          f"(vs {gmm_res['n_above']}/{gmm_res['n_total']} under K=2, "
                          f"ratio={bic_frac}) -- {interp}. The reported K=2 threshold above "
                          f"is still what's used for gmm_high_importance.")
                else:
                    print(f"    [GMM threshold] NOTE: BIC would prefer K={gmm_res['bic_best_k']} "
                          f"over K=2 -- the reported K=2 threshold above is still what's "
                          f"used for gmm_high_importance.")
            gmm_plot_dir = os.path.join(out_dir, "gmm_plots")
            os.makedirs(gmm_plot_dir, exist_ok=True)
            gmm_fig_path = os.path.join(gmm_plot_dir, f"gmm_threshold_{tag}.jpg")
            plot_gmm_threshold(ranked["score_causal"].values, ranked["taxon"].values,
                               gmm_res, tag, gmm_fig_path)
            print(f"    [GMM threshold] figure saved: {gmm_fig_path}")
            gmm_records.append((tag, ranked["score_causal"].values, gmm_res))

            # Re-save with the gmm_high_importance column included.
            ranked.to_csv(os.path.join(out_dir, f"rank_{tag}_Dual_rescored.tsv"),
                         sep="\t", index=False)

            tvt = tail_vs_top_networkness(ranked)
            if tvt is not None:
                t, l = tvt["top"], tvt["tail"]
                print(f"    [Top vs tail networkness] "
                      f"top(n={t['n']}, isolated={t['n_isolated']}): "
                      f"mean_net={t['mean_networkness']:.3f}  "
                      f"frac_J_dominant={t['frac_J_dominant']:.1%}  "
                      f"frac_Healthy_enriched={t['frac_healthy_enriched']:.1%}   |   "
                      f"tail(n={l['n']}, isolated={l['n_isolated']}): "
                      f"mean_net={l['mean_networkness']:.3f}  "
                      f"frac_J_dominant={l['frac_J_dominant']:.1%}  "
                      f"frac_Healthy_enriched={l['frac_healthy_enriched']:.1%}")
                tail_vs_top_records.append({
                    "tag": tag,
                    "top_n": t["n"], "top_mean_net": t["mean_networkness"],
                    "top_frac_J": t["frac_J_dominant"],
                    "top_frac_healthy": t["frac_healthy_enriched"],
                    "tail_n": l["n"], "tail_mean_net": l["mean_networkness"],
                    "tail_frac_J": l["frac_J_dominant"],
                    "tail_frac_healthy": l["frac_healthy_enriched"],
                    "ashman_d": gmm_res["ashman_d"],
                    "gmm_threshold_value": gmm_res["threshold"],
                    "bic_best_k": gmm_res["bic_best_k"],
                    "bic_agrees_with_k2": gmm_res["bic_agrees_with_k2"],
                    "bic_top_n": gmm_res.get("bic_top_n"),
                    "bic_top_frac_of_k2": gmm_res.get("bic_top_frac_of_k2"),
                })

            bias_summary, bias_df = fj_score_fairness_check(ranked, tag)
            bias_check_records.append(bias_summary)
            bias_df.to_csv(os.path.join(out_dir, f"bias_check_{tag}.tsv"),
                           sep="\t", index=False)
            mw_p_str = (f"{bias_summary['mannwhitney_p']:.4f}"
                       if not (isinstance(bias_summary['mannwhitney_p'], float)
                              and np.isnan(bias_summary['mannwhitney_p'])) else "n/a")
            print(f"    [F/J score fairness check] isolated taxa: "
                  f"{bias_summary['n_isolated']}/{bias_summary['n_total']}  "
                  f"frac_isolated(top)={bias_summary['frac_isolated_top']:.1%}  "
                  f"frac_isolated(tail)={bias_summary['frac_isolated_tail']:.1%}")
            print(f"    [F/J score fairness check] among NON-isolated taxa only: "
                  f"F-dominant n={bias_summary['n_F_dominant_noniso']}, "
                  f"J-dominant n={bias_summary['n_J_dominant_noniso']}, "
                  f"Mann-Whitney U p={mw_p_str} "
                  f"(score_causal, F-dominant vs J-dominant)")
            print(f"    [F/J score fairness check] score_causal vs score_causal_strict "
                  f"(isolated taxa's missing J-info filled with the tag's typical/median "
                  f"penalty instead of a free pass): Spearman rho="
                  f"{bias_summary['spearman_corr_orig_vs_strict']}  "
                  f"high-importance count: {bias_summary['n_high_importance_orig']} (orig) "
                  f"vs {bias_summary['n_high_importance_strict']} (strict)  "
                  f"mean rank shift: isolated={bias_summary['mean_rank_shift_isolated']:+.1f}, "
                  f"non-isolated={bias_summary['mean_rank_shift_noniso']:+.1f} "
                  f"(positive = ranked worse under the strict score)")
        else:
            print(f"    [GMM threshold] skipped (fewer than 6 taxa)")
        print()

    if not all_ranked:
        return None

    combined = pd.concat(all_ranked, ignore_index=True)
    combined[["cohort", "rank"]] = combined["tag"].apply(lambda t: pd.Series(split_tag(t)))
    combined_path = os.path.join(out_dir, "all_ranked_rescored.tsv")
    combined.to_csv(combined_path, sep="\t", index=False)
    print(f"[rescore] Saved combined ranking: {combined_path}")

    if bias_check_records:
        bias_summary_df = pd.DataFrame(bias_check_records)
        bias_summary_path = os.path.join(out_dir, "bias_check_summary.tsv")
        bias_summary_df.to_csv(bias_summary_path, sep="\t", index=False)
        print(f"\n{'='*70}")
        print("F/J SCORE FAIRNESS CHECK (across all cohort/rank patterns)")
        print("Does score_causal's F-dominant advantage survive when isolated")
        print("(no-J-information) taxa no longer get a free pass on the J side?")
        print(f"{'='*70}")
        print(f"  {'tag':20s}  {'iso_top':>8}  {'iso_tail':>9}  {'MW_p':>7}  "
              f"{'spearman':>9}  {'n_high(orig)':>12}  {'n_high(strict)':>14}")
        for _, r in bias_summary_df.iterrows():
            mw = f"{r['mannwhitney_p']:.4f}" if pd.notna(r['mannwhitney_p']) else "n/a"
            print(f"  {r['tag']:20s}  {r['frac_isolated_top']*100:>7.1f}%  "
                  f"{r['frac_isolated_tail']*100:>8.1f}%  {mw:>7}  "
                  f"{r['spearman_corr_orig_vs_strict']:>9.3f}  "
                  f"{r['n_high_importance_orig']:>12}  {r['n_high_importance_strict']:>14}")
        n_sig = int((bias_summary_df["mannwhitney_p"] < 0.05).sum())
        n_tested = int(bias_summary_df["mannwhitney_p"].notna().sum())
        print(f"\n  Mann-Whitney U (F-dominant vs J-dominant score_causal, "
              f"non-isolated taxa only): {n_sig}/{n_tested} patterns significant (p<0.05)")
        print(f"  (saved: {bias_summary_path})")

    if gmm_records:
        grid_path = os.path.join(out_dir, "gmm_plots", "gmm_threshold_ALL_PATTERNS.jpg")
        plot_gmm_threshold_grid(gmm_records, grid_path)
        print(f"[rescore] Combined GMM overview figure saved: {grid_path}")

    if filter_selection_records:
        rows = []
        for rec in filter_selection_records:
            cohort_name, rank_code = split_tag(rec["tag"])
            p_before, p_after, n_edges_full = try_get_filter_stats(cohort_name, rank_code)
            rows.append({
                "cohort_rank": rec["tag"],
                "p_before_filter": p_before if p_before is not None else "n/a",
                "p_after_filter": p_after if p_after is not None else "n/a",
                "K": rec["K"],
                "n_edges_full_network": n_edges_full if n_edges_full is not None else "n/a",
                "n_edges_within_selected_K_approx": rec["n_edges"] if rec["n_edges"] is not None else "n/a",
                "n_pairs_total": (p_after * (p_after - 1) // 2) if p_after is not None else "n/a",
                "E_cv_pct": rec.get("E_cv_pct", "n/a"),
                "sa_jaccard": rec.get("sa_jaccard", "n/a"),
                "K_min_across_seeds": rec.get("K_min_across_seeds", "n/a"),
                "K_max_across_seeds": rec.get("K_max_across_seeds", "n/a"),
            })
        filt_sel_df = pd.DataFrame(rows)
        filt_sel_path = os.path.join(out_dir, "filter_and_selection_summary.tsv")
        filt_sel_df.to_csv(filt_sel_path, sep="\t", index=False)
        n_na = int((filt_sel_df["p_before_filter"] == "n/a").sum())
        if n_na > 0:
            print(f"[rescore] Filter/selection summary saved: {filt_sel_path} "
                  f"({n_na}/{len(filt_sel_df)} rows missing p_before/p_after -- "
                  f"dual_run.py's COHORTS dict/data files not found from this "
                  f"working directory).")
        else:
            print(f"[rescore] Filter/selection summary saved: {filt_sel_path}")

    if tail_vs_top_records:
        tvt_df = pd.DataFrame(tail_vs_top_records)
        tvt_path = os.path.join(out_dir, "tail_vs_top_networkness.tsv")
        tvt_df.to_csv(tvt_path, sep="\t", index=False)
        print(f"\n{'='*70}")
        print("TOP-TIER vs. TAIL: is the network term's added value concentrated")
        print("in the tail of each cohort's own K-taxa selection (not just the")
        print("cross-cohort-common taxa examined above)?")
        print(f"{'='*70}")
        print(f"  {'tag':20s}  {'top_n':>6}  {'top_frac_J':>11}  {'tail_n':>7}  {'tail_frac_J':>12}")
        for _, r in tvt_df.iterrows():
            print(f"  {r['tag']:20s}  {r['top_n']:>6}  {r['top_frac_J']*100:>10.1f}%  "
                  f"{r['tail_n']:>7}  {r['tail_frac_J']*100:>11.1f}%")
        mean_top_frac_j = tvt_df["top_frac_J"].mean()
        mean_tail_frac_j = tvt_df["tail_frac_J"].mean()
        print(f"\n  Mean across {len(tvt_df)} cohort/rank patterns: "
              f"top={mean_top_frac_j:.1%} J-dominant, tail={mean_tail_frac_j:.1%} J-dominant")
        print(f"  (saved: {tvt_path})")

    n_cohorts = combined["cohort"].nunique()
    if n_cohorts > 1:
        print(f"\n{'='*70}")
        print(f"CROSS-COHORT STABLE TAXA (stability>={min_stability_cross}, "
              f"re-ranked by mean score_causal)")
        print(f"{'='*70}")
        cross = cross_cohort_summary(combined, min_stability=min_stability_cross)
        for rank, (n_c, grp) in cross.items():
            print(f"\n  [{rank}]: {len(grp)} taxa in all {n_c} cohorts")
            cross_path = os.path.join(out_dir, f"cross_cohort_{rank}_rescored.tsv")
            grp.to_csv(cross_path, sep="\t", index=False)
            has_gmm = "all_gmm_high" in grp.columns
            has_fj = "networkness" in grp.columns
            for _, r in grp.iterrows():
                name = r["taxon"].replace("s__", "").replace("g__", "")[:38]
                de = (f"{r['delta_E']:.4f}" if not np.isnan(r["delta_E"]) else "  -  ")
                gmm_str = (f"  gmm_high={int(r['n_gmm_high'])}/{n_c}"
                          + (" *CONFIRMED*" if has_gmm and r["all_gmm_high"] else "")
                          if has_gmm else "")
                fj_str = ""
                if has_fj:
                    nw = r["networkness"]
                    nw_str = f"{nw:+.2f}" if not np.isnan(nw) else "  n/a"
                    iso = int(r["n_isolated_cohorts"])
                    iso_str = f" (isolated in {iso}/{n_c})" if iso > 0 else ""
                    fj_str = (f"  F={r['F_val']:.3f} I={r['I_val']:.3f} "
                              f"net={nw_str} [{r['dominant_side']}]{iso_str}")
                print(f"    score_causal={r['score_causal']:>7.4f}  "
                      f"stab={r['stability']:.2f}  dE={de}  "
                      f"{r['role']:17s}  {name}{gmm_str}{fj_str}")
            print(f"    (saved: {cross_path})")

        print(f"\n{'='*70}")
        print("TOP-TIER vs. TAIL: cross-cohort reproducibility (is the tail")
        print("more likely to be cohort-specific / interchangeable, while the")
        print("top tier is what actually reproduces across cohorts?)")
        print(f"{'='*70}")
        turnover_records = []
        cross_taxa_by_rank = {rank: set(grp["taxon"]) for rank, (_, grp) in cross.items()}
        print(f"  {'tag':20s}  {'top_n':>6}  {'top_in_cross':>13}  "
              f"{'tail_n':>7}  {'tail_in_cross':>14}")
        for ranked_df in all_ranked:
            if "gmm_high_importance" not in ranked_df.columns:
                continue
            tag = ranked_df["tag"].iloc[0]
            _, rank_code = split_tag(tag)
            cross_set = cross_taxa_by_rank.get(rank_code, set())
            if not cross_set:
                continue
            top = ranked_df[ranked_df["gmm_high_importance"]]
            tail = ranked_df[~ranked_df["gmm_high_importance"]]
            top_in_cross = float(top["taxon"].isin(cross_set).mean()) if len(top) > 0 else float("nan")
            tail_in_cross = float(tail["taxon"].isin(cross_set).mean()) if len(tail) > 0 else float("nan")
            turnover_records.append({
                "tag": tag, "top_n": len(top), "top_frac_in_cross_cohort": top_in_cross,
                "tail_n": len(tail), "tail_frac_in_cross_cohort": tail_in_cross,
            })
            print(f"  {tag:20s}  {len(top):>6}  {top_in_cross*100:>12.1f}%  "
                  f"{len(tail):>7}  {tail_in_cross*100:>13.1f}%")
        if turnover_records:
            turnover_df = pd.DataFrame(turnover_records)
            turnover_path = os.path.join(out_dir, "turnover_top_vs_tail.tsv")
            turnover_df.to_csv(turnover_path, sep="\t", index=False)
            mean_top = turnover_df["top_frac_in_cross_cohort"].mean()
            mean_tail = turnover_df["tail_frac_in_cross_cohort"].mean()
            print(f"\n  Mean across {len(turnover_df)} cohort/rank patterns: "
                  f"top={mean_top:.1%} in cross-cohort-common set, "
                  f"tail={mean_tail:.1%} in cross-cohort-common set")
            print(f"  (saved: {turnover_path})")

            if tail_vs_top_records:
                tvt_df = pd.DataFrame(tail_vs_top_records)
                combined_tvt = turnover_df.merge(tvt_df, on="tag", how="outer",
                                                 suffixes=("", "_nw"))
                summary_rows = []
                for _, r in combined_tvt.iterrows():
                    summary_rows.append({
                        "cohort_rank": r["tag"],
                        "top_n": int(r["top_n"]),
                        "tail_n": int(r["tail_n"]),
                        "top_frac_J_dominant": f"{r['top_frac_J']*100:.1f}%" if pd.notna(r.get("top_frac_J")) else "n/a",
                        "tail_frac_J_dominant": f"{r['tail_frac_J']*100:.1f}%" if pd.notna(r.get("tail_frac_J")) else "n/a",
                        "top_frac_healthy_enriched": f"{r['top_frac_healthy']*100:.1f}%" if pd.notna(r.get("top_frac_healthy")) else "n/a",
                        "tail_frac_healthy_enriched": f"{r['tail_frac_healthy']*100:.1f}%" if pd.notna(r.get("tail_frac_healthy")) else "n/a",
                        "top_cross_cohort_match_rate": f"{r['top_frac_in_cross_cohort']*100:.1f}%",
                        "tail_cross_cohort_match_rate": f"{r['tail_frac_in_cross_cohort']*100:.1f}%",
                        "ashman_d": round(float(r["ashman_d"]), 4) if pd.notna(r.get("ashman_d")) else "n/a",
                        "gmm_threshold_value": round(float(r["gmm_threshold_value"]), 4) if pd.notna(r.get("gmm_threshold_value")) else "n/a",
                        "bic_agrees_with_k2": bool(r["bic_agrees_with_k2"]) if pd.notna(r.get("bic_agrees_with_k2")) else "n/a",
                        "bic_top_n": int(r["bic_top_n"]) if pd.notna(r.get("bic_top_n")) else "n/a",
                        "bic_top_frac_of_k2": round(float(r["bic_top_frac_of_k2"]), 3) if pd.notna(r.get("bic_top_frac_of_k2")) else "n/a",
                    })
                summary_df = pd.DataFrame(summary_rows)
                summary_path = os.path.join(out_dir, "top_vs_tail_summary.tsv")
                summary_df.to_csv(summary_path, sep="\t", index=False)
                print(f"  (combined summary saved: {summary_path})")

        confirmed = confirmed_candidates(cross)
        any_confirmed = any(len(df) > 0 for df in confirmed.values())
        if any_confirmed:
            print(f"\n{'='*70}")
            print("CONFIRMED CANDIDATES (cross-cohort stable AND GMM-high-importance "
                  "in EVERY cohort)")
            print(f"{'='*70}")
            for rank, df in confirmed.items():
                if len(df) == 0:
                    continue
                conf_path = os.path.join(out_dir, f"confirmed_candidates_{rank}.tsv")
                df.to_csv(conf_path, sep="\t", index=False)
                print(f"\n  [{rank}]: {len(df)} confirmed taxa (saved: {conf_path})")
                for _, r in df.iterrows():
                    name = r["taxon"].replace("s__", "").replace("g__", "")[:38]
                    print(f"    score_causal={r['score_causal']:>7.4f}  "
                          f"stab={r['stability']:.2f}  {r['role']:17s}  {name}")
        elif "gmm_high_importance" in combined.columns:
            print(f"\n  No taxa were GMM-high-importance in EVERY cohort for any rank "
                  f"-- no confirmed-candidate list produced (cross-cohort tables above "
                  f"still show n_gmm_high per taxon).")

    return combined


META_STATS = ["n_edges", "K", "purity"]
META_RANKS = ["s", "g", "f"]
DEFAULT_N_MAP = {
    "WirbelJ_2019": 125,
    "Zeller_2014": 114,
    "Yachida_2019": 571,
}


def load_permtest_table(results_dir):
    path = os.path.join(results_dir, "all_permtest.tsv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, sep="\t")


def save_permtest_summary(results_dir, out_dir):
    """Re-save dual_run.py's raw all_permtest.tsv as a clean, readable
    per-cohort/rank summary table (permutation_test_summary.tsv), with a
    significance flag column added for each statistic tested (p<0.05).
    Does NOT rerun the permutation test itself (that's dual_run.py's own
    expensive step, this only reformats its already-saved output).
    Automatically adapts to whichever statistic columns are present --
    e.g. "purity" was dropped from later pipeline versions that no
    longer perform clustering, so a table produced by such a version
    won't have obs_purity/null_purity_*/p_purity/dir_purity columns, and
    this function simply omits them rather than erroring."""
    df = load_permtest_table(results_dir)
    if df is None:
        print(f"[rescore] No all_permtest.tsv found in {results_dir}/ -- "
              f"skipping permutation_test_summary.tsv (run dual_run.py "
              f"with --run_perm to generate it).")
        return None
    stats_present = [s for s in META_STATS if f"obs_{s}" in df.columns]
    cols = ["tag", "cohort", "rank", "n_perm", "perm_n_seeds"]
    for s in stats_present:
        cols += [f"obs_{s}", f"null_{s}_mean", f"null_{s}_std", f"p_{s}", f"dir_{s}"]
    out = df[[c for c in cols if c in df.columns]].copy()
    for s in stats_present:
        p_col = f"p_{s}"
        if p_col in out.columns:
            out[f"sig_{s}_p<0.05"] = out[p_col] < 0.05
    out_path = os.path.join(out_dir, "permutation_test_summary.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    print(f"\n[rescore] Permutation test summary saved: {out_path}")
    for s in stats_present:
        n_sig = int((out[f"p_{s}"] < 0.05).sum())
        dirs = out[f"dir_{s}"].value_counts().to_dict()
        print(f"    {s}: {n_sig}/{len(out)} patterns significant (p<0.05); "
              f"direction breakdown: {dirs}")
    return out


def stouffer_combine(z_values, weights=None):
    z_values = np.asarray(z_values, dtype=float)
    k = len(z_values)
    if weights is None:
        weights = np.ones(k)
    else:
        weights = np.asarray(weights, dtype=float)
    Z_combined = np.sum(weights * z_values) / np.sqrt(np.sum(weights ** 2))
    p_combined = 2 * stats.norm.sf(abs(Z_combined))
    z_bar = np.sum(weights * z_values) / np.sum(weights)
    Q = np.sum(weights * (z_values - z_bar) ** 2)
    df = max(k - 1, 1)
    Q_p = stats.chi2.sf(Q, df) if k > 1 else float("nan")
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    return {
        "Z_combined": Z_combined, "p_combined": p_combined,
        "Q": Q, "Q_df": df, "Q_p": Q_p, "I2_pct": I2,
    }


def run_meta_analysis(results_dir, weight_by_n=False, n_map=None):
    df = load_permtest_table(results_dir)
    if df is None:
        print(f"[meta] No all_permtest.tsv found in {results_dir}/ -- skipping meta-analysis.")
        return None, set()
    if "cohort" not in df.columns or "rank" not in df.columns:
        print("[meta] all_permtest.tsv is missing 'cohort'/'rank' columns -- "
              "skipping meta-analysis (need dual_run.py --run_perm across "
              "multiple cohorts/ranks).")
        return None, set()
    n_map = n_map or DEFAULT_N_MAP
    rows = []
    skipped_stats = set()
    for rank in META_RANKS:
        sub_rank = df[df["rank"] == rank]
        if len(sub_rank) == 0:
            continue
        for stat in META_STATS:
            obs_col, mean_col, std_col = f"obs_{stat}", f"null_{stat}_mean", f"null_{stat}_std"
            if obs_col not in sub_rank.columns:
                continue
            if std_col not in sub_rank.columns:
                skipped_stats.add(stat)
                continue
            cohorts = sub_rank["cohort"].tolist()
            obs = sub_rank[obs_col].values.astype(float)
            null_mean = sub_rank[mean_col].values.astype(float)
            null_std = sub_rank[std_col].values.astype(float)
            null_std_safe = np.where(null_std > 1e-8, null_std, np.nan)
            z_i = (obs - null_mean) / null_std_safe
            valid = np.isfinite(z_i)
            if valid.sum() < 2:
                continue
            if weight_by_n:
                w = np.array([np.sqrt(n_map.get(c, 1.0)) for c in cohorts])[valid]
            else:
                w = np.ones(valid.sum())
            combo = stouffer_combine(z_i[valid], weights=w)
            row = {
                "rank": rank, "statistic": stat,
                "k_cohorts": int(valid.sum()),
                "cohorts": ",".join(np.array(cohorts)[valid]),
                "z_per_cohort": ",".join(f"{z:.2f}" for z in z_i[valid]),
                **combo,
            }
            rows.append(row)
    result_df = pd.DataFrame(rows)
    for col in ["Z_combined", "p_combined", "Q", "Q_p", "I2_pct"]:
        if col in result_df.columns:
            result_df[col] = result_df[col].round(4)
    return result_df, skipped_stats


def run_meta(results_dir, out_dir, weight_by_n):
    save_permtest_summary(results_dir, out_dir)
    result_df, skipped_stats = run_meta_analysis(results_dir, weight_by_n=weight_by_n)
    if result_df is None:
        return
    out_path = os.path.join(out_dir, "meta_analysis_summary.tsv")
    result_df.to_csv(out_path, sep="\t", index=False)
    if skipped_stats:
        print(f"\n[meta] WARNING: skipped statistic(s) {sorted(skipped_stats)} -- "
              f"required null_*_std column(s) not found (older all_permtest.tsv). "
              f"Rerun 'dual_run.py --run_perm' to regenerate and include them.")
    print(f"\n[meta] Stouffer meta-analysis "
          f"({'sqrt(n)-weighted' if weight_by_n else 'equal-weighted'})")
    print(f"{'='*100}")
    print(f"{'rank':>4} {'statistic':>9} {'k':>2}  {'Z_combined':>10} {'p_combined':>10}  "
          f"{'Q':>6} {'Q_p':>6} {'I2%':>6}   z per cohort")
    print("-"*100)
    for _, r in result_df.iterrows():
        sig = "*" if r["p_combined"] < 0.05 else " "
        print(f"{r['rank']:>4} {r['statistic']:>9} {r['k_cohorts']:>2}  "
              f"{r['Z_combined']:>10.3f} {r['p_combined']:>9.4f}{sig}  "
              f"{r['Q']:>6.2f} {r['Q_p']:>6.3f} {r['I2_pct']:>5.1f}%   "
              f"[{r['cohorts']}] = [{r['z_per_cohort']}]")
    print(f"\n[meta] Saved: {out_path}")
    print("[meta] Note: I2% is the share of cross-cohort variation NOT explained by "
          "chance alone. A combined p<0.05 with high I2% still means 'the pooled "
          "direction is real' but 'the effect size differs a lot by cohort'.")
    if weight_by_n:
        print("[meta] Note: with --weight_by_n, Cochran's Q (and hence I2%) inflates "
              "mechanically when weights are very unequal -- compare both weightings "
              "rather than trusting one alone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-hoc re-analysis of DualModel results: re-rank by "
                     "score_causal (from taxa_{tag}.tsv) and run the "
                     "cross-cohort Stouffer meta-analysis (from "
                     "all_permtest.tsv). Neither step re-runs SA/bootstrap/KO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results_dir", default="dualmodel_results",
                        help="Directory containing dual_run.py's output "
                             "(taxa_{tag}.tsv, all_permtest.tsv).")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write outputs (default: same as --results_dir).")
    parser.add_argument("--top_n", type=int, default=10,
                        help="How many top taxa to print per tag / cross-cohort table.")
    parser.add_argument("--min_boot_prob", type=float, default=0.5,
                        help="Only print taxa with boot_prob >= this in the "
                             "per-tag rescore tables.")
    parser.add_argument("--min_stability_cross", type=float, default=0.5,
                        help="Stability threshold for the cross-cohort stable-taxa summary.")
    parser.add_argument("--weight_by_n", action="store_true", default=False,
                        help="[meta-analysis] Weight each cohort's z-score by "
                             "sqrt(sample size) instead of equal weighting.")
    parser.add_argument("--skip_rescore", action="store_true", default=False,
                        help="Skip step (A), only run the meta-analysis.")
    parser.add_argument("--skip_meta", action="store_true", default=False,
                        help="Skip step (B), only run the re-ranking.")
    args = parser.parse_args()

    out_dir = args.out_dir or args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    if not args.skip_rescore:
        run_rescore(args.results_dir, out_dir, args.top_n,
                    args.min_boot_prob, args.min_stability_cross)

    if not args.skip_meta:
        print(f"\n{'#'*100}")
        run_meta(args.results_dir, out_dir, args.weight_by_n)
