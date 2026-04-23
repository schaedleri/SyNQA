# -*- coding: utf-8 -*-
"""
Master_Pipeline_Integrated_Full_Strict_MultiBeta.py
--------------------------------------------------------
Purpose:
    Execute the ENTIRE SyNQA Pipeline sequentially with STRICT SCIENTIFIC INTEGRITY.
    
    [Update]
      - Beta range is now identical to Gamma range.
      - Phase 1 computes the full Beta x Gamma grid once.
      - Phase 1.5 to 4 are executed in a loop for each Beta, saving outputs into 
        Beta-specific subdirectories.
      - Baseline models now dynamically use the same number of features as the Rewiring Guild.

"""

import pandas as pd
import numpy as np
import neal
import itertools
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed
import sys
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Scipy & Sklearn
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, 
                             confusion_matrix, recall_score,
                             f1_score, matthews_corrcoef, precision_score)

try:
    from xgboost import XGBClassifier
except ImportError:
    warnings.warn("[Warning] XGBoost not installed. XGB selection will fail if used.")

warnings.filterwarnings("ignore")

# =====================================================================
# GLOBAL CONFIGURATION
# =====================================================================
WORK_DIR = Path(".")
DIR_RESULTS = WORK_DIR / "Final_Results_SyNQA_Strict_demo"
DIR_RESULTS.mkdir(exist_ok=True)

FILE_X_RAW    = "X_all_raw.tsv"                  
FILE_COV      = "covariates.tsv"
FILE_Y        = "y_all.tsv"
FILE_STUDY    = "study_id.tsv"

# Phase 1 output (Contains all betas and gammas)
OUT_MECH_DETAIL  = DIR_RESULTS / "mechanism_detailed.csv"

# --- PARAMETER SETTINGS ---
CHAMPION_COST = 0.15
ALPHA_RANGE = [1.0]

# BETA and GAMMA are now identical ranges
BETA_RANGE  =  [1.0, 5.0] #[0.0, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0] 
GAMMA_RANGE =  [1.0, 5.0] #[0.0, 0.5, 1.0, 3.0, 5.0, 7.0, 10.0] 
COST_RANGE  = [CHAMPION_COST]
SWEEPS_RANGE = [1] #[10000]

TOP_N_SELECTION = 60
N_ENSEMBLE_TRIALS = 10 #100
FREQ_THRESHOLD = 0.5
N_SA_READS = 50 #300
SEED_BASE = 420
N_JOBS = os.cpu_count() - 1 if os.cpu_count() and os.cpu_count() > 1 else 1

N_PERMUTATIONS = 1000
SIGNIFICANCE_LEVEL = 0.05
Z_SCORE_THRESHOLD = 1.96
FINAL_K = 25  
SEED = 2754
MIN_K = 5
MAX_K = 50
STEP_SIZE = 2 
TOP_N_SCREENING = 60

np.random.seed(SEED)

# =====================================================================
# COMMON HELPERS & DATA LOADING
# =====================================================================
def clean_col_names(df):
    df.columns = [c.split("|")[-1].replace("[", "").replace("]", "").replace("<", "").replace("'", "").strip() for c in df.columns]
    return df

def load_data_strict():
    required = [FILE_X_RAW, FILE_COV, FILE_Y, FILE_STUDY]
    if not all(os.path.exists(f) for f in required):
        sys.exit(f"[Error] Missing files. Ensure {required} are in {WORK_DIR}.")

    X_raw   = pd.read_csv(FILE_X_RAW, sep="\t", index_col=0)
    cov     = pd.read_csv(FILE_COV, sep="\t", index_col=0)
    y_raw   = pd.read_csv(FILE_Y, sep="\t", index_col=0).iloc[:, 0]
    study   = pd.read_csv(FILE_STUDY, sep="\t", index_col=0).iloc[:, 0]
    
    X_raw = clean_col_names(X_raw)

    common_idx = X_raw.index.intersection(cov.index).intersection(y_raw.index).intersection(study.index)
    X_raw = X_raw.loc[common_idx]
    cov   = cov.loc[common_idx]
    y_raw = y_raw.loc[common_idx]
    study = study.loc[common_idx]
    
    if set(y_raw.unique()) <= {0, 1}: 
        y = y_raw.astype(int)
    else: 
        y = y_raw.map(lambda x: 1 if any(k in str(x).upper() for k in ["CRC", "CASE", "CANCER", "1"]) else 0)
    
    cov_encoded = pd.get_dummies(cov, drop_first=True)
    cov_encoded = cov_encoded.fillna(cov_encoded.median())
    
    return X_raw, cov_encoded, y, study

def get_fisher_z_diff_matrix(X, y):
    n_features = X.shape[1]
    X_case = X.loc[y == 1]
    X_ctrl = X.loc[y == 0]
    n_case, n_ctrl = len(X_case), len(X_ctrl)
    
    if n_case <= 3 or n_ctrl <= 3: 
        return np.zeros((n_features, n_features))

    r_case = np.clip(X_case.corr().values, -0.99999, 0.99999)
    r_ctrl = np.clip(X_ctrl.corr().values, -0.99999, 0.99999)
    z_case, z_ctrl = np.arctanh(r_case), np.arctanh(r_ctrl)
    
    se = np.sqrt((1 / (n_case - 3)) + (1 / (n_ctrl - 3)))
    z_stat = (z_case - z_ctrl) / se
    np.fill_diagonal(z_stat, 0)
    return np.nan_to_num(z_stat)

def compute_3_metrics_fisher(X, y):
    z_diff = get_fisher_z_diff_matrix(X, y)
    pos_score = np.sum(np.maximum(z_diff, 0), axis=0)
    neg_score = np.sum(np.abs(np.minimum(z_diff, 0)), axis=0)
    abs_score = np.sum(np.abs(z_diff), axis=0)
    return pos_score, neg_score, abs_score

# =====================================================================
# DYNAMIC RESIDUALIZATION (CORE)
# =====================================================================
def process_train_data_strict(X_train_raw, cov_train):
    pseudocount = 1e-6
    X_log = np.log(X_train_raw + pseudocount)
    gm = X_log.mean(axis=1)
    X_clr = X_log.sub(gm, axis=0)
    
    scaler_cov = StandardScaler()
    C_train = scaler_cov.fit_transform(cov_train)
    
    reg = LinearRegression()
    reg.fit(C_train, X_clr)
    
    X_pred = reg.predict(C_train)
    X_resid = X_clr - X_pred
    
    scaler_minmax = MinMaxScaler()
    X_resid_scaled = pd.DataFrame(scaler_minmax.fit_transform(X_resid), 
                                  index=X_train_raw.index, 
                                  columns=X_train_raw.columns)
    return X_resid_scaled

def dynamic_residualization_split(X_train_raw, cov_train, X_test_raw, cov_test):
    pseudocount = 1e-6
    X_train_log = np.log(X_train_raw + pseudocount)
    X_test_log  = np.log(X_test_raw + pseudocount)
    
    X_train_clr = X_train_log.sub(X_train_log.mean(axis=1), axis=0)
    X_test_clr  = X_test_log.sub(X_test_log.mean(axis=1), axis=0)
    
    scaler_cov = StandardScaler()
    C_train = scaler_cov.fit_transform(cov_train)
    C_test  = scaler_cov.transform(cov_test) 
    
    reg = LinearRegression()
    reg.fit(C_train, X_train_clr)
    
    X_train_resid = X_train_clr - reg.predict(C_train)
    X_test_resid  = X_test_clr - reg.predict(C_test)
    
    scaler_mm = MinMaxScaler()
    X_train_final = pd.DataFrame(scaler_mm.fit_transform(X_train_resid), 
                                 index=X_train_raw.index, columns=X_train_raw.columns)
    X_test_final  = pd.DataFrame(scaler_mm.transform(X_test_resid),
                                 index=X_test_raw.index, columns=X_test_raw.columns)
    return X_train_final, X_test_final

# =====================================================================
# ------------------------- PART 1: PHASE 1 -------------------------
# =====================================================================
def prepare_fold_matrices(X_train_resid, X_train_raw, y_train):
    mean_abund = X_train_raw.mean(axis=0)
    feats_abund = X_train_raw.columns[np.argsort(mean_abund.values)[-TOP_N_SELECTION:]]
    
    mi = mutual_info_classif(X_train_resid, y_train, random_state=SEED_BASE)
    feats_mi = X_train_resid.columns[np.argsort(mi)[-TOP_N_SELECTION:]]
    
    union_feats = sorted(list(set(feats_abund) | set(feats_mi)))
    X_sub = X_train_resid[union_feats].copy()
    
    mi_sub = mutual_info_classif(X_sub, y_train, random_state=SEED_BASE)
    s = pd.Series(mi_sub, index=X_sub.columns)
    if s.max() > 0: s /= s.max()
    
    C = np.abs(X_sub.corr().values); np.fill_diagonal(C, 0)
    
    z_stat = get_fisher_z_diff_matrix(X_sub, y_train)
    S_raw = np.abs(z_stat)
    max_delta = np.max(S_raw)
    S = S_raw / max_delta if max_delta > 0 else S_raw
    
    return union_feats, s.fillna(0).values, np.nan_to_num(C), np.nan_to_num(S)

def run_qubo_task(matrices, params, feature_names, target_study):
    s_vec, C_mat, S_mat = matrices
    alpha, beta, gamma, cost, sweeps = params
    
    n = len(feature_names)
    Q = {}; sparsity = cost * alpha
    for i in range(n):
        Q[(i, i)] = -alpha * s_vec[i] + sparsity
        for j in range(i+1, n): 
            Q[(i, j)] = (beta * C_mat[i, j]) - (gamma * S_mat[i, j])
            
    counts = np.zeros(n)
    for k in range(N_ENSEMBLE_TRIALS):
        sampler = neal.SimulatedAnnealingSampler()
        ss = sampler.sample_qubo(Q, num_reads=N_SA_READS, num_sweeps=sweeps, seed=SEED_BASE + k*999)
        sample = ss.first.sample
        curr = [k for k, v in sample.items() if v == 1]
        if curr: counts[curr] += 1
        
    probs = counts / N_ENSEMBLE_TRIALS
    df_probs = pd.Series(probs, index=feature_names)
    consensus = df_probs[df_probs >= FREQ_THRESHOLD].index.tolist()
    if len(consensus) < 2: consensus = df_probs.sort_values(ascending=False).head(3).index.tolist()
        
    return {
        "Alpha": alpha, "Beta": beta, "Gamma": gamma, "CostFactor": cost, 
        "Sweeps": sweeps, "Test_Cohort": target_study, 
        "K": len(consensus), "Features": ",".join(consensus)
    }

def run_phase1_mechanism():
    print("\n[Phase 1] Running Mechanism Exploration (Grid Search Beta x Gamma)...")
    print("[Load] Loading Raw Data & Covariates (Strict Processing)...")
    
    X_raw, cov, y, study = load_data_strict()
    studies = sorted(study.unique())
    
    param_grid = list(itertools.product(ALPHA_RANGE, BETA_RANGE, GAMMA_RANGE, COST_RANGE, SWEEPS_RANGE))
    tasks = []
    
    print(f" > Processing {len(studies)} folds (Strict LOGO Residualization)...")
    
    for i, target_study in enumerate(studies):
        print(f"   [{i+1}/{len(studies)}] Fold: {target_study} ...")
        
        test_mask = (study == target_study)
        X_train_raw = X_raw.loc[~test_mask]
        cov_train   = cov.loc[~test_mask]
        y_train     = y.loc[~test_mask]
        
        X_train_resid = process_train_data_strict(X_train_raw, cov_train)
        
        feats, s_vec, C_mat, S_mat = prepare_fold_matrices(X_train_resid, X_train_raw, y_train)
        fold_matrices = (s_vec, C_mat, S_mat)
        
        for params in param_grid:
            tasks.append((fold_matrices, params, feats, target_study))
            
    print(f" > Dispatching {len(tasks)} tasks to {N_JOBS} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=N_JOBS) as executor:
        futures = {executor.submit(run_qubo_task, t[0], t[1], t[2], t[3]): i for i, t in enumerate(tasks)}
        for i, fut in enumerate(as_completed(futures)):
            try:
                res = fut.result()
                results.append(res)
                if (i+1) % 50 == 0: print(f"   - Progress: {i+1}/{len(tasks)}", end='\r')
            except Exception as e: print(f"Err: {e}")
                
    print("\n")
    df_detail = pd.DataFrame(results)
    df_detail = df_detail.sort_values(by=["Test_Cohort", "Beta", "Gamma"])
    df_detail.to_csv(OUT_MECH_DETAIL, index=False)
    print(f" > Saved full grid detailed results to {OUT_MECH_DETAIL}")


# =====================================================================
# --------------------- PART 2: PHASE 1.5 ~ PHASE 4 -------------------
# =====================================================================

# --- Phase 1.5 ---
def run_role_synthesis(target_beta, out_dir):
    print(f"\n[Phase 1.5] Synthesizing Guild Roles for Beta={target_beta}...")
    
    out_guilds_file = out_dir / "guild_roles.csv"
    
    if not os.path.exists(OUT_MECH_DETAIL): 
        print(f"[Warning] {OUT_MECH_DETAIL} not found. Returning empty roles.")
        return {"Rewiring": []}

    df_mech = pd.read_csv(OUT_MECH_DETAIL)
    # Filter by target Beta
    df_tgt = df_mech[(np.isclose(df_mech["Beta"], target_beta)) & (np.isclose(df_mech["CostFactor"], CHAMPION_COST))]
    
    all_feats = set()
    for fstr in df_tgt["Features"]:
        if pd.notna(fstr) and fstr != "": all_feats.update([x.strip() for x in fstr.split(",")])
    
    champion_feats = sorted(list(all_feats))
    
    X_raw, cov, y, _ = load_data_strict()
    X_resid, _ = dynamic_residualization_split(X_raw, cov, X_raw, cov) 
    
    valid_feats = [f for f in champion_feats if f in X_resid.columns]
    X_guild = X_resid[valid_feats]
    
    # Check if there are valid features for this Beta
    if X_guild.shape[1] < 2:
        print(f" > [Warning] Not enough features selected for Beta={target_beta}. Skipping permutations.")
        return {"Rewiring": []}

    real_pos, real_neg, real_abs = compute_3_metrics_fisher(X_guild, y)
    
    def perm_worker(seed):
        np.random.seed(seed)
        y_shuf = y.sample(frac=1, random_state=seed).values
        return compute_3_metrics_fisher(X_guild, y_shuf)

    null_results = Parallel(n_jobs=N_JOBS)(delayed(perm_worker)(s) for s in range(N_PERMUTATIONS))
    null_pos = np.array([r[0] for r in null_results])
    null_neg = np.array([r[1] for r in null_results])
    null_abs = np.array([r[2] for r in null_results])
    
    role_map = {"Rewiring": set()}
    metrics = [("Absolute", real_abs, null_abs, "Rewiring")]
    
    for label, real, nulls, role_name in metrics:
        m_null, s_null = np.mean(nulls, axis=0), np.std(nulls, axis=0)
        s_null[s_null==0] = 1e-9
        z_scores = (real - m_null) / s_null
        p_vals = (np.sum(nulls >= real, axis=0) + 1) / (N_PERMUTATIONS + 1)
        for i, taxon in enumerate(valid_feats):
            if p_vals[i] < SIGNIFICANCE_LEVEL and z_scores[i] > Z_SCORE_THRESHOLD:
                role_map[role_name].add(taxon)

    guild_groups = {k: sorted(list(v)) for k, v in role_map.items()}
    rows = [{"Taxon": t, "Roles": "|".join([r for r in guild_groups if t in guild_groups[r]])} 
            for t in set().union(*role_map.values())]
    pd.DataFrame(rows).to_csv(out_guilds_file, index=False)
    
    return guild_groups

# --- Phase 2 ---
def evaluate_model_LOGO_strict(X_raw, cov, y, study, features, method_label, n_features=None, selector=None):
    logo = LeaveOneGroupOut()
    fold_metrics = []
    
    for train_idx, test_idx in logo.split(X_raw, y, study):
        X_tr_raw, X_te_raw = X_raw.iloc[train_idx], X_raw.iloc[test_idx]
        cov_tr, cov_te     = cov.iloc[train_idx], cov.iloc[test_idx]
        y_tr, y_te         = y.iloc[train_idx], y.iloc[test_idx]
        
        X_tr_res, X_te_res = dynamic_residualization_split(X_tr_raw, cov_tr, X_te_raw, cov_te)
        
        X_tr_vals, X_te_vals = X_tr_res.values, X_te_res.values
        y_tr_vals, y_te_vals = y_tr.values, y_te.values
        
        if selector == "DA":
            p_vals = [mannwhitneyu(X_tr_vals[y_tr_vals==1, j], 
                                   X_tr_vals[y_tr_vals==0, j])[1] for j in range(X_tr_vals.shape[1])]
            selected_idx = np.argsort(p_vals)[:n_features]

        elif selector == "RF":
            sel = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1).fit(X_tr_vals, y_tr_vals)
            selected_idx = np.argsort(sel.feature_importances_)[::-1][:n_features]

        elif selector == "XGB":
            sel = XGBClassifier(n_estimators=100, random_state=SEED, n_jobs=-1, 
                                use_label_encoder=False, eval_metric='logloss')
            sel.fit(X_tr_vals, y_tr_vals)
            selected_idx = np.argsort(sel.feature_importances_)[::-1][:n_features]

        elif selector == "LASSO":
            sel = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=SEED, class_weight='balanced')
            sel.fit(X_tr_vals, y_tr_vals)
            selected_idx = np.argsort(np.abs(sel.coef_[0]))[::-1][:n_features]

        elif selector == "Abundance":
            selected_idx = np.argsort(X_tr_raw.mean(axis=0).values)[::-1][:n_features]

        elif selector == "MI":
            selected_idx = np.argsort(mutual_info_classif(X_tr_vals, y_tr_vals, random_state=SEED))[::-1][:n_features]

        elif features is not None:
            selected_idx = [X_raw.columns.get_loc(f) for f in features if f in X_raw.columns]
        else:
            selected_idx = range(X_tr_vals.shape[1])

        if len(selected_idx) == 0: continue
        
        X_tr_final = X_tr_vals[:, selected_idx]
        X_te_final = X_te_vals[:, selected_idx]
        
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=SEED))
        if method_label == "RF_All":
            clf = RandomForestClassifier(n_estimators=300, class_weight='balanced', random_state=SEED, n_jobs=-1)
            
        clf.fit(X_tr_final, y_tr_vals)
        try:
            probs = clf.predict_proba(X_te_final)[:, 1]
            if len(np.unique(y_te_vals)) < 2: continue
            
            preds = (probs >= 0.5).astype(int)
            
            auc_val = roc_auc_score(y_te_vals, probs)
            acc_val = balanced_accuracy_score(y_te_vals, preds)
            tn, fp, fn, tp = confusion_matrix(y_te_vals, preds, labels=[0, 1]).ravel()
            
            sens = recall_score(y_te_vals, preds, zero_division=0)
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = precision_score(y_te_vals, preds, zero_division=0)
            npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            f1   = f1_score(y_te_vals, preds, zero_division=0)
            mcc  = matthews_corrcoef(y_te_vals, preds)

            fold_metrics.append({
                "ROC_AUC": auc_val, 
                "Balanced_Acc": acc_val,
                "Sensitivity": sens, 
                "Specificity": spec,
                "Precision": prec,
                "NPV": npv,
                "F1_Score": f1,
                "MCC": mcc
            })
        except: pass

    df_f = pd.DataFrame(fold_metrics)
    stats = df_f.mean().to_dict()
    stats["ROC_AUC_Std"] = df_f["ROC_AUC"].std()
    return stats, df_f

def run_validation_pipeline_strict(guild_groups, out_dir):
    print(" > [Phase 2] LOGO Validation (Strict)...")
    X_raw, cov, y, study = load_data_strict()
    results = []
    
    # ---------------------------------------------------------
    # DYNAMIC BASELINE FEATURE COUNT
    # Evaluate SyNQA first to determine the number of features
    # ---------------------------------------------------------
    synqa_methods = []
    dynamic_k = FINAL_K # Fallback
    
    for name, feats in guild_groups.items():
        if feats:
            # Using the full length of the synthesized role features
            current_k = len(feats) 
            m, _ = evaluate_model_LOGO_strict(X_raw, cov, y, study, feats, f"Role: {name}")
            synqa_methods.append({**m, "Method": f"Role: {name}", "Group": "SyNQA"})
            dynamic_k = current_k # Update baseline K to match SyNQA length
            print(f"   * SyNQA Role: {name} found {dynamic_k} features. Applying to baselines.")

    # ---------------------------------------------------------
    # EVALUATE BASELINES WITH DYNAMIC K
    # ---------------------------------------------------------
    baseline_list = [
        ("DA Baseline", "DA"), 
        ("High Abundance", "Abundance"), 
        ("High MI", "MI"), 
        ("RF Selection", "RF"),
        ("XGB Selection", "XGB"),
        ("LASSO Selection", "LASSO") 
    ]
    
    for s_name, sel in baseline_list:
        m, _ = evaluate_model_LOGO_strict(X_raw, cov, y, study, None, s_name, n_features=dynamic_k, selector=sel)
        results.append({**m, "Method": s_name, "Group": "Baseline"})
    
    m_rf, _ = evaluate_model_LOGO_strict(X_raw, cov, y, study, None, "RF_All")
    results.append({**m_rf, "Method": "RF All Features", "Group": "Baseline"})

    # Append SyNQA results evaluated earlier
    results.extend(synqa_methods)

    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "validation_logo_metrics.csv", index=False)

    methods_to_plot = ["DA Baseline", "LASSO Selection", "XGB Selection", "Role: Rewiring", "RF All Features"]
    df_plot = df_res[df_res["Method"].isin(methods_to_plot)].reset_index(drop=True)
    df_plot["SortIdx"] = df_plot["Method"].apply(lambda x: methods_to_plot.index(x) if x in methods_to_plot else 99)
    df_plot = df_plot.sort_values("SortIdx").reset_index(drop=True)

    plt.figure(figsize=(13, 7))
    custom_palette = {'DA Baseline': '#CCCCCC', 'LASSO Selection': '#9467bd', 'XGB Selection': '#17becf', 'Role: Rewiring': '#D62728', 'RF All Features': '#444444'}
    bar_colors = [custom_palette.get(m, 'skyblue') for m in df_plot["Method"]]
    
    bars = plt.bar(df_plot["Method"], df_plot["ROC_AUC"], yerr=df_plot["ROC_AUC_Std"], capsize=10, color=bar_colors, alpha=0.9, edgecolor='black', linewidth=1.5)
    for i, rect in enumerate(bars):
        height = rect.get_height()
        std_val = df_plot.loc[i, "ROC_AUC_Std"] if pd.notna(df_plot.loc[i, "ROC_AUC_Std"]) else 0
        plt.text(rect.get_x() + rect.get_width()/2.0, height + std_val + 0.015, f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=12)

    plt.title(f"SyNQA vs SOTA (Strict LOGO): Generalizability (Top {dynamic_k} features)", fontsize=16, fontweight='bold')
    plt.ylabel("Mean AUC (+- LOGO Std Dev)", fontsize=13)
    plt.xlabel("Method", fontsize=13)
    plt.ylim(0.0, 1.05)
    plt.xticks(fontsize=11, rotation=15)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "Fig_LOGO_AUC_Barplot.png", dpi=300)
    plt.close()

    heatmap_cols = ["ROC_AUC", "Balanced_Acc", "MCC", "F1_Score", "Sensitivity", "Specificity", "Precision", "NPV"]
    available_cols = [c for c in heatmap_cols if c in df_res.columns]
    
    if len(df_plot) > 0:
        df_heat = df_plot.set_index("Method")[available_cols]
        plt.figure(figsize=(12, 6))
        sns.heatmap(df_heat, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0.0, vmax=1.0, linewidths=.5)
        plt.title(f"Comprehensive Performance Metrics (Strict LOGO, K={dynamic_k})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(out_dir / "Fig_LOGO_AllMetrics_Heatmap.png", dpi=300)
        plt.close()

# --- Phase 3 ---
def calculate_set_metrics_LOGO_proof(X_raw, cov, y, study_id, features):
    valid_feats = [f for f in features if f in X_raw.columns]
    if len(valid_feats) < 2: return 0.5, 0.0, 0.0
    
    logo = LeaveOneGroupOut()
    aucs, reds, strs = [], [], []
    
    for train_idx, test_idx in logo.split(X_raw, y, study_id):
        X_tr, X_te = dynamic_residualization_split(X_raw.iloc[train_idx], cov.iloc[train_idx], 
                                                   X_raw.iloc[test_idx], cov.iloc[test_idx])
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        
        clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=SEED))
        clf.fit(X_tr[valid_feats], y_tr)
        if len(np.unique(y_te)) > 1:
            aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te[valid_feats])[:, 1]))
        
        X_sub = X_tr[valid_feats]
        reds.append(np.nanmean(np.abs(np.corrcoef(X_sub.T)[np.triu_indices(len(valid_feats), k=1)])))
        
        X_ca, X_co = X_sub[y_tr==1], X_sub[y_tr==0]
        if len(X_ca) > 2 and len(X_co) > 2:
            strs.append(np.nanmean(np.abs(np.corrcoef(X_ca.T) - np.corrcoef(X_co.T))[np.triu_indices(len(valid_feats), k=1)]))
            
    return (np.mean(aucs) if aucs else 0.5), (np.mean(reds) if reds else 0.0), (np.mean(strs) if strs else 0.0)

def run_structural_proof_pipeline(guild_groups, out_dir, proof_dir):
    print(" > [Phase 3] Structural Proof (Strict)...")
    X_raw, cov, y, study = load_data_strict()
    
    X_res_global, _ = dynamic_residualization_split(X_raw, cov, X_raw, cov)
    mi = mutual_info_classif(X_res_global, y, random_state=SEED)
    top_pool = list(set(X_raw.columns[np.argsort(X_raw.mean(axis=0))[-TOP_N_SCREENING:]]) | 
                    set(X_raw.columns[np.argsort(mi)[-TOP_N_SCREENING:]]))
    
    naive_ranks = compute_3_metrics_fisher(X_res_global[top_pool], y)
    taxa_pool = X_raw[top_pool].columns
    naive_map = {
        "Rewiring": [x for _, x in sorted(zip(naive_ranks[2], taxa_pool), reverse=True)]
    }

    k_vals = range(MIN_K, MAX_K + 1, STEP_SIZE)
    for g_name, n_key in [("Rewiring", "Rewiring")]:
        p_feats, n_feats = guild_groups.get(g_name, []), naive_map.get(n_key, [])
        if not p_feats: continue

        res = Parallel(n_jobs=N_JOBS)(delayed(lambda k: {
            "k": k, "SyNQA": calculate_set_metrics_LOGO_proof(X_raw, cov, y, study, p_feats[:k]),
            "Naive": calculate_set_metrics_LOGO_proof(X_raw, cov, y, study, n_feats[:k])
        })(k) for k in k_vals)

        df = pd.DataFrame([{"k": r["k"], "AUC_SyNQA": r["SyNQA"][0], "AUC_Naive": r["Naive"][0],
                            "Str_SyNQA": r["SyNQA"][2], "Str_Naive": r["Naive"][2],
                            "Red_SyNQA": r["SyNQA"][1], "Red_Naive": r["Naive"][1]} for r in res])
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(df["k"], df["AUC_SyNQA"], 'o-', color='red', label='SyNQA (QUBO)')
        ax1.plot(df["k"], df["AUC_Naive"], 's--', color='gray', label='Naive Ranking')
        ax1.set_title(f"Efficiency: {g_name}")
        ax1.legend()
        
        ax2.plot(df["k"], df["Str_SyNQA"], 'o-', color='red', label='Structure (SyNQA)')
        ax2.plot(df["k"], df["Str_Naive"], 's-', color='blue', label='Structure (Naive)')
        ax2.plot(df["k"], df["Red_SyNQA"], 'o:', color='orange', label='Redundancy (SyNQA)')
        ax2.plot(df["k"], df["Red_Naive"], 's:', color='cyan', label='Redundancy (Naive)')
        ax2.set_title("Structure vs Redundancy")
        ax2.legend()
        
        plt.savefig(proof_dir / f"Proof_Fisher_LOGO_{g_name}.png", dpi=300)
        plt.close()
        df.to_csv(proof_dir / f"Proof_Data_{g_name}.csv", index=False)

# --- Phase 4 (STRICT DETERMINISTIC VERSION) ---
def plot_panel_A_slope(X, y, pair, title, filename):
    u, v = pair
    df = X[[u, v]].copy()
    df['Status'] = ['Case' if val == 1 else 'Control' for val in y]
    
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df, x=u, y=v, hue='Status', style='Status', palette={'Case': 'red', 'Control': 'blue'}, alpha=0.6)
    sns.regplot(data=df[df['Status']=='Case'], x=u, y=v, scatter=False, color='red', label='Case Trend', ci=None)
    sns.regplot(data=df[df['Status']=='Control'], x=u, y=v, scatter=False, color='blue', label='Control Trend', ci=None)
    
    plt.title(f"Panel A: Differential Correlation\n{title}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_panel_B_energy(X, y, feats, title, filename):
    corrs = [spearmanr(X[f], y)[0] for f in feats]
    df_e = pd.DataFrame({'Feature': feats, 'Correlation': corrs})
    df_e = df_e.sort_values('Correlation')
    
    plt.figure(figsize=(8, 6))
    colors = ['red' if x > 0 else 'blue' for x in df_e['Correlation']]
    plt.barh(df_e['Feature'], df_e['Correlation'], color=colors)
    plt.title(f"Panel B: Feature Association Strength\n{title}")
    plt.xlabel("Spearman Correlation with Disease")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_panel_C_composite(X, y, target_pair, control_pair, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    u, v = target_pair
    df_t = X[[u, v]].copy()
    df_t['Status'] = y
    r_case = pearsonr(df_t.loc[df_t['Status']==1, u], df_t.loc[df_t['Status']==1, v])[0]
    r_ctrl = pearsonr(df_t.loc[df_t['Status']==0, u], df_t.loc[df_t['Status']==0, v])[0]
    
    sns.regplot(data=df_t[df_t['Status']==1], x=u, y=v, ax=axes[0], color='red', scatter_kws={'alpha':0.3}, label=f'Case (r={r_case:.2f})')
    sns.regplot(data=df_t[df_t['Status']==0], x=u, y=v, ax=axes[0], color='blue', scatter_kws={'alpha':0.3}, label=f'Ctrl (r={r_ctrl:.2f})')
    axes[0].set_title(f"SyNQA Target Pair\n{u} vs {v}")
    axes[0].legend()

    u, v = control_pair
    df_c = X[[u, v]].copy()
    df_c['Status'] = y
    r_case = pearsonr(df_c.loc[df_c['Status']==1, u], df_c.loc[df_c['Status']==1, v])[0]
    r_ctrl = pearsonr(df_c.loc[df_c['Status']==0, u], df_c.loc[df_c['Status']==0, v])[0]

    sns.regplot(data=df_c[df_c['Status']==1], x=u, y=v, ax=axes[1], color='red', scatter_kws={'alpha':0.3}, label=f'Case (r={r_case:.2f})')
    sns.regplot(data=df_c[df_c['Status']==0], x=u, y=v, ax=axes[1], color='blue', scatter_kws={'alpha':0.3}, label=f'Ctrl (r={r_ctrl:.2f})')
    axes[1].set_title(f"Baseline Control Pair\n{u} vs {v}")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def run_figure4_smoking_gun_pipeline(guild_groups, out_dir):
    print(" > [Phase 4] Generating Figure 4 (Smoking Gun Analysis - Strict Selection)...")
    X_raw, cov, y, _ = load_data_strict()
    
    X, _ = dynamic_residualization_split(X_raw, cov, X_raw, cov) 
    
    X.columns = [c.split("|")[-1].replace("[", "").replace("]", "").replace("<", "").replace("'", "").strip() for c in X.columns]
    y_bin = y 

    rewiring_feats = guild_groups.get("Rewiring", [])
    valid_rewiring = [f for f in rewiring_feats if f in X.columns]
    
    target_pair = None
    if len(valid_rewiring) >= 2:
        X_sub = X[valid_rewiring]
        z_diff = get_fisher_z_diff_matrix(X_sub, y_bin)
        abs_z_diff = np.abs(z_diff)
        
        max_idx = np.unravel_index(np.argmax(abs_z_diff, axis=None), abs_z_diff.shape)
        target_pair = (valid_rewiring[max_idx[0]], valid_rewiring[max_idx[1]])
    else:
        print("   [Warning] Insufficient Rewiring features. Falling back to default.")
        target_pair = (X.columns[0], X.columns[1])

    mean_abund = X_raw.mean(axis=0).sort_values(ascending=False)
    non_rewiring_pool = [f for f in mean_abund.index if f not in valid_rewiring and f in X.columns]
    
    control_pair = None
    if len(non_rewiring_pool) >= 2:
        control_pair = (non_rewiring_pool[0], non_rewiring_pool[1])
    else:
        control_pair = (X.columns[2], X.columns[3])

    plot_panel_A_slope(X, y_bin, target_pair, f"Target: {target_pair}", out_dir / "Figure4A_Slope.png")
    plot_panel_B_energy(X, y_bin, list(target_pair) + list(control_pair), "Selected Feature Correlations", out_dir / "Figure4B_Energy.png")
    plot_panel_C_composite(X, y_bin, target_pair, control_pair, out_dir / "Figure4C_Composite.png")

    with open(out_dir / "Figure4_Statistics_Report.txt", "w") as f:
        f.write("Selection Methodology: Deterministic based on Phase 1.5 Guilds\n")
        f.write(f"Target Pair (Max Z-diff in Rewiring group): {target_pair}\n")
        f.write(f"Control Pair (Top Abundance, Non-Rewiring): {control_pair}\n")

# =====================================================================
# MAIN EXECUTION
# =====================================================================
def main():
    print("Starting Master Pipeline (Full Sequence across Multiple Betas)...")
    
    # 1. Phase 1: Exploration (Runs Full Beta x Gamma Grid once)
    run_phase1_mechanism()
    
    # Run Phase 1.5 to 4 individually for each Beta
    for beta in BETA_RANGE:
        print(f"\n" + "="*50)
        print(f" PROCESSING DOWNSTREAM PIPELINE FOR BETA = {beta}")
        print("="*50)
        
        # Create Beta-specific directory
        beta_dir = DIR_RESULTS / f"Beta_{beta}"
        beta_dir.mkdir(parents=True, exist_ok=True)
        
        proof_dir = beta_dir / "Structural_Proof_LOGO"
        proof_dir.mkdir(parents=True, exist_ok=True)

        # 2. Phase 1.5: Role Synthesis
        guilds = run_role_synthesis(target_beta=beta, out_dir=beta_dir)
        
        if not any(guilds.values()):
            print(f" > [Skip] No roles synthesized for Beta={beta}. Moving to next.")
            continue
            
        # 3. Phase 2: Validation (Strict LOGO)
        run_validation_pipeline_strict(guilds, out_dir=beta_dir)
        
        # 4. Phase 3: Structural Advantage Proof
        run_structural_proof_pipeline(guilds, out_dir=beta_dir, proof_dir=proof_dir)
        
        # 5. Phase 4: Smoking Gun Visualization (Strict Deterministic)
        run_figure4_smoking_gun_pipeline(guilds, out_dir=beta_dir)
    
    print("\n[Done] All strict validation steps completed for all Beta values.")

if __name__ == "__main__":
    main()
