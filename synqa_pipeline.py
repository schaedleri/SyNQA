"""
synqa_pipeline.py -- SyNQA: Synergistic Network QUBO Analysis.

Formulates microbiome biomarker discovery as a QUBO problem that jointly
optimizes standardized single-taxon effect sizes (F, Welch's t-statistic
on CLR-transformed abundances) and pairwise correlation divergence (J,
Fisher z-transformed difference in Pearson correlation) between two
disease groups. Simulated annealing selects the taxon subset minimizing
the resulting energy; bootstrap resampling and label-permutation testing
assess selection stability and statistical significance.

F and J are centered before being passed to the optimizer using a
data-driven Gaussian-mixture-model (GMM) threshold: when the absolute
effect-size distribution shows a statistically supported bimodal
structure (Ashman's D above a threshold, confirmed by a BIC comparison
against a single-component fit), the crossing point between the two
components is used as the centering point; otherwise the design falls
back to the empirical median. This threshold is recomputed independently
for each bootstrap resample and each permutation draw (--center_source
resample, the default) unless --center_source observed is passed, which
freezes the threshold at the value computed from the full observed
dataset.

Selection (F_energy/J_energy, centered) is kept separate from
interpretation: every reported/interpreted quantity (taxon role,
network centrality, clustering) uses the original signed, degree-
normalized F/J values, not the centered energies used to drive the
optimizer.

Usage:
  python synqa_pipeline.py \
      --regress_covariates age,BMI,sex --cohort all --rank all \
      --run_ko --run_perm --centering_mode gmm \
      --n_seeds 30 --n_boot 1000 --n_perm 2000
"""

import sys, os, warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = "/home/3a18004/research/microbiome_quantum/data/wirbel_data"
COHORTS = {
    "WirbelJ_2019": {"meta": f"{DATA_ROOT}/meta_WirbelJ_2019.tsv",
                     "X":    f"{DATA_ROOT}/X_WirbelJ_2019.tsv"},
    "Zeller_2014" : {"meta": f"{DATA_ROOT}/meta_ZellerG_2014.tsv",
                     "X":    f"{DATA_ROOT}/X_ZellerG_2014.tsv"},
    "Yachida_2019": {"meta": f"{DATA_ROOT}/meta_YachidaS_2019.tsv",
                     "X":    f"{DATA_ROOT}/X_YachidaS_2019.tsv"},
}
RANKS = ["s", "g", "f"]

COVARIATE_ALIASES = {
    "age": ["age", "Age", "age_years", "host_age"],
    "BMI": ["BMI", "bmi", "body_mass_index"],
    "sex": ["sex", "Sex", "gender", "Gender", "host_sex"],
}
MALE_SET = {"male", "Male", "M", "m"}
FEMALE_SET = {"female", "Female", "F", "f"}


def _find_col(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    lower_map = {c.lower(): c for c in df.columns}
    for a in aliases:
        if a.lower() in lower_map:
            return lower_map[a.lower()]
    return None


# =============================================================================
# DATA
# =============================================================================

def load_data(meta_path, X_path, rank_code="s",
              min_prev=0.10, group_mode="or",
              min_info_prev=0.10, noise_floor_pct=10.0, info_combine="and",
              return_raw=False, covariates=None):
    meta_raw = pd.read_csv(meta_path, sep="\t")
    sid_col  = next((c for c in ["SampleID","sample_id","Unnamed: 0"]
                     if c in meta_raw.columns), meta_raw.columns[0])
    grp_col  = next((c for c in ["Group","disease","phenotype","study_condition"]
                     if c in meta_raw.columns), None)
    meta_raw = meta_raw.rename(columns={sid_col:"SampleID", grp_col:"Group"})
    meta_raw["SampleID"] = meta_raw["SampleID"].astype(str).str.strip()
    crc_set     = {"CRC","crc","case","adenoma","Adenoma"}
    healthy_set = {"Healthy","healthy","CTR","ctr","control","Control","normal"}
    meta_raw["d"] = meta_raw["Group"].map(
        lambda g: 1 if str(g).strip() in crc_set
                  else (-1 if str(g).strip() in healthy_set else None))
    meta_raw = meta_raw.dropna(subset=["d"])

    cov_lookup = None
    if covariates:
        cov_lookup = meta_raw[["SampleID"]].copy()
        for cov_name in covariates:
            aliases = COVARIATE_ALIASES.get(cov_name, [cov_name])
            found = _find_col(meta_raw, aliases)
            if found is None:
                print(f"    [covariates] WARNING: '{cov_name}' column not found "
                      f"(tried {aliases}) -- this covariate will be all-NaN.")
                cov_lookup[cov_name] = np.nan
            else:
                col = meta_raw[found]
                if cov_name == "sex" and not pd.api.types.is_numeric_dtype(col):
                    col = col.map(lambda v: 1.0 if str(v).strip() in MALE_SET
                                            else (0.0 if str(v).strip() in FEMALE_SET
                                                  else np.nan))
                cov_lookup[cov_name] = pd.to_numeric(col, errors="coerce")
        cov_lookup = cov_lookup.drop_duplicates("SampleID")

    meta_df  = meta_raw[["SampleID","d"]].drop_duplicates("SampleID").reset_index(drop=True)

    X_raw = pd.read_csv(X_path, sep="\t", index_col=0)
    X_raw.index = X_raw.index.astype(str).str.strip()
    meta_ids = set(meta_df["SampleID"])
    if len(meta_ids & set(X_raw.columns.astype(str))) > len(meta_ids & set(X_raw.index)):
        X_raw = X_raw.T; X_raw.index = X_raw.index.astype(str).str.strip()
    keep    = [s for s in X_raw.index if s in meta_ids]
    X_raw   = X_raw.loc[keep].fillna(0.0).clip(lower=0.0)
    meta_df = meta_df[meta_df["SampleID"].isin(set(X_raw.index))].reset_index(drop=True)
    meta_df = meta_df.set_index("SampleID").loc[X_raw.index].rename_axis("SampleID").reset_index()
    assert "SampleID" in meta_df.columns

    prefix    = f"|{rank_code}__"
    rank_cols = [c for c in X_raw.columns if prefix in str(c)]
    if rank_cols:
        X_raw = X_raw[rank_cols].copy()
        def _rname(col):
            for part in col.split("|"):
                if part.startswith(f"{rank_code}__"): return part
            return col.split("|")[-1]
        X_raw.columns = [_rname(c) for c in rank_cols]
        X_raw = X_raw.T.groupby(level=0).sum().T
    raw  = X_raw.loc[meta_df.set_index("SampleID").index]

    d_arr    = meta_df["d"].values.astype(float)
    crc_mask = d_arr == 1
    hly_mask = d_arr == -1

    prev_all = (raw > 0).mean()
    prev_crc = (raw[crc_mask] > 0).mean() if crc_mask.any() else pd.Series(0.0, index=raw.columns)
    prev_hly = (raw[hly_mask] > 0).mean() if hly_mask.any() else pd.Series(0.0, index=raw.columns)
    if group_mode == "pool":
        presence_ok = prev_all >= min_prev
    elif group_mode == "and":
        presence_ok = (prev_crc >= min_prev) & (prev_hly >= min_prev)
    elif group_mode == "or":
        presence_ok = (prev_crc >= min_prev) | (prev_hly >= min_prev)
    else:
        raise ValueError(f"Unknown group_mode: {group_mode!r}")

    nz_vals = raw.values[raw.values > 0]
    noise_floor = float(np.percentile(nz_vals, noise_floor_pct)) if len(nz_vals) > 0 else 0.0
    strict = raw > noise_floor
    prev_crc_i = strict[crc_mask].mean() if crc_mask.any() else pd.Series(0.0, index=raw.columns)
    prev_hly_i = strict[hly_mask].mean() if hly_mask.any() else pd.Series(0.0, index=raw.columns)
    info_ok = (prev_crc_i >= min_info_prev) | (prev_hly_i >= min_info_prev)

    if info_combine == "and":
        keep_mask = presence_ok & info_ok
    elif info_combine == "or":
        keep_mask = presence_ok | info_ok
    else:
        raise ValueError(f"Unknown info_combine: {info_combine!r}")

    n_pres = int(presence_ok.sum()); n_info = int(info_ok.sum()); n_kept = int(keep_mask.sum())
    print(f"    [filter] group_mode={group_mode} min_prev={min_prev}  "
          f"noise_floor(p{noise_floor_pct})={noise_floor:.6g} min_info_prev={min_info_prev}  "
          f"presence_ok={n_pres} info_ok={n_info} combine={info_combine}  "
          f"kept={n_kept}/{raw.shape[1]}")

    raw  = raw.loc[:, keep_mask]
    taxa = list(raw.columns)

    nz_final = raw.values[raw.values > 0]
    pseudo = float(nz_final.min()) / 2.0 if len(nz_final) > 0 else 0.5
    print(f"    [pseudocount] min nonzero value={nz_final.min() if len(nz_final)>0 else float('nan'):.6g}  "
          f"pseudocount(=min/2)={pseudo:.6g}")

    X_clr  = np.log(raw.values.astype(float) + pseudo)
    X_clr -= X_clr.mean(axis=1, keepdims=True)

    cov_array = None
    if covariates:
        merged = meta_df[["SampleID"]].merge(cov_lookup, on="SampleID", how="left")
        cov_array = merged[covariates].values.astype(float)
        n_missing = np.isnan(cov_array).any(axis=1).sum()
        if n_missing > 0:
            print(f"    [covariates] {n_missing}/{len(cov_array)} samples have a "
                  f"missing value in {covariates} (will be dropped before "
                  f"regression, not before F/J -- see regress_out_covariates).")

    outputs = [X_clr, meta_df["d"].values.astype(float), taxa]
    if return_raw:
        outputs.append(raw.values.astype(float))
    if covariates:
        outputs.append(cov_array)
    return tuple(outputs)


def regress_out_covariates(X_clr, cov_array, covariate_names):
    X_clr = np.asarray(X_clr, dtype=float)
    cov_array = np.asarray(cov_array, dtype=float)
    n, p = X_clr.shape
    valid = ~np.isnan(cov_array).any(axis=1)
    if valid.sum() < cov_array.shape[1] + 2:
        print(f"    [covariates] WARNING: only {valid.sum()} samples have complete "
              f"{covariate_names} data -- skipping confound adjustment.")
        return X_clr.copy()

    Z = cov_array[valid]
    Z1 = np.column_stack([np.ones(valid.sum()), Z])
    beta, _, _, _ = np.linalg.lstsq(Z1, X_clr[valid], rcond=None)

    X_adj = X_clr.copy()
    fitted_valid = Z1 @ beta
    resid_valid = X_clr[valid] - fitted_valid
    resid_valid = resid_valid - resid_valid.mean(axis=0, keepdims=True) + X_clr[valid].mean(axis=0, keepdims=True)
    X_adj[valid] = resid_valid

    print(f"    [covariates] regressed out {covariate_names} from {valid.sum()}/{n} "
          f"samples' CLR values.")
    return X_adj


# =============================================================================
# BUILD LAYERS
# =============================================================================

def welch_t(Xa, Xb, na_eff=None, nb_eff=None):
    na, nb = Xa.shape[0], Xb.shape[0]
    na_se = na_eff if na_eff is not None else na
    nb_se = nb_eff if nb_eff is not None else nb
    ma, mb = Xa.mean(axis=0), Xb.mean(axis=0)
    va = Xa.var(axis=0, ddof=1) if na > 1 else np.zeros_like(ma)
    vb = Xb.var(axis=0, ddof=1) if nb > 1 else np.zeros_like(mb)
    se = np.sqrt(va / max(na_se, 1) + vb / max(nb_se, 1)) + 1e-10
    t  = (ma - mb) / se
    return t, ma - mb


def gmm_centering_point(values, ashman_d_threshold=1.0, n_init=5, seed=0):
    """Fit a 2-component Gaussian mixture to `values`. Returns
    (center, ashman_d, used_gmm: bool, reason: str, bic_1: float, bic_2: float).

    bic_1 / bic_2 are the raw BIC of the 1-component and 2-component fits
    (NaN if n<6 or the fit raised), so the BIC gate's margin is
    inspectable rather than only its pass/fail outcome. Note that when
    the BIC gate fails (bic_1 <= bic_2), Ashman's D is never computed
    (ashman_d is NaN): the two checks are a serial gate, not two
    independently-computed criteria.

    `reason` distinguishes why the fallback path was taken:
      "n<6"                      -- too few points to fit a mixture
      "gmm_fit_exception:<repr>" -- sklearn's GaussianMixture.fit() raised
      "bic_gate"                 -- 2-component fit did not beat
                                     1-component fit on BIC
      "ashman_below_threshold"   -- passed the BIC gate but Ashman's D
                                     was still below ashman_d_threshold
      ""                         -- GMM crossing point was used
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    med = float(np.median(x)) if n > 0 else 0.0
    if n < 6:
        return med, float("nan"), False, "n<6", float("nan"), float("nan")

    from sklearn.mixture import GaussianMixture
    xr = x.reshape(-1, 1)
    try:
        gm1 = GaussianMixture(n_components=1, n_init=n_init, random_state=seed).fit(xr)
        gm = GaussianMixture(n_components=2, n_init=n_init, random_state=seed).fit(xr)
    except Exception as e:
        return med, float("nan"), False, f"gmm_fit_exception:{e!r}", float("nan"), float("nan")

    bic_1 = float(gm1.bic(xr))
    bic_2 = float(gm.bic(xr))
    if bic_1 <= bic_2:
        return med, float("nan"), False, "bic_gate", bic_1, bic_2

    means = gm.means_.ravel()
    stds = np.sqrt(gm.covariances_.ravel())
    top_idx = int(np.argmax(means))
    top_mean, top_std = means[top_idx], stds[top_idx]

    post = gm.predict_proba(xr)[:, top_idx]
    rest_mask = post < 0.5
    rest_mean = float(x[rest_mask].mean()) if rest_mask.any() else float(x.min())
    rest_std = float(x[rest_mask].std()) if rest_mask.sum() > 1 else 1e-6

    ashman_d = (np.sqrt(2.0) * abs(top_mean - rest_mean)
                / np.sqrt(top_std ** 2 + rest_std ** 2 + 1e-20))

    if ashman_d < ashman_d_threshold:
        return med, float(ashman_d), False, "ashman_below_threshold", bic_1, bic_2

    grid = np.linspace(rest_mean, top_mean, 2000)
    grid_post = gm.predict_proba(grid.reshape(-1, 1))[:, top_idx]
    crossing = float(grid[np.argmin(np.abs(grid_post - 0.5))])
    return crossing, float(ashman_d), True, "", bic_1, bic_2


def fisher_z_diff(Xa, Xb, r_clip=0.98, na_eff=None, nb_eff=None):
    na, nb = Xa.shape[0], Xb.shape[0]
    na_se = na_eff if na_eff is not None else na
    nb_se = nb_eff if nb_eff is not None else nb
    ra = np.corrcoef(Xa.T) if na > 1 else np.zeros((Xa.shape[1], Xa.shape[1]))
    rb = np.corrcoef(Xb.T) if nb > 1 else np.zeros((Xb.shape[1], Xb.shape[1]))
    ra = np.nan_to_num(np.clip(ra, -r_clip, r_clip))
    rb = np.nan_to_num(np.clip(rb, -r_clip, r_clip))
    za = np.arctanh(ra); zb = np.arctanh(rb)
    se = float(np.sqrt(1.0 / max(na_se - 3, 1) + 1.0 / max(nb_se - 3, 1)))
    Z  = (za - zb) / se
    np.fill_diagonal(Z, 0.0)
    return Z

def degree_scale_normalize(J):
    degree = np.maximum((np.abs(J) > 0).sum(axis=1), 1).astype(float)
    return J / (degree.mean() + 1e-10), degree.mean()


def build_dual_layer_gmm(X, d, edge_z_thresh=1.96, r_clip=0.98,
                               n_eff=None, verbose=True, centering_mode="median",
                               ashman_d_threshold=1.0,
                               fixed_center_F=None, fixed_center_J=None):
    """Returns a dict with F_energy/J_energy (drive the SA optimizer only)
    and F_signed/Z_signed (drive everything else: role assignment,
    clustering, permutation-null purity role signs, etc.).

    fixed_center_F / fixed_center_J: if both are given, they are used
    directly as center_F/center_J (centering_mode and ashman_d_threshold
    are ignored; no GMM fit, no median recomputation). This implements
    --center_source=observed: the centering point is frozen at the value
    computed once from the real-label full dataset, and every bootstrap
    resample and permutation draw is measured against that same absolute
    bar. If either is None (the default, --center_source=resample), the
    centering point is recomputed fresh for this call's own data.
    """
    crc = d == 1; hly = d == -1
    Xc, Xh = X[crc], X[hly]
    nc_eff, nh_eff = n_eff if n_eff is not None else (None, None)
    p = X.shape[1]

    F_signed, _ = welch_t(Xc, Xh, na_eff=nc_eff, nb_eff=nh_eff)
    F_abs = np.abs(F_signed)

    Z_signed = fisher_z_diff(Xc, Xh, r_clip=r_clip, na_eff=nc_eff, nb_eff=nh_eff)
    if not np.all(np.isfinite(Z_signed)):
        n_bad = int(np.sum(~np.isfinite(Z_signed)))
        if verbose:
            print(f"    WARNING: {n_bad} non-finite Z entries -> set to 0")
        Z_signed = np.nan_to_num(Z_signed, nan=0.0, posinf=0.0, neginf=0.0)

    m_full = p * (p - 1) // 2

    edge_mask = np.abs(Z_signed) >= edge_z_thresh
    np.fill_diagonal(edge_mask, False)
    n_edges = int(edge_mask.sum()) // 2
    if verbose:
        print(f"    [Dual] edges with |Z|>={edge_z_thresh}: {n_edges} "
              f"(of {m_full} possible)")
    Z_abs_surviving = np.abs(Z_signed)[edge_mask]

    bic1_F = bic2_F = bic1_J = bic2_J = float("nan")
    ashman_d_F = ashman_d_J = float("nan")
    used_gmm_F = used_gmm_J = False
    reason_F = reason_J = ""

    if fixed_center_F is not None and fixed_center_J is not None:
        # center_source=observed: reuse the caller-supplied centers as-is
        # (no GMM fit, no median recomputation for this call).
        center_F = float(fixed_center_F)
        center_J = float(fixed_center_J)
        if verbose:
            print(f"    [Dual] F centering: FIXED (observed-data) center={center_F:.4f}  "
                  f"|  J centering: FIXED (observed-data) center={center_J:.4f}")
    elif centering_mode == "gmm":
        center_F, ashman_d_F, used_gmm_F, reason_F, bic1_F, bic2_F = gmm_centering_point(
            F_abs, ashman_d_threshold=ashman_d_threshold)
        center_J, ashman_d_J, used_gmm_J, reason_J, bic1_J, bic2_J = gmm_centering_point(
            Z_abs_surviving, ashman_d_threshold=ashman_d_threshold)
        if verbose:
            def _bic_str(bic1, bic2):
                if np.isnan(bic1):
                    return ""
                # BIC gap: positive means the 2-component fit was better
                # (lower BIC) by this many BIC units; negative means the
                # 1-component (unimodal) fit was better by this margin --
                # i.e. how much the bic_gate margin was, not just pass/fail.
                gap = bic1 - bic2
                return f", BIC(k=1)={bic1:.1f} BIC(k=2)={bic2:.1f} gap={gap:+.1f}"
            f_note = (f"GMM crossing (Ashman's D={ashman_d_F:.2f}{_bic_str(bic1_F, bic2_F)})" if used_gmm_F
                     else f"median fallback (Ashman's D={ashman_d_F:.2f}, reason={reason_F}{_bic_str(bic1_F, bic2_F)})"
                     if not np.isnan(ashman_d_F) else f"median fallback (reason={reason_F}{_bic_str(bic1_F, bic2_F)})")
            j_note = (f"GMM crossing (Ashman's D={ashman_d_J:.2f}{_bic_str(bic1_J, bic2_J)})" if used_gmm_J
                     else f"median fallback (Ashman's D={ashman_d_J:.2f}, reason={reason_J}{_bic_str(bic1_J, bic2_J)})"
                     if not np.isnan(ashman_d_J) else f"median fallback (reason={reason_J}{_bic_str(bic1_J, bic2_J)})")
            print(f"    [Dual] F centering: {f_note}  |  J centering: {j_note}")
    else:
        center_F = float(np.median(F_abs)) if len(F_abs) > 0 else 0.0
        center_J = float(np.median(Z_abs_surviving)) if len(Z_abs_surviving) > 0 else 0.0

    F_energy = F_abs - center_F
    J_energy_raw = np.zeros_like(Z_signed)
    J_energy_raw[edge_mask] = np.abs(Z_signed[edge_mask]) - center_J

    Z_signed_sparse = np.where(edge_mask, Z_signed, 0.0)

    J_energy, mean_degree = degree_scale_normalize(J_energy_raw)
    Z_signed_normalized = Z_signed_sparse / (mean_degree + 1e-10)

    return {"Dual": {
        "F_energy": F_energy, "F_signed": F_signed,
        "J_energy": J_energy, "Z_signed": Z_signed_normalized,
        "median_F": center_F, "median_J": center_J,
        "n_edges": n_edges, "mean_degree": mean_degree,
        "centering_mode": centering_mode,
        "bic1_F": bic1_F, "bic2_F": bic2_F, "bic1_J": bic1_J, "bic2_J": bic2_J,
        "ashman_d_F": ashman_d_F, "ashman_d_J": ashman_d_J,
        "used_gmm_F": used_gmm_F, "used_gmm_J": used_gmm_J,
        "reason_F": reason_F, "reason_J": reason_J,
    }}


# =============================================================================
# SA (verbatim)
# =============================================================================

def energy(s, F, J): return -np.dot(F,s) - np.dot(s, J@s)

def sa(F, J, fixed=None, T_start=2.0, T_end=0.001, n_steps=100000, seed=0,
       record_trace=False, trace_interval=1000):
    rng=np.random.default_rng(seed); p=len(F)
    s=rng.integers(0,2,size=p).astype(float)
    if fixed is not None: s[fixed]=0.0
    E=energy(s,F,J); sb=s.copy(); Eb=E
    temps=np.exp(np.linspace(np.log(T_start),np.log(T_end),n_steps))

    trace_E=[]; trace_acc=[]; n_acc=0; n_prop=0
    trace_steps=[]

    for step, T in enumerate(temps):
        i=rng.integers(p)
        if fixed is not None and i==fixed: continue
        sn=s.copy(); sn[i]=1.0-sn[i]; En=energy(sn,F,J); dE=En-E
        n_prop+=1
        if dE<0 or rng.random()<np.exp(-dE/T):
            s=sn; E=En; n_acc+=1
        if E<Eb: sb=s.copy(); Eb=E
        if record_trace and (step+1) % trace_interval == 0:
            trace_E.append(float(E))
            trace_acc.append(n_acc/n_prop if n_prop>0 else 0)
            trace_steps.append(step+1)
            n_acc=0; n_prop=0

    if record_trace:
        return sb, Eb, {"steps":trace_steps, "energy":trace_E, "accept_rate":trace_acc}
    return sb, Eb

def _sa_worker(args):
    F,J,T0,T1,ns,seed = args
    return sa(F,J,T_start=T0,T_end=T1,n_steps=ns,seed=seed)

def sa_best(F, J, n_seeds=20, n_jobs=-1, return_all=False):
    nw   = min(cpu_count() if n_jobs==-1 else n_jobs, n_seeds)
    args = [(F,J,2.0,0.001,100000,s) for s in range(n_seeds)]
    if nw>1:
        with Pool(nw) as pool: results=pool.map(_sa_worker,args)
    else:
        results=[_sa_worker(a) for a in args]
    best = min(results, key=lambda x:x[1])
    if return_all: return best, results
    return best


# =============================================================================
# BOOTSTRAP
# =============================================================================

_BG = {}

def _boot_worker(b):
    X=_BG["X"]; d=_BG["d"]; ci=_BG["ci"]; hi=_BG["hi"]
    ezt=_BG["edge_z_thresh"]; rc=_BG["rc"]
    seed=_BG["seed"]; ns=_BG["ns"]; cmode=_BG.get("centering_mode", "median")
    adt=_BG.get("ashman_d_threshold", 1.0)
    fcF=_BG.get("fixed_center_F", None); fcJ=_BG.get("fixed_center_J", None)
    p=X.shape[1]
    rng=np.random.default_rng(seed+b)
    cb=rng.choice(ci,size=len(ci),replace=True)
    hb=rng.choice(hi,size=len(hi),replace=True)
    nc_eff = int(len(np.unique(cb))); nh_eff = int(len(np.unique(hb)))
    Xb=np.vstack([X[cb],X[hb]]); db=np.concatenate([np.ones(len(cb)),-np.ones(len(hb))])
    try:
        Lb = build_dual_layer_gmm(Xb, db, edge_z_thresh=ezt, r_clip=rc,
                                        n_eff=(nc_eff, nh_eff), verbose=False,
                                        centering_mode=cmode, ashman_d_threshold=adt,
                                        fixed_center_F=fcF, fixed_center_J=fcJ)["Dual"]
        Fe=Lb["F_energy"]; Je=Lb["J_energy"]
        Fs=Lb["F_signed"]; Zs=Lb["Z_signed"]
        sb,_=sa_best(Fe,Je,n_seeds=ns,n_jobs=1)
        eps=1e-10
        Ib=Zs@sb; aF=np.abs(Fs); aI=np.abs(Ib)
        Qb=aI/(aF+eps)*sb
        total_I=aI[sb==1].sum()+eps
        NBb=aI/total_I*sb
        return sb, np.sign(Fs), np.abs(Fs), Qb, NBb, np.sign(Zs), np.abs(Zs), Ib
    except Exception:
        z=np.zeros(p); zz=np.zeros((p,p))
        return z,z,z,z,z,zz,zz,z

def run_bootstrap(X, d, n_boot=100, n_sa_seeds=1, edge_z_thresh=1.96,
                  r_clip=0.98, seed=0, centering_mode="median", ashman_d_threshold=1.0,
                  fixed_center_F=None, fixed_center_J=None):
    ci=np.where(d==1)[0]; hi=np.where(d==-1)[0]; p=X.shape[1]
    _BG.update({"X":X,"d":d,"ci":ci,"hi":hi,
                "edge_z_thresh":edge_z_thresh,
                "rc":r_clip,"seed":seed,"ns":n_sa_seeds,
                "centering_mode":centering_mode,
                "ashman_d_threshold":ashman_d_threshold,
                "fixed_center_F":fixed_center_F,
                "fixed_center_J":fixed_center_J})
    nw=min(cpu_count(),n_boot)
    with Pool(nw) as pool:
        results=pool.map(_boot_worker,range(n_boot))

    sel=np.zeros(p); Fp=np.zeros(p); Fn=np.zeros(p)
    Jp=np.zeros((p,p)); Jn=np.zeros((p,p))
    Qr=[]; NBr=[]; FAr=[]; JAr=[]; Ir=[]; valid=0
    for sb,Fsb,FAb,Qb,NBb,Jsb,JAb,Ib in results:
        if sb.sum()>0:
            sel+=sb; Fp+=(Fsb>0).astype(float); Fn+=(Fsb<0).astype(float)
            Jp+=(Jsb>0).astype(float)
            Jn+=(Jsb<0).astype(float)
            Qr.append(Qb); NBr.append(NBb); FAr.append(FAb); JAr.append(JAb); Ir.append(Ib)
            valid+=1
    n=max(valid,1)
    J_pos=Jp/n
    J_neg=Jn/n
    Jmed=np.zeros((p,p)); Jcv=np.full((p,p), np.nan)
    if JAr:
        Js=np.stack(JAr,axis=0); Jmed=np.median(Js,axis=0)
        Jstd=Js.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Jcv = np.where(Jmed > 1e-8, Jstd / Jmed, np.nan)
    Imed=np.zeros(p); Icv=np.full(p, np.nan)
    if Ir:
        Is=np.abs(np.stack(Ir,axis=0)); Imed=np.median(Is,axis=0)
        Istd=Is.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Icv = np.where(Imed > 1e-8, Istd / Imed, np.nan)
    Fmed=np.zeros(p); Fcv=np.full(p, np.nan)
    if FAr:
        FAs=np.stack(FAr,axis=0); Fmed=np.median(FAs,axis=0)
        Fstd=FAs.std(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            Fcv = np.where(Fmed > 1e-8, Fstd / Fmed, np.nan)
    Qmed=np.zeros(p); Qcv=np.full(p, np.nan)
    if Qr:
        Qs=np.stack(Qr,axis=0)
        for i in range(p):
            v=Qs[:,i]; v=v[v>0]
            if len(v)>=3:
                Qmed[i]=np.median(v)
                Qcv[i]=v.std()/v.mean() if v.mean() > 1e-8 else np.nan
    return {"sel":sel/n,"Fp":Fp/n,"Fn":Fn/n,"Jp":J_pos,"Jn":J_neg,
            "Jmed":Jmed,"Jcv":Jcv,"Imed":Imed,"Icv":Icv,
            "Fmed":Fmed,"Fcv":Fcv,
            "Qmed":Qmed,"Qcv":Qcv,"n_valid":valid}


# =============================================================================
# KO
# =============================================================================

_KO_GLOBAL = {}

def _ko_worker(i):
    F      = _KO_GLOBAL["F"]
    J      = _KO_GLOBAL["J"]
    s_star = _KO_GLOBAL["s_star"]
    E_star = _KO_GLOBAL["E_star"]
    taxa   = _KO_GLOBAL["taxa"]
    n_seeds= _KO_GLOBAL["n_seeds"]
    eps    = 1e-10

    I      = J @ s_star
    aF     = abs(float(F[i])); aI = abs(float(I[i]))

    dEs = [sa(F, J, fixed=i, seed=s*100+i)[1] - E_star for s in range(n_seeds)]

    s_ko, _ = sa(F, J, fixed=i, seed=999+i)
    ds      = s_ko - s_star
    lost    = [taxa[j] for j in range(len(taxa)) if j!=i and ds[j] < -0.5]
    gained  = [taxa[j] for j in range(len(taxa)) if ds[j] > 0.5]

    return {
        "taxon"    : taxa[i],
        "delta_E"  : round(float(np.mean(dEs)), 5),
        "abs_F"    : round(aF, 4),
        "abs_I"    : round(aI, 4),
        "Q_i"      : round(aI / (aF + eps), 4),
        "n_lost"   : len(lost),
        "n_gained" : len(gained),
        "cascade"  : len(lost) + len(gained),
        "lost_taxa": "|".join(lost[:10]),
    }


def run_ko(F, J, s_star, E_star, taxa, n_seeds=5, n_jobs=-1):
    sel_idx = list(np.where(s_star == 1)[0])
    nw      = min(cpu_count() if n_jobs==-1 else n_jobs, len(sel_idx))

    _KO_GLOBAL.update({
        "F": F, "J": J, "s_star": s_star, "E_star": E_star,
        "taxa": taxa, "n_seeds": n_seeds,
    })

    print(f"    KO: {len(sel_idx)} taxa x {n_seeds} seeds x {nw} workers")
    if nw > 1:
        with Pool(nw) as pool:
            rows = pool.map(_ko_worker, sel_idx)
    else:
        rows = [_ko_worker(i) for i in sel_idx]

    return pd.DataFrame(rows).sort_values("delta_E", ascending=False)


# =============================================================================
# INTERPRET -- VERBATIM
# =============================================================================

def build_full_network(J):
    p=J.shape[0]; G=nx.Graph()
    G.add_nodes_from(range(p))
    for i in range(p):
        for j in range(i+1,p):
            w=abs(float(J[i,j]))
            if w>0: G.add_edge(i,j,weight=w)
    return G

def full_network_centrality(J, btw_max_p=400):
    p=J.shape[0]
    default={i:0.0 for i in range(p)}
    if not HAS_NX or p<2: return default, default

    G=build_full_network(J)
    if G.number_of_edges()==0: return default, default

    if p <= btw_max_p:
        try:
            btw=nx.betweenness_centrality(G, weight="weight", normalized=True)
        except Exception:
            btw=default.copy()
    else:
        print(f"    [centrality] p={p}>{btw_max_p}: skipping betweenness, using eigenvector only")
        btw=default.copy()

    try:
        eig=nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eig=default.copy()
    return btw, eig

def cluster_selected_taxa(J, taxa, sel_idx):
    if len(sel_idx) == 0:
        return {}
    if not HAS_NX or len(sel_idx) < 2:
        return {taxa[i]: 0 for i in sel_idx}

    G = nx.Graph()
    G.add_nodes_from(sel_idx)
    for a, i in enumerate(sel_idx):
        for jx in sel_idx[a+1:]:
            w = abs(float(J[i, jx]))
            if w > 0:
                G.add_edge(i, jx, weight=w)

    if G.number_of_edges() == 0:
        return {taxa[i]: c for c, i in enumerate(sel_idx)}

    try:
        communities = list(nx.algorithms.community.greedy_modularity_communities(G, weight="weight"))
    except Exception:
        communities = [set(sel_idx)]

    multi = sorted([c for c in communities if len(c) > 1], key=len, reverse=True)
    single = [c for c in communities if len(c) == 1]
    ordered = multi + single

    cluster_map = {}
    for cid, comm in enumerate(ordered):
        for i in comm:
            cluster_map[taxa[i]] = cid
    return cluster_map

def cluster_purity(cluster_ids, role_signs):
    cluster_ids = np.asarray(cluster_ids); role_signs = np.asarray(role_signs)
    purities = []; sizes = []
    for c in np.unique(cluster_ids):
        mask = cluster_ids == c
        n = int(mask.sum())
        if n < 2:
            continue
        signs = role_signs[mask]
        maj = max(float((signs > 0).mean()), float((signs < 0).mean()))
        purities.append(maj); sizes.append(n)
    if not sizes:
        return float("nan")
    return float(np.average(purities, weights=sizes))

def interpret_layer(layer_name, F, J, s, taxa, boot, ko_df=None):
    p=len(taxa); eps=1e-10
    sel_idx=list(np.where(s==1)[0]); K=len(sel_idx)
    I=J@s; aF=np.abs(F); aI=np.abs(I)
    Q=aI/(aF+eps); EI=aI/(aF+aI+eps)
    total_I=aI[s==1].sum()+eps; NB=aI/total_I
    strength=np.abs(J).sum(axis=1); degree=(np.abs(J)>0).sum(axis=1)
    btw_full, eig_full = full_network_centrality(J)

    Jp=boot["Jp"]; Jn=boot.get("Jn", np.zeros_like(Jp))
    stable_edge=(Jp>=0.9)|(Jn>=0.9); np.fill_diagonal(stable_edge,False)
    n_stable_edges=int(stable_edge.sum())//2

    with np.errstate(invalid="ignore", divide="ignore"):
        edge_exist_frac = Jp + Jn
        edge_sign_consistency = np.where(edge_exist_frac > 1e-8,
                                         np.maximum(Jp, Jn) / edge_exist_frac, np.nan)

    J_stab=np.zeros(p); J_cv_sc=np.full(p, np.nan); J_sign_prob_sc=np.full(p, np.nan)
    for i in sel_idx:
        di=degree[i]
        if di>0:
            J_stab[i]=stable_edge[i].sum()/di
            nb_mask=np.abs(J[i])>0
            if nb_mask.any():
                vals=boot["Jcv"][i,nb_mask]
                J_cv_sc[i]=np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan
                sign_vals=edge_sign_consistency[i,nb_mask]
                sign_vals=sign_vals[np.isfinite(sign_vals)]
                J_sign_prob_sc[i]=float(sign_vals.mean()) if len(sign_vals)>0 else np.nan

    ko_map={}
    if ko_df is not None and len(ko_df)>0:
        for _,row in ko_df.iterrows(): ko_map[row["taxon"]]=row.to_dict()

    # When bootstrap is skipped (--n_boot 0), boot["Fp"] is all-zero, so
    # role/F_sign_dir falls back to the sign of this call's own signed F
    # directly rather than the (meaningless, all-zero) bootstrap fraction.
    boot_has_data = boot.get("n_valid", 0) > 0

    rows=[]
    for i in sel_idx:
        t=taxa[i]
        bp=float(boot["sel"][i]); fp=float(boot["Fp"][i])
        if boot_has_data:
            F_sgn_prob=max(fp,1-fp); F_sgn_dir="pos" if fp>=1-fp else "neg"
        else:
            F_sgn_dir = "pos" if F[i] >= 0 else "neg"
            F_sgn_prob = float("nan")
        F_med_boot=float(boot["Fmed"][i]); F_cv_boot=float(boot["Fcv"][i])
        networkness=float(np.log10(max(Q[i],1e-300)))
        j_sgn = J_sign_prob_sc[i] if np.isfinite(J_sign_prob_sc[i]) else 1.0
        stability=bp*float(np.sqrt(F_sgn_prob*j_sgn)); impact=float(NB[i])
        centrality  = float(btw_full.get(i, 0.0))
        eig_central = float(eig_full.get(i, 0.0))
        dE=ko_map.get(t,{}).get("delta_E",float("nan"))
        cascade=ko_map.get(t,{}).get("cascade",-1)
        rows.append({
            "layer":layer_name,"taxon":t,
            "F_val":round(float(F[i]),4),"I_val":round(float(I[i]),4),
            "Q_i":round(float(Q[i]),4),"EI_i":round(float(EI[i]),4),
            "strength":round(float(strength[i]),5),"degree":int(degree[i]),
            "boot_prob":round(bp,3),"F_sign_prob":round(F_sgn_prob,3),
            "F_sign_dir":F_sgn_dir,
            "F_med_boot":round(F_med_boot,4),
            "F_cv_boot":round(F_cv_boot,4) if np.isfinite(F_cv_boot) else float("nan"),
            "role":"CRC_enriched" if F_sgn_dir=="pos" else "Healthy_enriched",
            "networkness":round(networkness,3),
            "stability":round(stability,4),
            "impact":round(impact,6),
            "centrality":round(centrality,5),
            "eig_central":round(eig_central,5),
            "delta_E":round(float(dE),5) if not np.isnan(dE) else float("nan"),
            "cascade":int(cascade),
            "Q_med_boot":round(float(boot["Qmed"][i]),4),
            "Q_cv_boot":round(float(boot["Qcv"][i]),4),
            "I_med_boot":round(float(boot["Imed"][i]),5),
            "I_cv_boot":round(float(boot["Icv"][i]),4),
            "J_stab_score":round(float(J_stab[i]),4),
            "J_cv_score":round(float(J_cv_sc[i]),4),
            "J_sign_prob":round(float(J_sign_prob_sc[i]),4) if np.isfinite(J_sign_prob_sc[i]) else float("nan"),
            "K":K,
        })
    df=pd.DataFrame(rows)

    Q_sel=Q[s==1]; logQ=np.log10(np.maximum(Q_sel,1e-6))
    summary={
        "layer":layer_name,"K":K,
        "logQ_median":round(float(np.median(logQ)),3) if K>0 else 0,
        "logQ_10pct":round(float(np.percentile(logQ,10)),3) if K>0 else 0,
        "logQ_90pct":round(float(np.percentile(logQ,90)),3) if K>0 else 0,
        "frac_Q_lt_1":round(float(np.mean(Q_sel<1.0)),3) if K>0 else 0,
        "frac_Q_gt_3":round(float(np.mean(Q_sel>3.0)),3) if K>0 else 0,
        "med_stability":round(float(np.nanmedian(df["stability"])),3) if len(df)>0 else 0,
        "n_stable_edges":n_stable_edges,
        "med_J_stab":round(float(np.median(df["J_stab_score"])),3) if len(df)>0 else 0,
        "med_J_cv":round(float(np.nanmedian(df["J_cv_score"])),3) if len(df)>0 and df["J_cv_score"].notna().any() else float("nan"),
        "med_I_cv":round(float(np.nanmedian(df["I_cv_boot"])),3) if len(df)>0 and df["I_cv_boot"].notna().any() else float("nan"),
        "med_delta_E":round(float(np.nanmedian(df["delta_E"])),5) if len(df)>0 else float("nan"),
    }
    return df, summary


# =============================================================================
# IMPORTANCE RANKING -- verbatim
# =============================================================================

def compute_importance(df_layer):
    df=df_layer.copy(); eps=1e-10
    has_dE="delta_E" in df.columns and df["delta_E"].notna().any()
    abs_dE   = df["delta_E"].abs() if has_dE else pd.Series(np.ones(len(df)),index=df.index)
    boot     = df["boot_prob"].fillna(0)
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
    for col in ["score_causal","score_causal_freq_only","score_network","score_full"]:
        mx=df[col].max()
        df[f"{col}_norm"]=(df[col]/(mx+eps)).round(4) if mx>0 else 0.0
    return df.sort_values("score_causal",ascending=False)

def rank_dual(df_dual):
    if len(df_dual) == 0:
        return pd.DataFrame()
    return compute_importance(df_dual)


# =============================================================================
# PIPELINE: one cohort x one rank
# =============================================================================

def run_one(meta_path, X_path, rank_code, cohort_name,
            n_seeds=20, n_boot=100, run_ko_flag=True,
            r_clip=0.98, out_dir=".",
            group_mode="or", min_prev=0.10,
            min_info_prev=0.10, noise_floor_pct=10.0, info_combine="and",
            edge_z_thresh=1.96,
            run_perm_flag=False, n_perm=200, perm_n_seeds=5, perm_seed=0,
            regress_covariates=None, centering_mode="median",
            ashman_d_threshold=1.0, center_source="resample"):
    tag=f"{cohort_name}_{rank_code}"
    print(f"  [{tag}] loading...")
    if regress_covariates:
        X, d, taxa, cov_array = load_data(
            meta_path, X_path, rank_code=rank_code,
            min_prev=min_prev, group_mode=group_mode,
            min_info_prev=min_info_prev, noise_floor_pct=noise_floor_pct,
            info_combine=info_combine, covariates=regress_covariates)
        X = regress_out_covariates(X, cov_array, regress_covariates)
    else:
        X, d, taxa = load_data(
            meta_path, X_path, rank_code=rank_code,
            min_prev=min_prev, group_mode=group_mode,
            min_info_prev=min_info_prev, noise_floor_pct=noise_floor_pct,
            info_combine=info_combine)
    n,p=X.shape
    print(f"  [{tag}] n={n} p={p} CRC={np.sum(d==1)} Healthy={np.sum(d==-1)}")

    # centering_mode decides HOW the center is computed (median or GMM
    # crossing); center_source decides WHERE it's used: "resample"
    # (default) recomputes it fresh for every bootstrap/permutation
    # draw, while "observed" freezes it at this real-label value and
    # reuses it for every draw (see build_dual_layer_gmm() docstring).
    L = build_dual_layer_gmm(X, d, edge_z_thresh=edge_z_thresh, r_clip=r_clip,
                                   centering_mode=centering_mode,
                                   ashman_d_threshold=ashman_d_threshold)["Dual"]
    F_energy=L["F_energy"]; J_energy=L["J_energy"]
    F_signed=L["F_signed"]; Z_signed=L["Z_signed"]
    print(f"  [{tag}] centering_mode={centering_mode}  center_source={center_source}  "
          f"median(|F|)={L['median_F']:.4f}  median(|J|, surviving edges)={L['median_J']:.4f}  "
          f"n_edges={L['n_edges']}  mean_degree={L['mean_degree']:.3f}")

    fixed_center_F = float(L["median_F"]) if center_source == "observed" else None
    fixed_center_J = float(L["median_J"]) if center_source == "observed" else None
    if center_source == "observed":
        print(f"  [{tag}] center_source=observed: freezing center_F={fixed_center_F:.4f}, "
              f"center_J={fixed_center_J:.4f} for ALL bootstrap resamples and permutation draws.")

    print(f"  [{tag}][Dual] SA (n_seeds={n_seeds})...")
    best_res, all_res = sa_best(F_energy,J_energy,n_seeds=n_seeds,return_all=True)
    s,E = best_res
    K   = int(s.sum())
    all_E = np.array([r[1] for r in all_res])
    all_K = np.array([int(r[0].sum()) for r in all_res])
    E_cv  = abs(all_E.std()/all_E.mean())*100 if all_E.mean()!=0 else 0
    all_s = np.stack([r[0] for r in all_res])
    n_r=len(all_s); pairs=[(i,j) for i in range(n_r) for j in range(i+1,n_r)]
    if len(pairs)>50:
        rng2=np.random.default_rng(0)
        pairs=[pairs[k] for k in rng2.choice(len(pairs),50,replace=False)]
    jacs=[]
    for i,j in pairs:
        inter=float((all_s[i]*all_s[j]).sum())
        union=float(((all_s[i]+all_s[j])>0).sum())
        jacs.append(inter/union if union>0 else 0)
    jac_mean=float(np.mean(jacs)) if jacs else 0
    print(f"  [{tag}][Dual] K={K}  E={E:.3f}  E_cv={E_cv:.2f}%  "
          f"Jaccard={jac_mean:.3f}  K_range=[{int(all_K.min())},{int(all_K.max())}]")
    if E_cv > 2.0:
        print(f"    NOTE: E_cv={E_cv:.1f}% > 2% -- consider increasing n_seeds further")
    if jac_mean < 0.70:
        print(f"    NOTE: Jaccard={jac_mean:.3f} < 0.70 -- landscape may have multiple basins")

    # Default (used when --n_boot 0): all bootstrap-derived quantities
    # are zero/NaN, matching run_bootstrap()'s actual return schema.
    boot={"sel":np.zeros(p),"Fp":np.zeros(p),"Fn":np.zeros(p),
          "Jp":np.zeros((p,p)),"Jn":np.zeros((p,p)),"Jmed":np.zeros((p,p)),"Jcv":np.zeros((p,p)),
          "Imed":np.zeros(p),"Icv":np.zeros(p),
          "Fmed":np.zeros(p),"Fcv":np.full(p, np.nan),
          "Qmed":np.zeros(p),"Qcv":np.zeros(p),"n_valid":0}
    if n_boot>0:
        print(f"  [{tag}][Dual] bootstrap n={n_boot}...")
        boot=run_bootstrap(X,d,n_boot=n_boot,n_sa_seeds=1,edge_z_thresh=edge_z_thresh,
                           r_clip=r_clip,centering_mode=centering_mode,
                           ashman_d_threshold=ashman_d_threshold,
                           fixed_center_F=fixed_center_F, fixed_center_J=fixed_center_J)

    ko_df=None
    if run_ko_flag:
        print(f"  [{tag}][Dual] KO ({K} taxa)...")
        ko_df=run_ko(F_energy,J_energy,s,E,taxa,n_seeds=n_seeds)
        if ko_df is not None:
            ko_df.to_csv(f"{out_dir}/ko_{tag}_Dual.tsv",sep="\t",index=False)

    df,summary=interpret_layer("Dual",F_signed,Z_signed,s,taxa,boot,ko_df)
    summary["median_F"] = round(L["median_F"], 4)
    summary["median_J"] = round(L["median_J"], 4)
    summary["centering_mode"] = centering_mode
    summary["bic1_F"] = round(L["bic1_F"], 2) if not np.isnan(L["bic1_F"]) else float("nan")
    summary["bic2_F"] = round(L["bic2_F"], 2) if not np.isnan(L["bic2_F"]) else float("nan")
    summary["bic_gap_F"] = (round(L["bic1_F"] - L["bic2_F"], 2)
                            if not np.isnan(L["bic1_F"]) else float("nan"))
    summary["bic1_J"] = round(L["bic1_J"], 2) if not np.isnan(L["bic1_J"]) else float("nan")
    summary["bic2_J"] = round(L["bic2_J"], 2) if not np.isnan(L["bic2_J"]) else float("nan")
    summary["bic_gap_J"] = (round(L["bic1_J"] - L["bic2_J"], 2)
                            if not np.isnan(L["bic1_J"]) else float("nan"))
    # F_method / J_method: human-readable label matching the console log's
    # own wording ("GMM crossing" vs "median fallback"), plus the Ashman's
    # D actually used to decide it (NaN when the BIC gate short-circuited
    # before Ashman's D was even computed -- see gmm_centering_point()).
    summary["F_method"] = ("GMM crossing" if L["used_gmm_F"]
                           else "median fallback" if centering_mode == "gmm"
                           else "median (centering_mode=median)")
    summary["J_method"] = ("GMM crossing" if L["used_gmm_J"]
                           else "median fallback" if centering_mode == "gmm"
                           else "median (centering_mode=median)")
    summary["F_ashman_d"] = (round(L["ashman_d_F"], 2) if not np.isnan(L["ashman_d_F"]) else float("nan"))
    summary["J_ashman_d"] = (round(L["ashman_d_J"], 2) if not np.isnan(L["ashman_d_J"]) else float("nan"))
    summary["center_source"] = center_source
    summary["E_cv_pct"] = round(float(E_cv), 4)
    summary["sa_jaccard"] = round(float(jac_mean), 4)
    summary["K_min_across_seeds"] = int(all_K.min())
    summary["K_max_across_seeds"] = int(all_K.max())

    sel_idx = list(np.where(s == 1)[0])
    cluster_map = cluster_selected_taxa(Z_signed, taxa, sel_idx)
    df["cluster"] = df["taxon"].map(cluster_map).fillna(-1).astype(int)
    cluster_sizes = df.loc[df["cluster"] >= 0, "cluster"].value_counts()
    n_multi = int((cluster_sizes > 1).sum())
    n_single = int((cluster_sizes == 1).sum())

    df.to_csv(f"{out_dir}/taxa_{tag}.tsv",sep="\t",index=False)
    summary["tag"] = tag
    df_sum=pd.DataFrame([summary])
    df_sum.to_csv(f"{out_dir}/summary_{tag}.tsv",sep="\t",index=False)

    ranked=rank_dual(df)
    if len(ranked)>0:
        ranked.to_csv(f"{out_dir}/rank_{tag}_Dual.tsv",sep="\t",index=False)

    print(f"\n  Landscape [{tag}] (Dual):")
    print(f"    K={summary['K']}  logQ_med={summary['logQ_median']:+.3f}  "
          f"frac<1={summary['frac_Q_lt_1']:.3f}  frac>3={summary['frac_Q_gt_3']:.3f}  "
          f"stab={summary['med_stability']:.3f}  stbl_edges={summary['n_stable_edges']}  "
          f"J_cv={summary['med_J_cv']:.2f}  I_cv={summary['med_I_cv']:.2f}")

    if len(ranked)>0:
        top=ranked[ranked["boot_prob"]>=0.5].head(10)
        if len(top)>0:
            print(f"\n  Importance [Dual] (score_causal=|dE|xboot; interpretation uses "
                  f"signed F/J, selection used median-centered |F|/|J|):")
            n_healthy_top = int((top["role"] == "Healthy_enriched").sum())
            print(f"    {'score':>7}  {'net':>6}  {'stab':>5}  {'str':>6}  {'|dE|':>7}  "
                  f"{'role':17s}  Taxon")
            print("    "+"-"*80)
            for _,r in top.iterrows():
                name=r["taxon"].replace("s__","").replace("g__","").replace("f__","")[:26]
                de=f"{r['delta_E']:.4f}" if not np.isnan(r["delta_E"]) else "  -   "
                print(f"    {r['score_causal']:>7.4f}  {r['networkness']:>+6.2f}  "
                      f"{r['boot_prob']:>5.2f}  {r['strength']:>6.4f}  {de}  "
                      f"{r['role']:17s}  {name}")
            if n_healthy_top > 0:
                print(f"    ({n_healthy_top}/{len(top)} of the top-10 are Healthy_enriched)")

    n_healthy_all = int((df["role"] == "Healthy_enriched").sum())
    print(f"\n  [{tag}] Healthy_enriched taxa in the full selection: {n_healthy_all}/{K}")

    if len(df)>0:
        print(f"\n  Clusters [Dual] ({n_multi} multi-member + {n_single} singleton, K={K} total):")
        print(f"    {'cid':>3}  {'size':>4}  {'CRC':>4}  {'Hlth':>4}  top taxa (by strength)")
        print("    "+"-"*80)
        for cid, grp in df[df["cluster"]>=0].groupby("cluster"):
            if len(grp) < 2:
                continue
            n_crc=int((grp["role"]=="CRC_enriched").sum())
            n_hly=int((grp["role"]=="Healthy_enriched").sum())
            top_names=(grp.sort_values("strength",ascending=False)["taxon"]
                       .str.replace("s__","",regex=False).str.replace("g__","",regex=False)
                       .str.replace("f__","",regex=False).head(4).tolist())
            print(f"    {cid:>3}  {len(grp):>4}  {n_crc:>4}  {n_hly:>4}  {', '.join(top_names)}")
        if n_single>0:
            print(f"    (+{n_single} singleton taxa with no surviving edge to another selected taxon)")

    obs_n_edges = int((np.abs(Z_signed) > 0).sum()) // 2
    obs_purity = cluster_purity(
        df["cluster"].values,
        np.where(df["F_val"].values >= 0, 1, -1)
    ) if len(df) > 0 else float("nan")

    perm_df = None
    if run_perm_flag:
        perm_df = run_permutation_test(
            X, d, taxa, tag, out_dir,
            obs_n_edges=obs_n_edges, obs_K=K, obs_purity=obs_purity,
            n_perm=n_perm, edge_z_thresh=edge_z_thresh,
            r_clip=r_clip, n_seeds=perm_n_seeds, seed=perm_seed,
            centering_mode=centering_mode, ashman_d_threshold=ashman_d_threshold,
            fixed_center_F=fixed_center_F, fixed_center_J=fixed_center_J)

    return df, df_sum, ranked, perm_df


# =============================================================================
# PERMUTATION TEST
# =============================================================================

_PG = {}

def _perm_worker(i):
    X=_PG["X"]; d=_PG["d"]; taxa=_PG["taxa"]
    ezt=_PG["edge_z_thresh"]; rc=_PG["rc"]
    ns=_PG["n_seeds"]; seed=_PG["seed"]; cmode=_PG.get("centering_mode", "median")
    adt=_PG.get("ashman_d_threshold", 1.0)
    fcF=_PG.get("fixed_center_F", None); fcJ=_PG.get("fixed_center_J", None)
    rng = np.random.default_rng(seed + i)
    d_perm = rng.permutation(d)
    try:
        # center_source="observed": fcF/fcJ (if set) are frozen at the
        # real-label value and applied unchanged to this shuffled draw.
        L = build_dual_layer_gmm(X, d_perm, edge_z_thresh=ezt, r_clip=rc,
                                       verbose=False, centering_mode=cmode,
                                       ashman_d_threshold=adt,
                                       fixed_center_F=fcF, fixed_center_J=fcJ)["Dual"]
        Fe=L["F_energy"]; Je=L["J_energy"]; Zs=L["Z_signed"]
        # Purity's role_signs use F_signed (disease direction), matching
        # obs_purity in run_one() -- not F_energy, which only reflects
        # distance from this draw's own centering point.
        Fs=L["F_signed"]
        n_edges = L["n_edges"]
        s, _E = sa_best(Fe, Je, n_seeds=ns, n_jobs=1)
        K = int(s.sum())
        if K < 2:
            return n_edges, K, float("nan")
        sel_idx = list(np.where(s == 1)[0])
        cmap = cluster_selected_taxa(Zs, taxa, sel_idx)
        cluster_ids = np.array([cmap.get(taxa[i2], -1) for i2 in sel_idx])
        role_signs  = np.array([1 if Fs[i2] >= 0 else -1 for i2 in sel_idx])
        purity = cluster_purity(cluster_ids, role_signs)
        return n_edges, K, purity
    except Exception:
        return 0, 0, float("nan")

def run_permutation_test(X, d, taxa, tag, out_dir, obs_n_edges, obs_K, obs_purity,
                         n_perm=200, edge_z_thresh=1.96,
                         r_clip=0.98, n_seeds=5, seed=0, centering_mode="median",
                         ashman_d_threshold=1.0,
                         fixed_center_F=None, fixed_center_J=None):
    _PG.update({"X":X,"d":d,"taxa":taxa,"edge_z_thresh":edge_z_thresh,
                "rc":r_clip,"n_seeds":n_seeds,"seed":seed,
                "ashman_d_threshold":ashman_d_threshold,
                "centering_mode":centering_mode,
                "fixed_center_F":fixed_center_F,
                "fixed_center_J":fixed_center_J})
    nw = min(cpu_count(), n_perm)
    print(f"  [{tag}][Dual] permutation test: n_perm={n_perm}, "
          f"perm_n_seeds={n_seeds} (label-shuffled nulls)...")
    with Pool(nw) as pool:
        results = pool.map(_perm_worker, range(n_perm))
    n_edges_null = np.array([r[0] for r in results], dtype=float)
    K_null       = np.array([r[1] for r in results], dtype=float)
    purity_null  = np.array([r[2] for r in results], dtype=float)

    def emp_p_two_sided(obs, null):
        null = null[np.isfinite(null)]
        if obs is None or np.isnan(obs) or len(null) == 0:
            return float("nan"), "n/a"
        n = len(null)
        p_upper = (1 + np.sum(null >= obs)) / (n + 1)
        p_lower = (1 + np.sum(null <= obs)) / (n + 1)
        p_two = float(min(1.0, 2 * min(p_upper, p_lower)))
        direction = "high" if p_upper <= p_lower else "low"
        return p_two, direction

    p_edges, dir_edges   = emp_p_two_sided(obs_n_edges, n_edges_null)
    p_K, dir_K           = emp_p_two_sided(obs_K, K_null)
    p_purity, dir_purity = emp_p_two_sided(obs_purity, purity_null)

    result = {
        "tag": tag, "n_perm": n_perm, "perm_n_seeds": n_seeds,
        "obs_n_edges": obs_n_edges,
        "null_n_edges_mean": round(float(np.nanmean(n_edges_null)), 1),
        "null_n_edges_std": round(float(np.nanstd(n_edges_null)), 1),
        "p_n_edges": round(p_edges, 4) if not np.isnan(p_edges) else float("nan"),
        "dir_n_edges": dir_edges,
        "obs_K": obs_K,
        "null_K_mean": round(float(np.nanmean(K_null)), 1),
        "null_K_std": round(float(np.nanstd(K_null)), 1),
        "p_K": round(p_K, 4) if not np.isnan(p_K) else float("nan"),
        "dir_K": dir_K,
        "obs_purity": round(obs_purity, 4) if obs_purity is not None and not np.isnan(obs_purity) else float("nan"),
        "null_purity_mean": round(float(np.nanmean(purity_null)), 4) if np.isfinite(purity_null).any() else float("nan"),
        "null_purity_std": round(float(np.nanstd(purity_null)), 4) if np.isfinite(purity_null).any() else float("nan"),
        "p_purity": round(p_purity, 4) if not np.isnan(p_purity) else float("nan"),
        "dir_purity": dir_purity,
    }
    print(f"    n_edges: obs={obs_n_edges}  null={result['null_n_edges_mean']:.0f}"
          f"+/-{result['null_n_edges_std']:.0f}  p={result['p_n_edges']} (dir={dir_edges})")
    print(f"    K      : obs={obs_K}  null={result['null_K_mean']:.1f}"
          f"+/-{result['null_K_std']:.1f}  p={result['p_K']} (dir={dir_K})")
    print(f"    purity : obs={result['obs_purity']}  null={result['null_purity_mean']}  "
          f"p={result['p_purity']} (dir={dir_purity})")

    df = pd.DataFrame([result])
    df.to_csv(f"{out_dir}/permtest_{tag}.tsv", sep="\t", index=False)
    return df


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="DualModel Pipeline variant offering an ADAPTIVE GMM "
                     "THRESHOLD as a data-driven alternative to "
                     "median-centering. See --centering_mode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cohort",   default="all",
                        choices=list(COHORTS.keys())+["all"])
    parser.add_argument("--rank",     default="all",
                        choices=["s","g","f","all"])
    parser.add_argument("--n_seeds",  type=int, default=20)
    parser.add_argument("--n_boot",   type=int, default=100)
    parser.add_argument("--run_ko",   action="store_true", default=False)
    parser.add_argument("--out_dir",  default="dualmodel_results_confound_adjusted_gmm")
    parser.add_argument("--min_prev", type=float, default=0.10)
    parser.add_argument("--group_mode", default="or", choices=["pool","and","or"])
    parser.add_argument("--min_info_prev", type=float, default=0.10)
    parser.add_argument("--noise_floor_pct", type=float, default=10.0)
    parser.add_argument("--info_combine", default="and", choices=["and","or"])
    parser.add_argument("--r_clip", type=float, default=0.98)
    parser.add_argument("--edge_z_thresh", type=float, default=1.96,
                        help="Existence threshold: pairs with |Z| below this "
                             "are NOT edges at all (never centered/considered). "
                             "Same meaning as in the unmodified pipeline.")
    parser.add_argument("--run_perm", action="store_true", default=False)
    parser.add_argument("--n_perm", type=int, default=200)
    parser.add_argument("--perm_n_seeds", type=int, default=5)
    parser.add_argument("--regress_covariates", default=None,
                        help="Comma-separated covariate names to regress out "
                             "of CLR values before F/J construction, e.g. "
                             "'age,BMI,sex'.")
    parser.add_argument("--centering_mode", default="median",
                        choices=["median", "gmm"],
                        help="'median' (default) or 'gmm' (data-driven GMM "
                             "crossing point, falling back to median).")
    parser.add_argument("--ashman_d_threshold", type=float, default=1.0,
                        help="Only used when --centering_mode=gmm. Minimum "
                             "Ashman's D for a side (F or J) to use the GMM "
                             "crossing point instead of falling back to the "
                             "median.")
    parser.add_argument("--center_source", default="resample",
                        choices=["resample", "observed"],
                        help="'resample' (default, prior behavior): the "
                             "centering point (median or GMM crossing, per "
                             "--centering_mode) is recomputed FRESH from "
                             "each bootstrap resample's / each permutation "
                             "draw's OWN |F|/|J| distribution -- this is "
                             "what both dualmodel_run_confound_adjusted_"
                             "symmetric.py and the original centering_mode="
                             "gmm variant of this file did. Because the "
                             "center is always relative to that draw's own "
                             "data, roughly the same fraction of taxa/edges "
                             "clears the bar whether the labels are real or "
                             "shuffled, which can make K insensitive to "
                             "real-vs-permuted labels even when the "
                             "UNDERLYING |F|/|J| SCALE does differ (as "
                             "n_edges, an absolute-threshold statistic, "
                             "typically shows). "
                             "'observed' (NEW): the centering point is "
                             "computed ONCE from the real-label full "
                             "dataset and then FROZEN -- every bootstrap "
                             "resample and every permutation draw is "
                             "measured against that same fixed, absolute "
                             "bar. This is the variant to use to test "
                             "whether K's permutation-insensitivity is "
                             "caused by the resample-relative recentering "
                             "(rather than by the symmetric/absolute-value "
                             "F/J formulation itself).")
    args = parser.parse_args()

    regress_covariates = (args.regress_covariates.split(",")
                          if args.regress_covariates else None)

    cohorts_to_run=({args.cohort:COHORTS[args.cohort]}
                    if args.cohort in COHORTS else COHORTS)
    ranks_to_run  =([args.rank] if args.rank in RANKS else RANKS)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nDualModel Pipeline (confound-adjusted, GMM-THRESHOLD variant)")
    print(f"  cohorts: {list(cohorts_to_run.keys())}")
    print(f"  ranks  : {ranks_to_run}")
    print(f"  regress_covariates: {regress_covariates}")
    print(f"  centering_mode: {args.centering_mode}")
    print(f"  center_source: {args.center_source}")
    print(f"  n_seeds: {args.n_seeds}  n_boot: {args.n_boot}  run_ko: {args.run_ko}\n")

    all_taxa=[]; all_ranked=[]; all_perm=[]; all_summary=[]

    for cohort_name,paths in cohorts_to_run.items():
        if not os.path.exists(paths["meta"]):
            print(f"Skipping {cohort_name}: not found"); continue
        for rank_code in ranks_to_run:
            try:
                df_t,df_s,df_r,df_p=run_one(
                    paths["meta"],paths["X"],
                    rank_code=rank_code,cohort_name=cohort_name,
                    n_seeds=args.n_seeds,n_boot=args.n_boot,
                    run_ko_flag=args.run_ko,out_dir=out_dir,
                    group_mode=args.group_mode,min_prev=args.min_prev,
                    min_info_prev=args.min_info_prev,
                    noise_floor_pct=args.noise_floor_pct,
                    info_combine=args.info_combine,
                    r_clip=args.r_clip,
                    edge_z_thresh=args.edge_z_thresh,
                    run_perm_flag=args.run_perm,n_perm=args.n_perm,
                    perm_n_seeds=args.perm_n_seeds,
                    regress_covariates=regress_covariates,
                    centering_mode=args.centering_mode,
                    ashman_d_threshold=args.ashman_d_threshold,
                    center_source=args.center_source)
                df_t["cohort"]=cohort_name; df_t["rank"]=rank_code
                if len(df_r)>0:
                    df_r["cohort"]=cohort_name; df_r["rank"]=rank_code
                all_taxa.append(df_t); all_ranked.append(df_r)
                if len(df_s)>0:
                    all_summary.append(df_s)
                if df_p is not None and len(df_p)>0:
                    df_p["cohort"]=cohort_name; df_p["rank"]=rank_code
                    all_perm.append(df_p)
            except Exception as e:
                print(f"ERROR {cohort_name} {rank_code}: {e}")
                import traceback; traceback.print_exc()

    if len(all_taxa)>1:
        df_mega=pd.concat(all_taxa,ignore_index=True)
        df_mega.to_csv(f"{out_dir}/all_taxa.tsv",sep="\t",index=False)
        df_rank_all=pd.concat([r for r in all_ranked if len(r)>0],ignore_index=True) \
                    if any(len(r)>0 for r in all_ranked) else pd.DataFrame()
        if len(df_rank_all)>0:
            df_rank_all.to_csv(f"{out_dir}/all_ranked.tsv",sep="\t",index=False)

    if len(all_perm)>0:
        df_perm_all=pd.concat(all_perm,ignore_index=True)
        df_perm_all.to_csv(f"{out_dir}/all_permtest.tsv",sep="\t",index=False)

    if len(all_summary)>0:
        df_summary_all = pd.concat(all_summary, ignore_index=True)
        diag_cols = ["tag", "median_F", "F_method", "F_ashman_d", "bic_gap_F",
                    "median_J", "J_method", "J_ashman_d", "bic_gap_J"]
        diag_cols = [c for c in diag_cols if c in df_summary_all.columns]
        diag_df = df_summary_all[diag_cols].rename(columns={
            "tag": "pattern", "median_F": "F_threshold", "median_J": "J_threshold",
        })
        diag_path = f"{out_dir}/gmm_centering_bic_diagnostics.tsv"
        diag_df.to_csv(diag_path, sep="\t", index=False)
        print(f"[gmm_centering_bic_diagnostics] {len(diag_df)} pattern(s) -> {diag_path}")

    print(f"\nDone -> {out_dir}/")
