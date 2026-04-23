# -*- coding: utf-8 -*-
"""
Generate Residualized Microbiome Data for Annealing (NO Study Correction)
Input: X_all_raw.tsv, covariates.tsv (study_id.tsv is NOT used for regression)
Output: X_all_residual_no_study.tsv (Only Age/BMI/Gender corrected)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Load Data
print("Loading data for residualization (NO Study Correction)...")
try:
    X_raw = pd.read_csv("X_all_raw.tsv", sep="\t", index_col=0)
    cov = pd.read_csv("covariates.tsv", sep="\t", index_col=0)
    # Note: study_id is loaded only for alignment, not for regression
    studies = pd.read_csv("study_id.tsv", sep="\t", index_col=0)
except FileNotFoundError as e:
    print(f"Error: Required input file missing. {e}")
    exit()

# Align samples
common = X_raw.index.intersection(cov.index).intersection(studies.index)
X_raw = X_raw.loc[common]
cov = cov.loc[common]
# studies is not added to confounders, but we align just in case

print(f"Aligned N: {len(X_raw)}")

# 2. Preprocess Microbiome (CLR Transform)
print("Applying CLR transformation...")
pseudocount = 1e-6
X_log = np.log(X_raw + pseudocount)

gm = X_log.mean(axis=1)          # Geometric mean per sample
X_clr = X_log.sub(gm, axis=0)    # CLR transformation

# 3. Prepare Confounders (ONLY Covariates, NO Study Dummies)
# We regress out: Age, BMI, Gender ONLY.
# Study effects (Batch effects) are PRESERVED for rigorous LODO validation.
print("Preparing confounders matrix (Age, BMI, Gender only)...")
cov_encoded = pd.get_dummies(cov, drop_first=True)

# --- CHANGE START: Do not include study_dummies ---
# study_dummies = pd.get_dummies(studies, drop_first=True)
# confounders = pd.concat([cov_encoded, study_dummies], axis=1)

confounders = cov_encoded  # Use only biological covariates
# --- CHANGE END ---

# Fill missing values just in case
confounders = confounders.fillna(confounders.median())

# Standardize confounders
scaler = StandardScaler()
C = scaler.fit_transform(confounders)

# 4. Residualization
# Model: Microbiome ~ Age + BMI + Gender
# Residual = Microbiome - Predicted_by_Bio_Covariates
print("Calculating residuals (removing ONLY age/BMI/gender)...")
reg = LinearRegression()
reg.fit(C, X_clr)
X_pred = reg.predict(C)
X_resid = X_clr - X_pred

# 5. Save
output_file = "X_all_residual_no_study.tsv"
X_resid.to_csv(output_file, sep="\t")
print(f"Done! Residualized data (no study correction) saved to: {output_file}")
print("Shape:", X_resid.shape)
