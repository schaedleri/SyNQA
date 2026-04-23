import pandas as pd
import numpy as np
import os

def generate_demo_data():
    print("Generating demonstration data for SyNQA...")
    
    # 1. Basic settings
    np.random.seed(42)
    n_samples = 60  # Number of samples (set to a small number for testing)
    n_taxa = 150    # Number of microbiome features (taxa)

    sample_ids = [f"Sample_{i:03d}" for i in range(1, n_samples + 1)]
    
    # Generate taxa names (mixing in representative bacteria from papers)
    taxa_names = [f"s__Mock_Bacterium_{i}" for i in range(1, n_taxa + 1)]
    taxa_names[0] = "s__Fusobacterium_nucleatum" # Fixed: Was taxa_names = "..."
    taxa_names[1] = "s__Prevotella_intermedia"
    taxa_names[2] = "s__Dialister_pneumosintes"
    taxa_names[3] = "s__Parvimonas_micra"
    taxa_names[4] = "s__Bacteroides_vulgatus"

    # 2. Microbiome relative abundance data (X_all_raw.tsv)
    # Generate using a log-normal distribution, setting sparsity (number of zeros) to 60% for realism
    X_raw = np.random.lognormal(mean=0, sigma=2, size=(n_samples, n_taxa))
    X_raw[np.random.rand(n_samples, n_taxa) < 0.60] = 0
    
    # Convert to relative abundance (row sums equal 1)
    X_raw = X_raw / X_raw.sum(axis=1, keepdims=True)
    
    df_X = pd.DataFrame(X_raw, index=sample_ids, columns=taxa_names)
    df_X.to_csv("X_all_raw.tsv", sep="\t")

    # 3. Covariate data (covariates.tsv)
    age = np.random.randint(40, 80, size=n_samples)
    sex = np.random.choice(["Male", "Female"], size=n_samples)
    bmi = np.random.normal(24.5, 3.5, size=n_samples)
    
    df_cov = pd.DataFrame({"Age": age, "Sex": sex, "BMI": bmi}, index=sample_ids)
    df_cov.to_csv("covariates.tsv", sep="\t")

    # 4. Target variable / diagnostic labels (y_all.tsv)
    # 0: Control, 1: CRC
    y = np.random.choice([0, 1], size=n_samples) # Fixed: Changed [1] to [0, 1]
    
    df_y = pd.DataFrame({"CRC": y}, index=sample_ids)
    df_y.to_csv("y_all.tsv", sep="\t")

    # 5. Cohort information (study_id.tsv)
    # Generate 3 cohorts so LOGO (Leave-One-Group-Out) cross-validation can run
    study_ids = np.random.choice(["Cohort_Germany", "Cohort_France", "Cohort_Japan"], size=n_samples)
    
    df_study = pd.DataFrame({"Study_ID": study_ids}, index=sample_ids)
    df_study.to_csv("study_id.tsv", sep="\t")

    print("Success! Created 4 demonstration files:")
    print(" - X_all_raw.tsv")
    print(" - covariates.tsv")
    print(" - y_all.tsv")
    print(" - study_id.tsv")

if __name__ == "__main__":
    generate_demo_data()
