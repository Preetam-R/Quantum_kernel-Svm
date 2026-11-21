"""
scripts/reduce_kmer_features.py

Project:
Quantum Kernel SVM for Predicting Ampicillin Resistance in Escherichia coli
Using Bacterial Genomic Features

Phase B: Feature Selection & Dimensionality Reduction - K-mer Track

Objective:
1. Load split IDs (train_test_split_ids.csv) and K-mer matrix (kmer_6_feature_matrix.csv).
2. Separate into training (N=575) and testing (N=144) sets.
3. Fit Preprocessing & PCA on TRAINING data ONLY (Strict Anti-Leakage):
   - Standardize training k-mers (StandardScaler: zero mean, unit variance).
   - Fit PCA to decompose 2,080 continuous frequency features into orthogonal components.
   - Report explained variance and scree statistics.
   - Scale PCA features into [0, pi] (MinMaxScaler) for quantum angle encoding (e.g. ZZFeatureMap).
4. Supervised Feature Selection (SelectKBest with ANOVA F-test):
   - Identify top individual 6-mers most significantly correlated with ampicillin phenotype.
5. Transform test data using the fitted scalers, PCA, and selectors.
6. Save all reduced matrices and reports.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

SPLIT_FILE = ROOT_DIR / "data" / "processed" / "train_test_split_ids.csv"
KMER_MATRIX_FILE = ROOT_DIR / "data" / "processed" / "kmer_6_feature_matrix.csv"

REPORT_DIR = ROOT_DIR / "data" / "reports"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PCA_REPORT_FILE = REPORT_DIR / "kmer_pca_explained_variance.csv"
KMER_RANKING_FILE = REPORT_DIR / "kmer_selectkbest_rankings.csv"

KMER_TRAIN_PCA_FILE = PROCESSED_DIR / "kmer_train_pca.csv"
KMER_TEST_PCA_FILE = PROCESSED_DIR / "kmer_test_pca.csv"

QUANTUM_K_SIZES = [4, 8, 12, 16]
CLASSICAL_PCA_COMPONENTS = 50


def main():
    print("=" * 65)
    print("PHASE B: K-MER DIMENSIONALITY REDUCTION & PCA (ANTI-LEAKAGE)")
    print("=" * 65)

    # 1. Load data
    print("[1/6] Loading datasets...")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file missing: {SPLIT_FILE}. Run prepare_train_test_split.py first.")
    if not KMER_MATRIX_FILE.exists():
        raise FileNotFoundError(f"K-mer matrix missing: {KMER_MATRIX_FILE}")

    split_df = pd.read_csv(SPLIT_FILE)
    print(f"  - Loaded split table ({len(split_df)} isolates)")

    # Read k-mer matrix
    print("  - Loading 6-mer frequency matrix (33 MB)...")
    kmer_df = pd.read_csv(KMER_MATRIX_FILE)
    print(f"  - K-mer matrix shape: {kmer_df.shape}")

    # Verify ID alignment
    split_df["genome_id"] = split_df["genome_id"].astype(str).str.strip()
    kmer_df["genome_id"] = kmer_df["genome_id"].astype(str).str.strip()

    merged = pd.merge(split_df, kmer_df, on="genome_id", suffixes=("_split", ""))
    if len(merged) != len(split_df):
        raise ValueError("Mismatch between split IDs and K-mer matrix IDs.")

    # 2. Strict Partitioning into Train and Test
    print("\n[2/6] Partitioning into Train (N=575) and Test (N=144)...")
    train_mask = merged["split"] == "train"
    test_mask = merged["split"] == "test"

    train_data = merged[train_mask].copy().reset_index(drop=True)
    test_data = merged[test_mask].copy().reset_index(drop=True)

    y_train = train_data["label"].astype(int).values
    y_test = test_data["label"].astype(int).values

    train_ids = train_data["genome_id"].values
    test_ids = test_data["genome_id"].values

    meta_cols = ["genome_id", "label", "split", "label_split"]
    feature_cols = [c for c in train_data.columns if c not in meta_cols]

    X_train_raw = train_data[feature_cols].values
    X_test_raw = test_data[feature_cols].values

    print(f"  - Training samples: {X_train_raw.shape[0]} (Susceptible={sum(y_train==0)}, Resistant={sum(y_train==1)})")
    print(f"  - Testing samples:  {X_test_raw.shape[0]} (Susceptible={sum(y_test==0)}, Resistant={sum(y_test==1)})")
    print(f"  - K-mer feature dimensions: {len(feature_cols)}")

    # 3. Standardization (fit on train only)
    print("\n[3/6] Standardizing k-mer features (StandardScaler, fit on train only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    print("  ✓ Standardization fitted on training data and applied to test set.")

    # 4. Supervised Feature Selection (ANOVA F-statistic)
    print("\n[4/6] Computing Supervised SelectKBest (ANOVA F-test) on training data...")
    selector = SelectKBest(score_func=f_classif, k="all")
    selector.fit(X_train_scaled, y_train)

    f_scores = selector.scores_
    p_values = selector.pvalues_

    ranking_df = pd.DataFrame({
        "kmer": feature_cols,
        "f_score": np.round(f_scores, 4),
        "p_value": p_values
    }).sort_values(by="f_score", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ranking_df.to_csv(KMER_RANKING_FILE, index=False)
    print(f"  ✓ K-mer ANOVA rankings saved to: {KMER_RANKING_FILE}")
    print("  Top 5 6-mers by ANOVA F-statistic in training data:")
    for _, row in ranking_df.head(5).iterrows():
        print(f"    #{int(row['rank']):02d} {row['kmer']:<15} | F-score={row['f_score']:.2f} | p-value={row['p_value']:.2e}")

    # 5. Principal Component Analysis (fit on train only)
    max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])  # 575 components
    print(f"\n[5/6] Fitting PCA on training data (up to {max_components} components)...")

    pca = PCA(n_components=max_components, random_state=42)
    X_train_pca_all = pca.fit_transform(X_train_scaled)
    X_test_pca_all = pca.transform(X_test_scaled)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    # Build explained variance report
    pca_report_df = pd.DataFrame({
        "principal_component": [f"PC_{i+1}" for i in range(len(evr))],
        "explained_variance_ratio": np.round(evr, 6),
        "cumulative_variance_ratio": np.round(cum_evr, 6)
    })
    pca_report_df.to_csv(PCA_REPORT_FILE, index=False)
    print(f"  ✓ PCA scree and explained variance saved to: {PCA_REPORT_FILE}")

    # Check key variance thresholds
    for threshold in [0.50, 0.70, 0.80, 0.85, 0.90, 0.95]:
        n_needed = int(np.argmax(cum_evr >= threshold) + 1)
        print(f"    - Variance >= {int(threshold*100)}%: {n_needed} components")

    # 6. Save Classical PCA matrices (Top 50 components)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    classical_pc_cols = [f"PC_{i+1}" for i in range(CLASSICAL_PCA_COMPONENTS)]

    train_pca_df = pd.DataFrame(X_train_pca_all[:, :CLASSICAL_PCA_COMPONENTS], columns=classical_pc_cols)
    train_pca_df.insert(0, "genome_id", train_ids)
    train_pca_df["label"] = y_train
    train_pca_df.to_csv(KMER_TRAIN_PCA_FILE, index=False)

    test_pca_df = pd.DataFrame(X_test_pca_all[:, :CLASSICAL_PCA_COMPONENTS], columns=classical_pc_cols)
    test_pca_df.insert(0, "genome_id", test_ids)
    test_pca_df["label"] = y_test
    test_pca_df.to_csv(KMER_TEST_PCA_FILE, index=False)

    print(f"\n[6/6] Saved Classical PCA matrices:")
    print(f"  ✓ Train (50 PCs): {KMER_TRAIN_PCA_FILE} shape={train_pca_df.shape}")
    print(f"  ✓ Test (50 PCs):  {KMER_TEST_PCA_FILE} shape={test_pca_df.shape}")

    # 7. Quantum-ready Scaled Subsets (k = 4, 8, 12, 16 components scaled into [0, pi])
    print("\n  Preparing Quantum-Ready Subsets (scaled strictly to [0, π] for angle encoding):")
    for k in QUANTUM_K_SIZES:
        pc_subset_cols = [f"PC_{i+1}" for i in range(k)]
        
        # Fit MinMaxScaler on training subset only
        q_scaler = MinMaxScaler(feature_range=(0.0, np.pi))
        X_train_q = q_scaler.fit_transform(X_train_pca_all[:, :k])
        X_test_q = q_scaler.transform(X_test_pca_all[:, :k])

        q_train_df = pd.DataFrame(X_train_q, columns=pc_subset_cols)
        q_train_df.insert(0, "genome_id", train_ids)
        q_train_df["label"] = y_train
        q_train_path = PROCESSED_DIR / f"kmer_train_quantum_k{k}.csv"
        q_train_df.to_csv(q_train_path, index=False)

        q_test_df = pd.DataFrame(X_test_q, columns=pc_subset_cols)
        q_test_df.insert(0, "genome_id", test_ids)
        q_test_df["label"] = y_test
        q_test_path = PROCESSED_DIR / f"kmer_test_quantum_k{k}.csv"
        q_test_df.to_csv(q_test_path, index=False)

        cum_var_k = cum_evr[k-1] * 100
        print(f"  ✓ Quantum subset k={k:<2} qubits (Cumulative Variance={cum_var_k:.2f}%):")
        print(f"      Train: {q_train_path.name} | Test: {q_test_path.name} | Range: [{X_train_q.min():.2f}, {X_train_q.max():.2f}]")

    print("\n" + "=" * 65)
    print("K-MER DIMENSIONALITY REDUCTION COMPLETED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    main()
