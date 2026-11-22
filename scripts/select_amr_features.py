"""
scripts/select_amr_features.py

Project:
Quantum Kernel SVM for Predicting Ampicillin Resistance in Escherichia coli
Using Bacterial Genomic Features

Phase B: Feature Selection & Dimensionality Reduction - AMR Track

Objective:
1. Load split IDs (train_test_split_ids.csv) and AMR feature matrix (amr_gene_feature_matrix.csv).
2. Separate into training (N=575) and test (N=144) sets.
3. Fit feature selectors on TRAINING data ONLY (Strict Anti-Leakage):
   - Remove zero-variance (invariant) features.
   - Remove single-isolate ultra-rare features (min_count < 2).
   - Rank remaining features using Chi-Square (chi2) and Mutual Information.
4. Transform test set using the exact training-fitted feature mask.
5. Generate compact quantum-ready feature subsets (k = 4, 8, 12, 16) and classical subsets.
6. Save all matrices and ranking reports.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2, mutual_info_classif


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

SPLIT_FILE = ROOT_DIR / "data" / "processed" / "train_test_split_ids.csv"
AMR_MATRIX_FILE = ROOT_DIR / "data" / "processed" / "amr_gene_feature_matrix.csv"

REPORT_DIR = ROOT_DIR / "data" / "reports"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

REPORT_FILE = REPORT_DIR / "amr_feature_rankings.csv"

TRAIN_SELECTED_FILE = PROCESSED_DIR / "amr_train_selected.csv"
TEST_SELECTED_FILE = PROCESSED_DIR / "amr_test_selected.csv"

MIN_PRESENCE_COUNT = 2  # Prune features present in < 2 training isolates (noise)
QUANTUM_K_SIZES = [4, 8, 12, 16]


def main():
    print("=" * 65)
    print("PHASE B: AMR GENE FEATURE SELECTION & RANKING")
    print("=" * 65)

    # 1. Load data
    print("[1/5] Loading datasets...")
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Split file missing: {SPLIT_FILE}. Run prepare_train_test_split.py first.")
    if not AMR_MATRIX_FILE.exists():
        raise FileNotFoundError(f"AMR matrix missing: {AMR_MATRIX_FILE}")

    split_df = pd.read_csv(SPLIT_FILE)
    amr_df = pd.read_csv(AMR_MATRIX_FILE)

    print(f"  - Split file records:  {len(split_df)}")
    print(f"  - AMR matrix shape:    {amr_df.shape}")

    # Verify ID alignment
    split_ids = split_df["genome_id"].astype(str).tolist()
    amr_df["genome_id"] = amr_df["genome_id"].astype(str)

    # Merge to ensure exact alignment
    merged = pd.merge(split_df, amr_df, on="genome_id", suffixes=("_split", ""))
    if len(merged) != len(split_df):
        raise ValueError("Mismatch between split IDs and AMR matrix IDs.")

    # 2. Strict Partitioning into Train and Test
    print("\n[2/5] Partitioning into Train (N=575) and Test (N=144)...")
    train_mask = merged["split"] == "train"
    test_mask = merged["split"] == "test"

    train_data = merged[train_mask].copy().reset_index(drop=True)
    test_data = merged[test_mask].copy().reset_index(drop=True)

    y_train = train_data["label"].astype(int).values
    y_test = test_data["label"].astype(int).values

    train_ids = train_data["genome_id"].values
    test_ids = test_data["genome_id"].values

    # Extract raw feature columns (exclude meta columns)
    meta_cols = ["genome_id", "label", "split", "label_split"]
    feature_cols = [c for c in train_data.columns if c not in meta_cols]

    X_train_raw = train_data[feature_cols].values
    X_test_raw = test_data[feature_cols].values

    print(f"  - Training samples: {X_train_raw.shape[0]} (Susceptible={sum(y_train==0)}, Resistant={sum(y_train==1)})")
    print(f"  - Testing samples:  {X_test_raw.shape[0]} (Susceptible={sum(y_test==0)}, Resistant={sum(y_test==1)})")
    print(f"  - Initial AMR features: {len(feature_cols)}")

    # 3. Training-Only Filtering
    print(f"\n[3/5] Applying training-only filters (Zero-variance & count >= {MIN_PRESENCE_COUNT})...")
    train_counts = X_train_raw.sum(axis=0)

    # Zero-variance filter (never detected or detected in all 575)
    non_constant = (train_counts > 0) & (train_counts < len(train_data))
    # Low-frequency pruning
    frequent_enough = train_counts >= MIN_PRESENCE_COUNT

    retained_mask = non_constant & frequent_enough
    selected_indices = np.where(retained_mask)[0]
    selected_features = [feature_cols[i] for i in selected_indices]

    removed_count = len(feature_cols) - len(selected_features)
    print(f"  - Features removed (zero variance or < {MIN_PRESENCE_COUNT} counts): {removed_count}")
    print(f"  - Informative features retained: {len(selected_features)}")

    # 4. Statistical Ranking (Chi-square and Mutual Information)
    print("\n[4/5] Computing Chi-square and Mutual Information on training data...")
    X_train_filtered = X_train_raw[:, selected_indices]
    X_test_filtered = X_test_raw[:, selected_indices]

    chi2_scores, p_values = chi2(X_train_filtered, y_train)
    mi_scores = mutual_info_classif(X_train_filtered, y_train, random_state=42)

    # Calculate biological presence per class in training
    train_s_counts = X_train_filtered[y_train == 0].sum(axis=0)
    train_r_counts = X_train_filtered[y_train == 1].sum(axis=0)

    ranking_df = pd.DataFrame({
        "feature": selected_features,
        "train_total_count": train_counts[selected_indices].astype(int),
        "train_prevalence_pct": (train_counts[selected_indices] / len(y_train) * 100).round(2),
        "train_count_susceptible": train_s_counts.astype(int),
        "train_count_resistant": train_r_counts.astype(int),
        "chi2_score": np.round(chi2_scores, 4),
        "p_value": p_values,
        "mutual_info": np.round(mi_scores, 5)
    })

    # Sort primarily by chi2_score descending, then mutual_info
    ranking_df = ranking_df.sort_values(by=["chi2_score", "mutual_info"], ascending=[False, False]).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    print("\n  Top 10 AMR Features Associated with Ampicillin Resistance in Training Set:")
    top10 = ranking_df.head(10)[["rank", "feature", "train_total_count", "train_count_resistant", "train_count_susceptible", "chi2_score", "p_value"]]
    for _, row in top10.iterrows():
        print(f"    #{int(row['rank']):02d} {row['feature']:<20} | Train: R={int(row['train_count_resistant'])}, S={int(row['train_count_susceptible'])} | Chi2={row['chi2_score']:.2f} (p={row['p_value']:.2e})")

    # 5. Save Outputs
    print("\n[5/5] Saving processed matrices and reports...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Save rankings report
    ranking_df.to_csv(REPORT_FILE, index=False)
    print(f"  ✓ Full feature rankings saved to:\n    {REPORT_FILE}")

    # Build full filtered matrices for classical models
    train_selected_df = pd.DataFrame(X_train_filtered, columns=selected_features)
    train_selected_df.insert(0, "genome_id", train_ids)
    train_selected_df["label"] = y_train
    train_selected_df.to_csv(TRAIN_SELECTED_FILE, index=False)

    test_selected_df = pd.DataFrame(X_test_filtered, columns=selected_features)
    test_selected_df.insert(0, "genome_id", test_ids)
    test_selected_df["label"] = y_test
    test_selected_df.to_csv(TEST_SELECTED_FILE, index=False)

    print(f"  ✓ Classical AMR train matrix saved: {TRAIN_SELECTED_FILE} shape={train_selected_df.shape}")
    print(f"  ✓ Classical AMR test matrix saved:  {TEST_SELECTED_FILE} shape={test_selected_df.shape}")

    # Save compact subsets for Quantum Kernel SVM (k = 4, 8, 12, 16)
    ranked_feature_names = ranking_df["feature"].tolist()
    for k in QUANTUM_K_SIZES:
        if k <= len(ranked_feature_names):
            k_features = ranked_feature_names[:k]
            
            q_train_df = pd.DataFrame(train_data[k_features].values, columns=k_features)
            q_train_df.insert(0, "genome_id", train_ids)
            q_train_df["label"] = y_train
            q_train_file = PROCESSED_DIR / f"amr_train_quantum_k{k}.csv"
            q_train_df.to_csv(q_train_file, index=False)

            q_test_df = pd.DataFrame(test_data[k_features].values, columns=k_features)
            q_test_df.insert(0, "genome_id", test_ids)
            q_test_df["label"] = y_test
            q_test_file = PROCESSED_DIR / f"amr_test_quantum_k{k}.csv"
            q_test_df.to_csv(q_test_file, index=False)

            print(f"  ✓ Quantum subset k={k:<2} saved: amr_train_quantum_k{k}.csv & amr_test_quantum_k{k}.csv")

    print("\n" + "=" * 65)
    print("AMR FEATURE SELECTION COMPLETED SUCCESSFULLY.")
    print("=" * 65)


if __name__ == "__main__":
    main()
