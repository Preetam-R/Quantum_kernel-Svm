"""
scripts/prepare_train_test_split.py

Project:
Quantum Kernel SVM for Predicting Ampicillin Resistance in Escherichia coli
Using Bacterial Genomic Features

Objective:
Create a single, authoritative, stratified train/test split (80/20, random_state=42)
shared across all genomic feature representations (AMR gene features and k-mer features).

Inputs:
- data/processed/liverpool_ampicillin_matched.csv
- data/processed/amr_gene_feature_matrix.csv
- data/processed/kmer_6_feature_matrix.csv

Output:
- data/processed/train_test_split_ids.csv
"""

from pathlib import Path
import sys
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# 1. FILE PATH CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

MATCHED_FILE = ROOT_DIR / "data" / "processed" / "liverpool_ampicillin_matched.csv"
AMR_MATRIX_FILE = ROOT_DIR / "data" / "processed" / "amr_gene_feature_matrix.csv"
KMER_MATRIX_FILE = ROOT_DIR / "data" / "processed" / "kmer_6_feature_matrix.csv"

OUTPUT_SPLIT_FILE = ROOT_DIR / "data" / "processed" / "train_test_split_ids.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42
EXPECTED_COHORT_SIZE = 719


def main():
    print("=" * 65)
    print("PHASE A: COMMON STRATIFIED TRAIN / TEST SPLIT GENERATION")
    print("=" * 65)

    # ============================================================
    # 2. VERIFY FILE EXISTENCE
    # ============================================================
    for path, desc in [
        (MATCHED_FILE, "Matched cohort metadata"),
        (AMR_MATRIX_FILE, "AMR gene feature matrix"),
        (KMER_MATRIX_FILE, "K-mer 6 feature matrix"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path} ({desc})")

    print("[1/5] Loading datasets...")
    df_matched = pd.read_csv(MATCHED_FILE)
    df_amr = pd.read_csv(AMR_MATRIX_FILE)
    df_kmer = pd.read_csv(KMER_MATRIX_FILE)

    print(f"  - Matched cohort shape:      {df_matched.shape}")
    print(f"  - AMR feature matrix shape:  {df_amr.shape}")
    print(f"  - K-mer feature matrix shape: {df_kmer.shape}")

    # ============================================================
    # 3. PRE-SPLIT VALIDATION CHECKS
    # ============================================================
    print("\n[2/5] Running pre-split validation checks...")

    # Standardize genome_id and label types
    for name, df in [("matched", df_matched), ("amr", df_amr), ("kmer", df_kmer)]:
        if "genome_id" not in df.columns:
            raise ValueError(f"Column 'genome_id' missing from {name} dataset.")
        if "label" not in df.columns:
            raise ValueError(f"Column 'label' missing from {name} dataset.")

        df["genome_id"] = df["genome_id"].astype(str).str.strip()
        df["label"] = df["label"].astype(int)

        # Check 1: Unique genome IDs
        if df["genome_id"].duplicated().any():
            duplicates = df.loc[df["genome_id"].duplicated(), "genome_id"].tolist()
            raise ValueError(f"Duplicate genome IDs found in {name}: {duplicates[:5]}")

        # Check 2: No missing genome IDs or labels
        if df["genome_id"].isna().any():
            raise ValueError(f"Missing genome_id found in {name}.")
        if df["label"].isna().any():
            raise ValueError(f"Missing label found in {name}.")

        # Check 3: Cohort size check
        if len(df) != EXPECTED_COHORT_SIZE:
            raise ValueError(
                f"Expected {EXPECTED_COHORT_SIZE} isolates in {name}, but found {len(df)}."
            )

    # Check 4: ID alignment across all three datasets
    matched_ids = list(df_matched["genome_id"])
    amr_ids = set(df_amr["genome_id"])
    kmer_ids = set(df_kmer["genome_id"])

    if set(matched_ids) != amr_ids:
        diff = set(matched_ids).symmetric_difference(amr_ids)
        raise ValueError(f"ID mismatch between matched cohort and AMR matrix: {diff}")

    if set(matched_ids) != kmer_ids:
        diff = set(matched_ids).symmetric_difference(kmer_ids)
        raise ValueError(f"ID mismatch between matched cohort and K-mer matrix: {diff}")

    # Check 5: Label consistency across all three datasets
    matched_label_map = dict(zip(df_matched["genome_id"], df_matched["label"]))
    amr_label_map = dict(zip(df_amr["genome_id"], df_amr["label"]))
    kmer_label_map = dict(zip(df_kmer["genome_id"], df_kmer["label"]))

    for gid in matched_ids:
        ml = matched_label_map[gid]
        al = amr_label_map[gid]
        kl = kmer_label_map[gid]
        if not (ml == al == kl):
            raise ValueError(
                f"Label conflict for isolate {gid}: matched={ml}, amr={al}, kmer={kl}"
            )

    print("  ✓ All three datasets contain exactly 719 unique, aligned isolates.")
    print("  ✓ Zero missing values in IDs or labels.")
    print("  ✓ 100% label concordance verified across matched, AMR, and k-mer datasets.")

    # Target class distribution in full cohort
    full_counts = df_matched["label"].value_counts().sort_index()
    print(f"\n  Full Cohort Label Distribution:")
    print(f"    - Susceptible (0): {full_counts.get(0, 0)} ({full_counts.get(0, 0) / len(df_matched) * 100:.2f}%)")
    print(f"    - Resistant   (1): {full_counts.get(1, 0)} ({full_counts.get(1, 0) / len(df_matched) * 100:.2f}%)")

    # ============================================================
    # 4. PERFORM REPRODUCIBLE STRATIFIED SPLIT
    # ============================================================
    print("\n[3/5] Performing stratified 80/20 train/test split (random_state=42)...")

    cohort_records = df_matched[["genome_id", "label"]].copy()

    train_df, test_df = train_test_split(
        cohort_records,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=cohort_records["label"]
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    test_df["split"] = "test"

    split_combined = pd.concat([train_df, test_df], ignore_index=True)

    # Sort deterministically by genome_id
    split_combined = split_combined.sort_values(by="genome_id").reset_index(drop=True)

    # ============================================================
    # 5. POST-SPLIT VALIDATION CHECKS
    # ============================================================
    print("\n[4/5] Running post-split validation checks...")

    # Check 6: Exactly one split assignment per isolate
    if len(split_combined) != EXPECTED_COHORT_SIZE:
        raise ValueError(
            f"Expected {EXPECTED_COHORT_SIZE} isolates in combined split, found {len(split_combined)}."
        )

    if split_combined["genome_id"].duplicated().any():
        raise ValueError("Duplicate genome IDs found in split table.")

    train_ids = set(split_combined.loc[split_combined["split"] == "train", "genome_id"])
    test_ids = set(split_combined.loc[split_combined["split"] == "test", "genome_id"])

    # Check 7: Disjointness and completeness
    intersection = train_ids.intersection(test_ids)
    if intersection:
        raise ValueError(f"Data leakage detected! Isolates present in both train and test: {intersection}")

    if train_ids.union(test_ids) != set(matched_ids):
        raise ValueError("Combined split IDs do not equal the complete original cohort.")

    # Calculate split stats
    train_dist = split_combined[split_combined["split"] == "train"]["label"].value_counts().sort_index()
    test_dist = split_combined[split_combined["split"] == "test"]["label"].value_counts().sort_index()

    train_total = len(train_ids)
    test_total = len(test_ids)

    train_r_ratio = train_dist.get(1, 0) / train_total
    train_s_ratio = train_dist.get(0, 0) / train_total
    test_r_ratio = test_dist.get(1, 0) / test_total
    test_s_ratio = test_dist.get(0, 0) / test_total

    print("  ✓ Split disjointness verified: 0 overlapping isolates.")
    print("  ✓ Split completeness verified: Exactly 719 isolates accounted for.")
    print("  ✓ Stratification verified:")
    print(f"      Train: Total={train_total} | Susceptible (0)={train_dist.get(0, 0)} ({train_s_ratio*100:.2f}%) | Resistant (1)={train_dist.get(1, 0)} ({train_r_ratio*100:.2f}%)")
    print(f"      Test:  Total={test_total}  | Susceptible (0)={test_dist.get(0, 0)} ({test_s_ratio*100:.2f}%)  | Resistant (1)={test_dist.get(1, 0)} ({test_r_ratio*100:.2f}%)")

    # ============================================================
    # 6. SAVE OUTPUT
    # ============================================================
    print("\n[5/5] Saving split output...")
    OUTPUT_SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    split_combined[["genome_id", "label", "split"]].to_csv(OUTPUT_SPLIT_FILE, index=False)

    print(f"  ✓ Split file successfully written to:\n    {OUTPUT_SPLIT_FILE}")
    print("=" * 65)
    print("TRAIN / TEST SPLIT CREATION AND VALIDATION SUCCESSFUL.")
    print("=" * 65)


if __name__ == "__main__":
    main()
