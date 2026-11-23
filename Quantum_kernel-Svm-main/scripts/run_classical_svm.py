"""
Train and evaluate classical RBF-SVM baselines on the prepared AMR and k-mer data.

Run from any directory with:
    python scripts/run_classical_svm.py

The script uses the existing train/test split and training-only selected features.
It saves fitted models, per-sample predictions, and a metrics summary.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,


    
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "data" / "reports"
MODEL_DIR = ROOT_DIR / "models" / "classical_svm"

DATASETS = {
    "amr": (
        PROCESSED_DIR / "amr_train_selected.csv",
        PROCESSED_DIR / "amr_test_selected.csv",
    ),
    "kmer_pca": (
        PROCESSED_DIR / "kmer_train_pca.csv",
        PROCESSED_DIR / "kmer_test_pca.csv",
    ),
}


def load_split(train_path: Path, test_path: Path):
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Required input not found: {path}. "
                "Run the split and feature-preparation scripts first."
            )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    required_columns = {"genome_id", "label"}
    for path, frame in ((train_path, train_df), (test_path, test_df)):
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    feature_columns = [column for column in train_df.columns if column not in required_columns]
    test_feature_columns = [column for column in test_df.columns if column not in required_columns]
    if not feature_columns or set(feature_columns) != set(test_feature_columns):
        raise ValueError(
            f"Train/test feature columns do not match for {train_path.stem} and {test_path.stem}."
        )

    if train_df["genome_id"].isna().any() or test_df["genome_id"].isna().any():
        raise ValueError("Missing genome_id values found in train or test data.")
    train_ids = train_df["genome_id"].astype(str).str.strip()
    test_ids = test_df["genome_id"].astype(str).str.strip()
    if train_ids.duplicated().any() or test_ids.duplicated().any():
        raise ValueError("Duplicate genome_id values found in train or test data.")
    if set(train_ids).intersection(test_ids):
        raise ValueError("Train and test data contain overlapping genome IDs.")

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    if set(np.unique(y_train)) != {0, 1} or set(np.unique(y_test)) != {0, 1}:
        raise ValueError("Both train and test labels must contain binary classes 0 and 1.")

    X_train = train_df[feature_columns].to_numpy(dtype=float)
    X_test = test_df[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Feature data contains missing or non-finite numeric values.")

    return (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns,
        test_ids.to_numpy(),
    )


def run_model(dataset_name: str, train_path: Path, test_path: Path):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        feature_columns,
        test_ids,
    ) = load_split(train_path, test_path)

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced"),
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    decision_scores = model.decision_function(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    metrics = {
        "dataset": dataset_name,
        "kernel": "rbf",
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features": len(feature_columns),
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision_resistant": precision_score(y_test, predictions, zero_division=0),
        "recall_resistant": recall_score(y_test, predictions, zero_division=0),
        "f1_resistant": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, decision_scores),
        "specificity_susceptible": tn / (tn + fp) if tn + fp else 0.0,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = REPORT_DIR / f"classical_svm_{dataset_name}_predictions.csv"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "genome_id": test_ids,
            "label_true": y_test,
            "label_predicted": predictions,
            "decision_score": decision_scores,
        }
    ).to_csv(prediction_path, index=False)
    model_path = MODEL_DIR / f"classical_svm_{dataset_name}.joblib"
    joblib.dump(model, model_path)

    print(
        f"{dataset_name}: {len(y_train)} train / {len(y_test)} test, "
        f"{len(feature_columns)} features | "
        f"balanced accuracy={metrics['balanced_accuracy']:.3f}, "
        f"F1(resistant)={metrics['f1_resistant']:.3f}, "
        f"ROC-AUC={metrics['roc_auc']:.3f}"
    )
    print(f"  Model: {model_path}")
    print(f"  Predictions: {prediction_path}")
    return metrics


def main():
    print("Classical SVM baseline (RBF kernel, training-only scaling)")
    metrics = [
        run_model(dataset_name, train_path, test_path)
        for dataset_name, (train_path, test_path) in DATASETS.items()
    ]

    metrics_path = REPORT_DIR / "classical_svm_metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(metrics_path, index=False)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
