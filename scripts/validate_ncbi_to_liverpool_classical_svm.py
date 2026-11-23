

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

NCBI_FILE = (
    ROOT / "data" / "processed" / "additional_dataset"
    / "additional_amr_gene_feature_matrix.csv"
)

LIVERPOOL_FILE = (
    ROOT / "data" / "processed" / "amr_gene_feature_matrix.csv"
)

FEATURE_FILE = (
    ROOT / "results" / "quantum" / "external_validation"
    / "amr" / "ncbi_to_liverpool_k12"
    / "selected_ncbi_quantum_features.json"
)

OUTPUT_DIR = (
    ROOT / "results" / "classical_svm" / "external_validation"
    / "amr" / "ncbi_to_liverpool_k12"
)

NCBI_ID_COLUMN = "assembly_ID"
LIVERPOOL_ID_COLUMN = "genome_id"
LABEL_COLUMN = "label"

C_VALUES = [0.1, 1, 10, 100]
RANDOM_STATE = 42
CV_FOLDS = 5


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def validate_binary_labels(series, dataset_name):
    values = set(series.dropna().unique())

    if not values.issubset({0, 1, 0.0, 1.0, "0", "1"}):
        raise ValueError(
            f"{dataset_name}: labels must contain only 0 and 1. "
            f"Found: {values}"
        )

    if series.isnull().any():
        raise ValueError(f"{dataset_name}: missing labels found.")

    return series.astype(int)


def validate_features(df, features, dataset_name):
    missing = [feature for feature in features if feature not in df.columns]

    if missing:
        raise ValueError(
            f"{dataset_name}: required features missing: {missing}"
        )

    X = df[features].apply(pd.to_numeric, errors="raise")

    if X.isnull().any().any():
        raise ValueError(f"{dataset_name}: missing feature values found.")

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(f"{dataset_name}: non-finite feature values found.")

    values = set(np.unique(X.to_numpy()))

    if not values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"{dataset_name}: expected binary AMR features. "
            f"Found values: {values}"
        )

    return X.astype(float)


def main():
    start_time = time.perf_counter()

    print("=" * 72)
    print("NCBI -> LIVERPOOL EXTERNAL VALIDATION")
    print("CLASSICAL RBF SUPPORT VECTOR MACHINE")
    print("=" * 72)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 1. Load selected features from the quantum experiment
    # --------------------------------------------------------

    print("\n[1] Loading selected quantum features...")

    selected_features = load_json(FEATURE_FILE)

    if len(selected_features) != 12:
        raise ValueError(
            f"Expected 12 selected features, found {len(selected_features)}"
        )

    print("Selected features:", selected_features)

    # --------------------------------------------------------
    # 2. Load NCBI and Liverpool datasets
    # --------------------------------------------------------

    print("\n[2] Loading NCBI and Liverpool feature matrices...")

    ncbi_df = pd.read_csv(NCBI_FILE)
    liverpool_df = pd.read_csv(LIVERPOOL_FILE)

    print("NCBI matrix:", ncbi_df.shape)
    print("Liverpool matrix:", liverpool_df.shape)

    required_ncbi = [NCBI_ID_COLUMN, LABEL_COLUMN]
    required_liverpool = [LIVERPOOL_ID_COLUMN, LABEL_COLUMN]

    for column in required_ncbi:
        if column not in ncbi_df.columns:
            raise ValueError(f"NCBI missing required column: {column}")

    for column in required_liverpool:
        if column not in liverpool_df.columns:
            raise ValueError(f"Liverpool missing required column: {column}")

    # --------------------------------------------------------
    # 3. Validate IDs and labels
    # --------------------------------------------------------

    print("\n[3] Validating sample IDs and labels...")

    ncbi_df[NCBI_ID_COLUMN] = ncbi_df[NCBI_ID_COLUMN].astype(str)
    liverpool_df[LIVERPOOL_ID_COLUMN] = (
        liverpool_df[LIVERPOOL_ID_COLUMN].astype(str)
    )

    if ncbi_df[NCBI_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate NCBI assembly IDs found.")

    if liverpool_df[LIVERPOOL_ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate Liverpool genome IDs found.")

    y_train = validate_binary_labels(
        ncbi_df[LABEL_COLUMN], "NCBI"
    )

    y_external = validate_binary_labels(
        liverpool_df[LABEL_COLUMN], "Liverpool"
    )

    # --------------------------------------------------------
    # 4. Align exactly the same 12 features
    # --------------------------------------------------------

    print("\n[4] Aligning the same 12 features used by quantum SVM...")

    X_train = validate_features(
        ncbi_df, selected_features, "NCBI"
    )

    missing_liverpool = [
        feature for feature in selected_features
        if feature not in liverpool_df.columns
    ]

    for feature in missing_liverpool:
        liverpool_df[feature] = 0

    X_external = validate_features(
        liverpool_df, selected_features, "Liverpool"
    )

    print("NCBI feature shape:", X_train.shape)
    print("Liverpool feature shape:", X_external.shape)
    print("Liverpool zero-filled features:", missing_liverpool)

    # --------------------------------------------------------
    # 5. Check class distributions
    # --------------------------------------------------------

    print("\n[5] Dataset class distributions...")

    train_counts = y_train.value_counts().sort_index()
    external_counts = y_external.value_counts().sort_index()

    print(
        "NCBI - susceptible:",
        int(train_counts.get(0, 0)),
        "| resistant:",
        int(train_counts.get(1, 0)),
    )

    print(
        "Liverpool - susceptible:",
        int(external_counts.get(0, 0)),
        "| resistant:",
        int(external_counts.get(1, 0)),
    )

    if y_train.nunique() != 2:
        raise ValueError("NCBI training data must contain both classes.")

    if y_external.nunique() != 2:
        raise ValueError("Liverpool external data must contain both classes.")

    # --------------------------------------------------------
    # 6. Build classical RBF SVM pipeline
    # --------------------------------------------------------

    print("\n[6] Building classical RBF SVM...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "svc",
            SVC(
                kernel="rbf",
                class_weight="balanced",
            ),
        ),
    ])

    parameter_grid = {
        "svc__C": C_VALUES
    }

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # --------------------------------------------------------
    # 7. Tune using NCBI training data only
    # --------------------------------------------------------

    print("\n[7] Hyperparameter tuning using NCBI only...")
    print("C values:", C_VALUES)
    print("Cross-validation folds:", CV_FOLDS)

    tuning_start = time.perf_counter()

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="balanced_accuracy",
        cv=cv,
        refit=True,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(X_train, y_train)

    tuning_time = time.perf_counter() - tuning_start

    best_model = search.best_estimator_

    print("\nBest C:", search.best_params_["svc__C"])
    print(
        "NCBI CV balanced accuracy:",
        round(search.best_score_, 6),
    )

    # Save CV results
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(
        OUTPUT_DIR / "ncbi_cross_validation_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. External Liverpool prediction
    # --------------------------------------------------------

    print("\n[8] Evaluating on independent Liverpool dataset...")

    prediction_start = time.perf_counter()

    predictions = best_model.predict(X_external)
    decision_scores = best_model.decision_function(X_external)

    prediction_time = time.perf_counter() - prediction_start

    # --------------------------------------------------------
    # 9. Calculate metrics
    # --------------------------------------------------------

    print("\n[9] Calculating external validation metrics...")

    tn, fp, fn, tp = confusion_matrix(
        y_external,
        predictions,
        labels=[0, 1],
    ).ravel()

    accuracy = accuracy_score(y_external, predictions)
    balanced_accuracy = balanced_accuracy_score(
        y_external, predictions
    )
    precision = precision_score(
        y_external, predictions, zero_division=0
    )
    recall = recall_score(
        y_external, predictions, zero_division=0
    )
    f1 = f1_score(
        y_external, predictions, zero_division=0
    )
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    roc_auc = roc_auc_score(y_external, decision_scores)
    average_precision = average_precision_score(
        y_external, decision_scores
    )

    metrics = {
        "experiment": "NCBI to Liverpool external validation",
        "model": "Classical RBF SVM",
        "training_dataset": "Additional NCBI E. coli cohort",
        "external_dataset": "Liverpool E. coli cohort",
        "simulation": "Classical machine learning",
        "feature_count": len(selected_features),
        "feature_names": selected_features,
        "features_reused_from_quantum_experiment": True,
        "missing_liverpool_features_zero_filled": missing_liverpool,
        "kernel_type": "RBF",
        "scaling": "StandardScaler",
        "best_C": float(search.best_params_["svc__C"]),
        "gamma": "scale",
        "class_weight": "balanced",
        "cv_folds": CV_FOLDS,
        "cv_balanced_accuracy": float(search.best_score_),
        "training_samples": int(len(y_train)),
        "external_samples": int(len(y_external)),
        "training_susceptible": int((y_train == 0).sum()),
        "training_resistant": int((y_train == 1).sum()),
        "external_susceptible": int((y_external == 0).sum()),
        "external_resistant": int((y_external == 1).sum()),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "precision_resistant": float(precision),
        "recall_sensitivity": float(recall),
        "specificity": float(specificity),
        "f1_resistant": float(f1),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        },
        "tuning_time_seconds": float(tuning_time),
        "prediction_time_seconds": float(prediction_time),
        "total_experiment_time_seconds": float(
            time.perf_counter() - start_time
        ),
    }

    # --------------------------------------------------------
    # 10. Save predictions and experiment outputs
    # --------------------------------------------------------

    print("\n[10] Saving experiment outputs...")

    prediction_df = pd.DataFrame({
        "genome_id": liverpool_df[LIVERPOOL_ID_COLUMN],
        "actual_label": y_external,
        "predicted_label": predictions,
        "decision_score": decision_scores,
    })

    prediction_df["prediction_result"] = np.where(
        prediction_df["actual_label"]
        == prediction_df["predicted_label"],
        "Correct",
        "Incorrect",
    )

    prediction_df.to_csv(
        OUTPUT_DIR / "external_predictions.csv",
        index=False,
    )

    save_json(
        metrics,
        OUTPUT_DIR / "external_metrics.json",
    )

    confusion_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual_Susceptible", "Actual_Resistant"],
        columns=["Predicted_Susceptible", "Predicted_Resistant"],
    )

    confusion_df.to_csv(
        OUTPUT_DIR / "external_confusion_matrix.csv"
    )

    # Save fitted model
    import joblib

    joblib.dump(
        best_model,
        OUTPUT_DIR / "classical_rbf_svm.joblib",
    )

    metadata = {
        "experiment": "NCBI to Liverpool external validation",
        "training_input": str(NCBI_FILE),
        "external_input": str(LIVERPOOL_FILE),
        "selected_feature_file": str(FEATURE_FILE),
        "selected_features": selected_features,
        "feature_selection_cohort": "NCBI only (reused from quantum experiment)",
        "hyperparameter_tuning_cohort": "NCBI only",
        "external_test_cohort": "Liverpool",
        "external_test_used_for_tuning": False,
        "feature_selection_before_cv": True,
        "training_samples": int(len(y_train)),
        "external_samples": int(len(y_external)),
        "best_C": float(search.best_params_["svc__C"]),
        "cv_folds": CV_FOLDS,
        "random_state": RANDOM_STATE,
        "parameter_grid": C_VALUES,
        "model": "StandardScaler + RBF SVC",
    }

    save_json(
        metadata,
        OUTPUT_DIR / "external_experiment_metadata.json",
    )

    # --------------------------------------------------------
    # 11. Print final report
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("CLASSICAL RBF SVM - EXTERNAL VALIDATION RESULTS")
    print("=" * 72)

    for key in [
        "accuracy",
        "balanced_accuracy",
        "precision_resistant",
        "recall_sensitivity",
        "specificity",
        "f1_resistant",
        "roc_auc",
        "average_precision",
    ]:
        print(f"{key:25s}: {metrics[key]:.4f}")

    print("\nBest C:", metrics["best_C"])
    print("CV balanced accuracy:", metrics["cv_balanced_accuracy"])

    print("\nConfusion Matrix:")
    print("TN:", tn, "| FP:", fp)
    print("FN:", fn, "| TP:", tp)

    print("\nSaved outputs to:")
    print(OUTPUT_DIR)

    print("\nExperiment completed successfully.")


if __name__ == "__main__":
    main()


