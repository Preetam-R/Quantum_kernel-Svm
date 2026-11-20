
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

MATCHED_FILE = (
    ROOT / "data/processed/liverpool_ampicillin_matched.csv"
)

SUMMARY_FILE = (
    ROOT / "data/reports/amrfinder_processing_summary.csv"
)

RESULT_DIR = ROOT / "data/raw/amrfinder_results"

OUTPUT_FILE = (
    ROOT / "data/processed/amr_gene_feature_matrix.csv"
)

# 1. Load the original labelled cohort
cohort = pd.read_csv(MATCHED_FILE)

cohort["genome_id"] = cohort["genome_id"].astype(str)

if cohort["genome_id"].duplicated().any():
    raise ValueError("Duplicate genome IDs in matched dataset.")

if "label" not in cohort.columns:
    raise ValueError("The dataset must contain the label column.")

# 2. Load processing status
summary = pd.read_csv(SUMMARY_FILE)

summary["genome_id"] = summary["genome_id"].astype(str)

if summary["genome_id"].duplicated().any():
    raise ValueError("Duplicate IDs in processing summary.")

expected_ids = set(cohort["genome_id"])
processed_ids = set(summary["genome_id"])

if expected_ids != processed_ids:
    raise ValueError(
        "Processing summary does not match the 719-isolate cohort."
    )

allowed_status = ["SUCCESS", "SKIPPED_EXISTING"]

failed = summary[
    ~summary["status"].isin(allowed_status)
]

if not failed.empty:
    raise ValueError(
        f"{len(failed)} isolates have unsuccessful processing. "
        "Resolve these before constructing the final matrix."
    )

# 3. Read AMRFinderPlus results
feature_records = []
no_detection_ids = []
read_errors = []

for genome_id in cohort["genome_id"]:

    result_file = (
        RESULT_DIR / f"{genome_id}_amrfinder.tsv"
    )

    if not result_file.exists():
        read_errors.append(
            f"{genome_id}: result file missing"
        )
        continue

    try:
        data = pd.read_csv(
            result_file,
            sep="\t",
            dtype=str
        )

        if "Element symbol" not in data.columns:
            read_errors.append(
                f"{genome_id}: missing Element symbol column"
            )
            continue

        # Header-only TSV: valid file with no reported detections.
        if data.empty:
            no_detection_ids.append(genome_id)
            continue

        symbols = (
            data["Element symbol"]
            .dropna()
            .astype(str)
            .str.strip()
        )

        symbols = symbols[symbols != ""]

        if symbols.empty:
            no_detection_ids.append(genome_id)
            continue

        for symbol in symbols.unique():
            feature_records.append({
                "genome_id": genome_id,
                "feature": symbol,
                "value": 1
            })

    except Exception as error:
        read_errors.append(
            f"{genome_id}: {error}"
        )

# 4. Stop if any results could not be verified
if read_errors:
    print("Result validation errors:")
    for error in read_errors:
        print(error)

    raise RuntimeError(
        "Resolve result-file errors before building the matrix."
    )

# 5. Construct binary feature matrix
if feature_records:

    detections = pd.DataFrame(feature_records)

    matrix = detections.pivot_table(
        index="genome_id",
        columns="feature",
        values="value",
        aggfunc="max",
        fill_value=0
    )

    matrix.columns.name = None

else:
    matrix = pd.DataFrame(index=cohort["genome_id"])

# 6. Restore every isolate from the original cohort
matrix = matrix.reindex(
    cohort["genome_id"],
    fill_value=0
)

matrix = matrix.fillna(0).astype("int8")

matrix.insert(
    0,
    "genome_id",
    cohort["genome_id"].values
)

# 7. Attach the target label separately
matrix["label"] = cohort["label"].values

# 8. Validate
feature_columns = [
    col for col in matrix.columns
    if col not in ["genome_id", "label"]
]

if len(matrix) != len(cohort):
    raise ValueError("Incorrect number of isolates.")

if matrix["genome_id"].duplicated().any():
    raise ValueError("Duplicate IDs in feature matrix.")

if not matrix[feature_columns].isin([0, 1]).all().all():
    raise ValueError("Feature matrix contains non-binary values.")

if matrix["label"].isna().any():
    raise ValueError("Missing target labels.")

# 9. Save
matrix.to_csv(OUTPUT_FILE, index=False)

print("\nAMR FEATURE MATRIX CREATED")
print("=" * 50)

print("Total isolates:", len(matrix))
print("Total genomic features:", len(feature_columns))
print("Total feature values:", len(matrix) * len(feature_columns))

print("\nTarget distribution:")
print(matrix["label"].value_counts().sort_index())

print("\nIsolates without reported detections:",
      len(no_detection_ids))

print("\nNo-detection isolate IDs:")
print(no_detection_ids)

print("\nOutput:", OUTPUT_FILE)
print("\nValidation successful.")
