
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULT_DIR = (
    PROJECT_ROOT / "data" / "raw" / "amrfinder_results"
)

SUMMARY_FILE = (
    PROJECT_ROOT / "data" / "reports"
    / "amrfinder_processing_summary.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "amr_results_combined.csv"
)

# Load processing summary
summary = pd.read_csv(SUMMARY_FILE)

successful = summary[
    summary["status"].isin(["SUCCESS", "SKIPPED_EXISTING"])
]

all_results = []
errors = []

# Read individual TSV files
for _, row in successful.iterrows():

    genome_id = str(row["genome_id"])

    result_file = RESULT_DIR / f"{genome_id}_amrfinder.tsv"

    if not result_file.exists():
        errors.append(f"{genome_id}: result file missing")
        continue

    try:
        data = pd.read_csv(
            result_file,
            sep="\t",
            dtype=str
        )

        if data.empty:
            errors.append(f"{genome_id}: empty result file")
            continue

        data.insert(0, "genome_id", genome_id)

        all_results.append(data)

    except Exception as error:
        errors.append(f"{genome_id}: {error}")

# Combine results
if all_results:
    combined = pd.concat(
        all_results,
        ignore_index=True
    )

    combined.to_csv(OUTPUT_FILE, index=False)

    print("Combined results saved:", OUTPUT_FILE)
    print("Total AMR detection rows:", len(combined))
    print("Unique isolates with detections:",
          combined["genome_id"].nunique())

    print("\nTop detected features:")
    print(
        combined["Element symbol"]
        .value_counts()
        .head(20)
        .to_string()
    )

else:
    print("No valid AMRFinderPlus result rows found.")

print("\nFiles with issues:")
if errors:
    for error in errors:
        print(error)
else:
    print("None")
