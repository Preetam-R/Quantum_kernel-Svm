
from pathlib import Path
import subprocess
import pandas as pd
import time

# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "liverpool_ampicillin_matched.csv"
)

GENOME_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "FNA_GENOMES"
    / "genomes"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "amrfinder_results"
)

REPORT_DIR = PROJECT_ROOT / "data" / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = REPORT_DIR / "amrfinder_processing_summary.csv"

# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

if not INPUT_CSV.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

if "genome_id" not in df.columns:
    raise ValueError("The input CSV must contain genome_id.")

if df["genome_id"].duplicated().any():
    raise ValueError("Duplicate genome IDs found in input dataset.")

print("=" * 60)
print("AMRFINDERPLUS BATCH PROCESSING")
print("=" * 60)

print(f"Total labelled isolates: {len(df)}")
print(f"Genome directory: {GENOME_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# --------------------------------------------------
# PROCESS EACH GENOME
# --------------------------------------------------

results = []

start_time = time.time()

for index, row in df.iterrows():

    genome_id = str(row["genome_id"]).strip()

    genome_file = GENOME_DIR / f"{genome_id}.fna"

    output_file = OUTPUT_DIR / f"{genome_id}_amrfinder.tsv"

    print(f"\n[{index + 1}/{len(df)}] Processing {genome_id}")

    status = "FAILED"
    message = ""

    try:

        if not genome_file.exists():
            raise FileNotFoundError(
                f"Genome not found: {genome_file}"
            )

        # Skip already completed, non-empty results.
        if output_file.exists() and output_file.stat().st_size > 0:
            status = "SKIPPED_EXISTING"
            message = "Existing non-empty result file retained."

        else:

            genome_relative = genome_file.relative_to(
                PROJECT_ROOT
            ).as_posix()

            output_relative = output_file.relative_to(
                PROJECT_ROOT
            ).as_posix()

            command = [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{PROJECT_ROOT}:/data",
                "ncbi/amr",
                "amrfinder",
                "-n",
                f"/data/{genome_relative}",
                "-O",
                "Escherichia",
                "--threads",
                "4",
                "--output",
                f"/data/{output_relative}"
            ]

            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=1800
            )

            if completed.returncode != 0:
                raise RuntimeError(
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "AMRFinderPlus returned an error."
                )

            if not output_file.exists():
                raise RuntimeError(
                    "AMRFinderPlus completed without creating output."
                )

            status = "SUCCESS"
            message = "Analysis completed successfully."

            print("Result saved:", output_file.name)

    except Exception as error:

        status = "FAILED"
        message = str(error)

        print("ERROR:", message)

    results.append({
        "genome_id": genome_id,
        "genome_path": str(genome_file),
        "result_file": str(output_file),
        "status": status,
        "message": message
    })

    # Save progress after every isolate.
    pd.DataFrame(results).to_csv(
        SUMMARY_FILE,
        index=False
    )

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

summary = pd.DataFrame(results)

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("BATCH PROCESSING COMPLETED")
print("=" * 60)

print("\nProcessing statistics:")
print(summary["status"].value_counts().to_string())

print(f"\nTotal isolates: {len(summary)}")
print(f"Elapsed time: {elapsed / 60:.2f} minutes")

print(f"\nSummary saved to:\n{SUMMARY_FILE}")

print("\nFailed isolates:")

failed = summary[summary["status"] == "FAILED"]

if failed.empty:
    print("None")
else:
    print(failed[["genome_id", "message"]].to_string(index=False))
