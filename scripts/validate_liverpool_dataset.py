from pathlib import Path
import pandas as pd
from collections import Counter

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

METADATA_FILE = ROOT / "data" / "raw" / "meta_data_tidy.txt"
GENOME_DIR = ROOT / "data" / "raw" / "FNA_GENOMES"
PROCESSED_FILE = ROOT / "data" / "processed" / "liverpool_ampicillin_matched.csv"

REPORT_DIR = ROOT / "data" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD METADATA
# --------------------------------------------------

print("\nLoading metadata...")

metadata = pd.read_csv(
    METADATA_FILE,
    sep="\t",
    dtype=str
)

metadata["genome_id"] = metadata["genome_id"].str.strip()
metadata["disk_AMP"] = metadata["disk_AMP"].str.strip().str.upper()

print("Metadata rows:", len(metadata))
print("Unique genome IDs:", metadata["genome_id"].nunique())

# --------------------------------------------------
# CHECK DUPLICATE METADATA IDS
# --------------------------------------------------

duplicate_metadata = metadata[
    metadata["genome_id"].duplicated(keep=False)
].sort_values("genome_id")

print("\nDuplicate metadata IDs:", len(duplicate_metadata))

duplicate_metadata.to_csv(
    REPORT_DIR / "duplicate_metadata_ids.csv",
    index=False
)

# --------------------------------------------------
# SCAN GENOME FILES
# --------------------------------------------------

print("\nScanning FNA files...")

extensions = {".fna", ".fasta", ".fa"}

genome_files = [
    path for path in GENOME_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in extensions
]

print("Genome files:", len(genome_files))

file_map = {}

for path in genome_files:
    genome_id = path.stem.strip().upper()
    file_map.setdefault(genome_id, []).append(path)

duplicate_file_ids = {
    genome_id: paths
    for genome_id, paths in file_map.items()
    if len(paths) > 1
}

print("Duplicate FNA filename IDs:", len(duplicate_file_ids))

duplicate_file_rows = []

for genome_id, paths in duplicate_file_ids.items():
    for path in paths:
        duplicate_file_rows.append({
            "genome_id": genome_id,
            "file_path": str(path)
        })

pd.DataFrame(duplicate_file_rows).to_csv(
    REPORT_DIR / "duplicate_fna_ids.csv",
    index=False
)

# --------------------------------------------------
# COMPARE METADATA AND GENOME FILES
# --------------------------------------------------

metadata_ids = set(
    metadata["genome_id"].dropna().str.upper()
)

file_ids = set(file_map.keys())

metadata_without_genome = sorted(metadata_ids - file_ids)
genomes_without_metadata = sorted(file_ids - metadata_ids)

print("\n========== ID COMPARISON ==========")
print("Metadata IDs:", len(metadata_ids))
print("FNA IDs:", len(file_ids))
print("Metadata IDs without FNA:", len(metadata_without_genome))
print("FNA IDs without metadata:", len(genomes_without_metadata))

pd.DataFrame({
    "genome_id": metadata_without_genome
}).to_csv(
    REPORT_DIR / "metadata_without_genome.csv",
    index=False
)

pd.DataFrame({
    "genome_id": genomes_without_metadata
}).to_csv(
    REPORT_DIR / "genomes_without_metadata.csv",
    index=False
)

# --------------------------------------------------
# VALIDATE FASTA CONTENT
# --------------------------------------------------

print("\nValidating FASTA sequences...")
print("This may take a few minutes for the complete dataset.")

valid_dna = set("ACGTNRYKMSWBDHV")
fasta_results = []

for index, path in enumerate(genome_files, start=1):

    sequence_count = 0
    total_bases = 0
    invalid_characters = Counter()
    errors = []

    current_sequence_has_bases = False
    first_line_checked = False

    try:
        with path.open("r", encoding="ascii", errors="replace") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                if not line:
                    continue

                if not first_line_checked:
                    first_line_checked = True

                    if not line.startswith(">"):
                        errors.append("Missing FASTA header at first line")

                if line.startswith(">"):

                    if sequence_count > 0 and not current_sequence_has_bases:
                        errors.append("Empty FASTA sequence record")

                    sequence_count += 1
                    current_sequence_has_bases = False

                else:

                    if sequence_count == 0:
                        errors.append("Sequence found before first FASTA header")
                        continue

                    sequence = line.upper().replace(" ", "")

                    current_sequence_has_bases = True
                    total_bases += len(sequence)

                    for character in sequence:
                        if character not in valid_dna:
                            invalid_characters[character] += 1

            if sequence_count > 0 and not current_sequence_has_bases:
                errors.append("Last FASTA sequence record is empty")

    except Exception as error:
        errors.append(str(error))

    if sequence_count == 0:
        errors.append("No FASTA records found")

    fasta_results.append({
        "genome_id": path.stem.upper(),
        "file_path": str(path),
        "contig_count": sequence_count,
        "total_bases": total_bases,
        "invalid_character_count": sum(invalid_characters.values()),
        "invalid_characters": "".join(sorted(invalid_characters)),
        "is_valid": len(errors) == 0 and total_bases > 0,
        "errors": "; ".join(errors)
    })

    if index % 50 == 0:
        print(f"Validated {index}/{len(genome_files)} files")

fasta_report = pd.DataFrame(fasta_results)

fasta_report.to_csv(
    REPORT_DIR / "fasta_validation_report.csv",
    index=False
)

# --------------------------------------------------
# CHECK PROCESSED DATASET
# --------------------------------------------------

print("\nChecking processed dataset...")

processed = pd.read_csv(PROCESSED_FILE, dtype=str)

print("Processed rows:", len(processed))
print("Processed columns:", list(processed.columns))

if "disk_AMP" in processed.columns:

    processed["disk_AMP"] = (
        processed["disk_AMP"].str.strip().str.upper()
    )

    print("\nProcessed phenotype distribution:")
    print(processed["disk_AMP"].value_counts(dropna=False))

    unexpected = processed[
        ~processed["disk_AMP"].isin(["R", "S"])
    ]

    print("Unexpected or missing labels:", len(unexpected))

# --------------------------------------------------
# FINAL REPORT
# --------------------------------------------------

print("\n========== VALIDATION SUMMARY ==========")

print("Metadata records:", len(metadata))
print("Unique metadata IDs:", metadata["genome_id"].nunique())

print("Genome files:", len(genome_files))
print("Unique FNA IDs:", len(file_ids))

print("Duplicate metadata IDs:", len(duplicate_metadata))
print("Duplicate FNA IDs:", len(duplicate_file_ids))

print("Metadata IDs without genome:", len(metadata_without_genome))
print("Genome IDs without metadata:", len(genomes_without_metadata))

print("Valid FASTA files:", int(fasta_report["is_valid"].sum()))
print("Invalid FASTA files:", int((~fasta_report["is_valid"]).sum()))

print("\nSource disk_AMP distribution:")
print(metadata["disk_AMP"].value_counts(dropna=False))

print("\nReports saved in:")
print(REPORT_DIR)

print("\nDataset validation completed.")