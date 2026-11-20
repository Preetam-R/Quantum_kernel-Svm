"""Match genome files against AMR records."""

from pathlib import Path
import pandas as pd
import re
import gzip

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

CSV_PATH = Path("data/raw/amr_records.csv")
FNA_FOLDER = Path("data/raw/FNA_GENOMES")
OUTPUT_FOLDER = Path("data/processed")
REPORT_FOLDER = Path("data/reports")

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
REPORT_FOLDER.mkdir(parents=True, exist_ok=True)

# Actual column names in your uploaded CSV
PREFIX = "pheno_geno_merged-"

COL_ANTIBIOTIC = PREFIX + "antibiotic_name"
COL_PHENOTYPE = PREFIX + "resistance_phenotype"
COL_BIOSAMPLE = PREFIX + "BioSample_ID"
COL_ASSEMBLY = PREFIX + "assembly_ID"
COL_SPECIES = PREFIX + "species"

VALID_EXTENSIONS = (
    ".fna", ".fasta", ".fa",
    ".fna.gz", ".fasta.gz", ".fa.gz"
)

ASSEMBLY_PATTERN = re.compile(
    r"GCA_\d+\.\d+|GCF_\d+\.\d+", re.IGNORECASE
)

BIOSAMPLE_PATTERN = re.compile(
    r"SAM[NED]\d+", re.IGNORECASE
)


# --------------------------------------------------
# 2. LOAD AND FILTER CSV
# --------------------------------------------------

print("Loading AMR CSV...")

df = pd.read_csv(CSV_PATH, low_memory=False)

required = [
    COL_ANTIBIOTIC,
    COL_PHENOTYPE,
    COL_BIOSAMPLE,
    COL_ASSEMBLY,
    COL_SPECIES
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

# Clean text fields
for col in required:
    df[col] = df[col].astype("string").str.strip()

# Select E. coli and ampicillin
amp = df[
    df[COL_SPECIES].str.lower().eq("escherichia coli")
    & df[COL_ANTIBIOTIC].str.lower().eq("ampicillin")
].copy()

# Normalize phenotype labels
amp["phenotype_clean"] = (
    amp[COL_PHENOTYPE].str.strip().str.lower()
)

amp["label"] = amp["phenotype_clean"].map({
    "susceptible": 0,
    "resistant": 1
})

# Preserve only binary-labelled records
binary = amp[amp["label"].notna()].copy()

print("All CSV records:", len(df))
print("Ampicillin E. coli records:", len(amp))
print("Binary-labelled records:", len(binary))
print("Excluded non-binary records:", len(amp) - len(binary))


# --------------------------------------------------
# 3. BUILD ASSEMBLY TARGET TABLE
# --------------------------------------------------

binary[COL_ASSEMBLY] = binary[COL_ASSEMBLY].str.upper()
binary[COL_BIOSAMPLE] = binary[COL_BIOSAMPLE].str.upper()

# Do not silently resolve conflicting labels
assembly_summary = (
    binary.dropna(subset=[COL_ASSEMBLY])
    .groupby(COL_ASSEMBLY)
    .agg(
        biosample_ids=(COL_BIOSAMPLE,
                       lambda x: ";".join(sorted(x.dropna().unique()))),
        phenotype_count=("label", "nunique"),
        labels=("label",
                lambda x: ";".join(map(str, sorted(x.unique())))),
        record_count=("label", "size")
    )
    .reset_index()
)

assembly_summary["label_conflict"] = (
    assembly_summary["phenotype_count"] > 1
)

assembly_summary.to_csv(
    REPORT_FOLDER / "ampicillin_assembly_targets.csv",
    index=False
)

# Exclude assemblies with contradictory labels from matching targets
targets = set(
    assembly_summary.loc[
        ~assembly_summary["label_conflict"], COL_ASSEMBLY
    ].dropna()
)

print("Unique assembly accessions:", len(assembly_summary))
print("Assemblies with conflicting labels:",
      int(assembly_summary["label_conflict"].sum()))
print("Assemblies eligible for matching:", len(targets))


# --------------------------------------------------
# 4. SCAN FNA FILE PATHS
# --------------------------------------------------

print("\nScanning FNA folder...")

genome_files = [
    p for p in FNA_FOLDER.rglob("*")
    if p.is_file()
    and p.name.lower().endswith(VALID_EXTENSIONS)
]

print("Genome files discovered:", len(genome_files))

matched = []
unmatched_files = []

for i, path in enumerate(genome_files, start=1):

    # Search full path, including parent folder names
    path_text = str(path).upper()

    assembly_ids = set(ASSEMBLY_PATTERN.findall(path_text))
    assembly_ids = {x.upper() for x in assembly_ids}

    biosample_ids = set(BIOSAMPLE_PATTERN.findall(path_text))
    biosample_ids = {x.upper() for x in biosample_ids}

    matching_assemblies = assembly_ids.intersection(targets)

    if matching_assemblies:
        for assembly_id in matching_assemblies:
            matched.append({
                "assembly_ID": assembly_id,
                "genome_path": str(path.resolve()),
                "match_method": "assembly_accession_in_path",
                "file_size_MB": round(
                    path.stat().st_size / (1024**2), 3
                )
            })
    else:
        unmatched_files.append({
            "genome_path": str(path.resolve()),
            "assembly_ids_in_path": ";".join(sorted(assembly_ids)),
            "biosample_ids_in_path": ";".join(sorted(biosample_ids))
        })

    if i % 500 == 0:
        print(f"Scanned {i}/{len(genome_files)} files")


# --------------------------------------------------
# 5. SAVE MATCH REPORT
# --------------------------------------------------

matched_df = pd.DataFrame(matched)

if matched_df.empty:
    matched_df = pd.DataFrame(columns=[
        "assembly_ID",
        "genome_path",
        "match_method",
        "file_size_MB"
    ])

matched_df.to_csv(
    REPORT_FOLDER / "matched_genomes.csv",
    index=False
)

pd.DataFrame(unmatched_files).to_csv(
    REPORT_FOLDER / "unmatched_genome_files.csv",
    index=False
)

# Assembly-level match status
matched_ids = set(matched_df["assembly_ID"].unique())

assembly_summary["genome_found"] = (
    assembly_summary[COL_ASSEMBLY].isin(matched_ids)
)

assembly_summary.to_csv(
    REPORT_FOLDER / "assembly_match_status.csv",
    index=False
)

# Save matched phenotype records with genome paths
final_df = binary.merge(
    matched_df[["assembly_ID", "genome_path"]].drop_duplicates(),
    left_on=COL_ASSEMBLY,
    right_on="assembly_ID",
    how="inner"
)

final_df.to_csv(
    OUTPUT_FOLDER / "ampicillin_matched_records.csv",
    index=False
)


# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

print("\n========== MATCHING SUMMARY ==========")
print("FNA files scanned:", len(genome_files))
print("Matched genome files:", len(matched_df))
print("Unique matched assemblies:", matched_df["assembly_ID"].nunique())
print("Target assemblies:", len(assembly_summary))
print("Unmatched target assemblies:",
      len(targets - matched_ids))
print("Final matched phenotype records:", len(final_df))

print("\nReports saved in:", REPORT_FOLDER.resolve())
print("Matched records saved in:",
      (OUTPUT_FOLDER / "ampicillin_matched_records.csv").resolve())

print("\nNext step: inspect the match reports before feature extraction.")
