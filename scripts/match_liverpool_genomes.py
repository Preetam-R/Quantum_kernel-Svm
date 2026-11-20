
from pathlib import Path
import pandas as pd

# ------------------------------------------------
# PATHS
# ------------------------------------------------

BASE = Path(__file__).resolve().parents[1]

METADATA = BASE / "data/raw/meta_data_tidy.txt"
GENOME_DIR = BASE / "data/raw/FNA_GENOMES"

PROCESSED = BASE / "data/processed"
REPORTS = BASE / "data/reports"

PROCESSED.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------
# LOAD LIVERPOOL METADATA
# ------------------------------------------------

print("Loading Liverpool metadata...")

meta = pd.read_csv(METADATA, sep="\t", dtype=str)

meta.columns = meta.columns.str.strip()

print("Metadata records:", len(meta))
print("Metadata columns:", list(meta.columns))

required = ["genome_id", "genome_name", "disk_AMP"]

missing = [c for c in required if c not in meta.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

meta["genome_id"] = meta["genome_id"].str.strip().str.upper()
meta["disk_AMP"] = meta["disk_AMP"].str.strip().str.upper()

# ------------------------------------------------
# FIND GENOME FILES
# ------------------------------------------------

extensions = (".fna", ".fasta", ".fa")

genome_files = [
    p for p in GENOME_DIR.rglob("*")
    if p.is_file() and p.name.lower().endswith(extensions)
]

print("\nGenome files found:", len(genome_files))

# Map filename stem to full path
file_map = {}

for path in genome_files:
    genome_id = path.stem.strip().upper()

    if genome_id in file_map:
        print("WARNING: Duplicate genome filename:", genome_id)

    file_map.setdefault(genome_id, str(path.resolve()))

# ------------------------------------------------
# MATCH METADATA TO GENOMES
# ------------------------------------------------

meta["genome_path"] = meta["genome_id"].map(file_map)

meta["genome_found"] = meta["genome_path"].notna()

# ------------------------------------------------
# CHECK AMPICILLIN LABELS
# ------------------------------------------------

print("\nAmpicillin disk-diffusion labels:")

print(
    meta["disk_AMP"]
    .value_counts(dropna=False)
    .to_string()
)

# Keep only resistant and susceptible
binary = meta[
    meta["disk_AMP"].isin(["R", "S"])
    & meta["genome_found"]
].copy()

binary["label"] = binary["disk_AMP"].map({
    "S": 0,
    "R": 1
})

# ------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------

meta.to_csv(
    REPORTS / "liverpool_genome_match_status.csv",
    index=False
)

binary.to_csv(
    PROCESSED / "liverpool_ampicillin_matched.csv",
    index=False
)

unmatched = meta[~meta["genome_found"]]

unmatched.to_csv(
    REPORTS / "liverpool_unmatched_genomes.csv",
    index=False
)

# ------------------------------------------------
# SUMMARY
# ------------------------------------------------

print("\n========== MATCHING SUMMARY ==========")

print("Metadata records:", len(meta))
print("Genome files found:", len(genome_files))
print("Matched metadata records:", int(meta["genome_found"].sum()))
print("Unmatched metadata records:", len(unmatched))
print("Binary-labelled matched isolates:", len(binary))

print("\nFinal phenotype distribution:")
print(binary["disk_AMP"].value_counts())

print("\nOutput:")
print((PROCESSED / "liverpool_ampicillin_matched.csv").resolve())

print("\nMatching completed.")
