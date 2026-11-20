
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

GENOME_DIR = ROOT / "data/raw/FNA_GENOMES/genomes"

MATCHED_FILE = (
    ROOT / "data/processed/liverpool_ampicillin_matched.csv"
)

OUTPUT_FILE = (
    ROOT / "data/processed/kmer_6_feature_matrix.csv"
)

K = 6
TOTAL_CODES = 4 ** K


# ============================================================
# 2. DNA BASE ENCODING
# ============================================================

BASE_MAP = np.full(256, 255, dtype=np.uint8)

for base, value in {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3
}.items():

    BASE_MAP[ord(base)] = value
    BASE_MAP[ord(base.lower())] = value


# ============================================================
# 3. CONVERT NUMERICAL CODE TO DNA K-MER
# ============================================================

def code_to_kmer(code):

    bases = "ACGT"
    result = []

    for _ in range(K):

        result.append(bases[code & 3])
        code >>= 2

    return "".join(reversed(result))


# ============================================================
# 4. GENERATE REVERSE COMPLEMENT CODE
# ============================================================

def reverse_complement(code):

    result = 0

    for _ in range(K):

        base = code & 3

        result = (result << 2) | (3 - base)

        code >>= 2

    return result


# ============================================================
# 5. PROCESS ONE CONTIG AND COUNT CANONICAL K-MERS
# ============================================================

def process_contig(sequence, counts):

    if len(sequence) < K:
        return 0

    arr = np.frombuffer(
        sequence.encode("ascii"),
        dtype=np.uint8
    )

    bases = BASE_MAP[arr]

    windows = len(bases) - K + 1

    forward = np.zeros(
        windows,
        dtype=np.uint32
    )

    valid = np.ones(
        windows,
        dtype=bool
    )

    # Generate forward k-mer codes
    for j in range(K):

        current = bases[j:j + windows]

        valid &= current < 4

        safe = np.where(
            current < 4,
            current,
            0
        )

        forward = (forward << 2) | safe

    if not valid.any():
        return 0

    # Generate reverse-complement codes
    reverse = np.zeros(
        windows,
        dtype=np.uint32
    )

    for j in range(K):

        digit = (forward >> (2 * j)) & 3

        complement = 3 - digit

        reverse = (
            (reverse << 2) | complement
        )

    # Canonical code = minimum of forward and reverse
    canonical = np.minimum(
        forward,
        reverse
    )

    valid_codes = canonical[valid]

    # Count occurrences
    counts += np.bincount(
        valid_codes,
        minlength=TOTAL_CODES
    )

    return len(valid_codes)


# ============================================================
# 6. PROCESS FASTA FILE
# ============================================================

def process_fasta(filepath):

    counts = np.zeros(
        TOTAL_CODES,
        dtype=np.int64
    )

    sequence_parts = []

    valid_windows = 0

    def finish_contig():

        if not sequence_parts:
            return 0

        sequence = "".join(sequence_parts)

        return process_contig(
            sequence,
            counts
        )

    with open(filepath, "r") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            if line.startswith(">"):

                valid_windows += finish_contig()

                sequence_parts = []

            else:

                sequence_parts.append(line)

        # Process final contig
        valid_windows += finish_contig()

    return counts, valid_windows


# ============================================================
# 7. MAIN FUNCTION
# ============================================================

def main():

    print("Loading labelled cohort...")

    cohort = pd.read_csv(MATCHED_FILE)

    cohort["genome_id"] = (
        cohort["genome_id"].astype(str)
    )

    # Validate cohort
    if cohort["genome_id"].duplicated().any():

        raise ValueError(
            "Duplicate genome IDs found."
        )

    if cohort["label"].isna().any():

        raise ValueError(
            "Missing target labels found."
        )

    # --------------------------------------------------------
    # Generate canonical k-mer codes
    # --------------------------------------------------------

    canonical_codes = sorted({

        min(
            code,
            reverse_complement(code)
        )

        for code in range(TOTAL_CODES)

    })

    # Generate feature names
    feature_names = [

        "kmer_" + code_to_kmer(code)

        for code in canonical_codes

    ]

    print(
        "Canonical k-mer features:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # Process all genomes
    # --------------------------------------------------------

    rows = []

    failed = []

    print("\nProcessing genomes...")

    for index, genome_id in enumerate(
        cohort["genome_id"],
        start=1
    ):

        fasta_file = (
            GENOME_DIR / f"{genome_id}.fna"
        )

        if not fasta_file.exists():

            failed.append(
                f"{genome_id}: FASTA file missing"
            )

            continue

        try:

            counts, valid_windows = process_fasta(
                fasta_file
            )

            if valid_windows == 0:

                failed.append(
                    f"{genome_id}: No valid 6-mer windows"
                )

                continue

            # IMPORTANT FIX:
            # Select only canonical k-mer positions.
            # The original counts array has 4096 positions,
            # while canonical_codes contains 2080 positions.

            canonical_counts = counts[
                canonical_codes
            ]

            # Calculate normalized frequencies
            frequencies = (
                canonical_counts / valid_windows
            )

            rows.append(frequencies)

            if (
                index % 25 == 0
                or index == len(cohort)
            ):

                print(
                    f"Processed {index}/{len(cohort)} genomes"
                )

        except Exception as error:

            failed.append(
                f"{genome_id}: {error}"
            )

    # --------------------------------------------------------
    # Stop if any genome failed
    # --------------------------------------------------------

    if failed:

        print("\nProcessing errors:")

        for error in failed:
            print(error)

        raise RuntimeError(
            "Some genomes failed. Matrix was not saved."
        )

    if len(rows) != len(cohort):

        raise ValueError(
            "Unexpected number of processed genomes."
        )

    # --------------------------------------------------------
    # Construct feature matrix
    # --------------------------------------------------------

    matrix = pd.DataFrame(
        np.vstack(rows),
        columns=feature_names
    )

    matrix.insert(
        0,
        "genome_id",
        cohort["genome_id"].values
    )

    # Attach ampicillin resistance labels
    matrix["label"] = cohort["label"].values

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print("\nValidating feature matrix...")

    if len(matrix) != len(cohort):

        raise ValueError(
            "Incorrect number of isolates."
        )

    if matrix["genome_id"].duplicated().any():

        raise ValueError(
            "Duplicate genome IDs in matrix."
        )

    if matrix["label"].isna().any():

        raise ValueError(
            "Missing target labels."
        )

    if matrix.isna().any().any():

        raise ValueError(
            "Missing values detected."
        )

    feature_array = matrix[
        feature_names
    ].to_numpy()

    if not np.isfinite(feature_array).all():

        raise ValueError(
            "Invalid numerical values detected."
        )

    if (feature_array.sum(axis=1) <= 0).any():

        raise ValueError(
            "Some genomes have zero feature frequency."
        )

    # Each row contains normalized canonical k-mer frequencies
    row_sums = feature_array.sum(axis=1)

    if not np.allclose(row_sums, 1.0):

        raise ValueError(
            "K-mer frequencies do not sum to 1."
        )

    # --------------------------------------------------------
    # SAVE MATRIX
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    matrix.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # FINAL REPORT
    # --------------------------------------------------------

    print("\nK-MER FEATURE MATRIX CREATED")
    print("=" * 55)

    print(
        "Total isolates:",
        len(matrix)
    )

    print(
        "Total canonical k-mer features:",
        len(feature_names)
    )

    print(
        "Total columns:",
        matrix.shape[1]
    )

    print(
        "Matrix shape:",
        matrix.shape
    )

    print(
        "\nOutput file:",
        OUTPUT_FILE
    )

    print(
        "\nAmpicillin resistance label distribution:"
    )

    print(
        matrix["label"].value_counts().sort_index()
    )

    print(
        "\nMissing values:",
        matrix.isna().sum().sum()
    )

    print(
        "\nValidation successful."
    )


# ============================================================
# 8. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
