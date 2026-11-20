
from pathlib import Path
import gzip

FNA_FOLDER = Path("data/raw/FNA_GENOMES")

extensions = (
    ".fna", ".fasta", ".fa",
    ".fna.gz", ".fasta.gz", ".fa.gz"
)

files = sorted(
    p for p in FNA_FOLDER.rglob("*")
    if p.is_file() and p.name.lower().endswith(extensions)
)

print("Total files:", len(files))
print("\n========== FIRST 30 GENOME HEADERS ==========\n")

for path in files[:30]:
    try:
        opener = gzip.open if path.name.lower().endswith(".gz") else open

        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            headers = [
                line.strip()
                for line in f
                if line.startswith(">")
            ][:3]

        print(f"\nFILE: {path.name}")
        for header in headers:
            print(header)

    except Exception as error:
        print(f"ERROR: {path.name}: {error}")
