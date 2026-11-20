
from pathlib import Path
from collections import Counter
import re

FNA_FOLDER = Path("data/raw/FNA_GENOMES")

extensions = (".fna", ".fasta", ".fa", ".fna.gz", ".fasta.gz", ".fa.gz")

files = [
    p for p in FNA_FOLDER.rglob("*")
    if p.is_file() and p.name.lower().endswith(extensions)
]

print("\n========== FNA DATASET INSPECTION ==========")
print("Total genome files:", len(files))

if not files:
    print("No FNA/FASTA files found. Check FNA_FOLDER.")
    raise SystemExit(1)

sizes = [p.stat().st_size for p in files]

print("Total size (GB):", round(sum(sizes) / (1024**3), 2))
print("Largest file (MB):", round(max(sizes) / (1024**2), 2))

print("\nFirst 20 filenames:")
for p in files[:20]:
    print(p.relative_to(FNA_FOLDER))

print("\nAssembly IDs found in paths:")
pattern = re.compile(r"GCA_\d+\.\d+|GCF_\d+\.\d+", re.I)

found = 0
for p in files[:100]:
    matches = pattern.findall(str(p))
    if matches:
        print(p.name, "->", matches)
        found += 1

if found == 0:
    print("No GCA/GCF assembly accessions found in the first 100 paths.")
