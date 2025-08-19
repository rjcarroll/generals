#!/usr/bin/env python3
import csv
import shutil
from pathlib import Path

HEADER_ALIASES = {
    "imgpath": "img_path",
    "img_path": "img_path",
    "image_path": "img_path",
    "imagepath": "img_path",
}

def normalize_headers(fieldnames):
    norm = []
    for h in fieldnames:
        k = (h or "").strip().lower().replace(" ", "_")
        norm.append(k)
    return norm

def main():
    manifest = Path("samples/batch01/manifest.csv")  # adjust if needed
    outdir = Path("samples/batch01")
    outdir.mkdir(parents=True, exist_ok=True)

    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    with manifest.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(2048)
        f.seek(0)

        # Try sniffer first, then fall back to tab if present in sample, else comma
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
            delimiter = dialect.delimiter
        except Exception:
            delimiter = "\t" if "\t" in sample else ","

        reader = csv.DictReader(f, delimiter=delimiter)
        # Normalize fieldnames
        raw_fields = reader.fieldnames or []
        fields = normalize_headers(raw_fields)

        # Build a mapping from original header -> normalized
        header_map = {orig: norm for orig, norm in zip(raw_fields, fields)}

        # Figure out which column is img_path
        img_key = None
        for orig, norm in header_map.items():
            alias = HEADER_ALIASES.get(norm)
            if alias == "img_path" or norm == "img_path":
                img_key = orig
                break

        if not img_key:
            raise KeyError(
                f"Couldn't find an 'img_path' column. Headers seen: {raw_fields}"
            )

        rows = list(reader)

    copied = 0
    for row in rows:
        img_path = Path(row[img_key]).expanduser()
        if not img_path.exists():
            print(f"[WARN] missing {img_path}")
            continue
        dest = outdir / img_path.name
        shutil.copy2(img_path, dest)
        print(f"[OK] copied {img_path} → {dest}")
        copied += 1

    # copy manifest itself for provenance
    try:
        shutil.copy2(manifest, outdir / manifest.name)
    except Exception:
        pass

    print(f"[DONE] copied {copied}/{len(rows)} images into {outdir} (delimiter='{delimiter}')")

if __name__ == "__main__":
    main()

