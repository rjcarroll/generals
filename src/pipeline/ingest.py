#!/usr/bin/env python3
"""
Dictionary Ingest Script

Reads one or more dictionary sources (PDFs or images), normalizes them into
/data/raw/{volume}/{page}.{ext}, and writes a manifest.

- Supports PDF (via PyMuPDF or pdf2image) and common image types (png, jpg, tif).
- Computes checksums and basic metadata (size, DPI if available, bytes).
- Creates zero-padded page numbering per volume (configurable).
- Writes manifest as CSV (default) or JSONL.
- Idempotent: will skip existing outputs unless --force.

Examples
--------
# Ingest a single PDF as volume "Vol1" into ./data/raw
python ingest.py ./sources/DictVol1.pdf --volume Vol1 --out-root ./data/raw

# Ingest a directory of mixed files; derive volume name per file stem
python ingest.py ./sources/* --out-root ./data/raw

# Emit JSONL manifest as well
python ingest.py ./sources/DictVol1.pdf --volume Vol1 --manifest ./data/manifest.csv --jsonl ./data/manifest.jsonl

Requirements
------------
- Python 3.9+
- Pillow (PIL)
- Either PyMuPDF (fitz)  OR  pdf2image + poppler (for PDF rendering)

"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageOps
except ImportError as e:
    print("This script requires Pillow. Try: pip install pillow", file=sys.stderr)
    raise

# Optional PDF engines
_HAVE_FITZ = False
_HAVE_PDF2IMAGE = False
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    pass

if not _HAVE_FITZ:
    try:
        from pdf2image import convert_from_path  # type: ignore
        _HAVE_PDF2IMAGE = True
    except Exception:
        pass

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}
DEFAULT_OUTPUT_EXT = ".png"  # lossless-ish and widely supported


@dataclass
class ManifestRow:
    id: str
    volume: str
    page: int
    page_str: str
    src_path: str
    out_path: str
    ext: str
    width: int
    height: int
    dpi_x: Optional[int]
    dpi_y: Optional[int]
    bytes: int
    sha256: str
    created_at: str

    @staticmethod
    def header() -> List[str]:
        return list(asdict(ManifestRow(
            id="", volume="", page=0, page_str="", src_path="", out_path="", ext="",
            width=0, height=0, dpi_x=None, dpi_y=None, bytes=0, sha256="", created_at=""
        )).keys())


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_volume_name(path: Path) -> str:
    # Use parent dir name if it looks like a volume (e.g., Vol1, Volume_II),
    # else use file stem.
    vol = path.stem
    parent = path.parent.name
    if re.search(r"vol(ume)?", parent, re.IGNORECASE):
        return parent
    return vol


def zero_pad(n: int, width: int) -> str:
    return f"{n:0{width}d}"


def save_image(img: Image.Image, out_path: Path, dpi: Optional[Tuple[int, int]] = None, quality: int = 95):
    params = {}
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        params.update({"quality": quality, "subsampling": 0, "optimize": True})
    if dpi is not None:
        params["dpi"] = dpi
    img.save(out_path, **params)


def render_pdf_to_images_with_fitz(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    if not _HAVE_FITZ:
        raise RuntimeError("PyMuPDF (fitz) not available")
    doc = fitz.open(pdf_path)
    images: List[Image.Image] = []
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
    finally:
        doc.close()
    return images


def render_pdf_to_images_with_pdf2image(pdf_path: Path, dpi: int = 300) -> List[Image.Image]:
    if not _HAVE_PDF2IMAGE:
        raise RuntimeError("pdf2image not available (and/or poppler is missing)")
    imgs = convert_from_path(str(pdf_path), dpi=dpi)
    return imgs


def iter_input_paths(inputs: List[Path]) -> Iterable[Path]:
    for p in inputs:
        if p.is_dir():
            for ext in list(SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS):
                yield from sorted(p.rglob(f"*{ext}"))
        else:
            yield p


def process_pdf(src: Path, out_root: Path, volume: str, page_start: int, pad: int, out_ext: str, dpi: int, force: bool) -> List[ManifestRow]:
    # Render to PIL images
    if _HAVE_FITZ:
        pages = render_pdf_to_images_with_fitz(src, dpi=dpi)
    elif _HAVE_PDF2IMAGE:
        pages = render_pdf_to_images_with_pdf2image(src, dpi=dpi)
    else:
        raise RuntimeError("No PDF engine available. Install PyMuPDF (fitz) or pdf2image + poppler.")

    rows: List[ManifestRow] = []
    volume_dir = out_root / volume
    ensure_dir(volume_dir)

    for i, img in enumerate(pages, start=page_start):
        page_str = zero_pad(i, pad)
        out_path = volume_dir / f"{page_str}{out_ext}"
        if out_path.exists() and not force:
            pass
        else:
            # Try to preserve DPI if available (some renderers return no DPI; we set requested dpi)
            dpi_tuple = (dpi, dpi)
            save_image(img, out_path, dpi=dpi_tuple)
        stat = out_path.stat()
        # PIL stores DPI in info only on load; best effort here
        dpi_x = dpi_y = dpi
        width, height = img.size
        row = ManifestRow(
            id=f"{volume}:{page_str}",
            volume=volume,
            page=i,
            page_str=page_str,
            src_path=str(src.resolve()),
            out_path=str(out_path.resolve()),
            ext=out_ext,
            width=width,
            height=height,
            dpi_x=dpi_x,
            dpi_y=dpi_y,
            bytes=stat.st_size,
            sha256=sha256_file(out_path),
            created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
        )
        rows.append(row)
    return rows


def process_image(src: Path, out_root: Path, volume: str, page_index: int, pad: int, out_ext: str, force: bool) -> ManifestRow:
    # Load and normalize image
    img = Image.open(src)
    # Convert paletted/CMYK to RGB to avoid downstream surprises
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    # Read DPI if present
    dpi_x = dpi_y = None
    if isinstance(img.info.get("dpi"), tuple):
        dpi_x, dpi_y = img.info["dpi"]
    page_str = zero_pad(page_index, pad)
    volume_dir = out_root / volume
    ensure_dir(volume_dir)
    out_path = volume_dir / f"{page_str}{out_ext}"
    if not out_path.exists() or force:
        # Preserve DPI if known
        save_image(img, out_path, dpi=(dpi_x, dpi_y) if (dpi_x and dpi_y) else None)
    stat = out_path.stat()
    width, height = img.size
    row = ManifestRow(
        id=f"{volume}:{page_str}",
        volume=volume,
        page=page_index,
        page_str=page_str,
        src_path=str(src.resolve()),
        out_path=str(out_path.resolve()),
        ext=out_ext,
        width=width,
        height=height,
        dpi_x=int(dpi_x) if dpi_x else None,
        dpi_y=int(dpi_y) if dpi_y else None,
        bytes=stat.st_size,
        sha256=sha256_file(out_path),
        created_at=datetime.utcnow().isoformat(timespec="seconds") + "Z",
    )
    return row


def write_csv_manifest(rows: List[ManifestRow], path: Path):
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(ManifestRow.header())
        for r in rows:
            w.writerow([r.id, r.volume, r.page, r.page_str, r.src_path, r.out_path, r.ext,
                        r.width, r.height, r.dpi_x, r.dpi_y, r.bytes, r.sha256, r.created_at])


def write_jsonl_manifest(rows: List[ManifestRow], path: Path):
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Ingest dictionaries: normalize pages and write manifest")
    p.add_argument("inputs", nargs="+", help="Input files or directories (PDFs or images)")
    p.add_argument("--out-root", default="./data/raw", help="Root output directory (default: ./data/raw)")
    p.add_argument("--volume", default=None, help="Volume name override (else inferred per input)")
    p.add_argument("--page-start", type=int, default=1, help="Starting page number (default: 1)")
    p.add_argument("--pad", type=int, default=4, help="Zero-padding for page numbers (default: 4 → 0001)")
    p.add_argument("--out-ext", default=DEFAULT_OUTPUT_EXT, choices=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"], help="Output image extension")
    p.add_argument("--dpi", type=int, default=300, help="Target DPI for PDF rendering (default: 300)")
    p.add_argument("--manifest", default="./data/manifest.csv", help="CSV manifest path (append mode)")
    p.add_argument("--jsonl", default=None, help="Optional JSONL manifest path (append mode)")
    p.add_argument("--force", action="store_true", help="Overwrite existing outputs")
    args = p.parse_args(argv)

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    inputs = [Path(x) for x in args.inputs]

    all_rows: List[ManifestRow] = []
    for src in iter_input_paths(inputs):
        ext = src.suffix.lower()
        vol = args.volume or infer_volume_name(src)
        if ext in SUPPORTED_PDF_EXTS:
            rows = process_pdf(src, out_root, vol, args.page_start, args.pad, args.out_ext, args.dpi, args.force)
            all_rows.extend(rows)
        elif ext in SUPPORTED_IMAGE_EXTS:
            # For per-file ingestion, we treat each image as next page; we maintain a running index per volume
            # Build a per-volume counter based on existing outputs
            volume_dir = out_root / vol
            ensure_dir(volume_dir)
            existing = sorted(volume_dir.glob(f"*{args.out_ext}"))
            next_index = args.page_start + len(existing)
            row = process_image(src, out_root, vol, next_index, args.pad, args.out_ext, args.force)
            all_rows.append(row)
        else:
            print(f"[skip] Unsupported file type: {src}", file=sys.stderr)

    if not all_rows:
        print("No files processed. Nothing to write to manifest.", file=sys.stderr)
        return 1

    write_csv_manifest(all_rows, Path(args.manifest))
    if args.jsonl:
        write_jsonl_manifest(all_rows, Path(args.jsonl))

    # Summary
    by_vol = {}
    for r in all_rows:
        by_vol.setdefault(r.volume, 0)
        by_vol[r.volume] += 1
    print("Ingest complete.")
    for vol, n in sorted(by_vol.items()):
        print(f"  {vol}: {n} pages")
    print(f"Manifest appended: {args.manifest}")
    if args.jsonl:
        print(f"JSONL appended: {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
