#!/usr/bin/env python3
"""
ocr.py — Two‑column aware OCR with per‑volume page ranges and parallelism (NO fallback)

What it does
------------
- Two strategies; you pick one:
  * --split-columns none  → Tesseract PSM 4 handles columns
  * --split-columns auto  → detect central rule, OCR left & right, merge L→R
- Optional header crop (percent of page height)
- Per‑volume start/end filtering via --ranges-config (YAML/JSON) or flags
- Writes TXT (merged L→R in split mode), TSV/HOCR (per column in split mode, else per page)
- Skips existing outputs
- Parallel page OCR via --workers

YAML config example
-------------------
ranges:
  dict1: { start: 27, end: 639 }
  dict2: { start: 21, end: 599 }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
import pytesseract
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # YAML optional; JSON still works


# ---------------------------- Dataclasses ---------------------------- #

@dataclass
class PageReport:
    src: str
    out_txt: Optional[str]
    out_tsv: List[str]
    out_hocr: List[str]
    n_chars: int
    n_lines: int
    ms: int
    split: bool
    header_crop_px: int
    split_x: Optional[int]
    mode: str  # 'tess' or 'split'


@dataclass
class VolumeReport:
    volume: str
    in_root: str
    out_root: str
    lang: str
    psm: int
    formats: List[str]
    start_page: Optional[int]
    end_page: Optional[int]
    n_pages: int
    pages: List[PageReport]


# ---------------------------- Utilities ----------------------------- #

def parse_page_index(stem: str) -> Optional[int]:
    """Return trailing integer from filename stem (e.g., 0007 from '0007', 12 from 'page_0012')."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None


def list_pages(in_root: Path, exts: Iterable[str]) -> List[Path]:
    pages: List[Path] = []
    for e in exts:
        pages.extend(sorted(in_root.glob(f"*{e}")))
    pages.sort(key=lambda p: (parse_page_index(p.stem) is None,
                              parse_page_index(p.stem) or p.stem))
    return pages


def crop_header_nd(img: np.ndarray, header_height_pct: float) -> Tuple[np.ndarray, int]:
    if header_height_pct <= 0:
        return img, 0
    h = img.shape[0]
    cut = min(h, max(0, int(round(h * (header_height_pct / 100.0)))))
    return img[cut:, :], cut


def binarize(gray: np.ndarray) -> np.ndarray:
    if len(gray.shape) != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th


def find_vertical_split(img_gray: np.ndarray) -> Optional[int]:
    """Detect central vertical rule by projecting black pixels per column."""
    bin_img = binarize(img_gray)
    h, w = bin_img.shape[:2]
    x0, x1 = int(w * 0.25), int(w * 0.75)
    band = bin_img[:, x0:x1]
    black = (255 - band) // 255
    col_scores = black.sum(axis=0)
    x_rel = int(col_scores.argmax())
    score = int(col_scores[x_rel])
    if score < 0.7 * h:
        return None
    return x0 + x_rel


def split_columns(img: np.ndarray, split_x: int, gap: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    xL0, xL1 = 0, max(0, split_x - gap)
    xR0, xR1 = min(w, split_x + gap), w
    left = img[:, xL0:xL1]
    right = img[:, xR0:xR1]
    return left, right


def pil_from_nd(img: np.ndarray) -> Image.Image:
    if len(img.shape) == 2:
        return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def tess_config(psm: int, oem: int) -> str:
    # Keep spaces, and lean slightly on language model and document dictionary
    return (
        f"--psm {psm} --oem {oem} "
        "-c preserve_interword_spaces=1 "
        "-c doc_dict_enable=1 "
        "-c language_model_ngram_on=1"
    )


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def load_ranges_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"ranges config not found: {cfg_path}")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed but a YAML config was provided")
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("ranges config must be a dict with top-level key 'ranges'")
    ranges = data.get("ranges", {})
    if not isinstance(ranges, dict):
        raise ValueError("ranges config must contain a 'ranges' mapping")
    norm = {}
    for vol, se in ranges.items():
        if isinstance(se, dict):
            norm[vol] = {"start": se.get("start"), "end": se.get("end")}
    return norm


# ----------------------------- Core OCR ------------------------------ #

def ocr_any(img: Image.Image, lang: str, config: str, kind: str) -> str | bytes:
    if kind == "txt":
        return pytesseract.image_to_string(img, lang=lang, config=config)
    if kind == "tsv":
        return pytesseract.image_to_data(img, lang=lang, config=config, output_type=pytesseract.Output.STRING)
    if kind == "hocr":
        return pytesseract.image_to_pdf_or_hocr(img, lang=lang, config=config, extension="hocr")
    raise ValueError(f"Unknown kind: {kind}")


def ocr_page_tess(
    img_nohdr: np.ndarray,
    img_stem: str,
    out_dir: Path,
    lang: str,
    psm: int,
    oem: int,
    formats: List[str],
) -> Tuple[str, List[str], List[str], bool, Optional[int]]:
    """Single-pass Tesseract (PSM) page OCR. Returns (text, tsv_paths, hocr_paths, split, split_x)."""
    out_tsv_paths: List[str] = []
    out_hocr_paths: List[str] = []
    cfg_page = tess_config(psm=psm, oem=oem)
    P = pil_from_nd(img_nohdr)
    text_all = ""
    if "txt" in formats:
        text_all = str(ocr_any(P, lang, cfg_page, "txt"))
        write_text(out_dir / f"{img_stem}.txt", text_all)
    if "tsv" in formats:
        tsv = str(ocr_any(P, lang, cfg_page, "tsv"))
        (out_dir / f"{img_stem}.tsv").write_text(tsv, encoding="utf-8")
        out_tsv_paths.append(str((out_dir / f"{img_stem}.tsv").resolve()))
    if "hocr" in formats:
        hocr = ocr_any(P, lang, cfg_page, "hocr")
        (out_dir / f"{img_stem}.hocr").write_bytes(hocr)  # type: ignore[arg-type]
        out_hocr_paths.append(str((out_dir / f"{img_stem}.hocr").resolve()))
    return text_all, out_tsv_paths, out_hocr_paths, False, None

def extract_center_header(img_nohdr: np.ndarray, split_x: int, oem: int, lang: str) -> str | None:
    import re
    h, w = img_nohdr.shape[:2]
    band_w = max(40, int(0.10 * w))     # ~10% width centered on split
    x0 = max(0, split_x - band_w // 2)
    x1 = min(w, split_x + band_w // 2)
    y0, y1 = 0, int(0.50 * h)           # upper half; adjust if needed

    roi = img_nohdr[y0:y1, x0:x1]
    P = pil_from_nd(roi)
    cfg = tess_config(psm=6, oem=oem)   # simple uniform block
    raw = str(ocr_any(P, lang, cfg, "txt")).strip()
    if not raw:
        return None
    # tidy lines and prefer short all-caps
    lines = [re.sub(r"\s+", " ", ln).strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return None
    # score: more uppercase is better; shorter is better
    lines.sort(key=lambda s: (-(sum(ch.isupper() for ch in s)), len(s)))
    cand = re.sub(r"[^A-Za-zÉÈÊÀÂÎÔÛÇa-z’' .-]", "", lines[0]).strip()
    # accept “B”, “C”, “A.”, or short all-caps word(s)
    import re as _re
    if _re.fullmatch(r"[A-ZÉÈÊÀÂÎÔÛÇ]\.?", cand):
        return cand.rstrip(".")
    if _re.fullmatch(r"[A-ZÉÈÊÀÂÎÔÛÇ]{1,10}(?: [A-ZÉÈÊÀÂÎÔÛÇ]{1,10}){0,2}", cand):
        return cand
    return None

def _ocr_tsv(img_pil, lang: str, cfg: str):
    import pytesseract
    return pytesseract.image_to_data(img_pil, lang=lang, config=cfg, output_type=pytesseract.Output.DATAFRAME)

def _text_by_y(df):
    # Build line-level strings sorted by (top, left)
    if df is None or df.empty: return []
    wdf = df[(df.conf.astype(float) >= 0) & (df.text.notna()) & (df.text.str.strip() != "")]
    if wdf.empty: return []
    # Group by (block, par, line) to get line boxes and text
    lines = []
    for keys, g in wdf.groupby(["block_num","par_num","line_num"], sort=True):
        top = int(g.top.min()); left = int(g.left.min())
        txt = " ".join(t for t in g.sort_values(["word_num"]).text if isinstance(t,str))
        if txt.strip():
            lines.append((top, left, txt))
    lines.sort(key=lambda t: (t[0], t[1]))
    return lines

def _split_lines_by_y(lines, y_cut, pad=8):
    # Return (above_text, below_text) with a small pad to absorb minor misalignments
    if not lines: return ("","")
    above = [t for (y,x,t) in lines if y <= y_cut - pad]
    below = [t for (y,x,t) in lines if y >= y_cut + pad]
    return ("\n".join(above).strip(), "\n".join(below).strip())

def extract_center_header_and_y(img_nohdr: np.ndarray, split_x: int, oem: int, lang: str):
    """
    Returns (header_text or None, header_y or None). header_y is absolute Y in img_nohdr.
    """
    import re
    h, w = img_nohdr.shape[:2]
    band_w = max(40, int(0.10 * w))
    x0 = max(0, split_x - band_w // 2); x1 = min(w, split_x + band_w // 2)
    y0, y1 = 0, int(0.55 * h)  # search upper half
    roi = img_nohdr[y0:y1, x0:x1]
    P = pil_from_nd(roi)
    cfg = tess_config(psm=6, oem=oem)
    try:
        df = _ocr_tsv(P, lang, cfg)
    except Exception:
        return (None, None)
    if df is None or df.empty: return (None, None)
    wdf = df[(df.conf.astype(float) >= 0) & (df.text.notna()) & (df.text.str.strip() != "")]
    if wdf.empty: return (None, None)
    # Prefer short all-caps lines
    cand = None
    best_score = (-1, 0)  # (num_upper, -len)
    for (_,_,lnum), g in wdf.groupby(["block_num","par_num","line_num"], sort=True):
        text = " ".join(t for t in g.sort_values(["word_num"]).text if isinstance(t,str)).strip()
        clean = re.sub(r"[^A-Za-zÉÈÊÀÂÎÔÛÇa-z’' .-]", "", text).strip()
        if not clean: continue
        uppers = sum(ch.isupper() for ch in clean)
        score = (uppers, -len(clean))
        if score > best_score:
            best_score = score
            cand = (clean, int(g.top.min()))
    if not cand: return (None, None)
    header_txt, header_top_in_roi = cand
    # Accept “B”, “C”, “A.”, or short all-caps token(s)
    if re.fullmatch(r"[A-ZÉÈÊÀÂÎÔÛÇ]\.?", header_txt):
        header_txt = header_txt.rstrip(".")
    elif not re.fullmatch(r"[A-ZÉÈÊÀÂÎÔÛÇ]{1,10}(?: [A-ZÉÈÊÀÂÎÔÛÇ]{1,10}){0,2}", header_txt):
        return (None, None)
    header_y_abs = y0 + header_top_in_roi
    return (header_txt, header_y_abs)

def ocr_page_split(
    img_nohdr: np.ndarray,
    img_stem: str,
    out_dir: Path,
    lang: str,
    oem: int,
    formats: List[str],
) -> Tuple[str, List[str], List[str], bool, Optional[int]]:
    """Manual split at detected rule; OCR L and R columns; merge L→R."""
    split_x = find_vertical_split(img_nohdr)
    out_tsv_paths: List[str] = []
    out_hocr_paths: List[str] = []
    text_all = ""
    did_split = False
    if split_x is not None:
        L_nd, R_nd = split_columns(img_nohdr, split_x, gap=10)
        cfg_col = tess_config(psm=6, oem=oem)
        L = pil_from_nd(L_nd)
        R = pil_from_nd(R_nd)

        # 1) Try to find centered header + its Y
        header_txt, header_y = extract_center_header_and_y(img_nohdr, split_x, oem=oem, lang=lang)

        # 2) OCR columns to TSV so we can split at Y
        dfL = _ocr_tsv(L, lang, cfg_col) if "txt" in formats else None
        dfR = _ocr_tsv(R, lang, cfg_col) if "txt" in formats else None
        linesL = _text_by_y(dfL); linesR = _text_by_y(dfR)

        if header_y is not None and linesL and linesR:
            # Convert absolute header_y to column-local Y (same y in both since we didn’t crop vertically per column)
            # (img_nohdr and both L/R slices share the same top origin)
            aboveL, belowL = _split_lines_by_y([(y,x,t) for (y,x,t) in linesL], header_y)
            aboveR, belowR = _split_lines_by_y([(y,x,t) for (y,x,t) in linesR], header_y)
            # Assemble: A-left → A-right → [Header] → B-left → B-right
            left_text_above  = aboveL
            right_text_above = aboveR
            left_text_below  = belowL
            right_text_below = belowR
            parts = []
            if left_text_above:  parts.append(left_text_above)
            if right_text_above: parts.append(right_text_above)
            if header_txt:       parts.append(header_txt)
            if left_text_below:  parts.append(left_text_below)
            if right_text_below: parts.append(right_text_below)
            text_all = "\n\n".join(p for p in parts if p)
        else:
            # Fallback: simple L→R merge
            left_text  = "\n".join(t for (_,_,t) in linesL) if linesL else ""
            right_text = "\n".join(t for (_,_,t) in linesR) if linesR else ""
            text_all   = (left_text or "") + "\n\n" + (right_text or "")

        if "txt" in formats:
            write_text(out_dir / f"{img_stem}.txt", text_all)
        if "tsv" in formats:
            tsvL = str(ocr_any(L, lang, cfg_col, "tsv"))
            (out_dir / f"{img_stem}_L.tsv").write_text(tsvL, encoding="utf-8")
            out_tsv_paths.append(str((out_dir / f"{img_stem}_L.tsv").resolve()))
            tsvR = str(ocr_any(R, lang, cfg_col, "tsv"))
            (out_dir / f"{img_stem}_R.tsv").write_text(tsvR, encoding="utf-8")
            out_tsv_paths.append(str((out_dir / f"{img_stem}_R.tsv").resolve()))
        if "hocr" in formats:
            hL = ocr_any(L, lang, cfg_col, "hocr")
            (out_dir / f"{img_stem}_L.hocr").write_bytes(hL)  # type: ignore[arg-type]
            out_hocr_paths.append(str((out_dir / f"{img_stem}_L.hocr").resolve()))
            hR = ocr_any(R, lang, cfg_col, "hocr")
            (out_dir / f"{img_stem}_R.hocr").write_bytes(hR)  # type: ignore[arg-type]
            out_hocr_paths.append(str((out_dir / f"{img_stem}_R.hocr").resolve()))
        did_split = True
    return text_all, out_tsv_paths, out_hocr_paths, did_split, split_x


def ocr_page(
    img_path: Path,
    out_dir: Path,
    lang: str,
    psm: int,
    oem: int,
    formats: List[str],
    split_columns_mode: str,
    header_height_pct: float,
) -> PageReport:
    t0 = time.time()

    src = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError(f"Failed to read image: {img_path}")

    img_nohdr, cut_px = crop_header_nd(src, header_height_pct)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt_path = out_dir / f"{img_path.stem}.txt"

    # run in chosen mode (no fallback)
    if split_columns_mode == "auto":
        text_all, tsv_paths, hocr_paths, did_split, split_x = ocr_page_split(
        img_nohdr, img_path.stem, out_dir, lang, oem, formats
        )
        if not did_split or not (text_all and text_all.strip()):
            # Fallback: no split detected or empty text → run single-pass page OCR
            text_all_fallback, tsv2, hocr2, _, _ = ocr_page_tess(
                img_nohdr, img_path.stem, out_dir, lang, psm, oem, formats
            )
            # Prefer non-empty result
            if text_all_fallback and len(text_all_fallback.strip()) > 0:
                text_all = text_all_fallback
                # merge artifact lists so report.json has something
                tsv_paths = tsv_paths or tsv2
                hocr_paths = hocr_paths or hocr2
                used_mode = "tess"
            else:
                used_mode = "split"
        else:
            used_mode = "split"
    else:
        text_all, tsv_paths, hocr_paths, did_split, split_x = ocr_page_tess(
            img_nohdr, img_path.stem, out_dir, lang, psm, oem, formats
        )
        used_mode = "tess"

    lines = [ln for ln in (text_all or "").splitlines() if ln.strip()]
    n_chars = len(text_all or "")
    ms = int((time.time() - t0) * 1000)

    return PageReport(
        src=str(img_path),
        out_txt=str(out_txt_path.resolve()) if out_txt_path.exists() else None,
        out_tsv=tsv_paths,
        out_hocr=hocr_paths,
        n_chars=n_chars,
        n_lines=len(lines),
        ms=ms,
        split=did_split,
        header_crop_px=cut_px,
        split_x=split_x,
        mode=used_mode,
    )


# ------------------------------ CLI/Main ----------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="OCR a preprocessed volume with Tesseract (two‑column modes, ranges, parallel; NO fallback)")
    ap.add_argument("--in-root", required=True, help="Input volume directory (preprocessed pages)")
    ap.add_argument("--tessdata-dir", default=None,
                help="Path to tessdata (traineddata, user-words, user-patterns)")
    ap.add_argument("--out-root", required=True, help="Output volume directory for OCR artifacts")
    ap.add_argument("--lang", default="fra", help="Tesseract language code (default: fra)")
    ap.add_argument("--psm", type=int, default=4, help="Tesseract page segmentation mode (default: 4: columns)")
    ap.add_argument("--oem", type=int, default=1, help="OCR Engine Mode (0=Legacy,1=LSTM,2=Both,3=Default)")
    ap.add_argument("--formats", default="txt,tsv,hocr", help="Comma-separated outputs: txt,tsv,hocr")
    ap.add_argument("--tesseract-cmd", default=None, help="Path to tesseract binary if not on PATH")
    ap.add_argument("--skip-existing", action="store_true", help="Skip pages where requested outputs already exist")
    ap.add_argument("--split-columns", choices=["none", "auto"], default="none", help="Primary column strategy: none=tesseract PSM, auto=manual split")
    ap.add_argument("--header-height-pct", type=float, default=7.0, help="Crop this % from the top to remove headers (0 to disable)")
    # per-volume ranges
    ap.add_argument("--ranges-config", default=None, help="YAML/JSON file mapping volume→{start,end}")
    # optional ad‑hoc overrides
    ap.add_argument("--start-page", type=int, default=None, help="Only process pages with numeric index >= this (inclusive)")
    ap.add_argument("--end-page", type=int, default=None, help="Only process pages with numeric index <= this (inclusive)")
    # parallelism
    ap.add_argument("--workers", type=int, default=1, help="Number of parallel OCR workers (processes). 1 = no parallelism")
    args = ap.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    formats = [f.strip().lower() for f in args.formats.split(',') if f.strip()]

    pages = list_pages(in_root, [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"])
    if not pages:
        print(f"[ocr] No page images found in {in_root}")
        return 1

    volume = in_root.name

    # Resolve start/end from config, then CLI overrides
    start_page = end_page = None
    ranges = load_ranges_config(args.ranges_config)
    if volume in ranges:
        se = ranges[volume]
        start_page = se.get("start")
        end_page = se.get("end")
    if args.start_page is not None:
        start_page = args.start_page
    if args.end_page is not None:
        end_page = args.end_page

    # Filter by start/end if numeric indices available
    filt_pages: List[Path] = []
    for p in pages:
        idx = parse_page_index(p.stem)
        keep = True if idx is None else not ((start_page is not None and idx < start_page) or
                                             (end_page is not None and idx > end_page))
        if keep:
            filt_pages.append(p)

    # Skip-existing pruning before dispatch
    def needs_processing(p: Path) -> bool:
        if not args.skip_existing:
            return True
        need_txt = ("txt" in formats) and not (out_root / f"{p.stem}.txt").exists()
        need_tsv = False
        if "tsv" in formats:
            have_page = (out_root / f"{p.stem}.tsv").exists()
            have_cols = (out_root / f"{p.stem}_L.tsv").exists() and (out_root / f"{p.stem}_R.tsv").exists()
            need_tsv = not (have_page or have_cols)
        need_hocr = False
        if "hocr" in formats:
            have_page = (out_root / f"{p.stem}.hocr").exists()
            have_cols = (out_root / f"{p.stem}_L.hocr").exists() and (out_root / f"{p.stem}_R.hocr").exists()
            need_hocr = not (have_page or have_cols)
        return need_txt or need_tsv or need_hocr

    todo_pages = [p for p in filt_pages if needs_processing(p)]

    page_reports: List[PageReport] = []

    # Serial mode
    if args.workers <= 1 or len(todo_pages) <= 1:
        for p in todo_pages:
            try:
                rep = ocr_page(
                    img_path=p,
                    out_dir=out_root,
                    lang=args.lang,
                    psm=args.psm,
                    oem=args.oem,
                    formats=formats,
                    split_columns_mode=args.split_columns,
                    header_height_pct=args.header_height_pct,
                )
                page_reports.append(rep)
                note = " (split)" if rep.split else ""
                print(f"[ocr] {p.name}{note} mode={rep.mode} → chars={rep.n_chars} lines={rep.n_lines} ({rep.ms} ms)")
            except Exception as e:
                print(f"[ocr][ERROR] {p}: {e}")
    else:
        # Parallel execution
        max_workers = args.workers if args.workers > 0 else (os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    ocr_page,
                    p,
                    out_root,
                    args.lang,
                    args.psm,
                    args.oem,
                    formats,
                    args.split_columns,
                    args.header_height_pct,
                ): p
                for p in todo_pages
            }
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    rep = fut.result()
                    page_reports.append(rep)
                    note = " (split)" if rep.split else ""
                    print(f"[ocr] {p.name}{note} mode={rep.mode} → chars={rep.n_chars} lines={rep.n_lines} ({rep.ms} ms)")
                except Exception as e:
                    print(f"[ocr][ERROR] {p}: {e}")

    vrep = VolumeReport(
        volume=volume,
        in_root=str(in_root),
        out_root=str(out_root),
        lang=args.lang,
        psm=args.psm,
        formats=formats,
        start_page=start_page,
        end_page=end_page,
        n_pages=len(page_reports),
        pages=page_reports,
    )
    (out_root / "report.json").write_text(
        json.dumps(asdict(vrep), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[ocr] Done. Wrote {len(page_reports)} pages to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
