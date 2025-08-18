#!/usr/bin/env python3
"""
preprocess.py — Page cleaning for dictionary scans

Pipeline:
  1) Load → grayscale → denoise
  2) Deskew (small-angle Hough)
  3) Binarize (CLAHE + Otsu)  [0 = black text, 255 = white bg]
  4) Remove specks (morph open on inverted)
  5) Crop, one of:
      - fixed: shave same % margins on all pages
      - auto: per-page content crop (flood-fill borders → largest contour)
      - volume: calibrate one crop box from a sample, apply to all pages
  6) Optional tiny extra fixed shave
  7) Save PNG + per-volume JSON report
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Optional

import cv2  # type: ignore
import numpy as np

# --- ranges (YAML/JSON) support ---

def parse_page_index(stem: str) -> int | None:
    import re
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else None

def load_ranges_config(path: str | None) -> dict[str, dict[str, int | None]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ranges config not found: {p}")
    txt = p.read_text(encoding="utf-8")
    data = None
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            raise RuntimeError("YAML ranges provided but PyYAML is not installed")
        data = yaml.safe_load(txt) or {}
    else:
        import json
        data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError("ranges config must be a dict with top-level key 'ranges'")
    rngs = data.get("ranges", {})
    if not isinstance(rngs, dict):
        raise ValueError("ranges config must contain a 'ranges' mapping")
    out: dict[str, dict[str, int | None]] = {}
    for vol, se in rngs.items():
        if isinstance(se, dict):
            out[vol] = {"start": se.get("start"), "end": se.get("end")}
    return out


# ------------------------------- data ------------------------------- #

@dataclass
class PageReport:
    src: str
    out: str
    width: int
    height: int
    angle_deg: float
    cropped_width: int
    cropped_height: int
    ms: int


@dataclass
class VolumeReport:
    volume: str
    in_root: str
    out_root: str
    n_pages: int
    crop_mode: str
    volume_box: Optional[Tuple[int,int,int,int]]
    pages: List[PageReport]


# ------------------------------- ops ------------------------------- #

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)

def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180.0, threshold=200)
    if lines is None:
        return 0.0
    angs = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        deg = (theta*180.0/np.pi) - 90.0
        if deg < -45: deg += 90
        if deg > 45: deg -= 90
        if -30 <= deg <= 30:
            angs.append(deg)
    if not angs:
        return 0.0
    med = float(np.median(angs))
    if abs(med) < 0.2:
        return 0.0
    return max(-15.0, min(15.0, med))

def rotate_image(img: np.ndarray, angle_deg: float, border: int = 10) -> np.ndarray:
    if abs(angle_deg) < 0.01:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle_deg, 1.0)
    cos, sin = abs(M[0,0]), abs(M[0,1])
    new_w = int((h*sin) + (w*cos)) + 2*border
    new_h = int((h*cos) + (w*sin)) + 2*border
    M[0,2] += (new_w - w)/2
    M[1,2] += (new_h - h)/2
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255)

def binarize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return th  # 0 = text, 255 = background

def remove_specks(binary: np.ndarray) -> np.ndarray:
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    return 255 - opened

# --- cropping helpers ---

def crop_fixed_pct(img: np.ndarray, top=0.0, right=0.0, bottom=0.0, left=0.0) -> np.ndarray:
    h, w = img.shape[:2]
    t = int(round(h*(top/100.0))); b = int(round(h*(bottom/100.0)))
    r = int(round(w*(right/100.0))); l = int(round(w*(left/100.0)))
    y0, y1 = max(0,t), max(0, h-b)
    x0, x1 = max(0,l), max(0, w-r)
    if y1 <= y0 or x1 <= x0: return img
    return img[y0:y1, x0:x1]

def floodfill_border_to_white(img_bin: np.ndarray) -> np.ndarray:
    """Ensure anything connected to the border gets painted to 255 (bg)."""
    H, W = img_bin.shape[:2]
    work = img_bin.copy()
    mask = np.zeros((H+2, W+2), np.uint8)
    for sx, sy in [(1,1), (W-2,1), (1,H-2), (W-2,H-2)]:
        if work[sy, sx] != 255:
            cv2.floodFill(work, mask, (sx, sy), 255)
    return work

def content_bbox(img_bin: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """(x0,y0,x1,y1) of largest internal content after wiping border-connected darks."""
    H, W = img_bin.shape[:2]
    work = floodfill_border_to_white(img_bin)
    inv = 255 - work
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    x,y,w,h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return (x, y, x+w, y+h)

def crop_by_box(img: np.ndarray, box: Tuple[int,int,int,int], pad_pct: float=0.5) -> np.ndarray:
    H, W = img.shape[:2]
    x0,y0,x1,y1 = box
    pad_x = int(round(W*(pad_pct/100.0)))
    pad_y = int(round(H*(pad_pct/100.0)))
    x0 = max(0, x0 - pad_x); y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x); y1 = min(H, y1 + pad_y)
    if x1 <= x0 or y1 <= y0: return img
    return img[y0:y1, x0:x1]

def calibrate_volume_box(sample_pages, _calib_n, qlo: float, qhi: float):
    """Compute a robust content bbox from given sample pages."""
    boxes = []
    for p in sample_pages:
        img = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None: continue
        gray = denoise(to_gray(img), 3)
        bin_ = binarize(gray)
        bb = content_bbox(bin_)
        if bb: boxes.append(bb)
    if not boxes:
        return None
    xs0, ys0, xs1, ys1 = zip(*boxes)
    x0 = int(np.quantile(xs0, qlo));  y0 = int(np.quantile(ys0, qlo))
    x1 = int(np.quantile(xs1, qhi));  y1 = int(np.quantile(ys1, qhi))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


# --------------------------- per-page pipeline --------------------------- #

def process_page(
    src_path: Path,
    out_path: Path,
    crop_mode: str,
    fixed_crop: Tuple[float,float,float,float],
    vol_box: Optional[Tuple[int,int,int,int]],
    pad_pct: float,
    min_area_ratio: float,
) -> PageReport:
    t0 = time.time()
    img_bgr = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {src_path}")
    h0, w0 = img_bgr.shape[:2]

    gray = denoise(to_gray(img_bgr), 3)
    angle = estimate_skew_angle(gray)
    rotated = rotate_image(gray, angle)

    binary = binarize(rotated)
    cleaned = remove_specks(binary)

    # --- cropping strategy ---
    if crop_mode == "fixed":
        t,r,b,l = fixed_crop
        cropped = crop_fixed_pct(cleaned, t, r, b, l)
    elif crop_mode == "volume" and vol_box is not None:
        cropped = crop_by_box(cleaned, vol_box, pad_pct=pad_pct)
    else:  # auto
        bb = content_bbox(cleaned)
        if bb:
            cropped = crop_by_box(cleaned, bb, pad_pct=pad_pct)
        else:
            cropped = cleaned

    # optional gentle extra shave for safety (applied after auto/volume)
    if crop_mode != "fixed":
        t,r,b,l = fixed_crop
        if any([t,r,b,l]):
            cropped = crop_fixed_pct(cropped, t, r, b, l)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", cropped)
    if not ok:
        raise RuntimeError(f"Failed to encode PNG for: {out_path}")
    out_path.write_bytes(buf.tobytes())

    h1, w1 = cropped.shape[:2]
    ms = int((time.time() - t0) * 1000)
    return PageReport(
        src=str(src_path),
        out=str(out_path),
        width=w0,
        height=h0,
        angle_deg=float(round(angle, 2)),
        cropped_width=w1,
        cropped_height=h1,
        ms=ms,
    )


# ------------------------------- driver ------------------------------- #

def find_pages(in_root: Path, ext: str) -> List[Path]:
    if ext == "auto":
        globs = ["*.png","*.jpg","*.jpeg","*.tif","*.tiff","*.webp","*.bmp"]
    else:
        globs = [f"*{ext}"]
    pages: List[Path] = []
    for g in globs:
        pages.extend(sorted(in_root.glob(g)))
    return pages

def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess a volume of page images (deskew, binarize, crop)")
    ap.add_argument("--ranges-config", default=None,
                help="YAML/JSON file mapping volume→{start,end}; pages outside are ignored")
    ap.add_argument("--quantile-low", type=float, default=0.10, help="Lower quantile for volume crop (e.g., 0.15)")
    ap.add_argument("--quantile-high", type=float, default=0.90, help="Upper quantile for volume crop (e.g., 0.85)")
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--report", default="report.json")
    # cropping knobs
    ap.add_argument("--crop-mode", choices=["fixed","auto","volume"], default="volume",
                    help="fixed: same %% margins; auto: per-page content crop; volume: one box for whole volume.")
    ap.add_argument("--calib-sample", type=int, default=64,
                    help="Max pages sampled to calibrate volume box.")
    ap.add_argument("--auto-crop-pad-pct", type=float, default=0.5,
                    help="Padding (%% of page) when cropping to content box.")
    ap.add_argument("--crop-margins-pct", default="0,1.5,0,1.5",
                    help="Fixed crop top,right,bottom,left in %% (e.g. '0,2,0,2').")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # parse fixed crop tuple
    try:
        fc = tuple(float(x.strip()) for x in args.crop_margins_pct.split(","))
        assert len(fc) == 4
        fixed_crop = (fc[0], fc[1], fc[2], fc[3])
    except Exception:
        fixed_crop = (0.0, 0.0, 0.0, 0.0)

    pages = find_pages(in_root, args.ext)
    # Optional start/end filtering using ranges config
    ranges = load_ranges_config(args.ranges_config)
    start_page = end_page = None
    if in_root.name in ranges:
        se = ranges[in_root.name]
        start_page = se.get("start")
        end_page = se.get("end")

    def keep(p: Path) -> bool:
        if start_page is None and end_page is None:
            return True
        idx = parse_page_index(p.stem)
        if idx is None:
            return True  # filenames w/o trailing number are kept
        if start_page is not None and idx < start_page:
            return False
        if end_page is not None and idx > end_page:
            return False
        return True

    pages = [p for p in pages if keep(p)]
    if not pages:
        print(f"[preprocess] No pages within range for {in_root}")
        return 1

    vol_box: Optional[Tuple[int,int,int,int]] = None
    if args.crop_mode == "volume":
        # Prefer early text pages; sample evenly but from the filtered list
        sample_n = min(args.calib_sample, len(pages))
        # Bias toward the beginning (first half) so front matter doesn’t dominate
        half = max(sample_n, 1)
        sample_pool = pages[:max(len(pages)//2, sample_n)]
        idxs = np.linspace(0, len(sample_pool)-1, num=sample_n, dtype=int)
        sample_pages = [sample_pool[i] for i in idxs]
        print(f"[preprocess] Calibrating volume crop on {len(sample_pages)} pages in-range…")
        vol_box = calibrate_volume_box(sample_pages, len(sample_pages), args.quantile_low, args.quantile_high)
        if vol_box:
            print(f"[preprocess] Volume crop box = {vol_box}")
        else:
            print("[preprocess] Calibration failed; falling back to per-page auto.")


    page_reports: List[PageReport] = []
    for src in pages:
        dst = out_root / (src.stem + ".png")
        if args.skip_existing and dst.exists():
            continue
        try:
            rep = process_page(
                src_path=src,
                out_path=dst,
                crop_mode=args.crop_mode,
                fixed_crop=fixed_crop,
                vol_box=vol_box,
                pad_pct=args.auto_crop_pad_pct,
                min_area_ratio=0.3,  # kept internal; can expose if needed
            )
            page_reports.append(rep)
            print(f"[preprocess] {src.name} angle={rep.angle_deg}° → {dst.name} ({rep.ms} ms)")
        except Exception as e:
            print(f"[preprocess][ERROR] {src}: {e}")

    vol_report = VolumeReport(
        volume=in_root.name,
        in_root=str(in_root),
        out_root=str(out_root),
        n_pages=len(page_reports),
        crop_mode=args.crop_mode,
        volume_box=vol_box,
        pages=page_reports,
    )
    (out_root / args.report).write_text(
        json.dumps(asdict(vol_report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[preprocess] Done. Wrote {len(page_reports)} pages to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
