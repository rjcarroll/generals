#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preprocess.py — Page cleaning and two-column splitting for dictionary scans

Key option for this pass:
  --keep-full-height     → ZERO top/bottom cropping (split and non-split paths).
                           We preserve full vertical height and only crop horizontally.

Pipeline (per page):
  1) Load → grayscale → denoise
  2) Deskew (small-angle Hough)
  3) If --split-columns:
       a) detect gutter (projection valley + optional Hough assist) on deskewed GRAY
       b) confidence-gated split into left/right (trim around gutter)
       c) per-column: binarize → content crop (optionally full height) → save *_L.png, *_R.png
       d) optional JSON sidecar with decisions
     else:
       e) binarize → speck clean → content crop (optionally full height) → save
  4) Volume-level JSON report
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


# ------------------------------- ranges (YAML/JSON) ------------------------------- #

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
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            raise RuntimeError("YAML ranges provided but PyYAML is not installed")
        data = yaml.safe_load(txt) or {}
    else:
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


# ------------------------------------ data ------------------------------------ #

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
    volume_box: Optional[Tuple[int, int, int, int]]
    pages: List[PageReport]


# ------------------------------------ ops ------------------------------------ #

def calibrate_volume_box(img: np.ndarray,
                         pad_pct: float = 0.0,
                         margins: Tuple[float, float, float, float] = (0, 1, 0, 1)) -> Tuple[int, int, int, int]:
    """
    Estimate a bounding box for the page volume (ignores black borders).
    
    Args:
        img: input image as numpy array (grayscale or color).
        pad_pct: percentage (0.0–2.0) of extra padding applied uniformly on all sides.
        margins: (top, right, bottom, left) fractions of image dimension to keep.
    
    Returns:
        (x, y, w, h) bounding box in pixel coordinates.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # threshold: anything “ink” vs background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # find largest contour (the page region)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return (0, 0, img.shape[1], img.shape[0])
    
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    
    # apply padding
    pad_x = int(w * pad_pct)
    pad_y = int(h * pad_pct)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    w = min(img.shape[1] - x, w + 2 * pad_x)
    h = min(img.shape[0] - y, h + 2 * pad_y)
    
    # apply margin cropping
    top_m, right_m, bot_m, left_m = margins
    x += int(left_m * w)
    w -= int((left_m + right_m) * w)
    y += int(top_m * h)
    h -= int((top_m + bot_m) * h)
    
    return (x, y, w, h)

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def denoise(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(gray, ksize)


def estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, threshold=200)
    if lines is None:
        return 0.0
    angs = []
    for rho_theta in lines[:200]:
        _, theta = rho_theta[0]
        deg = (theta * 180.0 / np.pi) - 90.0
        if deg < -45:
            deg += 90
        if deg > 45:
            deg -= 90
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
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle_deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin) + (w * cos)) + 2 * border
    new_h = int((h * cos) + (w * sin)) + 2 * border
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=255)


def binarize(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th  # 0 = text, 255 = background


def remove_specks(binary: np.ndarray) -> np.ndarray:
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    return 255 - opened


# ----------------------------- cropping helpers ----------------------------- #

def crop_fixed_pct(img: np.ndarray, top=0.0, right=0.0, bottom=0.0, left=0.0) -> np.ndarray:
    h, w = img.shape[:2]
    t = int(round(h * (top / 100.0)))
    b = int(round(h * (bottom / 100.0)))
    r = int(round(w * (right / 100.0)))
    l = int(round(w * (left / 100.0)))
    y0, y1 = max(0, t), max(0, h - b)
    x0, x1 = max(0, l), max(0, w - r)
    if y1 <= y0 or x1 <= x0:
        return img
    return img[y0:y1, x0:x1]


def floodfill_border_to_white(img_bin: np.ndarray) -> np.ndarray:
    """Ensure anything connected to the border gets painted to 255 (bg)."""
    H, W = img_bin.shape[:2]
    work = img_bin.copy()
    mask = np.zeros((H + 2, W + 2), np.uint8)
    for sx, sy in [(1, 1), (W - 2, 1), (1, H - 2), (W - 2, H - 2)]:
        if work[sy, sx] != 255:
            cv2.floodFill(work, mask, (sx, sy), 255)
    return work


def robust_union_bbox(inv_img: np.ndarray, tiny_area_frac=0.0005, skinny_aspect=10.0, skinny_w_frac=0.03) -> Optional[Tuple[int, int, int, int]]:
    """Common core: union of non-skinny contours; ignore tiny/vertical-rule-like shapes."""
    H, W = inv_img.shape[:2]
    cnts, _ = cv2.findContours(inv_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    keep = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        aspect = h / max(1, w)
        if area < tiny_area_frac * W * H:
            continue
        if aspect > skinny_aspect and w < skinny_w_frac * W:
            continue
        keep.append((x, y, w, h))
    if not keep:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        return (x, y, x + w, y + h)
    xs0, ys0, xs1, ys1 = zip(*[(x, y, x + w, y + h) for (x, y, w, h) in keep])
    return (min(xs0), min(ys0), max(xs1), max(ys1))


def content_bbox(img_bin: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Robust bbox for full-page path."""
    work = floodfill_border_to_white(img_bin)
    inv = 255 - work
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    return robust_union_bbox(inv, tiny_area_frac=0.0005, skinny_aspect=10.0, skinny_w_frac=0.03)


def content_bbox_column(img_bin: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Robust bbox for a single column (slightly more permissive)."""
    work = floodfill_border_to_white(img_bin)
    inv = 255 - work
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    return robust_union_bbox(inv, tiny_area_frac=0.0004, skinny_aspect=10.0, skinny_w_frac=0.035)


def crop_by_box(img: np.ndarray, box: Tuple[int, int, int, int], pad_pct: float = 0.5) -> np.ndarray:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    pad_x = int(round(W * (pad_pct / 100.0)))
    pad_y = int(round(H * (pad_pct / 100.0)))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(W, x1 + pad_x)
    y1 = min(H, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]

def enforce_min_inner_span(box: Optional[Tuple[int,int,int,int]],
                           img_w: int,
                           side: str,
                           min_span_px: int) -> Optional[Tuple[int,int,int,int]]:
    """
    Ensure the crop spans at least `min_span_px` from the gutter inward.
    - For LEFT column: inner boundary is the RIGHT edge of the left image (x = img_w).
        Enforce width_from_inner = (img_w - x0) >= min_span_px  => x0 <= img_w - min_span_px
    - For RIGHT column: inner boundary is the LEFT edge (x = 0).
        Enforce width_from_inner = x1 >= min_span_px            => x1 >= min_span_px
    """
    if box is None or min_span_px <= 0:
        return box
    x0,y0,x1,y1 = box
    if side.lower() == "left":
        # push x0 outward if too tight
        x0 = min(x0, max(0, img_w - min_span_px))
        if x1 <= x0:  # keep sane
            x1 = min(img_w, x0 + min_span_px)
    else:  # "right"
        # pull x1 outward if too tight
        x1 = max(x1, min(img_w, min_span_px))
        if x1 <= x0:
            x0 = max(0, x1 - min_span_px)
    return (x0,y0,x1,y1)



# --------------------------- split helpers --------------------------- #

def adaptive_bin_for_projection(gray: np.ndarray) -> np.ndarray:
    """Lightweight binarization tuned for projections (text=1, bg=0)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    bin_ = cv2.adaptiveThreshold(eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY, 31, 15)
    inv = (255 - bin_) // 255  # text=1
    return inv


def detect_gutter(gray_rot: np.ndarray) -> tuple[int, float, dict]:
    """Return (gutter_x, confidence, debug)."""
    H, W = gray_rot.shape[:2]
    inv = adaptive_bin_for_projection(gray_rot)
    col_sum = inv.sum(axis=0).astype(np.float32)
    k = max(5, W // 200)
    smooth = cv2.blur(col_sum.reshape(1, -1), (1, k)).flatten()
    lo, hi = int(0.35 * W), int(0.65 * W)
    window = smooth[lo:hi]
    proj_x = int(np.argmin(window)) + lo

    left = smooth[max(0, proj_x - 40):proj_x]
    right = smooth[proj_x + 1:min(W, proj_x + 41)]
    neigh = float(np.mean(np.concatenate([left, right]))) if left.size + right.size > 0 else 1.0
    depth = max(0.0, (neigh - smooth[proj_x])) / (neigh + 1e-6)
    centered = 1.0 - min(1.0, abs((proj_x / W) - 0.5) * 2.0)
    proj_conf = 0.5 * depth + 0.5 * centered

    edges = cv2.Canny(gray_rot, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                            minLineLength=int(0.6 * H), maxLineGap=10)
    hough_x, hough_conf = None, 0.0
    if lines is not None:
        best, score = None, 0.0
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x2 - x1) < 2:
                xmid = int(round(0.5 * (x1 + x2)))
                centered_h = 1.0 - min(1.0, abs((xmid / W) - 0.5) * 2.0)
                length = (abs(y2 - y1) / H)
                s = 0.6 * centered_h + 0.4 * length
                if s > score:
                    score = s; best = xmid
        if best is not None:
            hough_x, hough_conf = best, float(score)

    if hough_x is not None and abs(hough_x - proj_x) <= int(0.03 * W):
        gutter_x = int(round(0.5 * (hough_x + proj_x)))
        conf = float(0.5 * (proj_conf + hough_conf))
        method = "agree"
    else:
        if (hough_x is not None) and (hough_conf >= proj_conf):
            gutter_x, conf, method = int(hough_x), float(hough_conf), "hough"
        else:
            gutter_x, conf, method = int(proj_x), float(proj_conf), "projection"

    dbg = {
        "W": W,
        "proj_x": int(proj_x),
        "proj_conf": round(float(proj_conf), 3),
        "hough_x": None if hough_x is None else int(hough_x),
        "hough_conf": round(float(hough_conf), 3),
        "method": method,
    }
    return gutter_x, conf, dbg


def split_by_gutter(
    gray_rot: np.ndarray,
    gutter_x: int,
    trim_px: int,
    overlap_px: int,
    strategy: str = "add",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (left, right) grayscale crops with optional overlap across the gutter.

    Params:
      gutter_x: detected gutter x position (in pixels)
      trim_px:  how much to shave around gutter (old behavior)
      overlap_px: how far to extend each side INTO the opposite column
                  so the gutter/inner margin appears in both crops
      strategy:
        - "add"              → ignore trim band near gutter; left ends at gutter+overlap,
                               right starts at gutter-overlap (show the rule clearly)
        - "trim_then_overlap"→ start from a trimmed split (gutter±trim), then pull each
                               side inward by overlap, i.e., left ends at (gutter-trim+overlap),
                               right starts at (gutter+trim-overlap)

    Notes:
      • If you want to GUARANTEE the gutter line is visible, use strategy="add" with overlap_px ≥ 3–8.
      • If you want to keep a small safety gap but still cross a bit, use "trim_then_overlap"
        and set overlap_px slightly less than trim_px.
    """
    H, W = gray_rot.shape[:2]

    if strategy == "trim_then_overlap":
        left_end  = min(W, max(0, gutter_x - max(0, trim_px) + max(0, overlap_px)))
        right_sta = max(0, min(W, gutter_x + max(0, trim_px) - max(0, overlap_px)))
    else:  # "add" (default)
        left_end  = min(W, gutter_x + max(0, overlap_px))
        right_sta = max(0, gutter_x - max(0, overlap_px))

    left  = gray_rot[:, :left_end]
    right = gray_rot[:, right_sta:]
    return left, right



# --------------------------- per-page pipeline --------------------------- #

def process_page(
    src_path: Path,
    out_path: Path,
    crop_mode: str,
    fixed_crop: Tuple[float, float, float, float],
    vol_box: Optional[Tuple[int, int, int, int]],
    pad_pct: float,
    min_area_ratio: float,   # reserved for future
    split_cfg: Optional[dict] = None,
    keep_full_height: bool = False,
) -> PageReport:
    t0 = time.time()
    img_bgr = cv2.imdecode(np.fromfile(str(src_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {src_path}")
    h0, w0 = img_bgr.shape[:2]

    # Deskew on grayscale
    gray = denoise(to_gray(img_bgr), 3)
    angle = estimate_skew_angle(gray)
    rotated = rotate_image(gray, angle)

    # --- two-column split path (BEFORE any global crop) ---
    if split_cfg is not None and split_cfg.get("enabled", False):
        gutter_x, conf, dbg = detect_gutter(rotated)
        W = rotated.shape[1]
        centered_ok = (0.3 * W) <= gutter_x <= (0.7 * W)
        if (conf >= split_cfg["min_conf"]) and centered_ok:
            # Split on deskewed gray
            left_gray, right_gray = split_by_gutter(
                rotated,
                gutter_x,
                split_cfg["gutter_trim_px"],
                split_cfg["gutter_overlap_px"],
                split_cfg["overlap_strategy"],
            )


            # Per-column binarize
            left_bin  = binarize(left_gray)
            right_bin = binarize(right_gray)

            # Column bboxes
            lb = content_bbox_column(left_bin)
            rb = content_bbox_column(right_bin)

            # If keeping full height, force y-range to entire column image
            if keep_full_height and lb is not None:
                x0, y0, x1, y1 = lb
                Hc, Wc = left_bin.shape[:2]
                lb = (x0, 0, x1, Hc)
            if keep_full_height and rb is not None:
                x0, y0, x1, y1 = rb
                Hr, Wr = right_bin.shape[:2]
                rb = (x0, 0, x1, Hr)

            # --- NEW: guarantee minimum inner span from the gutter into each column ---
            Wl = left_bin.shape[1]
            Wr = right_bin.shape[1]
            lb = enforce_min_inner_span(lb, Wl, side="left",  min_span_px=split_cfg.get("min_inner_span_left_px", 0))
            rb = enforce_min_inner_span(rb, Wr, side="right", min_span_px=split_cfg.get("min_inner_span_right_px", 0))

            left_crop  = crop_by_box(left_bin,  lb,  pad_pct=split_cfg["col_pad_pct"])  if lb else left_bin
            right_crop = crop_by_box(right_bin, rb,  pad_pct=split_cfg["col_pad_pct"])  if rb else right_bin

            # Optional extra outside shaves (left/right only; no top/bottom here)
            if split_cfg["outside_shave_left_pct"] > 0.0:
                left_crop = crop_fixed_pct(left_crop, 0.0, 0.0, 0.0, split_cfg["outside_shave_left_pct"])
            if split_cfg["outside_shave_right_pct"] > 0.0:
                right_crop = crop_fixed_pct(right_crop, 0.0, split_cfg["outside_shave_right_pct"], 0.0, 0.0)

            # Save L/R
            out_path.parent.mkdir(parents=True, exist_ok=True)
            stem = out_path.stem
            outL = out_path.parent / f"{stem}_L.png"
            outR = out_path.parent / f"{stem}_R.png"
            okL, bufL = cv2.imencode(".png", left_crop)
            okR, bufR = cv2.imencode(".png", right_crop)
            if not (okL and okR):
                raise RuntimeError(f"Failed to encode split PNGs for: {out_path}")
            outL.write_bytes(bufL.tobytes()); outR.write_bytes(bufR.tobytes())

            # Optional JSON sidecar
            if split_cfg.get("save_json", False):
                side = {
                    "page": str(src_path.name),
                    "mode": "two_column",
                    "skew_deg": float(round(angle, 2)),
                    "gutter_x": int(gutter_x),
                    "conf": float(round(conf, 3)),
                    "debug": dbg,
                    "left_shape": left_crop.shape[::-1],
                    "right_shape": right_crop.shape[::-1],
                }
                (out_path.parent / f"{stem}.json").write_text(
                    json.dumps(side, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            h1L, w1L = left_crop.shape[:2]
            ms = int((time.time() - t0) * 1000)
            return PageReport(
                src=str(src_path),
                out=str(outL),
                width=w0, height=h0,
                angle_deg=float(round(angle, 2)),
                cropped_width=w1L, cropped_height=h1L,
                ms=ms,
            )
        # else fall through to single-image path

    # --- single-image path (no split) ---
    binary = binarize(rotated)
    cleaned = remove_specks(binary)

    if crop_mode == "fixed":
        t, r, b, l = fixed_crop
        # If keeping full height, zero out top/bottom shaves
        if keep_full_height:
            t, b = 0.0, 0.0
        cropped = crop_fixed_pct(cleaned, t, r, b, l)
    elif crop_mode == "volume" and vol_box is not None:
        bb = vol_box
        if keep_full_height:
            x0, y0, x1, y1 = bb
            Hc, Wc = cleaned.shape[:2]
            bb = (x0, 0, x1, Hc)
        cropped = crop_by_box(cleaned, bb, pad_pct=pad_pct)
    else:  # auto
        bb = content_bbox(cleaned)
        if keep_full_height and bb is not None:
            x0, y0, x1, y1 = bb
            Hc, Wc = cleaned.shape[:2]
            bb = (x0, 0, x1, Hc)
        cropped = crop_by_box(cleaned, bb, pad_pct=pad_pct) if bb else cleaned

    # Optional extra shave AFTER auto/volume: if keeping full height, do not shave top/bottom
    if crop_mode != "fixed":
        t, r, b, l = fixed_crop
        if keep_full_height:
            t, b = 0.0, 0.0
        if any([t, r, b, l]):
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
        width=w0, height=h0,
        angle_deg=float(round(angle, 2)),
        cropped_width=w1, cropped_height=h1,
        ms=ms,
    )


# ----------------------------------- driver ----------------------------------- #

def find_pages(in_root: Path, ext: str) -> List[Path]:
    if ext == "auto":
        globs = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.webp", "*.bmp"]
    else:
        globs = [f"*{ext}"]
    pages: List[Path] = []
    for g in globs:
        pages.extend(sorted(in_root.glob(g)))
    return pages


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocess a volume of page images (deskew, crop, optional split)")
    ap.add_argument("--ranges-config", default=None,
                    help="YAML/JSON file mapping volume→{start,end}; pages outside are ignored")
    ap.add_argument("--quantile-low", type=float, default=0.10, help="Lower quantile for volume crop")
    ap.add_argument("--quantile-high", type=float, default=0.90, help="Upper quantile for volume crop")
    ap.add_argument("--in-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--report", default="report.json")
    

    # Cropping knobs
    ap.add_argument("--crop-mode", choices=["fixed", "auto", "volume"], default="auto",
                    help="fixed: same % margins; auto: per-page content crop; volume: one box for whole volume.")
    ap.add_argument("--calib-sample", type=int, default=64,
                    help="Max pages sampled to calibrate volume box.")
    ap.add_argument("--auto-crop-pad-pct", type=float, default=0.5,
                    help="Padding (%) when cropping to content box.")
    ap.add_argument("--crop-margins-pct", default="0,1.0,0,1.0",
                    help="Fixed crop top,right,bottom,left in % (e.g. '0,2,0,2').")
    ap.add_argument("--min-inner-span-left-px", type=int, default=0,
                help="Minimum pixels from the gutter into the LEFT column (prevents over-tight inner crops).")
    ap.add_argument("--min-inner-span-right-px", type=int, default=0,
                help="Minimum pixels from the gutter into the RIGHT column (prevents over-tight inner crops).")


    # Split knobs
    ap.add_argument("--split-columns", action="store_true",
                    help="Detect gutter and save two PNGs per page (_L/_R) with per-column crops.")
    ap.add_argument("--min-split-conf", type=float, default=0.35,
                    help="Minimum confidence required to split; otherwise fallback to single.")
    ap.add_argument("--gutter-trim-px", type=int, default=12,
                    help="Pixels shaved around gutter when splitting.")
    ap.add_argument("--column-pad-pct", type=float, default=0.6,
                    help="Padding %% used when cropping each column box.")
    ap.add_argument("--outside-fixed-shave-pct", default="0.0,0.0",
                    help="Extra fixed outside shave %% as 'leftColOutside,rightColOutside'.")
    ap.add_argument("--save-json", action="store_true",
                    help="Write a JSON sidecar per page with decisions/metrics (split mode).")
    ap.add_argument("--gutter-overlap-px", type=int, default=8,
                help="Pixels of symmetric overlap across the gutter (visible rule in both crops).")
    ap.add_argument("--overlap-strategy", choices=["add","trim_then_overlap"], default="add",
                help="How overlap combines with trimming: 'add' shows the rule clearly; 'trim_then_overlap' keeps a smaller gap.")


    # NEW: keep full vertical extent
    ap.add_argument("--keep-full-height", action="store_true",
                    help="Preserve full top/bottom (no vertical cropping) in both split and non-split modes.")

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

    # parse outside shaves for columns (leftColOutside,rightColOutside)
    try:
        oc = tuple(float(x.strip()) for x in args.outside_fixed_shave_pct.split(","))
        assert len(oc) == 2
        outside_shave_left, outside_shave_right = oc[0], oc[1]
    except Exception:
        outside_shave_left, outside_shave_right = 0.0, 0.0

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

    vol_box: Optional[Tuple[int, int, int, int]] = None
    if args.crop_mode == "volume":
        # Prefer early text pages; sample evenly but from the filtered list
        sample_n = min(args.calib_sample, len(pages))
        sample_pool = pages[:max(len(pages) // 2, sample_n)]
        idxs = np.linspace(0, len(sample_pool) - 1, num=sample_n, dtype=int)
        sample_pages = [sample_pool[i] for i in idxs]
        print(f"[preprocess] Calibrating volume crop on {len(sample_pages)} pages in-range…")
        vol_box = calibrate_volume_box(sample_pages, len(sample_pages), args.quantile_low, args.quantile_high)
        if vol_box:
            print(f"[preprocess] Volume crop box = {vol_box}")
        else:
            print("[preprocess] Calibration failed; falling back to per-page auto.")

    # Build split configuration from CLI
    split_cfg = {
        "enabled": True,
        "min_conf": args.min_split_conf,
        "gutter_trim_px": args.gutter_trim_px,
        "gutter_overlap_px": getattr(args, "gutter_overlap_px", 0),
        "overlap_strategy": getattr(args, "overlap_strategy", "add"),
        "col_pad_pct": args.column_pad_pct,
        "outside_shave_left_pct": outside_shave_left,
        "outside_shave_right_pct": outside_shave_right,
        "save_json": args.save_json,
        "min_inner_span_left_px": args.min_inner_span_left_px,    # NEW
        "min_inner_span_right_px": args.min_inner_span_right_px,  # NEW
    }



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
                min_area_ratio=0.3,  # reserved
                split_cfg=split_cfg,
                keep_full_height=args.keep_full_height,
            )
            page_reports.append(rep)
            print(f"[preprocess] {src.name} angle={rep.angle_deg}° → {Path(rep.out).name} ({rep.ms} ms)")
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
    print(f"[preprocess] Done. Wrote {len(page_reports)} page records to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
