#!/usr/bin/env python3
"""Compact image annotation, chip-local coordinate tables, and heuristic anomaly triage.

Important: this module does NOT contain a trained semiconductor defect classifier.
It writes localization records and optional unclassified local-contrast anomaly
candidates for review. Candidate labels are not confirmed physical defect types.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .nova_inference import locate
except ImportError:  # script execution from project root
    from nova_inference import locate


SITE_COLUMNS = [
    "sample_id", "lot_id", "wafer_id", "architecture", "die_row", "die_col", "base_chip_id", "chip_id", "chip_row", "chip_col",
    "reference_image", "search_image", "reference_sha256", "search_sha256",
    "search_width_px", "search_height_px", "chip_origin_x_px", "chip_origin_y_px",
    "pred_x_image_px", "pred_y_image_px", "pred_x_chip_local_px", "pred_y_chip_local_px",
    "ground_truth_x_image_px", "ground_truth_y_image_px", "ground_truth_x_chip_local_px",
    "ground_truth_y_chip_local_px", "localization_error_px", "match_score_ncc",
    "score_gap_ncc", "chosen_scale_ratio", "candidate_count", "center_prior_used",
    "localization_status", "score_semantics", "defect_assessment_status",
    "anomaly_candidate_count", "anomaly_candidates_retained", "anomaly_candidates_before_cap", "anomaly_candidates_omitted_by_cap",
    "high_severity_candidate_count", "medium_severity_candidate_count", "low_severity_candidate_count", "annotation_image"
]

ANOMALY_COLUMNS = [
    "sample_id", "lot_id", "wafer_id", "architecture", "die_row", "die_col", "chip_id", "candidate_id", "search_image",
    "x_image_px", "y_image_px", "x_chip_local_px", "y_chip_local_px",
    "bbox_x_image_px", "bbox_y_image_px", "bbox_width_px", "bbox_height_px",
    "area_px", "mean_intensity_0_255", "mean_signed_residual", "robust_anomaly_score",
    "peak_robust_anomaly_score", "candidate_kind", "observation", "severity", "defect_type", "confirmation_status", "recommended_action"
]

CHIP_COLUMNS = [
    "sample_id", "lot_id", "wafer_id", "architecture", "die_row", "die_col", "chip_id", "chip_row", "chip_col",
    "search_image", "image_width_px", "image_height_px", "chip_origin_x_px", "chip_origin_y_px",
    "chip_width_px", "chip_height_px", "predicted_site_count", "pred_x_chip_local_px",
    "pred_y_chip_local_px", "ground_truth_site_count", "ground_truth_x_chip_local_px",
    "ground_truth_y_chip_local_px", "localization_error_px", "match_score_ncc",
    "anomaly_candidate_count", "shortlisted_anomaly_candidate_count", "bright_candidate_count", "dark_candidate_count", "mixed_candidate_count",
    "high_severity_candidate_count", "medium_severity_candidate_count", "low_severity_candidate_count",
    "confirmed_defect_count", "defect_assessment_status"
]

REVIEWED_DEFECT_COLUMNS = [
    "annotation_id", "sample_id", "lot_id", "wafer_id", "architecture", "die_row", "die_col",
    "base_chip_id", "chip_id", "chip_row", "chip_col", "search_image", "x_image_px", "y_image_px",
    "x_chip_local_px", "y_chip_local_px", "chip_origin_x_px", "chip_origin_y_px",
    "bbox_x_image_px", "bbox_y_image_px", "bbox_width_px", "bbox_height_px",
    "defect_type", "severity", "confirmed", "confirmation_status", "annotator", "annotation_date", "evidence", "notes"
]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_gray_u8(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"Could not decode image: {path}")
    return image


def detect_anomaly_candidates(
    image: np.ndarray,
    *,
    threshold_sigma: float = 2.5,
    min_area: int = 6,
    max_area: int = 250,
    max_aspect_ratio: float = 5.0,
) -> list[dict[str, Any]]:
    """Return unclassified local-contrast outlier candidates from a grayscale image.

    This deliberately produces review candidates, not confirmed defects. A generic
    high-pass detector is vulnerable to periodic gates, edges, capture noise and
    contrast shifts; it must not be interpreted as a real semiconductor defect model.
    """
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image must be a non-empty grayscale 2-D array")
    if threshold_sigma <= 0 or min_area < 1 or max_area < min_area or max_aspect_ratio < 1:
        raise ValueError("invalid anomaly detector settings")

    img = image.astype(np.float32)
    # Residual against a medium-scale background suppresses slow illumination variation.
    background = cv2.GaussianBlur(img, (0, 0), sigmaX=2.5, sigmaY=2.5)
    residual = img - background
    center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - center)))
    robust_sigma = max(1.4826 * mad, 1.0)  # 1 intensity level prevents division blow-up.
    z = (residual - center) / robust_sigma
    mask = (np.abs(z) >= float(threshold_sigma)).astype(np.uint8) * 255

    # Remove isolated single pixels; preserve compact features for connected components.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[dict[str, Any]] = []
    h_img, w_img = image.shape
    for label in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < min_area or area > max_area or w < 2 or h < 2:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        if aspect > max_aspect_ratio:
            continue
        if x == 0 or y == 0 or x + w >= w_img or y + h >= h_img:
            continue
        component_mask = labels[y:y+h, x:x+w] == label
        comp_res = residual[y:y+h, x:x+w][component_mask]
        comp_z = z[y:y+h, x:x+w][component_mask]
        mean_res = float(np.mean(comp_res)) if comp_res.size else 0.0
        abs_z = float(np.mean(np.abs(comp_z))) if comp_z.size else 0.0
        max_z = float(np.max(np.abs(comp_z))) if comp_z.size else 0.0
        if mean_res >= 1.5:
            kind = "BRIGHT_LOCAL_CONTRAST"
        elif mean_res <= -1.5:
            kind = "DARK_LOCAL_CONTRAST"
        else:
            kind = "MIXED_TEXTURE_OUTLIER"
        severity = "HIGH" if max_z >= 6.0 else "MEDIUM" if max_z >= 4.5 else "LOW"
        cx, cy = centroids[label]
        candidates.append({
            "x_image_px": float(cx), "y_image_px": float(cy),
            "bbox_x_image_px": x, "bbox_y_image_px": y,
            "bbox_width_px": w, "bbox_height_px": h, "area_px": area,
            "mean_intensity_0_255": float(np.mean(image[y:y+h, x:x+w][component_mask])),
            "mean_signed_residual": mean_res,
            "robust_anomaly_score": abs_z,
            "max_robust_anomaly_score": max_z,
            "candidate_kind": kind,
            "observation": {"BRIGHT_LOCAL_CONTRAST":"localized bright contrast outlier", "DARK_LOCAL_CONTRAST":"localized dark contrast outlier", "MIXED_TEXTURE_OUTLIER":"mixed local texture outlier"}[kind],
            "severity": severity,
            "defect_type": "UNCLASSIFIED",
            "confirmation_status": "UNCONFIRMED_HEURISTIC_CANDIDATE",
            "recommended_action": "Review crop and compare with a validated clean reference; do not treat as a confirmed defect.",
        })
    candidates.sort(key=lambda r: (-r["max_robust_anomaly_score"], -r["robust_anomaly_score"], -r["area_px"], r["y_image_px"], r["x_image_px"]))
    for i, row in enumerate(candidates, 1):
        row["candidate_id"] = f"ANOM_{i:04d}"
    return candidates


def _tile_info(x: float, y: float, image_w: int, image_h: int,
               chip_w: int, chip_h: int, sample_id: str) -> dict[str, Any]:
    if chip_w < 1 or chip_h < 1:
        raise ValueError("chip width and height must be positive")
    col = min(max(int(math.floor(max(x, 0) / chip_w)), 0), max(0, math.ceil(image_w / chip_w) - 1))
    row = min(max(int(math.floor(max(y, 0) / chip_h)), 0), max(0, math.ceil(image_h / chip_h) - 1))
    origin_x, origin_y = col * chip_w, row * chip_h
    actual_w = min(chip_w, image_w - origin_x)
    actual_h = min(chip_h, image_h - origin_y)
    multi = chip_w < image_w or chip_h < image_h
    chip_id = f"{sample_id}_R{row:03d}_C{col:03d}" if multi else sample_id
    return {"chip_id": chip_id, "chip_row": row, "chip_col": col,
            "chip_origin_x_px": origin_x, "chip_origin_y_px": origin_y,
            "chip_width_px": actual_w, "chip_height_px": actual_h,
            "x_chip_local_px": float(x - origin_x), "y_chip_local_px": float(y - origin_y)}


def _font(size: int = 20):
    for font_path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int], mark: str, size: int = 10, width: int = 4):
    px, py = int(round(x)), int(round(y))
    if mark == "x":
        draw.line((px-size, py-size, px+size, py+size), fill=color, width=width)
        draw.line((px-size, py+size, px+size, py-size), fill=color, width=width)
    elif mark == "+":
        draw.line((px-size, py, px+size, py), fill=color, width=width)
        draw.line((px, py-size, px, py+size), fill=color, width=width)
    elif mark == "square":
        draw.rectangle((px-size, py-size, px+size, py+size), outline=color, width=width)


def render_inspection_overlay(
    reference_path: str | Path, search_path: str | Path, output_path: str | Path,
    *, sample_id: str, architecture: str, pred_x: float, pred_y: float,
    score_ncc: float, scale_ratio: float | None,
    score_gap_ncc: float | None = None, center_prior_used: bool | None = None, candidate_count: int | None = None,
    ground_truth_x: float | None = None, ground_truth_y: float | None = None,
    chip_local_x: float | None = None, chip_local_y: float | None = None,
    chip_width_px: int | None = None, chip_height_px: int | None = None,
    localization_error_px: float | None = None,
    anomaly_candidates: list[dict[str, Any]] | None = None,
    annotate_anomalies: int = 12,
) -> Path:
    """Render a simple side-by-side reference/search overlay like the user's example."""
    ref = _read_gray_u8(reference_path)
    search = _read_gray_u8(search_path)
    panel_h = 700
    margin = 20
    header_h = 150
    ref_w = max(1, int(round(ref.shape[1] * panel_h / ref.shape[0])))
    search_w = max(1, int(round(search.shape[1] * panel_h / search.shape[0])))
    gap = 28
    canvas_w = margin * 2 + ref_w + gap + search_w
    canvas_h = header_h + panel_h + margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    f_title, f_body, f_small = _font(25), _font(18), _font(15)
    err_txt = f"{localization_error_px:.1f}px" if localization_error_px is not None and math.isfinite(localization_error_px) else "n/a"
    truth_txt = f"({ground_truth_x:.1f}, {ground_truth_y:.1f}) px" if ground_truth_x is not None and ground_truth_y is not None else "not provided"
    scale_txt = f"{scale_ratio:.4f}" if scale_ratio is not None else "n/a"
    local_txt = (f"chip-local=({chip_local_x:.1f}, {chip_local_y:.1f}) px"
                 if chip_local_x is not None and chip_local_y is not None else "chip-local=(n/a, n/a)")
    draw.text((margin, 8), f"NOVA-Loc | {architecture} | {sample_id}", fill=(20, 20, 20), font=f_title)
    draw.text((margin, 43), f"pred=({pred_x:.1f}, {pred_y:.1f}) px | true={truth_txt} | error={err_txt}", fill=(25, 25, 25), font=f_body)
    draw.text((margin, 72), f"NCC score={score_ncc:.3f} (not probability) | scale={scale_txt} | {local_txt}", fill=(50, 50, 50), font=f_small)
    gap_txt = f"{score_gap_ncc:.4f}" if score_gap_ncc is not None else "n/a"
    prior_txt = "yes" if center_prior_used else "no" if center_prior_used is not None else "n/a"
    cand_txt = str(candidate_count) if candidate_count is not None else "n/a"
    draw.text((margin, 96), f"score gap={gap_txt} | center prior used={prior_txt} | localization candidates={cand_txt}", fill=(65, 65, 65), font=f_small)

    ref_img = Image.fromarray(ref).convert("RGB").resize((ref_w, panel_h), Image.Resampling.BILINEAR)
    search_img = Image.fromarray(search).convert("RGB").resize((search_w, panel_h), Image.Resampling.BILINEAR)
    ref_x, img_y = margin, header_h
    search_x = margin + ref_w + gap
    canvas.paste(ref_img, (ref_x, img_y))
    canvas.paste(search_img, (search_x, img_y))
    draw.text((ref_x, header_h - 25), f"Reference: {Path(reference_path).name}", fill=(25, 25, 25), font=f_small)
    draw.text((search_x, header_h - 25), f"Search: {Path(search_path).name}", fill=(25, 25, 25), font=f_small)

    sx, sy = search_w / search.shape[1], panel_h / search.shape[0]
    pred_disp = (search_x + pred_x * sx, img_y + pred_y * sy)
    _draw_cross(draw, *pred_disp, (238, 35, 35), "x", size=11, width=4)
    if ground_truth_x is not None and ground_truth_y is not None:
        gt_disp = (search_x + ground_truth_x * sx, img_y + ground_truth_y * sy)
        _draw_cross(draw, *gt_disp, (15, 210, 60), "+", size=12, width=4)
        # A subtle dashed line connects prediction and truth without obscuring the image.
        x0, y0 = pred_disp; x1, y1 = gt_disp
        length = math.hypot(x1-x0, y1-y0)
        if length > 0:
            for t in np.arange(0.0, 1.0, 12.0 / length):
                xa, ya = x0+(x1-x0)*t, y0+(y1-y0)*t
                tb = min(1.0, t + 5.0 / length)
                xb, yb = x0+(x1-x0)*tb, y0+(y1-y0)*tb
                draw.line((xa, ya, xb, yb), fill=(255, 215, 0), width=2)

    # Only the strongest few candidates are marked on the picture; the CSV retains all.
    # If an explicit regular chip/die grid is configured, show its boundaries.
    if chip_width_px and chip_height_px and (chip_width_px < search.shape[1] or chip_height_px < search.shape[0]):
        for gx in range(chip_width_px, search.shape[1], chip_width_px):
            x_disp = search_x + gx * sx
            draw.line((x_disp, img_y, x_disp, img_y + panel_h), fill=(70, 170, 255), width=1)
        for gy in range(chip_height_px, search.shape[0], chip_height_px):
            y_disp = img_y + gy * sy
            draw.line((search_x, y_disp, search_x + search_w, y_disp), fill=(70, 170, 255), width=1)

    if anomaly_candidates:
        for candidate in anomaly_candidates[:max(0, annotate_anomalies)]:
            ax, ay = candidate["x_image_px"], candidate["y_image_px"]
            _draw_cross(draw, search_x + ax*sx, img_y + ay*sy, (255, 165, 0), "square", size=5, width=2)

    # Compact legend at the bottom.
    legend_y = header_h + panel_h + 2
    _draw_cross(draw, margin + 12, legend_y + 9, (238, 35, 35), "x", size=7, width=3)
    draw.text((margin + 26, legend_y), "prediction", fill=(40, 40, 40), font=f_small)
    _draw_cross(draw, margin + 132, legend_y + 9, (15, 210, 60), "+", size=7, width=3)
    draw.text((margin + 146, legend_y), "ground truth", fill=(40, 40, 40), font=f_small)
    _draw_cross(draw, margin + 280, legend_y + 9, (255, 165, 0), "square", size=6, width=2)
    draw.text((margin + 294, legend_y), "unclassified anomaly candidate (review)", fill=(40, 40, 40), font=f_small)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92, optimize=True)
    return output


def _path_label(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        columns = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _resolve_input_path(value: str, root: Path, manifest_parent: Path | None = None) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    candidates = [root / p]
    if manifest_parent is not None:
        candidates.append(manifest_parent / p)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _load_ground_truth(root: Path, input_manifest: str | Path | None = None) -> list[dict[str, Any]]:
    """Load pairs from the bundled synthetic GT CSVs or a user-supplied manifest.

    Manifest required columns: sample_id (or id), reference, search. Optional columns:
    architecture, chip_id, gt_x/gt_y (or x/y). Paths can be absolute or relative to
    the project root / manifest directory. Empty GT fields are allowed for real images.
    """
    rows: list[dict[str, Any]] = []
    if input_manifest is not None:
        manifest_path = Path(input_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (root / manifest_path).resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Input manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as f:
            manifest_rows = list(csv.DictReader(f))
        for i, row in enumerate(manifest_rows, 1):
            sample_id = (row.get("sample_id") or row.get("id") or f"sample_{i:04d}").strip()
            ref_value, search_value = row.get("reference"), row.get("search")
            if not ref_value or not search_value:
                raise ValueError(f"Manifest row {i} must include reference and search paths")
            ref = _resolve_input_path(ref_value, root, manifest_path.parent)
            search = _resolve_input_path(search_value, root, manifest_path.parent)
            if not ref.is_file(): raise FileNotFoundError(f"Reference image missing for {sample_id}: {ref}")
            if not search.is_file(): raise FileNotFoundError(f"Search image missing for {sample_id}: {search}")
            gx = row.get("gt_x", row.get("x", ""))
            gy = row.get("gt_y", row.get("y", ""))
            gx = "" if gx is None else str(gx).strip()
            gy = "" if gy is None else str(gy).strip()
            if bool(gx) != bool(gy):
                raise ValueError(f"Manifest row {i} must provide both gt_x and gt_y, or neither")
            if gx and gy:
                float(gx); float(gy)
            rows.append({"id":sample_id, "sample_id":sample_id,
                         "architecture":(row.get("architecture") or "UNKNOWN").strip(),
                         "chip_id":(row.get("chip_id") or sample_id).strip(),
                         "lot_id":(row.get("lot_id") or "NOT_PROVIDED").strip(),
                         "wafer_id":(row.get("wafer_id") or "NOT_PROVIDED").strip(),
                         "die_row":(row.get("die_row") or "NOT_PROVIDED").strip(),
                         "die_col":(row.get("die_col") or "NOT_PROVIDED").strip(),
                         "reference":str(ref), "search":str(search), "gt_x":gx, "gt_y":gy})
        if not rows:
            raise ValueError(f"Input manifest has no data rows: {manifest_path}")
        return rows

    for arch in ("DRAM", "FinFET"):
        gt = root / "results" / "baseline" / arch / "ground_truth.csv"
        if not gt.is_file():
            raise FileNotFoundError(f"Ground-truth metadata not found: {gt}")
        with gt.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                r = dict(row)
                r["architecture"] = arch
                r["sample_id"] = r["id"]
                r["chip_id"] = r["id"]
                r["lot_id"] = "NOT_PROVIDED"
                r["wafer_id"] = "NOT_PROVIDED"
                r["die_row"] = "NOT_PROVIDED"
                r["die_col"] = "NOT_PROVIDED"
                r["gt_x"] = r.get("x", "")
                r["gt_y"] = r.get("y", "")
                r["reference"] = str(_resolve_input_path(r["reference"], root))
                r["search"] = str(_resolve_input_path(r["search"], root))
                rows.append(r)
    return rows


def create_inspection_report(
    root: str | Path,
    output_dir: str | Path,
    *,
    predictions_csv: str | Path | None = None,
    input_manifest: str | Path | None = None,
    defect_annotations_csv: str | Path | None = None,
    split: str = "holdout_centered",
    method: str = "NOVA_hybrid_roi_gate",
    chip_width_px: int | None = None,
    chip_height_px: int | None = None,
    scan_anomalies: bool = True,
    threshold_sigma: float = 2.5,
    min_area: int = 6,
    max_area: int = 250,
    max_candidates_per_image: int = 20,
    annotate_anomalies: int = 12,
) -> dict[str, Any]:
    """Create visual, site, chip-summary and anomaly-candidate output tables.

    By default, uses the bundled synthetic GT CSVs. Pass `input_manifest` for external
    image pairs with optional ground truth and lot/wafer/die metadata. Rows without GT
    are still localized, but localization error and physical-wafer coordinates remain unavailable.
    """
    root = Path(root).resolve()
    out_dir = Path(output_dir).resolve()
    visuals = out_dir / "visuals"
    visuals.mkdir(parents=True, exist_ok=True)
    gt_rows = _load_ground_truth(root, input_manifest=input_manifest)

    if chip_width_px is not None and chip_width_px < 1:
        raise ValueError("chip_width_px must be positive")
    if chip_height_px is not None and chip_height_px < 1:
        raise ValueError("chip_height_px must be positive")
    if (chip_width_px is None) != (chip_height_px is None):
        raise ValueError("provide both chip_width_px and chip_height_px, or neither")
    if max_candidates_per_image < 0:
        raise ValueError("max_candidates_per_image must be >= 0; 0 means no cap")

    pred_by_id: dict[str, dict[str, Any]] = {}
    pred_path: Path | None = None
    if predictions_csv is not None:
        pred_path = Path(predictions_csv)
        if not pred_path.is_absolute():
            pred_path = (root / pred_path).resolve()
    elif input_manifest is None:
        candidate = root / "results" / "integrated_benchmark" / "integrated_per_sample_results.csv"
        if candidate.is_file():
            pred_path = candidate
    if pred_path is not None and pred_path.is_file():
        with pred_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("split") == split and row.get("method") == method:
                    pred_by_id[row["id"]] = row

    site_rows: list[dict[str, Any]] = []
    anomaly_rows: list[dict[str, Any]] = []
    anomaly_rows_all: list[dict[str, Any]] = []
    chip_rows: list[dict[str, Any]] = []
    anomaly_count_by_sample: dict[str, list[dict[str, Any]]] = {}
    sample_record_by_id: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for index, gt in enumerate(gt_rows, 1):
        sample_id = gt.get("sample_id") or gt["id"]
        arch = gt.get("architecture") or "UNKNOWN"
        base_chip_id = gt.get("chip_id") or sample_id
        ref_path, search_path = Path(gt["reference"]), Path(gt["search"])
        search_img = _read_gray_u8(search_path)
        ih, iw = search_img.shape
        cw = int(chip_width_px or iw)
        ch = int(chip_height_px or ih)
        truth_x = float(gt["gt_x"]) if str(gt.get("gt_x", "")).strip() else None
        truth_y = float(gt["gt_y"]) if str(gt.get("gt_y", "")).strip() else None
        if (truth_x is None) != (truth_y is None):
            raise ValueError(f"Both ground truth coordinates must be provided for {sample_id}, or neither")

        pred = pred_by_id.get(sample_id)
        if pred is not None:
            px, py = float(pred["pred_x"]), float(pred["pred_y"])
            match_score = float(pred["score"])
            scale_ratio = float(pred["chosen_scale_ratio"]) if pred.get("chosen_scale_ratio") not in (None, "", "None") else None
            gap = float(pred["center_score_gap"]) if pred.get("center_score_gap") not in (None, "", "None") else None
            center_prior = str(pred.get("center_prior_used", "")).lower() in ("true", "1")
            loc_status = str(pred.get("gate_status") or "from_benchmark_prediction")
            candidate_count = None
        else:
            loc = locate(ref_path, search_path)
            px, py = float(loc.x), float(loc.y)
            match_score = float(loc.score)
            scale_ratio = float(loc.chosen_scale_ratio)
            gap = float(loc.score_gap_from_best)
            center_prior = bool(loc.used_center_prior)
            loc_status = str(loc.status)
            candidate_count = int(loc.candidate_count)

        err = float(math.hypot(px - truth_x, py - truth_y)) if truth_x is not None and truth_y is not None else None
        p_tile = _tile_info(px, py, iw, ih, cw, ch, base_chip_id)
        gt_tile = _tile_info(truth_x, truth_y, iw, ih, cw, ch, base_chip_id) if truth_x is not None and truth_y is not None else None
        if scan_anomalies:
            all_candidates = detect_anomaly_candidates(search_img, threshold_sigma=threshold_sigma,
                                                      min_area=min_area, max_area=max_area)
        else:
            all_candidates = []
        raw_candidate_count = len(all_candidates)
        candidates = (all_candidates[:max_candidates_per_image]
                      if max_candidates_per_image > 0 else all_candidates)
        omitted_candidate_count = raw_candidate_count - len(candidates)
        anomaly_count_by_sample[sample_id] = candidates

        overlay_path = visuals / arch / f"{sample_id}_inspection.jpg"
        render_inspection_overlay(
            ref_path, search_path, overlay_path, sample_id=sample_id, architecture=arch,
            pred_x=px, pred_y=py, score_ncc=match_score, scale_ratio=scale_ratio,
            score_gap_ncc=gap, center_prior_used=center_prior, candidate_count=candidate_count,
            ground_truth_x=truth_x, ground_truth_y=truth_y,
            chip_local_x=p_tile["x_chip_local_px"], chip_local_y=p_tile["y_chip_local_px"],
            chip_width_px=cw, chip_height_px=ch,
            localization_error_px=err, anomaly_candidates=candidates,
            annotate_anomalies=annotate_anomalies)

        local_pred_tile = p_tile
        high = sum(c["severity"] == "HIGH" for c in all_candidates)
        med = sum(c["severity"] == "MEDIUM" for c in all_candidates)
        low = sum(c["severity"] == "LOW" for c in all_candidates)
        site_rows.append({
            "sample_id": sample_id, "lot_id": gt.get("lot_id", "NOT_PROVIDED"),
            "wafer_id": gt.get("wafer_id", "NOT_PROVIDED"), "architecture": arch,
            "die_row": gt.get("die_row", "NOT_PROVIDED"), "die_col": gt.get("die_col", "NOT_PROVIDED"),
            "base_chip_id": base_chip_id, "chip_id": p_tile["chip_id"],
            "chip_row": p_tile["chip_row"], "chip_col": p_tile["chip_col"],
            "reference_image": _path_label(ref_path, root), "search_image": _path_label(search_path, root),
            "reference_sha256": sha256_file(ref_path), "search_sha256": sha256_file(search_path),
            "search_width_px": iw, "search_height_px": ih,
            "chip_origin_x_px": p_tile["chip_origin_x_px"], "chip_origin_y_px": p_tile["chip_origin_y_px"],
            "pred_x_image_px": px, "pred_y_image_px": py,
            "pred_x_chip_local_px": p_tile["x_chip_local_px"], "pred_y_chip_local_px": p_tile["y_chip_local_px"],
            "ground_truth_x_image_px": truth_x if truth_x is not None else "", "ground_truth_y_image_px": truth_y if truth_y is not None else "",
            "ground_truth_x_chip_local_px": gt_tile["x_chip_local_px"] if gt_tile else "NOT_PROVIDED",
            "ground_truth_y_chip_local_px": gt_tile["y_chip_local_px"] if gt_tile else "NOT_PROVIDED",
            "localization_error_px": err if err is not None else "", "match_score_ncc": match_score,
            "score_gap_ncc": gap, "chosen_scale_ratio": scale_ratio,
            "candidate_count": candidate_count, "center_prior_used": center_prior,
            "localization_status": loc_status,
            "score_semantics": "NCC/score-gap diagnostics; not calibrated probabilities",
            "defect_assessment_status": "UNCONFIRMED_HEURISTIC_ANOMALY_TRIAGE_ONLY" if scan_anomalies else "NOT_RUN",
            "anomaly_candidate_count": raw_candidate_count, "anomaly_candidates_retained": len(candidates),
            "anomaly_candidates_before_cap": raw_candidate_count, "anomaly_candidates_omitted_by_cap": omitted_candidate_count,
            "high_severity_candidate_count": high,
            "medium_severity_candidate_count": med, "low_severity_candidate_count": low,
            "annotation_image": _path_label(overlay_path, root),
        })

        # Export all thresholded components, plus a top-K shortlist for quick review.
        def candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
            tile = _tile_info(candidate["x_image_px"], candidate["y_image_px"], iw, ih, cw, ch, base_chip_id)
            return {
                "sample_id": sample_id, "lot_id": gt.get("lot_id", "NOT_PROVIDED"),
                "wafer_id": gt.get("wafer_id", "NOT_PROVIDED"), "architecture": arch,
                "die_row": gt.get("die_row", "NOT_PROVIDED"), "die_col": gt.get("die_col", "NOT_PROVIDED"),
                "chip_id": tile["chip_id"],
                "candidate_id": candidate["candidate_id"], "search_image": _path_label(search_path, root),
                "x_image_px": candidate["x_image_px"], "y_image_px": candidate["y_image_px"],
                "x_chip_local_px": tile["x_chip_local_px"], "y_chip_local_px": tile["y_chip_local_px"],
                "bbox_x_image_px": candidate["bbox_x_image_px"], "bbox_y_image_px": candidate["bbox_y_image_px"],
                "bbox_width_px": candidate["bbox_width_px"], "bbox_height_px": candidate["bbox_height_px"],
                "area_px": candidate["area_px"], "mean_intensity_0_255": candidate["mean_intensity_0_255"],
                "mean_signed_residual": candidate["mean_signed_residual"],
                "robust_anomaly_score": candidate["robust_anomaly_score"],
                "peak_robust_anomaly_score": candidate["max_robust_anomaly_score"],
                "candidate_kind": candidate["candidate_kind"], "observation": candidate["observation"],
                "severity": candidate["severity"], "defect_type": candidate["defect_type"],
                "confirmation_status": candidate["confirmation_status"],
                "recommended_action": candidate["recommended_action"],
            }
        anomaly_rows_all.extend(candidate_record(c) for c in all_candidates)
        anomaly_rows.extend(candidate_record(c) for c in candidates)

        # Summarize all chip tiles in this image; empty tiles are retained in the table.
        ncols, nrows = math.ceil(iw / cw), math.ceil(ih / ch)
        for row_idx in range(nrows):
            for col_idx in range(ncols):
                x0, y0 = col_idx * cw, row_idx * ch
                tw, th = min(cw, iw-x0), min(ch, ih-y0)
                multi = ncols > 1 or nrows > 1
                chip_id = f"{base_chip_id}_R{row_idx:03d}_C{col_idx:03d}" if multi else base_chip_id
                predicted_here = (x0 <= px < x0+tw and y0 <= py < y0+th)
                truth_here = (truth_x is not None and truth_y is not None and x0 <= truth_x < x0+tw and y0 <= truth_y < y0+th)
                local_candidates_all = [c for c in all_candidates if x0 <= c["x_image_px"] < x0+tw and y0 <= c["y_image_px"] < y0+th]
                local_candidates = [c for c in candidates if x0 <= c["x_image_px"] < x0+tw and y0 <= c["y_image_px"] < y0+th]
                chip_rows.append({
                    "sample_id": sample_id, "lot_id": gt.get("lot_id", "NOT_PROVIDED"),
                    "wafer_id": gt.get("wafer_id", "NOT_PROVIDED"), "architecture": arch,
                    "die_row": gt.get("die_row", "NOT_PROVIDED"), "die_col": gt.get("die_col", "NOT_PROVIDED"),
                    "chip_id": chip_id,
                    "chip_row": row_idx, "chip_col": col_idx, "search_image": _path_label(search_path, root),
                    "image_width_px": iw, "image_height_px": ih, "chip_origin_x_px": x0, "chip_origin_y_px": y0,
                    "chip_width_px": tw, "chip_height_px": th,
                    "predicted_site_count": int(predicted_here),
                    "pred_x_chip_local_px": round(px-x0, 4) if predicted_here else "",
                    "pred_y_chip_local_px": round(py-y0, 4) if predicted_here else "",
                    "ground_truth_site_count": (int(truth_here) if truth_x is not None and truth_y is not None else "NOT_PROVIDED"),
                    "ground_truth_x_chip_local_px": (round(truth_x-x0, 4) if truth_here else "") if truth_x is not None and truth_y is not None else "NOT_PROVIDED",
                    "ground_truth_y_chip_local_px": (round(truth_y-y0, 4) if truth_here else "") if truth_x is not None and truth_y is not None else "NOT_PROVIDED",
                    "localization_error_px": round(err, 4) if err is not None and predicted_here and truth_here else "",
                    "match_score_ncc": round(match_score, 6) if predicted_here else "",
                    "anomaly_candidate_count": len(local_candidates_all),
                    "shortlisted_anomaly_candidate_count": len(local_candidates),
                    "high_severity_candidate_count": sum(c["severity"] == "HIGH" for c in local_candidates_all),
                    "medium_severity_candidate_count": sum(c["severity"] == "MEDIUM" for c in local_candidates_all),
                    "low_severity_candidate_count": sum(c["severity"] == "LOW" for c in local_candidates_all),
                    "bright_candidate_count": sum(c["candidate_kind"] == "BRIGHT_LOCAL_CONTRAST" for c in local_candidates_all),
                    "dark_candidate_count": sum(c["candidate_kind"] == "DARK_LOCAL_CONTRAST" for c in local_candidates_all),
                    "mixed_candidate_count": sum(c["candidate_kind"] == "MIXED_TEXTURE_OUTLIER" for c in local_candidates_all),
                    "confirmed_defect_count": "NOT_AVAILABLE_NO_DEFECT_LABELS",
                    "defect_assessment_status": "UNCONFIRMED_HEURISTIC_ANOMALY_TRIAGE_ONLY" if scan_anomalies else "NOT_RUN",
                })

        if index % 10 == 0:
            print(f"Inspection reporting: {index}/{len(gt_rows)} images processed")

    # Optional expert-reviewed annotations remain separate from heuristic anomaly candidates.
    # Annotation coordinates are supplied in search-image pixels; the report adds chip-local pixels.
    reviewed_defects: list[dict[str, Any]] = []
    if defect_annotations_csv is not None:
        defect_path = Path(defect_annotations_csv)
        if not defect_path.is_absolute():
            defect_path = (root / defect_path).resolve()
        if not defect_path.is_file():
            raise FileNotFoundError(f"Defect annotation CSV not found: {defect_path}")
        site_by_id = {r["sample_id"]: r for r in site_rows}
        with defect_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"sample_id", "x_image_px", "y_image_px", "defect_type"}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                raise ValueError(f"Defect annotation CSV must contain columns: {sorted(required)}")
            for ann_idx, ann in enumerate(reader, 1):
                sid = str(ann.get("sample_id", "")).strip()
                if sid not in site_by_id:
                    raise ValueError(f"Defect annotation row {ann_idx}: sample_id {sid!r} is not present in this report")
                site = site_by_id[sid]
                x = float(ann["x_image_px"]); y = float(ann["y_image_px"])
                iw, ih = int(site["search_width_px"]), int(site["search_height_px"])
                if not (0 <= x < iw and 0 <= y < ih):
                    raise ValueError(f"Defect annotation row {ann_idx}: coordinate ({x}, {y}) is outside {iw}x{ih} search image")
                tile = _tile_info(x, y, iw, ih, int(chip_width_px or iw), int(chip_height_px or ih), str(site["base_chip_id"]))
                confirmed_value = str(ann.get("confirmed", "false")).strip().lower() in ("1", "true", "yes", "confirmed")
                reviewed_defects.append({
                    "annotation_id": ann.get("annotation_id") or f"DEFECT_{ann_idx:05d}",
                    "sample_id": sid, "lot_id": ann.get("lot_id") or site["lot_id"],
                    "wafer_id": ann.get("wafer_id") or site["wafer_id"],
                    "architecture": ann.get("architecture") or site["architecture"],
                    "die_row": ann.get("die_row") or site["die_row"], "die_col": ann.get("die_col") or site["die_col"],
                    "base_chip_id": site["base_chip_id"], "chip_id": tile["chip_id"],
                    "chip_row": tile["chip_row"], "chip_col": tile["chip_col"],
                    "search_image": site["search_image"], "x_image_px": x, "y_image_px": y,
                    "x_chip_local_px": round(tile["x_chip_local_px"], 4), "y_chip_local_px": round(tile["y_chip_local_px"], 4),
                    "chip_origin_x_px": tile["chip_origin_x_px"], "chip_origin_y_px": tile["chip_origin_y_px"],
                    "bbox_x_image_px": ann.get("bbox_x_image_px", ""), "bbox_y_image_px": ann.get("bbox_y_image_px", ""),
                    "bbox_width_px": ann.get("bbox_width_px", ""), "bbox_height_px": ann.get("bbox_height_px", ""),
                    "defect_type": str(ann["defect_type"]).strip(), "severity": ann.get("severity", "NOT_PROVIDED"),
                    "confirmed": confirmed_value,
                    "confirmation_status": "CONFIRMED_EXPERT_LABEL" if confirmed_value else "REVIEWED_BUT_NOT_CONFIRMED",
                    "annotator": ann.get("annotator", "NOT_PROVIDED"), "annotation_date": ann.get("annotation_date", "NOT_PROVIDED"),
                    "evidence": ann.get("evidence", "NOT_PROVIDED"), "notes": ann.get("notes", ""),
                })
    _write_csv(out_dir / "reviewed_defect_annotations.csv", reviewed_defects, REVIEWED_DEFECT_COLUMNS)

    # Keep the chip summary explicit: counts are known only for expert annotations actually supplied.
    if defect_annotations_csv is not None:
        from collections import defaultdict
        ann_counts: dict[tuple[str, str], int] = defaultdict(int)
        for ann in reviewed_defects:
            if ann["confirmed"]:
                ann_counts[(ann["sample_id"], ann["chip_id"])] += 1
        # If a sample has annotation records but none confirmed, the confirmed count is 0 for its populated tiles.
        annotated_tiles = {(ann["sample_id"], ann["chip_id"]) for ann in reviewed_defects}
        for chip in chip_rows:
            key = (chip["sample_id"], chip["chip_id"])
            if key in annotated_tiles:
                chip["confirmed_defect_count"] = ann_counts[key]
                chip["defect_assessment_status"] = "EXPERT_ANNOTATIONS_PROVIDED_FOR_TILE"

    _write_csv(out_dir / "inspection_sites.csv", site_rows, SITE_COLUMNS)
    _write_csv(out_dir / "chip_summary.csv", chip_rows, CHIP_COLUMNS)
    _write_csv(out_dir / "anomaly_candidates.csv", anomaly_rows, ANOMALY_COLUMNS)
    _write_csv(out_dir / "anomaly_candidates_all.csv", anomaly_rows_all, ANOMALY_COLUMNS)
    sample_overlay = out_dir / "sample_inspection_overlay.jpg"
    # Prefer a representative non-zero-error example so the prediction and GT markers
    # are visibly distinct, similar to a useful inspection review screenshot.
    def _error_value(row: dict[str, Any]) -> float:
        try:
            return float(row.get("localization_error_px", ""))
        except (TypeError, ValueError):
            return -1.0
    candidate_examples = [r for r in site_rows if int(r.get("anomaly_candidates_retained", 0) or 0) > 0]
    example_pool = candidate_examples or site_rows
    example_site = max(example_pool, key=lambda r: (_error_value(r), int(r.get("anomaly_candidate_count", 0) or 0))) if example_pool else None
    example_overlay = Path(example_site["annotation_image"]) if example_site else None
    if example_overlay is not None and not example_overlay.is_absolute():
        example_overlay = (root / example_overlay).resolve()
    if example_overlay is not None and example_overlay.is_file():
        import shutil
        shutil.copy2(example_overlay, sample_overlay)

    severity_counts = Counter(r["severity"] for r in anomaly_rows_all)
    report = {
        "project": "NOVA-Loc Research Inspection Report",
        "generated_at_unix": time.time(),
        "root": str(root), "output_dir": str(out_dir),
        "sample_count": len(site_rows), "chip_tile_count": len(chip_rows),
        "anomaly_candidate_count": len(anomaly_rows_all),
        "shortlist_candidate_count": len(anomaly_rows),
        "severity_candidate_counts": dict(severity_counts),
        "site_table": "inspection_sites.csv", "chip_table": "chip_summary.csv",
        "anomaly_table": "anomaly_candidates.csv", "all_anomaly_candidates_table": "anomaly_candidates_all.csv",
        "sample_overlay": "sample_inspection_overlay.jpg",
        "sample_overlay_example_sample_id": (example_site["sample_id"] if example_site else None),
        "sample_overlay_example_localization_error_px": (_error_value(example_site) if example_site else None),
        "reviewed_defects_table": "reviewed_defect_annotations.csv",
        "expert_defect_annotations_loaded": len(reviewed_defects),
        "confirmed_expert_defect_annotations": sum(bool(a["confirmed"]) for a in reviewed_defects),
        "visual_dir": _path_label(visuals, root),
        "coordinate_semantics": {
            "image_coordinates": "x/y pixel-center coordinates in the original search image, origin at top-left",
            "chip_local_coordinates": "image coordinates minus the configured rectangular chip/tile top-left origin",
            "default_chip_layout": "one inspection image equals one chip/field because chip boundaries were not supplied",
            "wafer_coordinates": "not computed; wafer registration and calibrated physical scale are not supplied",
        },
        "defect_semantics": {
            "confirmed_defect_count": "not available; there are no confirmed defect labels or trained/validated defect classifier in the supplied project",
            "anomaly_candidates": "unclassified local high-pass intensity outliers; heuristic triage only, may include normal repetitive structures or capture noise",
            "defect_type": "UNCLASSIFIED",
        },
        "localization_semantics": "template/reference localization only; a target match is not a defect diagnosis",
        "ground_truth_rows": sum(1 for r in site_rows if r["ground_truth_x_image_px"] != "" and r["ground_truth_y_image_px"] != ""),
        "ground_truth_unavailable_rows": sum(1 for r in site_rows if r["ground_truth_x_image_px"] == "" or r["ground_truth_y_image_px"] == ""),
        "parameters": {"input_manifest": str(input_manifest) if input_manifest else None,
                       "defect_annotations_csv": str(defect_annotations_csv) if defect_annotations_csv else None,
                       "chip_width_px": chip_width_px, "chip_height_px": chip_height_px,
                       "scan_anomalies": scan_anomalies, "threshold_sigma": threshold_sigma,
                       "min_area": min_area, "max_area": max_area,
                       "max_candidates_per_image": max_candidates_per_image,
                       "annotate_anomalies": annotate_anomalies,
                       "prediction_source": str(pred_path) if pred_path else "fresh locate() inference",
                       "split": split, "method": method},
        "runtime_seconds": time.perf_counter() - started,
    }
    (out_dir / "inspection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary_lines = [
        "# NOVA-Loc Inspection Report", "",
        f"- Inspection images/sites: **{len(site_rows)}**",
        f"- Chip/tile table rows: **{len(chip_rows)}**",
        f"- All thresholded heuristic anomaly candidates: **{len(anomaly_rows_all)}** (unconfirmed; not a defect count)",
        f"- Shortlist candidates retained for easy review: **{len(anomaly_rows)}**.",
        f"- Raw thresholded components before per-image cap: **{sum(r['anomaly_candidates_before_cap'] for r in site_rows)}**; candidates omitted by cap: **{sum(r['anomaly_candidates_omitted_by_cap'] for r in site_rows)}**.",
        f"- Candidate severity tags: {dict(severity_counts)}",
        f"- Rows with ground truth coordinates: **{sum(1 for r in site_rows if r['ground_truth_x_image_px'] != '' and r['ground_truth_y_image_px'] != '')}**; rows without ground truth: **{sum(1 for r in site_rows if r['ground_truth_x_image_px'] == '' or r['ground_truth_y_image_px'] == '')}**.",
        "- Coordinates are search-image pixel coordinates and chip-local pixel coordinates. Optional lot/wafer/die metadata is carried from the input manifest; missing values remain NOT_PROVIDED.",
        "- Default geometry treats each search image as one chip/inspection field. Provide chip width/height only when actual tile geometry is known.",
        "- The package lacks confirmed defect labels and a validated defect classifier. Candidate records are unclassified outliers for review, not confirmed physical defects. `anomaly_candidates.csv` is a top-K shortlist; `anomaly_candidates_all.csv` retains all thresholded components.",
        "- No wafer-space/physical coordinates are emitted because wafer registration and pixel-to-length calibration are absent.", "",
        "## Output files", "",
        "- `inspection_sites.csv`: localized target, image-global and chip-local coordinates, localization error (where ground truth is available), NCC score, scale and gate diagnostics. It is one row per reference/search pair.",
        "- `chip_summary.csv`: one row per chip/tile, localized-site fields and anomaly-candidate summary.",
        "- `anomaly_candidates.csv`: shortlist of unclassified high-pass candidates with coordinates, bounding box, contrast type and review action (top 20 per image by default).",
        "- `anomaly_candidates_all.csv`: all retained thresholded components, before the top-K shortlist cap.",
        "- `anomaly_candidates_all.csv`: every thresholded candidate component with image-global and chip-local coordinates (can be noisy because this is not a validated defect detector).",
        "- `reviewed_defect_annotations.csv`: optional user/expert-supplied defect labels mapped to image-global and chip-local coordinates; empty when no annotation file is supplied.",
        "- `visuals/<architecture>/<sample>_inspection.jpg`: compact reference/search visualization with red prediction, green ground truth, and orange top anomaly candidates.",
        "- `sample_inspection_overlay.jpg`: representative overlay with the largest available ground-truth localization error so the prediction/GT markers and displayed error can be reviewed.",
    ]
    (out_dir / "INSPECTION_REPORT.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return report