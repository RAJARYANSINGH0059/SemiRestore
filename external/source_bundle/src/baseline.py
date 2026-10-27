#!/usr/bin/env python3
"""Baselines for NOVA-Loc.

B1: Fixed 10x resize + normalized cross-correlation (NCC).
B2: Multi-scale NCC over a narrow scale band around the expected 10x ratio.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def load_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def ncc(search: np.ndarray, template: np.ndarray) -> Tuple[float, Tuple[float, float]]:
    res = cv2.matchTemplate(search.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    h, w = template.shape
    cx = maxloc[0] + (w - 1) / 2.0
    cy = maxloc[1] + (h - 1) / 2.0
    return float(maxv), (float(cx), float(cy))


def baseline_fixed(ref_path: str | Path, search_path: str | Path) -> Tuple[float, Tuple[float, float], float]:
    ref = load_gray(ref_path)
    search = load_gray(search_path)
    t0 = time.perf_counter()
    h = max(8, int(round(ref.shape[0] / 10.0)))
    w = max(8, int(round(ref.shape[1] / 10.0)))
    template = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
    score, xy = ncc(search, template)
    return score, xy, time.perf_counter() - t0


def baseline_multiscale(ref_path: str | Path, search_path: str | Path,
                        ratios: np.ndarray | None = None) -> Tuple[float, Tuple[float, float], float, float]:
    ref = load_gray(ref_path)
    search = load_gray(search_path)
    t0 = time.perf_counter()
    if ratios is None:
        ratios = np.linspace(0.085, 0.115, 7)
    best = (-1.0, (0.0, 0.0), 0.10)
    for ratio in ratios:
        w = max(8, int(round(ref.shape[1] * float(ratio))))
        h = max(8, int(round(ref.shape[0] * float(ratio))))
        if h >= search.shape[0] or w >= search.shape[1]:
            continue
        template = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
        score, xy = ncc(search, template)
        if score > best[0]:
            best = (score, xy, float(ratio))
    return best[0], best[1], time.perf_counter() - t0, best[2]



def baseline_center_aware(ref_path: str | Path, search_path: str | Path,
                          ratios: np.ndarray | None = None, topk: int = 50,
                          score_window: float = 0.02) -> Tuple[float, Tuple[float, float], float, float, int]:
    """Multi-scale NCC plus the challenge's center-nearest tie-break rule."""
    ref = load_gray(ref_path)
    search = load_gray(search_path)
    t0 = time.perf_counter()
    if ratios is None:
        ratios = np.linspace(0.085, 0.115, 7)
    cands = []
    center = np.array([(search.shape[1]-1)/2.0, (search.shape[0]-1)/2.0])
    for ratio in ratios:
        w = max(8, int(round(ref.shape[1] * float(ratio))))
        h = max(8, int(round(ref.shape[0] * float(ratio))))
        if h >= search.shape[0] or w >= search.shape[1]:
            continue
        template = cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        flat = res.ravel()
        k = min(topk, flat.size)
        idxs = np.argpartition(flat, -k)[-k:]
        max_score = float(flat[idxs].max())
        for idx in idxs:
            score = float(flat[idx])
            if score < max_score - score_window:
                continue
            yy, xx = np.unravel_index(int(idx), res.shape)
            xy = np.array([xx + (w-1)/2.0, yy + (h-1)/2.0])
            dist = float(np.linalg.norm(xy-center))
            cands.append((score, dist, tuple(xy), float(ratio)))
    if not cands:
        score, xy, dt, r = baseline_multiscale(ref_path, search_path, ratios)
        return score, xy, dt, r, 0
    best_score = max(c[0] for c in cands)
    eligible = [c for c in cands if c[0] >= best_score - score_window]
    chosen = min(eligible, key=lambda c: (c[1], -c[0]))
    return chosen[0], chosen[2], time.perf_counter()-t0, chosen[3], len(eligible)

def evaluate(gt_csv: str | Path, mode: str, tolerance: float = 5.0) -> dict:
    rows = list(csv.DictReader(Path(gt_csv).open()))
    errors = []
    latencies = []
    correct = 0
    for r in rows:
        if mode == "fixed":
            score, pred, dt = baseline_fixed(r["reference"], r["search"])
        elif mode == "center":
            score, pred, dt, _, _ = baseline_center_aware(r["reference"], r["search"])
        else:
            score, pred, dt, _ = baseline_multiscale(r["reference"], r["search"])
        gt = (float(r["x"]), float(r["y"]))
        err = float(np.hypot(pred[0] - gt[0], pred[1] - gt[1]))
        errors.append(err)
        latencies.append(dt)
        correct += int(err <= tolerance)
    return {
        "n": len(rows),
        "accuracy_within_tolerance": correct / max(1, len(rows)),
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(np.max(errors)),
        "mean_latency_ms": 1000.0 * float(np.mean(latencies)),
        "p95_latency_ms": 1000.0 * float(np.percentile(latencies, 95)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--mode", choices=["fixed", "multiscale"], default="multiscale")
    p.add_argument("--tolerance", type=float, default=5.0)
    args = p.parse_args()
    out = evaluate(args.gt, args.mode, args.tolerance)
    for k, v in out.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()