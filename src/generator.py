#!/usr/bin/env python3
"""NOVA-Loc synthetic paired-image generator.

Creates 1000x1000 search images and ~100x100 reference images (10x linear zoom).
The generator is deliberately synthetic/SEM-inspired rather than a claim of a
full physical SEM simulator.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

SEARCH_SIZE = 1000
DEFAULT_REF = 100


def rng_from_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gaussian_kernel_sigma(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def add_edge_brightening(img: np.ndarray, strength: float = 0.16) -> np.ndarray:
    """SEM-inspired edge brightening using gradient magnitude.

    This is a compact image-formation approximation, not a physics-accurate SEM model.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    if mag.max() > 0:
        mag = mag / mag.max()
    out = img + strength * mag
    return np.clip(out, 0.0, 1.0)


def add_independent_noise(img: np.ndarray, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Independent Gaussian + signal-dependent Poisson-like noise."""
    # Gaussian component.
    out = img + rng.normal(0.0, sigma, img.shape).astype(np.float32)
    # Signal-dependent Poisson-like component. Keep counts moderate to avoid huge spikes.
    photons = rng.uniform(20_000, 80_000)
    counts = np.clip(out, 0, 1) * photons
    poisson = rng.poisson(counts).astype(np.float32) / photons
    # Blend a small fraction of shot-noise fluctuation around the local mean.
    out += 0.65 * (poisson - np.clip(out, 0, 1))
    return np.clip(out, 0.0, 1.0)


def make_unit_cell(arch: str, size: int, rng: np.random.Generator) -> np.ndarray:
    """Create a simple architecture-shaped grayscale unit cell."""
    canvas = np.zeros((size, size), np.float32)
    if arch == "DRAM":
        # Word-line / bit-line style orthogonal periodic structure.
        pitch_x = max(12, int(size * rng.uniform(0.18, 0.28)))
        pitch_y = max(12, int(size * rng.uniform(0.18, 0.28)))
        width = max(2, int(size * rng.uniform(0.025, 0.06)))
        for x in range(width // 2, size, pitch_x):
            canvas[:, max(0, x-width//2):min(size, x+width//2+1)] = 0.75
        for y in range(width // 2, size, pitch_y):
            canvas[max(0, y-width//2):min(size, y+width//2+1), :] = np.maximum(
                canvas[max(0, y-width//2):min(size, y+width//2+1), :], 0.58
            )
        # Contacts at intersections.
        for y in range(pitch_y // 2, size, pitch_y):
            for x in range(pitch_x // 2, size, pitch_x):
                rr = max(2, int(size * rng.uniform(0.015, 0.03)))
                yy, xx = np.ogrid[:size, :size]
                disk = (xx-x) ** 2 + (yy-y) ** 2 <= rr * rr
                canvas[disk] = 0.92
    elif arch == "FinFET":
        pitch = max(10, int(size * rng.uniform(0.12, 0.20)))
        fin_width = max(2, int(size * rng.uniform(0.025, 0.05)))
        gate_width = max(2, int(size * rng.uniform(0.03, 0.055)))
        for x in range(fin_width // 2, size, pitch):
            canvas[:, max(0, x-fin_width//2):min(size, x+fin_width//2+1)] = 0.78
        # A second direction gives gate crossings.
        gate_pitch = max(20, int(size * rng.uniform(0.34, 0.50)))
        for y in range(gate_pitch // 2, size, gate_pitch):
            canvas[max(0, y-gate_width//2):min(size, y+gate_width//2+1), :] = np.maximum(
                canvas[max(0, y-gate_width//2):min(size, y+gate_width//2+1), :], 0.9
            )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return canvas


def generate_search_base(arch: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a 1000x1000 repeating field with architecture-specific motifs."""
    block = int(rng.integers(120, 190))
    cell = make_unit_cell(arch, block, rng)
    reps = math.ceil(SEARCH_SIZE / block) + 1
    tiled = np.tile(cell, (reps, reps))[:SEARCH_SIZE, :SEARCH_SIZE]

    # Add a lower-frequency background and gentle brightness field.
    yy, xx = np.mgrid[:SEARCH_SIZE, :SEARCH_SIZE]
    bg = 0.08 + 0.04 * np.sin(xx / rng.uniform(130, 220)) + 0.03 * np.cos(yy / rng.uniform(140, 240))
    base = 0.18 + 0.70 * tiled + bg
    base = np.clip(base, 0, 1).astype(np.float32)
    return base


def place_target(search: np.ndarray, rng: np.random.Generator, architecture: str) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    """Create reference and search target from a randomly selected local patch."""
    ref_size = int(rng.integers(260, 361))
    patch_size = max(18, int(round(ref_size / 10.0)))
    margin = patch_size + 6
    # Keep the true site near the search-image center so the challenge rule
    # “closest matching region to the center” can be exercised when periodic
    # copies create multiple near-equivalent matches.
    cx = SEARCH_SIZE // 2
    cy = SEARCH_SIZE // 2

    # Add a small site-specific structural signature before capture. Most of the
    # surrounding field remains periodic, but the target site has a local
    # asymmetric feature so the benchmark contains both solvable and ambiguous cases.
    y0, x0 = cy - patch_size//2, cx - patch_size//2
    sig = max(2, patch_size // 7)
    strength = float(rng.uniform(0.05, 0.24))
    search[max(0, y0+2):min(SEARCH_SIZE, y0+2+sig),
           max(0, x0+patch_size//2):min(SEARCH_SIZE, x0+patch_size//2+sig//2+1)] = np.clip(
        search[max(0, y0+2):min(SEARCH_SIZE, y0+2+sig),
               max(0, x0+patch_size//2):min(SEARCH_SIZE, x0+patch_size//2+sig//2+1)] + strength, 0, 1)
    patch = search[cy-patch_size//2:cy-patch_size//2+patch_size,
                   cx-patch_size//2:cx-patch_size//2+patch_size].copy()
    # Reference has a higher apparent scale, lower noise, and independent capture conditions.
    ref = cv2.resize(patch, (ref_size, ref_size), interpolation=cv2.INTER_CUBIC)

    # Small independent reference-side transformations.
    angle = float(rng.uniform(-3.0, 3.0))
    scale = float(rng.uniform(0.97, 1.03))
    M = cv2.getRotationMatrix2D((ref_size / 2, ref_size / 2), angle, scale)
    ref = cv2.warpAffine(ref, M, (ref_size, ref_size), borderMode=cv2.BORDER_REFLECT)
    ref = cv2.GaussianBlur(ref, (0, 0), sigmaX=gaussian_kernel_sigma(rng, 0.3, 0.9))
    ref = add_edge_brightening(ref, strength=float(rng.uniform(0.08, 0.18)))
    ref = add_independent_noise(ref, rng, sigma=float(rng.uniform(0.012, 0.028)))

    # Search-side independent degradation; deliberately stronger than reference.
    search_out = search.copy()
    search_out = cv2.GaussianBlur(search_out, (0, 0), sigmaX=gaussian_kernel_sigma(rng, 0.45, 1.4))
    search_out = add_edge_brightening(search_out, strength=float(rng.uniform(0.10, 0.22)))
    search_out = add_independent_noise(search_out, rng, sigma=float(rng.uniform(0.025, 0.075)))

    # Add subtle global contrast/illumination variation.
    gain = float(rng.uniform(0.88, 1.10))
    bias = float(rng.uniform(-0.04, 0.04))
    search_out = np.clip(gain * search_out + bias, 0, 1)
    ref = np.clip(float(rng.uniform(0.95, 1.05)) * ref + float(rng.uniform(-0.015, 0.015)), 0, 1)

    return ref.astype(np.float32), search_out.astype(np.float32), (float(cx), float(cy))


def save_u8(path: Path, img: np.ndarray) -> None:
    Image.fromarray(np.clip(img * 255.0, 0, 255).astype(np.uint8), mode="L").save(path)


def generate(architecture: str, num_pairs: int, output_dir: Path, seed: int = 20260816) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = output_dir / "reference"
    searches = output_dir / "search"
    refs.mkdir(exist_ok=True)
    searches.mkdir(exist_ok=True)
    rng = rng_from_seed(seed + (0 if architecture == "DRAM" else 10_000))

    gt_path = output_dir / "ground_truth.csv"
    with gt_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "architecture", "reference", "search", "x", "y", "reference_size", "search_size", "seed"])
        for i in range(num_pairs):
            pair_seed = int(seed + i * 7919 + (0 if architecture == "DRAM" else 1000003))
            pair_rng = rng_from_seed(pair_seed)
            base = generate_search_base(architecture, pair_rng)
            ref, search, (cx, cy) = place_target(base, pair_rng, architecture)
            ident = f"{architecture.lower()}_{i:04d}"
            rp = refs / f"{ident}.png"
            sp = searches / f"{ident}.png"
            save_u8(rp, ref)
            save_u8(sp, search)
            writer.writerow([ident, architecture, str(rp), str(sp), f"{cx:.3f}", f"{cy:.3f}", ref.shape[0], SEARCH_SIZE, pair_seed])
    return gt_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--architecture", choices=["DRAM", "FinFET"], required=True)
    p.add_argument("--num-pairs", type=int, default=30)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260816)
    args = p.parse_args()
    print(generate(args.architecture, args.num_pairs, args.output_dir, args.seed))


if __name__ == "__main__":
    main()