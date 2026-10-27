#!/usr/bin/env python3
"""Reproducible baseline experiment runner."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from generator import generate  # noqa: E402
from baseline import evaluate  # noqa: E402


def run(output_dir: Path, pairs: int, tolerance: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arch in ["DRAM", "FinFET"]:
        arch_dir = output_dir / arch
        gt = generate(arch, pairs, arch_dir)
        results[arch] = {
            "fixed_ncc": evaluate(gt, "fixed", tolerance),
            "multiscale_ncc": evaluate(gt, "multiscale", tolerance),
            "center_aware_ncc": evaluate(gt, "center", tolerance),
        }
    (output_dir / "baseline_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, default=ROOT / "results" / "baseline")
    p.add_argument("--pairs", type=int, default=30)
    p.add_argument("--tolerance", type=float, default=5.0)
    args = p.parse_args()
    run(args.output_dir, args.pairs, args.tolerance)


if __name__ == "__main__":
    main()