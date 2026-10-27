#!/usr/bin/env python3
"""Standalone evaluator-facing inference script.

Usage:
    python infer.py reference.png search.png

Outputs exactly one final center coordinate: x y
"""
from __future__ import annotations

import sys
from pathlib import Path
import cv2
import numpy as np

from baseline import baseline_center_aware


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python infer.py <reference.png> <search.png>", file=sys.stderr)
        return 2
    ref, search = sys.argv[1], sys.argv[2]
    _, (x, y), _, _, _ = baseline_center_aware(ref, search)
    print(f"{x:.3f} {y:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())