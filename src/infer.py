#!/usr/bin/env python3
"""Evaluator-facing NOVA-Loc inference. Prints exactly: x y"""
from __future__ import annotations
import sys
try:
    from nova_inference import locate
except ImportError:  # when imported as src.infer
    from .nova_inference import locate

def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python src/infer.py <reference.png> <search.png>", file=sys.stderr)
        return 2
    result = locate(sys.argv[1], sys.argv[2])
    # Preserve challenge contract: stdout is exactly one coordinate pair.
    print(f"{result.x:.3f} {result.y:.3f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())