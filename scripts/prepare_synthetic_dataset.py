#!/usr/bin/env python3
"""Ensure the fixed-seed synthetic DRAM/FinFET reference/search pairs exist."""
from __future__ import annotations
import csv, shutil, tempfile, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.generator import generate
SEED = 20260816

def main():
    for arch in ("DRAM", "FinFET"):
        base = ROOT / "results" / "baseline" / arch
        gt = base / "ground_truth.csv"
        if not gt.exists():
            generate(arch, 30, base, seed=SEED)
            print(f"Generated {arch} ground truth and images under {base}")
            continue
        with gt.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 30:
            raise RuntimeError(f"Expected 30 {arch} samples in {gt}; found {len(rows)}")
        missing=[]
        for row in rows:
            for key in ("reference", "search"):
                p=Path(row[key]); p = p if p.is_absolute() else ROOT/p
                if not p.exists(): missing.append((p, Path(key)/p.name))
        if not missing:
            print(f"{arch}: all {len(rows)} pairs already exist")
            continue
        with tempfile.TemporaryDirectory(prefix=f"semirestore_{arch.lower()}_") as tmp:
            generated=Path(tmp)/arch
            generate(arch, 30, generated, seed=SEED)
            for row in rows:
                for key, sub in (("reference","reference"),("search","search")):
                    dest=Path(row[key]); dest=dest if dest.is_absolute() else ROOT/dest
                    if dest.exists(): continue
                    src=generated/sub/dest.name
                    if not src.exists(): raise FileNotFoundError(src)
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(src,dest)
        print(f"{arch}: regenerated {len(missing)} missing image files; preserved existing files")
if __name__ == "__main__": main()