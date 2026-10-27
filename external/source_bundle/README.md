# NOVA-Loc

**N**oise-robust, **O**OD-aware, **V**erification-guided, **A**daptive localization for periodic semiconductor structures.

This repository is the reproducible prototype for the SEMICON India Hackathon 2026 – Drift-Sense problem statement. The current code implements the **data generator + classical baselines + evaluator-facing inference contract** first. The next prototype layer is the NOVA-Loc periodicity/context verifier.

## Official contract implemented

- `generator.py` accepts architecture (`DRAM`/`FinFET`), number of pairs, and output directory.
- Ground-truth center coordinates are written to `ground_truth.csv`.
- `infer.py` accepts reference-image path and search-image path and prints a single `(x, y)` center.
- 1000×1000 search images and approximately 10×-zoomed reference images are generated.
- Noise is sampled independently for reference and search captures.

The official challenge requires the same generator and inference interfaces and will score the inference script on independently generated, noisier DRAM/FinFET data. See `docs/proposal.md` for the design rationale.

## Quick start

```bash
pip install -r requirements.txt
python src/generator.py --architecture DRAM --num-pairs 30 --output-dir results/dram
python src/generator.py --architecture FinFET --num-pairs 30 --output-dir results/finfet
python src/baseline.py --gt results/dram/ground_truth.csv --mode fixed
python src/baseline.py --gt results/dram/ground_truth.csv --mode multiscale
python src/infer.py results/dram/reference/dram_0000.png results/dram/search/dram_0000.png
```

## Phase-2 prototype roadmap

1. Fixed 10× NCC baseline.
2. Multi-scale NCC baseline.
3. Context-ring candidate verification.
4. FFT/autocorrelation periodicity analysis.
5. Candidate uniqueness / ambiguity gate.
6. Hard-negative training for a lightweight ranker.
7. Subpixel refinement.
8. OOD stress suite and latency benchmark.

## Directory layout

```text
NOVA-Loc/
├── docs/proposal.md
├── src/generator.py
├── src/baseline.py
├── src/infer.py
├── src/experiments.py
├── requirements.txt
└── results/
```