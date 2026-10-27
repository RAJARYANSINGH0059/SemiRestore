# Experimental Protocol

## A. Baselines

B1 — fixed 10× NCC.

B2 — 7-point multi-scale NCC around the nominal 10× ratio.

B3 — center-aware multi-scale NCC using the official center-nearest rule only during final tie/ambiguity resolution.

## B. Proposed ablations

P1 — B3 + gradient similarity.
P2 — P1 + context-ring similarity.
P3 — P2 + periodicity/FFT descriptor.
P4 — P3 + ambiguity gate.
P5 — P4 + lightweight candidate ranker.
P6 — P5 + subpixel refinement.

## C. Dataset partitions

- 70% training-generation settings.
- 15% validation settings.
- 15% strict OOD settings with held-out noise/blur/pitch/rotation ranges.
- Separate periodic-stress suite for final robustness analysis.

## D. Metrics

1. Accuracy within 1, 2, 5, 10 px.
2. Mean, median, p95 and worst-case localization error.
3. DRAM / FinFET breakdown.
4. Periodicity-stress accuracy.
5. OOD accuracy and `R_OOD = Acc_OOD / Acc_ID`.
6. Mean and p95 runtime.
7. Model size and peak RAM/VRAM if a learned module is added.

## E. Statistical discipline

Use fixed seeds for benchmark generation, keep the final test seed set hidden from model-selection experiments, and report the full per-sample result file. Do not tune thresholds against the final holdout suite.

## F. Success criterion

The final model is not considered improved unless an ablation increases either robust accuracy or worst-case behavior without causing an unacceptable runtime increase.