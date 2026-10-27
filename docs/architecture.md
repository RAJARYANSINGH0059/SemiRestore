# Architecture and Data Flow

1. **Input resolution** — explicit reference and search paths are required. The CLI never samples another image silently.
2. **Scale-aware coarse matching** — reference resized across configured scale ratios; NCC maps generate candidates.
3. **Candidate selection** — current hybrid searches a bounded center-relative ROI and explicitly scores the center candidate. A late score-window gate may prefer the center when the top candidates are close.
4. **Output** — evaluator CLI emits `x y`; diagnostics CLI adds JSON; the notebook/report layer adds input provenance, ground truth when known, error, chip-local mapping and candidate tables.
5. **Research-only verifier** — gradient/FFT/Random Forest candidate features are evaluated in experiments but are not selected as default because they failed to improve held-out top-1 results reliably.
6. **Anomaly review** — local contrast/texture outliers are stored as unconfirmed candidates. Expert labels, if supplied, live in a separate table.
7. **Reproducibility** — generated images use fixed seeds; benchmark CSV records method/condition/sample-level predictions, ground truth, transform and error.

## Non-goals

No frontend, HTTP service, database, persistent model-serving endpoint, wafer-stage controller, physical calibration or validated defect classifier is included.