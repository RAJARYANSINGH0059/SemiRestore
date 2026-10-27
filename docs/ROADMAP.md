# Research roadmap and exit criteria

## Phase 1 — Reproducible classical baselines (included)

Fixed 10× NCC, seven-scale NCC, source center-aware NCC, generator, ground-truth CSVs, fixed-seed inference CLI and summary benchmark.

## Phase 2 — Drift-aware candidate generation (partially included)

Bounded center-relative ROI, explicit center candidate and late score-gap gate. Current translated holdout result is only 51.7% within 5 pixels over shifts up to ±50 pixels; do not mark phase complete for arbitrary drift.

## Phase 3 — Verification and ML ranking (experimental)

Gradient correlation, FFT periodicity features and a Random Forest candidate ranker were tested. The current experimental variants did not reliably improve top-1 localization, so the default CLI does not use them. Future acceptance requires held-out improvement without unacceptable p95 runtime or worst-case error.

## Phase 4 — Inspection reporting (included, heuristic only)

Annotated match output, site coordinate table, chip/tile summary, anomaly candidate shortlist/full table, optional expert defect annotations. The heuristic outlier scan is not a defect classifier; confirmed defect labels are input to the report, not inferred by a trained model.

## Phase 5 — Real SEM/OOD evaluation (blocked on data)

Acquire licensed real SEM sources, retain source citations/terms, obtain expert localization labels and preferably defect masks, make wafer/source-level train/validation/OOD partitions, and evaluate false positives plus localization metrics.

## Phase 6 — Physical coordinate calibration (not included)

Require a trustworthy tile/die map, orientation/registration, tool-stage metadata and calibrated pixel-to-physical conversion before writing wafer/stage/micrometre/nanometre coordinates.