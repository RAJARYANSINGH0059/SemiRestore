# SemiRestore / NOVA-Loc Source-Integrated Engineering and Test Report

**Report date:** 2026-10-27  
**Scope:** user-supplied `NOVA-Loc-source.zip`, `NOVA-Loc-prototype.zip`, `NOVA-Loc-unified-dataset-factory.zip`, `RESULTS.md`, `experimental_protocol.md`, `math.md`, and the earlier notebook/presentation/workflow materials.  
**Run environment:** Python 3.13.5, Linux x86_64, OpenCV 4.x, NumPy 2.x, scikit-learn/joblib available.  
**Final integrated state:** all 60 bundled synthetic holdout pairs evaluated in centered and translated conditions; 480 per-sample predictions; 10 automated integration tests passed; integrated notebook executed top-to-bottom with 23 total cells / 12 code cells and no notebook errors. The run freshly regenerated all 480 holdout predictions and executed a separate-seed ML/structure validation experiment.

## 1. Executive summary

The user-provided ZIPs resolve the key source-code and synthetic-test-data gap from the earlier work. This package now contains the actual NOVA-Loc baseline source, generator, evaluator-facing input/output contract, generated DRAM and FinFET image pairs with their `ground_truth.csv` files, and the supplied unified dataset-factory scripts. The previous standalone reconstruction is no longer the primary entrypoint: the final evaluator command is `python src/infer.py <reference.png> <search.png>`, and the notebook imports the source-integrated modules.

The integrated work preserves B1/B2/B3 baselines, adds a bounded-ROI multi-scale NCC method with a late center ambiguity gate, retains the previous inference implementation in `src/infer_baseline.py`, validates the output contract, and measures all methods against the bundled holdout set and a deterministic translation stress test.

**Main result:** the hybrid reaches 100.0% within 5 px on the bundled centered synthetic test set, but only 51.7% under translated stress up to ±50 px. The centered score is strongly aided by the generator's construction: it places the target at the center by design. The shifted stress test demonstrates that drift handling remains incomplete. Candidate-level ML ranking and gradient/periodicity ablations were tested on separate generated validation data and did not improve localization reliably, so they are retained as experimental evidence and are not enabled by default.

These results do **not** establish real-SEM performance, official competition performance, nanometre accuracy, or production readiness.

## 2. Materials inventory and provenance

| Supplied artifact | Integrated use | Status |
|---|---|---|
| `NOVA-Loc-source.zip` | Baseline/generator/inference source baseline, source docs and prior result summaries | Inspected and incorporated. Original `src/infer.py` behavior preserved as `src/infer_baseline.py` before adapting the evaluator entrypoint. |
| `NOVA-Loc-prototype.zip` | Same source plus 30 DRAM and 30 FinFET reference/search PNG pairs and ground-truth CSVs | Inspected and incorporated; these are generated synthetic SEM-inspired images, not acquired wafer images. |
| `NOVA-Loc-unified-dataset-factory.zip` | Manifest, pair generator, split maker, source config and source documentation | Incorporated under `external/unified_dataset_factory/`. Source images from the listed public datasets are not included and need to be acquired under their own terms. |
| `RESULTS.md` | Preliminary source-supplied baseline report | Preserved as `results/RESULTS_user_supplied.md`; values reproduced on the supplied prototype pair set. |
| `experimental_protocol.md` | Baseline/ablation/split/metric plan | Preserved under `docs/experimental_protocol.md`. |
| `math.md` | Coordinate, scoring and error definitions | Preserved under `docs/math.md`. |
| Earlier supplied `NOVA.ipynb` | Historical Colab archive-extraction/orchestration notebook | Preserved byte-for-byte in `source_materials/NOVA_original_backup.ipynb`. It originally required a separate Drive archive; the new integrated notebook runs from this delivered project. |

### Source scope boundary

The supplied source's own README frames the existing code as the first phase: synthetic generator, classical baselines, metrics/evaluator interface. The complete proposed context/FFT/ML verifier is a roadmap, not an original fully implemented component. The user-provided source did not include real wafer acquisitions, native SEM stage-coordinate labels, physical pixel calibration, or an official hidden evaluation set.

## 3. Requirements-to-code mapping

| Requirement | Code/artifact | Status/evidence |
|---|---|---|
| Deterministic, explicit image pair inputs | `src/infer.py`, `src/nova_inference.py` | Implemented. Requires reference and search paths explicitly; it does not select a random dataset pair. |
| Preserve the previous evaluator baseline | `src/infer_baseline.py` | Preserved before replacing the current `src/infer.py` entrypoint with the hybrid. |
| Fixed-scale NCC baseline | `src/baseline.py::baseline_fixed` | Existing baseline, measured over the same 60 pairs. |
| Seven-scale NCC baseline | `src/baseline.py::baseline_multiscale` | Existing baseline, measured over the same 60 pairs. |
| Source center-aware Top-K baseline | `src/baseline.py::baseline_center_aware` | Existing baseline, measured over the same 60 pairs. |
| Bounded drift-region search | `src/nova_inference.py::locate` | Implemented with ROI radius default 50 pixels, seven scale ratios around 0.1, and explicit ROI candidate scoring. |
| Center candidate explicitly scored | `src/nova_inference.py::locate` | Implemented per scale; center prior is used only through the documented score-gap gate. |
| Configurable ambiguity gate | `src/nova_inference.py::locate` | Default score window 0.05 NCC units, selected on separate generated validation data. This is not a probability. |
| Gradient/FFT/ML candidate verifier investigation | `src/verification.py`, `results/ablation/` | Experiment executed; it did not reliably improve top-1 localization under validation/shift stress. Not enabled by the default evaluator entrypoint. |
| True reference-context-ring matching | — | Not implemented. The supplied interface has a crop-only reference, no larger reference-neighborhood/context image to compare against; no artificial score is claimed. |
| Sub-pixel physical accuracy | — | No physical accuracy claim. Correlation response returns pixel-center estimates; no SEM calibration or physical ground truth was supplied. |
| Reproducible benchmark and detailed per-pair records | `scripts/run_integrated_benchmark.py`, `results/integrated_benchmark/integrated_per_sample_results.csv` | Complete 480-prediction ledger: 60 pairs × 4 methods × 2 conditions. Image hashes, transformations, coordinates, errors and latency are recorded. |
| Automated regression tests | `tests/test_source_pipeline.py` | 8 tests passed in the final run. |
| Runnable integrated notebook | `notebooks/SemiRestore_Research.ipynb` | Latest run: 25 cells / 13 code cells, zero notebook errors. Uses saved benchmark/ablation evidence by default; live single-pair inference, inspection report, coordinate tables, CLI checks and regression tests execute. Full 480-prediction recomputation and ranker retraining are opt-in switches. |

## 4. Original inconsistency issue

The earlier `NOVA.ipynb` is a Colab orchestration notebook and includes a runner that can randomly select a pair using `random.SystemRandom().choice(...)` when launched in random mode. That is a confirmed cause of different input images between those invocations. It also displays a most-recent image artifact, which can make stale or unrelated output look like the current result, and its upload cell had an empty-list indexing failure in the saved output.

The source-integrated final evaluator does not randomly choose files: `src/infer.py` requires both explicit file paths and prints only the coordinate pair. The benchmark ledger records the path names and source/processed image SHA-256 hashes. Repeated execution on the same pair in the same environment was tested to return the same stable coordinates/score/scale/status (timings are naturally variable).

## 5. Full held-out benchmark

The baseline methods and hybrid were evaluated on all 60 bundled pairs (30 DRAM + 30 FinFET). A second deterministic condition translated the search image up to ±50 px with seed `930091` and shifted the known coordinates by the same offsets. The CSV has 480 predictions (four methods × 60 images × two conditions).

### Centered holdout — target generated at image center

| Method | Arch | n | ≤1 px | ≤2 px | ≤5 px | ≤10 px | Mean error px | P95 error px | Mean latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 fixed NCC | ALL | 60 | 13.3% | 13.3% | 13.3% | 13.3% | 331.44 | 600.00 | 36.8 |
| B2 7-scale NCC | ALL | 60 | 13.3% | 13.3% | 13.3% | 13.3% | 340.36 | 600.00 | 177.8 |
| B3 source center-aware NCC | ALL | 60 | 50.0% | 50.0% | 50.0% | 50.0% | 37.77 | 105.71 | 246.3 |
| NOVA bounded ROI + center gate | ALL | 60 | 100.0% | 100.0% | 100.0% | 100.0% | 0.38 | 0.71 | 172.7 |

Architecture-specific ≤5 px accuracy: B1 23.3% DRAM / 3.3% FinFET; B2 23.3% / 3.3%; B3 70.0% / 30.0%; hybrid 100.0% / 100.0%.

**Interpretation warning:** the hybrid's 100% centered result is not evidence of unconstrained localization. The supplied generator deliberately places the target at `(500, 500)` and the hybrid includes a center candidate and late center ambiguity gate. The known benchmark prior is doing material work.

### Deterministic translated holdout — shifts up to ±50 px

| Method | Arch | n | ≤1 px | ≤2 px | ≤5 px | ≤10 px | Mean error px | P95 error px | Mean latency ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B1 fixed NCC | ALL | 60 | 13.3% | 13.3% | 13.3% | 13.3% | 332.92 | 567.27 | 34.8 |
| B2 7-scale NCC | ALL | 60 | 13.3% | 13.3% | 13.3% | 13.3% | 342.03 | 567.27 | 175.5 |
| B3 source center-aware NCC | ALL | 60 | 43.3% | 43.3% | 43.3% | 43.3% | 42.80 | 120.40 | 257.7 |
| NOVA bounded ROI + center gate | ALL | 60 | 51.7% | 51.7% | 51.7% | 51.7% | 22.64 | 79.53 | 176.5 |

Architecture-specific ≤5 px accuracy after translation: B1 23.3% DRAM / 3.3% FinFET; B2 23.3% / 3.3%; B3 56.7% / 30.0%; hybrid 66.7% / 36.7%.

**Interpretation:** the hybrid reduces overall mean and p95 error relative to B3 in this translated test, but the 51.7% ≤5 px success rate is still inadequate to claim robust drift recovery. FinFET remains particularly challenging, and the fixed translation test is not a realistic physical SEM stage/optics simulation.

Full per-architecture metrics for every threshold are in `results/integrated_benchmark/integrated_metrics_summary.json`; per-sample results are in `results/integrated_benchmark/integrated_per_sample_results.csv`.

## 6. Validation and ML experiments (negative results retained)

A separate generated validation set (12 DRAM + 12 FinFET) and a translated counterpart were used to examine gradient correlation, FFT/periodicity similarity, ambiguity handling, and a Random Forest candidate ranker. Validation evidence:

| Validation split | Method | Accuracy ≤5 px | Mean error px |
|---|---|---:|---:|
| Centered validation | B3 center-aware | 45.8% | 33.13 |
| Centered validation | P1 gradient | 12.5% | 259.19 |
| Centered validation | P2 gradient + periodicity | 12.5% | 273.12 |
| Centered validation | P4 ML ranker | 20.8% | 263.66 |
| Centered validation | P5 ML + structure | 8.3% | 331.08 |
| Translated validation | B3 center-aware | 29.2% | 47.94 |
| Translated validation | P1 gradient | 8.3% | 307.18 |
| Translated validation | P2 gradient + periodicity | 8.3% | 358.82 |
| Translated validation | P4 ML ranker | 8.3% | 335.58 |
| Translated validation | P5 ML + structure | 4.2% | 332.95 |

Candidate pool recall alone was 100% at 5 px on centered validation and 95.8% on translated validation for the later ROI-candidate extractor, but top-1 ranking remained poor. Therefore, the Random Forest is a real executable experiment but **not** a validated or recommended inference model. Its weights are not loaded by the final evaluator entrypoint. The evidence suggests feature scaling/definition and correct-candidate discrimination are unresolved; it is not justified to claim the ML stage improves localization.

No reference-context image is provided by the input contract, so the context-ring part of the paper/protocol formulation was not falsely marked implemented. Sub-pixel numerical estimates have not been calibrated to physical stage coordinates.

## 7. Tests and notebook execution

Final test command:

```bash
python -m pytest -q
```

Previous integrated-pipeline-only result: **10 passed**. After adding the inspection reporting and expert-annotation coordinate tests, the current suite is **16 passed** (see the addendum below).

Tests cover:
1. source reference/search files exist and decode;
2. repeated inference gives the same stable outputs;
3. representative centered sample fits the generator's center label;
4. missing file raises an explicit error;
5. invalid ROI/score window/scale list is rejected;
6. evaluator CLI prints exactly two numeric coordinates;
7. complete 480-row benchmark has 60 unique samples per condition and all four methods;
8. benchmark summary records the centered-to-translated degradation.

Historical run: the notebook was previously executed with the full 480-prediction benchmark and ML/structure diagnostic enabled. Latest clean run: 25 cells / 13 code cells, zero error outputs; it used the saved verified 480-row benchmark ledger and saved ablation evidence, ran live single-pair inference, regenerated the inspection tables/visuals, checked dataset-factory interfaces, and passed 16 embedded tests. The optional heavy recomputation/retraining switches remain available but are off by default for reliable all-cell execution. The interactive Colab upload branch has not been exercised against Google's live hosted runtime in this environment.

## 8. How to run

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python src/infer.py results/baseline/DRAM/reference/dram_0000.png results/baseline/DRAM/search/dram_0000.png
```

For full benchmarking, run:

```bash
bash scripts/run_all_benchmarks.sh
```

Or follow the three 20-pair chunks described in `results/integrated_benchmark/INTEGRATED_BENCHMARK_REPORT.md`.

## 9. Current notebook runtime experiment (new independent seeds)

When `RUN_ML_STRUCTURE_ABLATION=True`, the integrated notebook runs a lightweight ML/structure diagnostic. It generates 12 training pairs (6 DRAM + 6 FinFET) from separate deterministic seeds, trains `src.verification.train_ranker`, and evaluates the trained ranker on 6 validation pairs (3 DRAM + 3 FinFET) generated with different seeds. Results and the serialized experimental model are stored under `work_results/notebook_ml_ablation/`. This optional operation is disabled by default in the final Colab notebook so Run all completes promptly.

| Method | Validation n | Candidate recall ≤5 px | Accuracy ≤5 px | Mean error (px) | P95 error (px) |
|---|---:|---:|---:|---:|---:|
| B3 center-aware | 6 | 100.0% | 66.7% | 12.78 | 41.82 |
| G1 center candidate gate | 6 | 100.0% | 83.3% | 8.04 | 34.75 |
| P1 gradient | 6 | 100.0% | 0.0% | 389.60 | 505.70 |
| P2 gradient + periodicity | 6 | 100.0% | 0.0% | 449.56 | 520.70 |
| P3 ambiguity gate | 6 | 100.0% | 0.0% | 449.56 | 520.70 |
| P4 ML ranker | 6 | 100.0% | 0.0% | 295.64 | 591.59 |
| P5 ML + structure | 6 | 100.0% | 0.0% | 399.52 | 543.71 |

This tiny six-pair exercise is a smoke/diagnostic test, not a robust model estimate. It is consistent with the earlier, larger negative ablations: the tested hand-designed gradient/FFT and RF ranking variants can choose distant lookalikes despite the true candidate being present. They must not be promoted to the default inference path. Full per-sample scores are in `work_results/notebook_ml_ablation/validation_scored/seed_heldout_validation_per_sample.csv`, and the full summary JSON contains the P95 values.

## 10. Unresolved work before real-wafer use

- Acquire legally usable real wafer SEM source images from the supplied dataset factory's cited sources and preserve their licenses/citations.
- Obtain representative reference/search pairs with expert-validated localization labels, including tool/stage metadata and wafer-level splits.
- Evaluate on held-out wafers and true OOD patterns, not just transformed versions of the same synthetic image.
- Improve candidate generation to retain the true candidate under drift; then improve ranking with hard negatives and a leakage-resistant split.
- Reconsider the center prior against the target challenge specification and evaluate failures where the true target genuinely lies away from the image center.
- If physical units are required, obtain verified pixel-to-stage calibration and validate error against instrument ground truth.

**Final status:** source-integrated, benchmarked, regression-tested Colab research notebook and CLI prototype. The classical hybrid improves the supplied synthetic tests but remains weak under translated drift; the learned ranker did not improve the held-out/validation evidence and is disabled by default. Real-world semiconductor localization remains unvalidated.


## Addendum — per-chip inspection reporting and defect-annotation schema (2026-10-27)

The integrated project now includes `src/inspection_reporting.py` and `scripts/build_inspection_report.py`. This supplements template localization with a review-focused inspection presentation and tabular exports. It does not convert template matching into a trained defect classifier.

- Annotated side-by-side images report reference/search, predicted coordinate, ground truth where present, pixel error, NCC score (not a probability), chosen scale, score gap, center-prior use, candidate count and chip-local coordinates.
- `inspection_sites.csv`: one row per pair with global/local localization coordinates and diagnostics.
- `chip_summary.csv`: one row per configured chip/tile.
- `anomaly_candidates.csv`: shortlist; `anomaly_candidates_all.csv`: all thresholded local-contrast components.
- `reviewed_defect_annotations.csv`: optional expert-supplied defect type and coordinates transformed to local coordinates; all such labels stay separate from heuristic detections.

On the included 60 synthetic pairs, the current thresholding settings produced **10,032 unconfirmed local-contrast candidates** and a **593-row shortlist**. These are not confirmed defect counts. Synthetic generated patterns can be periodic and noisy, so candidate false positives are expected. No known defect masks/type labels were supplied. The template `data/defect_annotations_template.csv` and optional `--defect-annotations` CLI interface allow genuine reviewed labels to be introduced without conflating them with heuristic candidates.

Current independent regression suite after adding inspection report and reviewed-label support: **16 passed**. One added test verifies expert annotation type, confirmation state, tile mapping and local coordinate output. Full output schemas are defined in `docs/inspection_output_schema.md`.