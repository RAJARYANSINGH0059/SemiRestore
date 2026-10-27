# SemiRestore

**Semiconductor localization & inspection research toolkit** — a Colab-ready Python research notebook and command-line inference system built around the supplied NOVA-Loc (Drift-Sense) prototype.

[![Tests](https://github.com/RAJARYANSINGH0059/SemiRestore/actions/workflows/tests.yml/badge.svg)](https://github.com/RAJARYANSINGH0059/SemiRestore/actions/workflows/tests.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RAJARYANSINGH0059/SemiRestore/blob/main/notebooks/SemiRestore_Research.ipynb)

> **Scope:** this repository is a research notebook + Python command-line inference workflow. It does **not** start a web UI, web server, or persistent API. Generated DRAM/FinFET pairs are synthetic SEM-inspired images, not real microscope captures. Do not treat candidate anomalies as confirmed defects or the current localization score as industrial/physical accuracy.

## Contents

- [Problem and objective](#problem-and-objective)
- [What the pipeline does](#what-the-pipeline-does)
- [Quick start in Google Colab](#quick-start-in-google-colab)
- [Local installation](#local-installation)
- [Command-line inference](#command-line-inference)
- [Inspection report and coordinate tables](#inspection-report-and-coordinate-tables)
- [Benchmark and current results](#benchmark-and-current-results)
- [ML/structure experiments](#mlstructure-experiments)
- [Dataset factory](#dataset-factory)
- [Mathematical definitions](#mathematical-definitions)
- [Tests and reproducibility](#tests-and-reproducibility)
- [Project tree](#project-tree)
- [Limitations and next research steps](#limitations-and-next-research-steps)
- [Provenance, citations and licensing](#provenance-citations-and-licensing)

## Problem and objective

Semiconductor inspection localization has to identify the physical site represented by a small high-detail reference patch inside a larger search image. The supplied NOVA-Loc materials describe compounding failure modes:

1. **Physical drift:** thermal expansion, vibration, mechanical slack or other movement can move the target from its expected landing point.
2. **Scale mismatch:** the reference may be around 10× larger in linear pixel dimension than the target appearance in the search image; a single scale can fail.
3. **Image degradation:** SEM noise, blur, contrast changes and weaker edges can reduce true-match evidence.
4. **Repetitive layouts:** DRAM grids, FinFET gates and periodic nanostructures produce many near-identical candidate locations.
5. **Compounding ambiguity:** the raw highest normalized cross-correlation (NCC) score may be a visually similar but physically incorrect location.

The core objective is **repeatable localization with traceable uncertainty and failure handling**, not merely producing a plausible coordinate. The workflow aims to compare fixed-scale and multi-scale baselines, apply the source-defined center-nearest tie-break only late, examine drift/translation behavior, and create simple outputs a researcher or inspector can review.

The supplied workflow illustration is retained here in compressed form:

![Supplied wafer inspection workflow](docs/assets/wafer_inspection_workflow.jpg)

### What is and is not meant by “defect detection”

The current source code localizes a reference patch. The inspection-report module additionally flags local contrast/texture outliers as **unconfirmed heuristic anomaly candidates**. It does not classify a point as a particle, scratch, bridge, void, missing feature, or any other physical defect unless an expert annotation is separately supplied. The synthetic benchmark has localization coordinates, but no expert-labelled defect masks or defect taxonomy.

## What the pipeline does

### Input/coordinate contract

- **Reference image (`R`)**: small detailed patch captured at higher apparent magnification.
- **Search image (`S`)**: larger field to search.
- Current challenge scaffold uses a 1000×1000 search field; references are usually approximately 10× larger than the true target crop in linear pixels.
- Evaluator-facing CLI: `python src/infer.py REFERENCE SEARCH`; standard output is exactly two numeric values, `x y`.
- Image coordinate frame: origin at top-left; x increases right and y increases down. Predicted coordinate represents the matched template/site centre in the search image.

### Implemented classical methods

| Method | Role | Important caveat |
|---|---|---|
| **B1 — fixed 10× NCC** | One scale using normalized cross-correlation. | Fast/simple; susceptible to scale variation and periodic lookalikes. |
| **B2 — seven-scale NCC** | Searches seven scale ratios around 0.1. | More search effort; still relies heavily on raw visual similarity. |
| **B3 — supplied center-aware NCC** | Uses the closest-to-center rule at a late ambiguity/tie stage. | Good when the generator's target is centered; can be biased when the target drifts. |
| **SemiRestore / NOVA hybrid** | Bounded center-relative ROI, multi-scale NCC, explicitly scored center candidate, late score-gap gate. | Current hybrid remains weak under large translations; it is not a general drift solver. |

`src/infer_baseline.py` preserves the earlier baseline entrypoint. `src/nova_inference.py` implements the current hybrid inference logic. `src/infer.py` preserves the exact evaluator output contract. `src/infer_diagnostics.py` adds a JSON diagnostics path without changing the standard CLI stdout.

### Experimental verifier (not default inference)

`src/verification.py` contains experiments using gradient similarity, FFT/periodicity descriptors, ambiguity features, and an optional Random Forest candidate ranker. These components are retained to support controlled ablations. On the validation experiments supplied and generated here, these hand-designed features / learned rankers did not reliably rank the true location above repetitive false candidates, particularly after drift. **The ranker is therefore not used by the default inference command.** Its presence means an experiment is executable, not that it is a validated production model.

The proposed context-ring score in the original protocol needs a larger reference-neighborhood/context crop. The current evaluator contract supplies only a reference crop and a search image, so context-ring matching is not silently fabricated or marked as implemented.

## Quick start in Google Colab

1. Open the notebook from GitHub using the **Open in Colab** badge above (once this repository exists on GitHub), or open `notebooks/SemiRestore_Research.ipynb` via **Colab → File → Upload notebook**.
2. In Colab, click **Runtime → Run all**. The bootstrap cell first tries to clone `https://github.com/RAJARYANSINGH0059/SemiRestore.git`; if that repository is not available yet, it falls back to a project ZIP upload.
3. The bootstrap checks required imports and installs `requirements.txt` only if packages are missing.
4. It creates only missing synthetic sample images using the fixed-seed source generator; any existing image files are preserved.
5. The notebook checks the dataset contract, runs a fresh one-pair CLI inference, loads the recorded benchmark summary, creates the inspection tables/annotated images, and displays the key source-integrated findings. Regression tests, full benchmark recomputation, and the separate-seed ranker experiment are separate switches so the main inspection flow remains stable on hosted notebook runtimes.

The default **Runtime → Run all** runs bootstrap, one-pair inference, the compact benchmark tables, the inspection report, and the optional-section status messages. It does not start a web app/API or automatically run the nested-subprocess test suite. The full benchmark and fresh ML experiment are opt-in:

```python
RUN_FULL_HOLDOUT_BENCHMARK = False # skip live recomputation; use the saved ledger
RUN_ML_STRUCTURE_ABLATION = False  # skip separate-seed model experiment
RUN_INSPECTION_REPORT = True       # site / chip / anomaly tables and annotated images
RUN_UNIFIED_FACTORY = False        # requires source images you provide under their licenses
RUN_AUTOMATED_TESTS = False        # run separately; tests launch nested subprocesses
SHOW_BENCHMARK_PLOTS = False       # simple visual outputs by default
```

For step-by-step commands, output locations, and troubleshooting, see [`docs/NOVA_Loc_Colab_Runbook.md`](docs/NOVA_Loc_Colab_Runbook.md).

## Local installation

Python 3.10+ is recommended. From the repository root:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/prepare_synthetic_dataset.py
python -m pytest -q
```

The 60 generated reference/search pairs are deliberately not stored as 120 large PNGs in this code-first repository. Their ground-truth CSVs are kept, and `src/generator.py` recreates them from fixed seeds. Pixel values may differ slightly across OpenCV/Pillow/NumPy versions (occasionally by one gray level). The saved 480-row ledger records the original supplied PNG bytes and is best treated as the source-run reference. The bootstrap checks the image hashes and warns if regenerated PNGs differ slightly. The default inspection report uses the saved predictions matched by sample ID; set `RUN_FULL_HOLDOUT_BENCHMARK = True` to recompute all 480 predictions on the exact image bytes generated in the current runtime. This avoids bloating clone/download size while keeping the benchmark runnable.

## Command-line inference

### Minimal evaluator-facing output

```bash
python src/infer.py \
  results/baseline/DRAM/reference/dram_0000.png \
  results/baseline/DRAM/search/dram_0000.png
```

Output format:

```text
x y
```

The default hybrid parameters are approximately seven scale ratios around 0.1, ROI radius 50 search pixels, and score-window 0.05 NCC units. The score window is a difference/gate parameter—not a probability. The center prior is applied late, but the current centered generator gives that prior substantial leverage.

### JSON diagnostics

```bash
python src/infer_diagnostics.py \
  results/baseline/DRAM/reference/dram_0000.png \
  results/baseline/DRAM/search/dram_0000.png \
  --roi-radius 50 \
  --score-window 0.05 \
  --scale-min 0.085 \
  --scale-max 0.115 \
  --scale-steps 7 \
  --json-out results/final_cli_diagnostics.json
```

This writes a structured JSON record with input paths/existence, coordinates, score semantics, selected scale, and diagnostic fields. Exact options are available from `python src/infer_diagnostics.py --help`.

### Original and comparison baselines

```bash
python src/infer_baseline.py \
  results/baseline/DRAM/reference/dram_0000.png \
  results/baseline/DRAM/search/dram_0000.png

python -m src.baseline --gt results/baseline/DRAM/ground_truth.csv --mode fixed --tolerance 5
python -m src.baseline --gt results/baseline/DRAM/ground_truth.csv --mode multiscale --tolerance 5
```

## Inspection report and coordinate tables

The report builder is `scripts/build_inspection_report.py`. The notebook runs it by default; it can also be run directly:

```bash
python scripts/build_inspection_report.py \
  --root . \
  --output-dir results/inspection_report
```

For a regular grid of multiple chips/dies inside a search image, provide **measured** tile dimensions, for example:

```bash
python scripts/build_inspection_report.py \
  --root . \
  --output-dir results/inspection_report \
  --chip-width-px 500 \
  --chip-height-px 500
```

Do not set dimensions unless they are accurate for the input image. The bundled synthetic examples default to one inspection field per image, so their local coordinates equal their image coordinates.

![Example localized inspection image](docs/assets/inspection_overlay_example.jpg)

### Coordinate semantics

- **Image-global pixel coordinates**: `(x_image_px, y_image_px)` measured in the original search image, origin top-left.
- **Chip-local pixel coordinates**: `(x_chip_local_px, y_chip_local_px)` relative to the top-left corner of the corresponding tile. Calculated by subtracting the tile origin.
- **Reference-match localization**: predicted patch/site centre. It is not a defect centroid unless an independently validated defect detector/label defines one.
- **Ground-truth coordinate**: available for the synthetic generated localization pairs, or for real input only when supplied by the operator/annotation manifest.
- **Physical wafer/stage coordinate**: not emitted. It needs registration, microscope/stage metadata, and calibrated pixel-to-physical conversion. No micrometre/nanometre number is fabricated.

### Report files

Under `results/inspection_report/`:

| File | Granularity | Includes |
|---|---|---|
| `sample_inspection_overlay.jpg` | One representative image; per-sample visuals are in `visuals/` when created. | Reference, search, prediction, ground truth when available, error, NCC score/scale and local coordinate. |
| `inspection_sites.csv` | One row per reference/search pair. | IDs, image paths/hashes, predicted x/y, ground-truth x/y if present, pixel error, scale, score/gate diagnostics, tile/chip origin and chip-local coordinate. |
| `chip_summary.csv` | One row per chip/tile. | Tile origin/dimensions, localization summary and unconfirmed-candidate counts. |
| `anomaly_candidates.csv` | Shortlist (Top-K per input, default 20). | Candidate centre/bounding box, heuristic evidence, image-global and chip-local coordinates. |
| `anomaly_candidates_all.csv` | All thresholded heuristic components. | Full candidate register for review, without cluttering the overlay. |
| `reviewed_defect_annotations.csv` | One row per supplied human/expert annotation. | Defect label, severity, confirmation status and transformed local coordinates. Empty except for headers/status when no annotations are supplied. |
| `inspection_report.json` | Summary. | Sample counts, coordinate semantics, candidate counts, parameters, input/label availability and limitations. |

The current synthetic run created 60 site rows, 60 chip/tile rows, 593 shortlisted candidates and 10,032 thresholded heuristic components. The latter two counts are **candidate outliers—not confirmed defects**. Periodic features, normal edge brightness, noise, or contrast differences can be flagged by a simple heuristic.

### Adding expert-verified defects

1. Copy `data/defect_annotations_template.csv` to `data/defect_annotations.csv`.
2. Fill actual review results; do not invent defect labels. Minimum useful fields are `annotation_id`, `sample_id`, `x_image_px`, `y_image_px`, `defect_type`, `severity`, `confirmed`, `annotator`, and `notes`.
3. Set `DEFECT_ANNOTATIONS_CSV = 'data/defect_annotations.csv'` in the notebook, or pass `--defect-annotations data/defect_annotations.csv` to the report builder.
4. Rerun report generation. The output maps image-global annotations into configured tile-local coordinates and preserves the confirmation state.

## Benchmark and current results

`RESULTS.md` from the supplied source and `docs/experimental_protocol.md` define the original baselines/metrics. The integrated ledger covers 60 fixed synthetic samples (30 DRAM + 30 FinFET), four methods, and two conditions (centered + deterministic translation up to ±50 pixels), resulting in **480 per-sample prediction rows**.

### Centered holdout, accuracy within 5 pixels

| Method | DRAM | FinFET | All 60 |
|---|---:|---:|---:|
| B1 fixed 10× NCC | 23.3% | 3.3% | 13.3% |
| B2 seven-scale NCC | 23.3% | 3.3% | 13.3% |
| B3 source center-aware NCC | 70.0% | 30.0% | 50.0% |
| SemiRestore/NOVA bounded ROI + center gate | 100.0% | 100.0% | 100.0% |

### Translated holdout, accuracy within 5 pixels

| Method | DRAM | FinFET | All 60 |
|---|---:|---:|---:|
| B1 fixed 10× NCC | 23.3% | 3.3% | 13.3% |
| B2 seven-scale NCC | 23.3% | 3.3% | 13.3% |
| B3 source center-aware NCC | 56.7% | 30.0% | 43.3% |
| SemiRestore/NOVA bounded ROI + center gate | 66.7% | 36.7% | 51.7% |

The integrated hybrid is improved on this synthetic translated test relative to B3, but 51.7% at ≤5 px is not adequate evidence of robust drift recovery. The centered 100% is strongly helped by the generator placing the true target at `(500,500)` and should not be interpreted as an unconstrained localization success rate.

### Reproduce the full benchmark

```bash
python scripts/prepare_synthetic_dataset.py
python scripts/run_integrated_benchmark.py --start-index 0 --limit 20 --out-dir results/benchmark_parts/part1
python scripts/run_integrated_benchmark.py --start-index 20 --limit 20 --out-dir results/benchmark_parts/part2
python scripts/run_integrated_benchmark.py --start-index 40 --limit 20 --out-dir results/benchmark_parts/part3
python scripts/merge_benchmark_parts.py
```

Or use `bash scripts/run_all_benchmarks.sh`. The benchmark uses the same samples/methods for each comparison and logs per-sample predictions, known transformations, errors and latency. Runtime values depend on hardware/software and are not a promised Colab latency.

Full results are saved at:
- `results/integrated_benchmark/integrated_per_sample_results.csv`
- `results/integrated_benchmark/integrated_metrics_summary.json`
- `results/integrated_benchmark/INTEGRATED_BENCHMARK_REPORT.md`

## ML/structure experiments

The supplied protocol proposes gradient support (P1), context ring (P2), periodicity/FFT (P3), ambiguity gate (P4), a ranker (P5), and sub-pixel refinement (P6). In this project:

- gradient/FFT/Random Forest ranking was implemented as an experiment in `src/verification.py`;
- experiments use distinct fixed seeds for generated training and validation data;
- held-out validation and translated validation did not show reliable top-1 improvement from the tested learned/handcrafted rankers;
- the ranker is excluded from default inference and remains research-only;
- context-ring verification is not claimed implemented because the input interface has no larger reference-context crop;
- sub-pixel physical accuracy is not claimed without physical calibration/ground truth.

To rerun the optional notebook experiment, set `RUN_ML_STRUCTURE_ABLATION = True`. Keep the final holdout untouched while choosing thresholds/features; the source protocol explicitly requires fixed seeds and strict OOD settings.

## Dataset factory

The supplied `external/unified_dataset_factory/` can build a source manifest, construct synthetic task-aligned pairs from licensed source imagery, and create source-level splits. Do not redistribute third-party raw images unless their licence permits it.

Example workflow after placing source datasets under `data/sources/<source-name>/`:

```bash
python external/unified_dataset_factory/scripts/build_manifest.py \
  --root data/sources --out data/source_manifest.csv --hash

python external/unified_dataset_factory/scripts/generate_unified_pairs.py \
  --source-root data/sources \
  --out data/generated \
  --pairs 5000 \
  --reference-size 64 \
  --search-size 1000 \
  --scale 10 \
  --seed 42 \
  --architecture mixed

python external/unified_dataset_factory/scripts/make_splits.py \
  --metadata data/generated/metadata.jsonl \
  --out data/generated/splits.json
```

See [`external/unified_dataset_factory/README.md`](external/unified_dataset_factory/README.md) and [`external/unified_dataset_factory/docs/SOURCES.md`](external/unified_dataset_factory/docs/SOURCES.md) for source registry/citations and the factory schema. Example listed sources include REFICS, Carinthia SEM, Edmond metal-layer SEM, Cheng polysilicon SEM, FinFET SEM, NFFA-EUROPE SEM, Bath periodic SEM and GaAs periodic data; verify availability, exact terms and citation requirements before use. **The repo does not bundle third-party real SEM images.**

## Mathematical definitions

For candidate location `(x,y)` and resized reference/template `T_r`, the core score is normalized cross-correlation:

`ρ_r(x,y) = ZNCC(T_r, S[x:x+h_r, y:y+w_r])`.

Candidates are scored across scale ratios around `r≈0.1`. The supplied ambiguity measure compares the best two candidate scores, conceptually `U = (J₁ - J₂) / (|J₁| + ε)`; a small score margin means multiple candidates are nearly equivalent and should trigger additional verification rather than false certainty. The final hybrid has a configurable late center score-gap gate, not a calibrated probability.

Pixel localization error is Euclidean distance:

`e = sqrt((x_pred-x_gt)^2 + (y_pred-y_gt)^2)`.

Accuracy at tolerance `τ` is the fraction of examples with `e ≤ τ`, typically reported at 1, 2, 5 and 10 pixels. OOD degradation ratio is `R_OOD = Acc_OOD / Acc_ID` where a valid held-out source/degradation split exists. No physical scale conversion is used without calibrated pixel-to-stage mapping.

## Tests and reproducibility

Run the tests from a shell / a fresh Colab cell, after the setup cell has completed:

```bash
python -m pytest -q
```

The notebook setting `RUN_AUTOMATED_TESTS` defaults to `False` because the test suite launches child Python processes; run it separately instead of immediately after the report pipeline in constrained notebook kernels. The suite covers input file decoding, repeated inference on fixed inputs, missing-file and invalid-configuration handling, evaluator CLI stdout contract, 480-row ledger completeness, centered-vs-translated degradation, chip-local coordinate transforms, simple overlay generation, report schema, optional ground-truth absence, and expert-label coordinate mapping. The suite has passed 16/16 tests in the development environment.

GitHub Actions runs the preparation script and test suite on pushes and pull requests. The generated pairs use fixed seed `20260816` for the base generator; the translated benchmark uses deterministic seed `930091`. The loader only creates missing images, so existing files are not silently overwritten. The test suite validates software behavior and saved synthetic fixtures, not real wafer performance.

## Project tree

```text
SemiRestore/
├── README.md
├── notebooks/SemiRestore_Research.ipynb
├── src/                         # generator, baselines, hybrid, CLI, diagnostics, inspection report, verifier experiment
├── scripts/                     # dataset bootstrap, benchmark runner/merger, inspection report builder
├── tests/                       # integration, report, dataset-factory tests
├── requirements.txt
├── data/                        # annotation + input-manifest templates
├── external/source_bundle/      # supplied original source implementation/docs
├── external/unified_dataset_factory/  # manifest, pair-generation, source-level split scripts
├── docs/                        # proposal, math, protocol, source audit, output schema, references
├── source_materials/             # original notebook backup, small source archives, provenance
└── results/                     # ground truth CSVs, metrics, 480-row benchmark ledger, inspection table/example outputs
```

Generated baseline reference/search images are intentionally ignored by Git and regenerated by `scripts/prepare_synthetic_dataset.py` after clone. Runtime scratch outputs and trained experimental model files are also ignored.

## Limitations and next research steps

1. The base generator places its target at the center; the hybrid benefits from this known prior. Expand validation to independently generated off-center/drift settings and unseen layout pitches.
2. Translated stress accuracy remains only 51.7% within 5 px. Improve candidate recall and ranking before claiming robust drift handling.
3. FinFET periodic ambiguity remains hard. The tested Random Forest ranking and gradient/FFT combinations did not improve top-1 sufficiently and remain disabled in default inference.
4. No real SEM acquisition, instrument native ground-truth localization, wafer-level training/test split, or pixel-to-nanometre calibration is included. The factory only supplies tooling for sources you obtain separately.
5. Heuristic anomaly candidates may flag normal circuitry, blur, edge enhancement or noise. A reliable defect classifier needs expert-labelled defects, a clear taxonomy, wafer/source-separated splits, and false-positive evaluation.
6. Physical/wafer/stage coordinate output needs chip/wafer registration, die map, orientation, stage metadata and calibration. `chip_summary.csv` only uses configured pixel tile dimensions and must not be called a physical die map without verified geometry.
7. Runtime numbers vary by machine, OpenCV build, thread count and Colab runtime. They are measurements, not deployment guarantees.
8. This package is a research prototype, not production-ready semiconductor inspection software.

### Next recommended work

- Obtain properly licensed real SEM source images and retain each source's terms and citation.
- Create verified reference/search coordinates and defect-type masks from expert review; split by source/wafer, not random crops alone.
- Add off-center and drifted procedural generator modes and an untouched OOD evaluation suite.
- Fix candidate recall first, then train a ranker with hard negatives and group-separated validation.
- Add context-ring evidence only when a larger reference context image is available.
- Calibrate confidence and physical units only against independent ground truth and instrument calibration.

## Provenance, citations and licensing

The project combines user-supplied `NOVA-Loc-source.zip`, `NOVA-Loc-prototype.zip` (its generated image pair set is reproduced from source rather than duplicated in this repo), `NOVA-Loc-unified-dataset-factory.zip`, `RESULTS.md`, `experimental_protocol.md`, `math.md`, the original notebook, two presentation decks and a workflow diagram. Source-derived requirements and slide summaries are captured in `docs/proposal.md`, `docs/round1_slides.md`, `docs/math.md`, `docs/experimental_protocol.md`, `docs/NOVA_Loc_Test_Report.md`, and `docs/supplied/`.

External dataset citations are listed in `external/unified_dataset_factory/docs/SOURCES.md` and `docs/references.md`. Check the latest source records for each dataset's terms before acquiring/redistributing image files. See [`LICENSE_STATUS.md`](LICENSE_STATUS.md): no project-wide license was provided, so this repository does not invent one.