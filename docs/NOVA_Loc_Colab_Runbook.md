# SemiRestore — Google Colab Runbook

This repository is a Python research notebook and CLI only (no UI/server/API).

## Launch from GitHub

1. Open `notebooks/SemiRestore_Research.ipynb` on GitHub and click **Open in Colab**, or use:
   `https://colab.research.google.com/github/RAJARYANSINGH0059/SemiRestore/blob/main/notebooks/SemiRestore_Research.ipynb`
2. Run the first setup cell. It clones the public repository; if that is unavailable, it asks you to upload a repository ZIP.
3. Select **Runtime → Run all**. Missing synthetic sample images are recreated from the fixed-seed generator.
4. The default notebook loads the saved 480-row benchmark ledger, runs fresh one-pair inference, builds the inspection report and output tables, and skips nested-subprocess tests by default. The original bootstrap failure was caused by overly strict root checks; the updated code now detects the project from core files after either a clone or ZIP extraction.

## One-pair CLI

```bash
python scripts/prepare_synthetic_dataset.py
python src/infer.py results/baseline/DRAM/reference/dram_0000.png results/baseline/DRAM/search/dram_0000.png
python src/infer_diagnostics.py results/baseline/DRAM/reference/dram_0000.png results/baseline/DRAM/search/dram_0000.png --json-out results/final_cli_diagnostics.json
```

## Report tables and visuals

```bash
python scripts/build_inspection_report.py --root . --output-dir results/inspection_report
```

Coordinate table: `inspection_sites.csv`; tile table: `chip_summary.csv`; shortlist: `anomaly_candidates.csv`; full candidate ledger: `anomaly_candidates_all.csv`; expert labels: `reviewed_defect_annotations.csv`. Candidate anomalies are not confirmed defects.

## Full benchmark recomputation

```bash
python scripts/run_integrated_benchmark.py --start-index 0 --limit 20 --out-dir results/benchmark_parts/part1
python scripts/run_integrated_benchmark.py --start-index 20 --limit 20 --out-dir results/benchmark_parts/part2
python scripts/run_integrated_benchmark.py --start-index 40 --limit 20 --out-dir results/benchmark_parts/part3
python scripts/merge_benchmark_parts.py
```

Or set `RUN_FULL_HOLDOUT_BENCHMARK=True` in the notebook. The learned ranker experiment is separate and currently not part of default inference; set `RUN_ML_STRUCTURE_ABLATION=True` for research.

## Save outputs

In Colab, outputs live under `/content/SemiRestore/results/`. Copy the `results/` folder into Drive before disconnecting; Colab runtime storage is temporary.

## Run regression tests separately

The notebook has `RUN_AUTOMATED_TESTS=False` by default because its test suite launches nested Python subprocesses, which can be unstable when combined with image-report generation in constrained notebook runtimes. After setup, run the test suite in a fresh Colab cell/runtime:

```bash
python -m pytest -q
```

The development environment passed 16/16 tests. This does not establish real-wafer accuracy; the bundled pairs are synthetic.