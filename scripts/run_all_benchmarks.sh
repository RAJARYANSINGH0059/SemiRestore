#!/usr/bin/env bash
set -euo pipefail
python scripts/run_integrated_benchmark.py --start-index 0 --limit 20 --out-dir results/benchmark_parts/part1
python scripts/run_integrated_benchmark.py --start-index 20 --limit 20 --out-dir results/benchmark_parts/part2
python scripts/run_integrated_benchmark.py --start-index 40 --limit 20 --out-dir results/benchmark_parts/part3
python scripts/merge_benchmark_parts.py