# Preliminary prototype results

These are **internal measurements on NOVA-Loc's own synthetic generator**, not official Applied Materials test-set results.

| Architecture | Baseline | ≤1 px | ≤2 px | ≤5 px | ≤10 px | Mean error (px) | Mean latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| DRAM | Center-aware multi-scale NCC | 66.7% | 70.0% | 70.0% | 70.0% | 24.09 | 166.68 |
| FinFET | Center-aware multi-scale NCC | 30.0% | 30.0% | 30.0% | 30.0% | 51.44 | 166.67 |

## Interpretation

The strongest classical baseline is still unreliable under the deliberately periodic synthetic fields. The center-aware rule improves localization materially, but the residual failures are exactly where the proposed NOVA-Loc layers should be tested: periodic ambiguity, structural-context discrimination, and noisy low-resolution candidate matching.

**Do not present these numbers as official competition performance.** They are a reproducibility baseline and an engineering target for the next prototype iteration.