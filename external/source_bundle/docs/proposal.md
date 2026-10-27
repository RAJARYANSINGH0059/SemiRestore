# NOVA-Loc — Round-1 Proposal

## Title
**NOVA-Loc: Noise-Robust, OOD-Aware, Verification-Guided Sensorless Navigation Recovery for Periodic Semiconductor Structures**

## Problem statement
**Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools**

## 1. Executive concept

Wafer inspection tools repeatedly revisit the same die site. When navigation drifts, the system must recover the intended site from a reference image and a much larger search image. The challenge is difficult because the reference/search pair has an approximately 10× linear scale difference, imaging noise is independent between captures, and repeated DRAM/FinFET structures can create multiple visually plausible matches.

NOVA-Loc treats recovery as a **structural site-identity problem**, not a single correlation-peak problem:

**local appearance + neighborhood context + periodicity + geometry + scale consistency → final site decision.**

The initial prototype implements the required synthetic generator and two NCC baselines. The full Round-2 system adds context-ring verification, frequency-domain periodicity analysis, hard-negative ranking, confidence-gated adaptive compute, and subpixel refinement.

## 2. Why simple template matching is insufficient

For a search image S and a resized reference template T, normalized cross-correlation can be written as

\[
\rho(x,y)=\frac{\sum_{u,v}(T_{u,v}-\bar T)(S_{x+u,y+v}-\bar S_{x,y})}{\sqrt{\sum_{u,v}(T_{u,v}-\bar T)^2}\sqrt{\sum_{u,v}(S_{x+u,y+v}-\bar S_{x,y})^2}}.
\]

A conventional system chooses

\[
(x^*,y^*)=\arg\max_{x,y}\rho(x,y).
\]

This is vulnerable when a periodic array produces many high-scoring, spatially separated candidates. NOVA-Loc therefore keeps the top-K candidates and verifies their structural identity.

## 3. Core mathematical formulation

For candidate i, define:

- \(C_i\): local visual similarity / NCC-derived score
- \(G_i\): geometric consistency of local gradients and edge structure
- \(Q_i\): contextual-ring similarity around the candidate
- \(F_i\): frequency-domain periodicity consistency
- \(A_i\): ambiguity penalty derived from repeated candidate scores
- \(D_i\): normalized distance from the search-image center
- \(S_i\): scale consistency

The proposed score is

\[
J_i=w_C C_i+w_Q Q_i+w_G G_i+w_F F_i+w_S S_i-w_A A_i-\lambda D_i.
\]

The predicted site is

\[
(\hat x,\hat y)=\operatorname*{arg\,max}_{i\in\mathcal C_K}J_i,
\]

where \(\mathcal C_K\) is the top-K candidate set from the coarse matcher.

### Candidate ambiguity

A simple uniqueness statistic is

\[
U=\frac{J_{(1)}-J_{(2)}}{|J_{(1)}|+\epsilon},
\]

where \(J_{(1)}\) and \(J_{(2)}\) are the best and second-best candidate scores. Small U indicates ambiguity and triggers deeper verification.

### Center prior

The official problem defines the center-nearest matching region as the correct choice when multiple matches exist. We therefore apply the center distance as a final decision term, not as a shortcut for candidate generation.

## 4. Dataset generator

The generator creates:

- 1000×1000 search images.
- Approximately 10×-zoomed reference images.
- DRAM-style orthogonal periodic cells.
- FinFET-style parallel-fin + gate-crossing structures.
- Independent reference and search noise.
- SEM-inspired edge brightening.
- Blur, rotation, scale and illumination variation.
- Ground-truth reference-center coordinates.

The generator is intentionally described as **SEM-inspired** rather than a full physical SEM simulator. Its purpose is to create controlled paired data with known geometry and an explicit domain-randomization gap.

### Proposed stress dimensions

1. Noise level.
2. Blur level.
3. Scale mismatch.
4. Rotation.
5. Periodic ambiguity.
6. Structural/pitch variation.
7. Combined out-of-distribution cases.

### Training philosophy

Training data should cover a family of imaging conditions instead of one fixed distribution. A held-out stress suite should contain parameter ranges excluded from training so that robustness is measured rather than assumed.

## 5. Localization architecture

### Stage A — Coarse retrieval

Resize the reference through a narrow scale band around the nominal 10× relationship and use fast NCC to obtain top-K candidates.

### Stage B — Structural verification

For each candidate, compare the reference with both the local patch and a surrounding context ring. This reduces false matches that share the same repeating local motif but differ in surrounding structure.

### Stage C — Periodicity analysis

Use autocorrelation and/or FFT peak analysis to estimate dominant spatial periodicity. Highly regular candidate families are flagged as ambiguous rather than accepted from one correlation maximum.

### Stage D — Learned ranking

A lightweight ranker receives candidate features such as local NCC, context score, edge consistency, frequency consistency, scale, candidate-rank gap, and center distance. Hard negatives are mined from visually similar periodic candidates.

### Stage E — Adaptive verification

High-confidence cases stop early. Ambiguous cases receive additional geometric/frequency verification. The design therefore spends compute in proportion to uncertainty.

### Stage F — Subpixel refinement

After selecting a candidate, refine the local response peak using a neighborhood fit so the system has a continuous internal estimate before producing the required pixel-coordinate output.

## 6. Why this is novel for the challenge

The novelty is not “using AI” alone. It is the integration of:

1. **Structural site identity:** local appearance + context + periodicity + geometry.
2. **Periodicity-aware ambiguity handling:** repeated structures are explicitly detected rather than treated as ordinary maxima.
3. **Confidence-gated adaptive compute:** the pipeline is fast for easy cases and deeper for ambiguous cases.
4. **Domain-randomized synthetic generation:** the training distribution is intentionally broader than the hidden-test distribution.

## 7. Experimental plan

### Baselines

B1. Fixed 10× normalized cross-correlation.

B2. Multi-scale NCC over a narrow ratio band around 10×.

B3. Multi-scale NCC + gradient representation.

### Proposed ablations

A1. + context ring.

A2. + periodicity feature.

A3. + ambiguity gate.

A4. + learned ranker.

A5. + subpixel refinement.

### Metrics

- Accuracy within 1 px, 2 px, 5 px and 10 px.
- Mean/median/p95 localization error.
- DRAM accuracy.
- FinFET accuracy.
- Periodic-subset accuracy.
- OOD accuracy.
- Average and p95 inference time per 1000×1000 pair.
- Model size and peak memory for the learned ranker.
- Confidence calibration error.

### OOD protocol

Exclude selected ranges of noise, blur, pitch, scale and rotation from training. Evaluate only on those excluded ranges. The key quantity is not only absolute accuracy but degradation from in-distribution to OOD conditions.

## 8. Failure analysis

The required honest failure case should be a deliberately hard periodic example. The analysis should report the candidate scores, why the true site was not uniquely identifiable under the first model, and how the next iteration addresses the observed failure.

## 9. Deployment and reproducibility

The evaluator-facing contract is intentionally minimal:

```bash
python infer.py reference.png search.png
```

The script must load all required components automatically and print one coordinate pair. No internet access, notebook interaction, manual path editing or hidden setup should be required.

## 10. Expected impact

NOVA-Loc is designed as a vision-only recovery layer that can complement existing tool navigation when visual drift causes a system to land at the wrong site. The technical emphasis is robustness to repeated semiconductor structures and unseen imaging conditions, with a controlled computation path suitable for repeated inspection workloads.

## 11. Evidence plan for Round 2

The final presentation should not claim performance before measurement. Every claimed improvement will be tied to an ablation result, every augmentation choice to 2–3 credible references, and every runtime claim to a measured benchmark on a specified machine.

## 12. Official challenge alignment

The official page requires a documented DRAM/FinFET generator, independent noise, SEM-like edge brightening, known ground-truth coordinates, realistic blur/rotation/scaling variation, at least 30 randomized self-test pairs, a standalone inference script, and a public repository. It also states that the official Phase-2 set is independently generated, noisier than training, covers DRAM and FinFET, and includes at least one highly periodic difficult region. NOVA-Loc is designed directly around those requirements.