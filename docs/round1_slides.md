# Round-1 Slide Content — NOVA-Loc

## Slide 1 — Team Details

**Team:** NOVA-Loc
**Problem:** Drift-Sense — Navigation-Error Recovery
**Members:** [fill names / roles]
**Institution:** Faculty of Technology, University of Delhi

One-line positioning: **Vision-only recovery of the intended wafer site when periodic structures make ordinary matching ambiguous.**

## Slide 2 — Problem Statement

Inspection tools repeatedly revisit the same wafer/die site. Navigation drift can place the tool on the wrong location. Drift-Sense asks for the center `(x,y)` of a high-resolution reference pattern inside a much larger search image with an approximately 10× scale relationship. The official Phase-2 test uses independently generated, noisier DRAM- and FinFET-style cases and includes a highly periodic hard case.

## Slide 3 — Idea Description

### Core idea
**Structural Site Identity = local appearance + neighborhood context + periodicity + geometry + scale.**

### Why not simple template matching?
A single NCC maximum can be misleading when the same local motif repeats throughout DRAM/FinFET arrays. NOVA-Loc first retrieves candidates, then disambiguates them instead of trusting one peak.

## Slide 4 — Proposed Solution

**Pipeline**
`Reference/Search → multi-scale retrieval → Top-K → context verification → periodicity analysis → ambiguity gate → lightweight ranker → subpixel refinement → center rule → (x,y)`

**Generator**
- 1000×1000 search image
- ~10× reference magnification
- DRAM and FinFET motifs
- independent noise per capture
- SEM-inspired edge brightening
- blur / rotation / scale / illumination variation
- recorded ground-truth center

## Slide 5 — Innovation & Uniqueness

1. **Structural Site Identity** — goes beyond local patch similarity.
2. **Periodicity-Aware Disambiguation** — explicitly detects repeated candidate families using spatial/frequency evidence.
3. **Confidence-Gated Verification** — easy cases remain fast; ambiguous cases receive deeper verification.
4. **OOD-First Dataset Design** — training covers a broader family of degradation conditions than the held-out stress suite.

## Slide 6 — Results

Current internal generator results (replace with final Round-2 numbers after the full NOVA-Loc verifier is complete):

- Fixed 10× NCC: very high localization error on periodic fields.
- Multi-scale NCC: does not by itself solve periodic ambiguity.
- Center-aware NCC: substantially improves performance but still leaves hard periodic/low-resolution failures.

Required final plots:
- accuracy within 1/2/5/10 px
- DRAM vs FinFET
- ID vs OOD
- periodic subset
- latency per 1000×1000 pair
- success case + honest failure case

## Slide 7 — Technology & Feasibility

Python, NumPy, OpenCV, Pillow; optional lightweight PyTorch ranker in Round 2.

Evaluator-facing script:
`python infer.py reference.png search.png`

Design target: deterministic, offline, CPU/GPU compatible, no manual edits, no network dependency.

## Slide 8 — GitHub & Video

GitHub: [insert repository URL]
Demo: [insert max-5-minute video URL]

Show one live example where NCC selects a repeated false match and NOVA-Loc selects the intended site.

## Slide 9 — References

Use the references in `docs/references.md` and put a small citation number next to every augmentation/noise/structural choice.