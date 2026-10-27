# Source presentation and workflow audit

This summary records the requirements extracted from the two user-supplied NOVA-Loc PowerPoint decks and workflow graphic. The original decks are image-based slides, so the claims below are treated as source-stated design intent rather than proof of implementation.

## `NOVA-Loc_Precision_Navigation_Recovery.pptx` — 15 slides

- Slides 1–2 frame the drift-to-precision problem and the four interacting ambiguity sources.
- Slide 3 illustrates a highest-score trap: a false candidate at 0.980 versus a true candidate at 0.979. This is a schematic example, not a measured dataset result.
- Slides 4–5 propose a hybrid, effort-scaled triage pipeline.
- Slides 6–7 describe a roughly 10× scale gap and keeping Top-K candidates rather than immediately trusting one response maximum.
- Slide 8 proposes geometric relationships and periodicity/context for repeated layouts.
- Slide 9 proposes a learned verifier. Its claim about training for real SEM effects is not backed by supplied pretrained weights or labelled real SEM data.
- Slides 10–11 introduce a confidence/ambiguity gate and accept-or-recover branch.
- Slide 12 outlines expansion of search radius, higher-resolution crops and re-ranking.
- Slide 13 illustrates sub-pixel output and a ±15–20 nm tolerance claim. The supplied materials do not include physical calibration or real ground-truth evidence that could substantiate that accuracy.
- Slides 14–15 summarize proposed ambiguity mitigation and operational benefits.

## `NOVA-Loc_Navigation_Recovery.pptx` — 14 slides

- Slides 1–4 introduce physical drift, compound ambiguity and false matches from periodic layouts.
- Slides 5–8 outline effort-scaled search, multi-scale candidate generation and ambiguity handling.
- Slide 9 emphasizes context beyond the central reference pixels, including local pitch and periodicity.
- Slides 10–12 describe the confidence gate, differentiators and intended outputs.
- Slides 13–14 summarize the design philosophy and semiconductor/SEM motivation.

## `Wafer_Inspection_Error_Recovery_Workflow.png`

The diagram proposes four ambiguity sources: physical drift, reference/search scale mismatch, image degradation, and repetitive layouts. Its response flow is scale handling/coarse search → Top-K candidate generation → structural/geometric/ML tri-factor verification → confidence estimation → sub-pixel output on a high-confidence branch or bounded recovery on a low-confidence branch.

## Traceability to the code

| Source idea | Current implementation status |
|---|---|
| Fixed-scale and multi-scale search | Implemented in the supplied baseline. |
| Center-nearest rule for tie/ambiguity handling | Implemented in the source baseline; hybrid applies a late center-score gate. |
| Bounded ROI around expected position | Implemented in `src/nova_inference.py`. |
| Structural gradient and FFT/periodicity evidence | Experimental in `src/verification.py`; not promoted into default inference because validation was poor. |
| Learned candidate ranker | Executable research experiment, but validation did not support using it by default. |
| Context-ring comparison | Not implemented: the evaluator input contract has only the cropped reference and search image, not a larger reference-context image. |
| Confidence gate | Current NCC score gap is a heuristic, not a calibrated probability. |
| Sub-pixel physical accuracy | Not established. No calibrated pixel-to-stage/nanometre transform or physical ground truth supplied. |
| Bounded recovery | The hybrid uses a bounded ROI, but its translated holdout performance shows that arbitrary drift recovery is not solved. |
| Per-chip outputs | Pixel-space tile summary supported when measured tile dimensions are provided. No native die map/stage coordinate conversion is included. |
| Defect classification | Not implemented. Local contrast outliers are unconfirmed review candidates, not confirmed defects. |

The source presentations contain several aspirational claims. `docs/NOVA_Loc_Test_Report.md` distinguishes these intended capabilities from code that was actually present, tested or left experimental.