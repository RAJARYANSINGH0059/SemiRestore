# SemiRestore — Presenter Guide (15 slides)

This guide contains fuller talking points for the editable PowerPoint. All numeric benchmark values below refer to the bundled **synthetic** 60-pair dataset, not real wafer production data.

## Slide 1: SemiRestore

Introduce the project as a Python research notebook and command-line localization workflow. The goal is not to claim a production defect classifier; it is to make reference localization reproducible and add an inspection-oriented report with coordinates and reviewable anomaly candidates. Team members: Raj Aryan Singh, Aman Kumar, Abhishek Kumar, Nishant Kumar Goyal.

## Slide 2: Project in one minute

Explain the distinction between localization and defect classification. The baseline task knows a reference patch and searches a larger image. The added report flags high-contrast/texture components to guide review but it does not know whether each component is a defect. The reported 10,032 rows are candidate components, not 10,032 true defects.

## Slide 3: The core problem

Walk through the supplied workflow graphic. Physical drift moves the landing point; scale disparity enlarges the candidate search; noise and blur degrade edges; periodic DRAM/FinFET structure creates lookalike locations. The combined effect means the raw maximum similarity is not equivalent to the correct physical location.

## Slide 4: Four interacting failure modes

For each failure mode, pair it with the mitigation implemented or tested. Be careful not to claim that the system fully solves physical drift: the translated stress benchmark proves the remaining weakness. A bounded ROI is only as good as the drift range and image geometry it covers.

## Slide 5: Data contract and coordinates

Explain that the reference and search are separate captures or scaled representations. The generator has exact synthetic target coordinates which are recorded in CSV. The coordinate origin is the top-left of the search image, x rightwards and y downwards. Tile-local position is computed by subtracting a known tile origin. The current one-field dataset makes local equal global; it does not contain a calibrated wafer map.

## Slide 6: Baselines

B1 fixes the scale using the nominal 10× linear difference. B2 searches seven ratios 0.085–0.115. B3 implements the source rule of choosing the centre-nearest among near-equivalent good candidates. SemiRestore adds a bounded centre-relative ROI, an explicitly scored centre candidate, and a late score-gap gate. Preserve the weak methods in the benchmark; do not report the hybrid without baselines.

## Slide 7: End-to-end pipeline

Explain each stage. Validate images first. Resize reference over a seven-point scale band around 0.1. Use NCC to build candidate positions in a bounded ROI. Also score the nominal centre explicitly. Use the score-gap gate to decide whether the centre prior should resolve an ambiguous result. Return localization plus diagnostics and generate inspection tables. The review pathway is for anomaly candidates and uncertainty; it is not a confirmed defect classifier.

## Slide 8: Core algorithm

The NCC map compares normalized grayscale patterns. Multiscale search handles ratio variation. The ROI constrains search around expected drift. Explicit centre scoring ensures the centre remains a candidate even if it lies outside the global top-K peaks. The gate uses a score gap in NCC units; 0.05 is a tuned setting for the current synthetic validation and is not a calibrated confidence threshold. If the ROI best is substantially better than centre, the ROI candidate wins.

## Slide 9: Result view

Use this slide to show the professor the style of output. The reference is on the left and search field on the right. Red X indicates prediction, green plus indicates ground truth, orange boxes are unconfirmed local anomaly components. This shown example is a good localized sample with 0.7 px error, but it is synthetic and centre-biased. The user-provided screenshot also demonstrated a failure with roughly 141 px error; that is why every prediction must be paired with ground truth during evaluation when available.

## Slide 10: Inspection reporting tables

The inspection-sites table preserves per-pair predicted and ground-truth coordinates, errors, selected scale and diagnostic fields. The chip summary gives one row per tile; in the current benchmark each image is one tile, so local and global coordinates are equal. To split a larger field into multiple chips, measured tile dimensions or a die map are required. The anomaly shortlist is review assistance. The reviewed-defect table is empty until an expert adds annotations; this avoids inventing defect classes.

## Slide 11: Current measured results

These are exact values from the integrated metrics JSON: 60 synthetic samples, 30 DRAM and 30 FinFET. The translated condition is deterministic up to ±50 pixel offsets. Fixed and multiscale NCC achieve 13.3% ≤5 px; center-aware reaches 50.0% centered and 43.3% translated; hybrid reaches 100% centered and 51.7% translated. The centered results are heavily influenced by known generator centre. State this prominently and do not characterize as official or real-wafer performance.

## Slide 12: Experiments and negative results

The supplied protocol calls for ablations from gradient similarity to context ring, periodicity and learned ranking. Gradient and FFT descriptors plus a Random Forest were tested, but did not reliably raise true candidates over repetitive lookalikes. The ranker stays experimental and is excluded from default inference. A context ring requires a broader reference context image, which is not in the current contract. Subpixel offsets are interpolation values only, not validated physical precision.

## Slide 13: Colab and CLI workflow

The repository uses a research notebook and a command-line inference system only. Setup should first clone the public repository; if not available, upload the ZIP. The bootstrap then detects the root based on source files, generates missing ground truth and images, and recomputes the benchmark only when the saved ledger is absent. The CLI prints x y for evaluator compatibility; detailed diagnostics go to JSON. Reports save in the inspection_report folder. This slide directly relates to the Colab FileNotFoundError and its fix: project detection should not require an optional benchmark JSON, and compact ZIPs must be allowed to generate missing files.

## Slide 14: Roadmap and limits

The current code is a research prototype. To advance, collect licensed real SEM imagery, native localization ground truth, and expert defect annotations. Use source/wafer-level split to prevent leakage from adjacent crops. Create OOD stress tests with held-out pitch, blur, noise and rotation ranges. Calibrate image pixels to stage coordinates only using instrument metadata. Report false positives, misses and runtime—not only mean localization error. The design does not yet meet the bar for industrial deployment.

## Slide 15: Close and invite questions

Summarize: SemiRestore preserves baseline methods, adds a bounded ROI and ambiguity gate, and turns outputs into an annotated image and coordinate tables. The engineering value is reproducibility and explicit failure evidence, not unsupported industrial claims. Ask for feedback on the next validation step: real SEM data, source-separated ground truth, and improved drift robustness. Names shown: Raj Aryan Singh, Aman Kumar, Abhishek Kumar, Nishant Kumar Goyal.

## Suggested opening (30–45 seconds)

SemiRestore is a research prototype for a specific semiconductor vision problem: given a small high-detail reference patch and a larger search image, find the correct matching site despite scale variation, blur/noise, drift, and repetitive layouts. The project combines reproducible baseline comparisons with a bounded-ROI hybrid and a report that exposes coordinates, error, and candidate anomalies. It is a notebook and CLI system—not a deployed service or validated manufacturing tool.

## Recommended questions to prepare for

1. **Why can the highest NCC score be wrong?** Repetitive layouts produce near-identical patterns; normalized similarity does not encode physical location or neighborhood context by itself.
2. **Why does the centre prior exist?** The challenge/source specification uses nearest-to-centre resolution for near-tied candidates. It is applied late, not as an unconditional answer, because real drift can move the site.
3. **Why is centred accuracy 100%?** The synthetic generator places the true target at (500, 500), and the hybrid uses a centre-relative region and late centre gate. This is a strong generator prior, so the translated stress set is more informative for drift.
4. **Is this defect detection?** Only partially. The report creates heuristic anomaly candidates for review. Without expert labels and a validated defect classifier, the system cannot call them confirmed defects or assign physical defect types.
5. **What would be needed to claim real accuracy?** Licensed real SEM images, native reference/search localization labels, expert defect annotations, source-/wafer-level train/validation/test splits, instrument calibration, and measurements of false positives, misses, latency, and worst-case error.
6. **Why are ML results not the default?** Initial gradient/FFT/Random Forest ranking did not reliably promote the correct candidate under translated repeated layouts. We keep the experiment to make the negative result reproducible rather than presenting a failed method as an innovation.

## Benchmark summary to quote accurately

- 60 synthetic reference/search pairs (30 DRAM + 30 FinFET).
- Four methods × 60 pairs × 2 conditions = 480 prediction records.
- Accuracy within 5 px: B1 fixed NCC 13.3% centred / 13.3% translated; B2 7-scale NCC 13.3% / 13.3%; B3 centre-aware NCC 50.0% / 43.3%; SemiRestore hybrid 100.0% / 51.7%.
- Hybrid centred mean error: approximately 0.38 px. Hybrid translated mean error: approximately 22.64 px. These results reflect the synthetic benchmark configuration; do not generalize to real chips.
- Inspection report: 60 site rows, 60 tile rows, 593 shortlisted heuristic candidates, 10,032 total heuristic components in the saved run. Those are candidates, not confirmed defects.

## Colab bootstrap fix explained

The earlier setup error occurred because project-root detection required `results/integrated_benchmark/integrated_metrics_summary.json`. The compact ZIP did not contain that optional saved-result file, so the setup rejected a valid project even though the source code was present. The patched setup identifies the root using core files (`src/nova_inference.py`, `src/generator.py`, `requirements.txt`, and `scripts/prepare_synthetic_dataset.py`), supports flat or nested ZIP layouts, generates missing synthetic inputs, and recomputes benchmark outputs only if saved results are absent.