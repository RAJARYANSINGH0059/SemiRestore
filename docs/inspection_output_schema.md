# NOVA-Loc inspection output schema

The research notebook writes its main inspection products under `results/inspection_report/`. Outputs are intentionally small-table-first: one annotated image for a quick visual check, plus CSVs for complete row-level review. No interactive UI or server is created.

## Coordinate frames

- **Image-global (`*_image_px`)**: `(x, y)` in the original search image, origin at the top-left; x increases right and y increases down. These are pixel coordinates, not wafer-stage or nanometre coordinates.
- **Chip-local (`*_chip_local_px`)**: image-global coordinate minus the configured rectangular chip/tile origin `(chip_origin_x_px, chip_origin_y_px)`.
- **Default layout**: one search image is treated as one inspection field/chip because actual chip/die boundaries were not supplied. In that mode, chip-local coordinates equal image-global coordinates.
- **Multiple chips/dies per image**: set both `CHIP_WIDTH_PX` and `CHIP_HEIGHT_PX` in the notebook configuration only after measuring/confirming the real grid. The code splits the image into rectangular tiles. Irregular layouts require a real chip/die map and are not inferred automatically.
- **Wafer coordinates and physical units**: not produced. These require registration to wafer/stage coordinates and a verified pixel-to-physical calibration.

## Output tables

### `inspection_sites.csv` — one row per reference/search pair

Contains sample, lot, wafer, architecture, die and chip identifiers where supplied; resolved reference/search paths and SHA-256 hashes; image size; predicted localization coordinates in image-global and chip-local frames; ground-truth coordinates where supplied; Euclidean localization error; NCC score; score gap; selected scale; candidate count; whether the center prior was used; localization status; anomaly counts by severity; and the annotated image path. `match_score_ncc` and `score_gap_ncc` are diagnostics, **not calibrated probabilities**.

### `chip_summary.csv` — one row per chip/tile

Contains tile origin and dimensions; prediction/ground-truth counts and coordinates; localization error where truth is available; counts by heuristic candidate kind/severity; and confirmed defect counts only for tiles with reviewed annotations supplied. Without expert labels, confirmed defect count is `NOT_AVAILABLE_NO_DEFECT_LABELS`.

### `anomaly_candidates.csv` — review shortlist

At most `MAX_ANOMALY_CANDIDATES_PER_IMAGE` high-ranking local-contrast components per image (20 by default). It includes candidate ID, sample/chip metadata, image-global and chip-local coordinates, bounding box, area, mean intensity/residual, robust outlier score, candidate kind, heuristic severity, defect type and confirmation state.

### `anomaly_candidates_all.csv` — complete heuristic candidate register

Every component that passes the current image-statistic/area thresholds. It can include ordinary repeated circuitry, grain, image noise and contrast boundaries. Its rows are **unconfirmed triage candidates, not verified defects**. The heuristic `severity` is a priority tag from the outlier score; it is not a physical defect grade.

### `reviewed_defect_annotations.csv` — optional expert labels

Empty when no label file is supplied. To load labels, copy `data/defect_annotations_template.csv` to `data/defect_annotations.csv`, fill one row per actual reviewed defect, set `sample_id`, `x_image_px`, `y_image_px`, `defect_type`, and (only when the label is confirmed) `confirmed=true`. Optional fields include bounding box, severity, lot/wafer/die metadata, annotator, date, evidence and notes. Then set `DEFECT_ANNOTATIONS_CSV = 'data/defect_annotations.csv'` in the notebook config, or pass `--defect-annotations data/defect_annotations.csv` to the report CLI. Coordinates must be in the original search image; out-of-image locations fail explicitly. Expert annotations are kept separate from heuristic anomaly detections.

## Annotated image

`sample_inspection_overlay.jpg` and one file in `visuals/<architecture>/` per sample show the reference and search images side-by-side; red X is predicted localization, green + is ground-truth localization when available, and orange boxes show a small number of heuristic candidate outliers. The heading reports predicted/true coordinates, pixel error, NCC score (not a probability), selected scale, chip-local coordinates, score gap, center-prior use and candidate count.

## What this system does not currently classify

The bundled DRAM/FinFET images are generated synthetic patterns with ground-truth **localization** labels. They do not have defect-type labels or confirmed physical defect annotations. NOVA-Loc localizes a reference motif and calculates heuristic local-contrast candidates; it does not identify a particle/scratch/bridge/void as a validated class. Populate the optional expert-label CSV from real inspection records to report actual defect types; a validated defect-detection/classification model and independently labelled real SEM data are necessary for automated classification.