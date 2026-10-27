# NOVA-Loc Inspection Report

- Inspection images/sites: **60**
- Chip/tile table rows: **60**
- All thresholded heuristic anomaly candidates: **10032** (unconfirmed; not a defect count)
- Shortlist candidates retained for easy review: **593**.
- Raw thresholded components before per-image cap: **10032**; candidates omitted by cap: **9439**.
- Candidate severity tags: {'LOW': 7710, 'MEDIUM': 2322}
- Rows with ground truth coordinates: **60**; rows without ground truth: **0**.
- Coordinates are search-image pixel coordinates and chip-local pixel coordinates. Optional lot/wafer/die metadata is carried from the input manifest; missing values remain NOT_PROVIDED.
- Default geometry treats each search image as one chip/inspection field. Provide chip width/height only when actual tile geometry is known.
- The package lacks confirmed defect labels and a validated defect classifier. Candidate records are unclassified outliers for review, not confirmed physical defects. `anomaly_candidates.csv` is a top-K shortlist; `anomaly_candidates_all.csv` retains all thresholded components.
- No wafer-space/physical coordinates are emitted because wafer registration and pixel-to-length calibration are absent.

## Output files

- `inspection_sites.csv`: localized target, image-global and chip-local coordinates, localization error (where ground truth is available), NCC score, scale and gate diagnostics. It is one row per reference/search pair.
- `chip_summary.csv`: one row per chip/tile, localized-site fields and anomaly-candidate summary.
- `anomaly_candidates.csv`: shortlist of unclassified high-pass candidates with coordinates, bounding box, contrast type and review action (top 20 per image by default).
- `anomaly_candidates_all.csv`: all retained thresholded components, before the top-K shortlist cap.
- `anomaly_candidates_all.csv`: every thresholded candidate component with image-global and chip-local coordinates (can be noisy because this is not a validated defect detector).
- `reviewed_defect_annotations.csv`: optional user/expert-supplied defect labels mapped to image-global and chip-local coordinates; empty when no annotation file is supplied.
- `visuals/<architecture>/<sample>_inspection.jpg`: compact reference/search visualization with red prediction, green ground truth, and orange top anomaly candidates.
- `sample_inspection_overlay.jpg`: representative overlay with the largest available ground-truth localization error so the prediction/GT markers and displayed error can be reviewed.