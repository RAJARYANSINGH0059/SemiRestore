#!/usr/bin/env python3
"""Build per-site/per-chip inspection tables and compact visual overlays."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inspection_reporting import create_inspection_report

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, default=None)
    p.add_argument('--predictions-csv', type=Path, default=None)
    p.add_argument('--input-manifest', type=Path, default=None, help='Optional CSV with sample_id,reference,search and optional architecture,chip_id,gt_x,gt_y. Omit for bundled benchmark GT pairs.')
    p.add_argument('--defect-annotations', type=Path, default=None, help='Optional reviewed defect CSV: sample_id,x_image_px,y_image_px,defect_type; optionally severity,confirmed,annotator,notes and bounding box fields.')
    p.add_argument('--split', default='holdout_centered', choices=['holdout_centered','holdout_translated'])
    p.add_argument('--method', default='NOVA_hybrid_roi_gate', choices=['B1_fixed_ncc','B2_multiscale_ncc','B3_center_aware_ncc','NOVA_hybrid_roi_gate'])
    p.add_argument('--chip-width-px', type=int, default=None, help='Known chip/die tile width in search-image pixels; omit to treat each search image as one field/chip.')
    p.add_argument('--chip-height-px', type=int, default=None, help='Known chip/die tile height in search-image pixels; must be set with --chip-width-px.')
    p.add_argument('--no-anomaly-scan', action='store_true', help='Skip heuristic local-contrast anomaly candidates.')
    p.add_argument('--threshold-sigma', type=float, default=2.5)
    p.add_argument('--min-area', type=int, default=6)
    p.add_argument('--max-area', type=int, default=250)
    p.add_argument('--max-candidates-per-image', type=int, default=20, help='Top candidate cap per search image; use 0 to retain all thresholded components.')
    p.add_argument('--annotate-anomalies', type=int, default=12, help='Number of strongest candidates to mark on each image; full list stays in CSV.')
    a=p.parse_args()
    root=a.root.resolve()
    out=(a.output_dir or (root/'results'/'inspection_report')).resolve()
    pred=a.predictions_csv
    report=create_inspection_report(root,out,predictions_csv=pred,input_manifest=a.input_manifest,
        defect_annotations_csv=a.defect_annotations,split=a.split,method=a.method,
        chip_width_px=a.chip_width_px,chip_height_px=a.chip_height_px,
        scan_anomalies=not a.no_anomaly_scan,threshold_sigma=a.threshold_sigma,
        min_area=a.min_area,max_area=a.max_area,max_candidates_per_image=a.max_candidates_per_image,
        annotate_anomalies=a.annotate_anomalies)
    print(json.dumps(report,indent=2))
    return 0
if __name__=='__main__': raise SystemExit(main())