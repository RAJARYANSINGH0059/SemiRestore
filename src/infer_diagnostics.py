#!/usr/bin/env python3
"""Detailed JSON command-line interface for NOVA-Loc; evaluator CLI remains src/infer.py."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
try:
    from .nova_inference import locate
except ImportError:
    from nova_inference import locate

def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('search', type=Path)
    p.add_argument('--roi-radius', type=int, default=50)
    p.add_argument('--score-window', type=float, default=0.05)
    p.add_argument('--scale-min', type=float, default=0.085)
    p.add_argument('--scale-max', type=float, default=0.115)
    p.add_argument('--scale-steps', type=int, default=7)
    p.add_argument('--json-out', type=Path, default=None)
    p.add_argument('--chip-id', default='CHIP_0000', help='Human-readable chip/die identifier for local coordinates.')
    p.add_argument('--chip-width-px', type=int, default=None, help='Known regular chip/die tile width; omit to treat the full search image as one chip/field.')
    p.add_argument('--chip-height-px', type=int, default=None, help='Known regular chip/die tile height; must be used with --chip-width-px.')
    a=p.parse_args()
    if a.scale_steps < 1: p.error('--scale-steps must be >= 1')
    if a.scale_min <= 0 or a.scale_max <= 0 or a.scale_min > a.scale_max: p.error('invalid scale range')
    r=locate(a.reference, a.search, ratios=np.linspace(a.scale_min, a.scale_max, a.scale_steps),
             roi_radius=a.roi_radius, score_window=a.score_window)
    if (a.chip_width_px is None) != (a.chip_height_px is None):
        p.error('provide both --chip-width-px and --chip-height-px, or neither')
    search_img = __import__('cv2').imread(str(a.search), __import__('cv2').IMREAD_GRAYSCALE)
    if search_img is None:
        p.error(f'could not decode search image: {a.search}')
    image_h, image_w = search_img.shape
    chip_w = int(a.chip_width_px or image_w)
    chip_h = int(a.chip_height_px or image_h)
    if chip_w <= 0 or chip_h <= 0:
        p.error('chip width/height must be positive')
    chip_col = min(max(int(np.floor(max(r.x, 0) / chip_w)), 0), max(0, int(np.ceil(image_w/chip_w))-1))
    chip_row = min(max(int(np.floor(max(r.y, 0) / chip_h)), 0), max(0, int(np.ceil(image_h/chip_h))-1))
    origin_x, origin_y = chip_col*chip_w, chip_row*chip_h
    selected_chip_id = (f'{a.chip_id}_R{chip_row:03d}_C{chip_col:03d}'
                        if chip_w < image_w or chip_h < image_h else a.chip_id)
    payload=r.to_dict()
    payload.update({'reference_path':str(a.reference.resolve()),'search_path':str(a.search.resolve()),
                    'reference_exists':a.reference.is_file(),'search_exists':a.search.is_file(),
                    'coordinate_system':'(x,y) pixel-center coordinates in the original search image; origin at top-left',
                    'image_width_px':image_w, 'image_height_px':image_h,
                    'chip_id':selected_chip_id, 'chip_row':chip_row, 'chip_col':chip_col,
                    'chip_origin_x_px':origin_x, 'chip_origin_y_px':origin_y,
                    'chip_width_px':min(chip_w,image_w-origin_x), 'chip_height_px':min(chip_h,image_h-origin_y),
                    'x_chip_local_px':float(r.x-origin_x), 'y_chip_local_px':float(r.y-origin_y),
                    'wafer_coordinates':'not available without wafer registration and physical calibration',
                    'score_semantics':'normalized cross-correlation score; not a probability',
                    'scale_configuration':{'min':a.scale_min,'max':a.scale_max,'steps':a.scale_steps}})
    out=json.dumps(payload, indent=2)
    print(out)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True); a.json_out.write_text(out+'\n')
        print(f'Wrote diagnostics JSON: {a.json_out}')
    return 0
if __name__=='__main__': raise SystemExit(main())