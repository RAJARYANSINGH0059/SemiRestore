from __future__ import annotations
import json, csv, sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inspection_reporting import _tile_info, detect_anomaly_candidates, render_inspection_overlay, create_inspection_report


def test_chip_local_coordinate_transform():
    tile = _tile_info(720.5, 160.25, 1000, 1000, 500, 400, "sampleA")
    assert tile["chip_id"] == "sampleA_R000_C001"
    assert tile["chip_origin_x_px"] == 500
    assert tile["chip_origin_y_px"] == 0
    assert tile["x_chip_local_px"] == 220.5
    assert tile["y_chip_local_px"] == 160.25


def test_anomaly_candidate_schema_is_unclassified():
    img = np.full((128, 128), 120, np.uint8)
    cv2.circle(img, (70, 65), 4, 235, -1)
    candidates = detect_anomaly_candidates(img, threshold_sigma=2.0, min_area=4, max_area=300)
    assert isinstance(candidates, list)
    if candidates:
        candidate = candidates[0]
        assert "x_image_px" in candidate and "y_image_px" in candidate
        assert candidate["defect_type"] == "UNCLASSIFIED"
        assert candidate["confirmation_status"] == "UNCONFIRMED_HEURISTIC_CANDIDATE"


def test_overlay_renders_prediction_and_ground_truth(tmp_path):
    ref = tmp_path / "ref.png"
    search = tmp_path / "search.png"
    cv2.imwrite(str(ref), np.full((64,64), 128, np.uint8))
    cv2.imwrite(str(search), np.full((128,128), 128, np.uint8))
    out = tmp_path / "overlay.jpg"
    render_inspection_overlay(ref, search, out, sample_id="S1", architecture="DRAM",
        pred_x=70, pred_y=65, score_ncc=0.87, scale_ratio=0.1,
        ground_truth_x=72, ground_truth_y=66, chip_local_x=70, chip_local_y=65,
        localization_error_px=float(np.hypot(2,1)))
    assert out.is_file()
    with Image.open(out) as im:
        assert im.width > 250 and im.height > 200


def test_single_pair_report_exports_coordinate_tables(tmp_path):
    # Build a tiny structurally valid project root with one DRAM and one FinFET pair.
    for arch in ("DRAM", "FinFET"):
        ref_dir = tmp_path / "results" / "baseline" / arch / "reference"
        search_dir = tmp_path / "results" / "baseline" / arch / "search"
        ref_dir.mkdir(parents=True)
        search_dir.mkdir(parents=True)
        ref = ref_dir / f"{arch.lower()}_0000.png"
        search = search_dir / f"{arch.lower()}_0000.png"
        rng = np.random.default_rng(11 if arch == "DRAM" else 12)
        base = np.tile(np.array([[0,20,0,20],[30,100,30,100]],np.uint8), (32,32))
        base = cv2.resize(base, (128,128))
        ref_img = cv2.resize(base[32:96,32:96], (64,64))
        # Keep the same motif and an explicit 10x-scale-compatible 640x640 image.
        search_img = cv2.resize(base, (640,640))
        cv2.imwrite(str(ref), ref_img)
        cv2.imwrite(str(search), search_img)
        gt = tmp_path / "results" / "baseline" / arch / "ground_truth.csv"
        gt.write_text("id,architecture,reference,search,x,y,reference_size,search_size,seed\n"
                      f"{arch.lower()}_0000,{arch},{ref.relative_to(tmp_path)},{search.relative_to(tmp_path)},320,320,64,640,11\n")
    out = tmp_path / "results" / "inspection_report"
    report = create_inspection_report(tmp_path, out, scan_anomalies=False)
    assert report["sample_count"] == 2
    for name in ("inspection_sites.csv", "chip_summary.csv", "anomaly_candidates.csv", "inspection_report.json", "sample_inspection_overlay.jpg"):
        assert (out / name).is_file()
    with (out / "inspection_sites.csv").open(newline="") as f:
        sites = list(csv.DictReader(f))
    assert len(sites) == 2
    assert sites[0]["pred_x_chip_local_px"] != ""
    with (out / "chip_summary.csv").open(newline="") as f:
        chips = list(csv.DictReader(f))
    assert len(chips) == 2
    saved = json.loads((out / "inspection_report.json").read_text())
    assert "defect_semantics" in saved
    assert "not computed" in saved["coordinate_semantics"]["wafer_coordinates"]


def test_input_manifest_without_ground_truth_is_supported(tmp_path):
    root = tmp_path / "project"
    ref = root / "external" / "ref.png"
    search = root / "external" / "search.png"
    ref.parent.mkdir(parents=True)
    rng = np.random.default_rng(3)
    base = np.tile(np.array([[30, 60, 120, 60], [80, 160, 80, 160]], np.uint8), (32, 32))
    base = cv2.resize(base, (128, 128))
    cv2.imwrite(str(ref), cv2.resize(base[32:96, 32:96], (64, 64)))
    cv2.imwrite(str(search), cv2.resize(base, (640, 640)))
    manifest = root / "data" / "inspection_manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "sample_id,lot_id,wafer_id,architecture,die_row,die_col,chip_id,reference,search,gt_x,gt_y\n"
        f"SAMPLE001,LOT_A,WAFER_02,DRAM,3,4,DIE_03_04,{ref},{search},,\n"
    )
    out = root / "results" / "inspection_report"
    report = create_inspection_report(root, out, input_manifest=manifest, scan_anomalies=False,
                                      chip_width_px=320, chip_height_px=320)
    assert report["sample_count"] == 1
    assert report["ground_truth_rows"] == 0
    assert report["ground_truth_unavailable_rows"] == 1
    with (out / "inspection_sites.csv").open(newline="") as f:
        site = next(csv.DictReader(f))
    assert site["wafer_id"] == "WAFER_02"
    assert site["chip_id"].startswith("DIE_03_04_R")
    assert site["localization_error_px"] == ""
    assert site["ground_truth_x_chip_local_px"] == "NOT_PROVIDED"
    with (out / "chip_summary.csv").open(newline="") as f:
        chips = list(csv.DictReader(f))
    assert len(chips) == 4
    assert all(r["defect_assessment_status"] == "NOT_RUN" for r in chips)


def test_expert_defect_annotations_are_exported_with_chip_local_coordinates(tmp_path):
    # Exercise the optional confirmed-label path independently of heuristic candidates.
    for arch in ("DRAM", "FinFET"):
        ref_dir = tmp_path / "results" / "baseline" / arch / "reference"
        search_dir = tmp_path / "results" / "baseline" / arch / "search"
        ref_dir.mkdir(parents=True)
        search_dir.mkdir(parents=True)
        ref = ref_dir / f"{arch.lower()}_0000.png"
        search = search_dir / f"{arch.lower()}_0000.png"
        base = np.tile(np.array([[0, 40, 10, 50], [20, 90, 30, 100]], np.uint8), (32, 32))
        base = cv2.resize(base, (128, 128))
        cv2.imwrite(str(ref), cv2.resize(base[32:96, 32:96], (64, 64)))
        cv2.imwrite(str(search), cv2.resize(base, (640, 640)))
        gt = tmp_path / "results" / "baseline" / arch / "ground_truth.csv"
        gt.write_text("id,architecture,reference,search,x,y,reference_size,search_size,seed\n"
                      f"{arch.lower()}_0000,{arch},{ref.relative_to(tmp_path)},{search.relative_to(tmp_path)},320,320,64,640,11\n")
    annotations = tmp_path / "data" / "reviewed_defects.csv"
    annotations.parent.mkdir(parents=True)
    annotations.write_text(
        "annotation_id,sample_id,x_image_px,y_image_px,defect_type,severity,confirmed,annotator,notes\n"
        "A1,dram_0000,100,120,particle,MEDIUM,true,reviewer_a,confirmed under review\n"
        "A2,dram_0000,510,515,scratch,HIGH,false,reviewer_a,needs second review\n"
    )
    out = tmp_path / "results" / "inspection_with_labels"
    report = create_inspection_report(tmp_path, out, scan_anomalies=False,
        chip_width_px=320, chip_height_px=320, defect_annotations_csv=annotations)
    with (out / "reviewed_defect_annotations.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["defect_type"] == "particle"
    assert rows[0]["confirmed"] == "True"
    assert rows[0]["confirmation_status"] == "CONFIRMED_EXPERT_LABEL"
    assert rows[0]["x_chip_local_px"] == "100.0"
    assert rows[1]["chip_col"] == "1"
    assert rows[1]["x_chip_local_px"] == "190.0"
    assert report["expert_defect_annotations_loaded"] == 2
    assert report["confirmed_expert_defect_annotations"] == 1