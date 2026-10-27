from __future__ import annotations
import csv, json, subprocess, sys
from pathlib import Path
import cv2
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.nova_inference import locate

REF=ROOT/'results'/'baseline'/'DRAM'/'reference'/'dram_0000.png'
SEARCH=ROOT/'results'/'baseline'/'DRAM'/'search'/'dram_0000.png'

def test_sample_files_are_present_and_decodable():
    assert REF.is_file() and SEARCH.is_file()
    assert cv2.imread(str(REF), cv2.IMREAD_GRAYSCALE) is not None
    assert cv2.imread(str(SEARCH), cv2.IMREAD_GRAYSCALE) is not None

def test_locate_repeatable_on_same_pair():
    a=locate(REF,SEARCH); b=locate(REF,SEARCH)
    assert (a.x,a.y,a.score,a.chosen_scale_ratio,a.status)==(b.x,b.y,b.score,b.chosen_scale_ratio,b.status)
    assert 0 <= a.x < 1000 and 0 <= a.y < 1000

def test_centered_generator_sample_coordinates_near_center():
    result=locate(REF,SEARCH)
    assert np.hypot(result.x-500,result.y-500) <= 5

def test_missing_reference_raises_instead_of_reusing_old_image():
    with pytest.raises(FileNotFoundError): locate(ROOT/'does-not-exist.png', SEARCH)

def test_invalid_configuration_rejected():
    with pytest.raises(ValueError): locate(REF,SEARCH,roi_radius=-1)
    with pytest.raises(ValueError): locate(REF,SEARCH,score_window=1.5)
    with pytest.raises(ValueError): locate(REF,SEARCH,ratios=np.array([]))

def test_evaluator_cli_prints_one_coordinate_pair():
    p=subprocess.run([sys.executable,str(ROOT/'src'/'infer.py'),str(REF),str(SEARCH)],cwd=ROOT,capture_output=True,text=True,check=True)
    parts=p.stdout.strip().split()
    assert len(parts)==2
    assert all(np.isfinite(float(x)) for x in parts)
    assert p.stderr==''

def test_merged_holdout_ledger_is_complete():
    path=ROOT/'results'/'integrated_benchmark'/'integrated_per_sample_results.csv'
    assert path.is_file()
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    assert len(rows)==480
    for split in ('holdout_centered','holdout_translated'):
        rr=[r for r in rows if r['split']==split]
        assert len({r['id'] for r in rr})==60
        assert len(rr)==240
    assert {r['method'] for r in rows}=={'B1_fixed_ncc','B2_multiscale_ncc','B3_center_aware_ncc','NOVA_hybrid_roi_gate'}

def test_benchmark_summary_reports_drift_degradation():
    p=ROOT/'results'/'integrated_benchmark'/'integrated_metrics_summary.json'
    d=json.loads(p.read_text())
    centered=d['holdout_centered']['NOVA_hybrid_roi_gate']['ALL']['accuracy_5px']
    shifted=d['holdout_translated']['NOVA_hybrid_roi_gate']['ALL']['accuracy_5px']
    assert centered==1.0
    assert shifted < centered


def test_diagnostics_cli_emits_json_and_writes_report(tmp_path):
    out = tmp_path / "diagnostics.json"
    p = subprocess.run(
        [sys.executable, str(ROOT / "src" / "infer_diagnostics.py"),
         str(REF), str(SEARCH), "--chip-id", "TESTCHIP", "--chip-width-px", "500", "--chip-height-px", "500",
         "--json-out", str(out)],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    payload = json.loads(p.stdout.split("Wrote diagnostics JSON:")[0])
    assert payload["coordinate_system"].startswith("(x,y)")
    assert payload["reference_exists"] and payload["search_exists"]
    assert "score_semantics" in payload
    saved = json.loads(out.read_text())
    assert saved["x"] == payload["x"] and saved["y"] == payload["y"]