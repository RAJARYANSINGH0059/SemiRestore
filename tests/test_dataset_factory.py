import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SCRIPT = ROOT / "external" / "unified_dataset_factory" / "scripts" / "make_splits.py"

def test_make_splits_groups_by_source_under_sources(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    records = [
        {"id": "a1", "structure_source": "data/sources/refics/a.png"},
        {"id": "a2", "structure_source": "/content/drive/MyDrive/data/sources/refics/b.png"},
        {"id": "b1", "structure_source": "data/sources/carinthia/c.png"},
        {"id": "c1", "structure_source": "/content/data/sources/finfet/d.png"},
        {"id": "d1", "structure_source": "data/sources/bath_periodic/e.png"},
        {"id": "e1", "structure_source": "data/sources/gaas_periodic/f.png"},
    ]
    metadata.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    out = tmp_path / "splits.json"
    subprocess.run([sys.executable, str(SPLIT_SCRIPT), "--metadata", str(metadata), "--out", str(out)], check=True)
    split = json.loads(out.read_text())
    sources = set(split["train_sources"] + split["val_sources"] + split["ood_sources"])
    assert sources == {"refics", "carinthia", "finfet", "bath_periodic", "gaas_periodic"}
    assert "data" not in sources
    assert "content" not in sources
    all_ids = split["train_ids"] + split["val_ids"] + split["ood_ids"]
    assert sorted(all_ids) == sorted(r["id"] for r in records)