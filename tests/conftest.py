"""Pytest bootstrap: create only missing fixed-seed synthetic sample images."""
from pathlib import Path
import subprocess, sys
ROOT = Path(__file__).resolve().parents[1]
def pytest_sessionstart(session):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_synthetic_dataset.py")], cwd=ROOT, check=True)