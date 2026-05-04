"""
Run all CIE tasks in order (from the MLOPs_Lab_CIE folder):

    python run_pipeline.py

Then start MLflow UI or the API using the commands printed at the end.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ROOT / "src" / "train.py",
    ROOT / "src" / "tune.py",
    ROOT / "src" / "record_step3.py",
    ROOT / "src" / "register_model.py",
]


def main() -> None:
    for script in STEPS:
        print(f"\n>>> Running {script.relative_to(ROOT)} ...\n")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

    print("\n" + "=" * 60)
    print("All tasks finished. Proof JSON is in: results/")
    print("Saved model: models/model.pkl")
    print("MLflow tracking data: mlruns/  (local file store)")
    print("=" * 60)
    print("\n--- MLflow UI (experiments, runs, metrics, registry) ---")
    print(f"1. Open a NEW terminal and run:\n   cd \"{ROOT}\"")
    print("   mlflow ui")
    print("2. In your browser open:  http://127.0.0.1:5000")
    print("   (Pick experiment: gallerypulse-auction-price-lakhs)")
    print("\n--- FastAPI (live /status and /forecast) ---")
    print(f"   cd \"{ROOT}\"")
    print(
        "   python -m uvicorn api:app --app-dir src --host 127.0.0.1 --port 8080"
    )
    print("   Browser: http://127.0.0.1:8080/status")
    print("   POST JSON to: http://127.0.0.1:8080/forecast\n")


if __name__ == "__main__":
    main()
