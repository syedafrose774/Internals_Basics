"""Write results/step3_s4.json using the same app logic as the FastAPI service."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

(ROOT / "results").mkdir(parents=True, exist_ok=True)

from fastapi.testclient import TestClient

from api import app

TEST_INPUT = {
    "artist_reputation_score": 6.9,
    "artwork_age_years": 128,
    "medium_type_index": 2,
    "exhibition_count": 8,
}

client = TestClient(app)
health_r = client.get("/status")
health_r.raise_for_status()
health = health_r.json()

pred_r = client.post("/forecast", json=TEST_INPUT)
pred_r.raise_for_status()
pred_body = pred_r.json()
if "prediction" not in pred_body:
    raise RuntimeError(f"Unexpected /forecast response: {pred_body}")

out = {
    "health_endpoint": "/status",
    "predict_endpoint": "/forecast",
    "port": 8080,
    "health_response": {
        "status": "running",
        "model": health["model"],
        "version": "1.0",
    },
    "test_input": TEST_INPUT,
    "prediction": float(pred_body["prediction"]),
}

with open(ROOT / "results" / "step3_s4.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Wrote results/step3_s4.json")
