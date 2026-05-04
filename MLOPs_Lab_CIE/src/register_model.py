import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent

(ROOT / "results").mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")

model_name = "gallerypulse-auction-price-lakhs-predictor"

with open(ROOT / "results" / "step2_s2.json", encoding="utf-8") as f:
    step2 = json.load(f)

run_id = step2["parent_run_id"]
source_metric_value = float(step2["best_cv_mae"])
# MLflow 3: use models:/<model_id> from log_model; legacy file store used runs:/<run_id>/model.
model_uri = step2.get("register_model_uri") or f"runs:/{run_id}/model"

result = mlflow.register_model(
    model_uri,
    model_name,
)

out = {
    "registered_model_name": model_name,
    "version": int(result.version),
    "run_id": run_id,
    "source_metric": "mae",
    "source_metric_value": source_metric_value,
}

with open(ROOT / "results" / "step4_s6.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("Task 4 done")
