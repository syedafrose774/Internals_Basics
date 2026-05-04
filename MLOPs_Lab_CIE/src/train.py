import pandas as pd
import numpy as np
import json
import pickle
import mlflow
import mlflow.sklearn
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent

# load data
df = pd.read_csv(ROOT / "data" / "training_data.csv")

X = df.drop("auction_price_lakhs", axis=1)
y = df["auction_price_lakhs"]

# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# mlflow setup
mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
mlflow.set_experiment("gallerypulse-auction-price-lakhs")

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge()
}

results = []

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("experiment_type", "baseline_comparison")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mape", mape)

        mlflow.sklearn.log_model(model, name=name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        })

# best model by MAE
best = min(results, key=lambda x: x["mae"])

# save model
(ROOT / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(parents=True, exist_ok=True)
with open(ROOT / "models" / "model.pkl", "wb") as f:
    pickle.dump(models[best["name"]], f)

# save result json
json.dump({
    "experiment_name": "gallerypulse-auction-price-lakhs",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "mae",
    "best_metric_value": best["mae"]
}, open(ROOT / "results" / "step1_s1.json", "w", encoding="utf-8"), indent=2)

print("Task 1 done")