import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid, cross_val_score, KFold, train_test_split

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(ROOT / "data" / "training_data.csv")

X = df.drop("auction_price_lakhs", axis=1)
y = df["auction_price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "n_estimators": [50, 150, 250],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 3, 5],
}

mlflow.set_tracking_uri(f"file:{ROOT / 'mlruns'}")
mlflow.set_experiment("gallerypulse-auction-price-lakhs")

(ROOT / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "results").mkdir(parents=True, exist_ok=True)

total_trials = len(list(ParameterGrid(param_grid)))
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

best_cv_mae = float("inf")
best_params = None

with mlflow.start_run(run_name="tuning-gallerypulse") as parent:
    parent_run_id = parent.info.run_id

    for i, params in enumerate(ParameterGrid(param_grid)):
        with mlflow.start_run(run_name=f"trial-{i}", nested=True):
            model = RandomForestRegressor(random_state=42, **params)
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=kfold,
                scoring="neg_mean_absolute_error",
            )
            cv_mae = float(-scores.mean())
            mlflow.log_params(params)
            mlflow.log_metric("cv_mae", cv_mae)
            if cv_mae < best_cv_mae:
                best_cv_mae = cv_mae
                best_params = dict(params)

    best_model = RandomForestRegressor(random_state=42, **best_params)
    best_model.fit(X_train, y_train)
    mlflow.log_metric("best_cv_mae", best_cv_mae)
    # MLflow 3 logs models under models:/<model_id>; runs:/.../model is not valid for register_model.
    model_info = mlflow.sklearn.log_model(best_model, name="model")

    with open(ROOT / "models" / "model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    out = {
        "search_type": "grid",
        "n_folds": 5,
        "total_trials": total_trials,
        "best_params": best_params,
        "best_mae": best_cv_mae,
        "best_cv_mae": best_cv_mae,
        "parent_run_name": "tuning-gallerypulse",
        "parent_run_id": parent_run_id,
        "register_model_uri": model_info.model_uri,
    }
    with open(ROOT / "results" / "step2_s2.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

print("Task 2 done")
