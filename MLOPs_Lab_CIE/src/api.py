from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle

FEATURE_COLS = [
    "artist_reputation_score",
    "artwork_age_years",
    "medium_type_index",
    "exhibition_count",
]

ROOT = Path(__file__).resolve().parent.parent

app = FastAPI()

with open(ROOT / "models" / "model.pkl", "rb") as f:
    model = pickle.load(f)

MODEL_DISPLAY_NAME = type(model).__name__


class InputData(BaseModel):
    artist_reputation_score: float = Field(..., ge=1, le=10)
    artwork_age_years: int = Field(..., ge=1, le=200)
    medium_type_index: int = Field(..., ge=1, le=5)
    exhibition_count: int = Field(..., ge=0, le=20)


@app.get("/status")
def status():
    return {
        "status": "running",
        "model": MODEL_DISPLAY_NAME,
        "version": "1.0",
    }


@app.post("/forecast")
def forecast(data: InputData):
    X_in = pd.DataFrame(
        [[
            data.artist_reputation_score,
            data.artwork_age_years,
            data.medium_type_index,
            data.exhibition_count,
        ]],
        columns=FEATURE_COLS,
    )
    pred = model.predict(X_in)[0]
    return {"prediction": float(pred)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
