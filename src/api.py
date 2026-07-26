

import os
import time
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = "models/model.pkl"
MAX_RETRIES = 10
RETRY_DELAY_SECONDS = 5

app = FastAPI(title="Auto MPG Prediction API")


def load_model_with_retry():
    """
    Wait for the trained model file to exist before loading it.
    This handles the case where the API container starts before
    the trainer container has finished producing model.pkl.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                return pickle.load(f)

        print(f"Model not found yet. Retry {attempt}/{MAX_RETRIES} in {RETRY_DELAY_SECONDS}s...")
        time.sleep(RETRY_DELAY_SECONDS)

    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH} after {MAX_RETRIES} retries."
    )


# ----- Load the trained model once, at startup -----
model = load_model_with_retry()


# ----- Request schema -----
class CarFeatures(BaseModel):
    weight: float
    model_year: int
    acceleration: float


# ----- Health check endpoint -----
@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "ok"}


# ----- Prediction endpoint -----
@app.post("/predict")
def predict_mpg(car: CarFeatures):
    """Predict MPG for a given car based on weight, model_year, and acceleration."""
    input_data = pd.DataFrame([{
        "weight": car.weight,
        "model_year": car.model_year,
        "acceleration": car.acceleration
    }])

    prediction = model.predict(input_data)[0]

    return {"predicted_mpg": round(float(prediction), 2)}


