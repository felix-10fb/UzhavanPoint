# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

MODEL_FILE = "crop_recommendation.pkl"

def ensure_model():
    if not os.path.exists(MODEL_FILE):
        # Attempt to generate the model by running the training script (train.py)
        try:
            # Importing train will execute its top-level training code and save the model
            import train  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"Model file '{MODEL_FILE}' not found and training failed: {e}")

    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

model = ensure_model()


class FarmInputs(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/predict_crop")
def predict_crop(inputs: FarmInputs):
    data = np.array([[
        inputs.N, inputs.P, inputs.K,
        inputs.temperature, inputs.humidity,
        inputs.ph, inputs.rainfall
    ]])

    probs = model.predict_proba(data)[0]
    top_indices = probs.argsort()[-3:][::-1]

    recommendations = []
    for idx in top_indices:
        recommendations.append({
            "crop": model.classes_[idx],
            "confidence": round(float(probs[idx]) * 100, 2)
        })

    return {"status": "success", "recommendations": recommendations}