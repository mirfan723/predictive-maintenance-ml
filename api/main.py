import os
import json
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure risk from sensor readings.",
)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "model_artifacts")

# ---- Load model + preprocessing artifacts at startup ----
try:
    model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
    label_encoder = joblib.load(os.path.join(ARTIFACTS_DIR, "label_encoder.pkl"))

    with open(os.path.join(ARTIFACTS_DIR, "feature_config.json")) as f:
        feature_config = json.load(f)

    FEATURE_COLS = feature_config["feature_cols"]
    NEEDS_SCALING = feature_config["needs_scaling"]
    MODEL_NAME = feature_config["model_name"]

except FileNotFoundError as e:
    raise FileNotFoundError(
        "Missing model artifacts. Copy 'model.pkl', 'scaler.pkl', 'label_encoder.pkl', "
        "and 'feature_config.json' (generated in your Kaggle notebook) into the "
        "'model_artifacts/' folder before running the app. See README.md."
    ) from e


# ---- Request/response schemas ----
class SensorReading(BaseModel):
    machine_type: str = Field(..., description="Product quality variant: 'L', 'M', or 'H'", examples=["M"])
    air_temperature_k: float = Field(..., description="Air temperature in Kelvin", examples=[300.5])
    process_temperature_k: float = Field(..., description="Process temperature in Kelvin", examples=[310.2])
    rotational_speed_rpm: float = Field(..., description="Rotational speed in rpm", examples=[1500])
    torque_nm: float = Field(..., description="Torque in Nm", examples=[40.0])
    tool_wear_min: float = Field(..., description="Tool wear in minutes", examples=[100])


class PredictionResponse(BaseModel):
    failure_predicted: bool
    failure_probability: float
    model_used: str


# ---- Routes ----
@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading):
    try:
        type_encoded = label_encoder.transform([reading.machine_type])[0]
    except ValueError:
        valid_types = [str(c) for c in label_encoder.classes_]
        raise HTTPException(
            status_code=400,
            detail=f"Unknown machine_type '{reading.machine_type}'. Expected one of: {valid_types}",
        )

    # Build the feature vector in the exact order the model was trained on
    raw_features = {
        "Type_encoded": type_encoded,
        "Air_temperature_K": reading.air_temperature_k,
        "Process_temperature_K": reading.process_temperature_k,
        "Rotational_speed_rpm": reading.rotational_speed_rpm,
        "Torque_Nm": reading.torque_nm,
        "Tool_wear_min": reading.tool_wear_min,
    }
    x = np.array([[raw_features[col] for col in FEATURE_COLS]])

    if NEEDS_SCALING:
        x = scaler.transform(x)

    prediction = bool(model.predict(x)[0])
    probability = float(model.predict_proba(x)[0][1])

    return PredictionResponse(
        failure_predicted=prediction,
        failure_probability=round(probability, 4),
        model_used=MODEL_NAME,
    )
