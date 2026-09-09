import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = "models/logistic_baseline.joblib"

model = None

def get_model():
    global model

    if model is None:
        model = joblib.load(MODEL_PATH)

    return model

app = FastAPI(
    title="AQI Next-Day Prediction API",
    description="Predicts tomorrow's AQI category from today's air-quality data.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    City: str = Field(..., min_length=1)

    PM_2_5: float | None = None
    NO: float | None = None
    NO2: float | None = None
    NOx: float | None = None
    CO: float | None = None
    SO2: float | None = None
    O3: float | None = None
    Benzene: float | None = None
    Toluene: float | None = None

    month: int = Field(..., ge=1, le=12)
    day_of_week: int = Field(..., ge=0, le=6)


@app.get("/")
def root():
    return {"message": "AQI prediction API is running."}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: PredictionRequest):
    data = pd.DataFrame([{
        "City": request.City,
        "PM2.5": request.PM_2_5,
        "NO": request.NO,
        "NO2": request.NO2,
        "NOx": request.NOx,
        "CO": request.CO,
        "SO2": request.SO2,
        "O3": request.O3,
        "Benzene": request.Benzene,
        "Toluene": request.Toluene,
        "month": request.month,
        "day_of_week": request.day_of_week,
    }])

    prediction = get_model().predict(data)[0]

    return {
        "predicted_aqi_bucket": prediction
    }