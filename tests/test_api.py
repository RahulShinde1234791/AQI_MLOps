from fastapi.testclient import TestClient

from src.api.app import app

class FakeModel:
    def predict(self, data):
        return ["Moderate"]
    
client = TestClient(app)

def valid_request():
    return {
        "City": "Delhi",
        "PM_2_5": 85.2,
        "NO": 18.1,
        "NO2": 42.3,
        "NOx": 58.7,
        "CO": 0.8,
        "SO2": 12.4,
        "O3": 76.5,
        "Benzene": 2.1,
        "Toluene": 5.4,
        "month": 9,
        "day_of_week": 2
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_endpoint(monkeypatch):
    from src.api import app as app_module

    monkeypatch.setattr(
        app_module,
        "model",
        FakeModel()
    )

    response = client.post(
        "/predict",
        json=valid_request()
    )

    assert response.status_code == 200

    result = response.json()

    assert result["predicted_aqi_bucket"] == "Moderate"


def test_invalid_month_rejected():
    request = valid_request()
    request["month"] = 13

    response = client.post(
        "/predict",
        json=request
    )

    assert response.status_code == 422