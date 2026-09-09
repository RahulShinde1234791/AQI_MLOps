import pandas as pd
import pytest

from src.data.validate import validate_dataframe


def test_valid_data_passes():
    data = pd.DataFrame({
        "City": ["Delhi"],
        "Date": ["2020-01-01"],
        "PM2.5": [50],
        "PM10": [80],
        "NO": [10],
        "NO2": [20],
        "NOx": [30],
        "NH3": [5],
        "CO": [0.5],
        "SO2": [10],
        "O3": [40],
        "Benzene": [1],
        "Toluene": [2],
        "Xylene": [3],
        "AQI": [75],
        "AQI_Bucket": ["Satisfactory"]
    })

    validate_dataframe(data)


def test_invalid_aqi_bucket_fails():
    data = pd.DataFrame({
        "City": ["Delhi"],
        "Date": ["2020-01-01"],
        "PM2.5": [50],
        "PM10": [80],
        "NO": [10],
        "NO2": [20],
        "NOx": [30],
        "NH3": [5],
        "CO": [0.5],
        "SO2": [10],
        "O3": [40],
        "Benzene": [1],
        "Toluene": [2],
        "Xylene": [3],
        "AQI": [75],
        "AQI_Bucket": ["Help"]
    })

    with pytest.raises(ValueError):
        validate_dataframe(data)


def test_duplicate_city_date_fails():
    data = pd.DataFrame({
        "City": ["Delhi", "Delhi"],
        "Date": ["2020-01-01", "2020-01-01"],
        "PM2.5": [50, 55],
        "PM10": [80, 82],
        "NO": [10, 11],
        "NO2": [20, 21],
        "NOx": [30, 31],
        "NH3": [5, 5],
        "CO": [0.5, 0.6],
        "SO2": [10, 11],
        "O3": [40, 41],
        "Benzene": [1, 1],
        "Toluene": [2, 2],
        "Xylene": [3, 3],
        "AQI": [75, 80],
        "AQI_Bucket": ["Satisfactory", "Moderate"]
    })

    with pytest.raises(ValueError):
        validate_dataframe(data)