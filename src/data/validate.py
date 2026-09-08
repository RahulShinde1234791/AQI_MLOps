import pandas as pd

REQUIRED_COLUMNS = [
    "City", "Date", "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3", "Benzene", "Toluene",
    "Xylene", "AQI", "AQI_Bucket"
]

df = pd.read_csv("city_day.csv")

missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

print("Schema validation passed.")

ALLOWED_AQI_BUCKETS = {
    "Good",
    "Satisfactory",
    "Moderate",
    "Poor",
    "Very Poor",
    "Severe"
}

actual_buckets = set(df["AQI_Bucket"].dropna().unique())

unexpected_buckets = actual_buckets - ALLOWED_AQI_BUCKETS

if unexpected_buckets:
    raise ValueError(
        f"Unexpected AQI_Bucket values: {unexpected_buckets}"
    )

print("AQI bucket validation passed.")

duplicate_rows = df.duplicated(subset=["City", "Date"]).sum()

if duplicate_rows > 0:
    raise ValueError(
        f"Found {duplicate_rows} duplicate City-Date rows."
    )

print("City-Date uniqueness validation passed.")

if df["City"].isna().any():
    raise ValueError("Missing City values found.")

print("City validation passed.")