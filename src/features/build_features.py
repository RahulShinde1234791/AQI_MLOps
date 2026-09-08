import os
import pandas as pd

from src.data.feature_selection import select_features


INPUT_FILE = "city_day.csv"
OUTPUT_FILE = "data/processed/forecast_dataset.csv"


def build_forecasting_dataset(df):
    # Make sure dates are actual datetime values
    df["Date"] = pd.to_datetime(df["Date"])

    # Sort chronologically within each city
    df = df.sort_values(["City", "Date"]).reset_index(drop=True)

    # Select pollutant features using our 70% coverage policy
    selected_features, _ = select_features(df)

    # Create temporal features
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    # Tomorrow's AQI category becomes today's target
    df["target"] = (
        df.groupby("City")["AQI_Bucket"]
        .shift(-1)
    )

    # Keep only the information available TODAY
    feature_columns = (
        ["City", "Date"]
        + selected_features
        + ["month", "day_of_week"]
    )

    result = df[feature_columns + ["target"]].copy()

    # Rows where tomorrow's target is unknown cannot be training examples
    result = result.dropna(subset=["target"])

    return result


def main():
    df = pd.read_csv(INPUT_FILE)

    result = build_forecasting_dataset(df)

    os.makedirs("data/processed", exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    print(f"Created: {OUTPUT_FILE}")
    print(f"Rows: {len(result)}")
    print(f"Columns: {len(result.columns)}")
    print("\nFeatures:")
    print(result.columns.tolist())
    print("\nTarget distribution:")
    print(result["target"].value_counts())


if __name__ == "__main__":
    main()