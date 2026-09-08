import pandas as pd
from pathlib import Path

# Locate the CSV file in the same folder as this script
base_dir = Path(__file__).resolve().parent
csv_path = None
for candidate in [base_dir / "city_day.csv", base_dir / "City_day.csv"]:
    if candidate.exists():
        csv_path = candidate
        break

if csv_path is None:
    raise FileNotFoundError("city_day.csv not found in the project folder.")

# Read the dataset
city_day_df = pd.read_csv(csv_path)

def analyze_pollutant_features(df):
    # Pollutant columns to evaluate
    pollutant_columns = [
        "PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "O3",
        "Benzene", "Toluene", "Xylene", "AQI"
    ]

    # Keep only pollutant columns that exist in the file
    available_pollutants = [col for col in pollutant_columns if col in df.columns]

    if not available_pollutants:
        print("No pollutant columns were found in the dataset.")
        return

    # Count actual (non-missing) measurements for each pollutant
    actual_measurements = df[available_pollutants].count()

    print("Number of actual (non-missing) measurements by pollutant:")
    print(actual_measurements.sort_values(ascending=False))

    # Also provide city-wise counts if a city column exists
    city_col = None
    for candidate in ["city", "City", "City_name", "City_Name", "city_name"]:
        if candidate in df.columns:
            city_col = candidate
            break

    if city_col:
        # Group by city and count non-missing values for each pollutant
        city_counts = df.groupby(city_col)[available_pollutants].count()
        # Show total measurements per city (sum across pollutants) to get busiest cities
        city_counts["total_measurements"] = city_counts.sum(axis=1)
        city_counts = city_counts.sort_values("total_measurements", ascending=False)

        print(f"\nTop cities by number of pollutant measurements (showing top 20):")
        print(city_counts.head(20))
    else:
        print("\nNo city column found; skipping city-wise breakdown.")


# Run the integrated pollutant analysis directly on the loaded dataset
analyze_pollutant_features(city_day_df)
