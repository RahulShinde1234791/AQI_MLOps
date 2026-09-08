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

# Pollutant columns to evaluate
pollutant_columns = [
    "PM2.5", "PM10", "NO", "NO2", "NH3", "SO2", "CO", "O3",
    "Benzene", "Toluene", "Xylene", "AQI"
]

# Keep only pollutant columns that exist in the file
available_pollutants = [col for col in pollutant_columns if col in city_day_df.columns]

# Calculate percentage of missing values per pollutant
missing_percent = (city_day_df[available_pollutants].isna().mean() * 100).sort_values(ascending=False)

print("Percentage of missing values by pollutant:")
print(missing_percent)

# Also compute missing percentages per city (handle possible column name variations)
city_col = "City"

if city_col is not None:
    missing_percent_by_city = (
        city_day_df.groupby(city_col)[available_pollutants]
        .apply(lambda df: df.isna().mean() * 100)
    )
    print("\nPercentage of missing values by pollutant (city-wise):")
    print(missing_percent_by_city)
else:
    print("\nNo city column found to compute city-wise missing percentages.")

missing_by_city = (
    city_day_df.groupby("City")[available_pollutants]
    .apply(lambda df: df.isna().mean() * 100)
)

print(missing_by_city.max().sort_values(ascending=False))
print(missing_by_city["Xylene"].sort_values(ascending=False))

aqi_missing_by_city = (
    city_day_df.groupby("City")["AQI"]
    .apply(lambda x: x.isna().mean() * 100)
    .sort_values(ascending=False)
)

print(aqi_missing_by_city)

bucket_missing_by_city = (
    city_day_df.groupby("City")["AQI_Bucket"]
    .apply(lambda x: x.isna().sum())
    .sort_values(ascending=True)
)

print(bucket_missing_by_city.tail())

total = city_day_df.groupby("City")["AQI_Bucket"].size()
usable = city_day_df.groupby("City")["AQI_Bucket"].count()

coverage = usable / total * 100

print(coverage.sort_values())

mumbai = city_day_df[city_day_df["City"] == "Mumbai"].copy()

mumbai["Date"] = pd.to_datetime(mumbai["Date"])

print(
    mumbai.groupby(mumbai["Date"].dt.year)["AQI_Bucket"]
    .apply(lambda x: x.isna().mean() * 100)
)

city_day_df["Date"] = pd.to_datetime(city_day_df["Date"])

date_range = city_day_df.groupby("City")["Date"].agg(["min", "max"])

print(date_range)

city_stats = city_day_df.groupby("City")["Date"].agg(["min", "max", "count"])

city_stats["expected_days"] = (
    city_stats["max"] - city_stats["min"]
).dt.days + 1

city_stats["coverage_percent"] = (
    city_stats["count"] / city_stats["expected_days"] * 100
)

print(city_stats.sort_values("coverage_percent"))

missing_dates = []

for city, group in city_day_df.groupby("City"):
    dates = group["Date"].sort_values().reset_index(drop=True)
    expected = pd.Series(
        pd.date_range(dates.min(), dates.max())
    )

    if len(dates) != len(expected) or not dates.equals(expected):
        missing_dates.append(city)

print(missing_dates)