import pandas as pd

POLLUTANTS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx",
    "NH3", "CO", "SO2", "O3",
    "Benzene", "Toluene", "Xylene"
]

MIN_COVERAGE = 0.70


def select_features(df):
    coverage = df[POLLUTANTS].notna().mean()

    selected = coverage[coverage >= MIN_COVERAGE].index.tolist()

    return selected, coverage

df = pd.read_csv("city_day.csv")

selected_features, coverage = select_features(df)

print("Feature coverage:")
print((coverage * 100).sort_values(ascending=False))

print("\nSelected features:")
print(selected_features)