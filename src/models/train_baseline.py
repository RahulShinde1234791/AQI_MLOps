import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.data.split import chronological_split


INPUT_FILE = "data/processed/forecast_dataset.csv"
MODEL_FILE = "models/logistic_baseline.joblib"


NUMERIC_FEATURES = [
    "PM2.5",
    "NO",
    "NO2",
    "NOx",
    "CO",
    "SO2",
    "O3",
    "Benzene",
    "Toluene",
    "month",
    "day_of_week",
]

CATEGORICAL_FEATURES = ["City"]


def build_pipeline():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = LogisticRegression(
        max_iter=1000
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    return pipeline


def main():
    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    train, test = chronological_split(df)

    X_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train["target"]

    X_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test["target"]

    pipeline = build_pipeline()

    print("Training model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    joblib.dump(pipeline, MODEL_FILE)

    print(f"\nSaved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()