import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

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

    mlflow.set_experiment("AQI Forecasting")

    with mlflow.start_run(run_name="Logistic Regression - Scaled"):

        pipeline = build_pipeline()

        print("Training model...")
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True
        )

        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("coverage_threshold", 0.70)
        mlflow.log_param("scaling", "StandardScaler")
        mlflow.log_param("numeric_imputation", "median")
        mlflow.log_param("categorical_imputation", "most_frequent")
        mlflow.log_param("train_end", "2020-01-01")

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("weighted_f1", weighted_f1)

        # Log per-class F1
        for class_name in [
            "Good",
            "Moderate",
            "Poor",
            "Satisfactory",
            "Severe",
            "Very Poor"
        ]:
            mlflow.log_metric(
                f"f1_{class_name.lower().replace(' ', '_')}",
                report[class_name]["f1-score"]
            )

        # Save model locally
        joblib.dump(pipeline, MODEL_FILE)

        # Log model to MLflow
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="logistic_baseline",
            skops_trusted_types=["numpy.dtype"]
        )

        print(f"\nAccuracy: {accuracy:.4f}")

        print("\nClassification report:")
        print(classification_report(y_test, predictions))

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions))

        print(f"\nSaved model to: {MODEL_FILE}")


if __name__ == "__main__":
    main()