## Baseline Experiment

### Experiment 1 — Logistic Regression

- Features: 9 pollutant features + month + day_of_week + City
- Feature coverage threshold: 70%
- Imputation: median for numerical features, most-frequent for City
- Train period: 2015–2019
- Test period: 2020
- Accuracy: 0.6627
- Macro F1: 0.58

### Observations

- Overall accuracy was 66.27%.
- Good class had extremely poor recall (0.01).
- Macro F1 was considerably lower than accuracy.
- Logistic Regression produced a convergence warning.
- Next experiment: add StandardScaler to numerical features.

### Experiment 2 — Logistic Regression + StandardScaler

- Model: Logistic Regression
- Features: same as Experiment 1
- Coverage threshold: 70%
- Imputation: median / most-frequent
- Scaling: StandardScaler for numerical features
- Train period: 2015–2019
- Test period: 2020
- Accuracy: 0.7086
- Macro F1: 0.66

### Comparison with Experiment 1

- Accuracy improved from 0.6627 to 0.7086.
- Macro F1 improved from 0.58 to 0.66.
- Good-class recall improved from 0.01 to 0.32.
- The Logistic Regression convergence warning disappeared.

### Registry-backed container inference

The FastAPI inference service runs inside a Docker container and
loads the model through the MLflow Model Registry using the
`@champion` alias.

Verified:
- Docker container starts successfully.
- `/health` returns HTTP 200.
- `/predict` returns HTTP 200.
- Model is loaded from `AQI_NextDay_Classifier@champion`.