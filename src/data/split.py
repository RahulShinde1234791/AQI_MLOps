import pandas as pd


TRAIN_END_DATE = "2020-01-01"


def chronological_split(df):
    train = df[df["Date"] < TRAIN_END_DATE].copy()
    test = df[df["Date"] >= TRAIN_END_DATE].copy()

    return train, test


if __name__ == "__main__":
    df = pd.read_csv("data/processed/forecast_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])

    train, test = chronological_split(df)

    print("Training rows:", len(train))
    print("Testing rows:", len(test))

    print("\nTraining date range:")
    print(train["Date"].min(), "→", train["Date"].max())

    print("\nTesting date range:")
    print(test["Date"].min(), "→", test["Date"].max())