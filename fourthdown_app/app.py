from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_PATH = Path(__file__).parent / "data" / "went_for_it_features.csv"
MODEL_PATH = Path(__file__).parent / "models" / "logit_4th_down_pipeline.joblib"

NUMERIC_FEATURES = [
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "off_epa_db",
    "off_epa_rush",
    "def_epa_db_allowed",
    "def_epa_rush_allowed",
]

CATEGORICAL_FEATURES = ["qtr"]


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing data file at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    train_df = df[df["season"].between(2019, 2023)]
    test_df = df[df["season"] == 2024]

    if train_df.empty or test_df.empty:
        raise ValueError("Expected data for seasons 2019-2023 (train) and 2024 (test)")

    X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df["fourth_down_converted"]
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df["fourth_down_converted"]

    return X_train, y_train, X_test, y_test


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
    )

    model = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])
    return model


def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_data()
    model = build_pipeline()
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")
    print(f"Test AUC: {auc:.3f}")
    print("Confusion matrix (threshold=0.5):")
    print(cm)
    print(f"Accuracy:  {acc:.4f}")


if __name__ == "__main__":
    train_and_evaluate()
