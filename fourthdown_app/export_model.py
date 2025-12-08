from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from app import CATEGORICAL_FEATURES, MODEL_PATH, NUMERIC_FEATURES


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _extract_bundle(model) -> Dict[str, Any]:
    pre = model.named_steps["pre"]
    clf = model.named_steps["clf"]

    feature_names: List[str] = pre.get_feature_names_out().tolist()
    coef = clf.coef_.ravel().tolist()
    intercept = float(clf.intercept_[0])

    num_imputer = pre.named_transformers_["num"].named_steps["imputer"]
    numeric_medians = dict(zip(NUMERIC_FEATURES, num_imputer.statistics_.tolist()))

    cat_encoder = pre.named_transformers_["cat"].named_steps["encoder"]
    categorical_levels = {
        feature: categories.tolist()
        for feature, categories in zip(CATEGORICAL_FEATURES, cat_encoder.categories_)
    }

    return {
        "feature_names": feature_names,
        "coef": coef,
        "intercept": intercept,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_medians": numeric_medians,
        "categorical_levels": categorical_levels,
    }


def main():
    model = _load_model()
    bundle = _extract_bundle(model)
    target = Path(__file__).parent / "web" / "model_bundle.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    print(f"Wrote bundle to {target}")


if __name__ == "__main__":
    main()
