import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from interpret.glassbox import ExplainableBoostingClassifier
except Exception:  # Optional dependency for Hugging Face/local installs.
    ExplainableBoostingClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # Optional dependency.
    XGBClassifier = None

from config import DEFAULT_DATA_PATH, LEAKAGE_COLUMNS, MODEL_DIR, REPORT_DIR, TARGET_COLUMN


def specificity_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) else 0.0


def make_preprocessor(X, scale_numeric=False):
    categorical = X.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        from sklearn.preprocessing import StandardScaler

        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(steps=numeric_steps)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def build_models(random_state):
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=40,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    if ExplainableBoostingClassifier is not None:
        models["explainable_boosting_machine"] = ExplainableBoostingClassifier(
            random_state=random_state,
            interactions=5,
        )
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.04,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
        )
    return models


def evaluate(name, pipeline, X_test, y_test):
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
        "pr_auc": average_precision_score(y_test, probs),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "specificity": specificity_score(y_test, preds),
        "f1": f1_score(y_test, preds, zero_division=0),
        "brier_score": brier_score_loss(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds, labels=[0, 1]).tolist(),
    }


def subgroup_metrics(frame, group_col):
    rows = []
    for group, part in frame.groupby(group_col, dropna=False):
        if len(part) < 20 or part["y_true"].nunique() < 2:
            continue
        rows.append(
            {
                "grouping": group_col,
                "group": str(group),
                "n": int(len(part)),
                "prevalence": float(part["y_true"].mean()),
                "mean_predicted_risk": float(part["probability"].mean()),
                "accuracy": accuracy_score(part["y_true"], part["prediction"]),
                "roc_auc": roc_auc_score(part["y_true"], part["probability"]),
                "precision": precision_score(part["y_true"], part["prediction"], zero_division=0),
                "recall": recall_score(part["y_true"], part["prediction"], zero_division=0),
                "specificity": specificity_score(part["y_true"], part["prediction"]),
                "f1": f1_score(part["y_true"], part["prediction"], zero_division=0),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Train MLTC prediction models.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to NHANES CSV.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    y = df[TARGET_COLUMN].astype(str).str.lower().map({"yes": 1, "no": 0})
    keep = y.notna()
    df = df.loc[keep].copy()
    y = y.loc[keep].astype(int)

    feature_columns = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    X = df[feature_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=args.seed,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    results = []
    fitted = {}
    for name, estimator in build_models(args.seed).items():
        preprocessor = make_preprocessor(X_train, scale_numeric=name == "logistic_regression")
        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results.append(evaluate(name, pipe, X_test, y_test))
        joblib.dump(pipe, MODEL_DIR / f"{name}.joblib")

    metrics = sorted(results, key=lambda r: (r["roc_auc"], r["pr_auc"]), reverse=True)
    best_name = metrics[0]["model"]
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")

    best_pipeline = fitted[best_name]
    test_predictions = X_test.copy()
    test_predictions["y_true"] = y_test.to_numpy()
    test_predictions["probability"] = best_pipeline.predict_proba(X_test)[:, 1]
    test_predictions["prediction"] = (test_predictions["probability"] >= 0.5).astype(int)
    test_predictions.to_csv(REPORT_DIR / "test_predictions.csv", index=False)

    subgroup_rows = []
    for col in ["age_category", "gender", "ethnicity", "BMI_category"]:
        if col in test_predictions.columns:
            subgroup_rows.extend(subgroup_metrics(test_predictions, col))
    pd.DataFrame(subgroup_rows).to_csv(REPORT_DIR / "subgroup_metrics.csv", index=False)

    feature_summary = {
        "target": TARGET_COLUMN,
        "positive_label": "yes",
        "excluded_columns": sorted(LEAKAGE_COLUMNS),
        "feature_columns": feature_columns,
        "best_model": best_name,
        "row_count": int(len(df)),
        "class_balance": y.value_counts().sort_index().to_dict(),
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "feature_summary.json").write_text(
        json.dumps(feature_summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"best_model": best_name, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
