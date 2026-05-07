import numpy as np
import pandas as pd
import os
from pathlib import Path


os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".tmp" / "matplotlib"),
)


def clean_feature_label(name):
    label = str(name)
    label = label.replace("numeric__", "").replace("categorical__", "")
    label = label.replace("total_mono_unsat_FA", "Monounsaturated fat")
    label = label.replace("total_poly_unsat_FA", "Polyunsaturated fat")
    label = label.replace("total_sat_FA", "Saturated fat")
    label = label.replace("thiamin_vit_B1", "Thiamin (B1)")
    label = label.replace("riboflavin_vit_B2", "Riboflavin (B2)")
    label = label.replace("vit_A_RAE", "Vitamin A (RAE)")
    label = label.replace("vit_C", "Vitamin C")
    label = label.replace("vit_D", "Vitamin D")
    label = label.replace("vit_E", "Vitamin E")
    label = label.replace("vit_B6", "Vitamin B6")
    label = label.replace("vit_B12", "Vitamin B12")
    label = label.replace("n3_fat", "Omega-3 fat")
    label = label.replace("DII", "Dietary Inflammatory Index (DII)")
    label = label.replace("BMI", "BMI")
    label = label.replace("age_category_", "Age: ")
    label = label.replace("gender_", "Gender: ")
    label = label.replace("ethnicity_", "Ethnicity: ")
    return label.replace("_", " ").strip().title().replace("Bmi", "BMI").replace("Dii", "DII")


def get_feature_names(pipeline):
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def map_ebm_term_name(term_name, feature_names):
    label = str(term_name)
    for i, feature in enumerate(feature_names):
        label = label.replace(f"feature_{i:04d}", clean_feature_label(feature))
    return label


def global_importance(pipeline, top_n=20):
    model = pipeline.named_steps["model"]
    names = get_feature_names(pipeline)

    if hasattr(model, "term_importances"):
        values = model.term_importances()
        term_names = getattr(model, "term_names_", [f"term_{i}" for i in range(len(values))])
        names = [map_ebm_term_name(name, get_feature_names(pipeline)) for name in term_names]
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    if not names:
        names = [f"feature_{i}" for i in range(len(values))]

    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )


def ebm_terms(pipeline):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "term_importances"):
        return pd.DataFrame(columns=["term_index", "feature", "importance"])
    names = get_feature_names(pipeline)
    return (
        pd.DataFrame(
            {
                "term_index": range(len(model.term_names_)),
                "feature": [map_ebm_term_name(name, names) for name in model.term_names_],
                "importance": model.term_importances(),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def ebm_shape_data(pipeline, term_index):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "explain_global"):
        return pd.DataFrame(), ""
    names = get_feature_names(pipeline)
    explanation = model.explain_global()
    data = explanation.data(int(term_index))
    title = map_ebm_term_name(model.term_names_[int(term_index)], names)
    if data.get("type") != "univariate":
        return pd.DataFrame(), title
    values = list(data["names"])
    scores = list(data["scores"])
    lower = list(data.get("lower_bounds", np.repeat(np.nan, len(scores))))
    upper = list(data.get("upper_bounds", np.repeat(np.nan, len(scores))))
    n = min(len(values), len(scores), len(lower), len(upper))
    return (
        pd.DataFrame(
            {
                "value": values[:n],
                "score": scores[:n],
                "lower": lower[:n],
                "upper": upper[:n],
            }
        ),
        title,
    )


def shap_values_for_tree_pipeline(pipeline, X, max_rows=400):
    import shap

    sample = X.sample(min(max_rows, len(X)), random_state=42)
    transformed = pipeline.named_steps["preprocess"].transform(sample)
    names = [clean_feature_label(n) for n in get_feature_names(pipeline)]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[1]
    if getattr(values, "ndim", 0) == 3:
        values = values[:, :, 1]
    return sample, transformed, np.asarray(values), names


def shap_global_importance(pipeline, X, max_rows=400, top_n=20):
    _, _, values, names = shap_values_for_tree_pipeline(pipeline, X, max_rows=max_rows)
    return (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )


def shap_local_importance(pipeline, row):
    import shap

    transformed = pipeline.named_steps["preprocess"].transform(row)
    names = [clean_feature_label(n) for n in get_feature_names(pipeline)]
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(transformed)
    expected = explainer.expected_value
    if isinstance(values, list):
        values = values[1]
        expected = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
    if getattr(values, "ndim", 0) == 3:
        values = values[:, :, 1]
        expected = expected[1] if isinstance(expected, (list, np.ndarray)) else expected
    frame = (
        pd.DataFrame({"feature": names, "shap_value": np.asarray(values)[0]})
        .assign(abs_shap=lambda d: d["shap_value"].abs())
        .sort_values("abs_shap", ascending=False)
        .head(15)
        .drop(columns=["abs_shap"])
    )
    return frame, float(np.asarray(expected).ravel()[0])


def local_linear_contributions(pipeline, row):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["feature", "contribution"])

    transformed = pipeline.named_steps["preprocess"].transform(row)
    names = get_feature_names(pipeline)
    contributions = transformed[0] * model.coef_[0]
    return (
        pd.DataFrame({"feature": names, "contribution": contributions})
        .assign(abs_contribution=lambda d: d["contribution"].abs())
        .sort_values("abs_contribution", ascending=False)
        .drop(columns=["abs_contribution"])
        .head(15)
    )
