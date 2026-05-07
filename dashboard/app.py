import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SUMMARY_PATH = ROOT / "reports" / "feature_summary.json"
SAMPLE_PATH = ROOT / "data" / "dashboard_sample.csv"
DATA_PATH = ROOT / "data" / "NHANES_final_completed_C.csv"
SUBGROUP_PATH = ROOT / "reports" / "subgroup_metrics.csv"

AGE_LABEL_MAP = {
    "Young Adult": "Young adult (18-39)",
    "Middle Aged": "Midlife (40-59)",
    "Senior": "Older adult (60-74)",
    "Elderly": "Advanced age (75+)",
}
AGE_LABEL_ORDER = [
    "Young adult (18-39)",
    "Midlife (40-59)",
    "Older adult (60-74)",
    "Advanced age (75+)",
]
ANTI_INFLAMMATORY_NUTRIENTS = [
    "magnesium",
    "dietary_fiber",
    "vit_A_RAE",
    "vit_C",
    "vit_D",
    "vit_E",
    "riboflavin_vit_B2",
    "vit_B6",
    "folic_acid",
    "niacin",
    "zinc",
    "total_poly_unsat_FA",
    "beta_carotene",
]
BIOMARKER_COLUMNS = [
    "DII",
    "BMI",
    "alcohol",
    "beta_carotene",
    "carbohydrate",
    "cholesterol",
    "total_fat",
    "dietary_fiber",
    "energy",
    "folic_acid",
    "iron",
    "magnesium",
    "niacin",
    "total_mono_unsat_FA",
    "protein",
    "selenium",
    "thiamin_vit_B1",
    "vit_A_RAE",
    "vit_C",
    "vit_D",
    "zinc",
    "riboflavin_vit_B2",
    "vit_B6",
    "vit_B12",
    "vit_E",
    "total_poly_unsat_FA",
    "total_sat_FA",
    "caffeine",
    "n3_fat",
]
MODEL_LABELS = {
    "explainable_boosting_machine": "Explainable Boosting Machine",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "decision_tree": "Decision Tree",
    "logistic_regression": "Logistic Regression",
}
GENDER_COLORS = {
    "female": "#d95f8d",
    "male": "#1f77b4",
    "Female": "#d95f8d",
    "Male": "#1f77b4",
}

st.set_page_config(page_title="NHANES MLTC Risk Dashboard", layout="wide")


@st.cache_data
def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data
def load_sample():
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    if SAMPLE_PATH.exists():
        return pd.read_csv(SAMPLE_PATH)
    return pd.DataFrame()


@st.cache_data
def load_subgroup_metrics():
    if SUBGROUP_PATH.exists():
        return pd.read_csv(SUBGROUP_PATH)
    return pd.DataFrame()


@st.cache_resource
def load_model(model_name):
    path = MODEL_DIR / f"{model_name}.joblib"
    if path.exists():
        return joblib.load(path)
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


def metric_table(metrics):
    df = pd.DataFrame(metrics)
    cols = ["model", "roc_auc", "pr_auc", "accuracy", "recall", "specificity", "f1", "brier_score"]
    return df[[c for c in cols if c in df.columns]]


def prepare_display_data(df):
    if df.empty:
        return df
    out = df.copy()
    if "multimorbidity" in out.columns:
        out["mltc"] = out["multimorbidity"].astype(str).str.lower().map({"yes": 1, "no": 0})
    if "age_category" in out.columns:
        out["age_group"] = out["age_category"].map(AGE_LABEL_MAP).fillna(out["age_category"])
    return out


def ordered_group_frame(series, value_name):
    frame = series.reset_index()
    frame.columns = ["group", value_name]
    if set(frame["group"]).issubset(set(AGE_LABEL_ORDER)):
        frame["group"] = pd.Categorical(frame["group"], AGE_LABEL_ORDER, ordered=True)
        frame = frame.sort_values("group")
    return frame


def grouped_rate_frame(df, group, hue=None):
    if hue and hue in df.columns and hue != group:
        frame = df.groupby([group, hue], dropna=False)["mltc"].mean().reset_index()
        frame.columns = ["group", hue, "mltc_rate"]
    else:
        frame = df.groupby(group, dropna=False)["mltc"].mean().reset_index()
        frame.columns = ["group", "mltc_rate"]
    if set(frame["group"]).issubset(set(AGE_LABEL_ORDER)):
        frame["group"] = pd.Categorical(frame["group"], AGE_LABEL_ORDER, ordered=True)
        frame = frame.sort_values("group")
    return frame


def biomarker_summary(df, columns):
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        for gender, part in df.groupby("gender", dropna=False):
            values = pd.to_numeric(part[col], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "marker": col,
                    "gender": str(gender),
                    "n": int(values.size),
                    "min": float(values.min()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def add_intersectionality_columns(df):
    out = df.copy()
    if {"age_group", "gender"}.issubset(out.columns):
        out["age_gender"] = out["age_group"].astype(str) + " | " + out["gender"].astype(str)
    if {"ethnicity", "gender"}.issubset(out.columns):
        out["ethnicity_gender"] = out["ethnicity"].astype(str) + " | " + out["gender"].astype(str)
    if {"age_group", "ethnicity", "gender"}.issubset(out.columns):
        out["age_ethnicity_gender"] = (
            out["age_group"].astype(str)
            + " | "
            + out["ethnicity"].astype(str)
            + " | "
            + out["gender"].astype(str)
        )

    risk_age = out.get("age_category", pd.Series(index=out.index, dtype=object)).isin(["Senior", "Elderly"])
    risk_ethnicity = out.get("ethnicity", pd.Series(index=out.index, dtype=object)).isin(
        ["Non-Hispanic Black", "Non-Hispanic White"]
    )
    risk_bmi = out.get("BMI_category", pd.Series(index=out.index, dtype=object)).eq("Overweight")
    out["risk_context_count"] = risk_age.astype(int) + risk_ethnicity.astype(int) + risk_bmi.astype(int)
    out["risk_context_label"] = out["risk_context_count"].map(
        {
            0: "0 observed high-risk contexts",
            1: "1 observed high-risk context",
            2: "2 observed high-risk contexts",
            3: "3 observed high-risk contexts",
        }
    )
    return out


def prettify_features(features):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from explain import clean_feature_label

    return features.astype(str).map(clean_feature_label)


def available_models(metrics, summary):
    names = []
    if metrics:
        names = [m["model"] for m in metrics if (MODEL_DIR / f"{m['model']}.joblib").exists()]
    if not names and summary:
        names = [summary.get("best_model", "explainable_boosting_machine")]
    return [name for name in names if name]


st.title("NHANES MLTC Risk Dashboard")
st.caption("Glassbox MLTC prediction with Random Forest and XGBoost comparison models.")

metrics = load_json(METRICS_PATH)
summary = load_json(SUMMARY_PATH)
sample = load_sample()
display_data = prepare_display_data(sample)
display_data = add_intersectionality_columns(display_data)
subgroup_metrics = load_subgroup_metrics()
model_options = available_models(metrics, summary)
default_model = summary.get("best_model") if summary else None
default_index = model_options.index(default_model) if default_model in model_options else 0
selected_model_name = st.sidebar.selectbox(
    "Model",
    options=model_options,
    index=default_index,
    format_func=lambda name: MODEL_LABELS.get(name, name.replace("_", " ").title()),
    help="Choose which trained model powers the prediction and feature insight views.",
)
model = load_model(selected_model_name) if selected_model_name else None

if metrics is None or summary is None or model is None:
    st.warning(
        "Model artifacts are not present yet. Run `python src/train.py` first, then restart this app."
    )

tabs = st.tabs(
    [
        "Overview",
        "DII & Nutrition",
        "Fairness",
        "Intersectionality",
        "Model Comparison",
        "Risk Explorer",
        "Feature Insights",
        "Deployment",
    ]
)

with tabs[0]:
    st.subheader("Dataset Overview")
    if summary:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{summary['row_count']:,}")
        c2.metric("Predictors", f"{len(summary['feature_columns'])}")
        c3.metric("Best model", summary["best_model"].replace("_", " ").title())
        st.info(
            f"Selected display model: {MODEL_LABELS.get(selected_model_name, selected_model_name)}"
        )
        balance = pd.Series(summary["class_balance"]).rename(index={0: "No MLTC", 1: "MLTC"})
        st.plotly_chart(px.pie(values=balance.values, names=balance.index, title="MLTC class balance"))
    if not display_data.empty:
        st.dataframe(display_data.head(50), use_container_width=True)

with tabs[1]:
    st.subheader("DII & Nutrition Insights")
    if not display_data.empty and "DII" in display_data.columns:
        c1, c2, c3 = st.columns(3)
        c1.metric("Mean DII", f"{display_data['DII'].mean():.2f}")
        c2.metric("Median DII", f"{display_data['DII'].median():.2f}")
        c3.metric("DII range", f"{display_data['DII'].min():.1f} to {display_data['DII'].max():.1f}")

        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.histogram(
                    display_data,
                    x="DII",
                    color="gender" if "gender" in display_data.columns else None,
                    color_discrete_map=GENDER_COLORS,
                    nbins=40,
                    marginal="box",
                    barmode="overlay",
                    opacity=0.65,
                    title="Dietary Inflammatory Index Distribution",
                ),
                use_container_width=True,
            )
        with right:
            if "multimorbidity" in display_data.columns:
                st.plotly_chart(
                    px.box(
                        display_data,
                        x="multimorbidity",
                        y="DII",
                        color="gender" if "gender" in display_data.columns else None,
                        color_discrete_map=GENDER_COLORS,
                        points="outliers",
                        title="DII by MLTC Status",
                        labels={"multimorbidity": "MLTC", "DII": "DII score"},
                    ),
                    use_container_width=True,
                )

        group = st.selectbox(
            "Compare DII by group",
            [c for c in ["age_group", "gender", "ethnicity", "BMI_category"] if c in display_data.columns],
        )
        st.plotly_chart(
            px.box(
                display_data,
                x=group,
                y="DII",
                color="gender" if "gender" in display_data.columns and group != "gender" else group,
                color_discrete_map=GENDER_COLORS,
                title=f"DII by {group.replace('_', ' ').title()}",
            ),
            use_container_width=True,
        )

        nutrient_cols = [c for c in ANTI_INFLAMMATORY_NUTRIENTS if c in display_data.columns]
        if nutrient_cols:
            corr = (
                display_data[["DII", *nutrient_cols]]
                .corr(numeric_only=True)["DII"]
                .drop("DII")
                .sort_values()
                .reset_index()
            )
            corr.columns = ["nutrient", "correlation_with_dii"]
            st.plotly_chart(
                px.bar(
                    corr,
                    x="correlation_with_dii",
                    y="nutrient",
                    orientation="h",
                    title="Correlation of Anti-Inflammatory Nutrients With DII",
                ),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Male/Female Marker Ranges")
        marker_options = [c for c in BIOMARKER_COLUMNS if c in display_data.columns]
        selected_markers = st.multiselect(
            "Markers to display",
            marker_options,
            default=marker_options[:8],
        )
        if selected_markers and "gender" in display_data.columns:
            marker_ranges = biomarker_summary(display_data, selected_markers)
            st.dataframe(marker_ranges.style.format(precision=2), use_container_width=True)
            st.plotly_chart(
                px.box(
                    display_data,
                    x="gender",
                    y=selected_markers,
                    color="gender",
                    color_discrete_map=GENDER_COLORS,
                    points=False,
                    title="Male/Female Distribution Ranges for Selected Markers",
                ),
                use_container_width=True,
            )
    else:
        st.info("DII is not available in the loaded data.")

with tabs[2]:
    st.subheader("Fairness & Subgroups")
    if not display_data.empty and "mltc" in display_data.columns:
        group = st.selectbox(
            "Subgroup view",
            [c for c in ["age_group", "gender", "ethnicity", "BMI_category"] if c in display_data.columns],
            key="fairness_group",
        )

        rep = ordered_group_frame(display_data[group].value_counts(normalize=True), "participant_share")
        out = grouped_rate_frame(
            display_data,
            group,
            hue="gender" if "gender" in display_data.columns and group != "gender" else None,
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                px.bar(rep, x="group", y="participant_share", title="Representation in Dataset"),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                px.bar(
                    out,
                    x="group",
                    y="mltc_rate",
                    color="gender" if "gender" in out.columns else None,
                    barmode="group",
                    color_discrete_map=GENDER_COLORS,
                    title="Observed MLTC Prevalence",
                ),
                use_container_width=True,
            )

        if not subgroup_metrics.empty:
            metric_group = "age_category" if group == "age_group" else group
            perf = subgroup_metrics[subgroup_metrics["grouping"] == metric_group].copy()
            if metric_group == "age_category":
                perf["group"] = perf["group"].map(AGE_LABEL_MAP).fillna(perf["group"])
                perf["group"] = pd.Categorical(perf["group"], AGE_LABEL_ORDER, ordered=True)
                perf = perf.sort_values("group")
            st.dataframe(perf.style.format(precision=3), use_container_width=True)
            st.plotly_chart(
                px.bar(
                    perf,
                    x="group",
                    y=["recall", "specificity", "accuracy"],
                    barmode="group",
                    title="Best Model Performance by Subgroup",
                ),
                use_container_width=True,
            )
        else:
            st.info("Run `python src/train.py` again to generate subgroup model metrics.")
    else:
        st.info("Fairness views need the full dataset and MLTC target column.")

with tabs[3]:
    st.subheader("Intersectionality & Multiple Contexts")
    if not display_data.empty and "mltc" in display_data.columns:
        intersection_options = [
            c
            for c in ["age_gender", "ethnicity_gender", "age_ethnicity_gender"]
            if c in display_data.columns
        ]
        intersection_group = st.selectbox("Combined subgroup", intersection_options)
        min_n = st.slider("Minimum subgroup size", min_value=20, max_value=200, value=50, step=10)
        inter = (
            display_data.groupby(intersection_group, dropna=False)
            .agg(n=("mltc", "size"), mltc_rate=("mltc", "mean"), mean_dii=("DII", "mean"))
            .reset_index()
            .rename(columns={intersection_group: "subgroup"})
        )
        inter = inter[inter["n"] >= min_n].sort_values("mltc_rate", ascending=False)
        st.dataframe(
            inter.style.format({"mltc_rate": "{:.3f}", "mean_dii": "{:.2f}"}),
            use_container_width=True,
        )
        st.plotly_chart(
            px.bar(
                inter.head(20).sort_values("mltc_rate"),
                x="mltc_rate",
                y="subgroup",
                color="mean_dii",
                orientation="h",
                title="Highest Observed MLTC Rates Across Combined Subgroups",
                labels={"mltc_rate": "Observed MLTC prevalence", "mean_dii": "Mean DII"},
            ),
            use_container_width=True,
        )

        context = (
            display_data.groupby(["risk_context_label", "gender"], dropna=False)
            .agg(n=("mltc", "size"), mltc_rate=("mltc", "mean"), mean_dii=("DII", "mean"))
            .reset_index()
        )
        st.plotly_chart(
            px.bar(
                context,
                x="risk_context_label",
                y="mltc_rate",
                color="gender",
                barmode="group",
                color_discrete_map=GENDER_COLORS,
                title="MLTC Prevalence by Count of Observed High-Risk Contexts",
                labels={
                    "risk_context_label": "Observed high-risk context count",
                    "mltc_rate": "Observed MLTC prevalence",
                },
            ),
            use_container_width=True,
        )
        st.markdown(
            "This view is descriptive, not a causal measure of disadvantage. The context count combines "
            "older age group, higher-prevalence ethnicity group in this dataset, and overweight BMI category "
            "to show where observed MLTC burden concentrates."
        )
    else:
        st.info("Intersectionality views need the full dataset and MLTC target column.")

with tabs[4]:
    st.subheader("Model Comparison")
    if metrics:
        table = metric_table(metrics)
        selected_metrics = table[table["model"] == selected_model_name]
        if not selected_metrics.empty:
            c1, c2, c3 = st.columns(3)
            c1.metric("Selected ROC AUC", f"{selected_metrics.iloc[0]['roc_auc']:.3f}")
            c2.metric("Selected PR AUC", f"{selected_metrics.iloc[0]['pr_auc']:.3f}")
            c3.metric("Selected F1", f"{selected_metrics.iloc[0]['f1']:.3f}")
        st.dataframe(table.style.format(precision=3), use_container_width=True)
        st.plotly_chart(
            px.bar(
                table,
                x="model",
                y=["roc_auc", "pr_auc", "recall", "specificity", "f1"],
                barmode="group",
                title="Held-out Test Metrics",
            ),
            use_container_width=True,
        )

with tabs[5]:
    st.subheader("Risk Explorer")
    if model is not None and summary and not sample.empty:
        st.caption(
            f"Prediction model: {MODEL_LABELS.get(selected_model_name, selected_model_name)}"
        )
        features = summary["feature_columns"]
        base = sample[features].iloc[[0]].copy()
        with st.form("predict_form"):
            edited = {}
            for col in features:
                value = base.iloc[0][col]
                if pd.api.types.is_numeric_dtype(sample[col]):
                    edited[col] = st.number_input(col, value=float(value))
                else:
                    options = sorted(sample[col].dropna().astype(str).unique().tolist())
                    current = str(value)
                    index = options.index(current) if current in options else 0
                    edited[col] = st.selectbox(col, options=options, index=index)
            submitted = st.form_submit_button("Predict MLTC risk")
        if submitted:
            row = pd.DataFrame([edited])[features]
            probability = model.predict_proba(row)[0, 1]
            st.metric("Predicted MLTC probability", f"{probability:.1%}")
            st.progress(float(probability))
    else:
        st.info("Train the model first to enable individual predictions.")

with tabs[6]:
    st.subheader("Feature Insights")
    if model is not None:
        st.caption(
            f"Feature insight model: {MODEL_LABELS.get(selected_model_name, selected_model_name)}"
        )
        import sys

        sys.path.insert(0, str(ROOT / "src"))
        from explain import (
            clean_feature_label,
            ebm_shape_data,
            ebm_terms,
            global_importance,
            shap_global_importance,
            shap_local_importance,
        )

        importance = global_importance(model)
        if importance.empty:
            st.info("This model does not expose simple global feature importances.")
        else:
            importance["feature_display"] = importance["feature"].astype(str).map(clean_feature_label)
            st.plotly_chart(
                px.bar(
                    importance.sort_values("importance"),
                    x="importance",
                    y="feature_display",
                    orientation="h",
                    title="Top Global Predictors",
                ),
                use_container_width=True,
            )
            dii_rows = importance[importance["feature"].str.contains("DII", case=False, na=False)]
            if not dii_rows.empty:
                st.metric("DII importance rank", f"#{int(dii_rows.index[0]) + 1}")

        if selected_model_name == "explainable_boosting_machine":
            st.divider()
            st.subheader("EBM Shape Interpretation")
            terms = ebm_terms(model)
            if not terms.empty:
                selected_term = st.selectbox(
                    "Feature or interaction",
                    terms["term_index"].tolist(),
                    format_func=lambda idx: terms.loc[terms["term_index"] == idx, "feature"].iloc[0],
                )
                shape, title = ebm_shape_data(model, selected_term)
                if shape.empty:
                    st.info(
                        "This selected term is an interaction. Interaction heatmaps can be added next; "
                        "choose a single feature to see a risk-contribution curve."
                    )
                else:
                    st.plotly_chart(
                        px.line(
                            shape,
                            x="value",
                            y="score",
                            title=f"EBM Risk Contribution: {title}",
                            labels={
                                "value": title,
                                "score": "Additive contribution to MLTC log-odds",
                            },
                        ),
                        use_container_width=True,
                    )
                    st.markdown(
                        "Positive values push the prediction toward higher MLTC risk; negative values "
                        "push it toward lower MLTC risk. Flat regions indicate little change in model "
                        "risk contribution across that feature range."
                    )

        if selected_model_name == "random_forest" and summary and not sample.empty:
            st.divider()
            st.subheader("Random Forest SHAP")
            features = summary["feature_columns"]
            X = sample[features].copy()
            shap_global = shap_global_importance(model, X)
            st.plotly_chart(
                px.bar(
                    shap_global.sort_values("mean_abs_shap"),
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="SHAP Global Feature Importance",
                    labels={"mean_abs_shap": "Mean absolute SHAP value"},
                ),
                use_container_width=True,
            )

            row_number = st.number_input(
                "Participant row for local SHAP explanation",
                min_value=0,
                max_value=int(len(X) - 1),
                value=0,
                step=1,
            )
            row = X.iloc[[int(row_number)]]
            local, expected = shap_local_importance(model, row)
            probability = model.predict_proba(row)[0, 1]
            c1, c2 = st.columns(2)
            c1.metric("Baseline RF log-odds", f"{expected:.3f}")
            c2.metric("Selected participant RF risk", f"{probability:.1%}")
            local["direction"] = local["shap_value"].map(lambda v: "Increases risk" if v > 0 else "Decreases risk")
            st.plotly_chart(
                px.bar(
                    local.sort_values("shap_value"),
                    x="shap_value",
                    y="feature",
                    color="direction",
                    orientation="h",
                    title="Local SHAP Contributions",
                    labels={"shap_value": "SHAP contribution"},
                ),
                use_container_width=True,
            )

with tabs[7]:
    st.subheader("Hugging Face Spaces Deployment")
    st.code(
        "python src/train.py --data data/NHANES_final_completed_C.csv\n"
        "streamlit run dashboard/app.py",
        language="bash",
    )
    st.markdown(
        "Deploy as a Streamlit Space by uploading this project, keeping `app.py` at the repository root "
        "or setting the Space app file to `dashboard/app.py`."
    )
