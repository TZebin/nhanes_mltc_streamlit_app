# NHANES MLTC Prediction Dashboard

This project trains interpretable glassbox models for MLTC prediction from the NHANES nutrition dataset and deploys a Streamlit dashboard suitable for Hugging Face Spaces.

Active development copy:

```text
F:\Codex\nhanes_mltc_streamlit_app
```

The original `C:` virtual environment was deleted because the `C:` drive had no free space. The active virtual environment and installed packages now live on `F:`.

## Dataset Context

Dataset:

```text
data\NHANES_final_completed_C.csv
```

Profile:

- Rows: `6,655`
- Columns: `47`
- Target balance:
  - `multimorbidity = yes`: `3,286`
  - `multimorbidity = no`: `3,369`

Key columns:

- `DII`: Dietary Inflammatory Index, a main nutrition/inflammation predictor.
- `age_category`: current categorical age grouping.
- `gender`, `ethnicity`, `BMI`, `BMI_category`: demographic/body composition predictors.
- Nutrition variables including fiber, energy, fat, protein, carbohydrate, vitamins, minerals, caffeine, and fatty acid measures.

Thesis context inspected:

```text
C:\Users\Arham\Downloads\2360921_Beverly NHANES Thesis.pdf
```

The thesis compared continuous and categorical versions of DII/BMI. Its results support using BMI as a continuous predictor because continuous BMI preserved more predictive information than BMI category. Therefore, this app now uses `BMI` for modelling and keeps `BMI_category` only for dashboard subgroup/fairness views.

## Target

The primary target is `multimorbidity`, where `yes` means MLTC-positive and `no` means MLTC-negative.

The following columns are excluded from predictors to avoid target leakage:

- `multimorbidity`
- `chronic_conditions_count`
- `BMI_category`, because it duplicates continuous `BMI`; kept for subgroup plots only.
- Individual chronic-condition flags such as `diabetes`, `asthma`, `arthritis`, `COPD`, `cancer`, `stroke`, `hypertension`, and related disease indicators.

## Models

Trained models:

- Logistic Regression: transparent baseline.
- Decision Tree: simple rule-based glassbox model.
- Explainable Boosting Machine: main glassbox model.
- Random Forest: nonlinear comparison model.
- XGBoost: high-performance comparison model.

The trained model artifacts are saved in:

```text
models\
```

## Current Training Results

Training completed successfully on the `F:` drive environment.

| Model | ROC AUC | PR AUC | Accuracy | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Explainable Boosting Machine | 0.856 | 0.850 | 0.767 | 0.821 | 0.777 |
| XGBoost | 0.855 | 0.847 | 0.755 | 0.793 | 0.762 |
| Random Forest | 0.852 | 0.842 | 0.763 | 0.821 | 0.774 |
| Decision Tree | 0.850 | 0.827 | 0.748 | 0.872 | 0.774 |
| Logistic Regression | 0.834 | 0.822 | 0.758 | 0.785 | 0.762 |

Best model by ROC AUC:

```text
explainable_boosting_machine
```

This is a useful project result: the glassbox EBM slightly outperformed XGBoost by ROC AUC, so the deployed model can prioritize interpretability without sacrificing benchmark performance.

## Prior Notebook Context

Prior exploratory notebook:

```text
C:\Users\Arham\NHANES Analysis\NHANES_TZ_Bias_Fairness_ExplainerDahboard.ipynb
```

Useful notebook ideas to integrate into the Streamlit dashboard:

- DII distribution.
- DII by MLTC status.
- Participant representation by age, ethnicity, and gender.
- MLTC prevalence by age, ethnicity, and gender.
- Correlations between DII and anti-inflammatory nutrients such as dietary fiber, magnesium, vitamin C, vitamin E, zinc, folic acid, niacin, beta-carotene, and polyunsaturated fatty acids.
- Subgroup performance and fairness checks.

Do not copy the notebook's `LabelEncoder` approach directly for modeling because it creates artificial ordinal relationships in categorical variables. The current training pipeline uses `OneHotEncoder`.

Implemented dashboard additions:

- `DII & Nutrition` tab with DII distribution, DII by MLTC, DII by subgroup, and nutrient-DII correlations.
- `Fairness` tab with representation, observed MLTC prevalence, and held-out subgroup model performance.
- Age display labels with ranges.
- Saved held-out predictions for fairness diagnostics.
- Sidebar model selector for EBM, XGBoost, Random Forest, Decision Tree, and Logistic Regression.
- EBM feature names mapped back from transformed `feature_0000` labels to readable names.
- EBM shape/risk contribution curves for univariate terms.
- Random Forest SHAP global importance and local participant explanations.

## Age Group Display Plan

Keep the original `age_category` values for reproducibility, but use clearer dashboard labels:

```python
age_label_map = {
    "Young Adult": "Young adult (18-39)",
    "Middle Aged": "Midlife (40-59)",
    "Senior": "Older adult (60-74)",
    "Elderly": "Advanced age (75+)",
}
```

If exact ages become available later, prefer deriving age bands directly from age rather than relying on inferred category labels.

## Run Locally

```bash
cd F:\Codex\nhanes_mltc_streamlit_app
.\.venv\Scripts\python.exe src\train.py --data data\NHANES_final_completed_C.csv
.\.venv\Scripts\streamlit.exe run dashboard\app.py
```

If rebuilding from scratch:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python src\train.py --data data\NHANES_final_completed_C.csv
streamlit run dashboard\app.py
```

## Hugging Face Spaces

1. Create a new Hugging Face Space.
2. Choose `Streamlit` as the SDK.
3. Upload the project files.
4. Add the dataset as `data/NHANES_final_completed_C.csv`.
5. Train locally and upload `models/*.joblib` plus `reports/*.json`, or run the training script before deployment if your Space has enough resources.
6. Set the app entrypoint to the root `app.py`.

## Outputs

- `models/best_model.joblib`
- `models/logistic_regression.joblib`
- `models/decision_tree.joblib`
- `models/random_forest.joblib`
- `models/explainable_boosting_machine.joblib`
- `models/xgboost.joblib`
- `reports/metrics.json`
- `reports/feature_summary.json`
- `reports/test_predictions.csv`
- `reports/subgroup_metrics.csv`
- `data/dashboard_sample.csv`

## Next Development Steps

- Visually inspect each dashboard tab in a browser.
- Improve EBM-specific shape plots inside Streamlit.
- Add calibration plots.
- Prepare a clean Hugging Face upload that excludes `.venv`, `.pip-cache`, and `.tmp`.

## Current Local App

The Streamlit app has been launched locally and responded successfully at:

```text
http://localhost:8501
```
