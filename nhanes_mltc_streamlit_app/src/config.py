from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "NHANES_final_completed_C.csv"
SAMPLE_DATA_PATH = PROJECT_ROOT / "data" / "dashboard_sample.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COLUMN = "multimorbidity"
CHRONIC_CONDITION_COLUMNS = [
    "diabetes",
    "asthma",
    "arthritis",
    "COPD",
    "cancer",
    "stroke",
    "coronary_heart_disease",
    "liver_condition",
    "failing_kidneys",
    "hypertension",
    "obesity",
]

LEAKAGE_COLUMNS = {
    "SEQN",
    "multimorbidity",
    "chronic_conditions_count",
    "BMI_category",
    *CHRONIC_CONDITION_COLUMNS,
}

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
