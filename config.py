"""Central configuration for AI Heist: Model Stealing using API Queries.
Batch 39 - Final Year Project Phase-1.
"""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
WEB_DIR = PROJECT_ROOT / "web"
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# Ensure essential directories exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset configuration
# Look for employee_attrition_dataset_10000.csv first, then fallback to employee_attrition.csv
DEFAULT_RAW_FILE = DATA_RAW_DIR / "employee_attrition_dataset_10000.csv"
FALLBACK_RAW_FILE = DATA_RAW_DIR / "employee_attrition.csv"
DATASET_FILE = DEFAULT_RAW_FILE if DEFAULT_RAW_FILE.exists() else FALLBACK_RAW_FILE
DATASET_FILENAME = DATASET_FILE.name

# Feature columns
ID_COLUMN = "Employee_ID"
TARGET_COLUMN = "Attrition"
POSITIVE_CLASS = "Yes"
NEGATIVE_CLASS = "No"

CATEGORICAL_FEATURES = [
    "Gender",
    "Marital_Status",
    "Department",
    "Job_Role",
    "Overtime"
]

NUMERICAL_FEATURES = [
    "Age",
    "Job_Level",
    "Monthly_Income",
    "Hourly_Rate",
    "Years_at_Company",
    "Years_in_Current_Role",
    "Years_Since_Last_Promotion",
    "Work_Life_Balance",
    "Job_Satisfaction",
    "Performance_Rating",
    "Training_Hours_Last_Year",
    "Project_Count",
    "Average_Hours_Worked_Per_Week",
    "Absenteeism",
    "Work_Environment_Satisfaction",
    "Relationship_with_Manager",
    "Job_Involvement",
    "Distance_From_Home",
    "Number_of_Companies_Worked"
]

ALL_FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Split and training settings
RANDOM_STATE = 42
TEST_SIZE = 0.20
ATTACK_POOL_SIZE = 0.30  # Unlabeled query pool reserved for attacker simulation

# Model artifacts
ORIGINAL_MODEL_PATH = MODELS_DIR / "original_model.joblib"
ORIGINAL_METADATA_PATH = MODELS_DIR / "original_model_metadata.json"
SUBSTITUTE_MODEL_UNDEFENDED_PATH = MODELS_DIR / "substitute_model_undefended.joblib"
SUBSTITUTE_MODEL_DEFENDED_PATH = MODELS_DIR / "substitute_model_defended.joblib"
HELD_OUT_TEST_DATA_PATH = DATA_PROCESSED_DIR / "held_out_test_data.csv"
ATTACK_QUERY_POOL_PATH = DATA_PROCESSED_DIR / "attack_query_pool.csv"

# Attack simulation settings
DEFAULT_ATTACK_QUERIES = 1000
STOLEN_DATASET_UNDEFENDED_PATH = DATA_PROCESSED_DIR / "stolen_dataset_undefended.csv"
STOLEN_DATASET_DEFENDED_PATH = DATA_PROCESSED_DIR / "stolen_dataset_defended.csv"
EXPERIMENTS_JSONL_PATH = LOGS_DIR / "experiment_runs.jsonl"
EXPERIMENTS_CSV_PATH = LOGS_DIR / "experiment_runs.csv"

# Security and Defense configuration
SECURITY_CONFIG = {
    # API Key Authentication
    "auth_required": False,  # If True, requests require X-API-Key header
    "api_keys": {
        "genuine_user_key_hr_2026": {"role": "genuine", "name": "HR Department"},
        "analyst_key_ops_2026": {"role": "genuine", "name": "Operations Analyst"},
        "guest_demo_key": {"role": "guest", "name": "Web UI Guest"},
        "attacker_key_heist_2026": {"role": "attacker", "name": "Dedicated Attack Client"}
    },
    
    # Rate Limiting
    "rate_limiting_enabled": False,  # Toggleable via UI / API
    "rate_limit_per_minute": 60,     # Max allowed requests per minute per IP/client
    "burst_limit_per_second": 5,     # Max requests within a single second
    
    # Suspicious Query Detection
    "suspicious_detection_enabled": False,  # Toggleable
    "suspicion_threshold": 0.50,            # Anomaly score threshold (0.0 to 1.0)
    "min_query_interval_ms": 50,            # Interval below which requests are flagged as bot bursts
    
    # Output Perturbation Defense
    "perturbation_enabled": False,         # Toggleable
    "perturbation_mode": "adaptive",       # 'adaptive' (perturb suspicious queries) or 'always'
    "flip_probability": 0.35,              # Probability of flipping label on suspicious query
    "noise_std_dev": 0.15                  # Gaussian noise added to probabilities for soft perturbation
}

# Logging files
SECURITY_LOG_PATH = LOGS_DIR / "security_events.jsonl"
API_ACCESS_LOG_PATH = LOGS_DIR / "api_access.log"
EVALUATION_METRICS_PATH = LOGS_DIR / "evaluation_metrics.json"
