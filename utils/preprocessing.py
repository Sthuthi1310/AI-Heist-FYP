"""Data preprocessing and feature transformation utilities for AI Heist.
"""
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

import config

def load_raw_data(filepath: Optional[str] = None) -> pd.DataFrame:
    """Load raw employee attrition dataset from configured or specified path."""
    path = filepath or config.DATASET_FILE
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    # Strip any column whitespace
    df.columns = [col.strip() for col in df.columns]
    return df

def clean_and_prepare(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Clean data, remove identifier columns, and separate target if present."""
    df_clean = df.copy()
    
    # Drop identifier if present
    if config.ID_COLUMN in df_clean.columns:
        df_clean = df_clean.drop(columns=[config.ID_COLUMN])
        
    y = None
    if config.TARGET_COLUMN in df_clean.columns:
        target_series = df_clean[config.TARGET_COLUMN]
        if not pd.api.types.is_numeric_dtype(target_series):
            y = target_series.astype(str).str.strip().map({config.NEGATIVE_CLASS: 0, config.POSITIVE_CLASS: 1}).fillna(0).astype(int)
        else:
            y = target_series.astype(int)
        X = df_clean.drop(columns=[config.TARGET_COLUMN])
    else:
        X = df_clean
        
    # Ensure all required features are present
    missing_features = [f for f in config.ALL_FEATURE_COLUMNS if f not in X.columns]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")
        
    # Reorder columns to standard feature order
    X = X[config.ALL_FEATURE_COLUMNS]
    return X, y

def build_column_preprocessor() -> ColumnTransformer:
    """Construct Scikit-learn ColumnTransformer for standardizing numeric features
    and one-hot encoding categorical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                config.NUMERICAL_FEATURES
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                config.CATEGORICAL_FEATURES
            )
        ],
        remainder="drop"
    )
    return preprocessor

def prepare_dataset_splits(
    save_splits: bool = True
) -> Dict[str, Any]:
    """Split raw data into:
    1. Victim train set (for training the original model)
    2. Held-out test set (for fair ground-truth evaluation of original and substitute models)
    3. Attacker query pool (unlabeled feature pool for black-box API queries)
    """
    raw_df = load_raw_data()
    X, y = clean_and_prepare(raw_df)
    
    # First split: Hold out the definitive test set (e.g. 20%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    # Second split: From the remaining 80%, split into Victim Train (50% of total) and Attacker Pool (30% of total)
    # 0.30 / 0.80 = 0.375
    attacker_ratio = config.ATTACK_POOL_SIZE / (1.0 - config.TEST_SIZE)
    X_victim_train, X_attack_pool, y_victim_train, y_attack_pool = train_test_split(
        X_train_val, y_train_val,
        test_size=attacker_ratio,
        random_state=config.RANDOM_STATE,
        stratify=y_train_val
    )
    
    if save_splits:
        config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # Save held out test data with ground truth
        test_df = X_test.copy()
        test_df[config.TARGET_COLUMN] = y_test
        test_df.to_csv(config.HELD_OUT_TEST_DATA_PATH, index=False)
        
        # Save attack query pool WITHOUT target column (attacker is strictly black-box)
        X_attack_pool.to_csv(config.ATTACK_QUERY_POOL_PATH, index=False)
    
    return {
        "X_victim_train": X_victim_train,
        "y_victim_train": y_victim_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_attack_pool": X_attack_pool,
        "y_attack_pool": y_attack_pool  # kept internally for verification only
    }

def get_sample_employee(risk_profile: str = "high") -> Dict[str, Any]:
    """Generate realistic employee profile sample for testing the API/UI."""
    if risk_profile == "high":
        return {
            "Age": 28,
            "Gender": "Male",
            "Marital_Status": "Single",
            "Department": "Sales",
            "Job_Role": "Assistant",
            "Job_Level": 1,
            "Monthly_Income": 3800,
            "Hourly_Rate": 45,
            "Years_at_Company": 2,
            "Years_in_Current_Role": 1,
            "Years_Since_Last_Promotion": 1,
            "Work_Life_Balance": 1,
            "Job_Satisfaction": 1,
            "Performance_Rating": 2,
            "Training_Hours_Last_Year": 18,
            "Overtime": "Yes",
            "Project_Count": 6,
            "Average_Hours_Worked_Per_Week": 55,
            "Absenteeism": 14,
            "Work_Environment_Satisfaction": 1,
            "Relationship_with_Manager": 1,
            "Job_Involvement": 1,
            "Distance_From_Home": 38,
            "Number_of_Companies_Worked": 4
        }
    elif risk_profile == "moderate":
        return {
            "Age": 36,
            "Gender": "Male",
            "Marital_Status": "Married",
            "Department": "IT",
            "Job_Role": "Analyst",
            "Job_Level": 3,
            "Monthly_Income": 8900,
            "Hourly_Rate": 60,
            "Years_at_Company": 7,
            "Years_in_Current_Role": 4,
            "Years_Since_Last_Promotion": 2,
            "Work_Life_Balance": 3,
            "Job_Satisfaction": 3,
            "Performance_Rating": 3,
            "Training_Hours_Last_Year": 40,
            "Overtime": "No",
            "Project_Count": 4,
            "Average_Hours_Worked_Per_Week": 44,
            "Absenteeism": 5,
            "Work_Environment_Satisfaction": 3,
            "Relationship_with_Manager": 3,
            "Job_Involvement": 3,
            "Distance_From_Home": 15,
            "Number_of_Companies_Worked": 2
        }
    else:  # low risk / loyal
        return {
            "Age": 48,
            "Gender": "Female",
            "Marital_Status": "Married",
            "Department": "Finance",
            "Job_Role": "Manager",
            "Job_Level": 4,
            "Monthly_Income": 15500,
            "Hourly_Rate": 78,
            "Years_at_Company": 16,
            "Years_in_Current_Role": 8,
            "Years_Since_Last_Promotion": 3,
            "Work_Life_Balance": 4,
            "Job_Satisfaction": 4,
            "Performance_Rating": 3,
            "Training_Hours_Last_Year": 60,
            "Overtime": "No",
            "Project_Count": 4,
            "Average_Hours_Worked_Per_Week": 40,
            "Absenteeism": 2,
            "Work_Environment_Satisfaction": 4,
            "Relationship_with_Manager": 4,
            "Job_Involvement": 4,
            "Distance_From_Home": 5,
            "Number_of_Companies_Worked": 2
        }
