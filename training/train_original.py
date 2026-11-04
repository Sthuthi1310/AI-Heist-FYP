"""Train and evaluate the original Logistic Regression employee attrition model.
This model acts as the target victim deployed behind the API.
"""
import json
import logging
from typing import Dict, Any
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

import config
from utils.preprocessing import (
    prepare_dataset_splits,
    build_column_preprocessor,
    clean_and_prepare
)
from utils.evaluation import evaluate_classification, save_metrics_to_disk, load_metrics_from_disk

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainOriginal")

def train_original_model() -> Dict[str, Any]:
    """Train the original victim Logistic Regression pipeline and evaluate against held-out test data."""
    logger.info("Preparing dataset splits (Victim Train, Held-out Test, Attacker Query Pool)...")
    splits = prepare_dataset_splits(save_splits=True)
    
    X_train = splits["X_victim_train"]
    y_train = splits["y_victim_train"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    
    logger.info(f"Victim training samples: {len(X_train)}, Held-out test samples: {len(X_test)}")
    
    # Construct complete pipeline
    preprocessor = build_column_preprocessor()
    classifier = LogisticRegression(
        max_iter=1000,
        random_state=config.RANDOM_STATE,
        solver="lbfgs",
        class_weight="balanced"
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    
    logger.info("Fitting Logistic Regression pipeline...")
    pipeline.fit(X_train, y_train)
    
    # Evaluate against held-out test set
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    metrics = evaluate_classification(y_test, y_pred, y_prob)
    logger.info(f"Original Model Test Accuracy: {metrics['accuracy'] * 100:.2f}% | F1: {metrics['f1_score']:.4f}")
    
    # Save the pipeline
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.ORIGINAL_MODEL_PATH)
    logger.info(f"Saved original model pipeline to {config.ORIGINAL_MODEL_PATH}")
    
    # Save metadata
    metadata = {
        "model_type": "LogisticRegression",
        "library": "scikit-learn",
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "features": config.ALL_FEATURE_COLUMNS,
        "metrics": metrics
    }
    with open(config.ORIGINAL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
        
    # Update central metrics file
    existing_metrics = load_metrics_from_disk() or {}
    existing_metrics["original_model"] = {
        "name": "Original Logistic Regression (Target Victim)",
        "test_accuracy": metrics["accuracy"],
        "test_f1": metrics["f1_score"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "roc_auc": metrics["roc_auc"],
        "confusion_matrix": metrics["confusion_matrix"]
    }
    save_metrics_to_disk(existing_metrics)
    
    return {
        "status": "success",
        "model_path": str(config.ORIGINAL_MODEL_PATH),
        "metrics": metrics
    }

if __name__ == "__main__":
    train_original_model()
