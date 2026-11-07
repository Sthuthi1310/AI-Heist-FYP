"""Flask application for AI Heist: Model Stealing using API Queries.
Provides prediction services, defense enforcement, attack simulation, and comparative evaluation.
"""
import time
import json
import threading
import sys
import os
import math
from typing import Dict, Any, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, render_template, request
import pandas as pd
import numpy as np
import joblib

import config
from utils.preprocessing import (
    clean_and_prepare,
    get_sample_employee,
    prepare_dataset_splits
)
from utils.evaluation import (
    load_metrics_from_disk,
    save_metrics_to_disk,
    calculate_extraction_fidelity,
    evaluate_classification,
    generate_comparison_summary
)
from security.authentication import require_api_key, get_client_identifier
from security.rate_limiter import rate_limiter
from security.suspicious_detection import suspicious_detector
from security.perturbation import apply_output_perturbation
from security.security_logger import security_logger
from attack.state import attack_state_manager

app = Flask(
    __name__,
    template_folder=str(config.TEMPLATES_DIR),
    static_folder=str(config.STATIC_DIR)
)

# Global model cache
_original_model = None
_model_lock = threading.Lock()

CATEGORICAL_VALUES = {
    "Gender": {"Male", "Female"},
    "Marital_Status": {"Single", "Married", "Divorced"},
    "Department": {"Sales", "HR", "Finance", "Marketing", "IT"},
    "Job_Role": {"Manager", "Assistant", "Analyst", "Executive"},
    "Overtime": {"Yes", "No"},
}


def validate_prediction_payload(data: Any) -> Dict[str, Any]:
    """Validate and normalize a prediction request before model inference."""
    if not isinstance(data, dict):
        raise ValueError("Request JSON must be an object containing employee fields")

    normalized = dict(data)
    missing_features = [
        feature for feature in config.ALL_FEATURE_COLUMNS
        if feature not in normalized
    ]
    if missing_features:
        raise ValueError(f"Missing required features in data: {missing_features}")

    unexpected_features = [
        feature for feature in normalized
        if feature not in config.ALL_FEATURE_COLUMNS
    ]
    if unexpected_features:
        raise ValueError(f"Unexpected feature fields: {unexpected_features}")

    for feature in config.NUMERICAL_FEATURES:
        value = normalized.get(feature)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Feature '{feature}' must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"Feature '{feature}' must be a finite number")
        normalized[feature] = float(value)

    for feature, allowed_values in CATEGORICAL_VALUES.items():
        value = normalized.get(feature)
        if not isinstance(value, str):
            raise ValueError(f"Feature '{feature}' must be one of: {sorted(allowed_values)}")
        if value not in allowed_values:
            raise ValueError(
                f"Invalid value for '{feature}': {value}. "
                f"Expected one of: {sorted(allowed_values)}"
            )

    return normalized


def calculate_profile_risk_score(profile: Dict[str, Any]) -> float:
    """Combine common employee attrition risk indicators for interpretable predictions."""
    score = 0.08
    score += 0.22 if profile["Overtime"] == "Yes" else 0.0
    score += 0.16 if profile["Job_Satisfaction"] <= 2 else 0.0
    score += 0.12 if profile["Work_Life_Balance"] <= 2 else 0.0
    score += 0.08 if profile["Work_Environment_Satisfaction"] <= 2 else 0.0
    score += 0.08 if profile["Relationship_with_Manager"] <= 2 else 0.0
    score += 0.06 if profile["Job_Involvement"] <= 2 else 0.0
    score += 0.07 if profile["Distance_From_Home"] >= 25 else 0.0
    score += 0.08 if profile["Monthly_Income"] < 4500 else 0.0
    score += 0.05 if profile["Average_Hours_Worked_Per_Week"] >= 50 else 0.0
    score += 0.05 if profile["Absenteeism"] >= 10 else 0.0
    score += 0.05 if profile["Years_at_Company"] <= 2 else 0.0
    score += 0.04 if profile["Number_of_Companies_Worked"] >= 4 else 0.0

    if profile["Work_Life_Balance"] >= 4 and profile["Job_Satisfaction"] >= 4:
        score -= 0.12
    if profile["Overtime"] == "No" and profile["Years_at_Company"] >= 10:
        score -= 0.08
    return max(0.02, min(0.98, score))


def get_original_model():
    """Retrieve cached original model pipeline or load/train if needed."""
    global _original_model
    with _model_lock:
        if _original_model is None:
            needs_training = not config.ORIGINAL_MODEL_PATH.exists()
            model = None
            if not needs_training:
                try:
                    model = joblib.load(config.ORIGINAL_MODEL_PATH)
                    sample = pd.DataFrame([get_sample_employee("high")])
                    X_sample, _ = clean_and_prepare(sample)
                    model.predict(X_sample)
                    model.predict_proba(X_sample)
                except Exception:
                    needs_training = True
            if needs_training:
                from training.train_original import train_original_model
                train_original_model()
                model = joblib.load(config.ORIGINAL_MODEL_PATH)
            _original_model = model
        return _original_model

@app.get("/health")
def health():
    """Health check endpoint."""
    model_exists = config.ORIGINAL_MODEL_PATH.exists()
    return jsonify({
        "status": "ok",
        "service": "AI Heist API",
        "original_model_ready": model_exists,
        "security_config": {
            "rate_limiting": config.SECURITY_CONFIG.get("rate_limiting_enabled"),
            "suspicious_detection": config.SECURITY_CONFIG.get("suspicious_detection_enabled"),
            "perturbation": config.SECURITY_CONFIG.get("perturbation_enabled"),
            "auth_required": config.SECURITY_CONFIG.get("auth_required")
        }
    })

@app.get("/")
def index():
    """Serve main web application dashboard."""
    return render_template("index.html")

@app.get("/api/sample-employee")
def sample_employee():
    """Get sample employee profile for quick testing."""
    risk = request.args.get("risk", "high")
    profile = get_sample_employee(risk)
    return jsonify(profile)

@app.post("/api/predict")
@require_api_key
def predict():
    """Predict employee attrition for an employee feature profile.
    Enforces security checks (rate limiting, anomaly detection, output perturbation).
    """
    client_id, client_role = get_client_identifier()
    ip_addr = request.remote_addr or "127.0.0.1"
    start_time = time.time()
    
    # 1. Rate Limiting Check
    is_allowed, rl_info = rate_limiter.check_rate_limit(client_id)
    if not is_allowed:
        security_logger.log_event(
            event_type="rate_limit_exceeded",
            client_id=client_id,
            ip_address=ip_addr,
            endpoint="/api/predict",
            status_code=429,
            details=rl_info
        )
        return jsonify({
            "status": "error",
            "code": 429,
            "message": "Too Many Requests: Rate limit exceeded. Please throttle your queries.",
            "rate_limit_details": rl_info
        }), 429

    # 2. Parse feature payload
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"status": "error", "code": 400, "message": "No JSON payload provided"}), 400

        data = validate_prediction_payload(data)
        # Ensure single record DataFrame
        df_query = pd.DataFrame([data])
        X_clean, _ = clean_and_prepare(df_query)
    except Exception as e:
        security_logger.log_event(
            event_type="malformed_payload",
            client_id=client_id,
            ip_address=ip_addr,
            endpoint="/api/predict",
            status_code=400,
            details={"error": str(e)}
        )
        return jsonify({"status": "error", "code": 400, "message": f"Invalid feature data: {str(e)}"}), 400

    # 3. Model Inference
    try:
        model = get_original_model()
        raw_pred = int(model.predict(X_clean)[0])
        classes = list(model.classes_)
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        raw_prob = float(model.predict_proba(X_clean)[0, positive_index])
        profile_score = calculate_profile_risk_score(data)
        raw_prob = (0.35 * raw_prob) + (0.65 * profile_score)
        raw_pred = 1 if raw_prob >= 0.50 else 0
    except Exception as e:
        return jsonify({"status": "error", "code": 500, "message": f"Model inference error: {str(e)}"}), 500

    # 4. Suspicious Query Detection
    anomaly_result = suspicious_detector.analyze_query(
        client_id=client_id,
        features_dict=data,
        predicted_probability=raw_prob
    )
    is_suspicious = anomaly_result["is_suspicious"]
    suspicion_score = anomaly_result["suspicion_score"]

    # 5. Output Perturbation Defense
    final_label, final_prob, defense_applied, defense_reason = apply_output_perturbation(
        original_label=raw_pred,
        original_prob=raw_prob,
        is_suspicious=is_suspicious
    )

    latency_ms = round((time.time() - start_time) * 1000.0, 2)
    
    # 6. Structured Security Logging
    security_logger.log_event(
        event_type="prediction_served",
        client_id=client_id,
        ip_address=ip_addr,
        endpoint="/api/predict",
        status_code=200,
        details={
            "latency_ms": latency_ms,
            "suspicion_score": suspicion_score,
            "is_suspicious": is_suspicious,
            "defense_applied": defense_applied,
            "defense_reason": defense_reason,
            "original_label": raw_pred,
            "returned_label": final_label
        }
    )

    # Risk level categorization
    risk_level = "High" if final_prob >= 0.65 else ("Medium" if final_prob >= 0.40 else "Low")
    attrition_str = "Yes" if final_label == 1 else "No"

    return jsonify({
        "status": "success",
        "prediction": final_label,
        "attrition": attrition_str,
        "probability": final_prob,
        "risk_level": risk_level,
        "security": {
            "client_id": client_id,
            "client_role": client_role,
            "suspicious": is_suspicious,
            "suspicion_score": suspicion_score,
            "defense_applied": defense_applied,
            "defense_reason": defense_reason,
            "latency_ms": latency_ms
        }
    })

@app.post("/api/security/config")
def update_security_config():
    """Dynamically toggle and update security defenses from UI/API."""
    data = request.get_json(force=True) or {}
    for key, value in data.items():
        if key in config.SECURITY_CONFIG:
            config.SECURITY_CONFIG[key] = value
            
    return jsonify({
        "status": "success",
        "message": "Security configuration updated",
        "security_config": config.SECURITY_CONFIG
    })

@app.get("/api/security/stats")
def get_security_stats():
    """Retrieve live security monitoring metrics and recent audit events."""
    summary = security_logger.get_summary_stats()
    rl_stats = rate_limiter.get_stats()
    det_stats = suspicious_detector.get_stats()
    recent = security_logger.get_recent_events(limit=40)
    
    return jsonify({
        "status": "success",
        "summary": summary,
        "rate_limiter": rl_stats,
        "suspicious_detector": det_stats,
        "config": config.SECURITY_CONFIG,
        "recent_events": recent
    })

@app.post("/api/security/reset")
def reset_security_stats():
    """Reset security logs, detector history, and rate limiters."""
    rate_limiter.reset()
    suspicious_detector.reset()
    security_logger.clear()
    return jsonify({"status": "success", "message": "Security logs and detectors reset."})

@app.post("/api/attack/start")
def start_attack():
    """Simulate a black-box model extraction attack against the API."""
    data = request.get_json(force=True) or {}
    query_count = int(data.get("queries", 500))
    defended_mode = bool(data.get("defended", False))
    
    current_state = attack_state_manager.get_state()
    if current_state["is_running"]:
        return jsonify({
            "status": "error",
            "message": "An attack simulation is already actively running."
        }), 400
        
    # Run simulation in a background thread
    from attack.attacker import run_attack_simulation_task
    threading.Thread(
        target=run_attack_simulation_task,
        args=(query_count, defended_mode, request.host_url.rstrip("/")),
        daemon=True
    ).start()

    return jsonify({
        "status": "success",
        "message": f"Started {'defended' if defended_mode else 'undefended'} attack simulation with {query_count} queries.",
        "state": attack_state_manager.get_state()
    })

@app.get("/api/attack/status")
def attack_status():
    """Get current status of black-box attack simulation."""
    return jsonify(attack_state_manager.get_state())

@app.post("/api/train/substitute")
def train_substitute():
    """Train Random Forest substitute model from the collected stolen dataset."""
    data = request.get_json(force=True) or {}
    mode = data.get("mode", "undefended")  # 'undefended' or 'defended'
    dataset_path = data.get("dataset_path")
    
    from training.train_substitute import train_substitute_model
    try:
        result = train_substitute_model(mode=mode, dataset_path=dataset_path)
        return jsonify({
            "status": "success",
            "mode": mode,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Failed to train substitute model: {str(e)}"
        }), 500

@app.post("/api/train/original")
def retrain_original():
    """Retrain the original Logistic Regression victim model."""
    global _original_model
    from training.train_original import train_original_model
    try:
        result = train_original_model()
        with _model_lock:
            _original_model = joblib.load(config.ORIGINAL_MODEL_PATH)
        return jsonify({
            "status": "success",
            "result": result
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.get("/api/metrics")
def get_metrics():
    """Get comprehensive comparative metrics for Original vs Substitute models."""
    recompute = request.args.get("recompute", "false").lower() == "true"
    if recompute:
        from training.train_substitute import reevaluate_substitute_models
        try:
            reevaluate_substitute_models()
        except Exception:
            pass
            
    metrics = load_metrics_from_disk()
    if not metrics:
        return jsonify({
            "status": "empty",
            "message": "No evaluation metrics found yet. Train original model first."
        })
    return jsonify({
        "status": "success",
        "metrics": metrics
    })

@app.get("/api/experiments")
def get_experiments():
    """Return reproducible experiment run history and available export files."""
    records = []
    if config.EXPERIMENTS_JSONL_PATH.exists():
        with config.EXPERIMENTS_JSONL_PATH.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return jsonify({
        "status": "success",
        "runs": records[-100:],
        "jsonl_export": str(config.EXPERIMENTS_JSONL_PATH),
        "csv_export": str(config.EXPERIMENTS_CSV_PATH)
    })

if __name__ == "__main__":
    # Ensure the original model is loadable and compatible on startup.
    get_original_model()
    app.run(host="127.0.0.1", port=5000, debug=True)
