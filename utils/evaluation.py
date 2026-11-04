"""Evaluation helpers for classification performance, extraction fidelity, and defense impact.
"""
from typing import Dict, Any, Union, Optional
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import config

def evaluate_classification(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_prob: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics."""
    y_true_arr = np.asarray(y_true).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
    
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    # Ensure cm format [[TN, FP], [FN, TP]]
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(val) for val in cm.ravel()]
    else:
        tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0

    auc = None
    if y_prob is not None:
        try:
            auc = float(roc_auc_score(y_true_arr, np.asarray(y_prob)))
        except Exception:
            auc = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp
        }
    }

def calculate_extraction_fidelity(
    victim_preds: Union[np.ndarray, pd.Series],
    substitute_preds: Union[np.ndarray, pd.Series],
    victim_probs: Optional[Union[np.ndarray, pd.Series]] = None,
    substitute_probs: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, Any]:
    """Calculate extraction fidelity (agreement between victim and substitute models).
    Fidelity measures how accurately the copycat mimics the victim model's behavior.
    """
    v_preds = np.asarray(victim_preds).astype(int)
    s_preds = np.asarray(substitute_preds).astype(int)
    
    agreement = float(np.mean(v_preds == s_preds))
    
    mean_prob_diff = None
    if victim_probs is not None and substitute_probs is not None:
        v_p = np.asarray(victim_probs)
        s_p = np.asarray(substitute_probs)
        mean_prob_diff = float(np.mean(np.abs(v_p - s_p)))
        
    return {
        "fidelity_agreement": round(agreement, 4),
        "fidelity_percentage": round(agreement * 100, 2),
        "mean_prob_drift": round(mean_prob_diff, 4) if mean_prob_diff is not None else None
    }

def generate_comparison_summary(
    original_metrics: Dict[str, Any],
    substitute_undefended_metrics: Optional[Dict[str, Any]] = None,
    substitute_defended_metrics: Optional[Dict[str, Any]] = None,
    fidelity_undefended: Optional[Dict[str, Any]] = None,
    fidelity_defended: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Compile a full comparative evaluation report suitable for display and saving."""
    summary = {
        "original_model": {
            "name": "Original Logistic Regression (Target Victim)",
            "test_accuracy": original_metrics.get("accuracy"),
            "test_f1": original_metrics.get("f1_score"),
            "test_precision": original_metrics.get("precision"),
            "test_recall": original_metrics.get("recall"),
            "roc_auc": original_metrics.get("roc_auc"),
            "confusion_matrix": original_metrics.get("confusion_matrix")
        },
        "substitute_undefended": None,
        "substitute_defended": None,
        "defense_impact": None
    }
    
    if substitute_undefended_metrics:
        summary["substitute_undefended"] = {
            "name": "Substitute Random Forest (Trained on Undefended API)",
            "test_accuracy": substitute_undefended_metrics.get("accuracy"),
            "test_f1": substitute_undefended_metrics.get("f1_score"),
            "test_precision": substitute_undefended_metrics.get("precision"),
            "test_recall": substitute_undefended_metrics.get("recall"),
            "roc_auc": substitute_undefended_metrics.get("roc_auc"),
            "fidelity": fidelity_undefended.get("fidelity_percentage") if fidelity_undefended else None,
            "fidelity_agreement": fidelity_undefended.get("fidelity_agreement") if fidelity_undefended else None,
            "confusion_matrix": substitute_undefended_metrics.get("confusion_matrix")
        }
        
    if substitute_defended_metrics:
        summary["substitute_defended"] = {
            "name": "Substitute Random Forest (Trained on Defended API)",
            "test_accuracy": substitute_defended_metrics.get("accuracy"),
            "test_f1": substitute_defended_metrics.get("f1_score"),
            "test_precision": substitute_defended_metrics.get("precision"),
            "test_recall": substitute_defended_metrics.get("recall"),
            "roc_auc": substitute_defended_metrics.get("roc_auc"),
            "fidelity": fidelity_defended.get("fidelity_percentage") if fidelity_defended else None,
            "fidelity_agreement": fidelity_defended.get("fidelity_agreement") if fidelity_defended else None,
            "confusion_matrix": substitute_defended_metrics.get("confusion_matrix")
        }
        
    if substitute_undefended_metrics and substitute_defended_metrics and fidelity_undefended and fidelity_defended:
        fid_drop = round(
            fidelity_undefended.get("fidelity_percentage", 0) - fidelity_defended.get("fidelity_percentage", 0), 2
        )
        acc_drop = round(
            (substitute_undefended_metrics.get("accuracy", 0) - substitute_defended_metrics.get("accuracy", 0)) * 100, 2
        )
        summary["defense_impact"] = {
            "fidelity_reduction_percent": fid_drop,
            "substitute_accuracy_drop_percent": acc_drop,
            "genuine_user_accuracy_retention": None,
            "verdict": "Defense reduces observed substitute imitation; genuine-user utility requires separate measurement."
        }
        
    return summary

def save_metrics_to_disk(metrics: Dict[str, Any], filepath: Optional[str] = None) -> None:
    """Save metrics JSON to disk."""
    path = filepath or config.EVALUATION_METRICS_PATH
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def load_metrics_from_disk(filepath: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load metrics JSON from disk if it exists."""
    path = filepath or config.EVALUATION_METRICS_PATH
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
