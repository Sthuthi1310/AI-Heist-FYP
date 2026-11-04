"""Output perturbation defense for black-box model extraction.
Injects controlled label flipping and probability noise into API responses for suspicious extraction queries.
"""
import random
from typing import Tuple, Dict, Any
import numpy as np

import config

def apply_output_perturbation(
    original_label: int,
    original_prob: float,
    is_suspicious: bool
) -> Tuple[int, float, bool, str]:
    """Apply output perturbation defense if enabled.
    
    Args:
        original_label: Model predicted class (0 or 1)
        original_prob: Model predicted positive class probability
        is_suspicious: Whether the query was flagged by suspicious query detector
        
    Returns:
        Tuple of:
        - returned_label (int)
        - returned_prob (float)
        - defense_applied (bool)
        - defense_reason (str)
    """
    sec_cfg = config.SECURITY_CONFIG
    is_enabled = sec_cfg.get("perturbation_enabled", False)
    
    if not is_enabled:
        return original_label, original_prob, False, "perturbation_disabled"
        
    mode = sec_cfg.get("perturbation_mode", "adaptive")
    flip_prob = sec_cfg.get("flip_probability", 0.35)
    noise_std = sec_cfg.get("noise_std_dev", 0.15)
    
    # Adaptive mode: Perturb ONLY if the query was flagged as suspicious
    should_perturb = False
    reason = "benign_query_preserved"
    
    if mode == "adaptive":
        if is_suspicious:
            should_perturb = True
            reason = "adaptive_perturbation_on_suspicious_query"
        else:
            should_perturb = False
            reason = "genuine_query_unaltered"
    elif mode == "always":
        should_perturb = True
        reason = "always_perturb_mode"
        
    if not should_perturb:
        return original_label, original_prob, False, reason
        
    # Apply perturbation:
    # 1. Soft perturbation: inject Gaussian noise into the probability
    noise = np.random.normal(0.0, noise_std)
    perturbed_prob = float(np.clip(original_prob + noise, 0.01, 0.99))
    
    # 2. Hard label perturbation: flip the binary label with flip_prob
    if random.random() < flip_prob:
        perturbed_label = 1 - original_label  # Flip 0 -> 1 or 1 -> 0
        # Ensure probability direction matches flipped label
        if perturbed_label == 1 and perturbed_prob < 0.5:
            perturbed_prob = 1.0 - perturbed_prob
        elif perturbed_label == 0 and perturbed_prob > 0.5:
            perturbed_prob = 1.0 - perturbed_prob
    else:
        perturbed_label = original_label
        
    return perturbed_label, round(perturbed_prob, 4), True, reason
