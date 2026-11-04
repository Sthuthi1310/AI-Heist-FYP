"""Suspicious query detection for identifying black-box model stealing attacks.
Analyzes request inter-arrival velocity, burst volume, payload repetition, and decision boundary probing.
"""
import time
import threading
import hashlib
import json
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

import config

class SuspiciousQueryDetector:
    """Detects automated model extraction patterns across incoming API queries."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._client_history: Dict[str, List[Dict[str, Any]]] = {}
        self._suspicious_counts: Dict[str, int] = {}
        
    def _compute_payload_hash(self, payload: Dict[str, Any]) -> str:
        """Create a deterministic hash of the feature payload."""
        sorted_items = sorted([(k, str(v)) for k, v in payload.items()])
        dumped = json.dumps(sorted_items)
        return hashlib.md5(dumped.encode("utf-8")).hexdigest()
        
    def analyze_query(
        self,
        client_id: str,
        features_dict: Dict[str, Any],
        predicted_probability: Optional[float] = None
    ) -> Dict[str, Any]:
        """Analyze an incoming query and calculate a suspicion score (0.0 to 1.0).
        
        Returns:
            Dict containing:
            - suspicion_score (float 0.0-1.0)
            - is_suspicious (bool)
            - reasons (list of trigger explanations)
            - metrics (velocity, burst_rate, repetition, boundary_proximity)
        """
        now = time.time()
        payload_hash = self._compute_payload_hash(features_dict)
        
        with self._lock:
            if client_id not in self._client_history:
                self._client_history[client_id] = []
                self._suspicious_counts[client_id] = 0
                
            history = self._client_history[client_id]
            # Keep history within last 60 seconds
            history = [q for q in history if q["time"] > (now - 60.0)]
            self._client_history[client_id] = history
            
            reasons = []
            velocity_score = 0.0
            burst_score = 0.0
            repetition_score = 0.0
            boundary_score = 0.0
            
            if history:
                # 1. Velocity Analysis: Inter-arrival time between last query and current query
                last_query_time = history[-1]["time"]
                interval_ms = (now - last_query_time) * 1000.0
                min_interval = config.SECURITY_CONFIG.get("min_query_interval_ms", 50)
                
                if interval_ms < min_interval:
                    # High automated velocity
                    velocity_score = 0.50
                    reasons.append(f"Sub-human query interval ({interval_ms:.1f}ms < {min_interval}ms)")
                elif interval_ms < 250.0:
                    velocity_score = 0.25
                    reasons.append(f"Rapid programmatic query cadence ({interval_ms:.1f}ms)")
                    
                # 2. Burst Volume Analysis: Number of queries in last 10 seconds
                ten_sec_queries = [q for q in history if q["time"] > (now - 10.0)]
                burst_count = len(ten_sec_queries)
                if burst_count >= 15:
                    burst_score = 0.40
                    reasons.append(f"High-frequency query burst ({burst_count} req / 10s)")
                elif burst_count >= 8:
                    burst_score = 0.20
                    reasons.append(f"Moderate query burst ({burst_count} req / 10s)")
                    
                # 3. Payload Repetition Analysis: Duplicate feature payloads
                recent_hashes = [q["hash"] for q in history[-20:]]
                duplicate_count = recent_hashes.count(payload_hash)
                if duplicate_count >= 3:
                    repetition_score = 0.30
                    reasons.append(f"Repeated identical feature payload ({duplicate_count} times)")
            
            # 4. Decision Boundary Probing Analysis
            # Model extraction attacks often probe regions where P(Y=1) is close to 0.5
            if predicted_probability is not None:
                prob_distance_to_boundary = abs(predicted_probability - 0.5)
                if prob_distance_to_boundary < 0.05:
                    boundary_score = 0.25
                    reasons.append(f"Critical decision boundary probe (prob={predicted_probability:.3f})")
                elif prob_distance_to_boundary < 0.10:
                    boundary_score = 0.15
                    
            # Combined suspicion score (capped at 1.0)
            combined_score = min(1.0, velocity_score + burst_score + repetition_score + boundary_score)
            
            # If feature detector is disabled globally, mark not suspicious
            is_enabled = config.SECURITY_CONFIG.get("suspicious_detection_enabled", False)
            threshold = config.SECURITY_CONFIG.get("suspicion_threshold", 0.50)
            
            is_suspicious = is_enabled and (combined_score >= threshold)
            
            if is_suspicious:
                self._suspicious_counts[client_id] += 1
                
            # Append current query to history
            history.append({
                "time": now,
                "hash": payload_hash,
                "prob": predicted_probability,
                "score": combined_score,
                "suspicious": is_suspicious
            })
            
            return {
                "suspicion_score": round(combined_score, 3),
                "is_suspicious": is_suspicious,
                "detection_enabled": is_enabled,
                "threshold": threshold,
                "reasons": reasons,
                "metrics": {
                    "velocity_score": round(velocity_score, 2),
                    "burst_score": round(burst_score, 2),
                    "repetition_score": round(repetition_score, 2),
                    "boundary_score": round(boundary_score, 2)
                },
                "total_suspicious_for_client": self._suspicious_counts[client_id]
            }
            
    def get_stats(self) -> Dict[str, Any]:
        """Return summary of detection activity."""
        with self._lock:
            total_suspicious = sum(self._suspicious_counts.values())
            return {
                "active_clients": len(self._client_history),
                "total_suspicious_queries": total_suspicious,
                "suspicious_by_client": dict(self._suspicious_counts),
                "detection_enabled": config.SECURITY_CONFIG.get("suspicious_detection_enabled", False),
                "threshold": config.SECURITY_CONFIG.get("suspicion_threshold", 0.50)
            }
            
    def reset(self) -> None:
        """Reset detection history."""
        with self._lock:
            self._client_history.clear()
            self._suspicious_counts.clear()

# Global suspicious query detector instance
suspicious_detector = SuspiciousQueryDetector()
