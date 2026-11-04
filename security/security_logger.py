"""Structured security logging for AI Heist API.
Records access logs and security event audit trails.
"""
import time
import json
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import config

class SecurityLogger:
    """Thread-safe security event and access logger."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._recent_events: List[Dict[str, Any]] = []
        self._max_recent = 100
        
    def log_event(
        self,
        event_type: str,
        client_id: str,
        ip_address: str,
        endpoint: str,
        status_code: int,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Record a structured security event."""
        now = datetime.now()
        timestamp_iso = now.isoformat()
        
        record = {
            "timestamp": timestamp_iso,
            "event_type": event_type,
            "client_id": client_id,
            "ip_address": ip_address,
            "endpoint": endpoint,
            "status_code": status_code,
            "details": details or {}
        }
        
        with self._lock:
            self._recent_events.append(record)
            if len(self._recent_events) > self._max_recent:
                self._recent_events.pop(0)
                
            # Append to security events jsonl log
            try:
                config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
                with open(config.SECURITY_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as e:
                pass  # Graceful fallback if file writing fails
                
        return record
        
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the most recent security events in reverse chronological order."""
        with self._lock:
            return list(reversed(self._recent_events[-limit:]))
            
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute aggregate summary counts from the recorded events."""
        with self._lock:
            total = len(self._recent_events)
            blocked_429 = sum(1 for e in self._recent_events if e["status_code"] == 429)
            unauthorized_401 = sum(1 for e in self._recent_events if e["status_code"] == 401)
            perturbed = sum(1 for e in self._recent_events if e.get("details", {}).get("defense_applied"))
            suspicious = sum(1 for e in self._recent_events if e.get("details", {}).get("is_suspicious"))
            
            return {
                "recent_requests_count": total,
                "rate_limited_count": blocked_429,
                "unauthorized_count": unauthorized_401,
                "perturbed_responses_count": perturbed,
                "suspicious_queries_count": suspicious
            }
            
    def clear(self) -> None:
        """Clear memory cache of events."""
        with self._lock:
            self._recent_events.clear()

# Global security logger instance
security_logger = SecurityLogger()
