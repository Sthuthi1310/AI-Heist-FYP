"""Thread-safe state manager for black-box attack simulations.
Shared between web.app and attack.attacker to ensure single source of truth across modules.
"""
import threading
from typing import Dict, Any, List, Optional

class AttackStateManager:
    """Thread-safe manager for attack simulation progress and live query streaming."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._state: Dict[str, Any] = {
            "is_running": False,
            "mode": "idle",
            "total_queries": 0,
            "completed_queries": 0,
            "successful_queries": 0,
            "blocked_queries": 0,
            "perturbed_queries": 0,
            "progress_percent": 0.0,
            "error_message": None,
            "last_result": None,
            "recent_logs": []
        }
        
    def start_attack(self, query_count: int, mode: str) -> None:
        """Initialize attack state for a new simulation run."""
        with self._lock:
            self._state = {
                "is_running": True,
                "mode": mode,
                "total_queries": query_count,
                "completed_queries": 0,
                "successful_queries": 0,
                "blocked_queries": 0,
                "perturbed_queries": 0,
                "progress_percent": 0.0,
                "error_message": None,
                "last_result": None,
                "recent_logs": [f"[{mode.upper()}] Initialized heist target with budget of {query_count} queries."]
            }
            
    def update_progress(
        self,
        completed: int,
        total: int,
        successful: int,
        blocked: int,
        perturbed: int,
        log_line: Optional[str] = None
    ) -> None:
        """Update live progress and counters."""
        with self._lock:
            percent = round((completed / max(1, total)) * 100.0, 1)
            self._state["completed_queries"] = completed
            self._state["successful_queries"] = successful
            self._state["blocked_queries"] = blocked
            self._state["perturbed_queries"] = perturbed
            self._state["progress_percent"] = min(100.0, percent)
            
            if log_line:
                self._state["recent_logs"].append(log_line)
                if len(self._state["recent_logs"]) > 50:
                    self._state["recent_logs"].pop(0)

    def finish_attack(self, result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> None:
        """Mark attack simulation as completed or failed."""
        with self._lock:
            self._state["is_running"] = False
            self._state["progress_percent"] = 100.0
            if result:
                self._state["last_result"] = result
                self._state["recent_logs"].append(
                    f"✓ [HEIST COMPLETE] Harvested {result.get('harvested_samples', 0)} samples in {result.get('elapsed_seconds', 0)}s."
                )
            if error:
                self._state["error_message"] = error
                self._state["recent_logs"].append(f"❌ [HEIST ERROR] {error}")
                
    def get_state(self) -> Dict[str, Any]:
        """Return snapshot of current state."""
        with self._lock:
            # Return deep-ish copy
            snapshot = dict(self._state)
            snapshot["recent_logs"] = list(self._state["recent_logs"])
            return snapshot

# Global singleton
attack_state_manager = AttackStateManager()
