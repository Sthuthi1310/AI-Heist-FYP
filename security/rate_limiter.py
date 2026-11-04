"""Sliding-window rate limiter for the Flask API.
Defends against automated extraction loops by throttling high-frequency requests.
"""
import time
import threading
from typing import Dict, List, Tuple, Any
import config

class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter tracking requests per client."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._request_history: Dict[str, List[float]] = {}
        self._blocked_counts: Dict[str, int] = {}
        
    def check_rate_limit(self, client_id: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if request is permitted under rate limits.
        
        Returns:
            Tuple of (is_allowed: bool, metadata: dict)
        """
        # If rate limiting is disabled in security config, allow all
        if not config.SECURITY_CONFIG.get("rate_limiting_enabled", False):
            return True, {"enabled": False, "reason": "rate_limiting_disabled"}
            
        rate_limit_per_minute = config.SECURITY_CONFIG.get("rate_limit_per_minute", 60)
        burst_limit_per_second = config.SECURITY_CONFIG.get("burst_limit_per_second", 5)
        now = time.time()
        
        with self._lock:
            if client_id not in self._request_history:
                self._request_history[client_id] = []
                self._blocked_counts[client_id] = 0
                
            history = self._request_history[client_id]
            # Prune requests older than 60 seconds
            window_start = now - 60.0
            history = [t for t in history if t > window_start]
            self._request_history[client_id] = history
            
            # Check burst limit (last 1 second)
            recent_second = [t for t in history if t > (now - 1.0)]
            if len(recent_second) >= burst_limit_per_second:
                self._blocked_counts[client_id] += 1
                return False, {
                    "enabled": True,
                    "reason": "burst_limit_exceeded",
                    "burst_count": len(recent_second),
                    "burst_limit": burst_limit_per_second,
                    "retry_after_seconds": 1.0,
                    "total_blocked": self._blocked_counts[client_id]
                }
                
            # Check 1-minute limit
            if len(history) >= rate_limit_per_minute:
                self._blocked_counts[client_id] += 1
                retry_after = round(60.0 - (now - history[0]), 1)
                return False, {
                    "enabled": True,
                    "reason": "minute_limit_exceeded",
                    "window_count": len(history),
                    "window_limit": rate_limit_per_minute,
                    "retry_after_seconds": max(retry_after, 1.0),
                    "total_blocked": self._blocked_counts[client_id]
                }
                
            # Request allowed; record timestamp
            history.append(now)
            return True, {
                "enabled": True,
                "window_count": len(history),
                "window_limit": rate_limit_per_minute,
                "burst_count": len(recent_second) + 1,
                "burst_limit": burst_limit_per_second
            }
            
    def get_stats(self) -> Dict[str, Any]:
        """Return rate limiter statistics."""
        with self._lock:
            return {
                "active_clients": len(self._request_history),
                "blocked_counts": dict(self._blocked_counts),
                "rate_limiting_enabled": config.SECURITY_CONFIG.get("rate_limiting_enabled", False),
                "rate_limit_per_minute": config.SECURITY_CONFIG.get("rate_limit_per_minute", 60),
                "burst_limit_per_second": config.SECURITY_CONFIG.get("burst_limit_per_second", 5)
            }
            
    def reset(self) -> None:
        """Clear rate limit history."""
        with self._lock:
            self._request_history.clear()
            self._blocked_counts.clear()

# Global rate limiter instance
rate_limiter = SlidingWindowRateLimiter()
