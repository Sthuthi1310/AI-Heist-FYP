"""API key authentication helpers for AI Heist API.
"""
from functools import wraps
from typing import Dict, Any, Optional, Tuple
from flask import request, jsonify
import config

def get_client_identifier() -> Tuple[str, str]:
    """Retrieve client identifier (API key name or IP address) and client role."""
    api_key = request.headers.get("X-API-Key")
    ip_addr = request.remote_addr or "127.0.0.1"
    
    if api_key and api_key in config.SECURITY_CONFIG.get("api_keys", {}):
        key_info = config.SECURITY_CONFIG["api_keys"][api_key]
        return api_key, key_info.get("role", "genuine")
    
    # If no valid API key header
    return ip_addr, "anonymous"

def require_api_key(f):
    """Decorator to enforce API key authentication when configured."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not config.SECURITY_CONFIG.get("auth_required", False):
            # Auth is disabled globally
            return f(*args, **kwargs)
            
        api_key = request.headers.get("X-API-Key")
        valid_keys = config.SECURITY_CONFIG.get("api_keys", {})
        
        if not api_key or api_key not in valid_keys:
            return jsonify({
                "status": "error",
                "code": 401,
                "message": "Unauthorized: A valid X-API-Key header is required to access this endpoint."
            }), 401
            
        return f(*args, **kwargs)
    return decorated_function
