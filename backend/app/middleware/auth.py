"""
Authentication Middleware
Simple token-based authentication for WebSocket and API endpoints
"""

from fastapi import Header, HTTPException, WebSocket, status
from typing import Optional
import secrets
import time
from datetime import datetime, timedelta
import hashlib

# In production, use environment variables or secure key management
SECRET_KEY = "your-secret-key-change-in-production"

# Session storage (use Redis in production)
active_sessions = {}


class SessionManager:
    """Manages session tokens for WebSocket connections"""

    @staticmethod
    def generate_token() -> str:
        """Generate a secure session token"""
        token = secrets.token_urlsafe(32)
        timestamp = time.time()

        active_sessions[token] = {
            'created_at': timestamp,
            'expires_at': timestamp + 3600,  # 1 hour
            'requests': 0
        }

        return token

    @staticmethod
    def validate_token(token: str) -> bool:
        """Validate a session token"""
        if not token or token not in active_sessions:
            return False

        session = active_sessions[token]

        # Check expiration
        if time.time() > session['expires_at']:
            del active_sessions[token]
            return False

        # Rate limit per session (max 100 requests per hour)
        if session['requests'] > 100:
            return False

        session['requests'] += 1
        return True

    @staticmethod
    def cleanup_expired_sessions():
        """Remove expired sessions"""
        current_time = time.time()
        expired = [
            token for token, session in active_sessions.items()
            if current_time > session['expires_at']
        ]

        for token in expired:
            del active_sessions[token]


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Verify API key for protected endpoints

    For development: API key is optional
    For production: Make it required and use proper key management
    """
    # In development mode, allow requests without API key
    if not x_api_key:
        return "anonymous"

    # Validate API key format
    if len(x_api_key) < 16:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format"
        )

    # In production, validate against database
    # For now, accept any key with correct format
    return x_api_key


async def verify_websocket_token(websocket: WebSocket, token: str) -> bool:
    """
    Verify WebSocket connection token

    Args:
        websocket: WebSocket connection
        token: Session token

    Returns:
        True if valid, False otherwise
    """
    if not SessionManager.validate_token(token):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return False

    return True


# API Key middleware (optional - for production use)
async def api_key_auth(x_api_key: str = Header(...)):
    """
    Require API key for certain endpoints

    Usage:
        @router.get("/protected", dependencies=[Depends(api_key_auth)])
    """
    if not x_api_key or x_api_key != SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API key"
        )
    return x_api_key
