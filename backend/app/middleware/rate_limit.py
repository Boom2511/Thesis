# -*- coding: utf-8 -*-
"""Simple Rate Limiting Middleware"""

from fastapi import Request, HTTPException
from typing import Dict
import time

class SimpleRateLimiter:
    """Simple in-memory rate limiter"""

    def __init__(self, requests_per_minute: int = 10):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = {}

    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed"""
        now = time.time()
        minute_ago = now - 60

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if req_time > minute_ago
            ]
        else:
            self.requests[client_ip] = []

        # Check limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False

        # Add new request
        self.requests[client_ip].append(now)
        return True

    def get_remaining(self, client_ip: str) -> int:
        """Get remaining requests"""
        now = time.time()
        minute_ago = now - 60

        if client_ip not in self.requests:
            return self.requests_per_minute

        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]

        return max(0, self.requests_per_minute - len(recent_requests))

# Global rate limiter
rate_limiter = SimpleRateLimiter(requests_per_minute=20)

async def rate_limit_middleware(request: Request, call_next):
    """Middleware to check rate limits"""
    if request.url.path in ['/health', '/']:
        return await call_next(request)

    client_ip = request.client.host if request.client else 'unknown'

    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many requests. Limit: {rate_limiter.requests_per_minute}/minute",
            headers={"Retry-After": "60"}
        )

    response = await call_next(request)
    remaining = rate_limiter.get_remaining(client_ip)
    response.headers["X-RateLimit-Limit"] = str(rate_limiter.requests_per_minute)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    return response
