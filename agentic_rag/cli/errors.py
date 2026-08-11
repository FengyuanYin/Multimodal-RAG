"""Stable user-facing errors and process exit codes."""

from __future__ import annotations

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    USAGE = 2
    CONFIG = 3
    UPSTREAM = 4
    CANCELLED = 130
    INTERNAL = 70


class AutoMemoryError(Exception):
    code = "AUTOMEMORY_ERROR"
    exit_code = ExitCode.INTERNAL

    def __init__(self, message: str, *, hint: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class UsageError(AutoMemoryError):
    code = "USAGE_ERROR"
    exit_code = ExitCode.USAGE


class ConfigurationError(AutoMemoryError):
    code = "CONFIG_ERROR"
    exit_code = ExitCode.CONFIG


class UpstreamError(AutoMemoryError):
    code = "UPSTREAM_ERROR"
    exit_code = ExitCode.UPSTREAM


class CancelledError(AutoMemoryError):
    code = "CANCELLED"
    exit_code = ExitCode.CANCELLED

    def __init__(self, message: str = "Operation cancelled") -> None:
        super().__init__(message)


class SecurityError(AutoMemoryError):
    code = "SECURITY_ERROR"
    exit_code = ExitCode.USAGE


def classify_http_error(status: int, service: str) -> UpstreamError:
    if status in {401, 403}:
        error = UpstreamError(f"{service} rejected the configured credential", hint=f"Run /secret test {service.lower()}")
        error.code = "UPSTREAM_AUTH"
        error.status = status
        return error
    if status == 402:
        error = UpstreamError(f"{service} account has insufficient credit", hint="Check the provider account balance")
        error.code = "UPSTREAM_QUOTA"
        error.status = status
        return error
    if status == 429:
        error = UpstreamError(f"{service} rate limit or quota was reached", hint="Wait, reduce concurrency, or check account quota")
        error.code = "UPSTREAM_RATE_LIMIT"
        error.status = status
        return error
    if status in {400, 404, 422}:
        error = UpstreamError(f"{service} rejected the configured model or request (HTTP {status})", hint="Check the model name and provider-compatible endpoint")
        error.code = "UPSTREAM_MODEL"
        error.status = status
        return error
    error = UpstreamError(f"{service} request failed (HTTP {status})")
    error.code = "UPSTREAM_HTTP"
    error.status = status
    return error
