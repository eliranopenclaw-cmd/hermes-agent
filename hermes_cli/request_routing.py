from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

from hermes_cli.config import load_config
from hermes_cli.runtime_provider import resolve_runtime_provider
_PRIMARY_CODEX_ROUTE = {
    "provider": "openai-codex",
    "model": "gpt-5.4",
    "base_url": "https://chatgpt.com/backend-api/codex",
}

_DEFAULT_BASIC_ROUTE = {
    "provider": "custom",
    "model": "deepseek-r1:14b",
    "base_url": "http://localhost:11434/v1",
    "api_mode": "chat_completions",
}

_DEFAULT_BASIC_FALLBACKS = [
    {
        "provider": "custom",
        "model": "gemma3:4b",
        "base_url": "http://localhost:11434/v1",
        "api_mode": "chat_completions",
    },
    {
        "provider": "custom",
        "model": "gemma4:e4b",
        "base_url": "http://localhost:11434/v1",
        "api_mode": "chat_completions",
    },
]

_DEFAULT_COMPLEX_FALLBACKS = []

_COMPLEX_PATTERNS = [
    r"```",
    r"\b(debug|fix|bug|traceback|stack trace|exception|error log|failing test)\b",
    r"\b(implement|refactor|rewrite|patch|edit the code|write code|code review)\b",
    r"\b(repository|repo|git diff|pull request|pr review|branch|commit)\b",
    r"\b(function|class|module|api endpoint|schema|migration|build system)\b",
    r"\b(pytest|test suite|unit test|integration test|terminal command)\b",
    r"(/Users/|~/.hermes|\.py\b|\.ts\b|\.tsx\b|\.js\b|\.jsx\b|\.rs\b|\.go\b|\.java\b)",
]


def _config_section() -> dict:
    cfg = load_config() or {}
    return cfg.get("request_routing", {}) or {}


def classify_request_tier(user_message: str, routing_cfg: Optional[dict] = None) -> str:
    cfg = routing_cfg or {}
    text = (user_message or "").strip().lower()
    if not text:
        return "basic"

    custom_keywords = [str(item).strip().lower() for item in cfg.get("complex_keywords", []) if str(item).strip()]
    patterns = list(_COMPLEX_PATTERNS)
    if custom_keywords:
        patterns.append(r"\b(" + "|".join(re.escape(item) for item in custom_keywords) + r")\b")

    if len(text) >= int(cfg.get("complex_length_threshold", 700)):
        return "complex"
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return "complex"
    return "basic"


def _runtime_matches(current_model: str, current_runtime: dict, route_cfg: dict) -> bool:
    return (
        str(current_model or "").strip() == str(route_cfg.get("model") or "").strip()
        and str(current_runtime.get("provider") or "").strip().lower() == str(route_cfg.get("provider") or "").strip().lower()
        and str(current_runtime.get("base_url") or "").strip().rstrip("/") == str(route_cfg.get("base_url") or "").strip().rstrip("/")
    )


def _build_runtime_for_route(route_cfg: dict, runtime_resolver: Callable[..., dict], current_model: str = "", current_runtime: Optional[dict] = None) -> dict:
    provider = str(route_cfg.get("provider") or "").strip().lower()
    base_url = str(route_cfg.get("base_url") or "").strip()
    current_runtime = current_runtime or {}
    if _runtime_matches(current_model, current_runtime, route_cfg):
        return {
            "provider": current_runtime.get("provider"),
            "api_key": current_runtime.get("api_key"),
            "base_url": current_runtime.get("base_url"),
            "api_mode": current_runtime.get("api_mode"),
            "command": current_runtime.get("command"),
            "args": list(current_runtime.get("args") or []),
            "credential_pool": current_runtime.get("credential_pool"),
        }
    if provider == "custom":
        return {
            "provider": "custom",
            "api_key": str(route_cfg.get("api_key") or ""),
            "base_url": base_url,
            "api_mode": str(route_cfg.get("api_mode") or "chat_completions"),
            "command": None,
            "args": [],
            "credential_pool": None,
        }
    runtime = runtime_resolver(requested=provider or None, explicit_base_url=base_url or None)
    return {
        "provider": runtime.get("provider"),
        "api_key": runtime.get("api_key"),
        "base_url": runtime.get("base_url"),
        "api_mode": runtime.get("api_mode"),
        "command": runtime.get("command"),
        "args": list(runtime.get("args") or []),
        "credential_pool": runtime.get("credential_pool"),
    }


def resolve_turn_route(
    user_message: str,
    current_model: str,
    current_runtime: dict,
    *,
    routing_cfg: Optional[dict] = None,
    runtime_resolver: Callable[..., dict] = resolve_runtime_provider,
) -> dict:
    cfg = routing_cfg or _config_section()
    tier = classify_request_tier(user_message, cfg)

    complex_cfg = cfg.get("complex", {}) or {}
    basic_cfg = cfg.get("basic", {}) or _DEFAULT_BASIC_ROUTE
    basic_fallbacks = cfg.get("basic_fallbacks", _DEFAULT_BASIC_FALLBACKS) or _DEFAULT_BASIC_FALLBACKS

    if tier == "complex":
        route_cfg = {
            "provider": complex_cfg.get("provider") or _PRIMARY_CODEX_ROUTE["provider"],
            "model": complex_cfg.get("model") or current_model or _PRIMARY_CODEX_ROUTE["model"],
            "base_url": complex_cfg.get("base_url") or current_runtime.get("base_url") or _PRIMARY_CODEX_ROUTE["base_url"],
        }
        runtime = _build_runtime_for_route(route_cfg, runtime_resolver, current_model=current_model, current_runtime=current_runtime)
        fallback_model = complex_cfg.get("fallback_providers") or _DEFAULT_COMPLEX_FALLBACKS
    else:
        route_cfg = {
            "provider": basic_cfg.get("provider") or _PRIMARY_CODEX_ROUTE["provider"],
            "model": basic_cfg.get("model") or _PRIMARY_CODEX_ROUTE["model"],
            "base_url": basic_cfg.get("base_url") or _PRIMARY_CODEX_ROUTE["base_url"],
        }
        runtime = _build_runtime_for_route(route_cfg, runtime_resolver, current_model=current_model, current_runtime=current_runtime)
        fallback_model = basic_fallbacks

    model = str(route_cfg.get("model") or current_model).strip()
    runtime.setdefault("command", None)
    runtime.setdefault("args", [])
    runtime.setdefault("credential_pool", None)
    signature = (
        tier,
        model,
        runtime.get("provider"),
        runtime.get("base_url"),
        runtime.get("api_mode"),
        runtime.get("command"),
        tuple(runtime.get("args") or []),
    )
    return {
        "tier": tier,
        "model": model,
        "runtime": runtime,
        "fallback_model": fallback_model,
        "signature": signature,
    }
