from hermes_cli.request_routing import classify_request_tier, resolve_turn_route


def test_classify_basic_summary_request():
    assert classify_request_tier("Summarize the last three updates in plain English") == "basic"


def test_classify_complex_dev_request():
    assert classify_request_tier("Debug this failing pytest stack trace and patch the Python file under /Users/ailab/Hermes") == "complex"


def test_basic_route_prefers_codex_then_local_then_openrouter_fallbacks():
    route = resolve_turn_route(
        "Give me a short summary of today's status",
        current_model="gpt-5.4",
        current_runtime={"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"},
        routing_cfg={
            "basic": {"provider": "openai-codex", "model": "gpt-5.4", "base_url": "https://chatgpt.com/backend-api/codex"},
            "basic_fallbacks": [
                {"provider": "custom", "model": "gemma3:4b", "base_url": "http://localhost:11434/v1", "api_mode": "chat_completions"},
                {"provider": "custom", "model": "gemma4:e4b-it-q4_K_M", "base_url": "http://localhost:11434/v1", "api_mode": "chat_completions"},
                {"provider": "openrouter", "model": "openrouter/free", "base_url": "https://openrouter.ai/api/v1"},
            ],
        },
        runtime_resolver=lambda **_: {
            "provider": "openai-codex",
            "api_key": "***",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        },
    )
    assert route["tier"] == "basic"
    assert route["model"] == "gpt-5.4"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["fallback_model"][0]["model"] == "gemma3:4b"
    assert route["fallback_model"][1]["model"] == "gemma4:e4b-it-q4_K_M"
    assert route["fallback_model"][2]["model"] == "openrouter/free"


def test_complex_route_goes_direct_to_codex_with_fallback_chain():
    route = resolve_turn_route(
        "Implement the feature, update tests, and debug the failing Python code",
        current_model="gpt-5.4",
        current_runtime={"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"},
        routing_cfg={
            "complex": {
                "provider": "openai-codex",
                "model": "gpt-5.4",
                "base_url": "https://chatgpt.com/backend-api/codex",
                "fallback_providers": [
                    {"provider": "custom", "model": "gemma3:4b", "base_url": "http://localhost:11434/v1", "api_mode": "chat_completions"},
                    {"provider": "custom", "model": "gemma4:e4b-it-q4_K_M", "base_url": "http://localhost:11434/v1", "api_mode": "chat_completions"},
                    {"provider": "openrouter", "model": "openrouter/free", "base_url": "https://openrouter.ai/api/v1"},
                ],
            },
        },
        runtime_resolver=lambda **_: {
            "provider": "openai-codex",
            "api_key": "***",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        },
    )
    assert route["tier"] == "complex"
    assert route["model"] == "gpt-5.4"
    assert route["runtime"]["provider"] == "openai-codex"
    assert route["fallback_model"][0]["model"] == "gemma3:4b"
    assert route["fallback_model"][1]["model"] == "gemma4:e4b-it-q4_K_M"
    assert route["fallback_model"][2]["model"] == "openrouter/free"


def test_complex_route_uses_default_fallback_chain_when_not_configured():
    route = resolve_turn_route(
        "Refactor the Python module and update tests",
        current_model="gpt-5.4",
        current_runtime={"provider": "openai-codex", "base_url": "https://chatgpt.com/backend-api/codex"},
        routing_cfg={
            "complex": {"provider": "openai-codex", "model": "gpt-5.4", "base_url": "https://chatgpt.com/backend-api/codex"},
        },
        runtime_resolver=lambda **_: {
            "provider": "openai-codex",
            "api_key": "***",
            "base_url": "https://chatgpt.com/backend-api/codex",
            "api_mode": "codex_responses",
        },
    )
    assert route["tier"] == "complex"
    assert route["fallback_model"][0]["model"] == "gemma3:4b"
    assert route["fallback_model"][1]["model"] == "gemma4:e4b-it-q4_K_M"
    assert route["fallback_model"][2]["model"] == "openrouter/free"
