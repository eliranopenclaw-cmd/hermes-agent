import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry
from gateway.run import GatewayRunner


def test_checkpoint_continuity_capsule_writes_active_capsule(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner._continuity_storage_root = tmp_path

    history = [
        {"role": "user", "content": "Please continue the deployment fix."},
        {"role": "assistant", "content": "I inspected the failing restart path."},
    ]

    capsule_path = runner._checkpoint_continuity_capsule(
        session_key="agent:main:telegram:dm:42:u1",
        session_id="sess-1",
        history=history,
        reason="restart_timeout",
        latest_user_text="Please continue the deployment fix.",
        profile="ops",
    )

    assert capsule_path is not None
    active_pointer = tmp_path / "agent_main_telegram_dm_42_u1" / "active.json"
    assert active_pointer.exists()
    payload = json.loads(capsule_path.read_text(encoding="utf-8"))
    assert payload["scope_key"] == "agent_main_telegram_dm_42_u1"
    assert payload["source_session_ids"] == ["sess-1"]


def test_inject_continuity_resume_packet_appends_resume_state_note(tmp_path):
    runner = object.__new__(GatewayRunner)
    runner._continuity_storage_root = tmp_path

    session_entry = SessionEntry(
        session_key="agent:main:telegram:dm:42:u1",
        session_id="sess-1",
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="dm",
        resume_pending=True,
        resume_reason="restart_timeout",
    )

    runner._checkpoint_continuity_capsule(
        session_key=session_entry.session_key,
        session_id=session_entry.session_id,
        history=[{"role": "user", "content": "Resume the failed review."}],
        reason="restart_timeout",
        latest_user_text="Resume the failed review.",
        profile="ops",
    )

    prompt = runner._inject_continuity_resume_packet(
        "## Current Session Context\nSource: Telegram",
        session_entry,
        session_entry.session_key,
    )

    assert "CONTINUITY CAPSULE" in prompt
    assert "Resume the failed review." in prompt
    assert "interrupted by a gateway restart" in prompt.lower()


def test_compress_context_checkpoints_continuity_before_compress(monkeypatch, tmp_path):
    import run_agent

    captured = {}

    def _fake_checkpoint(**kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setenv("HERMES_SESSION_KEY", "agent:main:telegram:dm:42:u1")
    monkeypatch.setattr(run_agent, "checkpoint_runtime_continuity", _fake_checkpoint)

    class DummyCompressor:
        compression_count = 1
        last_prompt_tokens = 0
        last_completion_tokens = 0

        def compress(self, messages, current_tokens=None, focus_topic=None):
            return [{"role": "assistant", "content": "compressed summary"}]

    class DummyTodoStore:
        def format_for_injection(self):
            return ""

    class DummyAgent:
        def __init__(self):
            self.session_id = "sess-1"
            self.model = "demo-model"
            self.platform = "telegram"
            self.context_compressor = DummyCompressor()
            self._memory_manager = None
            self._todo_store = DummyTodoStore()
            self._session_db = None
            self.logs_dir = tmp_path
            self.session_log_file = tmp_path / "session.json"

        def flush_memories(self, *_args, **_kwargs):
            return None

        def _invalidate_system_prompt(self):
            return None

        def _build_system_prompt(self, system_message):
            return system_message or "system"

        def _vprint(self, *_args, **_kwargs):
            return None

    agent = DummyAgent()
    messages = [
        {"role": "user", "content": "Keep debugging the failing restart."},
        {"role": "assistant", "content": "I have the latest logs."},
    ]

    compressed, _ = run_agent.AIAgent._compress_context(
        agent,
        messages,
        "system prompt",
        approx_tokens=1234,
        task_id="default",
    )

    assert compressed == [{"role": "assistant", "content": "compressed summary"}]
    assert captured["session_key"] == "agent:main:telegram:dm:42:u1"
    assert captured["session_id"] == "sess-1"
    assert captured["reason"] == "context_compression"
    assert captured["latest_user_text"] == "Keep debugging the failing restart."
