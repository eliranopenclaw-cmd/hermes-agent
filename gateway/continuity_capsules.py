from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

DEFAULT_REPO_ROOT = Path(os.getenv("HERMES_REPO_ROOT", "/Users/ailab/Hermes"))
DEFAULT_STORAGE_ROOT = DEFAULT_REPO_ROOT / "data" / "continuity_capsules"

STATUS_VALUES = {"active", "blocked", "waiting", "historical", "superseded"}
PROFILE_VALUES = {"work", "ops", "research"}
RESET_POSTURE_VALUES = {"safe_to_resume", "needs_operator_review", "superseded"}
TOKEN_BUDGET_VALUES = {"tiny", "small", "medium"}
SCOPE_TYPE_VALUES = {"task", "incident", "research", "ops-run", "build"}
LIST_FIELDS = (
    "completed_items",
    "open_items",
    "important_paths",
    "important_commands",
    "decisions",
    "constraints",
    "risks",
    "source_session_ids",
    "source_note_paths",
)


def continuity_storage_root(explicit_root: str | Path | None = None) -> Path:
    return Path(explicit_root) if explicit_root else DEFAULT_STORAGE_ROOT


def sanitize_scope_key(value: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip())
    return text.strip("_") or "session"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _latest_message_by_role(messages: Iterable[dict[str, Any]], role: str) -> str:
    for msg in reversed(list(messages)):
        if msg.get("role") == role:
            text = _message_text(msg.get("content"))
            if text:
                return text
    return ""


def _trim_line(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text.strip())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


def normalize_capsule(capsule: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(capsule)
    for field in LIST_FIELDS:
        values = normalized.get(field, []) or []
        if not isinstance(values, list):
            raise ValueError(f"{field} must be a list")
        deduped: list[Any] = []
        seen: set[Any] = set()
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            deduped.append(value)
        normalized[field] = deduped
    return normalized


def validate_capsule(capsule: dict[str, Any]) -> None:
    normalized = normalize_capsule(capsule)
    required = (
        "capsule_id",
        "scope_type",
        "scope_key",
        "status",
        "profile",
        "title",
        "goal",
        "active_task",
        "completed_items",
        "open_items",
        "exact_next_action",
        "important_paths",
        "constraints",
        "source_session_ids",
        "source_note_paths",
        "freshness_ts",
        "reset_posture",
        "token_budget_class",
    )
    missing = [field for field in required if field not in normalized]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    if normalized["scope_type"] not in SCOPE_TYPE_VALUES:
        raise ValueError("Invalid scope_type")
    if normalized["status"] not in STATUS_VALUES:
        raise ValueError("Invalid status")
    if normalized["profile"] not in PROFILE_VALUES:
        raise ValueError("Invalid profile")
    if normalized["reset_posture"] not in RESET_POSTURE_VALUES:
        raise ValueError("Invalid reset_posture")
    if normalized["token_budget_class"] not in TOKEN_BUDGET_VALUES:
        raise ValueError("Invalid token_budget_class")
    if not str(normalized["exact_next_action"] or "").strip():
        raise ValueError("exact_next_action must be non-empty")


def _scope_dir(root: Path, scope_key: str) -> Path:
    return root / sanitize_scope_key(scope_key)


def _capsule_path(root: Path, scope_key: str, capsule_id: str) -> Path:
    return _scope_dir(root, scope_key) / f"{capsule_id}.json"


def _pointer_path(root: Path, scope_key: str) -> Path:
    return _scope_dir(root, scope_key) / "active.json"


def save_capsule(root: Path, capsule: dict[str, Any]) -> Path:
    normalized = normalize_capsule(capsule)
    validate_capsule(normalized)
    root.mkdir(parents=True, exist_ok=True)
    scope_key = sanitize_scope_key(normalized["scope_key"])
    scope_dir = _scope_dir(root, scope_key)
    scope_dir.mkdir(parents=True, exist_ok=True)
    pointer = _pointer_path(root, scope_key)
    if normalized["status"] == "active" and pointer.exists():
        current = json.loads(pointer.read_text(encoding="utf-8"))
        old_path = _capsule_path(root, scope_key, current["capsule_id"])
        if old_path.exists() and current["capsule_id"] != normalized["capsule_id"]:
            old_capsule = json.loads(old_path.read_text(encoding="utf-8"))
            old_capsule["status"] = "superseded"
            old_capsule["reset_posture"] = "superseded"
            old_path.write_text(json.dumps(old_capsule, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_path = _capsule_path(root, scope_key, normalized["capsule_id"])
    out_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if normalized["status"] == "active":
        pointer.write_text(json.dumps({"scope_key": scope_key, "capsule_id": normalized["capsule_id"]}, indent=2) + "\n", encoding="utf-8")
    return out_path


def load_active_capsule(root: Path, scope_key: str) -> dict[str, Any] | None:
    pointer = _pointer_path(root, scope_key)
    if not pointer.exists():
        return None
    current = json.loads(pointer.read_text(encoding="utf-8"))
    capsule_path = _capsule_path(root, scope_key, current["capsule_id"])
    if not capsule_path.exists():
        return None
    return json.loads(capsule_path.read_text(encoding="utf-8"))


def build_capsule_from_messages(
    *,
    session_key: str,
    session_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    latest_user_text: str = "",
    profile: str = "ops",
    reset_posture: str = "safe_to_resume",
) -> dict[str, Any]:
    scope_key = sanitize_scope_key(session_key)
    latest_user = latest_user_text.strip() or _latest_message_by_role(messages, "user")
    latest_assistant = _latest_message_by_role(messages, "assistant")
    active_task = _trim_line(latest_user or f"Resume interrupted session {scope_key}")
    title = _trim_line(active_task, limit=120)
    completed = [f"Captured continuity capsule before {reason.replace('_', ' ')}."]
    if latest_assistant:
        completed.append(f"Latest assistant state: {_trim_line(latest_assistant, limit=180)}")
    open_items = [f"Resume from latest user request: {active_task}"]
    exact_next_action = f"Continue from the latest user request: {active_task}"
    return {
        "capsule_id": f"ctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
        "scope_type": "task",
        "scope_key": scope_key,
        "status": "active",
        "profile": profile,
        "title": title,
        "goal": active_task,
        "active_task": active_task,
        "completed_items": completed,
        "open_items": open_items,
        "exact_next_action": exact_next_action,
        "important_paths": [],
        "important_commands": [],
        "decisions": [f"Continuity checkpoint reason: {reason}."],
        "constraints": ["Resume from continuity state before relying on deep transcript replay."],
        "risks": [],
        "source_session_ids": [session_id] if session_id else [],
        "source_note_paths": [],
        "freshness_ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "reset_posture": reset_posture,
        "token_budget_class": "small",
    }


def checkpoint_runtime_continuity(
    *,
    session_key: str,
    session_id: str,
    messages: list[dict[str, Any]],
    reason: str,
    latest_user_text: str = "",
    profile: str = "ops",
    root: str | Path | None = None,
    reset_posture: str = "safe_to_resume",
) -> Path:
    storage_root = continuity_storage_root(root)
    capsule = build_capsule_from_messages(
        session_key=session_key,
        session_id=session_id,
        messages=messages,
        reason=reason,
        latest_user_text=latest_user_text,
        profile=profile,
        reset_posture=reset_posture,
    )
    return save_capsule(storage_root, capsule)


def render_resume_packet(capsule: dict[str, Any], *, max_chars: int = 2400) -> str:
    normalized = normalize_capsule(capsule)
    validate_capsule(normalized)
    parts = [
        "[CONTINUITY CAPSULE — ACTIVE STATE]",
        f"Title: {normalized['title']}",
        f"Active task: {normalized['active_task']}",
        "Completed:\n- " + "\n- ".join(normalized["completed_items"] or ["None recorded."]),
        "Open:\n- " + "\n- ".join(normalized["open_items"] or ["None recorded."]),
        f"Exact next action: {normalized['exact_next_action']}",
        f"Freshness: {normalized['freshness_ts']}",
        f"Reset posture: {normalized['reset_posture']}",
    ]
    packet = "\n\n".join(parts)
    if len(packet) > max_chars:
        packet = packet[: max_chars - 3].rstrip() + "..."
    return packet


def build_resume_packet_for_scope(root: str | Path | None, scope_key: str) -> str | None:
    capsule = load_active_capsule(continuity_storage_root(root), scope_key)
    if not capsule:
        return None
    return render_resume_packet(capsule)
