"""Shared datatypes for the three Gemma 4 tiers.

Every backend (local llama.cpp, local Ollama, cloud Gemini API) returns the
same `BackendResult` shape so the router and the API layer never need to know
which tier actually produced the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

Tier = Literal["local", "cloud_fast", "cloud_strong"]


@dataclass
class ToolCall:
    """A single native function call emitted by a Gemma 4 model."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "args": self.args}


@dataclass
class BackendResult:
    """Uniform result envelope returned by every tier."""

    text: str = ""
    model: str = ""
    tier: Tier = "local"
    thinking: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "tier": self.tier,
            "thinking": self.thinking,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first well-formed JSON object out of a model response.

    Local models routinely wrap JSON in prose or a ```json fence, and Gemma
    sometimes prefixes it with "Here is the JSON:".  Rather than fight that
    with brittle string surgery we scan for the first balanced object.
    """
    if not text:
        return None

    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(.+?)```", text, flags=re.DOTALL)
    candidates.extend(fenced)
    candidates.append(text)

    for blob in candidates:
        blob = blob.strip()
        start = blob.find("{")
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(blob)):
            ch = blob[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    chunk = blob[start : idx + 1]
                    try:
                        parsed = json.loads(chunk)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
    return None
