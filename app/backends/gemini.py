"""Cloud tier: Gemma 4 through the Gemini API.

Verified behaviour of the Gemini API for Gemma 4 (see NOTES-api-findings.md):

* Only ``gemma-4-26b-a4b-it`` and ``gemma-4-31b-it`` are exposed.  The E2B and
  E4B variants are local-only and return 404 here.
* The service intermittently returns 500/503 for both models, including for
  plain text prompts.  Every call is therefore retried with backoff.
* New-style API keys are issued as ``AQ.Ab...``.  A key pasted without the
  dot is rejected with ``API_KEY_INVALID``, so we try a repaired candidate.
* ``google_search`` grounding is accepted but returns **zero grounding
  chunks**, so the answer is effectively ungrounded.  We therefore treat
  grounding as best-effort and expose a real HTTP fetch tool instead.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from google import genai
from google.genai import types

from ..config import settings
from .base import BackendResult, ToolCall

# Errors worth retrying: the backend is flaky, not the request.
_RETRYABLE = (500, 503, 429)
_RETRYABLE_KEYWORDS = (
    "500",
    "503",
    "429",
    "unavailable",
    "resource_exhausted",
    "internal error",
    "high demand",
    "deadline_exceeded",
    "timeout",
    "connection reset",
    "remote disconnected",
)


def is_retryable_error(exc: Exception) -> bool:
    status = getattr(exc, "code", None) or _status_of(exc)
    if status in _RETRYABLE:
        return True
    msg = str(exc).lower()
    return any(k in msg for k in _RETRYABLE_KEYWORDS)


def key_candidates(key: str) -> list[str]:
    """Return the key plus repaired variants we are willing to try.

    Google migrated Gemini API keys from ``AIza...`` to ``AQ.Ab...``.  When a
    key is pasted through a client that strips punctuation the dot is lost and
    the key is rejected as invalid.  Adding the dot back restores it.
    """
    candidates = [key]
    if key.startswith("AQ") and not key.startswith("AQ."):
        candidates.append("AQ." + key[2:])
    return candidates


class GeminiBackend:
    """Thin, retry-hardened wrapper around ``google.genai`` for Gemma 4."""

    def __init__(self) -> None:
        self._client: genai.Client | None = None
        self._client_key: str | None = None
        self._lock = asyncio.Lock()

    # -- client ----------------------------------------------------------
    def _build_client(self, key: str) -> genai.Client:
        return genai.Client(api_key=key)

    async def client(self) -> genai.Client | None:
        """Return a client bound to a key that actually authenticates."""
        if not settings.cloud_enabled:
            return None
        raw = settings.gemini_api_key
        if not raw:
            return None

        if self._client is not None and self._client_key == raw:
            return self._client

        async with self._lock:
            if self._client is not None and self._client_key == raw:
                return self._client
            last_error: Exception | None = None
            for candidate in key_candidates(raw):
                try:
                    client = self._build_client(candidate)
                    await asyncio.to_thread(client.models.list)
                    self._client = client
                    self._client_key = raw
                    return client
                except Exception as exc:  # noqa: BLE001 - probe each candidate
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
            return None

    # -- retry -----------------------------------------------------------
    async def _with_retry(self, coro_factory, attempts: int = 4, label: str = ""):
        """Run a generate_content call, retrying transient server errors."""
        delay = 2.0
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                client = await self.client()
                if client is None:
                    raise RuntimeError("Gemini API is not configured (missing GEMINI_API_KEY)")
                return await asyncio.to_thread(coro_factory, client)
            except Exception as exc:  # noqa: BLE001
                last = exc
                if not is_retryable_error(exc):
                    raise
                if attempt == attempts - 1:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 1.8, 12.0)
        assert last is not None
        raise last

    # -- generation ------------------------------------------------------
    async def generate(
        self,
        *,
        model: str,
        contents: Any,
        system_instruction: str | None = None,
        tools: list[types.Tool] | None = None,
        google_search: bool = False,
        thinking_level: str | None = None,
        response_schema: types.Schema | None = None,
        response_mime_type: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        tier: str = "cloud_fast",
    ) -> BackendResult:
        """Call Gemma 4 on the Gemini API and normalise the result."""
        started = time.perf_counter()

        def build(client: genai.Client):
            tool_list: list[Any] = list(tools or [])
            if google_search:
                tool_list.append({"google_search": {}})
            config_kwargs: dict[str, Any] = {}
            if system_instruction:
                config_kwargs["system_instruction"] = system_instruction
            if tool_list:
                config_kwargs["tools"] = tool_list
            if thinking_level:
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=thinking_level
                    )
                except Exception:  # noqa: BLE001 - older SDK field names
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_budget=-1
                    )
            if response_schema is not None:
                config_kwargs["response_schema"] = response_schema
                config_kwargs["response_mime_type"] = response_mime_type or "application/json"
            elif response_mime_type:
                config_kwargs["response_mime_type"] = response_mime_type
            if temperature is not None:
                config_kwargs["temperature"] = temperature
            if max_output_tokens is not None:
                config_kwargs["max_output_tokens"] = max_output_tokens

            config = types.GenerateContentConfig(**config_kwargs)
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )

        response = await self._with_retry(build, attempts=4, label=model)

        text_parts: list[str] = []
        thought_parts: list[str] = []
        for candidate in getattr(response, "candidates", []) or []:
            content = getattr(candidate, "content", None)
            for part in getattr(content, "parts", []) or []:
                chunk = getattr(part, "text", None)
                if not chunk:
                    continue
                if getattr(part, "thought", False):
                    thought_parts.append(chunk)
                else:
                    text_parts.append(chunk)

        tool_calls = [
            ToolCall(name=fc.name, args=dict(fc.args or {}))
            for fc in (getattr(response, "function_calls", None) or [])
        ]

        grounding = None
        for candidate in getattr(response, "candidates", []) or []:
            gm = getattr(candidate, "grounding_metadata", None)
            if gm is not None:
                grounding = {
                    "chunks": len(getattr(gm, "grounding_chunks", []) or []),
                    "queries": list(getattr(gm, "web_search_queries", []) or []),
                }
                break

        usage = {
            "prompt_tokens": getattr(getattr(response, "usage_metadata", None), "prompt_token_count", None),
            "candidates_tokens": getattr(getattr(response, "usage_metadata", None), "candidates_token_count", None),
            "grounding": grounding,
        }

        return BackendResult(
            text="\n".join(text_parts).strip(),
            model=model,
            tier=tier,  # type: ignore[arg-type]
            thinking="\n".join(thought_parts).strip() or None,
            tool_calls=tool_calls,
            usage=usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=response,
        )

    # -- tool round-trip -------------------------------------------------
    @staticmethod
    def tool_response_part(name: str, payload: dict[str, Any]) -> Any:
        """Build the `tool_response` part Gemma 4 expects in a tool loop."""
        try:
            return types.Part.from_function_response(name=name, response=payload)
        except Exception:  # noqa: BLE001 - fall back to the raw schema
            return {"function_response": {"name": name, "response": payload}}


def _status_of(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from a SDK exception."""
    text = str(exc)
    for token in text.replace("{", " ").replace(",", " ").split():
        if token.isdigit() and 400 <= int(token) <= 599:
            return int(token)
    return None


async def list_gemma_models() -> list[str]:
    """List the Gemma models this key can actually reach."""
    backend = GeminiBackend()
    try:
        client = await backend.client()
        if client is None:
            return []
        models = await asyncio.to_thread(client.models.list)
        return sorted(m.name for m in models if "gemma" in m.name.lower())
    except Exception:  # noqa: BLE001
        return []
