"""Local tier: open-weight Gemma 4 running on this machine.

Two runtimes are supported and auto-detected:

* **llama.cpp server** – ``llama-server.exe -m gemma-4-E2B-it-Q3_K_M.gguf``.
  This is the one that fits a 4 GB laptop: the Q3_K_M build of Gemma 4 E2B is
  ~2.4 GB, leaving room for the KV cache and the OS.
* **Ollama** – ``ollama run gemma4:e2b-it-qat``.  Convenient, but the published
  tags are 4.3 GB and up, which does not fit in 3.8 GB of RAM.

The local tier is what makes this project *open-weight AI* rather than a thin
wrapper around a hosted endpoint: routing, verification and redaction all run
on-device, with no network access.
"""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import httpx

from ..config import settings
from .base import BackendResult, ToolCall


class LocalBackend:
    """Adapter for a local llama.cpp server or an Ollama daemon."""

    def __init__(self) -> None:
        self._mode: str | None = None
        self._checked_at: float = 0.0
        self._has_checked: bool = False

    # -- discovery -------------------------------------------------------
    async def detect_mode(self, force: bool = False) -> str | None:
        """Return ``"llamacpp"``, ``"ollama"`` or None, caching the result."""
        if not settings.local_enabled:
            return None
        now = time.monotonic()
        if not force and self._has_checked and (now - self._checked_at < 30):
            return self._mode

        base = settings.local_base_url
        fast_timeout = httpx.Timeout(connect=0.5, read=0.8, write=0.8, pool=0.8)
        async with httpx.AsyncClient(timeout=fast_timeout) as client:
            if settings.local_mode == "ollama":
                order = ["ollama"]
            elif settings.local_mode == "llamacpp":
                order = ["llamacpp"]
            else:
                order = ["llamacpp", "ollama"]

            for mode in order:
                try:
                    if mode == "llamacpp":
                        resp = await client.get(f"{base}/v1/models")
                    else:
                        resp = await client.get(f"{base}/api/tags")
                    if resp.status_code == 200:
                        self._mode = mode
                        self._checked_at = now
                        self._has_checked = True
                        return mode
                except Exception:  # noqa: BLE001 - server simply not running
                    continue

        self._mode = None
        self._checked_at = now
        self._has_checked = True
        return None

    async def health(self) -> dict[str, Any]:
        mode = await self.detect_mode(force=True)
        info: dict[str, Any] = {
            "available": mode is not None,
            "mode": mode,
            "base_url": settings.local_base_url,
            "model": settings.local_model,
        }
        if mode is None:
            return info
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                if mode == "llamacpp":
                    resp = await client.get(f"{settings.local_base_url}/v1/models")
                    info["served_models"] = [m.get("id") for m in resp.json().get("data", [])]
                else:
                    resp = await client.get(f"{settings.local_base_url}/api/tags")
                    info["served_models"] = [
                        m.get("name") for m in resp.json().get("models", [])
                    ]
        except Exception:  # noqa: BLE001
            pass
        return info

    # -- chat ------------------------------------------------------------
    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        temperature: float = 0.1,
        max_tokens: int = 1024,
        json_mode: bool = False,
    ) -> BackendResult:
        """Run a chat completion on the local Gemma 4 instance."""
        mode = await self.detect_mode()
        if mode is None:
            raise RuntimeError(
                "No local Gemma 4 runtime detected. Start llama-server on "
                f"{settings.local_base_url} or run 'ollama serve'."
            )

        started = time.perf_counter()
        async with httpx.AsyncClient(timeout=settings.local_timeout) as client:
            if mode == "llamacpp":
                payload: dict[str, Any] = {
                    "model": settings.local_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
                if json_mode:
                    payload["response_format"] = {"type": "json_object"}
                resp = await client.post(
                    f"{settings.local_base_url}/v1/chat/completions", json=payload
                )
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                text = message.get("content") or ""
                calls = [
                    ToolCall(name=tc["function"]["name"], args=tc["function"].get("arguments") or {})
                    for tc in (message.get("tool_calls") or [])
                ]
                usage = data.get("usage") or {}
            else:
                payload = {
                    "model": settings.local_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                }
                if json_mode:
                    payload["format"] = "json"
                resp = await client.post(f"{settings.local_base_url}/api/chat", json=payload)
                resp.raise_for_status()
                data = resp.json()
                message = data.get("message") or {}
                text = message.get("content") or ""
                calls = [
                    ToolCall(name=tc.get("function", {}).get("name", ""), args=tc.get("function", {}).get("arguments") or {})
                    for tc in (message.get("tool_calls") or [])
                ]
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "candidates_tokens": data.get("eval_count"),
                }

        return BackendResult(
            text=text.strip(),
            model=settings.local_model,
            tier="local",
            tool_calls=calls,
            usage=usage,
            latency_ms=int((time.perf_counter() - started) * 1000),
            raw=data,
        )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def image_message(text: str, image_bytes: bytes, mime_type: str = "image/png") -> dict[str, Any]:
        """Build a multimodal message in OpenAI-compatible content-part form."""
        encoded = base64.b64encode(image_bytes).decode()
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
            ],
        }
