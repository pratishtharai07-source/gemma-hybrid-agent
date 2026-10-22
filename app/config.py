"""Environment-driven configuration.

Every knob can be set in the environment or in a `.env` file next to the
project root.  Nothing here reaches the network, so it is safe to import
from tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _str(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except (TypeError, ValueError):
        return default


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# The official Gemma 4 guidance says a "LOW" thinking instruction is a
# behaviour-shaping prompt rather than a trained switch, and it trims roughly
# 20% of thinking tokens.  See the Gemma 4 prompt-formatting guide.
LOW_THINKING_SYSTEM_SUFFIX = (
    "Think at a lower depth. Prefer the shortest correct reasoning path and "
    "skip deliberation on questions you can answer immediately."
)


@dataclass(frozen=True)
class Settings:
    # --- cloud tier -----------------------------------------------------
    gemini_api_key: str = _str("GEMINI_API_KEY") or _str("GOOGLE_API_KEY")
    cloud_enabled: bool = _bool("CLOUD_ENABLED", True)
    cloud_model_fast: str = _str("GEMMA_CLOUD_MODEL", "gemma-4-26b-a4b-it")
    cloud_model_strong: str = _str("GEMMA_CLOUD_MODEL_STRONG", "gemma-4-31b-it")
    cloud_timeout: float = _float("CLOUD_TIMEOUT", 120.0)

    # --- local tier -----------------------------------------------------
    local_enabled: bool = _bool("LOCAL_LLM_ENABLED", True)
    local_mode: str = _str("LOCAL_LLM_MODE", "auto").lower()  # auto|llamacpp|ollama
    local_base_url: str = _str("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    local_model: str = _str("LOCAL_LLM_MODEL", "gemma4-e2b")
    local_timeout: float = _float("LOCAL_LLM_TIMEOUT", 120.0)

    # --- router ---------------------------------------------------------
    escalate_threshold: float = _float("ESCALATE_THRESHOLD", 0.55)
    max_tool_iterations: int = _int("MAX_TOOL_ITERATIONS", 5)
    max_output_tokens: int = _int("MAX_OUTPUT_TOKENS", 2048)
    low_thinking: bool = _bool("LOW_THINKING_INSTRUCTION", True)

    @property
    def cloud_ready(self) -> bool:
        return self.cloud_enabled and bool(self.gemini_api_key)

    def public_dict(self) -> dict:
        """Config summary that is safe to expose over HTTP."""
        return {
            "cloud": {
                "enabled": self.cloud_enabled,
                "ready": self.cloud_ready,
                "model_fast": self.cloud_model_fast,
                "model_strong": self.cloud_model_strong,
                "api_key_present": bool(self.gemini_api_key),
            },
            "local": {
                "enabled": self.local_enabled,
                "mode": self.local_mode,
                "base_url": self.local_base_url,
                "model": self.local_model,
            },
            "router": {
                "escalate_threshold": self.escalate_threshold,
                "max_tool_iterations": self.max_tool_iterations,
                "max_output_tokens": self.max_output_tokens,
                "low_thinking": self.low_thinking,
            },
        }


settings = Settings()
