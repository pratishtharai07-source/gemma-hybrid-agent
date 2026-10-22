"""The hybrid router.

This is where the open-weight local model earns its place in the system.

A 2.3B-parameter model cannot read a receipt, cannot browse the web and
struggles with long multi-step reasoning.  A 31B cloud model can do all of
those things but costs money, adds latency and sends data off the device.

The router asks the local Gemma 4 E2B tier to look at a request and decide
where it should go, then layers a small set of **deterministic overrides** on
top so that the routing can never silently break because a small model
misjudged something:

===========================  ==========================================
Override                     Effect
===========================  ==========================================
an image is attached          ``needs_vision = True`` -> cloud
"today/current/latest/..."   ``needs_live_data = True`` -> cloud
local tier unavailable        fall back to ``cloud_fast``
user picked a tier            user wins, no triage at all
===========================  ==========================================

The result is a routing decision that is *explainable* — every decision
carries the reason it was made, which is what makes the demo legible.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from .backends.base import BackendResult, extract_json
from .backends.local import LocalBackend
from .config import settings
from .prompts import TRIAGE_PROMPT, TRIAGE_SYSTEM
from .schemas import RouteDecision

# Words that almost always mean "the answer changes over time".
_LIVE_PATTERNS = (
    r"\btoday\b",
    r"\bcurrent(?:ly)?\b",
    r"\blatest\b",
    r"\bright now\b",
    r"\bthis week\b",
    r"\bthis month\b",
    r"\brecent(?:ly)?\b",
    r"\bnews\b",
    r"\bprice\b",
    r"\brate\b",
    r"\bscore\b",
    r"\bweather\b",
    r"\bexchange rate\b",
    r"\bgst (?:rate|slab)s?\b",
)

_TOOL_PATTERNS = (
    r"\bcalculate\b",
    r"\bcompute\b",
    r"\bsum\b",
    r"\btotal\b",
    r"\bconvert\b",
    r"\bfetch\b",
    r"\blook up\b",
    r"\blookup\b",
    r"\bsearch\b",
    r"\bgraph\b",
    r"\bwhere is\b",
    r"\bwhich file\b",
)


def _matches(text: str, patterns: tuple[str, ...]) -> bool:
    lowered = (text or "").lower()
    return any(re.search(p, lowered) for p in patterns)


class Router:
    """Decides which Gemma 4 tier should answer a request."""

    def __init__(self, local: LocalBackend) -> None:
        self.local = local
        self.last_triage: dict[str, Any] | None = None
        self.last_triage_ms: int = 0

    # ------------------------------------------------------------------
    async def decide(
        self,
        *,
        prompt: str,
        has_image: bool = False,
        enable_tools: bool = False,
        forced: str = "auto",
        use_local: bool = True,
        extra_context: str = "",
    ) -> RouteDecision:
        """Produce a routing decision for a request."""
        started = time.perf_counter()

        local_ready = use_local and await self.local.detect_mode() is not None

        # 1. Deterministic overrides ----------------------------------
        needs_vision = has_image
        needs_live_data = _matches(prompt, _LIVE_PATTERNS)
        needs_tools = enable_tools and _matches(prompt, _TOOL_PATTERNS)

        # 2. Local triage ----------------------------------------------
        triage: dict[str, Any] | None = None
        if local_ready and forced == "auto":
            triage = await self._triage(prompt, extra_context)
            if triage:
                needs_live_data = needs_live_data or bool(triage.get("needs_live_data"))
                needs_tools = needs_tools or bool(triage.get("needs_tools"))

        # 3. Merge into a verdict --------------------------------------
        complexity = float(triage.get("complexity", 0.5)) if triage else 0.5
        sensitivity = str(triage.get("sensitivity", "unknown")) if triage else "unknown"
        confidence = float(triage.get("confidence", 0.5)) if triage else 0.5
        reason = str(triage.get("reason", "")) if triage else ""

        if forced != "auto":
            tier = forced  # type: ignore[assignment]
            return RouteDecision(
                tier=tier,
                reason=f"User selected the {tier} tier.",
                confidence=1.0,
                decided_by="user_override",
                complexity=complexity,
                sensitivity=sensitivity,
                needs_vision=needs_vision,
                needs_tools=needs_tools,
                needs_live_data=needs_live_data,
                triage=triage,
            )

        if needs_vision:
            tier = "cloud_fast"
            reason = reason or "An image was attached, so the request needs a multimodal model."
            decided_by = "rules" if not triage else "local_triage"
        elif needs_live_data:
            tier = "cloud_fast"
            reason = reason or "The request needs live information that only the cloud tier can fetch."
            decided_by = "rules" if not triage else "local_triage"
        elif needs_tools:
            tier = "cloud_fast"
            reason = reason or "The request needs tool calls, which the cloud tier supports."
            decided_by = "rules" if not triage else "local_triage"
        elif complexity >= settings.escalate_threshold:
            tier = "cloud_strong"
            reason = reason or (
                f"Local triage scored this {complexity:.2f} complexity, "
                f"above the {settings.escalate_threshold:.2f} escalation threshold."
            )
            decided_by = "local_triage"
        elif local_ready:
            tier = "local"
            reason = reason or "Simple enough for the on-device model, and it keeps the data local."
            decided_by = "local_triage" if triage else "rules"
        else:
            tier = "cloud_fast"
            reason = "No local runtime was detected, so the request goes to the cloud tier."
            decided_by = "rules"

        # The local model's own recommendation is a hint, not gospel: we only
        # override it for the deterministic rules above.
        self.last_triage = triage
        self.last_triage_ms = int((time.perf_counter() - started) * 1000)

        return RouteDecision(
            tier=tier,
            reason=reason,
            confidence=confidence,
            decided_by=decided_by,  # type: ignore[arg-type]
            complexity=complexity,
            sensitivity=sensitivity,
            needs_vision=needs_vision,
            needs_tools=needs_tools,
            needs_live_data=needs_live_data,
            triage=triage,
        )

    # ------------------------------------------------------------------
    async def _triage(self, prompt: str, extra_context: str) -> dict[str, Any] | None:
        """Ask the on-device model to classify the request."""
        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM},
            {
                "role": "user",
                "content": TRIAGE_PROMPT.format(
                    prompt=prompt[:1200],
                    context=extra_context[:800] or "(no extra context)",
                ),
            },
        ]
        try:
            result: BackendResult = await self.local.chat(
                messages=messages, temperature=0.0, max_tokens=512, json_mode=True
            )
        except Exception:  # noqa: BLE001 - a failed triage must never break the request
            return None

        parsed = extract_json(result.text)
        if not parsed:
            # Some local builds ignore the JSON mode; try to salvage it.
            parsed = _salvage_triage(result.text)
        if not parsed:
            return None

        # Normalise and clamp.
        try:
            parsed["complexity"] = max(0.0, min(1.0, float(parsed.get("complexity", 0.5))))
        except (TypeError, ValueError):
            parsed["complexity"] = 0.5
        try:
            parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        except (TypeError, ValueError):
            parsed["confidence"] = 0.5
        parsed["needs_live_data"] = bool(parsed.get("needs_live_data"))
        parsed["needs_tools"] = bool(parsed.get("needs_tools"))
        parsed["route"] = parsed.get("route") or "local"
        parsed["sensitivity"] = parsed.get("sensitivity") or "unknown"
        parsed["reason"] = str(parsed.get("reason", ""))[:400]
        return parsed


def _salvage_triage(text: str) -> dict[str, Any] | None:
    """Last-ditch parser for models that answer with a labelled blob."""
    if not text:
        return None
    match = re.search(
        r"task_type\s*[:=]\s*(\w+).*?complexity\s*[:=]\s*([0-9.]+)",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    return {
        "task_type": match.group(1),
        "complexity": float(match.group(2)),
        "sensitivity": "unknown",
        "needs_live_data": bool(re.search(r"needs_live_data\s*[:=]\s*true", text, re.I)),
        "needs_tools": bool(re.search(r"needs_tools\s*[:=]\s*true", text, re.I)),
        "route": "local",
        "reason": text.strip()[:300],
        "confidence": 0.4,
    }


def dumps(value: Any) -> str:
    """Compact JSON helper used when embedding structures in prompts."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
