"""Tool registry.

Every tool is a plain function with a JSON-schema signature.  The same
registry is rendered two ways:

* as Gemini ``function_declarations`` so Gemma 4 can call it natively;
* as a local dispatch table used by the executor.

The arithmetic tool exists because we deliberately do **not** trust a language
model with maths.  Python computes the sums; Gemma judges the result.
"""

from __future__ import annotations

import ast
import asyncio
import html
import ipaddress
import operator
import re
import shutil
import subprocess
import sys
import sysconfig
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

import httpx

# ---------------------------------------------------------------------------
# Safe arithmetic
# ---------------------------------------------------------------------------
_BIN_OPS: dict[type[ast.AST], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.AST], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_arithmetic(expression: str) -> float:
    """Evaluate a pure-arithmetic expression without executing arbitrary code."""

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("only numbers are allowed")
            return node.value
        if isinstance(node, ast.BinOp):
            op = _BIN_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported operator")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = _UNARY_OPS.get(type(node.op))
            if op is None:
                raise ValueError("unsupported unary operator")
            return op(_eval(node.operand))
        raise ValueError("unsupported expression")

    tree = ast.parse(expression, mode="eval")
    return float(_eval(tree))


# ---------------------------------------------------------------------------
# Graphify CLI discovery
# ---------------------------------------------------------------------------
def _find_graphify() -> str | None:
    """Locate the `graphify` executable, including the user scripts dir."""
    found = shutil.which("graphify")
    if found:
        return found
    candidates = [
        sysconfig.get_path("scripts"),
        sysconfig.get_path("purelib", "nt_user"),
    ]
    for base in candidates:
        if not base:
            continue
        for name in ("graphify.exe", "graphify"):
            path = f"{base}\\{name}" if sys.platform == "win32" else f"{base}/{name}"
            try:
                import os

                if os.path.isfile(path):
                    return path
            except Exception:  # noqa: BLE001
                continue
    return None


def run_graphify(args: list[str], cwd: str, timeout: float = 90.0) -> str:
    """Run a graphify subcommand and return its stdout."""
    exe = _find_graphify()
    if not exe:
        return (
            "Graphify CLI is not installed. Install it with 'uv tool install graphifyy' "
            "or 'pipx install graphifyy', then run 'graphify extract .' to build the graph."
        )
    try:
        proc = subprocess.run(
            [exe, *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"graphify {' '.join(args)} timed out after {timeout:.0f}s."
    except Exception as exc:  # noqa: BLE001
        return f"graphify failed: {exc}"
    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    if proc.returncode != 0 and not out:
        return f"graphify exited {proc.returncode}: {err[:400]}"
    return out[:6000] if out else err[:400]


# ---------------------------------------------------------------------------
# Indian GST reference (offline)
# ---------------------------------------------------------------------------
GST_SLABS: list[tuple[float, str]] = [
    (0, "fresh fruits, vegetables, milk, eggs, curd, bread, salt, bindi, footwear, books, plain bread rusk"),
    (5, "packaged food, cereals, sugar, tea, coffee, spices, medicines, railway fare, small restaurants that do not claim input credit"),
    (12, "mobile phones, processed food, butter, ghee, namkeen, cotton textiles, hotel rooms up to 1000 INR"),
    (18, "hair oil, soap, toothpaste, IT services, capital goods, restaurant inside a hotel priced 1001-5000 INR, cab and ride services"),
    (28, "cars, cement, air conditioners, refrigerators, tobacco, televisions above 43 inch, five-star hotel rooms"),
]


def gst_reference(item: str) -> str:
    """Look up likely GST slabs for a described item."""
    needle = (item or "").lower()
    hits: list[str] = []
    for rate, examples in GST_SLABS:
        if any(word in needle for word in examples.split(", ")):
            hits.append(f"{rate}% - {examples}")
    if not hits:
        rates = ", ".join(str(r) for r, _ in GST_SLABS)
        return f"No confident match for {item!r}. Common Indian GST slabs are {rates}%."
    return "Likely GST slabs for this item:\n- " + "\n- ".join(hits)


# ---------------------------------------------------------------------------
# URL fetching with SSRF guard
# ---------------------------------------------------------------------------
_TAG_RE = re.compile(r"<[^>]+>")


def _is_public_host(host: str) -> bool:
    try:
        infos = socket_getaddrinfo(host)
    except Exception:  # noqa: BLE001
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False
    return True


def socket_getaddrinfo(host: str) -> list[Any]:
    import socket

    return socket.getaddrinfo(host, None)


async def fetch_url(url: str, max_chars: int = 4000) -> str:
    """Fetch a public URL and return its readable text."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Unsupported URL scheme {parsed.scheme!r}. Use http or https."
    host = parsed.hostname or ""
    if not host:
        return "That URL has no host."
    if not _is_public_host(host):
        return f"Refusing to fetch {host}: only public hosts are allowed."
    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "gemma-hybrid-agent/0.1"})
            resp.raise_for_status()
            text = resp.text
    except Exception as exc:  # noqa: BLE001
        return f"Fetch failed: {exc}"

    stripped = _TAG_RE.sub(" ", text)
    stripped = html.unescape(stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return stripped[:max_chars] or "(empty response)"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Awaitable[Any]]
    open_weight: bool = True


@dataclass
class ToolRegistry:
    tools: list[Tool] = field(default_factory=list)

    def add(self, tool: Tool) -> None:
        self.tools.append(tool)

    def get(self, name: str) -> Tool | None:
        return next((t for t in self.tools if t.name == name), None)

    def declarations(self) -> list[dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.tools
        ]

    def gemini_tools(self):
        """Render as Gemini function declarations."""
        from google.genai import types

        return [types.Tool(function_declarations=self.declarations())]

    async def execute(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tool = self.get(name)
        if tool is None:
            return {"error": f"Unknown tool {name!r}.", "ok": False}
        try:
            result = await tool.handler(**(args or {}))
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}", "ok": False}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{name} failed: {exc}", "ok": False}
        if isinstance(result, dict):
            payload = dict(result)
            payload.setdefault("ok", True)
            return payload
        return {"result": result, "ok": True}


def build_registry(repo_root: str) -> ToolRegistry:
    """Create the default tool registry."""

    async def _calculate(expression: str) -> dict[str, Any]:
        value = safe_arithmetic(expression)
        return {"expression": expression, "value": value, "ok": True}

    async def _gst_slab(item_description: str) -> dict[str, Any]:
        return {"item": item_description, "reference": gst_reference(item_description), "ok": True}

    async def _current_datetime() -> dict[str, Any]:
        now = datetime.now(timezone.utc).astimezone()
        return {
            "iso": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.strftime("%H:%M:%S"),
            "timezone": now.tzname(),
            "ok": True,
        }

    async def _fetch_url(url: str) -> dict[str, Any]:
        body = await fetch_url(url)
        return {"url": url, "content": body, "ok": not body.startswith(("Refusing", "Fetch failed", "Unsupported"))}

    async def _graphify_query(question: str) -> dict[str, Any]:
        out = await asyncio.to_thread(run_graphify, ["query", question], repo_root)
        return {"question": question, "output": out, "ok": True}

    async def _graphify_explain(symbol: str) -> dict[str, Any]:
        out = await asyncio.to_thread(run_graphify, ["explain", symbol], repo_root)
        return {"symbol": symbol, "output": out, "ok": True}

    async def _graphify_path(source: str, target: str) -> dict[str, Any]:
        out = await asyncio.to_thread(run_graphify, ["path", source, target], repo_root)
        return {"source": source, "target": target, "output": out, "ok": True}

    registry = ToolRegistry()
    registry.add(
        Tool(
            name="calculate",
            description=(
                "Evaluate an arithmetic expression and return the exact numeric result. "
                "Use this for every sum, difference, percentage or total. Never compute "
                "these in your head - always call this tool."
            ),
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. (160 + 50) * 1.18"}},
                "required": ["expression"],
            },
            handler=_calculate,
        )
    )
    registry.add(
        Tool(
            name="gst_slab_reference",
            description=(
                "Look up the Indian GST slab that normally applies to a described item "
                "(for example 'restaurant meal', 'mobile phone', 'cotton shirt'). "
                "Use this to check whether the tax rate printed on a bill is plausible."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "item_description": {
                        "type": "string",
                        "description": "What was purchased, in plain words.",
                    }
                },
                "required": ["item_description"],
            },
            handler=_gst_slab,
        )
    )
    registry.add(
        Tool(
            name="current_datetime",
            description="Get the current date and time, including the local timezone.",
            parameters={"type": "object", "properties": {}},
            handler=_current_datetime,
        )
    )
    registry.add(
        Tool(
            name="fetch_url",
            description=(
                "Fetch a public http/https URL and return its readable text content. "
                "Use it when you need a live page, for example a current rate or "
                "an official notification."
            ),
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
                "required": ["url"],
            },
            handler=_fetch_url,
        )
    )
    registry.add(
        Tool(
            name="graphify_query",
            description=(
                "Query the knowledge graph of this project's own source code, built with "
                "Graphify. Ask about how a component works, what calls what, or where a "
                "behaviour is implemented. Returns evidence from the graph."
            ),
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string", "description": "A question about the codebase."}},
                "required": ["question"],
            },
            handler=_graphify_query,
        )
    )
    registry.add(
        Tool(
            name="graphify_explain",
            description="Explain a single code symbol from this project's knowledge graph.",
            parameters={
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "Symbol name to explain."}},
                "required": ["symbol"],
            },
            handler=_graphify_explain,
        )
    )
    registry.add(
        Tool(
            name="graphify_path",
            description="Trace the shortest relationship path between two code symbols in this project.",
            parameters={
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "target"],
            },
            handler=_graphify_path,
        )
    )
    return registry
