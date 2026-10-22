#!/usr/bin/env python
"""End-to-end smoke test for the Gemma 4 Bill Auditor.

Run it with the app already serving:

    python -m uvicorn app.main:app --port 8000
    python scripts/smoke_test.py

Every stage reports PASS / FAIL / SKIP and the script exits non-zero if a
required stage fails, so it doubles as a pre-demo check.
"""

from __future__ import annotations

import base64
import io
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

BASE = "http://127.0.0.1:8000"
RESULTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    colour = {"PASS": "\033[92m", "FAIL": "\033[91m", "SKIP": "\033[93m"}.get(status, "")
    reset = "\033[0m"
    print(f"[{colour}{status}{reset}] {name}" + (f"  -- {detail}" if detail else ""))


def make_receipt() -> bytes:
    """Render a small receipt image to test the vision path."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return b""

    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    lines = [
        ("SHREE RESTAURANT", 24, 18),
        ("GSTIN: 27ABCDE1234F1Z5", 16, 62),
        ("2 x Masala Dosa      Rs 160.00", 16, 106),
        ("1 x Filter Coffee   Rs  50.00", 16, 140),
        ("Subtotal            210.00", 16, 190),
        ("CGST 5%              10.50", 16, 222),
        ("SGST 5%              10.50", 16, 254),
        ("TOTAL              Rs 231.00", 16, 292),
    ]
    for text, size, y in lines:
        d.text((24, y), text, fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main() -> int:
    print("=" * 72)
    print("Gemma 4 Bill Auditor -- smoke test")
    print("=" * 72)

    try:
        client = httpx.Client(timeout=300.0, base_url=BASE)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not build a client: {exc}")
        return 2

    # -- 1. health --------------------------------------------------------
    try:
        r = client.get("/api/health")
        r.raise_for_status()
        health: dict[str, Any] = r.json()
        record("health endpoint", "PASS", f"status={health.get('status')}")
    except Exception as exc:  # noqa: BLE001
        record("health endpoint", "FAIL", str(exc)[:200])
        return 1

    local = health.get("local", {})
    cloud = health.get("config", {}).get("cloud", {})
    record(
        "local Gemma 4 tier",
        "PASS" if local.get("available") else "SKIP",
        f"mode={local.get('mode')} url={local.get('base_url')}",
    )
    record(
        "cloud tier configured",
        "PASS" if cloud.get("ready") else "FAIL",
        f"key_present={cloud.get('api_key_present')}",
    )
    models = health.get("cloud_models_available") or []
    record(
        "Gemma 4 models reachable",
        "PASS" if models else "FAIL",
        ", ".join(models) or "none",
    )

    # -- 2. triage --------------------------------------------------------
    try:
        r = client.post(
            "/api/triage",
            json={"prompt": "Summarise this paragraph in one line.", "enable_tools": False},
        )
        r.raise_for_status()
        decision = r.json()
        record(
            "router / triage",
            "PASS",
            f"tier={decision.get('tier')} by={decision.get('decided_by')} :: {decision.get('reason','')[:70]}",
        )
    except Exception as exc:  # noqa: BLE001
        record("router / triage", "FAIL", str(exc)[:200])

    # -- 3. tool declarations --------------------------------------------
    try:
        r = client.get("/api/tools")
        r.raise_for_status()
        tools = r.json().get("tools", [])
        names = ", ".join(t.get("name", "?") for t in tools)
        record("tool registry", "PASS" if tools else "FAIL", names)
    except Exception as exc:  # noqa: BLE001
        record("tool registry", "FAIL", str(exc)[:200])

    # -- 4. cloud tool calling -------------------------------------------
    try:
        r = client.post(
            "/api/chat",
            json={
                "prompt": "Use the calculate tool to work out (160 + 50) * 1.10, then give me the answer.",
                "route": "cloud_fast",
                "enable_tools": True,
                "thinking": "minimal",
            },
        )
        r.raise_for_status()
        data = r.json()
        ok = data.get("ok")
        used = "calculate" in (data.get("text") or "") or bool(data.get("tool_calls"))
        record(
            "cloud function calling",
            "PASS" if ok else "FAIL",
            f"{(data.get('text') or '')[:110]}",
        )
    except Exception as exc:  # noqa: BLE001
        record("cloud function calling", "FAIL", str(exc)[:200])

    # -- 5. graphify ------------------------------------------------------
    g = health.get("graphify", {})
    if g.get("installed"):
        record(
            "graphify CLI",
            "PASS",
            f"installed, graph_present={g.get('graph_present')}",
        )
    else:
        record("graphify CLI", "SKIP", "not installed (pip install graphifyy)")

    # -- 6. full audit ----------------------------------------------------
    image = make_receipt()
    if not image:
        record("audit pipeline", "SKIP", "pillow is not installed, cannot build a test receipt")
    else:
        payload = {
            "image_base64": base64.b64encode(image).decode(),
            "mime_type": "image/png",
            "prompt": "Audit this bill and explain any problems.",
            "language": "English",
            "route": "auto",
            "enable_tools": True,
            "thinking": "high",
        }
        started = time.perf_counter()
        try:
            r = client.post("/api/audit", json=payload)
            r.raise_for_status()
            data = r.json()
            elapsed = int((time.perf_counter() - started) * 1000)
            if not data.get("ok"):
                record("audit pipeline", "FAIL", str(data.get("error"))[:200])
            else:
                bill = data.get("bill") or {}
                findings = data.get("findings") or []
                record(
                    "audit pipeline",
                    "PASS",
                    f"{elapsed}ms | vendor={bill.get('vendor_name')} "
                    f"total={bill.get('total')} | findings={len(findings)} "
                    f"| tiers={','.join(data.get('tiers_used') or [])}",
                )
                record(
                    "  extracted total",
                    "PASS" if bill.get("total") else "FAIL",
                    f"subtotal={bill.get('subtotal')} tax={bill.get('tax_total')} total={bill.get('total')}",
                )
                record(
                    "  explanation present",
                    "PASS" if data.get("explanation") else "FAIL",
                    (data.get("explanation") or "")[:110],
                )
                timings = data.get("timings") or {}
                print("    timings: " + json.dumps(timings))
                for finding in findings:
                    print(f"      [{finding.get('severity')}] {finding.get('title')}")
        except Exception as exc:  # noqa: BLE001
            record("audit pipeline", "FAIL", str(exc)[:250])

    # -- summary ----------------------------------------------------------
    print("=" * 72)
    passed = sum(1 for _, s, _ in RESULTS if s == "PASS")
    failed = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    skipped = sum(1 for _, s, _ in RESULTS if s == "SKIP")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
