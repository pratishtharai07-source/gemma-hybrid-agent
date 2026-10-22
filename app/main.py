"""Bill/Receipt Auditor — a hybrid Gemma 4 pipeline.

    photo of a bill
        |
        v
    [0] ROUTER    local Gemma 4 E2B decides where the work should go
        |
        v
    [1] EXTRACT   cloud Gemma 4 reads the image  -> structured fields
        |
        v
    [2] VERIFY    exact arithmetic in Python + on-device Gemma 4 audit
        |
        v
    [3] REDACT    identifiers stripped before anything else happens
        |
        v
    [4] ADVISE    cloud Gemma 4 with thinking + native function calls
                  -> verdict, findings, ledger entries, plain-language note

Design rule that runs through the whole file: **Python does the maths, Gemma
does the semantics.**  A language model is a poor calculator and an
excellent auditor, so we never let one do the other's job.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from google.genai import types

from .backends.base import BackendResult, extract_json
from .backends.gemini import GeminiBackend, list_gemma_models
from .backends.local import LocalBackend
from .config import settings
from .prompts import (
    ADVISOR_PROMPT,
    EXTRACT_PROMPT,
    EXTRACT_SYSTEM,
    LOCAL_AUDIT_PROMPT,
    LOCAL_AUDIT_SYSTEM,
    build_advisor_system,
)
from .router import Router, dumps
from .schemas import (
    AccountingEntry,
    AuditFinding,
    AuditRequest,
    AuditResponse,
    BillFields,
    ChatRequest,
    ChatResponse,
    RouteDecision,
)
from .tools import build_registry

REPO_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Gemma 4 Bill Auditor",
    version="0.1.0",
    description=(
        "Hybrid local + cloud Gemma 4 pipeline that audits Indian bills and "
        "receipts: on-device routing, cloud vision extraction, exact Python "
        "arithmetic, on-device audit and redaction, and cloud reasoning with "
        "native function calling."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

gemini = GeminiBackend()
local = LocalBackend()
router = Router(local)
registry = build_registry(str(REPO_ROOT))


# ===========================================================================
# Deterministic helpers — the parts we refuse to delegate to a model
# ===========================================================================
GSTIN_RE = re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[-\s]?)?[6-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
ACCOUNT_RE = re.compile(r"\b\d{11,18}\b")


def _f(value: Any) -> float | None:
    """Best-effort float coercion from messy model output."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    if cleaned in ("", "-", "."):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def python_audit(bill: dict[str, Any]) -> tuple[list[AuditFinding], dict[str, Any]]:
    """Recompute every number on the bill with exact float arithmetic."""
    findings: list[AuditFinding] = []
    items = bill.get("line_items") or []

    computed_subtotal = 0.0
    computed_tax = 0.0
    counted = 0
    line_mismatches = 0

    for item in items:
        qty = _f(item.get("quantity"))
        unit = _f(item.get("unit_price"))
        amount = _f(item.get("amount"))
        rate = _f(item.get("tax_rate"))
        label = str(item.get("description") or "item")[:60]

        expected: float | None = None
        if qty is not None and unit is not None:
            expected = qty * unit
            if amount is not None and abs(expected - amount) > 0.01:
                line_mismatches += 1
                findings.append(
                    AuditFinding(
                        severity="warning",
                        title=f"Line total does not match quantity x price: {label}",
                        detail=(
                            f"{qty:g} x {unit:g} = {expected:.2f}, but the bill prints "
                            f"{amount:.2f}. Difference {expected - amount:+.2f}."
                        ),
                        source="python",
                    )
                )
            computed_subtotal += expected
            counted += 1
        elif amount is not None:
            computed_subtotal += amount
            counted += 1

        if rate is not None and amount is not None:
            computed_tax += amount * rate / 100.0

    printed_subtotal = _f(bill.get("subtotal"))
    printed_tax = _f(bill.get("tax_total"))
    printed_total = _f(bill.get("total"))

    if counted and printed_subtotal is not None and abs(computed_subtotal - printed_subtotal) > 0.01:
        findings.append(
            AuditFinding(
                severity="critical",
                title="Subtotal does not match the sum of the line items",
                detail=(
                    f"Line items add up to {computed_subtotal:.2f} but the printed "
                    f"subtotal is {printed_subtotal:.2f} (difference "
                    f"{computed_subtotal - printed_subtotal:+.2f})."
                ),
                source="python",
            )
        )

    if counted and printed_tax is not None and abs(computed_tax - printed_tax) > 0.01:
        findings.append(
            AuditFinding(
                severity="warning",
                title="Tax charged does not match the rates on the items",
                detail=(
                    f"Applying each line's tax rate gives {computed_tax:.2f}, but the "
                    f"bill prints {printed_tax:.2f}."
                ),
                source="python",
            )
        )

    if printed_subtotal is not None and printed_tax is not None and printed_total is not None:
        expected_total = printed_subtotal + printed_tax
        if abs(expected_total - printed_total) > 0.01:
            findings.append(
                AuditFinding(
                    severity="critical",
                    title="Subtotal plus tax does not equal the total",
                    detail=(
                        f"{printed_subtotal:.2f} + {printed_tax:.2f} = {expected_total:.2f}, "
                        f"but the bill prints {printed_total:.2f}."
                    ),
                    source="python",
                )
            )

    if not printed_total and not printed_subtotal:
        findings.append(
            AuditFinding(
                severity="warning",
                title="No total could be read from the document",
                detail="The image did not yield a readable grand total.",
                source="python",
            )
        )

    report = {
        "computed_subtotal": round(computed_subtotal, 2),
        "computed_tax": round(computed_tax, 2),
        "computed_total": round(computed_subtotal + computed_tax, 2),
        "printed_subtotal": printed_subtotal,
        "printed_tax": printed_tax,
        "printed_total": printed_total,
        "line_items_counted": counted,
        "line_mismatches": line_mismatches,
        "arithmetic_ok": not any(f.severity == "critical" for f in findings),
    }
    return findings, report


def scrub_pii(text: str) -> str:
    """Mask identifiers with regular expressions (deterministic, always runs)."""
    text = GSTIN_RE.sub(lambda m: m.group(0)[:2] + "X" * 13, text)
    text = PHONE_RE.sub(lambda m: "XXXXXXXXXX"[: 10 - min(len(m.group(0)), 3)] + "XX", text)
    text = EMAIL_RE.sub("redacted@email.invalid", text)
    return text


def redact_bill(bill: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Strip identifiers from the extracted fields."""
    redacted = dict(bill)
    removed: list[str] = []

    gstin = redacted.get("vendor_gstin")
    if gstin:
        text = str(gstin)
        redacted["vendor_gstin"] = text[:2] + "X" * max(0, len(text) - 4) + text[-2:]
        removed.append("vendor_gstin")

    phone = redacted.get("vendor_phone")
    if phone:
        redacted["vendor_phone"] = "XXXXXXXXXX"
        removed.append("vendor_phone")

    items = []
    for item in redacted.get("line_items") or []:
        item = dict(item)
        if item.get("description"):
            item["description"] = scrub_pii(str(item["description"]))
        items.append(item)
    redacted["line_items"] = items

    return redacted, removed


# ===========================================================================
# Model helpers
# ===========================================================================
def _split_image(data_url: str) -> tuple[bytes, str]:
    """Accept a bare base64 string or a full data: URL."""
    mime = "image/png"
    payload = data_url
    if data_url.startswith("data:"):
        head, _, payload = data_url.partition(",")
        if ";" in head and ":" in head:
            mime = head.split(":", 1)[1].split(";", 1)[0] or mime
    payload = payload.strip()
    padding = (-len(payload)) % 4
    try:
        return base64.b64decode(payload + "=" * padding), mime
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}") from exc


def generate_fallback_accounting_entries(bill: dict[str, Any]) -> list[AccountingEntry]:
    """Generate balanced double-entry accounting entries based on verified bill totals."""
    total = _f(bill.get("total"))
    if not total or total <= 0:
        return []

    subtotal = _f(bill.get("subtotal")) or total
    tax = _f(bill.get("tax_total")) or 0.0
    vendor = str(bill.get("vendor_name") or "Vendor").strip()

    account = "Office & General Expenses"
    v_lower = vendor.lower()
    if any(k in v_lower for k in ("restaurant", "cafe", "bistro", "food", "hotel", "dosa", "coffee")):
        account = "Staff Welfare & Refreshment Expenses"
    elif any(k in v_lower for k in ("tech", "computer", "hardware", "software", "solution")):
        account = "Computer & IT Equipment"
    elif any(k in v_lower for k in ("supermarket", "store", "grocery", "mart")):
        account = "Pantry & Office Consumables"
    elif any(k in v_lower for k in ("cab", "taxi", "travel", "transport", "airport")):
        account = "Travel & Conveyance Expenses"

    entries: list[AccountingEntry] = []
    if tax > 0 and abs((subtotal + tax) - total) < 0.1:
        half_tax = round(tax / 2.0, 2)
        other_half = round(tax - half_tax, 2)
        entries.append(AccountingEntry(account=account, debit=subtotal, credit=0.0, narration=f"Purchase from {vendor}"))
        entries.append(AccountingEntry(account="GST Input (CGST)", debit=half_tax, credit=0.0, narration="CGST input credit claim"))
        entries.append(AccountingEntry(account="GST Input (SGST)", debit=other_half, credit=0.0, narration="SGST input credit claim"))
    else:
        entries.append(AccountingEntry(account=account, debit=total, credit=0.0, narration=f"Bill expense from {vendor}"))

    entries.append(AccountingEntry(account="Cash / Bank Account", debit=0.0, credit=total, narration=f"Payment for {vendor} bill"))
    return entries


def extract_accounting_entries_from_text(text: str) -> list[AccountingEntry]:
    """Extract structured accounting entries from JSON or markdown tables/bullets."""
    entries: list[AccountingEntry] = []
    if not text:
        return entries

    # 1. Try JSON structure first
    parsed = extract_json(text)
    if parsed and isinstance(parsed.get("entries"), list):
        for item in parsed["entries"]:
            try:
                acc = str(item.get("account", "Unspecified")).strip()[:120]
                debit = float(_f(item.get("debit")) or 0.0)
                credit = float(_f(item.get("credit")) or 0.0)
                narration = str(item.get("narration", "")).strip()[:300]
                if acc and (debit > 0 or credit > 0 or narration):
                    entries.append(
                        AccountingEntry(
                            account=acc,
                            debit=debit,
                            credit=credit,
                            narration=narration,
                        )
                    )
            except Exception:
                continue
        if entries:
            return entries

    # 2. Parse Markdown table rows: | Account | Debit | Credit | Narration |
    lines = text.splitlines()
    in_entries = False
    for line in lines:
        lower = line.lower()
        if any(h in lower for h in ("accounting entr", "journal entr", "ledger entr")):
            in_entries = True
            continue
        if in_entries and line.startswith("#"):
            in_entries = False

        if "|" in line:
            parts = [p.strip() for p in line.split("|")]
            # Filter empty boundary parts if line starts/ends with |
            if line.strip().startswith("|"):
                parts = parts[1:]
            if line.strip().endswith("|"):
                parts = parts[:-1]
            if len(parts) >= 3:
                col0 = parts[0].lower()
                if any(k in col0 for k in ("account", "ledger", "---", ":---")):
                    continue
                acc = re.sub(r"[*_`]", "", parts[0]).strip()
                debit_val = _f(parts[1]) or 0.0
                credit_val = _f(parts[2]) or 0.0
                narration = ""
                if len(parts) >= 4:
                    narration = re.sub(r"[*_`]", "", parts[3]).strip()
                if acc and (debit_val > 0 or credit_val > 0):
                    entries.append(
                        AccountingEntry(
                            account=acc,
                            debit=debit_val,
                            credit=credit_val,
                            narration=narration,
                        )
                    )

    # 3. Parse bullet points if table had no rows
    if not entries:
        bullet_re = re.compile(
            r"[-*•]\s+(?:Debit|Dr|Credit|Cr)[:\s]+([^–\-:]+?)[:–\-]\s*(?:Rs\.?|₹)?\s*([\d,.]+)",
            re.IGNORECASE,
        )
        for line in lines:
            m = bullet_re.search(line)
            if m:
                is_debit = "debit" in line.lower() or "dr" in line.lower()
                acc = m.group(1).strip()
                val = _f(m.group(2)) or 0.0
                if acc and val > 0:
                    entries.append(
                        AccountingEntry(
                            account=acc,
                            debit=val if is_debit else 0.0,
                            credit=0.0 if is_debit else val,
                            narration="Posted from bill advice",
                        )
                    )

    return entries


def extract_findings_from_text(text: str) -> list[AuditFinding]:
    """Extract findings from JSON or markdown bullet points."""
    findings: list[AuditFinding] = []
    if not text:
        return findings

    # 1. Try JSON structure
    parsed = extract_json(text)
    if parsed and isinstance(parsed.get("findings"), list):
        for item in parsed["findings"]:
            if isinstance(item, dict):
                sev = str(item.get("severity", "info")).lower()
                if sev not in ("critical", "warning", "info"):
                    sev = "info"
                findings.append(
                    AuditFinding(
                        severity=sev,
                        title=str(item.get("title", "Observation"))[:160],
                        detail=str(item.get("detail", ""))[:800],
                        source="cloud_gemma",
                    )
                )
        if findings:
            return findings

    # 2. Parse Markdown bullets: * **Warning: Tax Mismatch** – Details...
    pattern = re.compile(
        r"[-*•]\s+\*{0,2}(critical|warning|info)[:\s*]+([^*–\-:\n]+)\*{0,2}\s*[–\-:]\s*(.+)",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            sev = m.group(1).lower()
            title = m.group(2).strip()
            detail = m.group(3).strip()
            findings.append(
                AuditFinding(
                    severity=sev,
                    title=title[:160],
                    detail=detail[:800],
                    source="cloud_gemma",
                )
            )

    return findings


async def _structured(
    *,
    model: str,
    contents: Any,
    system_instruction: str,
    schema: Any,
    tier: str,
    thinking_level: str | None = None,
    temperature: float = 0.1,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """Call Gemma 4 with a response schema, degrading to prompted JSON.

    The Gemini API occasionally rejects ``response_schema`` under load, so we
    always have a fallback that asks for the same JSON in prose and parses
    it ourselves, plus automatic failover between Gemma 4 models.
    """
    calls: list[dict[str, Any]] = []
    models_to_try = [model]
    alt_model = (
        settings.cloud_model_strong
        if model != settings.cloud_model_strong
        else settings.cloud_model_fast
    )
    if alt_model and alt_model not in models_to_try:
        models_to_try.append(alt_model)

    for m in models_to_try:
        # First attempt: response_schema
        try:
            result = await gemini.generate(
                model=m,
                contents=contents,
                system_instruction=system_instruction,
                response_schema=schema,
                temperature=temperature,
                tier=tier,
            )
            calls.append(result.to_dict())
            parsed = extract_json(result.text)
            if parsed:
                return parsed, calls, result.text
        except Exception as exc:  # noqa: BLE001
            calls.append({"tier": tier, "model": m, "stage": "schema", "error": str(exc)[:300]})

        # Fallback attempt: prompt for JSON, no schema.
        try:
            fallback_prompt = (
                contents
                if isinstance(contents, str)
                else "Reply with a single JSON object and nothing else."
            )
            if isinstance(contents, list):
                extra = "Reply with a single JSON object conforming to the extraction schema, and nothing else."
                fallback_prompt = [*contents, extra]

            result = await gemini.generate(
                model=m,
                contents=fallback_prompt,
                system_instruction=system_instruction,
                temperature=temperature,
                tier=tier,
            )
            calls.append(result.to_dict())
            parsed = extract_json(result.text)
            if parsed:
                return parsed, calls, result.text
        except Exception as exc:  # noqa: BLE001
            calls.append({"tier": tier, "model": m, "stage": "fallback", "error": str(exc)[:300]})

    return None, calls, ""


async def _run_tool_loop(
    *,
    model: str,
    contents: list[Any],
    system_instruction: str,
    tier: str,
    thinking_level: str | None = None,
    max_iterations: int | None = None,
) -> tuple[BackendResult, list[dict[str, Any]]]:
    """Run Gemma 4's native function-calling loop until it stops asking."""
    limit = max_iterations or settings.max_tool_iterations
    calls: list[dict[str, Any]] = []
    history = list(contents)

    active_model = model
    try:
        result = await gemini.generate(
            model=active_model,
            contents=history,
            system_instruction=system_instruction,
            tools=registry.gemini_tools(),
            thinking_level=thinking_level,
            tier=tier,
        )
    except Exception as exc:
        calls.append({"tier": tier, "model": active_model, "error": str(exc)[:300]})
        alt_model = (
            settings.cloud_model_strong
            if active_model != settings.cloud_model_strong
            else settings.cloud_model_fast
        )
        if alt_model and alt_model != active_model:
            active_model = alt_model
            result = await gemini.generate(
                model=active_model,
                contents=history,
                system_instruction=system_instruction,
                tools=registry.gemini_tools(),
                thinking_level=thinking_level,
                tier=tier,
            )
        else:
            raise

    calls.append(result.to_dict())

    for _ in range(limit):
        if not result.tool_calls:
            break

        response_parts = []
        for call in result.tool_calls:
            payload = await registry.execute(call.name, call.args)
            calls.append(
                {
                    "tier": tier,
                    "model": active_model,
                    "tool": call.name,
                    "args": call.args,
                    "result": payload,
                }
            )
            response_parts.append(GeminiBackend.tool_response_part(call.name, payload))

        # Keep the model's own turn in the history, then add the tool results.
        model_turn = None
        for candidate in getattr(result.raw, "candidates", []) or []:
            model_turn = getattr(candidate, "content", None)
            break
        if model_turn is not None:
            history.append(model_turn)
        history.extend(response_parts)

        try:
            result = await gemini.generate(
                model=active_model,
                contents=history,
                system_instruction=system_instruction,
                tools=registry.gemini_tools(),
                thinking_level=thinking_level,
                tier=tier,
            )
            calls.append(result.to_dict())
        except Exception as exc:
            calls.append({"tier": tier, "model": active_model, "error": str(exc)[:300]})
            break

    return result, calls


# ===========================================================================
# Routes
# ===========================================================================
@app.get("/api/health")
async def health() -> JSONResponse:
    local_health = await local.health()
    models = await list_gemma_models() if settings.cloud_ready else []
    return JSONResponse(
        {
            "status": "ok",
            "config": settings.public_dict(),
            "local": local_health,
            "cloud_models_available": models,
            "graphify": {
                "installed": bool(_graphify_available()),
                "graph_present": (REPO_ROOT / "graphify-out" / "graph.html").exists(),
                "report_present": (REPO_ROOT / "GRAPH_REPORT.md").exists(),
            },
        }
    )


def _graphify_available() -> bool:
    from .tools import _find_graphify

    return _find_graphify() is not None


@app.post("/api/triage")
async def triage_endpoint(payload: dict[str, Any]) -> dict[str, Any]:
    """Expose the router on its own — handy for the demo."""
    prompt = str(payload.get("prompt", ""))
    decision = await router.decide(
        prompt=prompt,
        has_image=bool(payload.get("has_image")),
        enable_tools=bool(payload.get("enable_tools", True)),
        forced=str(payload.get("route", "auto")),
        use_local=bool(payload.get("use_local", True)),
    )
    return decision.model_dump()


@app.post("/api/audit")
async def audit(req: AuditRequest) -> AuditResponse:
    timings: dict[str, int] = {}
    model_calls: list[dict[str, Any]] = []
    tiers_used: list[str] = []
    findings: list[AuditFinding] = []
    decision: RouteDecision | None = None
    bill: dict[str, Any] | None = None

    try:
        image_bytes: bytes | None = None
        mime_type = req.mime_type
        if req.image_base64:
            try:
                image_bytes, mime_type = _split_image(req.image_base64)
            except HTTPException:
                raise
            if len(image_bytes) > 12 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Image is larger than 12 MB.")

        # -- Stage 0: routing ------------------------------------------------
        started = time.perf_counter()
        decision = await router.decide(
            prompt=req.prompt,
            has_image=image_bytes is not None,
            enable_tools=req.enable_tools,
            forced=req.route,
            use_local=req.use_local,
        )
        timings["route_ms"] = int((time.perf_counter() - started) * 1000)

        if image_bytes is None:
            return AuditResponse(
                ok=False,
                route=decision,
                tiers_used=tiers_used,
                timings=timings,
                model_calls=model_calls,
                error="No image was supplied. Send `image_base64` with the photo of the bill.",
            )

        cloud_model = (
            settings.cloud_model_strong if decision.tier == "cloud_strong" else settings.cloud_model_fast
        )
        if not settings.cloud_ready:
            return AuditResponse(
                ok=False,
                route=decision,
                tiers_used=tiers_used,
                timings=timings,
                model_calls=model_calls,
                error="GEMINI_API_KEY is not set, so the cloud tier cannot run the vision stage.",
            )

        # -- Stage 1: extraction (vision) ------------------------------------
        started = time.perf_counter()
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
        parsed_bill, calls, _raw = await _structured(
            model=cloud_model,
            contents=[image_part, EXTRACT_PROMPT],
            system_instruction=EXTRACT_SYSTEM,
            schema=BillFields,
            tier=decision.tier,
            temperature=0.05,
        )
        timings["extract_ms"] = int((time.perf_counter() - started) * 1000)
        model_calls.extend(calls)
        if decision.tier not in tiers_used:
            tiers_used.append(decision.tier)

        if not parsed_bill:
            return AuditResponse(
                ok=False,
                route=decision,
                tiers_used=tiers_used,
                timings=timings,
                model_calls=model_calls,
                error="The model could not read or extract structured fields from this image. Please check the image resolution and clarity.",
            )

        bill = parsed_bill

        # -- Stage 2: exact arithmetic in Python ------------------------------
        started = time.perf_counter()
        math_findings, math_report = python_audit(bill)
        findings.extend(math_findings)
        timings["python_audit_ms"] = int((time.perf_counter() - started) * 1000)

        # -- Stage 2b: on-device audit by local Gemma 4 ----------------------
        started = time.perf_counter()
        local_audit: dict[str, Any] | None = None
        if req.use_local and await local.detect_mode() is not None:
            try:
                result = await local.chat(
                    messages=[
                        {"role": "system", "content": LOCAL_AUDIT_SYSTEM},
                        {
                            "role": "user",
                            "content": LOCAL_AUDIT_PROMPT.format(bill_json=dumps(bill)),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=900,
                    json_mode=True,
                )
                model_calls.append(result.to_dict())
                local_audit = extract_json(result.text)
                if "local" not in tiers_used:
                    tiers_used.append("local")
            except Exception as exc:  # noqa: BLE001
                model_calls.append({"tier": "local", "error": str(exc)[:300]})
        timings["local_audit_ms"] = int((time.perf_counter() - started) * 1000)

        if local_audit:
            for anomaly in (local_audit.get("anomalies") or [])[:8]:
                findings.append(
                    AuditFinding(
                        severity="warning",
                        title="On-device observation",
                        detail=str(anomaly)[:400],
                        source="local_gemma",
                    )
                )

        # -- Stage 3: redaction ------------------------------------------------
        started = time.perf_counter()
        redacted_bill, removed = redact_bill(bill)
        local_redactions: dict[str, Any] = {}
        if local_audit and isinstance(local_audit.get("redacted_fields"), dict):
            local_redactions = local_audit["redacted_fields"]
        for key, value in list(local_redactions.items())[:12]:
            if key in redacted_bill and isinstance(value, str):
                redacted_bill[key] = value
        if removed:
            findings.append(
                AuditFinding(
                    severity="info",
                    title="Identifiers were removed before cloud reasoning",
                    detail="Redacted fields: " + ", ".join(removed) + ".",
                    source="python",
                )
            )
        timings["redact_ms"] = int((time.perf_counter() - started) * 1000)

        # -- Stage 4: advisory reasoning with native tool calling -------------
        started = time.perf_counter()
        advisor_prompt = ADVISOR_PROMPT.format(
            doc_type=redacted_bill.get("document_type", "bill"),
            vendor=redacted_bill.get("vendor_name") or "unknown",
            date=redacted_bill.get("date") or "unknown",
            currency=redacted_bill.get("currency") or "INR",
            line_items=dumps(redacted_bill.get("line_items") or []),
            subtotal=redacted_bill.get("subtotal"),
            tax_total=redacted_bill.get("tax_total"),
            total=redacted_bill.get("total"),
            math_report=dumps(math_report),
            local_audit=dumps(local_audit) if local_audit else "(local tier unavailable)",
            language=req.language,
        )
        advisor_system = build_advisor_system(settings.low_thinking and req.thinking == "minimal")

        try:
            result, calls = await _run_tool_loop(
                model=cloud_model,
                contents=[advisor_prompt],
                system_instruction=advisor_system,
                tier=decision.tier,
                thinking_level="high" if req.thinking == "high" else "minimal",
            )
            model_calls.extend(calls)
            if decision.tier not in tiers_used:
                tiers_used.append(decision.tier)
            final_text = result.text
        except Exception as exc:  # noqa: BLE001
            model_calls.append({"tier": decision.tier, "error": str(exc)[:300]})
            final_text = ""
        timings["advise_ms"] = int((time.perf_counter() - started) * 1000)

        # Robust extraction of accounting entries and findings from both JSON and Markdown
        entries = extract_accounting_entries_from_text(final_text)
        if not entries and bill:
            entries = generate_fallback_accounting_entries(bill)
        cloud_findings = extract_findings_from_text(final_text)
        for cf in cloud_findings:
            # Avoid duplicate titles
            if not any(f.title.lower() == cf.title.lower() for f in findings):
                findings.append(cf)

        explanation = final_text.strip()
        parsed_final = extract_json(final_text) if final_text else None
        if parsed_final:
            if parsed_final.get("explanation") or parsed_final.get("summary"):
                explanation = str(
                    parsed_final.get("explanation") or parsed_final.get("summary")
                ).strip()

        if not explanation:
            explanation = (
                "The advisory stage did not return text. "
                f"The arithmetic check is still available: {dumps(math_report)}"
            )

        return AuditResponse(
            ok=True,
            bill=bill,
            findings=findings,
            entries=entries,
            explanation=explanation,
            language=req.language,
            route=decision,
            tiers_used=tiers_used,
            timings=timings,
            model_calls=model_calls,
        )
    except HTTPException:
        raise
    except Exception as exc:
        err_str = str(exc)
        if "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str:
            friendly = "The Gemma 4 model is currently experiencing high demand upstream on the Gemini API. Please retry in a moment."
        elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            friendly = "Gemini API rate limit reached. Please wait a few seconds and try again."
        elif "500" in err_str or "INTERNAL" in err_str:
            friendly = "Gemini API returned an internal error. Please retry the audit."
        else:
            friendly = f"Audit encountered an error: {err_str[:250]}"

        return AuditResponse(
            ok=False,
            bill=bill,
            findings=findings,
            entries=[],
            explanation="",
            language=req.language,
            route=decision or RouteDecision(tier="cloud_fast", reason="Fallback", confidence=0.5, decided_by="rules"),
            tiers_used=tiers_used,
            timings=timings,
            model_calls=model_calls,
            error=friendly,
        )


@app.post("/api/chat")
async def chat(req: ChatRequest) -> ChatResponse:
    """A general chat endpoint that routes like everything else."""
    timings: dict[str, int] = {}
    model_calls: list[dict[str, Any]] = []
    tiers_used: list[str] = []

    started = time.perf_counter()
    decision = await router.decide(
        prompt=req.prompt,
        has_image=False,
        enable_tools=req.enable_tools,
        forced=req.route,
    )
    timings["route_ms"] = int((time.perf_counter() - started) * 1000)

    if decision.tier == "local":
        try:
            messages = [
                {"role": "system", "content": req.system_instruction or "You are a helpful assistant."},
                {"role": "user", "content": req.prompt},
            ]
            result = await local.chat(
                messages=messages,
                temperature=0.3,
                max_tokens=settings.max_output_tokens,
            )
            model_calls.append(result.to_dict())
            tiers_used.append("local")
            return ChatResponse(
                ok=True,
                text=result.text,
                route=decision,
                tiers_used=tiers_used,
                timings=timings,
            )
        except Exception as exc:  # noqa: BLE001
            model_calls.append({"tier": "local", "error": str(exc)[:300]})
            decision = decision.model_copy(
                update={
                    "tier": "cloud_fast",
                    "reason": "The local runtime failed mid-request, so the cloud tier took over.",
                    "decided_by": "rules",
                }
            )

    model = (
        settings.cloud_model_strong if decision.tier == "cloud_strong" else settings.cloud_model_fast
    )
    if not settings.cloud_ready:
        return ChatResponse(
            ok=False,
            route=decision,
            tiers_used=tiers_used,
            timings=timings,
            error="GEMINI_API_KEY is not set.",
        )

    started = time.perf_counter()
    try:
        if req.enable_tools:
            result, calls = await _run_tool_loop(
                model=model,
                contents=[req.prompt],
                system_instruction=req.system_instruction or "You are a helpful assistant powered by Gemma 4.",
                tier=decision.tier,
                thinking_level="high" if req.thinking == "high" else "minimal",
            )
        else:
            result = await gemini.generate(
                model=model,
                contents=[req.prompt],
                system_instruction=req.system_instruction or "You are a helpful assistant powered by Gemma 4.",
                thinking_level="high" if req.thinking == "high" else "minimal",
                tier=decision.tier,
            )
            calls = [result.to_dict()]
        model_calls.extend(calls)
        tiers_used.append(decision.tier)
        timings["generate_ms"] = int((time.perf_counter() - started) * 1000)
        return ChatResponse(
            ok=True,
            text=result.text,
            thinking=result.thinking,
            tool_calls=[c.to_dict() for c in result.tool_calls],
            route=decision,
            tiers_used=tiers_used,
            timings=timings,
        )
    except Exception as exc:  # noqa: BLE001
        model_calls.append({"tier": decision.tier, "error": str(exc)[:300]})
        return ChatResponse(
            ok=False,
            route=decision,
            tiers_used=tiers_used,
            timings=timings,
            error=str(exc)[:500],
        )


@app.get("/api/tools")
async def tools_endpoint() -> dict[str, Any]:
    return {"tools": registry.declarations()}


@app.get("/api/graph")
async def graph_endpoint() -> Any:
    """Serve the Graphify knowledge-graph report if it has been generated."""
    target = REPO_ROOT / "graphify-out" / "graph.html"
    if not target.exists():
        raise HTTPException(
            status_code=404,
            detail="No graph found. Run 'graphify extract .' in the project root first.",
        )
    return FileResponse(target)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index() -> Any:
    target = STATIC_DIR / "index.html"
    if not target.exists():
        return JSONResponse({"detail": "UI not installed."}, status_code=404)
    return FileResponse(target)


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":  # pragma: no cover
    main()
