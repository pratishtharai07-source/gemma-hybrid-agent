"""Pydantic models for the API surface and for structured model output.

The `Bill*`, `TriageVerdict` and `LocalAudit` models double as response
schemas for Gemma 4 — `google-genai` accepts a Pydantic model directly for
`response_schema`, which gives us guaranteed-shape JSON back.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

RouteName = Literal["local", "cloud_fast", "cloud_strong"]
TierName = Literal["local", "cloud_fast", "cloud_strong"]


# ---------------------------------------------------------------------------
# Structured model output
# ---------------------------------------------------------------------------
class LineItem(BaseModel):
    description: str = Field(description="What was purchased, exactly as printed.")
    quantity: float | None = Field(default=None, description="Units purchased.")
    unit_price: float | None = Field(default=None, description="Price for one unit.")
    amount: float | None = Field(default=None, description="Line total as printed.")
    tax_rate: float | None = Field(default=None, description="Tax percentage, e.g. 5 for 5% GST.")


class BillFields(BaseModel):
    """Everything we can read off a bill or receipt."""

    document_type: str = Field(description="invoice | receipt | bill | quotation | other")
    vendor_name: str | None = Field(default=None, description="Registered name of the seller.")
    vendor_gstin: str | None = Field(default=None, description="15 character GSTIN if printed.")
    vendor_phone: str | None = Field(default=None, description="Contact number if printed.")
    date: str | None = Field(default=None, description="Invoice date as printed, or null.")
    currency: str = Field(default="INR", description="ISO currency code of the amounts.")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: float | None = Field(default=None, description="Taxable value before tax.")
    tax_total: float | None = Field(default=None, description="Total tax charged.")
    total: float | None = Field(default=None, description="Grand total payable.")
    payment_method: str | None = Field(default=None, description="cash, card, upi, netbanking, etc.")
    confidence: float = Field(default=0.5, description="How confident you are that the fields are correct (0-1).")


class TriageVerdict(BaseModel):
    """The local router's structured decision about how to handle a request."""

    task_type: str = Field(description="audit | vision | extraction | summarize | explain | general")
    sensitivity: str = Field(description="low | medium | high — how sensitive the data appears.")
    complexity: float = Field(description="How hard the task is, 0.0 (trivial) to 1.0 (very hard).")
    needs_live_data: bool = Field(description="Does answering require current information from the web?")
    needs_tools: bool = Field(description="Does answering require calling a tool?")
    route: str = Field(description="local | cloud_fast | cloud_strong — your recommendation.")
    reason: str = Field(description="One short sentence explaining the recommendation.")
    confidence: float = Field(description="How confident you are in this verdict, 0.0 to 1.0.")


class LocalAudit(BaseModel):
    """The on-device verdict produced by Gemma 4 E2B."""

    arithmetic_consistent: bool | None = Field(default=None, description="Do the printed totals add up?")
    tax_rates_plausible: bool | None = Field(default=None, description="Are the tax rates reasonable for these items?")
    anomalies: list[str] = Field(default_factory=list, description="Anything odd you noticed.")
    pii_fields: list[str] = Field(default_factory=list, description="Field names holding personal or business identifiers.")
    redacted_fields: dict[str, str] = Field(default_factory=dict, description="Redacted replacement values, keyed by field name.")
    confidence: float = Field(default=0.5, description="Your confidence in this audit, 0.0 to 1.0.")


class AuditFinding(BaseModel):
    severity: str = Field(description="info | warning | critical")
    title: str
    detail: str
    source: str = Field(description="python | local_gemma | cloud_gemma")


class AccountingEntry(BaseModel):
    account: str = Field(description="Ledger account name, e.g. 'Purchases', 'GST Input (CGST)'.")
    debit: float = 0.0
    credit: float = 0.0
    narration: str = ""


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------
class AuditRequest(BaseModel):
    image_base64: str | None = Field(default=None, description="Base64 image bytes, with or without a data: prefix.")
    mime_type: str = Field(default="image/png")
    prompt: str = Field(default="Audit this bill and explain any problems.", description="Free-form instruction.")
    language: str = Field(default="English", description="Language for the human-readable explanation.")
    route: Literal["auto", "local", "cloud_fast", "cloud_strong"] = Field(default="auto")
    enable_tools: bool = Field(default=True)
    enable_search: bool = Field(default=False, description="Try Google Search grounding (best effort).")
    thinking: Literal["minimal", "high"] = Field(default="high")
    use_local: bool = Field(default=True, description="Allow the on-device Gemma 4 tier to be used.")


class RouteDecision(BaseModel):
    tier: TierName
    reason: str
    confidence: float
    decided_by: str = Field(description="local_triage | rules | user_override")
    complexity: float = 0.0
    sensitivity: str = "unknown"
    needs_vision: bool = False
    needs_tools: bool = False
    needs_live_data: bool = False
    triage: dict[str, Any] | None = None


class AuditResponse(BaseModel):
    ok: bool
    bill: dict[str, Any] | None = None
    findings: list[AuditFinding] = Field(default_factory=list)
    entries: list[AccountingEntry] = Field(default_factory=list)
    explanation: str = ""
    language: str = "English"
    route: RouteDecision
    tiers_used: list[str] = Field(default_factory=list)
    timings: dict[str, int] = Field(default_factory=dict)
    model_calls: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None


class ChatRequest(BaseModel):
    prompt: str
    system_instruction: str | None = None
    route: Literal["auto", "local", "cloud_fast", "cloud_strong"] = Field(default="auto")
    enable_tools: bool = Field(default=True)
    thinking: Literal["minimal", "high"] = Field(default="high")


class ChatResponse(BaseModel):
    ok: bool
    text: str = ""
    thinking: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    route: RouteDecision
    tiers_used: list[str] = Field(default_factory=list)
    timings: dict[str, int] = Field(default_factory=dict)
    error: str | None = None
