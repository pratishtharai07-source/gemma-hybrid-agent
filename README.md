# LedgerLens — a hybrid Gemma 4 bill auditor

**Photo of a bill → structured line items → exact arithmetic audit → live GST-slab check → accounting entries you can actually post.**

Built for the **Best Use of Gemma 4** and **Best Open-Source AI Project**
tracks. Python + FastAPI. Runs a 2.3B open-weight Gemma 4 on your own
laptop and escalates to the hosted Gemma 4 models through the Gemini API
only when the work actually needs them.

---

## Why this exists

A shopkeeper in Pune gets a stack of bills at month end. They need to know:

* does the arithmetic actually add up?
* is the GST rate right for what was bought?
* what do I post to my ledger?

Existing tools make you choose between a cloud API (data leaves the device,
costs money per call) and a heavyweight local model (won't run on a 4 GB
laptop). LedgerLens refuses that trade-off: **a small open-weight model on
your own machine does the cheap, private, high-volume work, and the cloud
model only handles what genuinely needs a big brain.**

---

## The rule that shapes the whole codebase

> **Python does the maths. Gemma does the semantics.**

A language model is a poor calculator and an excellent auditor. So every
number on a bill is recomputed in Python with exact float arithmetic, and
Gemma 4 is asked for judgement, explanation and taxonomy. The model is
*never* trusted with a sum.

---

## Architecture

```
                photo of a bill
                      │
  [0] ROUTE ─────────┤  local Gemma 4 E2B (llama.cpp, 2.4 GB Q3_K_M)
                      │  classifies the request → local | cloud_fast | cloud_strong
                      │  + deterministic overrides (image ⇒ vision, "today" ⇒ live data)
                      ▼
  [1] EXTRACT ───────┤  cloud Gemma 4 (26B-A4B / 31B) · vision
                      │  → structured JSON: vendor, GSTIN, line items, totals
                      ▼
  [2] VERIFY ────────┤  Python: recompute subtotal / tax / total exactly
                      │  Local Gemma 4 E2B: offline plausibility audit
                      ▼
  [3] REDACT ────────┤  local Gemma 4 E2B + regex scrubber strip GSTIN, phone, email
                      ▼
  [4] ADVISE ────────┤  cloud Gemma 4 · thinking=high · native function calling
                      │  tools: calculate, gst_slab_reference, current_datetime,
                      │         fetch_url, graphify_query, graphify_explain, graphify_path
                      ▼
        verdict · findings · ledger entries · plain-language note
```

### Why the local tier earns its place

| Stage | Runs on | Why |
|---|---|---|
| Routing | **local** | every request passes through it; a cloud model here would cost a call for every request |
| Audit + redaction | **local** | the sensitive half of the work never needs to leave the device |
| Vision extraction | cloud | E2B is text-only on the Gemini API — E2B/E4B model IDs return **404** |
| Advisory reasoning | cloud | this is where a 31B model actually earns its tokens |

The local tier is auto-detected. Stop `llama-server` and the pipeline
degrades to cloud-only with no code change and no error.

---

## The Gemma 4 features this actually uses

| Feature | Where |
|---|---|
| **Vision / multimodal input** | stage 1 reads the receipt image |
| **Structured output** (`response_schema`) | stage 1 returns schema-validated JSON |
| **Native function calling** | stage 4 tool loop — `calculate`, `fetch_url`, `graphify_*` |
| **Thinking** (`thinking_level`) | stage 4 uses `high`; the router uses `minimal` |
| **Open-weight local execution** | Gemma 4 E2B Q3_K_M via llama.cpp — routing, audit, redaction |
| **Gemma 4 chat control tokens** | server-side on the Gemini API; raw `<\|turn\|>`/`<\|tool\|>` handled by llama.cpp |

---

## Graphify integration

[Graphify](https://graphify.com) (Apache-2.0) turns a codebase into a
queryable knowledge graph. It is integrated in two places:

**1. It is our own map.** Running `graphify extract .` produces
`graphify-out/graph.html` (an interactive graph) and `GRAPH_REPORT.md`.
This repo ships both — the project's architecture is itself generated
by the tool the event asked us to use.

```
graphify-out/
  graph.json      204 nodes, 409 edges, 14 communities
  graph.html      interactive visualisation
  GRAPH_REPORT.md
```

**2. It is a tool Gemma 4 can call.** The registry in `app/tools.py`
exposes `graphify_query`, `graphify_explain` and `graphify_path` as native
function declarations. The "Ask the codebase" tab in the UI asks Gemma 4 a
question about *our own source*, and Gemma reaches for the graph:

```
Q: How does the router decide which tier to use?
→ Gemma calls graphify_query("router tier decision")
→ graph.json BFS returns the relevant nodes and edges
→ Gemma answers with evidence
```

**3. Graphify can run on our local model.** Graphify's `--backend openai`
route reaches any OpenAI-compatible server, including llama.cpp:

```powershell
$env:OPENAI_BASE_URL = "http://127.0.0.1:8080/v1"
$env:OPENAI_MODEL    = "gemma4-e2b"
graphify extract . --backend openai
```

So the *semantic* pass of the graph — the inferred edges between functions
that never call each other — is generated by a 2.3B open-weight model
running on the same 4 GB laptop. Nothing leaves the machine.

---

## Quick start

### 1. Cloud tier (required)

```powershell
pip install -r requirements.txt
copy .env.example .env      # then paste your GEMINI_API_KEY
```

Get a key at <https://aistudio.google.com/apikey>.

> **Key format note.** Google now issues keys as `AQ.Ab...`. If yours is
> `AQAb...` the dot was lost in transit — the app repairs this
> automatically, but store the dotted form. See `NOTES-api-findings.md`.

### 2. Local tier (optional but recommended)

```powershell
.\scripts\setup_local.ps1
```

This downloads `gemma-4-E2B-it-Q3_K_M.gguf` (~2.4 GB, resumable) and starts
`llama-server` on `127.0.0.1:8080`. Flags are tuned for a 4 GB laptop:
`-c 2048`, 4 threads, no warmup.

Already have Ollama? `LOCAL_LLM_MODE=ollama` and it will be detected
automatically — though note the published `gemma4:e2b` tags are 4.3 GB and
up, which will not fit in 3.8 GB of RAM. llama.cpp Q3_K_M is the only local
option that does.

### 3. Run it

```powershell
python -m uvicorn app.main:app --port 8000
```

Open <http://127.0.0.1:8000>.

### 4. Check it works

```powershell
python scripts\smoke_test.py
```

---

## API

| Endpoint | What it does |
|---|---|
| `GET /api/health` | which tiers are live, which models the key can reach, Graphify status |
| `POST /api/audit` | the full 4-stage pipeline on a base64 image |
| `POST /api/triage` | the routing decision on its own — no model answering, just the router |
| `POST /api/chat` | routed chat with the tool loop |
| `GET /api/tools` | the tool registry as JSON |
| `GET /api/graph` | serves `graphify-out/graph.html` |

```powershell
curl -X POST http://127.0.0.1:8000/api/audit `
  -H "Content-Type: application/json" `
  -d "{\"image_base64\":\"<base64>\",\"mime_type\":\"image/png\",\"language\":\"Hindi\"}"
```

---

## Tools available to Gemma 4

| Tool | Does |
|---|---|
| `calculate` | exact arithmetic via an AST sandbox — the model is told to never add anything up in its head |
| `gst_slab_reference` | offline table of Indian GST slabs (0/5/12/18/28%) with typical examples |
| `current_datetime` | current date, time and timezone |
| `fetch_url` | real HTTP fetch of a public page, with an SSRF guard (private IPs refused) |
| `graphify_query` | BFS a question through our own code knowledge graph |
| `graphify_explain` | explain a symbol and its neighbours |
| `graphify_path` | shortest relationship path between two symbols |

`fetch_url` exists because **Google Search grounding does not actually
ground on Gemma 4** through the Gemini API — it is accepted, returns a
confident answer, and produces **0 grounding chunks**. We measured it; see
`NOTES-api-findings.md`. Rather than depend on a tool that silently does
nothing, we fetch the page ourselves.

---

## Reliability

The Gemini API is flaky for Gemma 4 — `500 INTERNAL` and `503 UNAVAILABLE`
come back for plain text prompts and clear up on the next attempt. Every
call is retried with exponential backoff (`app/backends/gemini.py`).
Structured output also degrades gracefully: if `response_schema` is
rejected, the pipeline asks for the same JSON in prose and parses it with
a lenient scanner.

The app never dies because a model failed. Every stage is wrapped, every
failure degrades to the next best thing, and the trace tab shows exactly
which tier produced which part of the answer.

---

## Layout

```
app/
  main.py           FastAPI app, the 4-stage pipeline, deterministic audit
  router.py         hybrid routing: local triage + deterministic overrides
  config.py         env-driven settings
  schemas.py        Pydantic models (double as Gemini response schemas)
  prompts.py        prompt templates per stage
  tools.py          tool registry, safe arithmetic, GST reference, Graphify bridge
  backends/
    base.py         shared result envelope + lenient JSON parser
    gemini.py       cloud tier: retries, key repair, tool round-trip
    local.py        local tier: llama.cpp and Ollama, auto-detected
  static/index.html demo UI
scripts/
  setup_local.ps1   download + start the local Gemma 4 tier
  smoke_test.py     end-to-end pre-demo check
graphify-out/       our own knowledge graph (graph.json, graph.html)
GRAPH_REPORT.md     Graphify's report on this codebase
NOTES-api-findings.md  everything we measured about the Gemini API
```

---

## What we would do next

* **Local vision.** E2B is text-only on the API and the local Q3_K_M build
  has no projector, so extraction needs the cloud tier. A machine with
  8 GB+ could run `mmproj` locally and keep the *pixels* on-device too.
* **Duplicate detection.** Fingerprint bills across months to catch
  re-submitted expenses.
* **A Tally/Zoho Books exporter** for the ledger entries.

---

## Licence

MIT — see `LICENSE`.

Gemma 4 weights are released by Google under the Gemma terms.
Graphify is Apache-2.0.
