# Graph Report - gemma-hybrid-agent  (2026-10-22)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 204 nodes · 409 edges · 14 communities (9 shown, 5 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 30 edges (avg confidence: 0.91)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11

## God Nodes (most connected - your core abstractions)
1. `audit()` - 18 edges
2. `BackendResult` - 14 edges
3. `LocalBackend` - 14 edges
4. `GeminiBackend` - 14 edges
5. `build_registry()` - 13 edges
6. `extract_json()` - 11 edges
7. `ToolCall` - 10 edges
8. `Router` - 9 edges
9. `list_gemma_models()` - 8 edges
10. `_run_tool_loop()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `GeminiBackend` --uses--> `BackendResult`  [INFERRED]
  app/backends/gemini.py → app/backends/base.py
- `_run_tool_loop()` --uses--> `BackendResult`  [INFERRED]
  app/main.py → app/backends/base.py
- `Router` --uses--> `BackendResult`  [INFERRED]
  app/router.py → app/backends/base.py
- `GeminiBackend` --uses--> `ToolCall`  [INFERRED]
  app/backends/gemini.py → app/backends/base.py
- `Router` --uses--> `LocalBackend`  [INFERRED]
  app/router.py → app/backends/local.py

## Import Cycles
- None detected.

## Communities (14 total, 5 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.08
Nodes (17): build_registry(), _calculate(), _fetch_url(), _graphify_explain(), _graphify_path(), _graphify_query(), _gst_slab(), fetch_url() (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.10
Nodes (5): BackendResult, ToolCall, key_candidates(), list_gemma_models(), LocalBackend

### Community 2 - "Community 2"
Cohesion: 0.17
Nodes (6): extract_json(), dumps(), _matches(), Router, _salvage_triage(), RouteDecision

### Community 4 - "Community 4"
Cohesion: 0.15
Nodes (6): _bool(), _float(), _int(), Settings, _str(), build_advisor_system()

### Community 5 - "Community 5"
Cohesion: 0.19
Nodes (9): AccountingEntry, AuditRequest, AuditResponse, BillFields, ChatRequest, ChatResponse, LineItem, LocalAudit (+1 more)

### Community 6 - "Community 6"
Cohesion: 0.22
Nodes (4): _split_image(), main(), make_receipt(), record()

### Community 8 - "Community 8"
Cohesion: 0.25
Nodes (3): chat(), _run_tool_loop(), triage_endpoint()

### Community 9 - "Community 9"
Cohesion: 0.32
Nodes (4): graph_endpoint(), index(), _structured(), tools_endpoint()

### Community 10 - "Community 10"
Cohesion: 0.47
Nodes (4): audit(), _f(), python_audit(), AuditFinding

## Knowledge Gaps
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `build_registry()` connect `Community 0` to `Community 7`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Are the 6 inferred relationships involving `audit()` (e.g. with `AccountingEntry` and `AuditFinding`) actually correct?**
  _`audit()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.07549361207897794 - nodes in this community are weakly interconnected._
- **Why does `LocalBackend` connect `Community 1` to `Community 2`, `Community 7`?**
  _High betweenness centrality (0.085) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `BackendResult` (e.g. with `GeminiBackend` and `LocalBackend`) actually correct?**
  _`BackendResult` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.10160427807486631 - nodes in this community are weakly interconnected._
- **Why does `GeminiBackend` connect `Community 3` to `Community 8`, `Community 1`, `Community 7`?**
  _High betweenness centrality (0.079) - this node is a cross-community bridge._