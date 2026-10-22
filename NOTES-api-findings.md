# Gemini API findings for Gemma 4

Everything below was measured against the live Gemini API with the key in
`.env` (project `814315076808`). It is written down so nobody has to
rediscover it during the demo.

## The key format changed

Google migrated Gemini API keys from `AIza...` to a new **auth key** format
that begins `AQ.Ab...`.

A key pasted as `AQAb8RN6...` (dot lost) is rejected with:

```
400 INVALID_ARGUMENT
API key not valid. Please pass a valid API key.
reason: API_KEY_INVALID
```

Restoring the dot (`AQ.Ab8RN6...`) makes the same key work. The app repairs
this automatically — `app/backends/gemini.py::key_candidates()` returns
`[key, "AQ." + key[2:]]` and the backend probes each one until one
authenticates.

The same key also authenticates against Vertex AI express mode, but only
after the Agent Platform API is enabled on the project:

```
403 PERMISSION_DENIED
Agent Platform API has not been used in project 814315076808 before or it is disabled.
```

We use the plain Gemini API (`generativelanguage.googleapis.com`), which
needs no extra API enablement.

## Which Gemma 4 models exist on the Gemini API

`models.list` returns 63 models for this key; exactly **two** are Gemma 4:

| Model id | Status |
|---|---|
| `gemma-4-31b-it` | works |
| `gemma-4-26b-a4b-it` | works, but needs retries |
| `gemma-4-e2b-it` | 404 — local-only (GGUF) |
| `gemma-4-e4b-it` | 404 — local-only (GGUF) |

The E2B / E4B checkpoints are open-weight and **not** served by the Gemini
API. That is exactly why this project runs them locally.

## Capability matrix

| Capability | Result | Notes |
|---|---|---|
| Text generation | works | on both models |
| **Vision** (image in) | works | read `Rs 243.80` and `27ABCDE1234F1Z5` correctly off a generated receipt |
| **Thinking** (`thinking_level="high"`) | works | `17 * 23 = 391` |
| **Function calling** | works | native `function_calls`, tool-response round trip OK |
| **Structured output** (`response_schema`) | works | text-only is reliable; image + schema is flaky under load |
| **Google Search grounding** | **silently ungrounded** | see below |

## The 500s are transient, not fatal

`gemma-4-26b-a4b-it` intermittently returns:

```
500 INTERNAL  "Internal error encountered."
503 UNAVAILABLE "This model is currently experiencing high demand."
```

This happens on plain text prompts with no tools and no schema, and it
recovers on the next attempt. **Every call is retried with backoff**
(`app/backends/gemini.py::_with_retry`, 4 attempts, exponential delay). A
build that does not retry will look broken on stage.

## Google Search grounding does not actually ground

`tools=[{"google_search": {}}]` is accepted by the API and returns a
normal-looking answer, but:

* `grounding_metadata.grounding_chunks` is **0**
* `grounding_metadata.web_search_queries` is **None**
* the model says *"As of my last knowledge update in January 2025"*

So the tool is accepted and then ignored. The project therefore does **not**
depend on Search grounding. Live data comes from the `fetch_url` tool
instead, which is a real HTTP request with an SSRF guard, and from
`current_datetime`.

`enable_search` is still exposed in the API for completeness, and the
grounding chunk count is reported back so the behaviour is visible rather
than hidden.

## Quota

`gemini-3.8-flash` returns `429 RESOURCE_EXHAUSTED` — the free tier quota
for the Gemini models is exhausted on this key. `gemini-2.5-flash` returns
404 (retired). The Gemma 4 models have their own quota and still respond.

Practical consequence: **the demo stays on Gemma 4 only**, and the pipeline
is designed to make a small number of calls — one vision call, one tool-loop
call, plus the on-device tier.

## Structured output caveats

* `response_schema` + **text only**: reliable.
* `response_schema` + **image**: failed 3/3 under load, worked on retry.

The project always keeps a fallback: ask for the same JSON in prose and
parse it with the lenient scanner in `app/backends/base.py::extract_json()`.

## Key hygiene

The key lives in `.env`, which is git-ignored. It was pasted into a chat
session during development, so it should be rotated before the repository is
made public.
