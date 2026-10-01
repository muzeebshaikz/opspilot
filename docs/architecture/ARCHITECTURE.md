# OpsPilot AI — Layered Architecture

Companion to [`TECHNOLOGY-RATIONALE.md`](TECHNOLOGY-RATIONALE.md) (the *why* for
each choice, with comparison tables) and [`../DIVERGENCE.md`](../DIVERGENCE.md)
(what is built vs deferred). This document answers: **what sits at each layer,
what technology implements it, and what was rejected.**

All numbers cited are measured. See `artifacts/reports/`.

---

## 1. What kind of chatbot is this, exactly?

A fair question, because "RAG chatbot", "agent" and "MCP" get used
interchangeably and they are not the same thing. OpsPilot is all three, layered:

| Label | Does it apply? | What it actually means here |
| --- | --- | --- |
| **RAG chatbot** | **Yes — this is the core** | Every knowledge answer is retrieved from a document corpus first, then generated *only* from that evidence, with citations. The model is never the source of Acme facts. |
| **Tool-calling agent** | **Yes** | It also reads live structured state — tickets, incidents, device inventory — through 16 typed tools. This is what RAG alone cannot do: "is there an outage *right now*" is not in any document. |
| **MCP application** | **Yes, as the transport** | Those tools are exposed over Model Context Protocol servers. MCP is *how* tools are reached, not *what* the system is. |
| **Agentic / autonomous** | **Deliberately limited** | It routes, investigates and proposes. It does **not** act on its own: consequential actions stop for human approval. |
| **Multi-agent** | **No, on purpose** | One routed workflow, not a crew of negotiating personas. Nothing in the problem needs that, and it would add cost and nondeterminism. |
| **Fine-tuned model** | **No** | No training. See §1.1. |

**One sentence:** *a retrieval-grounded, tool-using IT copilot with server-side
authorization and human-in-the-loop approval, where the tools are reached over MCP.*

### 1.1 Why RAG and not fine-tuning

| | RAG (chosen) | Fine-tuning (rejected) |
| --- | --- | --- |
| Update a policy | Re-index one document, seconds | Retrain, hours-to-days |
| Show sources | Native — citations are the mechanism | Impossible |
| Per-role access | Filter at retrieval (`access_level`) | **Cannot be done** — weights don't have an ACL |
| Wrong answer | Trace to the retrieved passage | Opaque |

The access-control row is disqualifying on its own. You cannot fine-tune a model
that knows a document for `IT_admin` and not for `employee`. Retrieval can simply
not return it.

---

## 2. The request path

```
Browser ──SSE──► FastAPI ──► Rate limit ──► AuthN ──► LangGraph
                                                          │
                                        ┌─────────────────┴──────────────────┐
                                        ▼                                    ▼
                                 classify_intent                       (routing)
                                        │
        ┌───────────────┬───────────────┼────────────────┬────────────────┐
        ▼               ▼               ▼                ▼                ▼
    knowledge     troubleshooting    incident         ticket           action
        │               │               │                │                │
        ▼               ▼               ▼                ▼                ▼
   Hybrid RAG    RAG + incidents    incident MCP    ticketing MCP    authorize
        │           + device                                              │
        │               │                                          human approval
        └───────────────┴───────────────┬────────────────┘                │
                                        ▼                                 ▼
                            evidence sufficiency gate            execute (in-process,
                                        │                         idempotency-keyed)
                        ┌───────────────┴───────────────┐
                        ▼                               ▼
                   abstain                     generate + ground-check
                                                        │
                                              answer + citations
```

---

## 3. Layer-by-layer: technology, role, and what was rejected

### L1 — Presentation

| Concern | Technology | Why | Rejected |
| --- | --- | --- | --- |
| Chat UI | Single-file HTML/CSS/JS served at `/` | Demonstrates the SSE contract end to end with no Node toolchain | Next.js + Tailwind (spec §24) — **scaffold only, not built** |
| Streaming | **SSE** (`text/event-stream`) | One-directional, plain HTTP, survives corporate proxies, `EventSource` auto-reconnects | WebSockets (bidirectional not needed, more proxy friction); long polling (worse latency) |
| Approval UX | `approval_required` event → separate POST | An SSE stream **cannot block** for human input | Blocking the stream (would hold a connection open indefinitely) |

### L2 — API

| Concern | Technology | Why | Rejected |
| --- | --- | --- | --- |
| HTTP framework | **FastAPI** + Pydantic v2 | Schema validation at the edge, OpenAPI for free, native async for SSE | Flask (no async, no schema); Django (ORM/admin not needed) |
| Server | Uvicorn (ASGI) | Required for SSE | Gunicorn sync workers |
| Rate limiting | In-process sliding window | No Redis locally. **Honest limit: single-instance only** | Redis limiter (target; needs a server) |
| AuthN | `X-Employee-Id` header — **development shim** | Labelled as a shim, not dressed up as SSO | JWT/OIDC — **the one genuinely weak control**, next on the list |

### L3 — Orchestration

| Concern | Technology | Why | Rejected |
| --- | --- | --- | --- |
| Workflow | **LangGraph** | Earns its place for exactly two things: **checkpointing** (approval must survive the HTTP request dying) and **inspectable routing state** (which makes the UI activity trace truthful) | Plain functions (fine until HITL, then you write your own checkpointer badly); LangChain AgentExecutor (opaque); CrewAI/AutoGen (multi-agent theatre for a deterministic workflow); Temporal (right at scale, overkill here) |
| Intent routing | `gpt-oss-20b`, JSON-constrained | Cheap, fast, high call volume | Rules only (brittle); the 120b model (wasteful for a 6-way classification) |
| Memory | Bounded recent-turn tail | Spec §14 forbids injecting all history | Full-history injection (blows the 8k TPM ceiling) |

### L4 — Retrieval (RAG)

| Stage | Technology | Why | Rejected |
| --- | --- | --- | --- |
| Chunking | **Structure-aware** (markdown headings, procedures kept intact) | A truncated 5-step remediation is actively harmful, not just lower-scoring | Fixed 500/100 — *kept as the control arm*, spec §8 forbids it as the only strategy |
| Dense | Hashing embedder (8,192-dim) — **by constraint** | Corporate proxy blocks HuggingFace model files; this keeps everything runnable and measurable | bge-small via ONNX — **implemented, one config flag away** |
| Sparse | **BM25, hand-implemented** | Needed IDF for sparse vectors, per-term explainability, and explicit identifier tokenisation (`ERR_VPN_809` → `err_vpn_809` + `err`,`vpn`,`809`) | `rank_bm25` (no IDF access, no tokenisation control); SPLADE (needs a blocked model) |
| Fusion | **RRF (k=60)**, fused in Python | Rank-based, so no score normalisation between incomparable scales. Python-side keeps per-arm lists intact for the per-class analysis | Weighted score fusion (hidden normalisation constants); Qdrant server-side fusion (**used in production path**, but collapses the arms the evaluation needs) |
| Reranking | **Feature reranker** (`lexical-v2`): IDF-weighted coverage, identifier match, status recency, product agreement | Scores signals fusion structurally cannot see. Zero model, ~2 ms | Cross-encoder ONNX (implemented, model blocked); Cohere rerank (another vendor + hop) |
| Vector store | **Qdrant, embedded mode** | Verified by spike: sparse vectors + prefetch + server-side RRF + payload filters, **with no server and no Docker** | pgvector (no sparse arm, needs a server); FAISS (no filtering — the RBAC filter rules it out); Chroma (weaker filtering) |
| Grounding | Evidence-sufficiency gate + lexical grounding check | Abstention threshold **chosen from a 101-point measured sweep**, not guessed | A guessed threshold — *tried, and it failed*: 54.5% hallucination rate |

### L5 — Tools & MCP

| Concern | Technology | Why | Rejected |
| --- | --- | --- | --- |
| Tool contracts | 16 Pydantic-typed tools | validate → authorize → resource-check → execute → audit | Free-form function calling (no validation, no authz) |
| Protocol | **MCP over stdio** (SDK 2.x, `MCPServer`) | Spec §7. Honest benefit: **interoperability** — any MCP client can drive these tools | Claiming MCP improves performance (it does not — it is pure overhead in-process) |
| Domains | 4 servers: ticketing · incident · asset · knowledge | Spec §7 boundaries | One monolith server |
| Identity | **HMAC-signed, expiring grant, minted server-side per call** | See §4 — this is the security-critical part | Plain `employee_id` argument (model-controlled = spoofable); env-pinned identity (measured: forces one process per user, 4–7 s cold start each) |
| Failure | Falls back in-process, **fallback counted** in `/metrics` | Spec §29. A silent fallback would let "MCP never actually runs" hide | Failing the request |

### L6 — Data

| Store | Technology | Holds | Why not elsewhere |
| --- | --- | --- | --- |
| Relational | **SQLite** (PostgreSQL-portable schema) | users, assets, tickets, incidents, conversations, messages, **audit_logs**, approvals | Spec §13 forbids putting this in the vector DB — it needs transactions, FKs and exact filters |
| Vector | **Qdrant embedded** | 363 chunks: dense + sparse vectors + metadata | Not a substitute for the relational store |
| Cache | In-process, Redis-shaped interface | LLM responses | Redis is the target; **honest limit: single-instance, lost on restart** |

### L7 — Models

| Role | Model | Why this one | Measured |
| --- | --- | --- | --- |
| Routing | `openai/gpt-oss-20b` | ~1000 tok/s, cheapest, high call volume | ~235 tok/call |
| Generation | `openai/gpt-oss-120b` | Called once per answer | ~2,000 prompt tok/answer |
| Judge (eval) | `qwen/qwen3.8-27b` | **Independent family** — a different size of the generator would be self-evaluation | Not yet used |
| Injection | `meta-llama/llama-prompt-guard-2-86m` | Purpose-trained classifier, **token-free** (usage reports 0) | +3.1pp recall, **+14pp false positives** |

> **Correction to an earlier assumption.** `llama-3.3-70b-versatile` is **not
> available** on this account — verified against `GET /v1/models`. The judge was
> moved to `qwen/qwen3.8-27b`. Also: `gpt-oss` are **reasoning models** that
> spend completion tokens on an internal `reasoning` field before emitting
> `content`; a small `max_tokens` returns an **empty answer**. The provider now
> floors the completion budget and sends `max_completion_tokens`, never the
> legacy `max_tokens` (which makes `qwen` return 404).

### L8 — Security

| Control | Where | Measured |
| --- | --- | --- |
| Authorization | Service layer, table-driven, **outside the LLM** | **5/5 probes pass** |
| Identity integrity | HMAC grant, 120 s TTL | Forged/expired/tampered all rejected |
| Injection — direct | Regex + Prompt Guard 2 | recall 81.2%, precision 86.7% |
| Injection — **indirect** | Trust tiering; retrieved/tool text is **never instruction-bearing**; tool args derive only from the user turn | Structural, test-enforced |
| Approval integrity | Single-use token bound to `hash(tool, args, actor)` | Replay → `already_applied` |
| Duplicate actions | Idempotency key from approval token | Verified in Demo 4 |
| Audit | Append-only table, secrets redacted | Every consequential action |

### L9 — Observability & evaluation

| Concern | Technology | Status |
| --- | --- | --- |
| Logging | Structured JSON, request/user/conversation correlation | Done |
| Metrics | `/metrics`: latency percentiles, cache, **token budget**, MCP stats, authz denials | Done |
| Tracing | In-process spans | OTel exporter **not wired** |
| Dashboards | — | Prometheus/Grafana **not built** |
| Retrieval eval | Span-based qrels, per-class nDCG, bootstrap CIs | **Measured** |
| Generation eval | LLM judge + κ | **Not yet measured** |

---

## 4. The one design decision worth singling out

**Identity across the MCP boundary.** Get this wrong and every other control is
decoration.

The naive port exposes `get_ticket(ticket_id, employee_id)`. The model writes
`employee_id`, so "create a high-priority ticket for another employee" (spec §17)
succeeds at the data layer no matter what the graph believes.

Two safe designs, and why the second won:

| | Identity pinned to process (env var) | **Signed grant per call (chosen)** |
| --- | --- | --- |
| Spoofable by the model? | No | No — forging needs the HMAC secret |
| Processes needed | One **per (domain, user)** | One **per domain** |
| Cold start | **4–7 s for every new user** | 3.5–6 s once, then shared |
| Warm latency | 8 ms | **7–36 ms, any user** |
| Matches HTTP MCP? | No | Yes — the grant is what an `Authorization` header carries |

The invariant is not "identity must never appear in arguments". It is **identity
must never be model-controlled**. A plain field fails that; an HMAC-signed,
expiring grant minted server-side from the authenticated session does not.

Measured: a grant tampered to `role: IT_admin` is rejected; a call with no grant
is rejected; the same warm process serves an `employee` (denied) and an
`IT_admin` (121 devices) correctly.

---

## 5. Cost of each boundary (measured, not estimated)

| Path | Latency | Note |
| --- | --- | --- |
| In-process tool call | ~0.3 ms | Direct function call |
| MCP call, warm | 7–36 ms | Serialisation + stdio round trip |
| MCP call, cold | 3.5–6 s | One-time subprocess spawn per domain |
| Retrieval (hybrid + rerank) | p95 9.7 ms | 363 chunks, CPU |
| LLM routing call | ~0.6 s | gpt-oss-20b |
| LLM generation | ~1.3 s | gpt-oss-120b |
| **End-to-end, warm** | **~2 s** | knowledge answer |
| **End-to-end, cold MCP** | **~12.7 s** | first multi-tool investigation |

**MCP costs roughly 25× a function call when warm and is still trivially fast.**
The cold start dominates and is one-time per domain. That is the honest price of
protocol interoperability, and it is a price, not a benefit.

---

## 6. The binding operational constraint

Groq free tier, **measured**: 30 RPM · 1,000 RPD · **8,000 TPM** · 200,000 TPD.
At ~2,400 tokens per grounded answer that is **~3 answers per minute**.

Demonstrated live: eight requests in one minute hit `tokens_this_minute:
7722/8000` and the budget guard returned a clean **503 before Groq could 429**.
The demo script now paces itself against `/metrics`.

This single constraint shaped the architecture more than any preference:

1. **Retrieval evaluation is LLM-free** — Recall/MRR/nDCG need no generation, so
   the most rigorous measurement in the project costs nothing and runs unlimited.
2. **Caching is justified by quota, not latency** — a hit returns budget.
   Metrics are `llm_calls_avoided` and `tokens_saved`, both countable.
3. **Models are tiered by task** — 20b for routing, 120b only for the final answer.
4. **Reasoning effort is capped** — `gpt-oss` reasoning tokens bill against the
   same ceiling.
5. **The budget is enforced before the call**, not discovered as a 429 in front
   of a user.
