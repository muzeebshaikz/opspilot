# OpsPilot AI

An enterprise IT-operations copilot for a fictional company, **Acme Technologies**.
Ask → Investigate → Verify → Explain → Act safely.

A **retrieval-grounded, tool-using copilot** with server-side authorization and
human-in-the-loop approval, where tools are reached over MCP. All enterprise data
is synthetic; no real company system is touched.

> **Every number in this README came from an actual run of this code.** Anything
> not measured says `Not yet measured.` See [`docs/DIVERGENCE.md`](docs/DIVERGENCE.md)
> for what works, what does not, and what is unverified.

---

## 1. Requirements

| | Minimum | Notes |
| --- | --- | --- |
| Python | 3.12+ | Verified on **3.14**. No PyTorch needed — embeddings use ONNX Runtime. |
| Node | 22+ | Only for the Next.js frontend. The backend serves its own UI without it. |
| RAM | 4 GB | |
| API key | optional | **Not required.** The default provider is deterministic and offline. |
| Docker | not required | Everything runs without it. |

---

## 2. Quickstart (5 minutes, no API key)

```bash
python -m venv .venv
```

```bash
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Generate the corpus, build the index, and run the retrieval experiments:

```bash
./.venv/Scripts/python.exe -m scripts.build_index
```

Start the server, then open <http://127.0.0.1:8000>:

```bash
./.venv/Scripts/python.exe -m uvicorn opspilot.api.main:app --port 8000
```

You will get a sign-in screen. Pick any account — **Priya Nair** (employee) and
**Reshma Iyer** (IT_admin) show the RBAC difference most clearly.

On Linux/macOS use `.venv/bin/python` instead of `./.venv/Scripts/python.exe`.

---

## 3. Enabling the live model (optional)

```bash
cp .env.example .env
```

Put your Groq key in `.env` (**not** `.env.example` — that file is committed):

```
OPSPILOT_LLM_BACKEND=groq
OPSPILOT_GROQ_API_KEY=gsk_...
```

Restart the server. `/ready` should now report `"provider": "cached:groq"`.

> **Corporate networks:** if `api.groq.com` fails certificate verification, that
> is TLS interception. `truststore` is already wired in and uses the OS
> certificate store, which carries the corporate root CA. Without it, the failure
> looks exactly like a bad API key.

---

## 4. Running the tests

```bash
./.venv/Scripts/python.exe -m pytest -q
```

**42 tests, ~4 seconds, zero API calls.** The deterministic fake provider means
the suite is free and reproducible offline.

Lint and format (what CI enforces):

```bash
./.venv/Scripts/python.exe -m ruff check . && ./.venv/Scripts/python.exe -m ruff format --check .
```

### The full evaluation suite

```bash
./.venv/Scripts/python.exe scripts/run_all_evals.py --base http://127.0.0.1:8000
```

Seven stages; **five need no model access at all**:

| # | Stage | API calls | Command |
| --- | --- | --- | --- |
| 1 | Unit + integration (42 tests) | none | `pytest -q` |
| 2 | Retrieval experiments E1–E4 | none | `python -m scripts.build_index` |
| 3 | Abstention calibration sweep | none | `python scripts/calibrate_abstention.py` |
| 4 | Adversarial + authorization probes | none | `python scripts/run_adversarial.py` |
| 5 | MCP layer (7 checks) | none | `python scripts/test_mcp.py` |
| 6 | End-to-end scenarios (9) | live | `python scripts/demo_scenarios.py` |
| 7 | Judge calibration | none (replays) | `python scripts/calibrate_judge.py` |

Generation quality (spends quota, paces itself against the live budget):

```bash
./.venv/Scripts/python.exe scripts/run_generation_eval.py --limit 14
```

Reports land in `artifacts/reports/`; the roll-up is `EVALUATION_SUMMARY.md`.

---

## 5. The frontend (optional)

The backend already serves a full UI at `/`. The Next.js app is the spec'd
frontend:

```bash
cd frontend && npm install && npm run dev
```

Open <http://localhost:3000>. It proxies `/api/v1/*` to the backend, so no CORS
configuration is needed. `npm run typecheck`, `npm run lint` and `npm run build`
all pass.

---

## 6. What it does

| Capability | How |
| --- | --- |
| Answers IT questions | Hybrid retrieval → reranking → grounded generation with citations |
| Investigates a live problem | Fans out to knowledge + active incidents + the caller's own device |
| Reads tickets and incidents | 16 typed tools, schema-validated and authorized |
| Acts safely | Authorize → human approval → idempotent execution |
| Refuses when it should | Calibrated abstention; answering with no evidence is a failure |
| Resists prompt injection | Trust tiering + server-side authorization the model cannot influence |
| Exposes tools over MCP | 4 MCP servers over the same service layer |

**Three distinct outcomes**, never conflated:

| Outcome | Meaning |
| --- | --- |
| `answered` | Grounded in retrieved evidence, with citations |
| `abstained_no_evidence` | Evidence was insufficient — refuses rather than guessing |
| `denied_authorization` | The data exists; your role may not see it |

---

## 7. Measured results

446 documents, 695 chunks, 311 scored queries. 42 topics, exactly 3 per category.

### Retrieval (k = 10)

| Experiment | Recall | Precision | MRR | nDCG | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 Dense only | 0.660 | — | 0.414 | 0.311 | 27.3 ms |
| E2 Sparse (BM25) | 0.739 | — | 0.450 | 0.384 | 2.8 ms |
| E3 Hybrid (RRF) | 0.718 | — | 0.455 | 0.359 | 30.9 ms |
| **E4 Hybrid + reranker** | **0.813** | — | **0.667** | **0.482** | 35.7 ms |

**Three findings worth reading:**

1. **BM25 beats dense 2× on exact identifiers** (0.654 vs 0.315 on error-code
   queries). Embeddings smear rare tokens; lexical matching does not.
2. **Fusion helps the candidate pool, not the ranking — and the distinction
   matters.** Judged on fused *ordering*, hybrid looks worse than BM25 alone
   (0.372 vs 0.406). Judged on **candidate-pool recall@30**, which is what
   fusion actually feeds the reranker, hybrid is **best at 0.914** vs 0.887 for
   sparse alone, and the dense arm contributes **14 gold documents that BM25
   never retrieves**. An earlier version of this README reported only the first
   half and drew the wrong conclusion.
3. **Reranking is the largest win** (+0.139 nDCG, CI `[+0.110, +0.166]`), driven
   by version-staleness (0.133 → 0.519) and near-duplicate distractors
   (0.191 → 0.416).

### Abstention — three tiers, not two

A binary answer/refuse gate bounced 37% of answerable questions. Adding a
**partial-evidence** tier — a hedged, cited "related material, not a confirmed
answer" — cut that by more than half at **zero safety cost**:

| | Two-tier (before) | Three-tier (now) |
| --- | ---: | ---: |
| Hard refusal on answerable questions | 37.0% | **17.7%** |
| Partial (hedged, cited) | — | 35.7% |
| Full confident answer | 63.0% | 46.6% |
| **Confident but wrong** | 12.1% | **6.7%** |

Better on *both* axes than the original: half the confident-wrong rate and half
the refusal rate. The cost is that more answers are explicitly hedged.

> **The threshold is a property of the corpus, not a constant.** Widening the
> corpus from 240 to 446 documents pushed confident-wrong from 12.1% to **26.7%**
> at the old setting — more documents means more chances an unanswerable
> question finds something above the bar. It had to be re-swept. A safety
> ceiling is now enforced in CI so this cannot regress silently again.

The confident-wrong rate is unchanged *by construction*: the bar for a full
answer never moved. And 30 of the 33 remaining hard refusals are paraphrase
(S) and multi-hop (M) queries — the classes that need semantic matching, and so
are blocked on the embedding model rather than on the gate.

### Security

- **5/5 authorization probes pass**, including where injection detection misses entirely
- Injection detection: regex **78.1%** recall; **+ Prompt Guard 2 → 81.2%**, at the cost of **+14pp false positives**
- MCP identity grants: forged, tampered and expired all rejected
- **9/9 end-to-end scenarios** pass against live Groq + MCP

### Generation (LLM-as-judge, sampled n=14)

| Metric | Mean | 95% CI | κ | Status |
| --- | ---: | --- | ---: | --- |
| faithfulness | 1.000 | [1.000, 1.000] | — | **UNTESTED** |
| answer_relevance | 0.643 | [0.357, 0.857] | 1.00 | reliable |
| context_precision | 0.571 | [0.334, 0.798] | — | unvalidated |

`UNTESTED` is the important one: all faithfulness labels were positive, so chance
agreement is 1.0 and κ is **mathematically undefined**. Reporting "1.00, reliable"
there would look like perfect validation while meaning the judge was never challenged.

### MCP overhead

| Path | Latency |
| --- | --- |
| In-process tool call | ~0.3 ms |
| MCP warm | 7–36 ms |
| MCP cold (one-time per domain) | 3.5–6 s |

---

## 8. Architecture

```
Browser ──SSE──► FastAPI ──► session ──► LangGraph ──► classify ──► route
                                                                     │
        ┌────────────────────────┬───────────────────┬───────────────┤
        ▼                        ▼                   ▼               ▼
   Hybrid RAG              MCP servers          authorize      clarify
   dense + BM25         ticketing · incident         │
        │ RRF            asset · knowledge      human approval
        ▼                        │                   │
   rerank → top 6                ▼                   ▼
        │                 service layer       idempotent execute
        ▼                  (16 tools)
  evidence gate                  │
   ├─ sufficient → generate + ground-check → answer + citations
   └─ insufficient → abstain

  Qdrant (embedded) · SQLite (Postgres-shaped) · in-process cache
  Prometheus /metrics/prometheus · OpenTelemetry spans · audit log
```

Full diagrams and decision tables: [`docs/DESIGN.md`](docs/DESIGN.md).

---

## 9. Documentation

| Document | Contents |
| --- | --- |
| [`docs/DESIGN.md`](docs/DESIGN.md) | **Start here.** Architecture diagrams, decisions, request flow |
| [`docs/audit/SPEC-AUDIT.md`](docs/audit/SPEC-AUDIT.md) | Where the original spec was insufficient or contradictory |
| [`docs/architecture/TECHNOLOGY-RATIONALE.md`](docs/architecture/TECHNOLOGY-RATIONALE.md) | 18 comparison tables — why each technology beat its alternatives |
| [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md) | Layer-by-layer technology map |
| [`docs/data/DATA-STRATEGY.md`](docs/data/DATA-STRATEGY.md) | Corpus design, dataset licences, gold-label protocol |
| [`docs/DIVERGENCE.md`](docs/DIVERGENCE.md) | Status against all 44 spec sections |
| [`docs/PENDING.md`](docs/PENDING.md) | What is blocked, why, and what stands in its place |
| [`docs/deployment/cloud.md`](docs/deployment/cloud.md) | Deployment procedure (never executed) |

---

## 10. Repository layout

```
opspilot/            backend package
  api/               FastAPI app + built-in UI
  agents/            LangGraph workflow
  rag/               chunking · embeddings · sparse · retrieval · generator
  mcp/               MCP servers + client gateway
  eval/              metrics · experiments · abstention · adversarial · judge
  data/              synthetic corpus + eval-set generators
  tools.py           16 typed tools
  security.py        RBAC · injection defence · approval tokens
  auth.py            session tokens
  memory.py          short-term + long-term memory
  observability.py   OpenTelemetry + Prometheus
frontend/            Next.js + TypeScript + Tailwind
scripts/             build, evaluate, calibrate, CI gates
tests/               42 tests
infra/               docker · kubernetes · monitoring   (authored, not run)
artifacts/reports/   generated evaluation reports
```

---

## 11. CI

`.github/workflows/ci.yml` runs on every push: **lint** (ruff check + format),
**tests** (3.12 and 3.13), **offline evaluation** with a regression gate,
**security** (dependency audit + secret scan), **frontend** (typecheck/lint/build)
and **docker** build. Deployment is gated behind all of them.

CI makes **zero API calls** — `OPSPILOT_LLM_BACKEND=fake`.

`scripts/check_regressions.py` fails the build if measured retrieval quality drops
below a floor. Tests prove the code runs; this proves it still works *well*.

---

## 12. Limitations

1. **The dense arm is a hashing baseline, not a neural encoder.** The corporate proxy blocks HuggingFace model files. The ONNX path is implemented — `OPSPILOT_EMBEDDING_BACKEND=onnx`. All dense and hybrid numbers understate the architecture.
2. **Faithfulness is UNTESTED** — no negative human labels yet.
3. **Authentication is passwordless.** Session tokens are real (signed, expiring, verified); identity proof is not.
4. **Docker, Kubernetes and the monitoring stack have never been run.** Authored and labelled unverified.
5. **Single-instance only.** Cache and rate limiter are per-process, so rate limiting is not a real limit behind a load balancer.
6. **No load test.**
7. **Groq free tier is the binding constraint** — 8,000 tokens/minute ≈ 3 answers/minute. Demonstrated live: 8 requests in a minute hit the guard, which returned 503 *before* Groq could 429.
8. **The synthetic corpus is more self-consistent than a real enterprise wiki**, so retrieval scores are optimistic relative to real data.
