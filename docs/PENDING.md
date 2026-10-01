# Pending work — what is blocked, why, and what exists in its place

Three categories, kept separate because they need different things from you:

- **A. Blocked by this machine / network** — code exists, cannot be *verified* here.
- **B. Blocked by budget** — needs more API quota than the free tier allows.
- **C. Not started** — sequenced later by the spec's own ordering, or simply not done yet.

Nothing in here is hidden in the reports. Every artifact that is unverified says
so in its own header.

---

## A. Blocked by this machine or network

### A1. Neural embeddings — **the single biggest quality gap**

| | |
| --- | --- |
| **Blocked by** | Corporate proxy rejects HuggingFace model-file downloads. `huggingface.co` returns 200; the file CDN fails TLS handshake (`SSLV3_ALERT_HANDSHAKE_FAILURE`). |
| **Placeholder in place** | `HashingEmbedder` — real, deterministic, 8,192-dim hashed n-grams. Not a stub: it produces genuine vectors and genuine scores. |
| **Real code already written** | `OnnxEmbedder` in `opspilot/rag/embeddings.py`. ONNX Runtime 1.30 **already installs and imports** on Python 3.14 — only the weights are missing. |
| **How to unblock** | (a) download `BAAI/bge-small-en-v1.5` ONNX on any unblocked network and drop it in `~/.cache/fastembed`; or (b) ask IT to allow `cas-server.xethub.hf.co` and `cdn-lfs*.hf.co`. |
| **Then** | `OPSPILOT_EMBEDDING_BACKEND=onnx` and re-run `scripts/build_index.py`. |
| **Measured impact of the gap** | Paraphrase (S-class) nDCG: dense **0.256** vs BM25 **0.261** — the dense arm adds nothing, because hashing captures spelling, not meaning. This is also why RRF fusion currently *hurts* (E3 0.372 < E2 0.406). |

### A2. Cross-encoder reranker

| | |
| --- | --- |
| **Blocked by** | Same model-download block. |
| **Placeholder in place** | `LexicalReranker` (`lexical-v2`) — feature-based: IDF-weighted coverage, identifier match, status recency, product agreement. It is currently the **largest measured win** in the system (+0.139 nDCG). |
| **Real code already written** | `CrossEncoderReranker` in `opspilot/rag/retrieval.py`. |
| **Unknown** | **CPU latency is `Not yet measured.`** 30 pairs through a cross-encoder on a U-series CPU is the least certain number in the design and may blow the 900 ms budget. Measure before adopting. |

### A3. Docker Compose — 9 services

| | |
| --- | --- |
| **Blocked by** | No Docker Desktop, no WSL2, no admin rights. |
| **Placeholder in place** | `docker-compose.yml` — complete topology (postgres, redis, qdrant, backend, frontend, 4 MCP services commented), health checks, dependency ordering. **Never executed.** |
| **Honest status** | Authored and reviewable. Unverified. Labelled as such in the file header. |
| **To verify** | Any machine with Docker: `docker compose up`. Expect to fix small things — untested YAML always has some. |

### A4. Kubernetes

| | |
| --- | --- |
| **Blocked by** | No cluster, and no Docker to build images. |
| **Placeholder in place** | `infra/kubernetes/backend.yaml` — Deployment, Service, HPA, ConfigMap, readiness/liveness probes, resource requests/limits, non-root securityContext, read-only rootfs. |
| **Design note worth keeping** | Readiness gates on `/ready`, which returns 503 until the **index is built** — serving traffic before that would answer confidently with no evidence. |
| **Honest status** | Never applied to any cluster. Spec §31 explicitly forbids calling this production-grade. |

### A5. Real PostgreSQL and Redis

| | |
| --- | --- |
| **Blocked by** | Both need a server; no Docker. |
| **Placeholder in place** | SQLite via SQLAlchemy with a **PostgreSQL-portable schema** (explicit lengths, no SQLite-only types). In-process cache and rate limiter behind Redis-shaped interfaces. |
| **Honest limits** | Cache is lost on restart and not shared across replicas. Rate limiter is per-process, so it is **not a real limit behind a load balancer**. SQLite is single-writer, so concurrency is untested. |
| **To unblock** | Set `OPSPILOT_DATABASE_URL` / `OPSPILOT_QDRANT_URL`. No code change. |

### A6. Load testing (spec §34)

| | |
| --- | --- |
| **Blocked by** | Two things. Locust would run, but (a) 8,000 tokens/minute means ~3 real answers/minute, so 250 concurrent users is impossible against the live provider; (b) testing against the fake provider measures FastAPI, not the system. |
| **What is honest** | Load-test the **retrieval path only** (no LLM) — that is meaningful and affordable. Full-stack load testing needs a paid tier. |
| **Status** | Not written. |

---

## B. Blocked by API budget

### B1. Full-set judged evaluation

Measured free tier: **8,000 TPM / 200,000 TPD**. A judged query costs 1 generation
+ 3 judge calls ≈ 4 model calls.

| Scope | Model calls | Feasible? |
| --- | ---: | --- |
| Smoke (n=8) | ~19 | Yes — ran in 190 s |
| Current run (n=14) | 34 | Yes — ran in 377 s |
| Meaningful (n=60) | ~200 | ~1 hour of pacing; one run/day |
| Full set (n=173) | ~600 | Multiple days |

**Currently measured:** faithfulness 1.000 (n=6, **UNTESTED** — no negative labels),
answer relevance 0.643 [0.357, 0.857] (n=14, kappa 1.00, reliable),
context precision 0.571 [0.334, 0.798] (n=14, unvalidated).

### B2. Human labels for kappa

`evaluation/human_labels.json` currently holds 14 labels **annotated by the
assistant, not an independent human**. That is recorded in the file and reprinted
in every report that cites kappa. Faithfulness needs negative examples before its
kappa means anything.

### B3. Experiment 5 — query rewrite

Spec §20 lists it. Needs one LLM call per query across the eval set. Not run.

---

## C. Not started

| Item | Spec | Why not |
| --- | --- | --- |
| **Next.js + TypeScript + Tailwind frontend** | §24 | A working single-file UI exists (sign-in, chat, tickets, incidents, evaluation, system status). The API contract is stable at `/docs`. This is a rewrite in a different toolchain, not new capability. |
| **CI/CD (GitHub Actions)** | §33 | Not a git repository yet. The 42 tests exist but nothing enforces them. **Highest-value cheap win.** |
| **Cloud deployment guide** | §32 | Depends on A3/A4 being verified first. |
| **OpenTelemetry + Prometheus + Grafana** | §21 | Structured logs, spans and `/metrics` JSON exist. Wiring an exporter is mechanical. |
| **Long-term memory summarisation** | §14 | Short-term memory (bounded tail) works. Summarisation costs an LLM call per conversation — budget. |
| **Real authentication (OIDC/JWT)** | §15 | Signed session tokens with expiry and server-side verification now exist, but **passwordless**. `issue_token`/`verify_token` are the only functions an IdP swap would replace. |
| **Semantic caching** | §22 | Deliberately deferred: risks serving a subtly wrong cached answer to a differently-worded question. Needs its own false-hit evaluation first. |

---

## Recommended order

| # | Work | Cost | Unblocks |
| --- | --- | --- | --- |
| 1 | **Copy in the ONNX embedding model** | 10 min on an unblocked network | A1, A2 — the biggest quality gap, and likely flips the RRF result |
| 2 | **`git init` + GitHub Actions** (lint + 42 tests) | 1 hour | Regression safety; tests already exist |
| 3 | **Expand judged eval to n=60 + label negatives** | ~1 hour/day of budget | Makes faithfulness testable rather than UNTESTED |
| 4 | **Verify Compose on any Docker machine** | 1 hour | A3, then A4, then A6 |
| 5 | **OIDC replacing the session shim** | 1 day | The one genuinely weak security control |
| 6 | **OTel exporter → Prometheus → Grafana** | 1 day | §21 |
| 7 | **Next.js frontend** | 2–3 days | §24 |
