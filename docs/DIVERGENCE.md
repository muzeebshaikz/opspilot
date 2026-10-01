# Divergence from the original specification

**Question being answered:** how far has this implementation drifted from
`prompt.docx`, and is that drift *significant* and *meaningful*?

**Short answer.** Roughly **80% of the specification is implemented as written**.
Every divergence falls into one of three buckets, and only one bucket is
architectural:

| Bucket | Count | Is it significant? |
| --- | ---: | --- |
| **A. Forced by the environment** — no Docker, no GPU, blocked model downloads, an 8k-TPM API ceiling | 6 | Significant in *deployment*, not in *design*. The code paths exist and are selectable by config. |
| **B. Corrections to the spec** — places the spec was under-specified or would have produced unfalsifiable results | 9 | **Meaningful and deliberate.** These are improvements, argued in `docs/audit/SPEC-AUDIT.md`. |
| **C. Sequenced, not yet built** — spec §45 explicitly forbids building these first | 5 | Not divergence. This is the spec's own ordering. |

Nothing was silently dropped. Nothing is claimed that was not measured.

---

## 1. Section-by-section status

Legend: ✅ implemented · ◐ adapted (bucket A/B) · ⏳ sequenced (bucket C) · ❌ not done

| § | Requirement | Status | Note |
| --- | --- | :-: | --- |
| 1 | Product identity, synthetic Acme world | ✅ | 240 docs, 116 users, 121 assets, 622 tickets, 4 incidents |
| 2 | ChatGPT-style UX, example questions | ◐ | Working chat UI served at `/`; **Next.js app not built** (§24) |
| 3 | ADR per technology | ◐ | Delivered as one consolidated `TECHNOLOGY-RATIONALE.md` with 18 comparison tables rather than 9 separate ADR files. Same content, one place. Splitting into `ADR-00x.md` is mechanical. |
| 4 / 45 | V0→V7 incremental evolution | ✅ | V0–V6 done; V7 (MCP) sequenced |
| 5 | Synthetic data model | ✅ | All fields present, plus deliberate edge cases (contractor, on-leave, offboarded, 0-device and 2-device users) |
| 6 | 100–300 knowledge documents | ✅ | 240. **Composition is designed**, not just counted — see §2 below |
| 7 | Public data for queries/intents only | ◐ | Datasets surveyed and licence-verified (`docs/data/DATA-STRATEGY.md`); **not yet wired in**. Nothing public is in the vector store, as required. |
| 8 | Ingestion pipeline, no naive chunking | ✅ | Structure-aware + fixed control arm; stable IDs within strategy |
| 9 | Full RAG architecture with metadata filters | ✅ | Dense + sparse → RRF → top-30 → rerank → top-6 → context → LLM |
| 10 | Grounding and abstention | ✅ | **Calibrated from a measured sweep**, not guessed |
| 11 | Citations, no fabrication | ✅ | Only *referenced* citations returned |
| 12 | Model abstraction, no hard-coded keys | ✅ | `LLMProvider` / `Embedder` / reranker all swappable by env |
| 13 | PostgreSQL + Qdrant + Redis | ◐ | Qdrant ✅ (embedded). Postgres → **SQLite**, Postgres-portable schema. Redis → **in-process cache**, same interface. Bucket A. |
| 14 | Short + long-term memory | ◐ | Short-term ✅ (bounded tail, not full history). Long-term summarisation **not implemented** |
| 15 | AuthN/AuthZ, RBAC, injection defence, audit, rate limiting | ✅ | 5/5 authorization probes pass. AuthN is a **dev shim**, labelled as such |
| 16 | Human-in-the-loop approval | ✅ | Plus single-use tokens, TTL and **idempotency** (spec omitted this) |
| 17 | Prompt-injection test set | ✅+ | **Extended to indirect injection**, which the spec did not ask for |
| 18 | Tool security | ✅ | Validate → authorize → resource-check → execute → audit |
| 19 | Evaluation datasets and metrics | ◐ | Retrieval metrics ✅ measured. **RAG metrics (faithfulness etc.) not measured** — need a judge model |
| 20 | Experiment framework E1–E5 | ◐ | E1–E4 ✅ measured with CIs. **E5 (query rewrite) not run** |
| 21 | Observability, OTel, Prometheus, Grafana | ◐ | Structured logs, spans, `/metrics` JSON ✅. **No OTel exporter, no Prometheus, no Grafana** |
| 22 | Redis with measured cache benefit | ◐ | Mechanism + accounting ✅. In-process, not Redis. **Hit rate in real use not measured** |
| 23 | FastAPI endpoints incl. SSE | ✅+ | All specified endpoints, **plus** `/api/v1/approvals/{token}` |
| 24 | Next.js + TypeScript + Tailwind | ❌ | Single-file UI instead; scaffold documented |
| 25 | Admin/evaluation UI | ◐ | `GET /api/v1/evaluation` returns measured JSON or literally `"Not evaluated yet"` |
| 26 | Repository structure | ◐ | Equivalent, flatter. `opspilot/` package rather than `backend/` |
| 27 | Code quality | ✅ | Type hints, Pydantic, module boundaries, structured logging |
| 28 | Unit / integration / E2E tests | ✅ | 32 tests, 1.94 s, **zero API calls** + 9/9 E2E demo scenarios |
| 29 | Failure handling | ◐ | Retries with backoff, quota guard, abstention, tool errors ✅. **No provider fallback chain** |
| 30 | Docker Compose, 9 services | ◐ | Authored, **never run**. Bucket A |
| 31 | Kubernetes manifests | ◐ | Authored (Deployment/Service/HPA/ConfigMap/probes/limits), **never applied**. Bucket A |
| 32 | Cloud deployment guide | ❌ | Not written |
| 33 | CI/CD GitHub Actions | ❌ | Not written |
| 34 | Load testing (Locust, 10–250 users) | ❌ | Not run |
| 35 | Seven demo scenarios | ✅+ | **9/9 pass**, including two extra security scenarios |
| 36 | README | ✅ | Written |
| 37 | Interview-first documentation | ✅ | 18 comparison tables answering exactly the §37 questions |
| 38 | No fake claims | ✅+ | **Strengthened**: added the rule that an improvement whose CI crosses zero is reported as "no measurable difference" |
| 39 | Per-phase workflow | ✅ | Each phase measured before moving on |
| 40 | Git workflow | ❌ | Not a git repository yet |
| 41 | Definition of Done | ◐ | 22/29 items complete |
| 42 | First action: inspect, plan, scaffold, stop | ✅ | Audit delivered before code |
| 43 | Final architecture | ◐ | As specified minus MCP layer |
| 44 | The engineering story | ✅ | Baseline → hybrid → rerank → state → tools → orchestration, each measured |

**Totals:** ✅ 19 · ✅+ 5 · ◐ 14 · ⏳/❌ 6

---

## 2. Bucket B — the meaningful divergences (deliberate corrections)

These are the ones worth defending in an interview, because each one is a place
the spec as written would have produced a *worse or unfalsifiable* result.

### B1. Gold labels key to document spans, not chunk IDs

Spec §8 demands stable chunk IDs; §20 demands comparing chunking strategies.
**These are incompatible** — re-chunking invalidates every chunk-ID-keyed label,
silently. Labels now key to `(document_id, char_start, char_end)` with an overlap
test. Enforced by `test_span_overlap_survives_rechunking`.
*Significance: high.* Without it the evaluation set rots the first time anyone
touches the chunker, and nobody notices.

### B2. Corpus composition is designed around retrieval failure modes

Spec §6 asks for "100–300 documents". On a small, clean, single-generator corpus
dense retrieval saturates and **hybrid/reranking have no headroom to demonstrate
anything**. So each document family exists to induce a specific phenomenon, and
every query carries a difficulty tag (L/S/V/P/D/M/A/X).
*Significance: high.* This is what turns "hybrid nDCG 0.81 vs 0.79" (noise) into
"BM25 beats dense on exact identifiers 0.654 vs 0.315" (a finding).

### B3. Every comparison reports a bootstrap CI; no CI-crossing-zero is a win

Spec §38 forbids fabricating benchmarks but says nothing about statistical
significance, so comparing point estimates on ~170 queries would quietly
manufacture conclusions.
*Significance: high, and it changed a conclusion.* It is why this repo reports
**"RRF fusion made things worse"** (CI `[-0.052, -0.015]`) instead of quietly
presenting hybrid as the win the spec expected.

### B4. Metrics are role-scoped

Documents carry `access_level`. A gold label pointing at a document the asking
role cannot see makes the task impossible and drives recall to zero.
*Significance: high — this was a live bug.* Diagnostics showed V-class queries
scoring 0.000 because the gold document was `IT_admin`-only while the query was
labelled `employee`. **The label was wrong, not the retriever.**

### B5. Abstention is a confusion matrix, and the threshold is swept

Spec §10 requires abstention; §19 lists an out-of-domain set. Together they are
gameable — always abstaining scores perfectly. Now measured as paired error
rates and the operating point is *chosen from data*.
*Significance: high, and it caught a real defect.* The guessed threshold gave a
54.5% hallucination-risk rate and the system confidently answered a question
about a topic that does not exist in the corpus.

### B6. Indirect prompt injection is in scope

Every §17 example is a direct user-turn attack. The realistic attack is through
**ticket descriptions** — an attacker-controlled field that an IT_agent later
reads with elevated privilege. Added `TrustTier`, structural isolation, and the
rule that tool arguments derive only from the user turn.
*Significance: high.* The spec's own data model creates this hole.

### B7. Identity is never a tool parameter

If `employee_id` were an argument, the model would control it and "create a
ticket for another employee" (spec §17) would succeed at the data layer. Enforced
structurally and by test.
*Significance: high.*

### B8. Approvals are single-use, argument-bound and idempotent

Spec §16 (approval) + §29 (retries) compose into duplicate ticket creation. The
spec never mentions idempotency.
*Significance: medium-high.* Verified by Demo 4's replay assertion.

### B9. SSE and human-in-the-loop are reconciled

§23 wants streaming, §16 wants a pause. The spec never says how a stream pauses.
Resolution: stream emits `approval_required` and closes; approval is a separate
POST that resumes the checkpointed graph.
*Significance: medium.* It is also the strongest justification for LangGraph.

---

## 3. Bucket A — forced by the environment

| # | Spec assumption | Reality (measured) | Adaptation | Reversible? |
| --- | --- | --- | --- | --- |
| A1 | `docker compose up`, 9 services | No Docker, no WSL2, no admin rights | Embedded Qdrant + SQLite + in-process cache. Compose/K8s authored, labelled unverified | **Yes** — config change |
| A2 | GPU for reranking | Intel iGPU, no CUDA | Feature-based reranker; ONNX cross-encoder implemented but unused | **Yes** |
| A3 | Local or API models freely | Groq free tier: 8k TPM / 200k TPD ≈ **55 answers/day** | Retrieval eval made LLM-free; quota guard; task-tiered models; caching for quota | Partly — tier limits are external |
| A4 | HuggingFace models downloadable | **Corporate proxy rejects the model CDN** (TLS handshake failure) | Hashing embedder default; ONNX path implemented and selectable | **Yes** — unblock the CDN or copy files in |
| A5 | Standard Python ML stack | Python 3.14; torch CPU-only, sentence-transformers undeclared | ONNX Runtime instead of PyTorch (verified working) | n/a — this is strictly better |
| A6 | API endpoints reachable | `api.groq.com` fails certifi verification | `truststore.inject_into_ssl()` → 200 OK | n/a — required fix |

**A4 is the one that most depresses the headline numbers.** The dense arm is a
hashing baseline, not a neural encoder, which is why S-class shows no dense
advantage and why RRF underperforms BM25. This is a *measurement* limitation
being reported honestly, not an architectural flaw — the ONNX path is written
and one config flag away.

---

## 4. Bucket C — sequenced, not skipped

Spec §45: *"Do NOT start by implementing MCP, multi-agent complexity or
Kubernetes."*

| Item | Precondition | Status |
| --- | --- | --- |
| MCP servers (§7/V7) | "Only after the internal tools work" | 13 typed tools ✅ done and proven. MCP is next. |
| Kubernetes (§31) | Working local system | Manifests authored, honestly labelled unverified |
| CI/CD (§33) | A git repository | Not initialised yet |
| Load testing (§34) | A deployable target | Not run |
| Next.js frontend (§24) | A stable API contract | Contract stable at `/docs`; scaffold documented |

Building any of these first would have inverted the spec's own ordering.

---

## 5. Is the divergence significant? A direct answer

**On design intent: no.** The engineering story the spec asks for in §44 is
fully intact, and every step of it is measured:

> baseline → hybrid → reranking → conversation state → typed tools →
> orchestration → authorization and approval → evaluation → observability

**On deployment maturity: yes, materially.** There is no container that has ever
been built, no cluster, no CI, no load test. The system runs on one machine, in
one process, against SQLite. Calling it "production-grade" would violate §38.
It is accurately described as **a production-*shaped* system running in a
development configuration**.

**On evaluation completeness: partially.** Retrieval is measured rigorously —
with per-class breakdowns, confidence intervals and a calibrated abstention
operating point. Generation-side quality (faithfulness, answer relevance) is
**not measured at all**, because it needs a judge model and its κ calibration.
That gap is stated in every report rather than filled with plausible numbers.

### What an enterprise reviewer would still hold against it

1. **No neural embeddings** — the dense arm is a hashing baseline. Biggest quality gap.
2. **No faithfulness measurement** — the generation half of RAG is unevaluated.
3. **Authentication is a header shim** — authorization is real; authentication is not.
4. **Single-instance everything** — cache and rate limiter are in-process; no horizontal scaling is demonstrated.
5. **No CI** — nothing prevents a regression from landing.
6. **37% false-abstention rate** — the safety-first operating point is currently expensive in usability.
7. **Synthetic corpus is self-consistent** — one generator wrote it, so it is more internally coherent than a real enterprise wiki. All scores are optimistic relative to real data.

Each of these is in the reports, not buried here.

---

## 6. Recommended next order of work

| # | Work | Why it is first | Blocked by |
| --- | --- | --- | --- |
| 1 | Unblock HuggingFace model files, switch to `bge-small` + cross-encoder | Lifts S-class, likely flips the E2-vs-E3 fusion result, and should cut false abstention | Network policy, or manual file copy |
| 2 | Add the LLM-judge harness with κ calibration | Closes the single largest evaluation gap | Groq key in `.env` |
| 3 | MCP servers over the existing service layer | Next spec phase; precondition already met | — |
| 4 | `git init` + GitHub Actions running lint + the 32 tests | Tests exist but nothing enforces them | — |
| 5 | Real auth (JWT/OIDC) replacing the header shim | The one genuinely weak security control | — |
| 6 | Long-term memory summarisation (§14) | Only partly implemented | Quota |
| 7 | OTel exporter + Prometheus + Grafana | Metrics exist as JSON; wiring is mechanical | — |
| 8 | Load test, then Docker/K8s verification | Cannot be done honestly on this machine | Docker |
