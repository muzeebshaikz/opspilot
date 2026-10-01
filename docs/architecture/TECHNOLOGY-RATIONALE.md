# Technology Rationale — why each choice, and what it cost

Spec §3 and §37 require, for every significant component: what problem it
solves, what alternatives were considered, why this one, what trade-off was
accepted, and what would make us replace it. This document answers that for
every decision in the system.

**Reading rule.** Where a claim is measured, the number is cited and the report
it came from is named. Where a claim is *not* measured, it says
`Not yet measured.` — per spec §38, unmeasured claims are never dressed up as
findings.

**Measured context for everything below** (from `artifacts/reports/`):
240 documents · 363 chunks · 206 eval queries (173 scored) · hashing embedder ·
lexical-v2 reranker · seed 20260929.

| Experiment | Recall@10 | Precision@10 | MRR | nDCG@10 | p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E1 Dense only | 0.674 | 0.160 | 0.439 | 0.324 | 8.0 ms |
| E2 Sparse only (BM25) | 0.744 | 0.198 | 0.484 | 0.406 | 0.8 ms |
| E3 Hybrid (RRF) | 0.732 | 0.184 | 0.483 | 0.372 | 8.4 ms |
| **E4 Hybrid + reranker** | **0.829** | **0.245** | **0.709** | **0.511** | 9.7 ms |

---

## 1. RAG instead of fine-tuning

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Retrieval-augmented generation** | **Chosen** | IT knowledge changes weekly (a VPN client release invalidates a procedure). RAG updates by re-indexing a document. It also gives citations, which is what makes the answer auditable — and an IT copilot that cannot show its source is not trustworthy for actions people take on production systems. |
| Fine-tuning a base model | Rejected | Bakes knowledge into weights, so every policy change needs a retraining cycle. No citations. No access control — you cannot fine-tune a model that "knows" a document for IT_admin but not for an employee. That last point is disqualifying here. |
| Long-context stuffing (whole KB in prompt) | Rejected | 240 documents ≈ 190k characters. Against the measured 8,000 TPM free-tier ceiling this is roughly **six minutes of quota for one question**. Also degrades with distractors. |
| Hybrid (RAG + light fine-tune) | Deferred | Defensible later for *style*, never for *facts*. No training data exists yet. |

**Trade-off accepted:** retrieval quality becomes the ceiling on answer quality. If retrieval misses, the answer is wrong or absent. That is why most of the measurement effort in this project is on retrieval, not generation.
**Replace when:** the corpus stabilises AND citations stop being required AND per-role access disappears. All three are unlikely.

---

## 2. Vector store — Qdrant

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Qdrant (embedded local mode)** | **Chosen** | Verified by spike on this machine: local mode (`QdrantClient(path=…)`) supports sparse vectors, `prefetch`, **server-side RRF fusion** and metadata filters with **no server and no Docker** — which is decisive given no Docker/WSL2 here. Same client code moves to a server by swapping `path=` for `url=`. Payload filtering is expressive enough for the `access_level` security filter. |
| pgvector | Strong runner-up | One less service, transactional consistency with the relational data. Rejected because it has no native sparse/BM25 arm, so hybrid retrieval would need a second system anyway — and because Postgres itself needs a server we cannot run locally. |
| FAISS | Rejected | A library, not a store: no payload filtering, no persistence story, no metadata. The RBAC filter alone rules it out. |
| Chroma | Rejected | Embeds easily but weaker filtering and sparse support. |
| Weaviate / Milvus / Elasticsearch | Rejected | Capable, heavier operationally. Elasticsearch would actually be a good single-system hybrid answer at larger scale — but it cannot run embedded here. |

**Trade-off accepted:** embedded mode is single-process and unclustered, so nothing here demonstrates distributed vector search. Stated plainly rather than implied otherwise.
**Replace when:** corpus exceeds a few million chunks, or transactional consistency between vectors and rows becomes a correctness requirement — then pgvector becomes the better answer.

---

## 3. Dense retrieval / embeddings

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Hashing embedder (current default)** | **Chosen by constraint** | Pure numpy, 8,192 dims, deterministic, zero downloads. The corporate proxy rejects HuggingFace model-file downloads (TLS handshake failure on the CDN), so no neural encoder can be obtained on this machine. This keeps the *entire* pipeline runnable and measurable today. |
| bge-small-en-v1.5 via fastembed/ONNX | **Intended upgrade, blocked** | ONNX Runtime 1.30 installs and imports on Python 3.14 (verified) — no PyTorch needed. Only the model weights are blocked. Code path implemented and selectable via `OPSPILOT_EMBEDDING_BACKEND=onnx`. |
| sentence-transformers (PyTorch) | Rejected | Does not declare Python 3.14 support, and torch ships CPU-only wheels for 3.14. Heavier for no gain over ONNX here. |
| OpenAI/Cohere embedding APIs | Rejected | Groq offers **no embedding endpoint** (verified), so this would add a second vendor, a second key and a per-query network hop on the hot path. |
| BGE-M3 (dense + sparse + ColBERT in one) | Considered | Genuinely attractive: one model produces both arms. 568M params on a CPU-only U-series laptop makes ingestion slow, and it is blocked by the same download problem. |

**What the measurement says — and this is the honest part.** The hashing embedder is a *weak* dense arm, and the evaluation shows exactly that:

- On **S-class** (semantic paraphrase, vocabulary deliberately disjoint from the document) dense scores **0.256** versus BM25's **0.261** — i.e. **no advantage at all**, when semantic matching is the entire reason dense retrieval exists.
- Consequently **RRF fusion makes things worse**: E3 (0.372) is significantly *below* E2 (0.406), CI `[-0.052, -0.015]`.

That is the strongest single argument in this repo for a real neural embedder, and it is evidence rather than assertion. **Not yet measured:** what bge-small does to S-class and to E3-vs-E2. That is the first experiment to run once model files are available.

**Trade-off accepted:** current dense numbers understate what the architecture can do. Better to report a weak measured arm than a strong imagined one.
**Replace when:** model files become reachable — by whitelisting `cdn-lfs*.hf.co` / `cas-server.xethub.hf.co`, or by copying the ONNX files in manually.

---

## 4. Sparse retrieval — BM25, implemented directly

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Own BM25 implementation** | **Chosen** | Three concrete needs: (a) IDF weights are required to emit Qdrant sparse vectors; (b) per-term score contributions are needed to explain *why* a lexical match won, which is the L-class analysis; (c) **tokenisation policy is where identifier retrieval is won or lost**, and it must be explicit. `ERR_VPN_809` is indexed as `err_vpn_809` **and** `err`,`vpn`,`809`, so both exact-code and partial queries match. |
| `rank_bm25` library | Rejected | No IDF access for sparse-vector emission, no control over identifier tokenisation — which is the one thing that matters most here. |
| SPLADE (learned sparse) | Deferred | Usually beats BM25, but needs a model (blocked) and inference cost per query. |
| Elasticsearch BM25 | Rejected | Cannot run locally; overkill at 363 chunks. |

**Measured payoff:** on **L-class** (exact error codes) BM25 scores **0.654** against dense's **0.315** — better than **2×**. This is the clearest confirmation of the corpus design hypothesis: embeddings smear rare identifiers, lexical matching does not.
**Trade-off accepted:** ~200 lines to maintain, and no learned term expansion.
**Replace when:** SPLADE becomes runnable, or the corpus grows past what an in-process inverted index should hold.

---

## 5. Fusion — Reciprocal Rank Fusion

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **RRF (k=60), fused in Python** | **Chosen** | Rank-based, so it needs no score normalisation between two arms whose scores are not comparable (cosine similarity vs unbounded BM25). Robust, one parameter, hard to get wrong. |
| Weighted score fusion | Rejected | Requires normalising incomparable scales; the normalisation constants become hidden hyperparameters that silently drift with the corpus. |
| Distribution-Based Score Fusion (Qdrant native) | Considered | Qdrant supports it. Worth benchmarking against RRF later. `Not yet measured.` |
| Qdrant server-side fusion | **Available, deliberately not used for evaluation** | Qdrant can fuse server-side and that is the production path. Fusing in Python keeps the **per-arm ranked lists intact**, and without those the "BM25 beat dense on L-class" analysis is impossible — the arms would already be collapsed inside the database. Measurement path and production path differ on purpose. |

**Measured — and the first conclusion drawn from it was wrong. Corrected here.**

E3 (0.372) scores below E2 alone (0.406), which reads as "fusion hurts". But nDCG of the *fused ordering* is the wrong yardstick: in the production pipeline the reranker re-scores the entire candidate pool, so fusion's only real job is to assemble a pool that **contains** the right documents.

Measured on that metric — candidate-pool recall@30:

| Configuration | Pool recall@30 |
| --- | ---: |
| Sparse only | 0.887 |
| Dense only | 0.846 |
| **Hybrid (equal weights)** | **0.914** |

The dense arm contributes **14 gold documents that BM25 never retrieves at all**. Those are unreachable any other way, and the reranker can only rank what fusion hands it.

A weight sweep confirms the ordering question is moot in production: with the reranker enabled, nDCG varies only between **0.509 and 0.512** across dense weights from 0.0 to 1.0. The reranker washes out the fusion weighting entirely.

**Conclusion:** keep hybrid at equal weights. Report E3 honestly as the *unreranked* ordering — which is not the production path — and judge fusion on pool recall, which is the thing it actually controls.
**Replace when:** pool recall for hybrid stops exceeding sparse-only. Re-measure the day a neural embedder lands.

---

## 6. Reranking

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Feature-based lexical reranker (`lexical-v2`)** | **Chosen** | Not a second BM25 pass — that would add nothing. It scores signals fusion structurally cannot see: IDF-weighted query coverage, exact-identifier presence, heading/title alignment, product-name agreement, and document `status` (current > superseded > deprecated). Zero model, ~2 ms, runs anywhere. |
| ONNX cross-encoder (ms-marco-MiniLM-L-6-v2) | **Implemented, blocked** | The quality option. Code path exists (`CrossEncoderReranker`). Blocked by the same model-download problem. **Latency on this CPU is `Not yet measured.`** and is the single least-certain number in the design — 30 pairs through a cross-encoder on a U-series CPU is the main risk to the ≤900 ms budget. |
| bge-reranker-v2-m3 | Rejected for local | Stronger and much heavier; not viable CPU-only. |
| Hosted reranker API (Cohere) | Rejected | Another vendor, another key, network hop per query. |
| No reranker | **Rejected by measurement** | See below. |

**This is the biggest measured win in the system.** E4 vs E3: **+0.139 nDCG@10**, CI `[+0.110, +0.166]` — significant. And the per-class table shows *where* it comes from, which is what makes it credible rather than lucky:

| Class | Hybrid (E3) | + reranker (E4) | What the reranker is doing |
| --- | ---: | ---: | --- |
| **V** version/recency | 0.133 | **0.519** | Status boost demotes deprecated/superseded revisions that contain a deliberately wrong step |
| **D** distractor | 0.191 | **0.416** | Product-name agreement separates "Corporate VPN" from the near-identical "Lab VPN" doc |
| **B** baseline | 0.183 | **0.429** | IDF-weighted coverage outranks generic term overlap |
| **S** paraphrase | 0.292 | **0.378** | Partial — cannot manufacture semantics the embedder never had |

**Trade-off accepted:** a hand-built feature reranker encodes assumptions about *this* corpus (that `status` and product names matter). A cross-encoder would learn relevance instead of being told it. The features are few, explicit and individually justified rather than tuned until the number rose.
**Replace when:** cross-encoder weights are available — then benchmark E4-lexical against E4-cross-encoder and keep the winner on measured nDCG *and* measured p95.

**Caveat, stated rather than buried:** E4 reranks 30 candidates down to 10 while E3 takes the top 10 of the fused list, so E4 sees more candidates. That is the pipeline design, not a rigged comparison — without a reranker you cannot use 30 candidates, you would have to put them all in the context — but the depth difference is part of the effect and should not be read as pure reranker magic.

---

## 7. LLM provider and model tiering

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Groq (OpenAI-compatible)** | **Chosen** | The key you have. Very fast inference (500–1000 tok/s on gpt-oss). Reachable from this machine once `truststore` is injected. |
| Local model (Ollama / llama.cpp) | Rejected | 15.5 GB RAM, no GPU, low-power U-series CPU. A 7B at single-digit tok/s makes the eval loop unusable. |
| OpenAI / Anthropic direct | Deferred | Better quality, needs a key and budget. The provider abstraction makes this a config change. |
| **Deterministic fake provider** | **Chosen as the default** | Not a stub — it does extractive grounding over the retrieved context. It makes the whole pipeline testable offline at zero quota, and it is why the 32-test suite runs in 1.94 s with **zero API calls**. |

**Model tiering** (task → model), because the binding constraint is tokens, not capability:

| Task | Model | Why |
| --- | --- | --- |
| Intent routing, query rewrite | `openai/gpt-oss-20b` | ~1000 tok/s, cheapest. High call volume, low difficulty. |
| Final grounded generation | `openai/gpt-oss-120b` | ~500 tok/s. Only called once per answer. |
| LLM-as-judge (evaluation) | `llama-3.3-70b-versatile` | **Must differ from the generator** or faithfulness is self-evaluation (audit G5). |
| Injection classification | Prompt Guard 2 (86M), **run locally** | At 86M this costs milliseconds on CPU and **zero tokens**. Routing every retrieved chunk through a hosted model would spend the daily budget on defence alone. |

**The constraint that shaped this whole project.** Measured Groq free tier: **30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD**. At ~3.5k prompt tokens per grounded answer that is **~2 requests/minute and ~55 answers/day**. Consequences, all of which improved the engineering:

1. **Retrieval experiments are LLM-free** — Recall/MRR/nDCG need no generation, so the most rigorous measurement in the project costs nothing and runs unlimited.
2. **Caching became load-bearing**, not decorative (§11 below).
3. **Context compression became a design goal**, which also helps faithfulness.
4. **A token budget is enforced before the call**, not discovered via 429s in front of a user.

**Action required:** the free-tier table lists the `gpt-oss` models explicitly; `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` did **not** appear in those free-tier rows. Confirm your account's real limits in the Groq console before relying on the judge model.

---

## 8. Orchestration — LangGraph

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **LangGraph** | **Chosen** | Installs and runs on Python 3.14 (verified). Earns its place for exactly two things: **checkpointing** (the approval flow must survive the death of the HTTP request and resume later — `thread_id` gives this directly) and **explicit routing state** (intent, branch outcomes and retries are inspectable data, which is what makes the UI activity trace truthful instead of decorative). |
| Plain functions / hand-rolled state machine | Rejected | Fine until human-in-the-loop. Then you are writing your own checkpointer and serialisation, badly. |
| LangChain AgentExecutor | Rejected | Opaque control flow; hard to assert "the incident tool ran before generation" in a test. |
| CrewAI / AutoGen | Rejected | Multi-agent conversation frameworks. This problem is a **deterministic routed workflow**, not a negotiation between agents. Adding agent personas would be exactly the "technology because it looks impressive" the spec forbids. |
| Temporal / durable workflow engine | Rejected | Genuinely the right answer for long-running approvals at scale, and massive overkill at this size. |

**Trade-off accepted:** a dependency and its state-serialisation constraints (graph state must stay JSON-shaped, which is why `Hit` objects are flattened to dicts at the node boundary).
**Honest note:** if human-in-the-loop were removed, LangGraph would be hard to justify here and a plain function chain would do. The approval requirement is what makes it earn its keep.

---

## 9. MCP — deliberately not built yet

Spec §7 places MCP after internal tools work, and §45 forbids starting with it. **It is not implemented.** The service layer it must wrap (`opspilot/tools.py`, 13 typed tools) is complete and proven, which is the precondition.

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **MCP servers over the existing service layer** | **Planned** | Honest justification: **interoperability and protocol demonstration** — any MCP-capable client could drive these tools. |
| Claiming MCP improves performance/reliability | **Refused** | It would not. The spec itself says MCP must call the same service layer without duplicating logic, so in-process it is pure overhead: serialisation plus a transport hop. Saying "this is overhead I accepted in exchange for interoperability" is a stronger and more defensible position than inventing a benefit (spec §38, audit C3). |

**The one design rule it must respect** (audit G9): identity must cross the MCP boundary **out of band** as signed request context, never as a tool argument. If `employee_id` were a parameter, the model would control it and "create a ticket for another employee" would succeed at the data layer. A test already enforces this (`test_no_tool_exposes_an_identity_parameter`).

---

## 10. Relational store — SQLite now, PostgreSQL shape

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **SQLite via SQLAlchemy** | **Chosen locally** | No Docker, so no Postgres server. Schema written PostgreSQL-portable (explicit lengths, no SQLite-only types), so the move is a connection-string change. |
| PostgreSQL | **Target** | Correct for concurrent writes, real constraints, JSONB and an append-only audit table. `docker-compose.yml` is configured for it. |
| Putting this data in Qdrant | **Rejected** | Spec §13 forbids it, correctly. Tickets, approvals and audit logs need transactions, foreign keys and exact filters. A vector DB is the wrong tool and the audit trail would be unreliable. |

**Trade-off accepted:** SQLite's single-writer model means the concurrency story is untested. No claim is made about write throughput.

---

## 11. Caching — justified by quota, not by a latency hand-wave

Spec §22 warns against adding Redis and claiming a performance improvement. So the justification here is different and checkable: **under a 200k tokens/day ceiling, a cache hit returns budget.**

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **In-process exact-match cache** | **Chosen now** | No Redis server available. Interface matches a Redis implementation. Honest limitation: **single-instance only** — it does not survive a restart and is not shared across replicas. |
| Redis | **Target** | Needed the moment there is more than one backend replica. Configured in compose. |
| Semantic cache (embedding-similarity) | Deferred | Attractive under this quota ceiling, but risks serving a subtly wrong cached answer to a differently-worded question. Would need its own false-hit evaluation before being trusted. |

**Metrics reported** (`/metrics`): `llm_calls_avoided`, `tokens_saved`, `hit_rate` — countable facts, not a latency claim. Verified by test.

---

## 12. Chunking

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Structure-aware** | **Chosen** | Splits on markdown headings, packs small sections, avoids cutting numbered procedures. Measured: 363 chunks, mean 522 chars, max 896. A truncated 5-step remediation is an actively harmful answer, not just a lower score. |
| Fixed 500/100 | **Kept as the control arm** | Spec §8 forbids it as the *only* strategy. Retained so chunking is a measurable variable. |
| Semantic (embedding-boundary) chunking | Deferred | Needs a good embedder (blocked) and is slow to ingest. |

**The design decision that makes this measurable at all** (audit G2): chunk IDs are stable *within* a strategy, but gold labels key to **document character spans**, so labels survive re-chunking. Enforced by `test_span_overlap_survives_rechunking`. Without this, changing chunking silently invalidates the entire evaluation set.

---

## 13. Streaming — SSE

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **SSE** | **Chosen** | One-directional server→client, which is exactly the shape of this traffic. Plain HTTP: works through corporate proxies, no special infrastructure, auto-reconnect built into `EventSource`. |
| WebSockets | Rejected | Bidirectional capability that is not needed; more proxy and load-balancer friction. |
| Long polling | Rejected | Worse latency and more connection churn. |

**The non-obvious part** (audit G15): an SSE stream **cannot block for human approval**. So the stream does not wait — it emits an `approval_required` event carrying a single-use token and closes cleanly. Approval is a separate `POST /api/v1/approvals/{token}` that resumes the checkpointed LangGraph thread. Verified end to end by Demo 4, including idempotent replay.

---

## 14. Security

| Concern | Choice | Reasoning |
| --- | --- | --- |
| Authorization | **Server-side, table-driven, outside the LLM** | Spec §15. The model may *request*; `security.py` decides. Testable with no model involved — which is also why the RBAC suite costs zero quota. |
| Identity propagation | **Ambient `Principal`, never a tool argument** | Audit G9. Enforced structurally and by test. |
| Direct injection | Regex detector | 78.1% recall, 89.3% precision, 42.9% FPR on deliberately adversarial benign look-alikes. Reported as the **weak** layer. |
| **Indirect injection** (KB docs, ticket bodies, tool output) | **Trust tiering + structural isolation** | The attack the spec omits. `TrustTier` marks retrieved/tool/user-record content as **never instruction-bearing**, untrusted blocks are fenced and labelled, and **tool arguments derive only from the user turn**. |
| Approval integrity | Single-use token bound to a hash of (tool, args, actor), TTL 300 s | Audit G10. Prevents replay against different arguments. |
| Duplicate actions | Idempotency key derived from the approval token | Approve-once-retry-twice would otherwise create three tickets. Verified by Demo 4. |
| Secrets | Env only, redacted in audit writes | Spec §15/§40. |

**Measured:** **5/5 authorization probes pass** — including the case where detection misses entirely. The security claim rests on the probes, not on detection recall. The neural Prompt Guard 2 second stage is **not implemented** (model blocked) and **no score is claimed for it**.

---

## 15. Abstention — calibrated, not guessed

| Option | Verdict | Reasoning |
| --- | --- | --- |
| **Threshold chosen from a measured sweep** | **Chosen** | 101-point sweep over the evidence score, scored as a confusion matrix across 173 answerable and 33 unanswerable queries. Policy: cap the **safety** error first (hallucination risk ≤ 15%), then minimise the usability error. |
| A guessed threshold | **Rejected — and this actually failed** | The initial guess (0.28) produced a 54.5% hallucination-risk rate and Demo 7 failed: the system confidently answered a question about tape-backup retrieval, a topic that does not exist in the corpus. |

**The fix was a real finding.** The reranker weighted all query terms equally, so an out-of-domain question full of common words ("what is the **procedure** for…") scored like a genuine hit. Changing to **IDF-weighted coverage, with out-of-vocabulary terms treated as maximally informative and never satisfied**, made unanswerable questions score low by construction. Balanced accuracy **0.635 → 0.754**.

**Chosen operating point: threshold 1.5** — hallucination risk **12.1%**, false abstention **37.0%**.
**Trade-off accepted, explicitly:** 37% false abstention is high — the system refuses roughly one in three answerable questions. That is the deliberate cost of the safety-first policy on a weak dense arm. It is the metric most likely to improve with a real embedder, and it is reported rather than hidden.

---

## 16. Python 3.14 and the dependency strategy

| Constraint | Decision |
| --- | --- |
| Python 3.14 installed; torch is CPU-only there and sentence-transformers does not declare 3.14 | **Avoid PyTorch entirely.** Use ONNX Runtime (1.30 verified working on 3.14). |
| HuggingFace model files blocked by the corporate proxy | Provider abstraction with a **working no-model default**, so nothing is blocked on network policy. |
| `api.groq.com` fails certificate verification with certifi | **`truststore.inject_into_ssl()`** — delegates to the Windows certificate store, which carries the corporate root CA. Verified: 200 OK. **Without this, Groq is unreachable and the failure looks like a bad API key.** |

---

## 17. Things deliberately NOT used

| Not used | Why not |
| --- | --- |
| Multi-agent frameworks (CrewAI, AutoGen) | This is a routed deterministic workflow. Agent personas would add cost and nondeterminism for no capability. |
| A graph/knowledge-graph store | No measured retrieval failure that a graph would fix. Revisit if multi-hop (M-class, currently 0.209) stays poor after a real embedder. |
| Fine-tuned embeddings | Requires labelled pairs; hard negatives are being collected, but training before a baseline exists is premature. |
| Semantic caching | Risk of serving a wrong cached answer; needs its own false-hit evaluation first. |
| Celery / task queue | No long-running background work yet. Would be needed for async ingestion at scale. |
| Kubernetes locally | Cannot run it; manifests authored and labelled unverified rather than claimed working. |

---

## 18. Summary of what is measured vs not

| Claim | Status |
| --- | --- |
| Hybrid + reranker beats all other arms | **Measured**, significant, CI `[+0.110, +0.166]` |
| BM25 beats dense on exact identifiers | **Measured**, 0.654 vs 0.315 |
| RRF fusion *hurts* with a weak dense arm | **Measured**, significant, CI `[-0.052, -0.015]` |
| Reranker fixes version-staleness and distractors | **Measured**, V 0.133→0.519, D 0.191→0.416 |
| Abstention operating point | **Measured**, 12.1% hallucination risk / 37.0% false abstention |
| Authorization cannot be bypassed by injection | **Measured**, 5/5 probes |
| Injection detection recall | **Measured**, 78.1% (weak layer, stated) |
| Neural embeddings would improve S-class | **Not yet measured** — model files blocked |
| Cross-encoder reranker quality and CPU latency | **Not yet measured** — the least certain number in the design |
| Faithfulness / answer relevance | **Not yet measured** — needs a judge model and its κ |
| End-to-end latency under load | **Not yet measured** — no load test run |
| Docker Compose / Kubernetes | **Not verified** — authored, never executed |
| Cache hit rate in real use | **Not yet measured** — mechanism verified by unit test only |
