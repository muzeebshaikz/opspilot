# OpsPilot AI — Specification Sufficiency Audit

**Status:** Phase 0, pre-implementation
**Inputs audited:** `prompt.docx` (1,306 lines, §1–§44) + architecture infographic
**Audited on:** 2026-09-29
**Verdict:** The spec is strong on *breadth* and unusually disciplined about honesty (§38). It is **insufficient in four areas**, **internally contradictory in four places**, and **infeasible as written** on the measured hardware. None of these are fatal; all have concrete fixes below.

> Read this before `DATA-STRATEGY.md` and `TECHNOLOGY-RATIONALE.md`. The gaps here are *why* those two documents look the way they do.

---

## 0. Executive summary

| # | Finding | Severity | Fix lives in |
|---|---|---|---|
| G1–G6 | The measurement foundation is missing. Metrics are demanded but ground truth has no defined provenance. | **Critical** | `DATA-STRATEGY.md` §3 |
| G7 | The corpus is specified so small and so clean that the experiment *cannot demonstrate* the thing it sets out to prove. | **Critical** | `DATA-STRATEGY.md` §2 |
| G8–G11 | Security covers direct prompt injection but not the attack surface that actually matters for RAG. | **High** | §3 below |
| G12–G17 | Operational unknowns: no SLO, no token budget, no deterministic tests, no SSE/HITL protocol. | **High** | §4 below |
| C1–C4 | Internal contradictions. | Medium | §5 below |
| E1–E5 | Environment does not match spec assumptions. | **Blocking** | §6 below |

**The single most important finding is E5:** the Groq free tier allows roughly **55 grounded RAG answers per day**. That number, not the architecture diagram, determines how this project must be built.

---

## 1. What the spec gets right (worth preserving)

Before the criticism — these are genuinely above-average design instincts and should not be traded away:

- **§38 "No fake claims"** and the `"Not yet measured."` requirement. Most portfolio projects fail exactly here.
- **§3 ADR-per-technology** with alternatives and trade-offs. This is the right unit of documentation.
- **§4/§45 incremental evolution** (V0→V7), explicitly forbidding MCP/K8s-first. Correct sequencing.
- **§15 "The LLM must NEVER be treated as the authorization authority."** This is the correct security posture and most agent projects get it wrong.
- **§10 Abstention as a valid outcome.** Treating "I do not know" as a success state rather than a failure.
- **§7** forbidding "upload a public dataset and call it enterprise RAG."

The audit below assumes all of the above stays.

---

## 2. Gap category A — The measurement foundation (Critical)

The spec demands Recall@K, Precision@K, MRR, nDCG, faithfulness, context precision/recall, and says "never fabricate evaluation values" (§19, §38). **It never says where ground truth comes from.** Without a labelling protocol, every one of those numbers is unfalsifiable — which is a subtler failure of §38 than outright fabrication, and a more likely one.

### G1 — Gold-label provenance is undefined (circularity risk)

If one LLM writes the knowledge base, the evaluation queries, *and* the relevance judgments, the resulting metrics measure **generator self-consistency**, not retrieval quality. The numbers will look excellent and mean nothing.

**Fix:** a three-tier labelling protocol with tiers reported *separately and never averaged* — see `DATA-STRATEGY.md` §3. Tier 2 uses TREC-style pooling so that judgments are made independently of the system that produced the candidate.

### G2 — "Stable chunk IDs" is incompatible with comparing chunking strategies

§8 requires *"Each chunk must have a stable ID."* §8 also forbids a single fixed chunking strategy, and §20 requires reproducible experiments across configurations. **The moment you re-chunk, every chunk-ID-keyed gold label is invalid**, and your entire evaluation set silently rots.

**Fix — the highest-value correction in this audit:** gold labels key to a **document span**, not a chunk.

```
qrel := (query_id, document_id, char_start, char_end, grade)
```

A retrieved chunk scores as a hit when its `[start, end)` overlaps the gold span by at least a declared threshold (proposal: the chunk covers ≥ 50% of the gold span, or the gold span covers ≥ 80% of the chunk). Labels then survive **any** re-chunking, and chunking strategy becomes a legitimately measurable variable instead of a label-invalidating change. Chunk IDs stay stable *within* a chunking config for caching and citation, which satisfies §8's actual intent.

### G3 — No significance testing (the spec implicitly invites noise-chasing)

§20 requires comparing five retrieval experiments. At the spec's corpus size the test set will be a few hundred queries. **An nDCG@10 delta of 0.02 on 150 queries is noise**, and reporting it as "hybrid beat dense" is a fabricated claim in substance if not in intent.

**Fix:** every experiment comparison reports paired bootstrap 95% CIs (10,000 resamples) and a paired significance test over per-query scores. House rule to add to §38: **no improvement may be claimed when its confidence interval crosses zero.** Report it as "no measurable difference at this sample size" — an honest and defensible result.

### G4 — Metrics are not role-aware, but documents are

Documents carry `access_level` (§5) and retrieval must filter on it (§9). But a document an `employee` may not see must neither count as a hit nor a miss for that employee's query. Computing corpus-wide metrics over an access-filtered retriever produces a systematically wrong number.

**Fix:** qrels are role-scoped; metrics computed per-role against role-filtered qrels and reported per-role. This yields a free security test: if role-filtered recall for `employee` ever counts an `IT_admin`-only document, retrieval is leaking.

### G5 — LLM-judge metrics are specified without judge validation

Faithfulness, answer relevance, context precision/recall (§19) all require an LLM judge. The spec never mentions judge–generator independence, position bias, or calibration. Self-evaluation by the generating model is a well-known failure mode.

**Fix:**
- The judge model **must differ** from the generator model. This is a hard requirement and it drives model selection in `TECHNOLOGY-RATIONALE.md` §7.
- Hand-label a 50-item calibration slice; report judge-vs-human agreement (Cohen's κ) alongside every judge-derived metric.
- **No judge metric is publishable without its κ stated.** A faithfulness score of 0.91 at κ = 0.34 means nothing.

### G6 — Abstention is required but has no metric, and is trivially gameable

§10 mandates abstention and §19 lists "out-of-domain questions" as a test set. But a system that abstains on *everything* scores perfectly on the out-of-domain set. The spec never pairs it with the opposing error.

**Fix:** score abstention as a confusion matrix over two paired sets — `answerable-hard` and `unanswerable` — reporting:
- **False-abstention rate** — refused despite evidence being present (the usability failure)
- **Hallucination-under-pressure rate** — answered confidently with no supporting evidence (the safety failure)

These trade off against each other. Report the operating point and the curve, not a single number.

---

## 3. Gap category B — Security covers the wrong attack surface (High)

### G8 — Only *direct* prompt injection is addressed; the real RAG threat is *indirect*

Every adversarial example in §17 is a direct user-turn attack ("Ignore previous instructions…"). Those are the easy ones. The attack that actually compromises a system of this shape is **injection through content the system retrieves**, because that content is attacker-influenced and arrives already trusted.

| Injection channel | Who controls it | Why it is dangerous here |
|---|---|---|
| **Ticket `description` / `add_ticket_comment`** | **Any employee** | This is the live one. A malicious employee writes an injection into a ticket; an `IT_agent` later asks OpsPilot about that ticket; the injected text enters the agent's context **carrying the agent's elevated privileges**. §5 and §18 both define these as user-writable fields that flow into the LLM. |
| Knowledge-base document body | KB author / ingestion pipeline | A poisoned doc persists and fires on every matching query. |
| Incident `workaround` / `description` | IT staff | Lower volume, higher privilege. |
| Asset `installed_software` | Device agent / endpoint | Attacker-nameable strings reaching context. |
| Tool output | Downstream services | Model may treat tool results as instructions. |

**This is a genuine hole, not a theoretical one.** The spec's own data model creates it, and the spec's own defence ("authorization is enforced outside the LLM") does not close it — authorization stops the *action*, but injection can still exfiltrate context, poison an answer, or induce the agent to instruct a human to do something harmful.

**Fix:**
1. **Trust tiering.** Every string entering context is tagged `system | user_turn | retrieved_kb | tool_output | user_generated_record`. The last three are untrusted by construction.
2. **Structural isolation.** Untrusted content is delimited and provenance-tagged; the system prompt states untrusted blocks contain data only, never instructions.
3. **Classifier on the retrieval path, not just the input path.** Run an injection classifier over *retrieved* content too. Meta's Prompt Guard 2 (86M, mDeBERTa) fits — and because it is only 86M it should run **locally on CPU**, not via API, costing zero quota (`TECHNOLOGY-RATIONALE.md` §14).
4. **Retrieved text may never influence tool selection or tool arguments.** Tool args derive only from the user turn plus validated state. A structural guarantee, testable by unit test.
5. **Output-side controls:** URL allowlisting; never emit executable command strings sourced from retrieved content without explicit marking.
6. Extend the adversarial eval set with indirect cases — `DATA-STRATEGY.md` §4.

### G9 — Identity propagation across the MCP boundary is unspecified

§15 correctly says the backend decides authorization. But §7 moves tools behind MCP servers, and at that point the MCP server must not trust anything the model wrote. If a tool signature is `get_ticket(ticket_id, employee_id)`, the model controls `employee_id` and the "create a high-priority ticket for another employee" attack (§17) succeeds at the data layer regardless of what the graph believes.

**Fix:** identity is **never a model-writable tool parameter**. It travels out-of-band as a signed, short-lived request context injected by the MCP client from the authenticated session. Tool signatures are `get_ticket(ticket_id)`; the acting identity is ambient and verified server-side. Add a lint/unit test that fails the build if any tool schema declares an identity-like field.

### G10 — No idempotency on consequential actions

§16 (human approval) + §29 (retry policies) + LLM re-planning compose into **duplicate ticket creation**. Approve once, retry twice, get three tickets.

**Fix:** approval tokens are single-use, time-limited, and bound to a hash of the exact serialized action (tool + args + actor). `create_ticket` carries an idempotency key derived from that token. Replay returns the original result rather than acting again. The spec omits this entirely.

### G11 — No PII, retention, or redaction policy

`audit_logs` store raw user queries; assets store device and location data; §5 users carry `manager` and `location`. The spec mandates audit logging but never bounds it.

**Fix:** declare a retention window, define which fields are redacted at log-write time, and keep the threat model honest that this is synthetic data, so the policy is *demonstrated* rather than *relied upon*.

---

## 4. Gap category C — Operational unknowns (High)

### G12 — No latency SLO

§21/§34 require measuring p50/p95/p99, but "measure latency" without a target produces uninterpretable numbers. Proposed budgets, committed in Phase 0 and then tested against:

| Stage | p95 target | Note |
|---|---|---|
| Dense + sparse retrieval + RRF | ≤ 250 ms | CPU, local Qdrant |
| Cross-encoder rerank (30 → 8) | ≤ 900 ms | **Highest risk on CPU** — drives reranker choice |
| Tool / MCP call | ≤ 300 ms | Local service layer |
| LLM time-to-first-token | ≤ 1.2 s | Groq is fast; this should hold |
| End-to-end knowledge answer | ≤ 6 s | |
| End-to-end multi-tool investigation | ≤ 12 s | |

These are **targets, not measurements**. They stay marked `Not yet measured.` until a run exists.

### G13 — No token or cost budget, though "LLM calls per request" is a required metric

See §6 / E5 below. This is the binding constraint on the whole project.

### G14 — No deterministic test strategy

§28 requires E2E tests for nine scenarios. Against a live LLM those tests are flaky, slow, and — critically — would consume the entire daily quota in one CI run.

**Fix:** a `FakeLLMProvider` returning deterministic scripted responses, plus record/replay cassettes for integration tests. **CI runs with zero API calls.** A separate, manually-triggered `eval` workflow makes real calls. This also gives the §12 LLM abstraction a second reason to exist beyond provider-swapping.

### G15 — SSE streaming and human-in-the-loop are specified independently and do not compose

§23 requires SSE streaming; §16 requires the graph to pause for approval. The spec never says how an open SSE stream pauses and resumes.

**Fix:** the stream does not block. It emits an `approval_required` event carrying an opaque single-use token, then closes cleanly. Approval is a separate `POST /api/v1/approvals/{token}` which resumes the checkpointed graph and opens a new stream. This maps exactly onto LangGraph's `interrupt` + checkpointer — and is the **strongest honest justification for LangGraph** in the system (`TECHNOLOGY-RATIONALE.md` §8).

### G16 — "Goal completion" is a required metric with no rubric

§19 lists it; nothing defines it. **Fix:** per-scenario machine-checkable success predicates (e.g. *ticket created with priority=High AND assigned_team=NetworkOps AND audit row exists AND approval token consumed*), not an LLM opinion.

### G17 — `dataset version` is a required experiment field with no versioning mechanism

**Fix:** eval datasets are content-addressed — a manifest hash over the query set plus qrels, recorded in every experiment row. Changing a label changes the hash, so results are never silently compared across different ground truth.

---

## 5. Internal contradictions

| # | Contradiction | Resolution |
|---|---|---|
| **C1** | "Production-grade RAG" (§0) vs "approximately 100–300 documents" (§6). | Raise the corpus to ~520 docs / ~7k chunks and — more importantly — *design* its difficulty. See G7 and `DATA-STRATEGY.md` §2. |
| **C2** | "Each chunk must have a stable ID" (§8) vs comparing chunking strategies (§8, §20). | Span-based qrels. See G2. |
| **C3** | "Do not add technologies merely because they look impressive" (§3) vs mandating MCP and Kubernetes for a single-node portfolio app (§7, §31). | **Keep both, but the ADRs must state the honest reason.** MCP's real benefit here is *interoperability and protocol demonstration*, not performance — the spec itself concedes it wraps the same service layer (§7). Kubernetes' real benefit is *deployment-portability demonstration*. Claiming a throughput or reliability win would violate §38. "This is overhead I accepted in exchange for X" is a *stronger* interview answer than an invented benefit. |
| **C4** | §45 "Do NOT start by implementing MCP / K8s" vs §41 Definition of Done requiring both. | Not a real conflict, but the DoD should be **per-phase** rather than one flat list, otherwise nothing is "done" until everything is. |

---

## 6. Environment reality check (measured, not assumed)

Measured on the target machine on 2026-09-29:

| Spec assumption | Measured reality | Consequence |
|---|---|---|
| **E1** `docker compose up` brings up 9 services (§30) | No Docker, no WSL2, no admin install | Compose + K8s manifests are **authored and CI-linted but not run locally**. Per §31/§38 they must be labelled *"not verified locally"* — never "production-grade". |
| **E2** GPU available for reranking / local LLM | Intel Core Ultra 5 135U, **integrated graphics, no CUDA** | Reranker must be CPU-viable. Rules out the large cross-encoders; drives `TECHNOLOGY-RATIONALE.md` §6. |
| **E3** Local model hosting (§12) | 15.5 GB RAM, low-power U-series CPU | Local LLM generation is impractical; Groq API is correct. Local models stay viable for **embeddings** and the **86M injection classifier**. |
| **E4** Modern Python toolchain | **Python 3.14** installed | PyTorch ships CPU-only wheels for 3.14 and `sentence-transformers` does not yet declare 3.14 support. **Pin a Python 3.12 virtualenv.** Do not fight this. |
| **E5** Unlimited LLM calls for evaluation | **Groq free tier: 30 RPM · 1,000 RPD · 8,000 TPM · 200,000 TPD** (gpt-oss models) | **See below. This reshapes the project.** |

### E5 — The Groq free tier is the binding constraint on this entire project

At roughly 3,500 prompt tokens for a grounded RAG answer (system prompt + 5–8 reranked chunks + conversation state):

- **8,000 TPM ⇒ about 2 requests per minute.**
- **200,000 TPD ⇒ about 55 full RAG answers per day.**

A naive execution of §20 — 150 queries × 5 configurations × generation × 4 judge metrics — is several million tokens. That is **weeks** of free-tier budget for one experiment sweep.

This is not a reason to scale down the ambition. Handled correctly it *improves* the engineering, and every consequence below is a better design decision on its own merits:

1. **Retrieval experiments must be LLM-free — and they already can be.** Recall@K, Precision@K, MRR and nDCG require *no generation whatsoever*. Experiments 1–4 of §20 (dense / sparse / hybrid / hybrid+rerank) run entirely offline, unlimited, in seconds, thousands of times. **This is where the real measurement rigour lives, and it costs nothing.** Only Experiment 5 (query rewrite) needs an LLM, and only for the rewrite step.
2. **Generation-side metrics move to a sampled subset** (~60 queries) with bootstrap CIs, explicitly labelled as sampled with n stated. Honest, and standard practice.
3. **Caching stops being decorative and becomes load-bearing.** §22 warns "do not merely add Redis and claim performance improvement." Here caching genuinely *buys quota*, and the honest metric is not a latency hand-wave but **LLM calls avoided** and **tokens saved** — hard, countable numbers that directly extend how much evaluation is affordable.
4. **Context compression becomes a first-class design goal.** Every token trimmed is quota returned. Fewer, tighter, better-reranked chunks *also* improve faithfulness. The constraint and the quality goal point the same way.
5. **A token-budget meter belongs in observability** (§21) and in the UI. "Tokens remaining today" is a real operational concern here, which is more than most portfolio projects can claim.
6. **Model tiering by task** — a cheap fast model for classification / routing / rewrite (high call volume, low difficulty), a stronger one only for final grounded generation.

**Verification required on day 1:** the free-tier table retrieved lists the `openai/gpt-oss-*` models explicitly; `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` did **not** appear in those free-tier rows. Groq's docs note per-organisation exceptions. **Confirm actual limits on the account's own settings page once the key exists**, and record them in `docs/architecture/decisions/ADR-009-model-provider.md`. Do not design against assumed numbers.

---

## 7. What changes as a result

| Spec section | Change |
|---|---|
| §6 Knowledge base | 100–300 docs → ~520 docs, **composition designed** around retrieval failure modes (G7) |
| §8 Ingestion | Chunk IDs stable *within a config*; gold labels keyed to document spans (G2) |
| §17 Injection testing | Extended to indirect injection via tickets, KB, tool output (G8) |
| §19 Evaluation | Three-tier label provenance; per-role metrics; bootstrap CIs; judge κ; abstention confusion matrix (G1–G6) |
| §20 Experiments | Experiments 1–4 are **LLM-free and unlimited**; Experiment 5 sampled (E5) |
| §22 Redis | Justified by **quota preservation**, measured in calls-avoided and tokens-saved (E5) |
| §23 API | Add `POST /api/v1/approvals/{token}`; define the `approval_required` SSE event (G15) |
| §28 Testing | `FakeLLMProvider` + cassettes; CI makes **zero** API calls (G14) |
| §30/§31 Docker/K8s | Authored, linted, **explicitly labelled not-run-locally** (E1) |
| §38 No fake claims | Add the rule: **no improvement claimed when its CI crosses zero** (G3) |
| §41 Definition of Done | Split into per-phase DoD (C4) |

---

## 8. Open questions requiring a decision

1. **Confirm the real Groq limits** on the account settings page once the key is added (E5).
2. **Qdrant local-mode feature parity** — confirm by spike that sparse vectors plus prefetch/fusion work in `QdrantClient(path=...)` without a server. *Risk is already mitigated:* RRF is implemented in Python anyway for per-arm measurement transparency, so local-mode fusion support is **not on the critical path**.
3. **Reranker latency on this CPU** — the ≤ 900 ms p95 target for 30 pairs is the least certain number in §4. Spike it before committing to the pipeline shape.
4. **Tier-3 human labelling effort** — 100 hand-verified queries is real work. Confirm willingness; it is what makes the headline metrics defensible.

---

## Sources

- Groq model catalogue — <https://console.groq.com/docs/models>
- Groq rate limits — <https://console.groq.com/docs/rate-limits>
- Llama Prompt Guard 2 on Groq — <https://console.groq.com/docs/model/llama-prompt-guard-2-86m>
- Qdrant hybrid queries / RRF / DBSF — <https://qdrant.tech/documentation/concepts/hybrid-queries/>
- PyTorch Python 3.14 support tracking — <https://github.com/pytorch/pytorch/issues/156856>, <https://github.com/pytorch/pytorch/issues/169929>
- sentence-transformers supported Python versions — <https://pypi.org/project/sentence-transformers/>
