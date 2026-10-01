# OpsPilot AI — Data Strategy

**Status:** Phase 0 design, no data generated yet
**Depends on:** `docs/audit/SPEC-AUDIT.md` (G1–G7 are the reason this document exists)
**Covers:** spec §5 (synthetic world), §6 (knowledge base), §7 (public data), §19 (evaluation datasets)

---

## 0. The core argument

The spec asks for "approximately 100–300 documents" of synthetic enterprise knowledge and then asks us to prove that hybrid retrieval beats dense, and that reranking beats hybrid.

**On a small, clean, well-separated synthetic corpus, those things may simply not be true** — dense retrieval alone tends to saturate Recall@10 near ceiling when every document is about a distinct topic and was written by a single coherent generator. If dense@10 lands at 0.97, hybrid has nowhere to go, and you are left choosing between reporting "no improvement" (honest, weak story) or quietly massaging the setup (dishonest, violates §38).

The fix is not to make the corpus bigger. **The fix is to design the corpus so that the failure modes each technique exists to solve are actually present in it, in known quantities, with known labels.** Then the experiment can succeed *or* fail on its merits, and either outcome is a real result.

That principle drives everything below.

---

## 1. Public data — ruling and licence register

### 1.1 The rule we are applying

Spec §7 permits public datasets **only** for intent examples, query examples, evaluation examples and synthetic query generation, and forbids loading a public dataset into the vector store and calling it enterprise RAG.

**Our stricter rule:** no public dataset text is ever ingested into Qdrant, ever embedded, or ever quoted in the knowledge base. Public data is used for *method*, *taxonomy*, *phrasing calibration* and *adversarial seeds* only. The repository stays licence-clean (MIT/Apache-compatible body) and anything with a viral licence is quarantined.

### 1.2 Licence register (verified 2026-09-29)

| Dataset | Licence | Size / shape | What we use it for | Constraint we must honour |
|---|---|---|---|---|
| **CLINC150** — `clinc/clinc_oos` | **CC-BY-3.0** | 150 intents across 10 domains + a dedicated **out-of-scope** label; 19.2k rows (imbalanced config), 59.3k across all configs | (a) template for our intent taxonomy and its granularity; (b) **source of out-of-scope queries → abstention negatives** (G6) | Attribution required. Domains are consumer-assistant, **not IT** — use as a *method and OOS source*, never as IT content. |
| **deepset/prompt-injections** | **Apache-2.0** | 662 rows (546 train / 116 test), German + English, binary label | Seed corpus for the adversarial suite; baseline sanity check for our injection classifier | Permissive, no friction. **Small and direct-injection-only** — must be extended with the *indirect* RAG injections we author ourselves (G8). |
| **TechQA** — `rojagtap/tech-qa` | **MIT** | 621 rows (450/160/11) over real IBM Technotes — WebSphere, SPSS, DataPower, security bulletins | Phrasing calibration: how real enterprise users actually word a support question. Question-form templates. | IBM-product-specific. **Never ingested.** Read to calibrate tone; no text copied into our corpus. |
| **Stack Exchange dumps** — ServerFault, Super User | **CC-BY-SA 4.0** | Large; XML dumps (Posts, Comments, Users, …) | Realistic user *phrasing* of IT problems — the messy, under-specified way people actually describe a broken VPN | ⚠️ **Share-alike is a licence hazard.** Any derivative dataset inherits SA and could infect the repo. **Use for human/LLM style reference only.** If any derived file is ever committed, it lives alone under `data/external/stackexchange/` with its own `LICENSE` + per-post attribution, and is excluded from the main licence. Also note access to the dumps has become less reliably public since 2024. |
| **BEIR / MS MARCO** | Research-use terms vary by sub-corpus | — | **Methodology reference only**: qrels file format, pooling depth conventions, nDCG cut-off conventions | No data used. Cited as prior art for the evaluation design. |

### 1.3 Recommendation

**Adopt as real dependencies:** CLINC150 (out-of-scope split) and `deepset/prompt-injections`. Both are permissively licensed, small, and fill gaps we genuinely cannot fill credibly ourselves — OOS negatives written by the same person who wrote the corpus are not convincing negatives.

**Use as read-only style references:** TechQA and ServerFault. Humans and the generation prompt read them to calibrate phrasing; no text is copied. This keeps the repo licence-clean while still making the query realism evidence-based rather than imagined.

All of this is recorded in `data/external/DATA-LICENCES.md` with retrieval date and URL, per §7's "record dataset licensing/source information". **Licences change — re-verify at ingest time, do not trust this table's age.**

---

## 2. Synthetic corpus design — the difficulty ledger

### 2.1 Query difficulty taxonomy

Every evaluation query carries a **difficulty class tag**. This is the mechanism that makes the §20 experiments meaningful, because it lets us report *where* each retrieval arm wins instead of one aggregate number that hides everything.

| Tag | Class | What it looks like | Which technique should win | Why it belongs in the corpus |
|---|---|---|---|---|
| **L** | Lexical-anchored | `ERR_VPN_809`, `HKLM\SOFTWARE\...`, `--mtu 1400`, CVE IDs, hostnames | **BM25** | Embeddings smear rare tokens and exact identifiers. This is the canonical case where dense retrieval fails and sparse rescues it. |
| **S** | Semantic paraphrase | "my internet dies whenever I join a Teams call" → *Split-tunnel MTU mismatch* | **Dense** | Zero vocabulary overlap with the document. BM25 scores near zero. |
| **V** | Version / recency | Answer exists in v2.1 *and* v3.2; only v3.2 is current | **Metadata filter + reranker** | Tests that recency and version metadata actually reach the ranker. |
| **P** | Platform variant | Same question, Windows vs macOS vs Linux answer | **Metadata filter** | Tests attribute-aware retrieval. |
| **D** | Distractor | "Corporate VPN" vs "Lab VPN" — near-identical docs, one correct | **Reranker** | Fusion alone cannot separate these; cross-encoding can. This is reranking's entire justification. |
| **M** | Multi-hop | Answer requires two documents combined | Hybrid + context builder | Tests that recall@k is doing real work, not just top-1. |
| **A** | Access-gated | Answer exists but the asking role may not see it | RBAC filter | Must return abstention, **not** a leak (G4). |
| **X** | Unanswerable | Topic genuinely absent from the corpus | Abstention | The other half of the abstention confusion matrix (G6). |
| **C** | Conversational | Resolvable only with prior-turn context | Query rewrite | Justifies Experiment 5. |
| **T** | Tool-required | "Status of INC-10482" — structured data, not KB | Tool routing | Tests the router, not retrieval. |
| **H** | Hybrid KB+tool | "VPN broken — outage? my laptop?" | Multi-tool + RAG | The §35 Demo 3 scenario. |

**The payoff:** instead of "hybrid nDCG@10 = 0.81 vs dense 0.79" (noise, unfalsifiable, boring), the result reads *"BM25 beats dense on L-class by +0.31 [CI 0.24–0.38]; dense beats BM25 on S-class by +0.28 [CI 0.21–0.35]; RRF captures ~85% of both."* That is a real finding, it is falsifiable, and it is the interview answer to "why hybrid?".

### 2.2 Corpus composition

Target **≈ 542 documents / ≈ 7,000 chunks** across the 14 categories in §6 (`vpn, password, email, microsoft365, github, git, docker, kubernetes, aws, databases, networking, laptops, security, onboarding`).

| Document family | Count | Induces | Notes |
|---|---:|---|---|
| Canonical topic guides | 70 | baseline | The happy path; one per major topic. |
| Error-code reference docs | 65 | **L** | Dense tables of codes, symptoms, fixes. Deliberately identifier-heavy. |
| Version pairs (v2.x superseded by v3.x) | 90 (45 pairs) | **V** | Both indexed; only one current. `effective_date` and `version` carry the signal. |
| Platform variants (Win / macOS / Linux) | 60 (20 triples) | **P** | Same procedure, different specifics. |
| Near-duplicate distractors | 45 | **D** | "Lab VPN" / "Contractor VPN" / "Corporate VPN". Deliberately confusable. |
| Policy docs with graded `access_level` | 60 | **A** | Spread across employee / IT_agent / IT_admin. |
| Procedural runbooks | 75 | chunking | Numbered steps, prerequisites, rollback. **Tests structure-aware chunking** — naive 500-char splitting will cut steps in half and this is where that shows up as a metric. |
| Multi-hop pairs | 40 (20 pairs) | **M** | Doc A gives the precondition, doc B the procedure. |
| Short FAQ docs (< 400 chars) | 25 | chunking edge | Document shorter than one chunk. |
| Long docs (> 15k chars) | 12 | chunking edge | Forces hierarchy detection. |
| **Total** | **542** | | ≈ 7,000 chunks at ~13 chunks/doc |

Plus a declared list of **30 "absent topics"** — subjects cross-referenced in passing but deliberately never documented. These generate the **X** class. Keeping them in a manifest means unanswerability is *designed and verifiable*, not accidental.

### 2.3 Structured enterprise data (§5)

| Entity | Count | Deliberate edge cases (these are the point) |
|---|---:|---|
| `users` | 120 | 100 employee / 14 IT_agent / 6 IT_admin. Plus: a contractor with reduced access, an on-leave user, a two-level manager chain, a user with no assigned device, a recently-offboarded user. |
| `assets` | 240 | Users with 2 devices and users with 0. Stale `last_seen`. A device with an outdated `VPN_client_version` that *explains* a known incident — so troubleshooting can genuinely correlate. |
| `tickets` | 620 | Spread across all statuses/priorities. **~25 containing benign injection-shaped text** (a user quoting an error that contains "ignore previous instructions") and **~15 genuinely adversarial** (G8). Tickets owned by other employees, for RBAC tests. |
| `incidents` | 55 | Active / monitoring / resolved. **Time-relative windows** so "is there *currently* an outage?" depends on `now()` and cannot be answered from a static string. |
| `conversations` / `messages` | 80 / ~400 | Multi-turn histories for the memory tests (§14) and C-class queries. |

The device-version-explains-the-incident detail matters: without it, "check whether there is an outage and check my laptop" (§35 Demo 3) has no correct joined answer, and the multi-tool demo is theatre.

---

## 3. Gold-label protocol (fixes G1, G2, G5)

### 3.1 Labels key to document spans, not chunks

```
qrel := (query_id, document_id, char_start, char_end, grade, role_scope, difficulty_tag)
grade ∈ {0 = irrelevant, 1 = partially supports, 2 = fully supports}
```

A retrieved chunk counts as a hit when its character range overlaps the gold span by the declared threshold (**chunk covers ≥ 50% of the gold span, OR gold span covers ≥ 80% of the chunk**). Labels therefore survive re-chunking, and chunking strategy becomes a measurable variable rather than a label-destroying change (G2).

Graded relevance (0/1/2) is required for nDCG to mean anything — binary labels make nDCG collapse toward MRR.

### 3.2 Three tiers of label provenance, reported separately

| Tier | Method | Volume | Strength | Known bias |
|---|---|---:|---|---|
| **Tier 1 — Constructive** | Generate the query *from* a chunk; the gold span is that chunk's source span | ~500 | Weak | **Optimistic by construction.** The query was written while looking at the answer, so vocabulary overlap is inflated and retrieval looks better than it is. Must be labelled as such everywhere it is reported. |
| **Tier 2 — Pooled / independent** | Queries written against the *topic* (never the text), phrasing calibrated from ServerFault/TechQA style. Then **TREC-style pooling**: run all 5 retrieval configurations, pool the union of top-10, judge each pooled candidate blind to which arm produced it | ~200 | Strong | Pool bias — a document no configuration ever retrieves is never judged, so recall is slightly overstated for all arms equally. Standard, disclosed, and fair across arms. |
| **Tier 3 — Human-verified** | 100 queries labelled by hand | 100 | Strongest | Small n; wide CIs. |

**These three are never averaged into one number.** Tier 2 and Tier 3 carry the headline results; Tier 1 exists for high-volume regression detection and rapid iteration.

This structure is what kills the circularity in G1: the Tier-2 judgment is made without knowing which retriever produced the candidate, so it cannot flatter the system.

### 3.3 Hard negatives

For each Tier-2/3 query, store the top-ranked **non-relevant** documents explicitly as mined hard negatives. These serve three purposes: they make Precision@K meaningful, they are the seed set for any future embedding fine-tuning, and they are the regression fixture for "this used to rank wrong".

### 3.4 Judge calibration (fixes G5)

- Generator and judge **must be different models**.
- The 100 Tier-3 human labels double as the judge calibration set.
- Report **Cohen's κ** for judge-vs-human on every judge-derived metric.
- **House rule: no faithfulness / answer-relevance / context-precision number is published without its κ printed beside it.**

### 3.5 Dataset versioning (fixes G17)

Every eval dataset is content-addressed: `sha256` over the canonicalised query set + qrels + difficulty tags. That hash is a required column in every experiment record. Changing one label changes the hash, which makes silently comparing results across different ground truth impossible.

---

## 4. Evaluation set inventory

Spec §19 names eleven test sets. Sized, sourced, and — critically — marked for whether they cost LLM quota (E5).

| Set | Size | Label tier | Difficulty classes | Needs LLM? |
|---|---:|---|---|---|
| Factual knowledge | 180 | T1 + T2 | L, S, V, P | ❌ retrieval only |
| Retrieval benchmark (the §20 core) | 300 | T1/T2/T3 | all of L,S,V,P,D,M | ❌ **free, unlimited** |
| Troubleshooting | 90 | T2 + T3 | S, M, H | ✅ sampled |
| Ticket lookup | 60 | deterministic | T | ❌ exact-match assertions |
| Incident lookup | 45 | deterministic | T | ❌ exact-match assertions |
| Multi-tool workflows | 40 | rubric predicates | H | ✅ sampled |
| Multi-turn conversations | 50 | T2 | C | ✅ sampled |
| Action confirmation (HITL) | 35 | deterministic | — | partial |
| Unauthorised requests (RBAC) | 55 | deterministic | A | ❌ **assert at the service layer, not via the LLM** |
| Prompt injection (direct + **indirect**) | 220 | deterministic | — | partial |
| Out-of-domain / abstention | 120 | deterministic | X | ✅ sampled |

**Two things to notice.** First, the largest and most rigorous set — the 300-query retrieval benchmark — costs **zero LLM quota** and can run thousands of times. That is deliberate, and it is where the strongest claims in this project will come from. Second, RBAC and unauthorised-request tests assert **at the service layer**, which both costs nothing and is the correct test: §15 says the LLM is not the authorization authority, so testing authorization through the LLM would be testing the wrong component.

### 4.1 Adversarial set composition (220 items)

| Source | Count | Type |
|---|---:|---|
| `deepset/prompt-injections` (Apache-2.0) | 116 | Direct, external baseline |
| Authored direct attacks (spec §17 examples + variants) | 40 | Direct |
| **Authored indirect — poisoned KB document** | 25 | **Indirect (G8)** |
| **Authored indirect — malicious ticket `description` / comment** | 25 | **Indirect — the highest-realism attack** |
| **Authored indirect — tool output / incident `workaround`** | 14 | **Indirect** |

The 64 indirect cases are the ones the spec does not ask for and the ones that would actually matter. Each is paired with an assertion that is *structural*, not textual: authorization was not bypassed, no tool argument was sourced from retrieved text, no disallowed URL was emitted.

---

## 5. Generation method and reproducibility

- **Seeded and deterministic.** A fixed RNG seed plus a pinned generator model and pinned prompts, so `make data` reproduces the corpus byte-for-byte. Without this, "dataset version" is meaningless.
- **Template + LLM hybrid.** Structure (headings, metadata, step numbering, error tables) comes from templates so it is consistent and parseable; prose comes from the LLM so it is realistic. Pure-LLM documents drift in structure and make structure-aware chunking untestable.
- **Generation is a one-off cost, not a per-run cost.** The corpus is generated once and committed. It does **not** consume daily evaluation quota (E5). Budget the generation run across several days or on a paid burst.
- **Distractors and near-duplicates are generated from a shared seed document**, then perturbed along one controlled axis (product name / platform / version). This is what makes them genuinely confusable rather than accidentally different.
- **Injection payloads live in a single registry** (`data/adversarial/payloads.yaml`) and are *injected* into records at build time, so the same payload can be placed in a KB doc, a ticket and a tool output and the three results compared.

---

## 6. Directory layout

```
data/
├── external/
│   ├── DATA-LICENCES.md          # register + retrieval dates + URLs
│   ├── clinc_oos/                # CC-BY-3.0, attribution file alongside
│   └── prompt_injections/        # Apache-2.0
├── generated/
│   ├── documents/                # 542 docs, 14 categories
│   ├── users.jsonl               # 120
│   ├── assets.jsonl              # 240
│   ├── tickets.jsonl             # 620
│   ├── incidents.jsonl           # 55
│   └── conversations.jsonl       # 80
├── adversarial/
│   ├── payloads.yaml             # injection payload registry
│   └── placements.yaml           # where each payload is injected
├── manifests/
│   ├── absent_topics.yaml        # the 30 deliberately-undocumented topics
│   └── corpus_manifest.json      # content hash, seed, generator model, counts
└── eval/
    ├── queries/                  # tagged by difficulty class
    ├── qrels/                    # span-based, role-scoped, graded
    └── VERSIONS.md               # hash → what changed
```

---

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Designed difficulty classes still do not separate the retrieval arms | Medium | Measure early — run the L/S split on a 50-doc pilot **before** generating all 542. If BM25 does not beat dense on L-class at pilot scale, the corpus design is wrong and cheap to fix. |
| Tier-3 human labelling (100 queries) stalls the project | Medium | Front-load 30; publish Tier-2 headline numbers with Tier-3 as a calibration slice that grows. |
| Generation quota (E5) blocks corpus creation | Medium | Generation is a one-off; spread across days, or use templates for the identifier-heavy L-class docs which barely need an LLM at all. |
| Synthetic corpus is too self-consistent, inflating all scores | **High** | This is the residual risk and it must simply be **disclosed**. Synthetic data written by one generator is more internally consistent than a real enterprise wiki. State it in the README limitations section rather than pretending otherwise. |

---

## Sources

- CLINC150 — <https://huggingface.co/datasets/clinc/clinc_oos> (CC-BY-3.0; Larson et al., EMNLP 2019, <https://www.aclweb.org/anthology/D19-1131>)
- deepset/prompt-injections — <https://huggingface.co/datasets/deepset/prompt-injections> (Apache-2.0)
- TechQA — <https://huggingface.co/datasets/rojagtap/tech-qa> (MIT)
- Stack Exchange data dumps — <https://archive.org/details/stackexchange> (CC-BY-SA 4.0; note restricted access since 2024, <https://devclass.com/2024/07/30/stack-exchange-restricts-access-to-dump-of-user-contributed-data-as-critics-complain-license-permits-reuse-for-any-purpose/>)
