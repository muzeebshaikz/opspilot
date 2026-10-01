"""Central configuration.

Everything that varies between environments lives here and nowhere else.
Spec §12: no provider is hard-coded; §15: no secret is hard-coded.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACT_DIR = REPO_ROOT / "artifacts"


class EmbeddingBackend(StrEnum):
    """Which embedding implementation to use.

    HASHING requires no model download and no network. It is the default so the
    whole pipeline runs in a locked-down environment (see docs/audit/SPEC-AUDIT.md E6).
    ONNX is the quality option and is a drop-in swap once model files are available.
    """

    HASHING = "hashing"
    ONNX = "onnx"


class RerankerBackend(StrEnum):
    LEXICAL = "lexical"  # no model, deterministic, runs anywhere
    CROSS_ENCODER = "cross_encoder"  # ONNX cross-encoder, needs model files
    NONE = "none"


class LLMBackend(StrEnum):
    FAKE = "fake"  # deterministic, offline, zero quota -- used by CI (§28 / G14)
    GROQ = "groq"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="OPSPILOT_", extra="ignore")

    environment: str = "local"
    log_level: str = "INFO"

    # ---- storage -----------------------------------------------------------
    # SQLite locally (no Docker available); the schema is written to be
    # PostgreSQL-portable so the same models run against Postgres unchanged.
    database_url: str = f"sqlite:///{(DATA_DIR / 'opspilot.db').as_posix()}"
    qdrant_path: str = str(DATA_DIR / "qdrant")
    qdrant_url: str | None = None  # set this to use a real server instead
    collection_name: str = "opspilot_knowledge"

    # ---- retrieval ---------------------------------------------------------
    embedding_backend: EmbeddingBackend = EmbeddingBackend.HASHING
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    # Hashing needs far more dimensions than a neural encoder: with only 384
    # buckets, hash collisions dominate and cosine similarity becomes noise.
    hashing_dim: int = 8192
    reranker_backend: RerankerBackend = RerankerBackend.LEXICAL
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    candidate_k: int = 30  # after fusion, before rerank  (spec §9: 20-30)
    final_k: int = 6  # after rerank, into context   (spec §9: 5-8)
    rrf_k: int = 60  # RRF smoothing constant

    # ---- generation --------------------------------------------------------
    llm_backend: LLMBackend = LLMBackend.FAKE
    groq_api_key: str | None = Field(default=None, repr=False)
    # Task-tiered models (§6/E5: cheap model for high-volume low-difficulty calls)
    # Verified against this account's GET /v1/models on 2026-09-30.
    # llama-3.x is NOT available here -- do not configure it.
    model_router: str = "openai/gpt-oss-20b"
    model_generator: str = "openai/gpt-oss-120b"
    # Judge must be an INDEPENDENT family, not just a different size of the
    # generator, or faithfulness scoring is self-evaluation (audit G5).
    model_judge: str = "qwen/qwen3.8-27b"
    # Meta Prompt Guard 2 -- neural injection classifier, 86M, 512-token context.
    model_prompt_guard: str = "meta-llama/llama-prompt-guard-2-86m"
    # gpt-oss are reasoning models; reasoning tokens bill against the same
    # TPM ceiling, so keep them cheap unless a task needs depth.
    reasoning_effort: str = "low"
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 3

    # ---- grounding / abstention (§10) --------------------------------------
    # Chosen from a measured sweep, not guessed: see
    # artifacts/reports/abstention_calibration.md. Policy is safety-first --
    # cap hallucination risk at 15%, then minimise false abstention.
    # Recalibrated for the 446-document corpus. At 1.5 the wider corpus pushed
    # confident-wrong answers to 26.7%: more documents means more chances that
    # an unanswerable question finds something scoring above the bar. A
    # retrieval threshold is a property of the CORPUS, not a constant, and must
    # be re-swept whenever the corpus changes.
    min_evidence_score: float = 1.6
    # Below min_evidence_score but above this, the system gives a HEDGED answer
    # from the evidence it did find rather than refusing outright. Measured:
    # hard refusals fall 37.0% -> 17.3% with confident-wrong unchanged at 12.1%
    # (artifacts/reports/abstention_policy.json).
    partial_evidence_score: float = 0.85
    # Share of the question's IDF weight allowed to sit on terms that appear in
    # NO document. Above this, refuse at any score -- the user is asking about
    # something the corpus has never heard of.
    max_oov_mass: float = 0.34
    min_evidence_chunks: int = 1

    # ---- quota guard (§6/E5 -- the binding constraint) ---------------------
    daily_token_budget: int = 200_000
    tokens_per_minute_budget: int = 8_000

    # ---- MCP (spec §7 / V7) ------------------------------------------------
    # Routes tool calls through MCP servers instead of calling the service layer
    # in-process. Measured cost: ~640 ms mean per call vs ~0.3 ms in-process,
    # because stdio MCP spawns a subprocess per (domain, principal). Kept
    # configurable so the overhead is a choice, not an accident.
    mcp_enabled: bool = True
    mcp_max_sessions: int = 8
    mcp_timeout_s: float = 20.0

    # ---- injection defence --------------------------------------------------
    # Prompt Guard 2 via API. Token-free (usage reports 0) but RPM-limited, so
    # it is applied to the user turn always, and to retrieved content only when
    # the regex stage already flagged something.
    neural_injection_enabled: bool = True

    # ---- security ----------------------------------------------------------
    jwt_secret: str = Field(default="dev-only-not-a-secret", repr=False)
    approval_ttl_seconds: int = 300
    rate_limit_per_minute: int = 30

    # ---- observability (spec §21) -------------------------------------------
    # Tracing is a no-op unless an endpoint is set, so nothing is required to
    # run locally and a broken collector can never take the service down.
    otel_enabled: bool = True
    otel_endpoint: str | None = None  # e.g. http://localhost:4318/v1/traces
    otel_console: bool = False
    prometheus_enabled: bool = True

    # ---- memory (spec §14) ---------------------------------------------------
    # Summarisation costs one LLM call, so it runs on a threshold, never per turn.
    long_term_memory_enabled: bool = True

    # ---- determinism -------------------------------------------------------
    seed: int = 20260929

    @property
    def offline(self) -> bool:
        return self.llm_backend is LLMBackend.FAKE


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def enable_corporate_tls() -> None:
    """Use the OS trust store instead of certifi.

    Measured on the target machine: api.groq.com fails CERTIFICATE_VERIFY_FAILED
    with certifi because corporate TLS interception re-signs with a private root
    CA that certifi does not carry. truststore delegates to the Windows cert
    store, which does carry it. Without this, Groq is unreachable.
    """
    if os.environ.get("OPSPILOT_SKIP_TRUSTSTORE"):
        return
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - best effort, never fatal
        pass
