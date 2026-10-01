"""LLM provider abstraction (spec §12) with a hard token-budget guard.

Two backends:
  FakeLLMProvider -- deterministic, offline, zero quota. This is what CI uses,
                     which is why the whole test suite costs nothing (audit G14).
  GroqProvider    -- OpenAI-compatible chat completions against api.groq.com.

The TokenBudget is not decoration. Measured free-tier limits are 30 RPM /
1,000 RPD / 8,000 TPM / 200,000 TPD, which at ~3.5k prompt tokens per grounded
answer is roughly two requests per minute and ~55 answers per day. A system that
discovers this by getting 429s in front of a user is badly built; the budget is
checked before the call and surfaced as a first-class metric.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol

from opspilot.config import LLMBackend, Settings, enable_corporate_tls, get_settings
from opspilot.core import ProviderError, QuotaExhaustedError, get_logger, log_event

log = get_logger("llm")


@dataclass(slots=True)
class LLMMessage:
    role: str  # system | user | assistant
    content: str


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(Protocol):
    name: str

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse: ...


def estimate_tokens(text: str) -> int:
    """~4 characters per token. Deliberately an ESTIMATE and named as one; the
    authoritative count comes back from the provider's usage field."""
    return max(1, len(text) // 4)


# --------------------------------------------------------------------------
# budget guard
# --------------------------------------------------------------------------
@dataclass
class TokenBudget:
    tpm_limit: int
    tpd_limit: int
    rpm_limit: int = 30
    _minute: deque[tuple[float, int]] = field(default_factory=deque)
    _day_tokens: int = 0
    _day_start: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def _roll(self) -> None:
        now = time.time()
        while self._minute and now - self._minute[0][0] > 60:
            self._minute.popleft()
        if now - self._day_start > 86400:
            self._day_start, self._day_tokens = now, 0

    def check(self, estimated: int) -> None:
        with self._lock:
            self._roll()
            minute_tokens = sum(t for _, t in self._minute)
            if self._day_tokens + estimated > self.tpd_limit:
                raise QuotaExhaustedError(
                    f"daily budget {self._day_tokens}/{self.tpd_limit}",
                    safe_message="The daily model token budget is exhausted. "
                    "Retrieval-only answers remain available.",
                )
            if minute_tokens + estimated > self.tpm_limit:
                raise QuotaExhaustedError(
                    f"per-minute budget {minute_tokens}/{self.tpm_limit}",
                    safe_message="Model rate limit reached. Please retry in a moment.",
                )
            if len(self._minute) >= self.rpm_limit:
                raise QuotaExhaustedError(
                    "requests per minute exceeded",
                    safe_message="Too many model requests. Retry shortly.",
                )

    def record(self, used: int) -> None:
        with self._lock:
            self._roll()
            self._minute.append((time.time(), used))
            self._day_tokens += used

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._roll()
            return {
                "tokens_this_minute": sum(t for _, t in self._minute),
                "tpm_limit": self.tpm_limit,
                "tokens_today": self._day_tokens,
                "tpd_limit": self.tpd_limit,
                "requests_this_minute": len(self._minute),
                "daily_remaining": max(0, self.tpd_limit - self._day_tokens),
            }


# --------------------------------------------------------------------------
# fake provider
# --------------------------------------------------------------------------
class FakeLLMProvider:
    """Deterministic provider used by CI, tests and offline operation.

    It does NOT pretend to be a language model. It performs extractive
    grounding: it selects the sentences from the supplied context that best
    overlap the question and returns them with citation markers. That makes the
    surrounding pipeline (retrieval, citation, abstention, approval, tools)
    fully exercisable and end-to-end testable without any API access, and it
    never invents content, so faithfulness is trivially satisfied.
    """

    name = "fake"

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        question = next((m.content for m in reversed(messages) if m.role == "user"), "")
        context = "\n".join(m.content for m in messages if m.role == "system")

        # Routing / classification calls expect a compact JSON answer.
        if "Respond with JSON" in context or "classify" in context.lower():
            return self._classify(question, context, t0)

        evidence = self._extract(question, context)
        if not evidence:
            text = (
                "I could not find enough evidence in the IT knowledge base to answer "
                "that reliably. Please rephrase, or raise a ticket and a human agent "
                "will pick it up."
            )
        else:
            text = (
                "Based on the retrieved documentation:\n\n"
                + "\n".join(f"- {s}" for s in evidence[:4])
                + "\n\nSee the cited sources for the full procedure."
            )
        return LLMResponse(
            text=text,
            model="fake-extractive-v1",
            prompt_tokens=estimate_tokens(context + question),
            completion_tokens=estimate_tokens(text),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    # Weighted cues rather than first-match. First-match ordering mis-routes
    # compound questions: "my VPN is broken, is there an outage and is my laptop
    # affected?" carries symptom + incident + device signals at once, and only
    # the troubleshooting branch fans out to all three.
    _CUES: tuple[tuple[str, tuple[str, ...], float], ...] = (
        (
            "restricted",
            (
                "every employee",
                "all employees",
                "everyone",
                "all devices",
                "every device",
                "all users",
                "other employees",
                "everybody",
                "all tickets",
                "every user",
            ),
            4.0,
        ),
        (
            "action",
            (
                "create a",
                "create ticket",
                "raise a ticket",
                "open a ticket",
                "escalate",
                "add these logs",
                "add this to my ticket",
            ),
            3.0,
        ),
        (
            "ticket",
            ("inc-", "my ticket", "my tickets", "status of", "open tickets", "ticket i created"),
            2.5,
        ),
        (
            "incident",
            (
                "outage",
                "incident",
                "is it down",
                "service down",
                "other employees",
                "anyone else",
                "affecting",
            ),
            2.0,
        ),
        (
            "troubleshooting",
            (
                "not working",
                "isn't working",
                "keeps",
                "cannot",
                "can't",
                "broken",
                "fails",
                "failing",
                "stopped",
                "error",
                "issue",
                "problem",
                "won't",
                "disconnect",
                "slow",
                "my laptop",
                "my device",
                "check my",
            ),
            2.0,
        ),
        (
            "knowledge",
            (
                "how do i",
                "what is",
                "where do i",
                "policy",
                "guide",
                "procedure",
                "request access",
                "install",
            ),
            1.5,
        ),
    )

    @classmethod
    def _classify(cls, question: str, context: str, t0: float) -> LLMResponse:
        q = question.lower()
        scores: dict[str, float] = {}
        for intent, cues, weight in cls._CUES:
            hits = sum(1 for c in cues if c in q)
            if hits:
                scores[intent] = scores.get(intent, 0.0) + weight * hits

        if len(q.split()) <= 2 and not scores:
            intent, confidence = "ambiguous", 0.6
        elif not scores:
            intent, confidence = "knowledge", 0.5
        else:
            # A symptom plus an incident/device probe is an investigation, and the
            # troubleshooting branch is the only one that fans out to RAG +
            # incidents + assets together.
            if "restricted" in scores:
                intent = "restricted"
            elif "troubleshooting" in scores and ("incident" in scores or "check my" in q):
                intent = "troubleshooting"
            else:
                intent = max(scores, key=lambda k: scores[k])
            total = sum(scores.values())
            confidence = round(min(0.95, scores.get(intent, 0) / total), 2) if total else 0.5
        payload = json.dumps({"intent": intent, "confidence": confidence})
        return LLMResponse(
            text=payload,
            model="fake-extractive-v1",
            prompt_tokens=estimate_tokens(context + question),
            completion_tokens=estimate_tokens(payload),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _extract(question: str, context: str) -> list[str]:
        q_words = {w for w in re.findall(r"[a-z0-9_]+", question.lower()) if len(w) > 2}
        if not q_words:
            return []
        scored: list[tuple[float, str]] = []
        for raw in re.split(r"(?<=[.!?])\s+|\n", context):
            s = raw.strip(" -*#|")
            if len(s) < 35 or s.startswith("["):
                continue
            words = {w for w in re.findall(r"[a-z0-9_]+", s.lower()) if len(w) > 2}
            if not words:
                continue
            overlap = len(q_words & words) / len(q_words)
            if overlap > 0.12:
                scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        seen, out = set(), []
        for _, s in scored:
            key = s[:60]
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out


# --------------------------------------------------------------------------
# groq provider
# --------------------------------------------------------------------------
class GroqProvider:
    """OpenAI-compatible chat completions.

    Retries on 429/5xx with exponential backoff and honours `retry-after`.
    Reads the corporate trust store, because certifi alone cannot verify
    api.groq.com through TLS interception on the target network.
    """

    name = "groq"
    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, settings: Settings | None = None, budget: TokenBudget | None = None):
        import httpx

        self.settings = settings or get_settings()
        if not self.settings.groq_api_key:
            raise ProviderError(
                "OPSPILOT_GROQ_API_KEY is not set",
                safe_message="The model provider is not configured.",
            )
        enable_corporate_tls()
        self.budget = budget or TokenBudget(
            tpm_limit=self.settings.tokens_per_minute_budget,
            tpd_limit=self.settings.daily_token_budget,
        )
        self._client = httpx.Client(timeout=self.settings.llm_timeout_s)

    # gpt-oss models are REASONING models: they spend completion tokens on an
    # internal `reasoning` field before emitting `content`. A small max_tokens
    # is therefore consumed entirely by reasoning and `content` comes back
    # EMPTY -- which looks like a broken provider but is a budgeting mistake.
    # Measured: a 12-token cap returned '' while burning 27 reasoning tokens.
    REASONING_MODELS = ("gpt-oss",)
    MIN_COMPLETION_TOKENS = 256

    def _is_reasoning(self, model: str) -> bool:
        return any(m in model for m in self.REASONING_MODELS)

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        model = model or self.settings.model_generator
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature,
        }
        # Always `max_completion_tokens`, never the legacy `max_tokens`.
        # Measured: qwen/qwen3.8-27b returns 404 for a request carrying
        # `max_tokens`, which reads like "model not found" and is really
        # "parameter not supported". The modern field works for every model here.
        effective_cap = (
            max(max_tokens, self.MIN_COMPLETION_TOKENS) if self._is_reasoning(model) else max_tokens
        )
        payload["max_completion_tokens"] = effective_cap
        if self._is_reasoning(model):
            # Reasoning tokens bill against the same 8k TPM ceiling, so keep
            # them cheap unless a task genuinely needs depth.
            payload["reasoning_effort"] = reasoning_effort or self.settings.reasoning_effort

        estimated = sum(estimate_tokens(m.content) for m in messages) + effective_cap
        self.budget.check(estimated)

        last_exc: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            t0 = time.perf_counter()
            try:
                r = self._client.post(
                    self.BASE_URL,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.settings.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                )
                if r.status_code == 429:
                    wait = float(r.headers.get("retry-after", 2**attempt))
                    log_event(log, "groq rate limited", attempt=attempt, wait_s=wait)
                    time.sleep(min(wait, 20))
                    continue
                if r.status_code >= 500:
                    time.sleep(2**attempt)
                    continue
                r.raise_for_status()
                body = r.json()
                usage = body.get("usage", {})
                choice = body["choices"][0]
                used = usage.get("total_tokens", estimated)
                self.budget.record(used)
                text = (choice["message"].get("content") or "").strip()
                reasoning_tokens = (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens", 0
                )
                if not text:
                    # Surfaced rather than swallowed: an empty answer here almost
                    # always means the completion cap was spent on reasoning.
                    log_event(
                        log,
                        "empty content from model",
                        model=model,
                        finish_reason=choice.get("finish_reason"),
                        reasoning_tokens=reasoning_tokens,
                        cap=effective_cap,
                    )
                log_event(
                    log,
                    "groq ok",
                    model=model,
                    tokens=used,
                    reasoning_tokens=reasoning_tokens,
                    remaining_today=self.budget.snapshot()["daily_remaining"],
                )
                return LLMResponse(
                    text=text,
                    model=model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    reasoning_tokens=reasoning_tokens,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    finish_reason=choice.get("finish_reason", "stop"),
                )
            except Exception as exc:
                last_exc = exc
                if attempt == self.settings.llm_max_retries - 1:
                    break
                time.sleep(2**attempt)
        raise ProviderError(f"groq failed after retries: {last_exc}")


# --------------------------------------------------------------------------
# response cache (spec §22: the honest justification is quota, not latency)
# --------------------------------------------------------------------------
class ResponseCache:
    """Exact-match response cache with hit accounting.

    Under a 200k token/day ceiling a cache hit does not merely save milliseconds,
    it returns budget. The metrics that matter are therefore `llm_calls_avoided`
    and `tokens_saved`, both counted here, not a latency claim.

    In-process today; the interface matches a Redis-backed implementation.
    """

    def __init__(self, max_entries: int = 2048):
        self._store: dict[str, LLMResponse] = {}
        self._max = max_entries
        self.hits = 0
        self.misses = 0
        self.tokens_saved = 0

    @staticmethod
    def key(messages: list[LLMMessage], model: str) -> str:
        blob = json.dumps([[m.role, m.content] for m in messages] + [model], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> LLMResponse | None:
        hit = self._store.get(k)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        self.tokens_saved += hit.total_tokens
        from dataclasses import replace

        return replace(hit, cached=True)

    def put(self, k: str, resp: LLMResponse) -> None:
        if len(self._store) >= self._max:
            self._store.pop(next(iter(self._store)))
        self._store[k] = resp

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total) if total else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "llm_calls_avoided": self.hits,
            "tokens_saved": self.tokens_saved,
        }


class CachedProvider:
    """Decorator that adds caching + budget accounting to any provider."""

    def __init__(self, inner: LLMProvider, cache: ResponseCache | None = None):
        self.inner = inner
        self.cache = cache or ResponseCache()
        self.name = f"cached:{inner.name}"

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 700,
    ) -> LLMResponse:
        k = ResponseCache.key(messages, model or "default")
        if temperature <= 0.3 and (hit := self.cache.get(k)) is not None:
            return hit
        resp = self.inner.complete(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        if temperature <= 0.3:
            self.cache.put(k, resp)
        return resp


def build_provider(settings: Settings | None = None) -> LLMProvider:
    s = settings or get_settings()
    if s.llm_backend is LLMBackend.GROQ:
        try:
            return CachedProvider(GroqProvider(s))
        except ProviderError as exc:
            log_event(log, "groq unavailable, using fake provider", error=str(exc))
    return CachedProvider(FakeLLMProvider())
