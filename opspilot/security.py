"""Authorization, injection defence, approval tokens, rate limiting.

The controlling principle (spec §15): the LLM is never the authorization
authority. It may *request* an action; this module decides. Every check here is
reachable and testable without invoking a model at all, which is also why the
RBAC test suite costs zero API quota.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from opspilot.config import get_settings
from opspilot.core import (
    AuthorizationError,
    Principal,
    RateLimitError,
    Role,
    TaggedText,
    TrustTier,
)

# --------------------------------------------------------------------------
# RBAC
# --------------------------------------------------------------------------
# action -> minimum role. Deliberately a flat, readable table: authorization
# rules that require reading code are authorization rules nobody audits.
ACTION_MIN_ROLE: dict[str, Role] = {
    "search_knowledge": Role.EMPLOYEE,
    "get_document": Role.EMPLOYEE,
    "get_policy": Role.EMPLOYEE,
    "get_ticket": Role.EMPLOYEE,
    "search_tickets": Role.EMPLOYEE,
    "create_ticket": Role.EMPLOYEE,
    "add_ticket_comment": Role.EMPLOYEE,
    "update_ticket": Role.IT_AGENT,
    "escalate_ticket": Role.IT_AGENT,
    "get_incident": Role.EMPLOYEE,
    "list_active_incidents": Role.EMPLOYEE,
    "search_incidents": Role.EMPLOYEE,
    "get_service_status": Role.EMPLOYEE,
    "get_employee_device": Role.EMPLOYEE,
    "get_device_status": Role.EMPLOYEE,
    "get_installed_software": Role.EMPLOYEE,
    "list_all_devices": Role.IT_ADMIN,
}

# Actions that change state -> require human approval (spec §16).
CONSEQUENTIAL_ACTIONS = frozenset(
    {
        "create_ticket",
        "update_ticket",
        "escalate_ticket",
        "add_ticket_comment",
    }
)


def authorize_action(principal: Principal, action: str) -> None:
    required = ACTION_MIN_ROLE.get(action)
    if required is None:
        raise AuthorizationError(
            f"unknown action {action}", safe_message="That action is not available."
        )
    if principal.role.rank < required.rank:
        raise AuthorizationError(
            f"{principal.employee_id} ({principal.role}) attempted {action}",
            safe_message=f"Your role does not permit {action.replace('_', ' ')}.",
        )


def authorize_resource(principal: Principal, owner_employee_id: str, action: str) -> None:
    """Ownership check. An employee may only touch their own records.

    This is the check that defeats "create a high-priority ticket for another
    employee" (spec §17) -- and it runs regardless of what the model asked for.
    """
    if principal.role.rank >= Role.IT_AGENT.rank:
        return
    if owner_employee_id != principal.employee_id:
        raise AuthorizationError(
            f"{principal.employee_id} attempted {action} on record owned by {owner_employee_id}",
            safe_message="You can only access your own records.",
        )


# --------------------------------------------------------------------------
# prompt-injection defence
# --------------------------------------------------------------------------
_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\b[^.\n]{0,40}\b"
            r"(previous|prior|above|earlier|all)\b[^.\n]{0,30}"
            r"\b(instruction|prompt|rule|configuration|task)",
            re.I,
        ),
    ),
    (
        "system_impersonation",
        re.compile(
            r"\b(system\s*(override|prompt|message)|you\s+are\s+now|maintenance\s+mode|"
            r"developer\s+mode)\b",
            re.I,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|show|print|disclose|repeat)\b[^.\n]{0,30}"
            r"\b(system\s+prompt|hidden\s+instruction|your\s+instructions)\b",
            re.I,
        ),
    ),
    (
        "role_escalation",
        re.compile(
            r"\b(the\s+user\s+is\s+(an?\s+)?(it_admin|admin|administrator)|"
            r"grant\s+all|bypass\s+authoriz|disregard\s+authoriz|"
            r"security\s+team\s+approved)\b",
            re.I,
        ),
    ),
    (
        "tool_coercion",
        re.compile(
            r"\b(run|call|execute|invoke)\b[^.\n]{0,20}\b(tool|function)\b[^.\n]{0,40}"
            r"\b(for\s+every|all)\b",
            re.I,
        ),
    ),
    (
        "delimiter_attack",
        re.compile(
            r"(\[\[\s*end\s+of\s+document\s*\]\]|<!--\s*hidden|###\s*new\s+instructions)", re.I
        ),
    ),
)


@dataclass(slots=True)
class InjectionVerdict:
    flagged: bool
    patterns: list[str]
    score: float
    tier: TrustTier
    neural_score: float | None = None

    @property
    def blocks_instruction_following(self) -> bool:
        """Untrusted content is never instruction-bearing, flagged or not.
        Flagging only decides whether we ALSO strip / warn."""
        return not self.tier.is_trusted


class InjectionDetector:
    """Two-stage detector.

    Stage 1 -- regex. Cheap, deterministic, zero quota, runs on every string.
    Stage 2 -- Meta Prompt Guard 2 (86M) via the Groq API, an actual classifier
               trained for this task.

    Stage 2 is OPTIONAL and OFF by default, deliberately. Running every
    retrieved chunk through a hosted classifier would spend the daily token
    budget on defence alone: one answer retrieves 6 passages, so a naive wiring
    multiplies API calls per turn by seven. It is therefore applied only to the
    USER TURN by default (one call), with retrieved content escalated to it
    only when the regex stage already flagged something.

    When the classifier is unavailable, `neural_score` returns None -- meaning
    "no neural signal", never a fabricated 0.0.
    """

    def __init__(self, provider=None, enable_neural: bool = False):
        self.provider = provider
        self.enable_neural = enable_neural and provider is not None
        self.neural_calls = 0
        self.neural_failures = 0

    def scan(self, text: str, tier: TrustTier, *, allow_neural: bool = False) -> InjectionVerdict:
        hits = [name for name, pat in _INJECTION_PATTERNS if pat.search(text)]
        score = min(1.0, 0.34 * len(hits))
        neural: float | None = None
        if self.enable_neural and (allow_neural or hits):
            neural = self.neural_score(text)
            if neural is not None:
                # The classifier can both confirm and OVERRULE the regex stage.
                score = max(score, neural)
                if neural >= 0.5 and "neural_classifier" not in hits:
                    hits = [*hits, "neural_classifier"]
        return InjectionVerdict(
            flagged=bool(hits), patterns=hits, score=score, tier=tier, neural_score=neural
        )

    def neural_score(self, text: str) -> float | None:
        """P(injection) from Prompt Guard 2. None when unavailable.

        The model has a 512-token context, so long retrieved passages are
        truncated to the head -- injected instructions are overwhelmingly
        placed at a boundary, and a partial scan beats no scan.
        """
        if self.provider is None:
            return None
        try:
            from opspilot.config import get_settings
            from opspilot.llm.providers import LLMMessage

            self.neural_calls += 1
            resp = self.provider.complete(
                [LLMMessage("user", text[:1800])],
                model=get_settings().model_prompt_guard,
                temperature=0.0,
                max_tokens=8,
            )
            raw = (resp.text or "").strip()
            # Prompt Guard 2 on Groq returns a bare probability as the message
            # content, e.g. "0.9995923638343811" for an attack and
            # "0.00035985137219540775" for benign text. Parse the float FIRST:
            # a naive startswith("0") label check reads "0.9995" as benign and
            # silently disables the whole classifier.
            try:
                return max(0.0, min(1.0, float(raw.split()[0])))
            except (ValueError, IndexError):
                pass
            upper = raw.upper()
            if "JAILBREAK" in upper or "INJECTION" in upper:
                return 1.0
            if "BENIGN" in upper:
                return 0.0
            return None
        except Exception:
            self.neural_failures += 1
            return None  # never let the defence layer break the request path

    def scan_many(self, items: list[TaggedText]) -> list[InjectionVerdict]:
        return [self.scan(i.text, i.tier) for i in items]


def sanitize_untrusted(text: str, max_len: int = 4000) -> str:
    """Neutralise structure that could be read as prompt framing.

    This does NOT try to remove malicious meaning -- that is unachievable. It
    removes the *structural* affordances (fake delimiters, HTML comments) and
    relies on trust tiering for the rest.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"(?mi)^\s*(system|assistant|user)\s*:", r"[\1]:", text)
    text = text.replace("```", "'''")
    return text[:max_len]


# --------------------------------------------------------------------------
# approval tokens (human-in-the-loop, spec §16 + audit G10 idempotency)
# --------------------------------------------------------------------------
def action_hash(tool_name: str, arguments: dict, actor: str) -> str:
    payload = json.dumps({"t": tool_name, "a": arguments, "u": actor}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def mint_approval_token(tool_name: str, arguments: dict, actor: str) -> tuple[str, str]:
    """Returns (token, action_hash). The token is bound to the exact action, so
    an approval cannot be replayed against different arguments."""
    settings = get_settings()
    ah = action_hash(tool_name, arguments, actor)
    nonce = hashlib.sha256(f"{ah}{time.time_ns()}".encode()).hexdigest()[:16]
    sig = hmac.new(
        settings.jwt_secret.encode(), f"{ah}{nonce}".encode(), hashlib.sha256
    ).hexdigest()[:32]
    return f"{nonce}.{sig}", ah


def approval_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=get_settings().approval_ttl_seconds)


def idempotency_key_for(token: str, action_hash_value: str) -> str:
    return hashlib.sha256(f"{token}|{action_hash_value}".encode()).hexdigest()[:48]


# --------------------------------------------------------------------------
# rate limiting
# --------------------------------------------------------------------------
class SlidingWindowRateLimiter:
    """In-process sliding window.

    Interface matches what a Redis-backed limiter would expose, so swapping in
    Redis is a class substitution. Redis is unavailable locally (no Docker), and
    an in-process limiter is honestly single-instance-only -- documented rather
    than pretended otherwise.
    """

    def __init__(self, limit_per_minute: int | None = None):
        self.limit = limit_per_minute or get_settings().rate_limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
        now = time.monotonic()
        q = self._hits[key]
        while q and now - q[0] > 60.0:
            q.popleft()
        if len(q) >= self.limit:
            raise RateLimitError(f"rate limit for {key}")
        q.append(now)

    def remaining(self, key: str) -> int:
        return max(0, self.limit - len(self._hits.get(key, ())))
