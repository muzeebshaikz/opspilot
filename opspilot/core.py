"""Cross-cutting primitives: identity, trust tiers, errors, structured logging.

The TrustTier enum here is the mechanism that closes audit finding G8
(indirect prompt injection). Every string that reaches an LLM prompt must be
carried in a TaggedText so its provenance is known at render time.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

# --------------------------------------------------------------------------
# request context
# --------------------------------------------------------------------------
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")
user_id_var: ContextVar[str] = ContextVar("user_id", default="-")
conversation_id_var: ContextVar[str] = ContextVar("conversation_id", default="-")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def utcnow() -> datetime:
    return datetime.now(UTC)


# --------------------------------------------------------------------------
# roles & trust
# --------------------------------------------------------------------------
class Role(StrEnum):
    EMPLOYEE = "employee"
    IT_AGENT = "IT_agent"
    IT_ADMIN = "IT_admin"

    @property
    def rank(self) -> int:
        return {"employee": 1, "IT_agent": 2, "IT_admin": 3}[self.value]

    def can_see(self, access_level: str) -> bool:
        return self.rank >= Role(access_level).rank


class TrustTier(StrEnum):
    """Provenance of a piece of text entering the model context.

    SYSTEM and USER_TURN are instruction-bearing. Everything else is DATA and
    must never be interpreted as an instruction (audit G8).
    """

    SYSTEM = "system"
    USER_TURN = "user_turn"
    RETRIEVED_KB = "retrieved_kb"
    TOOL_OUTPUT = "tool_output"
    USER_GENERATED_RECORD = "user_generated_record"  # ticket bodies, comments

    @property
    def is_trusted(self) -> bool:
        return self in (TrustTier.SYSTEM, TrustTier.USER_TURN)


@dataclass(frozen=True, slots=True)
class TaggedText:
    """Text plus where it came from. The context builder refuses raw str."""

    text: str
    tier: TrustTier
    source_id: str | None = None


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated actor.

    Constructed only from a verified session. Never from model output.
    This object -- not any tool argument -- is the authorization subject (G9).
    """

    employee_id: str
    name: str
    role: Role
    department: str = ""

    @property
    def is_staff(self) -> bool:
        return self.role in (Role.IT_AGENT, Role.IT_ADMIN)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------
class OpsPilotError(Exception):
    """Base. Carries a safe message; internals never leak to the client (§15)."""

    code = "internal_error"
    http_status = 500
    safe_message = "An internal error occurred."

    def __init__(self, detail: str = "", *, safe_message: str | None = None):
        super().__init__(detail or self.safe_message)
        self.detail = detail
        if safe_message:
            self.safe_message = safe_message


class AuthorizationError(OpsPilotError):
    code = "forbidden"
    http_status = 403
    safe_message = "You are not authorized to perform that action."


class ValidationError(OpsPilotError):
    code = "invalid_request"
    http_status = 422
    safe_message = "The request was not valid."


class NotFoundError(OpsPilotError):
    code = "not_found"
    http_status = 404
    safe_message = "The requested resource was not found."


class RateLimitError(OpsPilotError):
    code = "rate_limited"
    http_status = 429
    safe_message = "Too many requests. Please retry shortly."


class QuotaExhaustedError(OpsPilotError):
    code = "quota_exhausted"
    http_status = 503
    safe_message = "The daily model token budget is exhausted."


class ProviderError(OpsPilotError):
    code = "provider_unavailable"
    http_status = 502
    safe_message = "An upstream model provider is unavailable."


# --------------------------------------------------------------------------
# structured logging
# --------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "user_id": user_id_var.get(),
            "conversation_id": conversation_id_var.get(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, msg: str, **fields: Any) -> None:
    logger.info(msg, extra={"extra_fields": fields})


# --------------------------------------------------------------------------
# lightweight span timing (OpenTelemetry-shaped, no dependency)
# --------------------------------------------------------------------------
@dataclass
class Span:
    name: str
    start: float = field(default_factory=time.perf_counter)
    attributes: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def set(self, **kw: Any) -> Span:
        self.attributes.update(kw)
        return self

    def finish(self) -> Span:
        self.duration_ms = (time.perf_counter() - self.start) * 1000
        return self


class Trace:
    """Collects spans for one request. Rendered into the SSE activity feed (§24)
    and into the observability record (§21)."""

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or new_id("req")
        self.spans: list[Span] = []

    def span(self, name: str, **attrs: Any) -> Span:
        s = Span(name=name, attributes=dict(attrs))
        self.spans.append(s)
        return s

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "spans": [
                {"name": s.name, "duration_ms": round(s.duration_ms or 0.0, 2), **s.attributes}
                for s in self.spans
            ],
            "total_ms": round(sum(s.duration_ms or 0.0 for s in self.spans), 2),
        }
