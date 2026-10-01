"""Persistence layer (spec §13).

SQLite locally because Docker is unavailable on the target machine; the schema
is deliberately PostgreSQL-portable (no SQLite-only types, explicit lengths,
timezone-aware timestamps stored as ISO strings) so moving to Postgres is a
connection-string change, not a rewrite.

The vector store is NOT used for any of this. Structured, transactional,
relational data belongs here -- spec §13: "Do not use the vector database as a
replacement for PostgreSQL."
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from opspilot.config import get_settings
from opspilot.core import new_id


class Base(DeclarativeBase):
    pass


class UserRow(Base):
    __tablename__ = "users"
    employee_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    department: Mapped[str] = mapped_column(String(64))
    role: Mapped[str] = mapped_column(String(16), index=True)
    location: Mapped[str] = mapped_column(String(64))
    manager: Mapped[str | None] = mapped_column(String(16), nullable=True)
    access_level: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="active")
    employment_type: Mapped[str] = mapped_column(String(16), default="permanent")


class AssetRow(Base):
    __tablename__ = "assets"
    asset_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    employee_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("users.employee_id"), index=True
    )
    hostname: Mapped[str] = mapped_column(String(64))
    operating_system: Mapped[str] = mapped_column(String(32))
    os_version: Mapped[str] = mapped_column(String(32))
    vpn_client_version: Mapped[str] = mapped_column(String(32))
    installed_software: Mapped[str] = mapped_column(Text)  # JSON array
    status: Mapped[str] = mapped_column(String(16))
    last_seen: Mapped[str] = mapped_column(String(40))


class TicketRow(Base):
    __tablename__ = "tickets"
    ticket_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    employee_id: Mapped[str] = mapped_column(
        String(16), ForeignKey("users.employee_id"), index=True
    )
    category: Mapped[str] = mapped_column(String(32), index=True)
    priority: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(24), index=True)
    summary: Mapped[str] = mapped_column(String(256))
    description: Mapped[str] = mapped_column(Text)
    assigned_team: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[str] = mapped_column(String(40))
    updated_at: Mapped[str] = mapped_column(String(40))
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    contains_injection: Mapped[bool] = mapped_column(Boolean, default=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class IncidentRow(Base):
    __tablename__ = "incidents"
    incident_id: Mapped[str] = mapped_column(String(24), primary_key=True)
    service: Mapped[str] = mapped_column(String(64), index=True)
    severity: Mapped[str] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(16), index=True)
    affected_users: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[str] = mapped_column(String(40))
    description: Mapped[str] = mapped_column(Text)
    workaround: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolved_at: Mapped[str | None] = mapped_column(String(40), nullable=True)


class ConversationRow(Base):
    __tablename__ = "conversations"
    conversation_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    employee_id: Mapped[str] = mapped_column(String(16), index=True)
    title: Mapped[str] = mapped_column(String(256), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)  # long-term memory (§14)


class MessageRow(Base):
    __tablename__ = "messages"
    message_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("conversations.conversation_id"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | system
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    trace: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class AuditLogRow(Base):
    """Append-only. Every consequential action lands here (spec §15, §18)."""

    __tablename__ = "audit_logs"
    audit_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(32), index=True)
    actor_employee_id: Mapped[str] = mapped_column(String(16), index=True)
    actor_role: Mapped[str] = mapped_column(String(16))
    action: Mapped[str] = mapped_column(String(64), index=True)
    target: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision: Mapped[str] = mapped_column(String(16))  # allowed | denied
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    arguments: Mapped[str | None] = mapped_column(Text, nullable=True)  # redacted JSON
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class EvaluationRunRow(Base):
    __tablename__ = "evaluation_runs"
    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(16))
    label: Mapped[str] = mapped_column(String(64))
    dataset_hash: Mapped[str] = mapped_column(String(64))
    corpus_hash: Mapped[str] = mapped_column(String(64))
    config: Mapped[str] = mapped_column(Text)
    metrics: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class ApprovalRow(Base):
    """Single-use, time-limited, bound to an exact action hash (audit G10)."""

    __tablename__ = "approvals"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String(32), index=True)
    actor_employee_id: Mapped[str] = mapped_column(String(16))
    action_hash: Mapped[str] = mapped_column(String(64))
    tool_name: Mapped[str] = mapped_column(String(48))
    arguments: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(String(512))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    consumed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    result_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)


_engine = None
_Session: sessionmaker | None = None


def init_engine(url: str | None = None, echo: bool = False):
    global _engine, _Session
    settings = get_settings()
    _engine = create_engine(
        url or settings.database_url,
        echo=echo,
        connect_args={"check_same_thread": False}
        if (url or settings.database_url).startswith("sqlite")
        else {},
    )
    _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    Base.metadata.create_all(_engine)
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    if _Session is None:
        init_engine()
    assert _Session is not None
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def audit(
    session: Session,
    *,
    request_id: str,
    actor: str,
    role: str,
    action: str,
    decision: str,
    target: str | None = None,
    reason: str | None = None,
    arguments: dict | None = None,
) -> None:
    session.add(
        AuditLogRow(
            audit_id=new_id("aud"),
            request_id=request_id,
            actor_employee_id=actor,
            actor_role=role,
            action=action,
            target=target,
            decision=decision,
            reason=reason,
            arguments=json.dumps(_redact(arguments or {}))[:4000],
        )
    )


_SENSITIVE = {"password", "token", "secret", "api_key", "authorization"}


def _redact(d: dict) -> dict:
    """Never write a secret into the audit table (spec §15 error sanitisation)."""
    out = {}
    for k, v in d.items():
        if any(s in k.lower() for s in _SENSITIVE):
            out[k] = "***redacted***"
        elif isinstance(v, dict):
            out[k] = _redact(v)
        elif isinstance(v, str) and len(v) > 500:
            out[k] = v[:500] + "...<truncated>"
        else:
            out[k] = v
    return out


def seed_from_generated(data) -> dict:
    """Load generated enterprise data into the database. Idempotent."""
    counts = {"users": 0, "assets": 0, "tickets": 0, "incidents": 0}
    with session_scope() as s:
        if s.scalar(select(UserRow).limit(1)) is not None:
            return {"skipped": "already seeded"}
        for u in data.users:
            s.add(UserRow(**asdict(u)))
            counts["users"] += 1
        for a in data.assets:
            d = asdict(a)
            d["installed_software"] = json.dumps(d["installed_software"])
            s.add(AssetRow(**d))
            counts["assets"] += 1
        for t in data.tickets:
            s.add(TicketRow(**asdict(t)))
            counts["tickets"] += 1
        for i in data.incidents:
            s.add(IncidentRow(**asdict(i)))
            counts["incidents"] += 1
    return counts


class FeedbackRow(Base):
    """Thumbs up/down on an answer.

    This is not a vanity metric. It is the only signal in the system that comes
    from a real user rather than from a synthetic label, which makes it the
    seed of a genuine evaluation set: a thumbs-down on an answer the system
    scored as confident is exactly the case worth adding to the regression set.
    """

    __tablename__ = "feedback"
    feedback_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    # Stable per (conversation, question). Rating a turn twice must UPDATE this
    # row, not insert another -- otherwise the satisfaction metric counts one
    # user clicking repeatedly as many opinions.
    turn_key: Mapped[str] = mapped_column(String(64), index=True)
    conversation_id: Mapped[str] = mapped_column(String(32), index=True)
    message_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    employee_id: Mapped[str] = mapped_column(String(16), index=True)
    rating: Mapped[str] = mapped_column(String(8))  # up | down
    reason: Mapped[str | None] = mapped_column(String(256), nullable=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(String(32))
    citations: Mapped[int] = mapped_column(Integer, default=0)
    retracted: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
