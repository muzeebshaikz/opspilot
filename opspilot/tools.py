"""Typed application tools (spec §5/V5, §18).

Every tool has: a Pydantic input schema, a typed output, an authorization check,
a resource-ownership check, error handling, a timeout, structured logging and an
audit record for consequential actions.

The most important design rule here (audit G9):
    NO TOOL TAKES AN IDENTITY ARGUMENT.
The acting principal is ambient, supplied by the authenticated session through
ToolContext. If `employee_id` were a tool parameter, the model would control it,
and "create a ticket for another employee" would succeed at the data layer no
matter what the graph believed. A unit test enforces this invariant.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import select

from opspilot.core import (
    AuthorizationError,
    NotFoundError,
    OpsPilotError,
    Principal,
    TrustTier,
    get_logger,
    log_event,
)
from opspilot.db import AssetRow, IncidentRow, TicketRow, audit, session_scope
from opspilot.security import (
    CONSEQUENTIAL_ACTIONS,
    authorize_action,
    authorize_resource,
    sanitize_untrusted,
)

log = get_logger("tools")


@dataclass(slots=True)
class ToolContext:
    """Ambient, verified execution context. Never model-writable."""

    principal: Principal
    request_id: str
    conversation_id: str = "-"
    approved_action_hash: str | None = None
    idempotency_key: str | None = None


@dataclass(slots=True)
class ToolResult:
    ok: bool
    tool: str
    data: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    trust_tier: TrustTier = TrustTier.TOOL_OUTPUT
    requires_approval: bool = False
    approval_description: str | None = None

    def to_json(self) -> dict:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 2),
        }


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------
class GetTicketIn(BaseModel):
    ticket_id: str = Field(pattern=r"^INC-\d{4,6}$")


class SearchTicketsIn(BaseModel):
    status: str | None = Field(default=None, max_length=24)
    category: str | None = Field(default=None, max_length=32)
    limit: int = Field(default=10, ge=1, le=50)


class CreateTicketIn(BaseModel):
    category: str = Field(max_length=32)
    priority: str = Field(pattern=r"^(Low|Medium|High|Critical)$")
    summary: str = Field(min_length=3, max_length=256)
    description: str = Field(min_length=3, max_length=4000)
    assigned_team: str = Field(default="Service Desk", max_length=64)


class AddCommentIn(BaseModel):
    ticket_id: str = Field(pattern=r"^INC-\d{4,6}$")
    comment: str = Field(min_length=1, max_length=2000)


class EscalateTicketIn(BaseModel):
    ticket_id: str = Field(pattern=r"^INC-\d{4,6}$")
    reason: str = Field(min_length=3, max_length=512)


class GetIncidentIn(BaseModel):
    incident_id: str = Field(pattern=r"^INCIDENT-\d{1,6}$")


class SearchIncidentsIn(BaseModel):
    service: str | None = Field(default=None, max_length=64)
    status: str | None = Field(default=None, max_length=16)


class ServiceStatusIn(BaseModel):
    service: str = Field(max_length=64)


class EmptyIn(BaseModel):
    pass


class SearchKnowledgeIn(BaseModel):
    query: str = Field(min_length=2, max_length=512)
    category: str | None = Field(default=None, max_length=32)
    top_k: int = Field(default=6, ge=1, le=20)


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
class Tool:
    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_model: ClassVar[type[BaseModel]] = EmptyIn
    timeout_s: ClassVar[float] = 5.0

    def run(self, ctx: ToolContext, args: BaseModel) -> Any:  # pragma: no cover
        raise NotImplementedError

    # -- execution wrapper: validate -> authorize -> execute -> audit -------
    def __call__(self, ctx: ToolContext, raw_args: dict) -> ToolResult:
        t0 = time.perf_counter()
        try:
            args = self.input_model.model_validate(raw_args)
        except PydanticValidationError as exc:
            return ToolResult(
                ok=False,
                tool=self.name,
                error=f"invalid arguments: {exc.error_count()} problem(s)",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        try:
            authorize_action(ctx.principal, self.name)
            data = self.run(ctx, args)
            elapsed = (time.perf_counter() - t0) * 1000
            if elapsed > self.timeout_s * 1000:
                log_event(log, "tool exceeded budget", tool=self.name, ms=elapsed)
            self._audit(ctx, args, "allowed", None)
            log_event(
                log,
                "tool ok",
                tool=self.name,
                ms=round(elapsed, 2),
                actor=ctx.principal.employee_id,
            )
            return ToolResult(ok=True, tool=self.name, data=data, latency_ms=elapsed)
        except OpsPilotError as exc:
            self._audit(ctx, args, "denied", exc.detail or str(exc))
            log_event(
                log, "tool denied", tool=self.name, reason=exc.code, actor=ctx.principal.employee_id
            )
            return ToolResult(
                ok=False,
                tool=self.name,
                error=exc.safe_message,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as exc:  # never leak internals (spec §15)
            log.exception("tool crashed: %s", self.name)
            self._audit(ctx, args, "error", type(exc).__name__)
            return ToolResult(
                ok=False,
                tool=self.name,
                error="The tool failed.",
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

    def _audit(self, ctx: ToolContext, args: BaseModel, decision: str, reason: str | None) -> None:
        if self.name not in CONSEQUENTIAL_ACTIONS and decision == "allowed":
            return  # read-only successes are logged, not audited
        with session_scope() as s:
            audit(
                s,
                request_id=ctx.request_id,
                actor=ctx.principal.employee_id,
                role=ctx.principal.role.value,
                action=self.name,
                decision=decision,
                reason=reason,
                arguments=args.model_dump(),
            )


# --------------------------------------------------------------------------
# ticketing
# --------------------------------------------------------------------------
def _ticket_json(t: TicketRow, *, redact_description: bool = True) -> dict:
    return {
        "ticket_id": t.ticket_id,
        "employee_id": t.employee_id,
        "category": t.category,
        "priority": t.priority,
        "status": t.status,
        "summary": t.summary,
        # Ticket bodies are user-written and therefore untrusted (audit G8).
        "description": sanitize_untrusted(t.description) if redact_description else t.description,
        "assigned_team": t.assigned_team,
        "created_at": t.created_at,
        "updated_at": t.updated_at,
        "resolution": t.resolution,
    }


class GetTicket(Tool):
    name = "get_ticket"
    description = "Fetch one ticket by its identifier."
    input_model = GetTicketIn

    def run(self, ctx: ToolContext, args: GetTicketIn) -> dict:
        with session_scope() as s:
            row = s.get(TicketRow, args.ticket_id)
            if row is None:
                raise NotFoundError(f"ticket {args.ticket_id}")
            authorize_resource(ctx.principal, row.employee_id, self.name)
            return _ticket_json(row)


class SearchTickets(Tool):
    name = "search_tickets"
    description = "Search the caller's tickets (staff may search all)."
    input_model = SearchTicketsIn

    def run(self, ctx: ToolContext, args: SearchTicketsIn) -> list[dict]:
        with session_scope() as s:
            stmt = select(TicketRow)
            # Scope is applied HERE, not by the model.
            if not ctx.principal.is_staff:
                stmt = stmt.where(TicketRow.employee_id == ctx.principal.employee_id)
            if args.status:
                stmt = stmt.where(TicketRow.status == args.status)
            if args.category:
                stmt = stmt.where(TicketRow.category == args.category)
            rows = s.scalars(stmt.order_by(TicketRow.updated_at.desc()).limit(args.limit)).all()
            return [_ticket_json(r) for r in rows]


class CreateTicket(Tool):
    name = "create_ticket"
    description = "Create a ticket on behalf of the caller. Requires approval."
    input_model = CreateTicketIn

    def run(self, ctx: ToolContext, args: CreateTicketIn) -> dict:
        with session_scope() as s:
            # Idempotency: replaying an approved action returns the original
            # ticket instead of creating a duplicate (audit G10).
            if ctx.idempotency_key:
                existing = s.scalar(
                    select(TicketRow).where(TicketRow.idempotency_key == ctx.idempotency_key)
                )
                if existing is not None:
                    return {**_ticket_json(existing), "idempotent_replay": True}
            now = datetime.now(UTC).isoformat()
            seq = s.scalar(select(TicketRow).order_by(TicketRow.ticket_id.desc()))
            next_num = (int(seq.ticket_id.split("-")[1]) + 1) if seq else 20000
            row = TicketRow(
                ticket_id=f"INC-{next_num}",
                employee_id=ctx.principal.employee_id,  # NEVER from args
                category=args.category,
                priority=args.priority,
                status="open",
                summary=args.summary,
                description=args.description,
                assigned_team=args.assigned_team,
                created_at=now,
                updated_at=now,
                idempotency_key=ctx.idempotency_key,
            )
            s.add(row)
            s.flush()
            return _ticket_json(row)


class AddTicketComment(Tool):
    name = "add_ticket_comment"
    description = "Append a comment to a ticket. Requires approval."
    input_model = AddCommentIn

    def run(self, ctx: ToolContext, args: AddCommentIn) -> dict:
        with session_scope() as s:
            row = s.get(TicketRow, args.ticket_id)
            if row is None:
                raise NotFoundError(f"ticket {args.ticket_id}")
            authorize_resource(ctx.principal, row.employee_id, self.name)
            stamp = datetime.now(UTC).isoformat()
            row.description = f"{row.description}\n\n[{stamp}] {ctx.principal.name}: {args.comment}"
            row.updated_at = stamp
            return {"ticket_id": row.ticket_id, "comment_added": True}


class EscalateTicket(Tool):
    name = "escalate_ticket"
    description = "Raise a ticket's priority and reassign. Staff only, requires approval."
    input_model = EscalateTicketIn

    def run(self, ctx: ToolContext, args: EscalateTicketIn) -> dict:
        with session_scope() as s:
            row = s.get(TicketRow, args.ticket_id)
            if row is None:
                raise NotFoundError(f"ticket {args.ticket_id}")
            row.priority = "Critical" if row.priority == "High" else "High"
            row.status = "in_progress"
            row.updated_at = datetime.now(UTC).isoformat()
            return {"ticket_id": row.ticket_id, "priority": row.priority, "reason": args.reason}


# --------------------------------------------------------------------------
# incidents
# --------------------------------------------------------------------------
def _incident_json(i: IncidentRow) -> dict:
    return {
        "incident_id": i.incident_id,
        "service": i.service,
        "severity": i.severity,
        "status": i.status,
        "affected_users": i.affected_users,
        "started_at": i.started_at,
        "description": i.description,
        "workaround": sanitize_untrusted(i.workaround or ""),
        "resolution": i.resolution,
    }


class GetIncident(Tool):
    name = "get_incident"
    input_model = GetIncidentIn
    description = "Fetch one incident by identifier."

    def run(self, ctx: ToolContext, args: GetIncidentIn) -> dict:
        with session_scope() as s:
            row = s.get(IncidentRow, args.incident_id)
            if row is None:
                raise NotFoundError(args.incident_id)
            return _incident_json(row)


class ListActiveIncidents(Tool):
    name = "list_active_incidents"
    input_model = EmptyIn
    description = "List incidents currently active or being monitored."

    def run(self, ctx: ToolContext, args: EmptyIn) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(
                select(IncidentRow).where(IncidentRow.status.in_(["active", "monitoring"]))
            ).all()
            return [_incident_json(r) for r in rows]


class SearchIncidents(Tool):
    name = "search_incidents"
    input_model = SearchIncidentsIn
    description = "Search incidents by service or status."

    def run(self, ctx: ToolContext, args: SearchIncidentsIn) -> list[dict]:
        with session_scope() as s:
            stmt = select(IncidentRow)
            if args.service:
                stmt = stmt.where(IncidentRow.service.ilike(f"%{args.service}%"))
            if args.status:
                stmt = stmt.where(IncidentRow.status == args.status)
            return [_incident_json(r) for r in s.scalars(stmt.limit(20)).all()]


class GetServiceStatus(Tool):
    name = "get_service_status"
    input_model = ServiceStatusIn
    description = "Current health of one named service."

    def run(self, ctx: ToolContext, args: ServiceStatusIn) -> dict:
        with session_scope() as s:
            rows = s.scalars(
                select(IncidentRow).where(
                    IncidentRow.service.ilike(f"%{args.service}%"),
                    IncidentRow.status.in_(["active", "monitoring"]),
                )
            ).all()
            return {
                "service": args.service,
                "status": "degraded" if rows else "operational",
                "incidents": [_incident_json(r) for r in rows],
            }


# --------------------------------------------------------------------------
# assets
# --------------------------------------------------------------------------
def _asset_json(a: AssetRow) -> dict:
    return {
        "asset_id": a.asset_id,
        "employee_id": a.employee_id,
        "hostname": a.hostname,
        "operating_system": a.operating_system,
        "os_version": a.os_version,
        "vpn_client_version": a.vpn_client_version,
        "installed_software": json.loads(a.installed_software),
        "status": a.status,
        "last_seen": a.last_seen,
    }


class GetEmployeeDevice(Tool):
    name = "get_employee_device"
    input_model = EmptyIn
    description = "Return the caller's own assigned devices."

    def run(self, ctx: ToolContext, args: EmptyIn) -> list[dict]:
        # No parameters at all: there is no way to ask for someone else's device.
        with session_scope() as s:
            rows = s.scalars(
                select(AssetRow).where(AssetRow.employee_id == ctx.principal.employee_id)
            ).all()
            return [_asset_json(r) for r in rows]


class GetDeviceStatus(Tool):
    name = "get_device_status"
    input_model = EmptyIn
    description = "Health summary of the caller's devices."

    def run(self, ctx: ToolContext, args: EmptyIn) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(
                select(AssetRow).where(AssetRow.employee_id == ctx.principal.employee_id)
            ).all()
            out = []
            for r in rows:
                stale = r.status == "stale"
                out.append(
                    {
                        "asset_id": r.asset_id,
                        "hostname": r.hostname,
                        "status": r.status,
                        "last_seen": r.last_seen,
                        "vpn_client_version": r.vpn_client_version,
                        "healthy": not stale,
                    }
                )
            return out


class GetInstalledSoftware(Tool):
    name = "get_installed_software"
    input_model = EmptyIn
    description = "Software inventory for the caller's devices."

    def run(self, ctx: ToolContext, args: EmptyIn) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(
                select(AssetRow).where(AssetRow.employee_id == ctx.principal.employee_id)
            ).all()
            return [
                {"asset_id": r.asset_id, "installed_software": json.loads(r.installed_software)}
                for r in rows
            ]


class ListAllDevices(Tool):
    """Exists specifically so the §17 attack has a real target to be denied."""

    name = "list_all_devices"
    input_model = EmptyIn
    description = "Estate-wide device listing. IT_admin only."

    def run(self, ctx: ToolContext, args: EmptyIn) -> list[dict]:
        with session_scope() as s:
            rows = s.scalars(select(AssetRow).limit(500)).all()
            return [_asset_json(r) for r in rows]


# --------------------------------------------------------------------------
REGISTRY: dict[str, Tool] = {}


def register(tool: Tool) -> Tool:
    REGISTRY[tool.name] = tool
    return tool


for _t in (
    GetTicket(),
    SearchTickets(),
    CreateTicket(),
    AddTicketComment(),
    EscalateTicket(),
    GetIncident(),
    ListActiveIncidents(),
    SearchIncidents(),
    GetServiceStatus(),
    GetEmployeeDevice(),
    GetDeviceStatus(),
    GetInstalledSoftware(),
    ListAllDevices(),
):
    register(_t)


IDENTITY_LIKE_FIELDS = frozenset(
    {
        "employee_id",
        "user_id",
        "actor",
        "principal",
        "role",
        "access_level",
        "on_behalf_of",
        "requester",
        "username",
    }
)


def tool_schemas() -> list[dict]:
    """JSON schemas for the model. Identity fields are absent by construction."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.input_model.model_json_schema(),
        }
        for t in REGISTRY.values()
    ]


def execute(name: str, ctx: ToolContext, args: dict) -> ToolResult:
    tool = REGISTRY.get(name)
    if tool is None:
        return ToolResult(ok=False, tool=name, error="Unknown tool.")
    return tool(ctx, args)


# --------------------------------------------------------------------------
# knowledge tools (spec §V5)
#
# Retrieval is also reachable directly from the graph; exposing it as a TOOL is
# what lets an external MCP client use the knowledge base. The index is injected
# at startup rather than imported, so the tool layer never owns model state.
# --------------------------------------------------------------------------
_INDEX_PROVIDER: Callable[[], Any] | None = None


def set_index_provider(fn: Callable[[], Any]) -> None:
    global _INDEX_PROVIDER
    _INDEX_PROVIDER = fn


def _index() -> Any:
    if _INDEX_PROVIDER is None:
        raise NotFoundError(
            "knowledge index is not available", safe_message="The knowledge base is not available."
        )
    return _INDEX_PROVIDER()


class GetDocumentIn(BaseModel):
    document_id: str = Field(pattern=r"^KB-[A-Z0-9]{8,12}$")


class GetPolicyIn(BaseModel):
    topic: str = Field(min_length=2, max_length=128)


class SearchKnowledge(Tool):
    name = "search_knowledge"
    description = "Search the IT knowledge base. Results are filtered to the caller's access level."
    input_model = SearchKnowledgeIn

    def run(self, ctx: ToolContext, args: SearchKnowledgeIn) -> list[dict]:
        from opspilot.rag.retrieval import LexicalReranker, RetrievalFilter

        index = _index()
        # The role filter is applied HERE, from the verified principal.
        flt = RetrievalFilter(
            role=ctx.principal.role, category=args.category, exclude_deprecated=True
        )
        hits = index.retrieve(
            args.query,
            arm="hybrid",
            flt=flt,
            candidate_k=30,
            final_k=args.top_k,
            reranker=LexicalReranker(bm25=index.bm25),
        )
        return [
            {
                **h.citation,
                "chunk_id": h.chunk.chunk_id,
                "text": sanitize_untrusted(h.chunk.text, 1200),
            }
            for h in hits
        ]


class GetDocument(Tool):
    name = "get_document"
    description = "Fetch one knowledge document by its identifier."
    input_model = GetDocumentIn

    def run(self, ctx: ToolContext, args: GetDocumentIn) -> dict:
        index = _index()
        chunks = [c for c in index.chunks if c.document_id == args.document_id]
        if not chunks:
            raise NotFoundError(args.document_id)
        first = chunks[0]
        if not ctx.principal.role.can_see(first.access_level):
            raise AuthorizationError(
                f"{ctx.principal.employee_id} -> {args.document_id}",
                safe_message="You do not have access to that document.",
            )
        chunks.sort(key=lambda c: c.ordinal)
        return {
            "document_id": args.document_id,
            "title": first.title,
            "category": first.category,
            "product": first.product,
            "version": first.version,
            "status": first.status,
            "effective_date": first.effective_date,
            "access_level": first.access_level,
            "content": sanitize_untrusted("\n\n".join(c.text for c in chunks), 8000),
        }


class GetPolicy(Tool):
    name = "get_policy"
    description = "Fetch the policy document covering a topic, if the caller may see it."
    input_model = GetPolicyIn

    def run(self, ctx: ToolContext, args: GetPolicyIn) -> dict:
        from opspilot.rag.retrieval import LexicalReranker, RetrievalFilter

        index = _index()
        flt = RetrievalFilter(role=ctx.principal.role)
        hits = index.retrieve(
            f"policy {args.topic}",
            arm="hybrid",
            flt=flt,
            candidate_k=30,
            final_k=10,
            reranker=LexicalReranker(bm25=index.bm25),
        )
        policy = next((h for h in hits if h.chunk.family == "policy"), None)
        if policy is None:
            raise NotFoundError(
                f"policy for {args.topic}",
                safe_message="No policy document is available to you for that topic.",
            )
        return {**policy.citation, "text": sanitize_untrusted(policy.chunk.text, 3000)}


for _t in (SearchKnowledge(), GetDocument(), GetPolicy()):
    register(_t)
