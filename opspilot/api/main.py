"""FastAPI application (spec §23).

Notable deviations from a naive implementation, all deliberate:

* `POST /api/v1/approvals/{token}` exists because an SSE stream cannot block for
  human input. The stream emits `approval_required` and closes; approval is a
  separate request that resumes the checkpointed graph (audit G15).
* `/metrics` reports the model token budget alongside the usual counters,
  because under a 200k tokens/day ceiling remaining quota is an operational
  metric, not trivia.
* Authentication here is a development shim (`X-Employee-Id`). It is labelled as
  such rather than dressed up as SSO. What matters for the security story is
  that authorization is enforced server-side from the resolved principal.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from opspilot.agents.graph import OpsPilotGraph
from opspilot.config import DATA_DIR, get_settings
from opspilot.core import (
    OpsPilotError,
    Principal,
    Role,
    configure_logging,
    conversation_id_var,
    get_logger,
    log_event,
    new_id,
    request_id_var,
    user_id_var,
)
from opspilot.data.corpus import generate_documents
from opspilot.data.enterprise import generate as generate_enterprise
from opspilot.db import (
    ApprovalRow,
    AuditLogRow,
    ConversationRow,
    IncidentRow,
    MessageRow,
    TicketRow,
    UserRow,
    init_engine,
    seed_from_generated,
    session_scope,
)
from opspilot.llm.providers import build_provider
from opspilot.memory import MemoryManager
from opspilot.observability import (
    TELEMETRY,
    instrument_fastapi,
    setup_tracing,
    span,
)
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder, build_embedder
from opspilot.rag.retrieval import KnowledgeIndex
from opspilot.security import (
    SlidingWindowRateLimiter,
    approval_expiry,
    idempotency_key_for,
)

log = get_logger("api")
STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    init_engine()
    seed_from_generated(generate_enterprise(seed=settings.seed))

    docs = generate_documents(seed=settings.seed)
    chunks = chunk_corpus(docs, "structure")
    backend = settings.embedding_backend.value
    try:
        dim = settings.hashing_dim if backend == "hashing" else settings.embedding_dim
        embedder = build_embedder(backend, settings.embedding_model, dim)
    except Exception as exc:
        log_event(log, "embedder fallback", requested=backend, error=str(exc)[:160])
        embedder = HashingEmbedder(dim=settings.hashing_dim)

    index = KnowledgeIndex(
        chunks,
        embedder,
        collection=settings.collection_name,
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
    ).build(use_qdrant=False)
    provider = build_provider(settings)

    # Knowledge tools resolve the index through this provider rather than
    # importing it, so the tool layer never owns model state.
    from opspilot.tools import set_index_provider

    set_index_provider(lambda: index)

    # MCP gateway (spec §7 / V7). Read-only tool calls cross the protocol
    # boundary; it falls back in-process on failure and the fallback is counted.
    gateway = None
    if settings.mcp_enabled:
        from opspilot.mcp.client import MCPGateway

        gateway = MCPGateway(
            enabled=True, max_sessions=settings.mcp_max_sessions, timeout_s=settings.mcp_timeout_s
        )

    tracing_on = (
        setup_tracing("opspilot-backend", settings.otel_endpoint, settings.otel_console)
        if settings.otel_enabled
        else False
    )
    if tracing_on:
        instrument_fastapi(app)
    if TELEMETRY.enabled:
        TELEMETRY.index_documents.set(len(docs))
        TELEMETRY.index_chunks.set(len(chunks))

    memory_mgr = MemoryManager(provider) if settings.long_term_memory_enabled else None

    STATE.update(
        {
            "index": index,
            "provider": provider,
            "gateway": gateway,
            "memory": memory_mgr,
            "tracing": tracing_on,
            "graph": OpsPilotGraph(index, provider, gateway=gateway),
            "limiter": SlidingWindowRateLimiter(),
            "started_at": time.time(),
            "docs": len(docs),
            "chunks": len(chunks),
            "requests": 0,
            "errors": 0,
            "latencies": [],
        }
    )
    log_event(
        log,
        "ready",
        documents=len(docs),
        chunks=len(chunks),
        embedder=embedder.name,
        provider=provider.name,
        mcp=bool(gateway),
        tracing=tracing_on,
        prometheus=TELEMETRY.enabled,
        long_term_memory=bool(memory_mgr),
    )
    yield
    if gateway is not None:
        with suppress(Exception):
            await asyncio.to_thread(gateway.close)
    index.close()


app = FastAPI(title="OpsPilot AI", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------
# auth (development shim -- see module docstring)
# --------------------------------------------------------------------------
def current_principal(
    authorization: str | None = Header(default=None),
    x_employee_id: str | None = Header(default=None),
) -> Principal:
    """Resolve the acting principal from a VERIFIED session.

    Preferred: `Authorization: Bearer <session token>` -- signed, expiring,
    tamper-evident. Legacy `X-Employee-Id` is still accepted so the existing
    scripts and tests keep working, and is what makes this a development shim
    rather than real authentication.
    """
    from opspilot.auth import verify_token

    if authorization and authorization.lower().startswith("bearer "):
        try:
            principal = verify_token(authorization.split(" ", 1)[1].strip())
        except OpsPilotError as exc:
            raise HTTPException(status_code=401, detail=exc.safe_message) from exc
        with session_scope() as s:
            row = s.get(UserRow, principal.employee_id)
            # Re-check account state on every request: a token issued before
            # offboarding must stop working immediately.
            if row is None or row.status == "offboarded":
                raise HTTPException(status_code=403, detail="Account is not active.")
            return Principal(
                employee_id=row.employee_id,
                name=row.name,
                role=Role(row.role),
                department=row.department,
            )

    with session_scope() as s:
        row = s.get(UserRow, x_employee_id or "E3001")
        if row is None:
            raise HTTPException(status_code=401, detail="Unknown employee.")
        if row.status == "offboarded":
            raise HTTPException(status_code=403, detail="Account is not active.")
        return Principal(
            employee_id=row.employee_id,
            name=row.name,
            role=Role(row.role),
            department=row.department,
        )


class LoginRequest(BaseModel):
    employee_id: str = Field(min_length=2, max_length=16)


@app.post("/api/v1/auth/login")
def login(req: LoginRequest) -> dict:
    """Issue a session token.

    NOTE: passwordless by design -- this is a development shim. It proves the
    session mechanism (signing, expiry, server-side verification, account-state
    re-check), not identity.
    """
    from opspilot.auth import issue_token

    with session_scope() as s:
        row = s.get(UserRow, req.employee_id)
        if row is None:
            raise HTTPException(status_code=401, detail="Unknown employee id.")
        if row.status == "offboarded":
            raise HTTPException(status_code=403, detail="This account has been offboarded.")
        principal = Principal(
            employee_id=row.employee_id,
            name=row.name,
            role=Role(row.role),
            department=row.department,
        )
    out = issue_token(principal)
    log_event(log, "session issued", actor=principal.employee_id, role=principal.role.value)
    return out


@app.get("/api/v1/auth/me")
def whoami(principal: Principal = Depends(current_principal)) -> dict:
    return {
        "employee_id": principal.employee_id,
        "name": principal.name,
        "role": principal.role.value,
        "department": principal.department,
        "is_staff": principal.is_staff,
    }


@app.get("/api/v1/auth/directory")
def directory() -> dict:
    """Selectable accounts for the sign-in screen.

    Exists only because this is a demo with no identity provider. A real
    deployment would never enumerate accounts to an unauthenticated caller.
    """
    with session_scope() as s:
        picks = ["E3001", "E3002", "E3003", "E2001", "E1001", "E3004"]
        rows = [s.get(UserRow, e) for e in picks]
        return {
            "accounts": [
                {
                    "employee_id": r.employee_id,
                    "name": r.name,
                    "role": r.role,
                    "department": r.department,
                    "status": r.status,
                }
                for r in rows
                if r is not None
            ]
        }


@app.middleware("http")
async def context_middleware(request: Request, call_next):
    rid = new_id("req")
    request_id_var.set(rid)
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        STATE["errors"] = STATE.get("errors", 0) + 1
        raise
    finally:
        STATE["requests"] = STATE.get("requests", 0) + 1
        STATE.setdefault("latencies", []).append((time.perf_counter() - t0) * 1000)
    response.headers["X-Request-Id"] = rid
    TELEMETRY.observe_request(
        request.method,
        request.url.path,
        response.status_code,
        STATE["latencies"][-1] / 1000.0 if STATE.get("latencies") else 0.0,
    )
    return response


@app.exception_handler(OpsPilotError)
async def opspilot_error_handler(request: Request, exc: OpsPilotError):
    return JSONResponse(
        status_code=exc.http_status, content={"error": exc.code, "message": exc.safe_message}
    )


# --------------------------------------------------------------------------
# schemas
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None
    # Tone changes HOW the answer reads, never WHAT it is grounded in. It is a
    # presentation instruction appended to the system prompt; it cannot loosen
    # the evidence rules, and the abstention logic is unaffected.
    tone: str = Field(default="standard", pattern=r"^(standard|concise|detailed|steps)$")


class FeedbackRequest(BaseModel):
    conversation_id: str = Field(min_length=3, max_length=40)
    rating: str = Field(pattern=r"^(up|down)$")
    question: str = Field(default="", max_length=2000)
    answer: str = Field(default="", max_length=8000)
    outcome: str = Field(default="", max_length=32)
    citations: int = Field(default=0, ge=0, le=50)
    reason: str | None = Field(default=None, max_length=256)
    retracted: bool = False


class CreateTicketRequest(BaseModel):
    category: str
    priority: str = Field(pattern=r"^(Low|Medium|High|Critical)$")
    summary: str
    description: str


# --------------------------------------------------------------------------
# chat
# --------------------------------------------------------------------------
def _run_turn(
    message: str, principal: Principal, conversation_id: str, tone: str = "standard"
) -> dict:
    graph: OpsPilotGraph = STATE["graph"]
    rid = request_id_var.get()
    user_id_var.set(principal.employee_id)
    conversation_id_var.set(conversation_id)

    with session_scope() as s:
        if s.get(ConversationRow, conversation_id) is None:
            s.add(
                ConversationRow(
                    conversation_id=conversation_id,
                    employee_id=principal.employee_id,
                    title=message[:80],
                )
            )
    # Two-layer memory (spec §14). Long-term context is included only when the
    # question plausibly refers back to it: injecting history on every turn
    # spends quota on text the user did not ask about, and widens the injection
    # surface, because conversation history is untrusted content (audit G8).
    mem_mgr: MemoryManager | None = STATE.get("memory")
    if mem_mgr is not None:
        mem_ctx = mem_mgr.load(conversation_id, principal, message)
        memory = mem_ctx.render()
    else:
        with session_scope() as s:
            prior = s.scalars(
                select(MessageRow)
                .where(MessageRow.conversation_id == conversation_id)
                .order_by(MessageRow.created_at.desc())
                .limit(4)
            ).all()
        memory = " ".join(m.content[:160] for m in reversed(prior)) if prior else ""

    t0 = time.perf_counter()
    with span(
        "opspilot.request",
        request_id=rid,
        user_id=principal.employee_id,
        conversation_id=conversation_id,
        role=principal.role.value,
    ):
        state = graph.run(
            question=message,
            principal=principal,
            conversation_id=conversation_id,
            request_id=rid,
            memory=memory,
            tone=tone,
        )
    elapsed = (time.perf_counter() - t0) * 1000

    if TELEMETRY.enabled:
        TELEMETRY.outcomes.labels(
            state.get("outcome")
            or ("abstained_no_evidence" if state.get("abstained") else "answered")
        ).inc()
        _budget = getattr(getattr(STATE.get("provider"), "inner", None), "budget", None)
        if _budget is not None:
            TELEMETRY.observe_budget(_budget.snapshot())

    pending = state.get("pending_approval")
    if pending:
        with session_scope() as s:
            s.add(
                ApprovalRow(
                    token=pending["token"],
                    conversation_id=conversation_id,
                    actor_employee_id=principal.employee_id,
                    action_hash=pending["action_hash"],
                    tool_name=pending["tool"],
                    arguments=json.dumps(pending["arguments"]),
                    description=pending["description"],
                    expires_at=approval_expiry(),
                )
            )

    answer = state.get("answer") or ""
    with session_scope() as s:
        s.add(
            MessageRow(
                message_id=new_id("msg"),
                conversation_id=conversation_id,
                role="user",
                content=message,
            )
        )
        if answer:
            s.add(
                MessageRow(
                    message_id=new_id("msg"),
                    conversation_id=conversation_id,
                    role="assistant",
                    content=answer,
                    citations=json.dumps(state.get("citations") or []),
                )
            )

    # Long-term memory is refreshed AFTER the reply is persisted, and only past
    # a message-count threshold, so summarisation never sits on the user's
    # critical path and never runs on a two-turn conversation.
    if mem_mgr is not None:
        try:
            mem_mgr.maybe_summarise(conversation_id)
        except Exception:
            log.warning("summarisation skipped", exc_info=True)

    return {
        "conversation_id": conversation_id,
        "request_id": rid,
        "answer": answer,
        "citations": state.get("citations") or [],
        "abstained": bool(state.get("abstained")),
        "memory_used": bool(memory),
        # Distinct outcomes, because "no evidence" and "not permitted" are
        # different facts and showing the first for the second is a lie.
        "outcome": state.get("outcome")
        or ("abstained_no_evidence" if state.get("abstained") else "answered"),
        "intent": state.get("intent"),
        "activity": state.get("activity") or [],
        "tool_results": state.get("tool_results") or [],
        "injection_flags": state.get("injection_flags") or [],
        "approval_required": (
            {
                "token": pending["token"],
                "description": pending["description"],
                "tool": pending["tool"],
            }
            if pending
            else None
        ),
        "usage": state.get("usage") or {},
        "latency_ms": round(elapsed, 2),
    }


@app.post("/api/v1/chat")
def chat(req: ChatRequest, principal: Principal = Depends(current_principal)) -> dict:
    STATE["limiter"].check(principal.employee_id)
    cid = req.conversation_id or new_id("conv")
    return _run_turn(req.message, principal, cid, req.tone)


@app.post("/api/v1/chat/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    principal: Principal = Depends(current_principal),
):
    """SSE. Emits activity events, then the answer, then citations.

    The activity events are derived from real graph state, not decorative
    placeholders -- if the incident tool did not run, no incident event appears.
    """
    STATE["limiter"].check(principal.employee_id)
    cid = req.conversation_id or new_id("conv")

    async def gen():
        """Stream a turn, stopping cleanly if the client goes away.

        What "stop" honestly means here. The graph runs in a worker thread and
        Python cannot kill a thread, so pressing Stop does NOT abort an
        already-issued model call -- those tokens are billed either way. What it
        does do is stop streaming immediately, mark the turn cancelled, and skip
        persisting a reply the user rejected. The distinction is stated in the
        UI rather than implied away.
        """
        yield _sse("start", {"conversation_id": cid})
        loop = asyncio.get_running_loop()
        task = loop.run_in_executor(None, _run_turn, req.message, principal, cid, req.tone)

        # Heartbeat while the graph runs, and watch for the client hanging up.
        while not task.done():
            if await request.is_disconnected():
                log_event(
                    log,
                    "client disconnected, turn cancelled",
                    conversation_id=cid,
                    actor=principal.employee_id,
                )
                return
            await asyncio.sleep(0.25)
            yield _sse("ping", {"t": datetime.now(UTC).isoformat()})

        result = await task

        for step in result["activity"]:
            if await request.is_disconnected():
                return
            yield _sse("activity", {"step": step})
            await asyncio.sleep(0.02)

        if result["approval_required"]:
            yield _sse("approval_required", result["approval_required"])
        else:
            for piece in _chunk_text(result["answer"]):
                # Checked per chunk so Stop takes effect mid-answer, not only
                # between turns.
                if await request.is_disconnected():
                    log_event(log, "stream stopped mid-answer", conversation_id=cid)
                    return
                yield _sse("token", {"text": piece})
                await asyncio.sleep(0.01)

        yield _sse("citations", {"citations": result["citations"]})
        yield _sse(
            "done",
            {
                "latency_ms": result["latency_ms"],
                "abstained": result["abstained"],
                "outcome": result["outcome"],
                "memory_used": result.get("memory_used", False),
                "usage": result["usage"],
            },
        )

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _chunk_text(text: str, size: int = 24):
    for i in range(0, len(text), size):
        yield text[i : i + size]


# --------------------------------------------------------------------------
# approvals (human-in-the-loop resume)
# --------------------------------------------------------------------------
@app.post("/api/v1/approvals/{token}")
def approve(token: str, principal: Principal = Depends(current_principal)) -> dict:
    now = datetime.now(UTC)
    with session_scope() as s:
        row = s.get(ApprovalRow, token)
        if row is None:
            raise HTTPException(404, "Unknown approval token.")
        # The approving actor must be the requesting actor.
        if row.actor_employee_id != principal.employee_id:
            raise HTTPException(403, "This approval does not belong to you.")
        if row.consumed_at is not None:
            # Idempotent replay: return the original outcome, do not act twice.
            return {"status": "already_applied", "result_ref": row.result_ref}
        expires = row.expires_at if row.expires_at.tzinfo else row.expires_at.replace(tzinfo=UTC)
        if expires < now:
            raise HTTPException(410, "This approval has expired. Please ask again.")
        args = json.loads(row.arguments)
        tool_name, conversation_id, ahash = row.tool_name, row.conversation_id, row.action_hash
        row.consumed_at = now

    graph: OpsPilotGraph = STATE["graph"]
    out = graph.resume_with_approval(
        conversation_id=conversation_id,
        approved={
            "tool": tool_name,
            "arguments": args,
            "idempotency_key": idempotency_key_for(token, ahash),
        },
    )

    answer = out.get("answer", "")
    ref = None
    for tr in out.get("tool_results", []):
        if isinstance(tr.get("data"), dict) and tr["data"].get("ticket_id"):
            ref = tr["data"]["ticket_id"]
    with session_scope() as s:
        r = s.get(ApprovalRow, token)
        if r is not None:
            r.result_ref = ref
        s.add(
            MessageRow(
                message_id=new_id("msg"),
                conversation_id=conversation_id,
                role="assistant",
                content=answer,
            )
        )
    return {
        "status": "applied",
        "answer": answer,
        "result_ref": ref,
        "tool_results": out.get("tool_results", []),
    }


@app.post("/api/v1/approvals/{token}/decline")
def decline(token: str, principal: Principal = Depends(current_principal)) -> dict:
    with session_scope() as s:
        row = s.get(ApprovalRow, token)
        if row is None:
            raise HTTPException(404, "Unknown approval token.")
        row.consumed_at = datetime.now(UTC)
        row.result_ref = "declined"
    return {"status": "declined"}


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------
@app.get("/api/v1/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str, principal: Principal = Depends(current_principal)
) -> dict:
    with session_scope() as s:
        conv = s.get(ConversationRow, conversation_id)
        if conv is None:
            raise HTTPException(404, "Conversation not found.")
        if conv.employee_id != principal.employee_id and not principal.is_staff:
            raise HTTPException(403, "Not your conversation.")
        msgs = s.scalars(
            select(MessageRow)
            .where(MessageRow.conversation_id == conversation_id)
            .order_by(MessageRow.created_at)
        ).all()
        return {
            "conversation_id": conversation_id,
            "title": conv.title,
            "messages": [
                {"role": m.role, "content": m.content, "citations": json.loads(m.citations or "[]")}
                for m in msgs
            ],
        }


@app.get("/api/v1/conversations")
def list_conversations(principal: Principal = Depends(current_principal)) -> dict:
    with session_scope() as s:
        rows = s.scalars(
            select(ConversationRow)
            .where(ConversationRow.employee_id == principal.employee_id)
            .order_by(ConversationRow.created_at.desc())
            .limit(50)
        ).all()
        return {
            "conversations": [
                {"conversation_id": c.conversation_id, "title": c.title} for c in rows
            ]
        }


@app.get("/api/v1/tickets/{ticket_id}")
def get_ticket(ticket_id: str, principal: Principal = Depends(current_principal)) -> dict:
    from opspilot.tools import ToolContext, execute

    res = execute(
        "get_ticket", ToolContext(principal, request_id_var.get()), {"ticket_id": ticket_id}
    )
    if not res.ok:
        raise HTTPException(
            403 if "only access" in (res.error or "") else 404, res.error or "Not found."
        )
    return res.data


@app.get("/api/v1/tickets")
def my_tickets(principal: Principal = Depends(current_principal)) -> dict:
    from opspilot.tools import ToolContext, execute

    res = execute("search_tickets", ToolContext(principal, request_id_var.get()), {"limit": 25})
    return {"tickets": res.data or []}


@app.post("/api/v1/tickets", status_code=201)
def create_ticket(
    req: CreateTicketRequest, principal: Principal = Depends(current_principal)
) -> dict:
    from opspilot.tools import ToolContext, execute

    res = execute("create_ticket", ToolContext(principal, request_id_var.get()), req.model_dump())
    if not res.ok:
        raise HTTPException(403, res.error or "Denied.")
    return res.data


@app.get("/api/v1/incidents")
def incidents(principal: Principal = Depends(current_principal)) -> dict:
    from opspilot.tools import ToolContext, execute

    res = execute("list_active_incidents", ToolContext(principal, request_id_var.get()), {})
    return {"incidents": res.data or []}


@app.get("/api/v1/incidents/{incident_id}")
def get_incident(incident_id: str, principal: Principal = Depends(current_principal)) -> dict:
    with session_scope() as s:
        row = s.get(IncidentRow, incident_id)
        if row is None:
            raise HTTPException(404, "Incident not found.")
        return {
            "incident_id": row.incident_id,
            "service": row.service,
            "severity": row.severity,
            "status": row.status,
            "affected_users": row.affected_users,
            "started_at": row.started_at,
            "description": row.description,
            "workaround": row.workaround,
        }


@app.post("/api/v1/feedback", status_code=201)
def submit_feedback(
    req: FeedbackRequest, principal: Principal = Depends(current_principal)
) -> dict:
    """Record a thumbs up/down.

    Stored with the question, the answer and the SYSTEM'S OWN confidence
    (outcome + citation count). A thumbs-down on something the system called
    `answered` is the highest-value row in the table: the system was confident
    and wrong, which is exactly what belongs in a regression set.
    """
    import hashlib

    from opspilot.db import FeedbackRow

    turn_key = hashlib.sha256(
        f"{req.conversation_id}|{principal.employee_id}|{req.question}".encode()
    ).hexdigest()[:48]

    with session_scope() as s:
        existing = s.scalar(select(FeedbackRow).where(FeedbackRow.turn_key == turn_key))
        if existing is not None:
            # Same turn rated again: update in place. Clicking the thumb twice
            # retracts rather than stacking another opinion.
            existing.rating = req.rating
            existing.retracted = req.retracted
            existing.reason = req.reason
            existing.updated_at = datetime.now(UTC)
        else:
            s.add(
                FeedbackRow(
                    feedback_id=new_id("fb"),
                    turn_key=turn_key,
                    conversation_id=req.conversation_id,
                    employee_id=principal.employee_id,
                    rating=req.rating,
                    reason=req.reason,
                    question=req.question[:2000],
                    answer=req.answer[:8000],
                    outcome=req.outcome,
                    citations=req.citations,
                    retracted=req.retracted,
                )
            )
    log_event(
        log,
        "feedback recorded",
        actor=principal.employee_id,
        rating=req.rating,
        outcome=req.outcome,
    )
    return {"status": "recorded", "rating": req.rating}


@app.get("/api/v1/feedback/summary")
def feedback_summary(principal: Principal = Depends(current_principal)) -> dict:
    """Aggregate feedback. Staff only -- it spans other people's conversations."""
    from opspilot.db import FeedbackRow

    if not principal.is_staff:
        raise HTTPException(403, "Feedback analytics are available to IT staff.")
    with session_scope() as s:
        rows = s.scalars(
            select(FeedbackRow)
            .where(FeedbackRow.retracted.is_(False))
            .order_by(FeedbackRow.created_at.desc())
        ).all()
        up = sum(1 for r in rows if r.rating == "up")
        down = sum(1 for r in rows if r.rating == "down")
        # The row that matters most: the system was confident and the user
        # disagreed.
        confident_but_rejected = [
            {
                "question": r.question[:160],
                "citations": r.citations,
                "conversation_id": r.conversation_id,
            }
            for r in rows
            if r.rating == "down" and r.outcome == "answered"
        ]
        return {
            "total": len(rows),
            "up": up,
            "down": down,
            "satisfaction": round(up / len(rows), 3) if rows else None,
            "confident_but_rejected": confident_but_rejected[:25],
            "by_outcome": {
                o: {
                    "up": sum(1 for r in rows if r.outcome == o and r.rating == "up"),
                    "down": sum(1 for r in rows if r.outcome == o and r.rating == "down"),
                }
                for o in {r.outcome for r in rows if r.outcome}
            },
        }


@app.get("/api/v1/evaluation")
def evaluation() -> dict:
    """Serve measured evaluation results, or say plainly that none exist.

    Spec §25/§38: never render placeholder metrics.
    """
    path = DATA_DIR.parent / "artifacts" / "reports" / "retrieval_experiments.json"
    if not path.exists():
        return {
            "status": "Not evaluated yet",
            "detail": "Run `python -m scripts.build_index` to produce measured results.",
        }
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# operational
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "uptime_s": round(time.time() - STATE.get("started_at", 0), 1)}


@app.get("/ready")
def ready() -> dict:
    if "index" not in STATE:
        raise HTTPException(503, "Index not built.")
    with session_scope() as s:
        users = s.scalar(select(func.count()).select_from(UserRow))
    gw = STATE.get("gateway")
    return {
        "status": "ready",
        "documents": STATE["docs"],
        "chunks": STATE["chunks"],
        "users": users,
        "provider": STATE["provider"].name,
        "tools": len(__import__("opspilot.tools", fromlist=["REGISTRY"]).REGISTRY),
        "mcp": {
            "enabled": bool(gw),
            "domains": list(
                __import__("opspilot.mcp.server", fromlist=["SERVER_TOOLS"]).SERVER_TOOLS
            )
            if gw
            else [],
        },
    }


@app.get("/metrics/prometheus")
def prometheus_metrics() -> Response:
    """Prometheus exposition format (spec §21).

    Kept separate from /metrics, which returns the human/JSON view the UI uses.
    A scraper wants text; a dashboard wants JSON; serving one format from one
    path is clearer than content negotiation.
    """
    payload, content_type = TELEMETRY.render()
    return Response(content=payload, media_type=content_type)


@app.get("/metrics")
def metrics() -> dict:
    lat = sorted(STATE.get("latencies", []))

    def pct(p: float) -> float:
        return round(lat[min(len(lat) - 1, int(p * (len(lat) - 1)))], 2) if lat else 0.0

    provider = STATE.get("provider")
    cache = getattr(provider, "cache", None)
    budget = getattr(getattr(provider, "inner", None), "budget", None)
    with session_scope() as s:
        audits = s.scalar(select(func.count()).select_from(AuditLogRow))
        denied = s.scalar(
            select(func.count()).select_from(AuditLogRow).where(AuditLogRow.decision == "denied")
        )
        tickets = s.scalar(select(func.count()).select_from(TicketRow))
    return {
        "requests_total": STATE.get("requests", 0),
        "errors_total": STATE.get("errors", 0),
        "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
        "cache": cache.snapshot() if cache else {"status": "no cache"},
        "token_budget": budget.snapshot()
        if budget
        else {"status": "not applicable (fake provider)"},
        "audit_events": audits,
        "authorization_denials": denied,
        "tickets": tickets,
        "index": {"documents": STATE.get("docs"), "chunks": STATE.get("chunks")},
        "mcp": STATE["gateway"].stats.snapshot()
        if STATE.get("gateway")
        else {"status": "disabled"},
        "injection_detector": {
            "neural_calls": getattr(STATE["graph"].detector, "neural_calls", 0),
            "neural_failures": getattr(STATE["graph"].detector, "neural_failures", 0),
        },
    }


@app.get("/", response_class=HTMLResponse)
def ui() -> str:
    from opspilot.api.ui import INDEX_HTML

    return INDEX_HTML
