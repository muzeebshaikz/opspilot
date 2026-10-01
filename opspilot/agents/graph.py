"""LangGraph workflow (spec §6/V6).

Why LangGraph earns its place here
----------------------------------
Not because it wraps an LLM call -- that would be pointless. It is used for the
two things a hand-rolled function chain does badly:

  1. CHECKPOINTING. The human-in-the-loop approval flow (spec §16) requires the
     workflow to stop mid-execution, survive the death of the HTTP request, and
     resume later from exactly where it paused. The checkpointer + thread_id
     gives that for free.
  2. EXPLICIT ROUTING STATE. Intent routing, retries and branch outcomes are
     inspectable state rather than control flow buried in call stacks, which is
     what makes the activity trace in the UI truthful.

The approval pause is modelled as a clean graph exit rather than a blocking
interrupt, because an SSE stream cannot block for human input (audit G15): the
stream closes with an `approval_required` event and a separate POST resumes the
checkpointed thread.
"""

from __future__ import annotations

import json
import re
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from opspilot.config import get_settings
from opspilot.core import (
    AuthorizationError,
    Principal,
    Role,
    TaggedText,
    TrustTier,
    get_logger,
    log_event,
)
from opspilot.db import audit, session_scope
from opspilot.llm.providers import LLMMessage, LLMProvider
from opspilot.rag.generator import GroundedGenerator
from opspilot.rag.retrieval import KnowledgeIndex, LexicalReranker, RetrievalFilter
from opspilot.security import (
    CONSEQUENTIAL_ACTIONS,
    InjectionDetector,
    authorize_action,
    mint_approval_token,
)
from opspilot.tools import ToolContext, execute

log = get_logger("agent")


# Sentinel that lets a reduced list field be CLEARED at the start of a turn.
# Without it, passing [] to an additive reducer means "append nothing", so the
# list keeps growing across every turn on the same checkpointed thread.
RESET = "__RESET__"


def _merge(a: list, b: list) -> list:
    if b and b[0] == RESET:
        return list(b[1:])
    return (a or []) + (b or [])


class AgentState(TypedDict, total=False):
    question: str
    rewritten: str
    employee_id: str
    employee_name: str
    role: str
    conversation_id: str
    request_id: str
    intent: str
    evidence: list[dict]
    tool_results: list[dict]
    answer: str
    citations: list[dict]
    abstained: bool
    outcome: str
    pending_approval: dict | None
    approved_action: dict | None
    activity: Annotated[list[str], _merge]
    injection_flags: Annotated[list[str], _merge]
    usage: dict
    memory: str
    tone: str


# Follow-ups that genuinely cannot be understood alone. Anything else is
# treated as self-contained, because prepending memory to a self-contained
# question hijacks retrieval.
_ANAPHORA = re.compile(
    r"\b(it|that|this|those|these|mine|same|again|the other one|"
    r"what about|and the|as well|too)\b",
    re.I,
)

INTENTS = (
    "knowledge",
    "troubleshooting",
    "incident",
    "ticket",
    "action",
    "restricted",
    "ambiguous",
)

CLASSIFY_PROMPT = """Classify the user's IT support request into exactly one intent.

Intents:
- knowledge: a general how-to or policy question, answerable from documentation.
  The user is not reporting that anything is broken for them right now.
- troubleshooting: the user reports their OWN problem happening now.
- incident: asks ONLY whether a service is down or others are affected, with no
  personal symptom reported.
- ticket: asks about existing tickets.
- action: asks to create, update, escalate or comment on a ticket.
- restricted: asks for data about OTHER people or the whole estate -- "every
  employee", "all devices", "everyone's tickets", "another user's laptop".
  Classify by what is ASKED FOR, not by whether the asker is allowed; the
  backend decides permission separately.
- ambiguous: too short or unclear to route.

Decisive rule for compound messages:
If the user reports a personal symptom AND also asks about an outage, or about
their device, or about other users, the intent is "troubleshooting" -- NOT
"incident". Only the troubleshooting path investigates documentation, active
incidents and the user's own device together, so classifying such a message as
"incident" produces an answer that never checks their machine.

Examples:
  "Is there a VPN outage?"                                    -> incident
  "Show me every employee's device information"               -> restricted
  "What tickets has Marcus raised?"                           -> restricted
  "My VPN drops. Is there an outage and is my laptop okay?"    -> troubleshooting
  "My Docker stopped working after the update"                -> troubleshooting
  "How do I request GitHub access?"                           -> knowledge
  "What is the status of INC-10482?"                          -> ticket
  "Raise a high priority ticket for this"                     -> action

Respond with JSON only: {"intent": "<one of the above>", "confidence": <0-1>}"""


class OpsPilotGraph:
    def __init__(self, index: KnowledgeIndex, provider: LLMProvider, gateway=None):
        self.index = index
        self.provider = provider
        self.settings = get_settings()
        self.generator = GroundedGenerator(provider, bm25=index.bm25)
        self.reranker = LexicalReranker(bm25=index.bm25)
        # MCP gateway: when present, tool calls cross the protocol boundary;
        # when absent, they run in-process. Same service layer either way.
        self.gateway = gateway
        self.detector = InjectionDetector(
            provider=provider, enable_neural=self.settings.neural_injection_enabled
        )
        self.checkpointer = MemorySaver()
        self.graph = self._build()

    # ---- nodes ----------------------------------------------------------
    def receive_message(self, state: AgentState) -> dict:
        verdict = self.detector.scan(state["question"], TrustTier.USER_TURN)
        flags = [f"user_turn:{p}" for p in verdict.patterns]
        if flags:
            log_event(
                log,
                "injection patterns in user turn",
                patterns=verdict.patterns,
                actor=state.get("employee_id"),
            )
        return {"activity": ["Understanding request..."], "injection_flags": flags}

    def classify_intent(self, state: AgentState) -> dict:
        resp = self.provider.complete(
            [LLMMessage("system", CLASSIFY_PROMPT), LLMMessage("user", state["question"])],
            model=self.settings.model_router,
            temperature=0.0,
            max_tokens=60,
        )
        intent = "knowledge"
        try:
            parsed = json.loads(resp.text[resp.text.find("{") : resp.text.rfind("}") + 1])
            if parsed.get("intent") in INTENTS:
                intent = parsed["intent"]
        except Exception:
            log_event(log, "intent parse failed, defaulting", raw=resp.text[:120])
        return {"intent": intent, "activity": [f"Intent: {intent}"]}

    def query_understanding(self, state: AgentState) -> dict:
        """Prepare the RETRIEVAL query.

        Deliberately NOT an LLM call in the default path: rewriting every query
        would double the request count against an 8k TPM ceiling. Rewrite is
        Experiment 5 and is enabled explicitly.

        What this must NOT do -- and previously did -- is concatenate the
        conversation memory onto a short question. Memory is dominated by the
        previous answer, so "i have laptop software issue" became
        "<entire previous VPN answer> i have laptop software issue" and
        retrieval matched VPN documents. The user got the previous answer back.

        Memory belongs in the GENERATION context, where the model can weigh it,
        not in the retrieval query, where it silently outvotes the actual
        question. Only a genuinely anaphoric follow-up ("what about mine?",
        "and the second one?") gets a narrow, bounded amount of prior context.
        """
        q = state["question"]
        memory = (state.get("memory") or "").strip()
        words = q.split()

        anaphoric = bool(_ANAPHORA.search(q)) and len(words) <= 8
        if anaphoric and memory:
            # Bounded: the tail only, never the whole transcript.
            rewritten = f"{memory[-220:]} {q}".strip()
        else:
            rewritten = q
        return {"rewritten": rewritten}

    def retrieve(self, state: AgentState) -> dict:
        flt = RetrievalFilter(role=Role(state["role"]), exclude_deprecated=True)
        hits = self.index.retrieve(
            state.get("rewritten") or state["question"],
            arm="hybrid",
            flt=flt,
            candidate_k=self.settings.candidate_k,
            final_k=self.settings.final_k,
            rrf_k=self.settings.rrf_k,
            reranker=self.reranker,
        )
        evidence = [
            {**h.citation, "text": h.chunk.text, "chunk_id": h.chunk.chunk_id} for h in hits
        ]
        return {
            "evidence": evidence,
            "activity": [f"Searching knowledge base ({len(hits)} passages)"],
        }

    def check_incidents(self, state: AgentState) -> dict:
        res = self._call("list_active_incidents", state, {})
        return {
            "tool_results": (state.get("tool_results") or []) + [res.to_json()],
            "activity": ["Checking active incidents..."],
        }

    def check_device(self, state: AgentState) -> dict:
        res = self._call("get_device_status", state, {})
        return {
            "tool_results": (state.get("tool_results") or []) + [res.to_json()],
            "activity": ["Checking your device..."],
        }

    def ticket_lookup(self, state: AgentState) -> dict:
        import re

        m = re.search(r"\bINC-\d{4,6}\b", state["question"], re.I)
        if m:
            res = self._call("get_ticket", state, {"ticket_id": m.group(0).upper()})
        else:
            res = self._call("search_tickets", state, {"status": "open", "limit": 10})
        return {
            "tool_results": (state.get("tool_results") or []) + [res.to_json()],
            "activity": ["Looking up tickets..."],
        }

    def incident_lookup(self, state: AgentState) -> dict:
        res = self._call("list_active_incidents", state, {})
        return {
            "tool_results": (state.get("tool_results") or []) + [res.to_json()],
            "activity": ["Checking incident board..."],
        }

    def prepare_action(self, state: AgentState) -> dict:
        """Authorize FIRST, then request human approval.

        Order matters: an unauthorized action must be refused outright, never
        surfaced to the user as an approvable request. Offering to do something
        the backend would reject teaches the user the wrong mental model and
        leaks what exists.
        """
        principal = self._principal(state)
        tool_name, args, description = self._plan_action(state, principal)
        try:
            authorize_action(principal, tool_name)
        except Exception as exc:
            return {
                "answer": getattr(exc, "safe_message", "That action is not permitted."),
                "abstained": False,
                "pending_approval": None,
                "activity": ["Authorization check: denied"],
            }
        if tool_name not in CONSEQUENTIAL_ACTIONS:
            return {"activity": ["Action does not require approval"]}
        token, ahash = mint_approval_token(tool_name, args, principal.employee_id)
        return {
            "pending_approval": {
                "token": token,
                "action_hash": ahash,
                "tool": tool_name,
                "arguments": args,
                "description": description,
            },
            "activity": ["Authorization check: passed", "Awaiting your approval"],
        }

    def execute_action(self, state: AgentState) -> dict:
        approved = state.get("approved_action") or {}
        # Consequential actions keep the IN-PROCESS path: the idempotency key
        # is part of ToolContext, and stdio MCP has no per-call channel to carry
        # it. Routing a state-changing call through a transport that cannot
        # carry its idempotency guard would trade a real safety property for a
        # protocol demonstration.
        ctx = self._ctx(state, idempotency_key=approved.get("idempotency_key"))
        res = execute(approved["tool"], ctx, approved.get("arguments", {}))
        data = res.data if res.ok else None
        answer = (
            f"Done. Ticket {data.get('ticket_id')} has been created with "
            f"{data.get('priority')} priority and assigned to "
            f"{data.get('assigned_team')}."
            if res.ok and isinstance(data, dict) and data.get("ticket_id")
            else ("The action completed." if res.ok else res.error or "Action failed.")
        )
        return {
            "tool_results": (state.get("tool_results") or []) + [res.to_json()],
            "answer": answer,
            "abstained": False,
            "activity": [f"Executed {approved['tool']}"],
        }

    def restricted_request(self, state: AgentState) -> dict:
        """Estate-wide or other-user data request.

        Answering "I could not find evidence" here would be FALSE: the evidence
        exists, the caller simply may not see it. Conflating "no evidence" with
        "not permitted" teaches the user the wrong model of the system, so the
        two outcomes are kept distinct in state and in the UI.

        The tool is never invoked for an unauthorised caller -- authorization is
        checked first, and a denial short-circuits before any data is touched.
        """
        principal = self._principal(state)
        tool = self._restricted_tool(state["question"])
        try:
            authorize_action(principal, tool)
        except AuthorizationError as exc:
            with session_scope() as sess:
                audit(
                    sess,
                    request_id=state.get("request_id", "-"),
                    actor=principal.employee_id,
                    role=principal.role.value,
                    action=tool,
                    decision="denied",
                    reason="restricted data request",
                )
            return {
                "answer": (
                    f"{exc.safe_message} This needs a higher access level than "
                    f"your **{principal.role.value}** role. If you need it for your "
                    f"work, ask an IT administrator, or I can raise a request ticket."
                ),
                "abstained": False,
                "outcome": "denied_authorization",
                "citations": [],
                "activity": [
                    f"Authorization check for {tool}: DENIED (role {principal.role.value})"
                ],
            }
        res = self._call(tool, state, {})
        return {
            "tool_results": (state.get("tool_results") or []) + [res.to_json()],
            "outcome": "answered",
            "activity": [
                f"Authorization check for {tool}: allowed",
                f"Retrieved estate data via {tool}",
            ],
        }

    @staticmethod
    def _restricted_tool(question: str) -> str:
        q = question.lower()
        if any(w in q for w in ("device", "laptop", "machine", "hardware", "asset")):
            return "list_all_devices"
        if "ticket" in q:
            return "search_tickets"
        return "list_all_devices"

    def clarify(self, state: AgentState) -> dict:
        return {
            "answer": (
                "I want to make sure I help with the right thing. Could you say a "
                "little more — what were you trying to do, and what happened?"
            ),
            "abstained": False,
            "outcome": "clarification",
            "citations": [],
            "activity": ["Requesting clarification"],
        }

    def generate(self, state: AgentState) -> dict:
        from opspilot.rag.chunking import Chunk
        from opspilot.rag.retrieval import Hit

        hits = []
        for e in state.get("evidence") or []:
            hits.append(
                Hit(
                    chunk=Chunk(
                        chunk_id=e.get("chunk_id", ""),
                        document_id=e["document_id"],
                        text=e.get("text", ""),
                        char_start=0,
                        char_end=0,
                        ordinal=0,
                        heading=e.get("section", ""),
                        title=e.get("title", ""),
                        version=e.get("version", ""),
                        status=e.get("status", "current"),
                        effective_date=e.get("effective_date", ""),
                        category=e.get("category", ""),
                        product=e.get("product", ""),
                    ),
                    score=e.get("score", 0.0),
                    rerank_score=e.get("score", 0.0),
                )
            )

        tool_blocks = []
        for tr in state.get("tool_results") or []:
            if tr.get("ok") and tr.get("data"):
                tool_blocks.append(
                    TaggedText(
                        text=json.dumps(tr["data"], indent=2)[:3000],
                        tier=TrustTier.TOOL_OUTPUT,
                        source_id=tr.get("tool", "tool"),
                    )
                )

        result = self.generator.generate(
            state.get("rewritten") or state["question"],
            hits,
            tool_blocks,
            memory=state.get("memory"),
            tone=state.get("tone", "standard"),
        )
        return {
            "answer": result.text,
            "citations": result.citations,
            "abstained": result.abstained,
            "usage": result.usage,
            "outcome": (
                "abstained_no_evidence"
                if result.abstained
                else ("partial_evidence" if result.evidence.tier == "partial" else "answered")
            ),
            "injection_flags": result.injection_flags,
            "activity": ["Grounding verification", "Preparing response..."],
        }

    # ---- routing ---------------------------------------------------------
    @staticmethod
    def route_intent(state: AgentState) -> str:
        intent = state.get("intent", "knowledge")
        return {
            "knowledge": "retrieve",
            "troubleshooting": "investigate",
            "incident": "incident_lookup",
            "ticket": "ticket_lookup",
            "action": "prepare_action",
            "restricted": "restricted_request",
            "ambiguous": "clarify",
        }.get(intent, "retrieve")

    @staticmethod
    def after_action(state: AgentState) -> str:
        if state.get("answer"):
            return END  # denied by authorization
        if state.get("pending_approval"):
            return END  # pause; resumed by the approval endpoint
        return "generate"

    # ---- assembly --------------------------------------------------------
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("receive_message", self.receive_message)
        g.add_node("classify_intent", self.classify_intent)
        g.add_node("query_understanding", self.query_understanding)
        g.add_node("retrieve", self.retrieve)
        g.add_node("check_incidents", self.check_incidents)
        g.add_node("check_device", self.check_device)
        g.add_node("ticket_lookup", self.ticket_lookup)
        g.add_node("incident_lookup", self.incident_lookup)
        g.add_node("restricted_request", self.restricted_request)
        g.add_node("prepare_action", self.prepare_action)
        g.add_node("execute_action", self.execute_action)
        g.add_node("clarify", self.clarify)
        g.add_node("generate", self.generate)

        g.add_edge(START, "receive_message")
        g.add_edge("receive_message", "classify_intent")
        g.add_edge("classify_intent", "query_understanding")
        g.add_conditional_edges(
            "query_understanding",
            self.route_intent,
            {
                "retrieve": "retrieve",
                "investigate": "retrieve",
                "incident_lookup": "incident_lookup",
                "ticket_lookup": "ticket_lookup",
                "prepare_action": "prepare_action",
                "clarify": "clarify",
                "restricted_request": "restricted_request",
            },
        )
        # Troubleshooting fans out: knowledge + incident + asset (spec Demo 3).
        g.add_conditional_edges(
            "retrieve",
            lambda s: "check_incidents" if s.get("intent") == "troubleshooting" else "generate",
            {"check_incidents": "check_incidents", "generate": "generate"},
        )
        g.add_edge("check_incidents", "check_device")
        g.add_edge("check_device", "generate")
        g.add_edge("ticket_lookup", "generate")
        g.add_edge("incident_lookup", "generate")
        g.add_conditional_edges(
            "prepare_action", self.after_action, {END: END, "generate": "generate"}
        )
        g.add_edge("execute_action", END)
        g.add_conditional_edges(
            "restricted_request",
            lambda st: END if st.get("answer") else "generate",
            {END: END, "generate": "generate"},
        )
        g.add_edge("clarify", END)
        g.add_edge("generate", END)
        return g.compile(checkpointer=self.checkpointer)

    # ---- helpers ---------------------------------------------------------
    def _principal(self, state: AgentState) -> Principal:
        return Principal(
            employee_id=state["employee_id"],
            name=state.get("employee_name", ""),
            role=Role(state["role"]),
        )

    def _call(self, tool: str, state: AgentState, args: dict):
        """Read-only tool call: through MCP when enabled, in-process otherwise."""
        principal = self._principal(state)
        if self.gateway is not None and self.gateway.enabled:
            return self.gateway.call(tool, principal, args, state.get("request_id", "-"))
        return execute(tool, self._ctx(state), args)

    def _ctx(self, state: AgentState, idempotency_key: str | None = None) -> ToolContext:
        return ToolContext(
            principal=self._principal(state),
            request_id=state.get("request_id", "-"),
            conversation_id=state.get("conversation_id", "-"),
            idempotency_key=idempotency_key,
        )

    def _plan_action(self, state: AgentState, principal: Principal) -> tuple[str, dict, str]:
        """Decide which consequential tool the user is asking for.

        Rule-based on purpose: the arguments of a state-changing action are
        derived from the USER TURN only, never from retrieved text or tool
        output (audit G8). A model free-forming these arguments is the exact
        path by which injected content causes real actions.
        """
        q = state["question"].lower()
        if "escalate" in q:
            import re

            m = re.search(r"\bINC-\d{4,6}\b", state["question"], re.I)
            tid = m.group(0).upper() if m else "INC-10482"
            return (
                "escalate_ticket",
                {"ticket_id": tid, "reason": "User requested escalation."},
                f"Escalate ticket {tid}",
            )
        priority = "High" if "high" in q or "urgent" in q else "Medium"
        category = next(
            (
                c
                for c in (
                    "vpn",
                    "docker",
                    "github",
                    "email",
                    "laptops",
                    "networking",
                    "databases",
                    "password",
                )
                if c in q
            ),
            "vpn",
        )
        summary = state["question"][:120]
        team = "Network Operations" if category in ("vpn", "networking") else "Service Desk"
        return (
            "create_ticket",
            {
                "category": category,
                "priority": priority,
                "summary": summary,
                "description": state["question"],
                "assigned_team": team,
            },
            f"Create a {priority}-priority {category} ticket assigned to {team}",
        )

    # ---- public API ------------------------------------------------------
    def run(
        self,
        *,
        question: str,
        principal: Principal,
        conversation_id: str,
        request_id: str,
        memory: str | None = None,
        tone: str = "standard",
    ) -> AgentState:
        cfg = {"configurable": {"thread_id": conversation_id}}
        # EVERY per-turn field is reset explicitly.
        #
        # The checkpointer persists state across turns on the same thread_id --
        # which is what makes approval-resume work, and is also a trap: any key
        # not overwritten here survives into the next turn. Two ways that bit:
        # `activity` has an additive reducer, so passing [] appended nothing and
        # the step list grew across turns; and stale `evidence`/`citations`/
        # `answer` meant a later turn could be served the PREVIOUS turn's
        # answer with the previous turn's sources.
        #
        # The RESET sentinel is how a reduced field is cleared: a plain [] is
        # indistinguishable from "no new items" to the reducer.
        state: AgentState = {
            "question": question,
            "employee_id": principal.employee_id,
            "employee_name": principal.name,
            "role": principal.role.value,
            "conversation_id": conversation_id,
            "request_id": request_id,
            "memory": memory or "",
            "tone": tone,
            "activity": [RESET],
            "injection_flags": [RESET],
            "tool_results": [],
            "evidence": [],
            "citations": [],
            "answer": "",
            "outcome": "",
            "rewritten": "",
            "intent": "",
            "abstained": False,
            "usage": {},
            "pending_approval": None,
            "approved_action": None,
        }
        return self.graph.invoke(state, cfg)

    def resume_with_approval(self, *, conversation_id: str, approved: dict) -> AgentState:
        """Resume a checkpointed thread after human approval."""
        cfg = {"configurable": {"thread_id": conversation_id}}
        snapshot = self.graph.get_state(cfg)
        state = dict(snapshot.values) if snapshot and snapshot.values else {}
        state["approved_action"] = approved
        state["pending_approval"] = None
        return self.execute_action(state) | {"_prior": True}
