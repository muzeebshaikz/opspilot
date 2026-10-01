"""MCP client gateway (spec §7 / V7).

Routes a tool call to the MCP server that owns that domain, over stdio.

Two design decisions worth defending
------------------------------------

**1. Identity is passed at process spawn, not per call.**
stdio MCP has no per-request header, so the verified principal is injected into
the server process environment. That means one server process per (domain,
principal) -- sessions are pooled and capped. This is the honest cost of stdio.
For a real deployment the transport is HTTP with a signed per-request auth
header, and the pool collapses to one process per domain; the server code does
not change, only `MCPIdentity` sourcing does.

**2. Graceful degradation is mandatory, not optional.**
MCP adds a process boundary and a serialisation hop to calls that already worked
in-process. If the boundary fails, falling back to the in-process service layer
is strictly better than failing the user's request (spec §29). The fallback is
COUNTED and surfaced in `/metrics`, so "MCP is silently never used" cannot hide.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass, field

from opspilot.core import Principal, get_logger, log_event
from opspilot.mcp.server import SERVER_TOOLS
from opspilot.tools import ToolContext, ToolResult, execute

log = get_logger("mcp.client")

TOOL_TO_DOMAIN: dict[str, str] = {
    tool: domain for domain, tools in SERVER_TOOLS.items() for tool in tools
}


@dataclass
class MCPStats:
    calls: int = 0
    failures: int = 0
    fallbacks: int = 0
    latency_ms: list[float] = field(default_factory=list)

    def snapshot(self) -> dict:
        lat = sorted(self.latency_ms)
        p95 = lat[min(len(lat) - 1, int(0.95 * (len(lat) - 1)))] if lat else 0.0
        return {
            "calls": self.calls,
            "failures": self.failures,
            "fallbacks_to_in_process": self.fallbacks,
            "p95_ms": round(p95, 2),
            "mean_ms": round(sum(lat) / len(lat), 2) if lat else 0.0,
        }


class MCPGateway:
    """Spawns and pools MCP server subprocesses; routes calls by domain."""

    def __init__(self, *, enabled: bool = True, max_sessions: int = 8, timeout_s: float = 20.0):
        self.enabled = enabled
        self.max_sessions = max_sessions
        self.timeout_s = timeout_s
        self.stats = MCPStats()
        self._sessions: dict[str, tuple[object, AsyncExitStack]] = {}
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = None
        import threading

        self._loop_lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------
    async def _session(self, domain: str, principal: Principal):
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        # Keyed by DOMAIN only. Identity travels per call as a signed grant,
        # so one process serves every caller (see server.py for why).
        key = domain
        if key in self._sessions:
            return self._sessions[key][0]
        async with self._lock:
            if key in self._sessions:
                return self._sessions[key][0]
            if len(self._sessions) >= self.max_sessions:
                old_key, (_, old_stack) = next(iter(self._sessions.items()))
                await old_stack.aclose()
                self._sessions.pop(old_key, None)

            env = dict(os.environ)
            # Multi-tenant mode: do NOT pin an identity to the process.
            env.pop("OPSPILOT_MCP_PRINCIPAL", None)
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "opspilot.mcp.server", domain],
                env=env,
                cwd=str(__import__("pathlib").Path(__file__).resolve().parents[2]),
            )
            stack = AsyncExitStack()
            read, write = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[key] = (session, stack)
            log_event(
                log,
                "mcp session opened",
                domain=domain,
                actor=principal.employee_id,
                pool=len(self._sessions),
            )
            return session

    async def aclose(self) -> None:
        for _, (_, stack) in list(self._sessions.items()):
            with suppress(Exception):
                await stack.aclose()
        self._sessions.clear()

    # -- calls -------------------------------------------------------------
    async def acall(
        self, tool: str, principal: Principal, args: dict, request_id: str = "-"
    ) -> ToolResult:
        domain = TOOL_TO_DOMAIN.get(tool)
        if not self.enabled or domain is None:
            return self._in_process(tool, principal, args, request_id, reason="disabled")

        t0 = time.perf_counter()
        try:
            session = await self._session(domain, principal)
            # The server handler takes a single `args` object, so tool arguments
            # are nested one level. Keeping the wire shape uniform means adding a
            # tool never changes the client.
            from opspilot.mcp.server import IDENTITY_GRANT_KEY, mint_identity_grant

            call_args = dict(args or {})
            # Minted here, server-side, from the authenticated session. The
            # model never sees or influences this value.
            call_args[IDENTITY_GRANT_KEY] = mint_identity_grant(principal)
            raw = await asyncio.wait_for(
                session.call_tool(tool, {"args": call_args}), timeout=self.timeout_s
            )
            elapsed = (time.perf_counter() - t0) * 1000
            self.stats.calls += 1
            self.stats.latency_ms.append(elapsed)
            payload = json.loads(raw.content[0].text) if raw.content else {}
            return ToolResult(
                ok=payload.get("ok", False),
                tool=tool,
                data=payload.get("data"),
                error=payload.get("error"),
                latency_ms=elapsed,
            )
        except Exception as exc:
            self.stats.failures += 1
            log_event(
                log,
                "mcp call failed, falling back in-process",
                tool=tool,
                domain=domain,
                error=str(exc)[:160],
            )
            return self._in_process(tool, principal, args, request_id, reason="error")

    def _in_process(
        self, tool: str, principal: Principal, args: dict, request_id: str, reason: str
    ) -> ToolResult:
        if reason == "error":
            self.stats.fallbacks += 1
        return execute(tool, ToolContext(principal=principal, request_id=request_id), args)

    # -- the loop problem ---------------------------------------------------
    # MCP sessions are anyio streams bound to the event loop that created them.
    # Running each call on a fresh loop therefore DEADLOCKS the second call
    # against a cached session, which presents as a request that simply hangs.
    # The gateway owns ONE persistent loop in a daemon thread and marshals every
    # call onto it, so sessions and their streams always live on the same loop.
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        import threading

        if self._loop is not None and not self._loop.is_closed():
            return self._loop
        with self._loop_lock:
            if self._loop is not None and not self._loop.is_closed():
                return self._loop
            ready = threading.Event()

            def runner() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._loop = loop
                ready.set()
                loop.run_forever()

            self._thread = threading.Thread(target=runner, daemon=True, name="mcp-gateway-loop")
            self._thread.start()
            ready.wait(timeout=10)
            assert self._loop is not None
            return self._loop

    def call(
        self, tool: str, principal: Principal, args: dict, request_id: str = "-"
    ) -> ToolResult:
        """Blocking wrapper for the synchronous graph nodes."""
        loop = self._ensure_loop()
        fut = asyncio.run_coroutine_threadsafe(self.acall(tool, principal, args, request_id), loop)
        try:
            return fut.result(timeout=self.timeout_s + 15)
        except Exception as exc:
            self.stats.failures += 1
            log_event(
                log, "mcp dispatch failed, falling back in-process", tool=tool, error=str(exc)[:160]
            )
            return self._in_process(tool, principal, args, request_id, reason="error")

    def close(self) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        fut = asyncio.run_coroutine_threadsafe(self.aclose(), self._loop)
        with suppress(Exception):
            fut.result(timeout=15)
        self._loop.call_soon_threadsafe(self._loop.stop)
