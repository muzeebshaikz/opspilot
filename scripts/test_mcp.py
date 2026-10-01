"""Verify the MCP layer: tool discovery, execution, and the identity boundary."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.core import Principal, Role
from opspilot.db import init_engine
from opspilot.mcp.client import TOOL_TO_DOMAIN, MCPGateway
from opspilot.mcp.server import SERVER_TOOLS


async def main() -> int:
    init_engine()
    gw = MCPGateway(enabled=True)
    emp = Principal("E3001", "Priya Nair", Role.EMPLOYEE)
    adm = Principal("E1001", "Reshma Iyer", Role.IT_ADMIN)
    ok = 0
    total = 0

    def check(name, cond, detail):
        nonlocal ok, total
        total += 1
        ok += bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}\n      {detail}")

    print(f"domains: {list(SERVER_TOOLS)}  |  {len(TOOL_TO_DOMAIN)} tools routed\n")

    # 1. discovery over the wire
    session = await gw._session("ticketing", emp)
    listed = await session.list_tools()
    names = sorted(t.name for t in listed.tools)
    check(
        "MCP tool discovery (ticketing)", set(names) == set(SERVER_TOOLS["ticketing"]), f"{names}"
    )

    # 2. real call through the protocol
    r = await gw.acall("get_ticket", emp, {"ticket_id": "INC-10482"})
    check(
        "MCP call: own ticket allowed",
        r.ok and r.data and r.data.get("ticket_id") == "INC-10482",
        f"ok={r.ok} ticket={(r.data or {}).get('ticket_id')} {r.latency_ms:.0f}ms",
    )

    # 3. authorization still enforced ACROSS the MCP boundary
    r = await gw.acall("get_ticket", emp, {"ticket_id": "INC-11001"})
    check(
        "MCP call: another employee's ticket denied",
        (not r.ok) and "own records" in (r.error or ""),
        f"error={r.error!r}",
    )

    # 4. THE identity-spoofing attack (audit G9)
    r = await gw.acall(
        "create_ticket",
        emp,
        {
            "category": "vpn",
            "priority": "High",
            "summary": "spoof attempt",
            "description": "created via MCP",
            "employee_id": "E3055",
        },
    )
    owner = (r.data or {}).get("employee_id")
    check(
        "MCP: injected employee_id is ignored (G9)",
        r.ok and owner == "E3001",
        f"requested owner=E3055 -> actual owner={owner}",
    )

    # 5. role boundary across the transport
    r_emp = await gw.acall("list_all_devices", emp, {})
    r_adm = await gw.acall("list_all_devices", adm, {})
    check(
        "MCP: estate-wide listing denied for employee, allowed for admin",
        (not r_emp.ok) and r_adm.ok,
        f"employee={'denied' if not r_emp.ok else 'ALLOWED'} admin={len(r_adm.data or [])} devices",
    )

    # 6. incident domain
    r = await gw.acall("list_active_incidents", emp, {})
    ids = [i["incident_id"] for i in (r.data or [])]
    check("MCP incident domain", r.ok and "INCIDENT-204" in ids, f"{ids}")

    # 7. asset domain
    r = await gw.acall("get_device_status", emp, {})
    check(
        "MCP asset domain",
        r.ok and len(r.data or []) > 0,
        f"{len(r.data or [])} device(s), vpn={(r.data or [{}])[0].get('vpn_client_version')}",
    )

    print(f"\nMCP stats: {gw.stats.snapshot()}")
    await gw.aclose()
    print(f"\n{ok}/{total} MCP checks passed")
    return 0 if ok == total else 1


raise SystemExit(asyncio.run(main()))
