"""MCP layer tests. No API calls -- the servers wrap the local service layer."""

from __future__ import annotations

import time

import pytest

from opspilot.core import Principal, Role
from opspilot.db import init_engine
from opspilot.mcp.server import (
    IDENTITY_GRANT_KEY,
    SERVER_TOOLS,
    assert_no_identity_parameters,
    mint_identity_grant,
    verify_identity_grant,
)
from opspilot.tools import IDENTITY_LIKE_FIELDS, REGISTRY


@pytest.fixture(scope="module", autouse=True)
def _db():
    init_engine()


def test_every_registry_tool_is_served_by_exactly_one_domain():
    served = [t for tools in SERVER_TOOLS.values() for t in tools]
    assert len(served) == len(set(served)), "a tool is served by two domains"
    assert set(served) == set(REGISTRY), "registry and MCP domains have drifted apart"


def test_no_mcp_domain_exposes_an_identity_parameter():
    """The structural guarantee behind audit G9."""
    for _domain, tools in SERVER_TOOLS.items():
        assert_no_identity_parameters(tools)


def test_identity_grant_roundtrip():
    p = Principal("E3001", "Priya", Role.EMPLOYEE)
    assert verify_identity_grant(mint_identity_grant(p)).employee_id == "E3001"


@pytest.mark.parametrize("field,value", [("role", "IT_admin"), ("employee_id", "E3055")])
def test_tampered_grant_is_rejected(field, value):
    g = mint_identity_grant(Principal("E3001", "Priya", Role.EMPLOYEE))
    g[field] = value
    with pytest.raises(PermissionError):
        verify_identity_grant(g)


def test_unsigned_grant_is_rejected():
    g = mint_identity_grant(Principal("E3001", "Priya", Role.EMPLOYEE))
    g.pop("sig")
    with pytest.raises(PermissionError):
        verify_identity_grant(g)


def test_expired_grant_is_rejected():
    g = mint_identity_grant(Principal("E3001", "Priya", Role.EMPLOYEE))
    g["exp"] = int(time.time()) - 5
    with pytest.raises(PermissionError):
        verify_identity_grant(g)


def test_grant_key_is_not_a_tool_field():
    """The reserved key must never collide with a real tool argument."""
    for tool in REGISTRY.values():
        props = tool.input_model.model_json_schema().get("properties") or {}
        assert IDENTITY_GRANT_KEY not in props


def test_gateway_falls_back_in_process_when_mcp_disabled():
    from opspilot.mcp.client import MCPGateway

    gw = MCPGateway(enabled=False)
    r = gw.call(
        "get_ticket", Principal("E3001", "Priya", Role.EMPLOYEE), {"ticket_id": "INC-10482"}
    )
    assert r.ok and r.data["ticket_id"] == "INC-10482"
    assert gw.stats.calls == 0, "disabled gateway must not count MCP calls"


def test_gateway_routes_every_tool_to_a_domain():
    from opspilot.mcp.client import TOOL_TO_DOMAIN

    assert set(TOOL_TO_DOMAIN) == set(REGISTRY)
