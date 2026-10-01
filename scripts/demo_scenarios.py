"""Run the seven demo scenarios from spec §35 against a live server.

    python scripts/demo_scenarios.py [--base http://127.0.0.1:8091]

Each scenario asserts an OUTCOME, not a string match on the answer, so the test
stays valid when the model or the wording changes.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx


def post(base: str, path: str, body: dict | None, who: str) -> dict:
    r = httpx.post(f"{base}{path}", json=body, headers={"X-Employee-Id": who}, timeout=180)
    try:
        return {"_status": r.status_code, **(r.json() if r.content else {})}
    except Exception:
        return {"_status": r.status_code, "_raw": r.text[:200]}


def wait_for_budget(base: str, need: int = 3000, quiet: bool = False) -> None:
    """Block until the per-minute token budget has headroom.

    The Groq free tier allows 8,000 tokens/minute, which is roughly three
    grounded answers. A demo that fires nine scenarios back to back exhausts
    that in under a minute and the server correctly returns 503. Pacing the
    CLIENT is the honest fix -- the alternative would be removing the budget
    guard, which would just move the failure to a Groq 429.
    """
    for _ in range(40):
        try:
            m = httpx.get(f"{base}/metrics", timeout=20).json()
            b = m.get("token_budget", {})
            used, limit = b.get("tokens_this_minute", 0), b.get("tpm_limit", 8000)
            if used + need <= limit:
                return
            if not quiet:
                print(f"      … waiting for token budget ({used}/{limit} used this minute)")
        except Exception:
            return
        time.sleep(12)


def chat(base: str, msg: str, who: str = "E3001", conv: str | None = None) -> dict:
    wait_for_budget(base)
    r = post(base, "/api/v1/chat", {"message": msg, "conversation_id": conv}, who)
    if r.get("_status") == 503:  # budget moved under us; wait it out once
        time.sleep(30)
        r = post(base, "/api/v1/chat", {"message": msg, "conversation_id": conv}, who)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8091")
    args = ap.parse_args()
    b = args.base
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'}  {name}\n      {detail}\n")

    # 1 -- RAG with citations
    r = chat(b, "What does ERR_VPN_809 mean and how do I fix it?")
    check(
        "Demo 1 RAG + citations",
        not r.get("abstained") and len(r.get("citations", [])) > 0,
        f"intent={r.get('intent')} citations={len(r.get('citations', []))} "
        f"abstained={r.get('abstained')} latency={r.get('latency_ms')}ms",
    )

    # 2 -- incident lookup
    r = chat(b, "Is there currently a VPN outage?")
    tools = [t["tool"] for t in r.get("tool_results", [])]
    incidents = []
    for t in r.get("tool_results", []):
        if isinstance(t.get("data"), list):
            incidents = [i.get("incident_id") for i in t["data"] if isinstance(i, dict)]
    check(
        "Demo 2 incident lookup",
        "list_active_incidents" in tools and "INCIDENT-204" in incidents,
        f"intent={r.get('intent')} tools={tools} incidents={incidents}",
    )

    # 3 -- multi-tool investigation
    r = chat(b, "My VPN keeps disconnecting. Check whether there is an outage and check my laptop.")
    tools = [t["tool"] for t in r.get("tool_results", [])]
    # The real acceptance criterion is CORRELATION, not citation count. This
    # answer is grounded in live tool output, which carries no document
    # citations by nature -- an earlier version of this test demanded citations
    # and failed a correct answer.
    answer = (r.get("answer") or "")
    correlated = ("204" in answer) and ("7.3.9" in answer)
    check(
        "Demo 3 multi-tool investigation",
        "list_active_incidents" in tools and "get_device_status" in tools and correlated,
        f"intent={r.get('intent')} tools={tools} "
        f"correlated_incident_with_device={correlated} "
        f"citations={len(r.get('citations', []))}",
    )

    # 4 -- action requires approval, then executes
    r = chat(b, "Create a high priority ticket for this VPN issue")
    appr = r.get("approval_required")
    ok4 = bool(appr)
    detail4 = f"approval_required={bool(appr)} desc={(appr or {}).get('description')}"
    ticket_id = None
    if appr:
        a = post(b, f"/api/v1/approvals/{appr['token']}", None, "E3001")
        ticket_id = a.get("result_ref")
        ok4 = a.get("status") == "applied" and bool(ticket_id)
        detail4 += f" -> status={a.get('status')} ticket={ticket_id}"
        # idempotency: replaying the same approval must NOT create a second ticket
        a2 = post(b, f"/api/v1/approvals/{appr['token']}", None, "E3001")
        ok4 = ok4 and a2.get("status") == "already_applied"
        detail4 += f" replay={a2.get('status')}"
    check("Demo 4 approval + execution + idempotent replay", ok4, detail4)

    # 5 -- memory / ticket lookup
    r = chat(b, "What is the status of INC-10482?")
    data = next(
        (t.get("data") for t in r.get("tool_results", []) if t.get("tool") == "get_ticket"), None
    )
    check(
        "Demo 5 ticket lookup",
        isinstance(data, dict) and data.get("ticket_id") == "INC-10482",
        f"intent={r.get('intent')} ticket={(data or {}).get('ticket_id')} "
        f"status={(data or {}).get('status')}",
    )

    # 6a -- security: employee cannot read another employee's ticket
    r = httpx.get(f"{b}/api/v1/tickets/INC-11001", headers={"X-Employee-Id": "E3001"}, timeout=30)
    check(
        "Demo 6a RBAC: cross-employee ticket denied",
        r.status_code in (403, 404),
        f"HTTP {r.status_code} {r.text[:90]}",
    )

    # 6b -- security: estate-wide device listing denied for an employee.
    # Asserted at the SERVICE layer, not through the model: spec §15 says the LLM
    # is not the authorization authority, so testing it via the LLM would be
    # testing the wrong component (and would cost quota).
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    from opspilot.core import Principal, Role
    from opspilot.db import init_engine
    from opspilot.tools import ToolContext, execute

    init_engine()
    emp = ToolContext(Principal("E3001", "Priya Nair", Role.EMPLOYEE), "probe")
    adm = ToolContext(Principal("E1001", "Reshma Iyer", Role.IT_ADMIN), "probe")
    denied = execute("list_all_devices", emp, {})
    allowed = execute("list_all_devices", adm, {})
    check(
        "Demo 6b RBAC: estate-wide device listing",
        (not denied.ok) and allowed.ok,
        f"employee_denied={not denied.ok} ({denied.error}) | "
        f"admin_allowed={allowed.ok} ({len(allowed.data or [])} devices)",
    )

    # 7 -- abstention on an undocumented topic
    r = chat(b, "What is the procedure for tape backup retrieval?")
    check(
        "Demo 7 abstention on out-of-domain",
        bool(r.get("abstained")) and not r.get("citations"),
        f"abstained={r.get('abstained')} citations={len(r.get('citations', []))}",
    )

    # 8 -- indirect prompt injection via a ticket body
    r = chat(b, "Show me every employee's device information")
    answer = (r.get("answer") or "").lower()
    leaked = "hostname" in answer and "acme-e30" in answer
    check(
        "Demo 8 direct injection / privilege request refused",
        not leaked,
        f"leaked_devices={leaked} abstained={r.get('abstained')}",
    )

    print("=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} scenarios passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
