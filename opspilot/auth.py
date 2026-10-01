"""Session tokens (spec §15 authentication).

What this IS: a signed, expiring, tamper-evident session token, verified
server-side on every request, from which the authorization principal is derived.

What this IS NOT: proof of identity. There is no password, no MFA and no
identity provider -- selecting a user grants a session for that user. It is a
**development authentication shim with a real session mechanism**, and it is
labelled that way everywhere rather than dressed up as SSO.

Why build it at all rather than keep the raw header: the header form made it
impossible to demonstrate login/logout, token expiry, or the fact that
authorization is derived from a *verified session* rather than a client claim.
The token makes that boundary visible and testable. Swapping in OIDC replaces
`issue_token`/`verify_token` only; nothing downstream changes, because
everything downstream already consumes a `Principal`.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from opspilot.config import get_settings
from opspilot.core import AuthorizationError, Principal, Role

SESSION_TTL_SECONDS = 8 * 3600


def _secret() -> bytes:
    return get_settings().jwt_secret.encode()


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(principal: Principal, ttl: int = SESSION_TTL_SECONDS) -> dict:
    payload = {
        "sub": principal.employee_id,
        "name": principal.name,
        "role": principal.role.value,
        "dept": principal.department,
        "exp": int(time.time()) + ttl,
        "iat": int(time.time()),
    }
    body = _b64e(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    sig = _b64e(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    return {"token": f"{body}.{sig}", "expires_in": ttl, "principal": payload}


def verify_token(token: str) -> Principal:
    """Verify signature and expiry. Raises AuthorizationError on any failure."""
    try:
        body, sig = token.split(".", 1)
    except ValueError as exc:
        raise AuthorizationError(
            "malformed session token", safe_message="Your session is invalid. Please sign in again."
        ) from exc
    expected = _b64e(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(sig, expected):
        raise AuthorizationError(
            "bad session signature", safe_message="Your session is invalid. Please sign in again."
        )
    payload = json.loads(_b64d(body))
    if int(payload.get("exp", 0)) < int(time.time()):
        raise AuthorizationError(
            "session expired", safe_message="Your session has expired. Please sign in again."
        )
    return Principal(
        employee_id=payload["sub"],
        name=payload.get("name", ""),
        role=Role(payload["role"]),
        department=payload.get("dept", ""),
    )
