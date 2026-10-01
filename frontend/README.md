# Frontend — scaffold only

Spec §24 calls for Next.js + TypeScript + Tailwind. **That app is not built yet.**

What exists instead: a working single-file UI served by the backend at `/`
(`opspilot/api/ui.py`). It is deliberately minimal and is *not* presented as the
deliverable frontend. Its job is to prove the SSE contract end to end — activity
events, the approval pause and resume, citation rendering, abstention display —
so the backend is demonstrable without a Node toolchain.

Planned sections (spec §24): Chat · Conversations · My tickets · Active
incidents · Knowledge · Evaluation · System status.

The API contract the Next.js app will consume is already stable and documented
at `/docs` (OpenAPI). The evaluation page should read `GET /api/v1/evaluation`,
which returns measured results or the literal string `"Not evaluated yet"` —
never placeholder numbers (spec §25/§38).
