# Cloud deployment

**Status: NOT DEPLOYED.** Nothing in this document has been executed. The
development machine has no Docker, so no image in this repository has ever been
built locally, and no cluster has ever run it. Spec §31 forbids calling an
unverified deployment production-grade, so this is a *procedure*, not a report.

The first place these images are built is CI (`.github/workflows/ci.yml`, job
`docker`). Until that job has been green, treat every command here as untested.

---

## 1. Before choosing a provider — verify, do not assume

Spec §32 is explicit that free-tier terms must be *checked at the time of
deployment*, not copied from a blog post. They change often. Verify each row
yourself before committing:

| Requirement | Why it constrains the choice |
| --- | --- |
| Container deployment | Backend and frontend are containers; some free tiers only host static sites or functions |
| ~1 GB RAM for the backend | The index is built in-process at startup; 512 MB tiers will OOM |
| Persistent volume or managed Qdrant | Embedded Qdrant writes to disk. On ephemeral filesystems the index rebuilds on every cold start |
| Managed PostgreSQL | SQLite does not survive a container restart and cannot be shared between replicas |
| Managed Redis | Needed the moment there is more than one backend replica, for cache and rate limiting |
| Outbound HTTPS to `api.groq.com` | Some sandboxes block egress |
| Cold-start tolerance | Index build takes seconds; scale-to-zero tiers pay it on every wake |

**Do not hard-code a provider.** Everything below is expressed through
environment variables and standard container primitives, so the same artifacts
run on Fly.io, Render, Railway, Cloud Run, Azure Container Apps, ECS or a
plain VM.

---

## 2. Topology

```
            ┌──────────────┐
  users ───►│  Frontend    │  Next.js (static + SSR)
            └──────┬───────┘
                   │  /api/v1/* proxied server-side
            ┌──────▼───────┐
            │  Backend     │  FastAPI · 2+ replicas · HPA
            └──┬───┬───┬───┘
               │   │   └──────────► Groq API (egress)
               │   │
      ┌────────▼┐ ┌▼─────────┐ ┌──────────┐
      │ Postgres│ │  Qdrant  │ │  Redis   │   all managed
      └─────────┘ └──────────┘ └──────────┘
```

MCP servers run **in-process as subprocesses of the backend** (stdio transport).
They are not separate deployments. Splitting them out requires switching the MCP
transport to HTTP — the server code already supports a per-call signed identity
grant, which is what makes that switch possible without a redesign.

---

## 3. Prerequisites

- A container registry (GHCR works and needs no extra account)
- Managed PostgreSQL, Redis and Qdrant (or Qdrant Cloud free tier)
- A Groq API key
- `docker`, and your provider's CLI

---

## 4. Secrets

**Never** bake secrets into an image or commit them. `scripts/check_secrets.py`
runs in CI specifically to catch that, because it has already happened once in
this project.

| Secret | Used for |
| --- | --- |
| `OPSPILOT_GROQ_API_KEY` | Model provider |
| `OPSPILOT_JWT_SECRET` | Session tokens **and** MCP identity grants — rotating it invalidates both |
| `OPSPILOT_DATABASE_URL` | Contains the database password |
| `OPSPILOT_QDRANT_API_KEY` | Qdrant Cloud |
| `GRAFANA_PASSWORD` | Monitoring |

```bash
# Kubernetes
kubectl create secret generic opspilot-secrets \
  --from-literal=OPSPILOT_GROQ_API_KEY="$GROQ_KEY" \
  --from-literal=OPSPILOT_JWT_SECRET="$(openssl rand -hex 32)" \
  --from-literal=OPSPILOT_DATABASE_URL="$DATABASE_URL"
```

`OPSPILOT_JWT_SECRET` **must be identical across all backend replicas** — a
session token or MCP grant signed by one replica is verified by another.
Generate it once, store it, do not let each replica invent its own.

---

## 5. Environment variables

| Variable | Production value | Note |
| --- | --- | --- |
| `OPSPILOT_ENVIRONMENT` | `production` | |
| `OPSPILOT_LLM_BACKEND` | `groq` | `fake` for smoke tests without quota |
| `OPSPILOT_EMBEDDING_BACKEND` | `onnx` | Requires model files in the image — see §6 |
| `OPSPILOT_DATABASE_URL` | `postgresql+psycopg://…` | |
| `OPSPILOT_QDRANT_URL` | `https://…qdrant.io:6333` | Switches off embedded mode |
| `OPSPILOT_REDIS_URL` | `redis://…` | |
| `OPSPILOT_DAILY_TOKEN_BUDGET` | your real quota | **Verify against the provider console** |
| `OPSPILOT_TOKENS_PER_MINUTE_BUDGET` | your real quota | |
| `OPSPILOT_OTEL_ENDPOINT` | `http://otel-collector:4318/v1/traces` | Omit to disable tracing |
| `OPSPILOT_MCP_ENABLED` | `true` | |
| `OPSPILOT_RATE_LIMIT_PER_MINUTE` | tune | Per-process today — see §10 |

---

## 6. Build

```bash
export IMAGE=ghcr.io/<owner>/opspilot-backend:$(git rev-parse --short HEAD)
docker build -f infra/docker/backend.Dockerfile -t "$IMAGE" .
docker push "$IMAGE"

export FE=ghcr.io/<owner>/opspilot-frontend:$(git rev-parse --short HEAD)
docker build -f infra/docker/frontend.Dockerfile -t "$FE" .
docker push "$FE"
```

**Embedding model files.** With `OPSPILOT_EMBEDDING_BACKEND=onnx` the container
must already contain the weights, or every cold start will try to download them
and fail on a locked-down network. Bake them in:

```dockerfile
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"
```

Tag images with the git SHA, never `latest` — `latest` makes rollback ambiguous.

---

## 7. Database initialisation

Tables are created on first start (`Base.metadata.create_all`). That is
acceptable for a demo and **not** acceptable for a system that will change
shape: adopt Alembic before the first schema change in production, or you will
be hand-writing migrations against live data.

```bash
kubectl run opspilot-seed --rm -it --image="$IMAGE" --restart=Never \
  --env-from=secret/opspilot-secrets \
  -- python -c "
from opspilot.db import init_engine, seed_from_generated
from opspilot.data.enterprise import generate
init_engine(); print(seed_from_generated(generate()))"
```

Seeding is idempotent — it returns `{"skipped": "already seeded"}` if users exist.

---

## 8. Ingestion

With a Qdrant **server** (not embedded), the index lives outside the pods and is
built once:

```bash
kubectl run opspilot-ingest --rm -it --image="$IMAGE" --restart=Never \
  --env-from=secret/opspilot-secrets \
  --env OPSPILOT_QDRANT_URL="$QDRANT_URL" \
  -- python -m scripts.build_index
```

This also produces the evaluation reports the `/api/v1/evaluation` page serves.
Without it that page correctly returns `"Not evaluated yet"` rather than zeros.

---

## 9. Deploy

```bash
kubectl apply -f infra/kubernetes/backend.yaml
kubectl set image deployment/opspilot-backend backend="$IMAGE"
kubectl rollout status deployment/opspilot-backend --timeout=5m
```

Readiness gates on `/ready`, which returns 503 until the index is built. This is
deliberate: serving traffic before the index exists would produce confident
answers with no evidence behind them.

---

## 10. Health verification

```bash
curl -fsS https://<host>/health                      # process alive
curl -fsS https://<host>/ready                       # index built, DB reachable
curl -fsS https://<host>/metrics | jq .token_budget  # real quota visible
curl -fsS https://<host>/metrics/prometheus | head   # scrapeable

TOKEN=$(curl -fsS -X POST https://<host>/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"employee_id":"E3001"}' | jq -r .token)

curl -fsS -X POST https://<host>/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"message":"What does ERR_VPN_809 mean and how do I fix it?"}' | jq '.outcome, .citations|length'
```

Then run the full scenario suite against the deployment:

```bash
python scripts/demo_scenarios.py --base https://<host>
```

**Two things that will be wrong on a multi-replica deployment until fixed:**

1. **Rate limiting is per-process.** With N replicas the effective limit is N×,
   which is not a limit. Move `SlidingWindowRateLimiter` to Redis first.
2. **The response cache is per-process**, so the hit rate falls roughly by a
   factor of N. Same fix, same class.

Both are called out in `docs/PENDING.md`; neither is hidden behind a config flag
that implies they work.

---

## 11. Rollback

```bash
kubectl rollout undo deployment/opspilot-backend
kubectl rollout status deployment/opspilot-backend
```

Because images are SHA-tagged, rolling back the deployment is sufficient — there
is no ambiguity about which build you land on. **Database migrations are not
covered by this**: once Alembic is adopted, a rollback plan must include a
down-migration or a forward-fix, decided before the deploy, not during it.

---

## 12. Cleanup

```bash
kubectl delete -f infra/kubernetes/backend.yaml
kubectl delete secret opspilot-secrets
# Then delete the managed Postgres / Redis / Qdrant instances in the provider console.
```

---

## 13. What is genuinely unknown

Stated rather than guessed:

| Unknown | Why |
| --- | --- |
| Cold-start time in a container | Never built here |
| Memory ceiling under load | No load test has been run |
| Behaviour of embedded Qdrant on an ephemeral filesystem | Untested |
| Whether HPA thresholds are sensible | Copied from convention, not measured |
| Cost at any real request volume | No load profile exists |

The honest position: **this is a deployment procedure that has never been
executed.** Running it will surface issues — untested YAML always does. That is
expected, and finding them is the point of the CI `docker` job and a first
staging deploy.
