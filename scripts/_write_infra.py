"""Write deployment + scaffold files.

These are authored so the deployment path exists and is reviewable, but they are
NOT run locally (no Docker/WSL2 on the target machine). Every file says so.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES: dict[str, str] = {}

FILES["infra/docker/backend.Dockerfile"] = """\
# NOT BUILT OR RUN LOCALLY -- no Docker on the development machine.
# Authored and CI-lintable; treat as unverified until a CI build runs it.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Python 3.12, not 3.14: sentence-transformers and parts of the ONNX/embedding
# ecosystem do not yet declare 3.14 support. The app itself runs on both.
RUN apt-get update && apt-get install -y --no-install-recommends curl \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY opspilot ./opspilot
COPY scripts ./scripts

RUN useradd -m -u 10001 opspilot && chown -R opspilot /app
USER opspilot

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \\
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "opspilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

FILES["infra/docker/frontend.Dockerfile"] = """\
# NOT BUILT OR RUN LOCALLY. Placeholder for the Next.js frontend (spec §24).
FROM node:22-alpine AS deps
WORKDIR /app
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --omit=dev || npm install --omit=dev

FROM node:22-alpine AS build
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY frontend/ .
RUN npm run build

FROM node:22-alpine AS run
WORKDIR /app
ENV NODE_ENV=production
COPY --from=build /app/.next ./.next
COPY --from=build /app/public ./public
COPY --from=build /app/package.json ./
USER node
EXPOSE 3000
CMD ["npm", "start"]
"""

FILES["docker-compose.yml"] = """\
# NOT RUN LOCALLY -- the development machine has no Docker or WSL2.
# This file is the intended deployment topology, authored for review and CI.
# Locally the same system runs with: Qdrant in embedded mode, SQLite, and an
# in-process cache. See docs/architecture/TECHNOLOGY-RATIONALE.md.
services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: opspilot
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?set POSTGRES_PASSWORD}
      POSTGRES_DB: opspilot
    volumes: [pgdata:/var/lib/postgresql/data]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U opspilot"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: ["redis-server", "--maxmemory", "256mb", "--maxmemory-policy", "allkeys-lru"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5

  qdrant:
    image: qdrant/qdrant:latest
    volumes: [qdrantdata:/qdrant/storage]
    healthcheck:
      test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333'"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build: {context: ., dockerfile: infra/docker/backend.Dockerfile}
    environment:
      OPSPILOT_DATABASE_URL: postgresql+psycopg://opspilot:${POSTGRES_PASSWORD}@postgres:5432/opspilot
      OPSPILOT_QDRANT_URL: http://qdrant:6333
      OPSPILOT_REDIS_URL: redis://redis:6379/0
      OPSPILOT_LLM_BACKEND: ${OPSPILOT_LLM_BACKEND:-fake}
      OPSPILOT_GROQ_API_KEY: ${OPSPILOT_GROQ_API_KEY:-}
      OPSPILOT_EMBEDDING_BACKEND: onnx
    ports: ["8000:8000"]
    depends_on:
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}
      qdrant: {condition: service_healthy}

  frontend:
    build: {context: ., dockerfile: infra/docker/frontend.Dockerfile}
    environment:
      NEXT_PUBLIC_API_URL: http://backend:8000
    ports: ["3000:3000"]
    depends_on: [backend]

  # MCP servers (spec §7 / V7). Not implemented yet -- deliberately.
  # The service layer they must wrap exists (opspilot/tools.py); introducing MCP
  # before that layer was proven would have inverted the spec's own ordering.
  # mcp-ticketing: {build: {context: ., dockerfile: infra/docker/mcp.Dockerfile},
  #                 command: ["python","-m","opspilot.mcp.ticketing"]}
  # mcp-incident:  ...
  # mcp-asset:     ...
  # mcp-knowledge: ...

volumes:
  pgdata:
  qdrantdata:
"""

FILES["infra/kubernetes/backend.yaml"] = """\
# NOT APPLIED TO ANY CLUSTER. Authored for review; unverified.
# Spec §31 explicitly forbids claiming a local deployment is production-grade.
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opspilot-backend
  labels: {app: opspilot, component: backend}
spec:
  replicas: 2
  selector: {matchLabels: {app: opspilot, component: backend}}
  template:
    metadata:
      labels: {app: opspilot, component: backend}
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 10001
        seccompProfile: {type: RuntimeDefault}
      containers:
        - name: backend
          image: ghcr.io/acme/opspilot-backend:0.1.0
          ports: [{containerPort: 8000}]
          envFrom:
            - configMapRef: {name: opspilot-config}
            - secretRef: {name: opspilot-secrets}
          # Readiness gates on the index being BUILT, not merely on the process
          # being alive -- serving traffic before the index exists would return
          # confident answers with no evidence.
          readinessProbe:
            httpGet: {path: /ready, port: 8000}
            initialDelaySeconds: 20
            periodSeconds: 10
            failureThreshold: 6
          livenessProbe:
            httpGet: {path: /health, port: 8000}
            initialDelaySeconds: 40
            periodSeconds: 20
          resources:
            requests: {cpu: "500m", memory: "1Gi"}
            limits: {cpu: "2", memory: "3Gi"}
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: {drop: ["ALL"]}
          volumeMounts: [{name: tmp, mountPath: /tmp}]
      volumes: [{name: tmp, emptyDir: {}}]
---
apiVersion: v1
kind: Service
metadata: {name: opspilot-backend}
spec:
  selector: {app: opspilot, component: backend}
  ports: [{port: 80, targetPort: 8000}]
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata: {name: opspilot-backend}
spec:
  scaleTargetRef: {apiVersion: apps/v1, kind: Deployment, name: opspilot-backend}
  minReplicas: 2
  maxReplicas: 8
  metrics:
    - type: Resource
      resource: {name: cpu, target: {type: Utilization, averageUtilization: 70}}
---
apiVersion: v1
kind: ConfigMap
metadata: {name: opspilot-config}
data:
  OPSPILOT_QDRANT_URL: "http://qdrant:6333"
  OPSPILOT_EMBEDDING_BACKEND: "onnx"
  OPSPILOT_LOG_LEVEL: "INFO"
# Secrets are supplied by the cluster's secret manager, never committed.
"""

FILES["requirements.txt"] = """\
# Core -- all verified to install and import on Python 3.14 (Windows, CPU-only).
numpy>=2.0
pydantic>=2.9
pydantic-settings>=2.5
fastapi>=0.115
uvicorn>=0.30
httpx>=0.27
sqlalchemy>=2.0
rank-bm25>=0.2.2
sse-starlette>=2.1
python-multipart>=0.0.9
qdrant-client>=1.12
langgraph>=0.2
truststore>=0.9          # OS trust store: certifi cannot verify api.groq.com
                         # through corporate TLS interception

# Optional -- real neural retrieval. ONNX Runtime, NOT PyTorch, so it installs
# on 3.14 and runs on CPU. Requires model files, which the corporate proxy
# currently blocks; the system falls back to the hashing embedder.
fastembed>=0.4
onnxruntime>=1.19

# Dev
pytest>=8.0
"""

FILES[".env.example"] = """\
# Copy to .env . NEVER commit .env (spec §40).

OPSPILOT_ENVIRONMENT=local
OPSPILOT_LOG_LEVEL=INFO

# ---- models -----------------------------------------------------------------
# fake = deterministic offline provider, zero quota (default; used by CI)
# groq = live API
OPSPILOT_LLM_BACKEND=fake
OPSPILOT_GROQ_API_KEY=

# Task tiering: cheap+fast for routing, stronger for final generation.
# The judge MUST differ from the generator, or faithfulness scores are
# self-evaluation (see docs/audit/SPEC-AUDIT.md G5).
OPSPILOT_MODEL_ROUTER=openai/gpt-oss-20b
OPSPILOT_MODEL_GENERATOR=openai/gpt-oss-120b
OPSPILOT_MODEL_JUDGE=llama-3.3-70b-versatile

# ---- quota guard ------------------------------------------------------------
# Measured Groq free tier: 30 RPM / 1,000 RPD / 8,000 TPM / 200,000 TPD.
# CONFIRM YOUR OWN LIMITS on the Groq console before trusting these.
OPSPILOT_DAILY_TOKEN_BUDGET=200000
OPSPILOT_TOKENS_PER_MINUTE_BUDGET=8000

# ---- retrieval --------------------------------------------------------------
# hashing = no model download, works offline; onnx = bge-small via fastembed
OPSPILOT_EMBEDDING_BACKEND=hashing
OPSPILOT_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
OPSPILOT_RERANKER_BACKEND=lexical
OPSPILOT_CANDIDATE_K=30
OPSPILOT_FINAL_K=6
OPSPILOT_RRF_K=60

# Chosen from a measured sweep, not guessed:
# artifacts/reports/abstention_calibration.md
OPSPILOT_MIN_EVIDENCE_SCORE=1.5

# ---- storage ----------------------------------------------------------------
# Local default is SQLite + embedded Qdrant (no Docker available).
# OPSPILOT_DATABASE_URL=postgresql+psycopg://opspilot:pw@localhost:5432/opspilot
# OPSPILOT_QDRANT_URL=http://localhost:6333

# ---- security ---------------------------------------------------------------
OPSPILOT_JWT_SECRET=change-me
OPSPILOT_APPROVAL_TTL_SECONDS=300
OPSPILOT_RATE_LIMIT_PER_MINUTE=30
"""

FILES["Makefile"] = """\
.PHONY: install data eval abstention adversarial test serve demo all clean
PY := .venv/Scripts/python.exe

install:
\t$(PY) -m pip install -r requirements.txt

data eval:          ## generate corpus + eval set, build index, run experiments
\t$(PY) -m scripts.build_index

abstention:
\t$(PY) scripts/calibrate_abstention.py

adversarial:
\t$(PY) scripts/run_adversarial.py

test:
\t$(PY) -m pytest -q

serve:
\t$(PY) -m uvicorn opspilot.api.main:app --reload --port 8000

demo:
\t$(PY) scripts/demo_scenarios.py --base http://127.0.0.1:8000

all: data abstention adversarial test

clean:
\trm -rf data/qdrant data/opspilot.db artifacts/reports/*.json
"""

FILES["frontend/README.md"] = """\
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
"""

FILES[".gitignore"] = """\
.venv/
__pycache__/
*.pyc
.env
data/qdrant/
data/opspilot.db
data/generated/
data/eval/
artifacts/
.pytest_cache/
node_modules/
.next/
"""

for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"wrote {rel}")
print(f"\n{len(FILES)} files")
