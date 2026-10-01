# NOT BUILT OR RUN LOCALLY -- no Docker on the development machine.
# Authored and CI-lintable; treat as unverified until a CI build runs it.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# Python 3.12, not 3.14: sentence-transformers and parts of the ONNX/embedding
# ecosystem do not yet declare 3.14 support. The app itself runs on both.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY opspilot ./opspilot
COPY scripts ./scripts

RUN useradd -m -u 10001 opspilot && chown -R opspilot /app
USER opspilot

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "opspilot.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
