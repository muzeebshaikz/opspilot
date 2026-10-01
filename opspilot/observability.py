"""OpenTelemetry tracing and Prometheus metrics (spec §21).

Both are OPTIONAL and degrade cleanly. If the OTel packages are missing, or no
collector endpoint is configured, tracing becomes a no-op and the application
runs exactly as before. An observability layer that can take the service down is
worse than no observability layer.

What is instrumented
--------------------
Spec §21 asks for a specific trace shape per request. These are the span names
and attributes actually emitted:

    opspilot.request          request_id · user_id · conversation_id · role
      opspilot.classify_intent    intent · model · tokens
      opspilot.retrieve           arm · candidate_k · final_k · hits · latency
      opspilot.rerank             reranker · in · out
      opspilot.tool               tool · transport(mcp|in_process) · ok
      opspilot.generate           model · prompt_tokens · completion_tokens
                                  · reasoning_tokens · cached · abstained

Metric naming follows Prometheus convention: `_total` for counters, `_seconds`
for durations, base units only.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opspilot.core import get_logger

log = get_logger("observability")

# --------------------------------------------------------------------------
# Prometheus
# --------------------------------------------------------------------------
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"


class Telemetry:
    """Prometheus collectors for the whole application.

    A private registry rather than the global default, so tests can build a
    fresh instance without duplicate-timeseries errors.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        if not self.enabled:
            return

        self.registry = CollectorRegistry()

        self.requests = Counter(
            "opspilot_requests_total",
            "HTTP requests",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "opspilot_request_duration_seconds",
            "End-to-end request duration",
            ["path"],
            registry=self.registry,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
        )

        self.retrieval_duration = Histogram(
            "opspilot_retrieval_duration_seconds",
            "Retrieval stage duration",
            ["arm"],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1),
        )
        self.rerank_duration = Histogram(
            "opspilot_rerank_duration_seconds",
            "Reranking duration",
            ["reranker"],
            registry=self.registry,
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2),
        )

        self.llm_calls = Counter(
            "opspilot_llm_calls_total", "LLM calls", ["model", "outcome"], registry=self.registry
        )
        self.llm_duration = Histogram(
            "opspilot_llm_duration_seconds",
            "LLM call duration",
            ["model"],
            registry=self.registry,
            buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
        )
        self.llm_tokens = Counter(
            "opspilot_llm_tokens_total",
            "Tokens consumed",
            ["model", "kind"],
            registry=self.registry,
        )

        # The binding constraint deserves a first-class gauge, not a log line.
        self.token_budget_remaining = Gauge(
            "opspilot_token_budget_remaining",
            "Daily model token budget remaining",
            registry=self.registry,
        )
        self.token_budget_minute = Gauge(
            "opspilot_token_budget_minute_used",
            "Tokens used in the current minute",
            registry=self.registry,
        )

        self.tool_calls = Counter(
            "opspilot_tool_calls_total",
            "Tool invocations",
            ["tool", "transport", "outcome"],
            registry=self.registry,
        )
        self.tool_duration = Histogram(
            "opspilot_tool_duration_seconds",
            "Tool call duration",
            ["tool", "transport"],
            registry=self.registry,
            buckets=(0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10),
        )

        self.cache_events = Counter(
            "opspilot_cache_events_total",
            "Response cache events",
            ["event"],
            registry=self.registry,
        )
        self.authorization = Counter(
            "opspilot_authorization_total",
            "Authorization decisions",
            ["action", "decision"],
            registry=self.registry,
        )
        self.injection_flags = Counter(
            "opspilot_injection_flags_total",
            "Injection patterns detected",
            ["channel", "stage"],
            registry=self.registry,
        )
        self.outcomes = Counter(
            "opspilot_answer_outcomes_total",
            "Terminal outcome of a chat turn",
            ["outcome"],
            registry=self.registry,
        )

        self.index_documents = Gauge(
            "opspilot_index_documents", "Documents indexed", registry=self.registry
        )
        self.index_chunks = Gauge("opspilot_index_chunks", "Chunks indexed", registry=self.registry)

    # -- helpers ---------------------------------------------------------
    def observe_request(self, method: str, path: str, status: int, seconds: float) -> None:
        if not self.enabled:
            return
        self.requests.labels(method, path, str(status)).inc()
        self.request_duration.labels(path).observe(seconds)

    def observe_llm(
        self,
        model: str,
        seconds: float,
        prompt: int,
        completion: int,
        reasoning: int = 0,
        cached: bool = False,
        ok: bool = True,
    ) -> None:
        if not self.enabled:
            return
        self.llm_calls.labels(model, "cached" if cached else ("ok" if ok else "error")).inc()
        if not cached:
            self.llm_duration.labels(model).observe(seconds)
            self.llm_tokens.labels(model, "prompt").inc(prompt)
            self.llm_tokens.labels(model, "completion").inc(completion)
            if reasoning:
                self.llm_tokens.labels(model, "reasoning").inc(reasoning)

    def observe_tool(self, tool: str, transport: str, ok: bool, seconds: float) -> None:
        if not self.enabled:
            return
        self.tool_calls.labels(tool, transport, "ok" if ok else "error").inc()
        self.tool_duration.labels(tool, transport).observe(seconds)

    def observe_budget(self, snapshot: dict[str, Any]) -> None:
        if not self.enabled or "daily_remaining" not in snapshot:
            return
        self.token_budget_remaining.set(snapshot["daily_remaining"])
        self.token_budget_minute.set(snapshot.get("tokens_this_minute", 0))

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            return (b"# prometheus_client is not installed\n", CONTENT_TYPE_LATEST)
        return (generate_latest(self.registry), CONTENT_TYPE_LATEST)


# --------------------------------------------------------------------------
# OpenTelemetry
# --------------------------------------------------------------------------
_tracer = None
_otel_enabled = False


def setup_tracing(
    service_name: str = "opspilot-backend", endpoint: str | None = None, console: bool = False
) -> bool:
    """Initialise tracing. Returns True when a real tracer was installed.

    Never raises: a misconfigured collector must not stop the service starting.
    """
    global _tracer, _otel_enabled
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name, "service.version": "0.1.0"})
        )
        if endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        elif console:
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        else:
            # No exporter configured: spans are created and dropped. Still
            # useful, because it keeps the instrumentation code exercised.
            pass

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name)
        _otel_enabled = True
        return True
    except Exception as exc:  # pragma: no cover - optional dependency
        log.warning("OpenTelemetry unavailable, tracing disabled: %s", exc)
        _otel_enabled = False
        return False


def instrument_fastapi(app) -> bool:
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="/metrics,/health,/ready")
        return True
    except Exception as exc:  # pragma: no cover
        log.warning("FastAPI OTel instrumentation unavailable: %s", exc)
        return False


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Trace a block. A no-op context manager when tracing is off."""
    if not _otel_enabled or _tracer is None:
        start = time.perf_counter()
        yield None
        del start
        return
    with _tracer.start_as_current_span(name) as s:
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, v)
        yield s


TELEMETRY = Telemetry()
