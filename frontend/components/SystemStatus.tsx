"use client";

import type { Metrics } from "@/lib/api";
import { PanelState } from "@/components/Panel";

export default function SystemStatus({ metrics }: { metrics: Metrics | null }) {
  if (!metrics) return <div className="p-6"><PanelState /></div>;

  const budget = "tpm_limit" in metrics.token_budget ? metrics.token_budget : null;
  const cache = metrics.cache as Record<string, number>;
  const mcp = metrics.mcp as Record<string, number>;

  const stats: [string, string, string?][] = [
    ["Requests", String(metrics.requests_total), `${metrics.errors_total} errors`],
    ["Latency p95", String(metrics.latency_ms.p95), "ms"],
    ["Tokens today", budget ? String(budget.tokens_today) : "—",
      budget ? `/ ${budget.tpd_limit}` : "offline provider"],
    ["MCP calls", String(mcp.calls ?? 0), `${mcp.fallbacks_to_in_process ?? 0} fallbacks`],
    ["Cache hit rate", `${(((cache.hit_rate as number) ?? 0) * 100).toFixed(0)}%`,
      `${cache.llm_calls_avoided ?? 0} calls avoided`],
    ["Authz denials", String(metrics.authorization_denials)],
    ["Corpus", String(metrics.index.documents), `docs / ${metrics.index.chunks} chunks`],
    ["Audit events", String(metrics.audit_events)],
  ];

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h3 className="mb-1 text-base font-semibold">System status</h3>
      <p className="mb-4 text-sm text-[color:var(--mut)]">Live counters from this process.</p>
      <div className="mb-6 grid gap-3" style={{ gridTemplateColumns: "repeat(auto-fit,minmax(190px,1fr))" }}>
        {stats.map(([k, v, sub]) => (
          <div key={k} className="rounded-xl border border-[color:var(--line)] bg-[color:var(--panel)] p-4">
            <div className="text-[11.5px] uppercase tracking-wide text-[color:var(--mut)]">{k}</div>
            <div className="mt-1 text-[22px] font-semibold">
              {v} {sub && <small className="text-xs font-normal text-[color:var(--mut)]">{sub}</small>}
            </div>
          </div>
        ))}
      </div>
      <pre className="overflow-auto rounded-lg border border-[color:var(--line)] bg-[color:var(--panel)] p-3 text-xs">
        {JSON.stringify(metrics, null, 2)}
      </pre>
    </div>
  );
}
