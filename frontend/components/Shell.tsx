"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Metrics, type Principal } from "@/lib/api";
import Chat from "@/components/Chat";
import Tickets from "@/components/Tickets";
import Incidents from "@/components/Incidents";
import Evaluation from "@/components/Evaluation";
import SystemStatus from "@/components/SystemStatus";

const VIEWS = [
  { key: "chat", label: "Chat" },
  { key: "tickets", label: "My tickets" },
  { key: "incidents", label: "Active incidents" },
  { key: "evaluation", label: "Evaluation" },
  { key: "system", label: "System status" },
] as const;

type ViewKey = (typeof VIEWS)[number]["key"];

export default function Shell({
  me,
  onSignOut,
}: {
  me: Principal;
  onSignOut: () => void;
}) {
  const [view, setView] = useState<ViewKey>("chat");
  const [metrics, setMetrics] = useState<Metrics | null>(null);

  const refresh = useCallback(async () => {
    try {
      setMetrics(await api.metrics());
    } catch {
      /* metrics are advisory; never break the page for them */
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(() => void refresh(), 15000);
    return () => clearInterval(t);
  }, [refresh]);

  const budget =
    metrics && "tpm_limit" in metrics.token_budget ? metrics.token_budget : null;
  const pct = budget ? Math.min(100, (100 * budget.tokens_this_minute) / budget.tpm_limit) : 0;

  return (
    <div className="flex h-screen overflow-hidden">
      <aside className="flex w-56 shrink-0 flex-col border-r border-[color:var(--line)] bg-[color:var(--panel)] p-3">
        <h2 className="mb-4 flex items-center gap-2 px-2 text-[15px] font-semibold">
          <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--ok)]" />
          OpsPilot AI
        </h2>
        <nav className="flex flex-col gap-0.5">
          {VIEWS.map((v) => (
            <button
              key={v.key}
              onClick={() => setView(v.key)}
              className={`rounded-lg px-2.5 py-2 text-left text-[13.5px] transition ${
                view === v.key
                  ? "bg-[color:var(--panel2)] font-medium"
                  : "text-[color:var(--mut)] hover:bg-[color:var(--panel2)]"
              }`}
            >
              {v.label}
            </button>
          ))}
          <a
            href="/docs"
            target="_blank"
            rel="noreferrer"
            className="rounded-lg px-2.5 py-2 text-[13.5px] text-[color:var(--mut)] hover:bg-[color:var(--panel2)]"
          >
            API docs ↗
          </a>
        </nav>

        <div className="mt-auto border-t border-[color:var(--line)] pt-3 text-xs text-[color:var(--mut)]">
          <div className="text-[13px] font-medium text-[color:var(--fg)]">{me.name}</div>
          <div>
            {me.role} · {me.department}
          </div>
          <button
            onClick={onSignOut}
            className="mt-2.5 w-full rounded-lg border border-[color:var(--line)] bg-[color:var(--panel2)] py-1.5 text-xs"
          >
            Sign out
          </button>
        </div>
      </aside>

      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-4 border-b border-[color:var(--line)] px-6 py-3">
          <b className="text-[15px]">{VIEWS.find((v) => v.key === view)?.label}</b>
          <div
            className="flex items-center gap-2 text-[11.5px] text-[color:var(--mut)]"
            title="Groq free tier: 8,000 tokens per minute"
          >
            {budget ? (
              <>
                <span>
                  {budget.tokens_this_minute}/{budget.tpm_limit} tok/min
                </span>
                <span className="h-1.5 w-24 overflow-hidden rounded bg-[color:var(--line)]">
                  <span
                    className="block h-full transition-all"
                    style={{
                      width: `${pct}%`,
                      background:
                        pct > 80
                          ? "var(--danger)"
                          : pct > 50
                            ? "var(--warn)"
                            : "var(--ok)",
                    }}
                  />
                </span>
              </>
            ) : (
              <span>offline provider</span>
            )}
          </div>
        </header>

        {view === "chat" && <Chat onTurnComplete={refresh} />}
        {view === "tickets" && <Tickets />}
        {view === "incidents" && <Incidents />}
        {view === "evaluation" && <Evaluation />}
        {view === "system" && <SystemStatus metrics={metrics} />}
      </main>
    </div>
  );
}
