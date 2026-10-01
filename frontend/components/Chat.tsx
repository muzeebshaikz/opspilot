"use client";

import { useEffect, useRef, useState } from "react";
import {
  api,
  OUTCOME_META,
  streamChat,
  type ApprovalRequest,
  type Citation,
  type Outcome,
  type Usage,
} from "@/lib/api";

interface Turn {
  role: "user" | "assistant";
  text: string;
  activity: string[];
  citations: Citation[];
  approval: ApprovalRequest | null;
  approvalResult: string | null;
  outcome: Outcome | null;
  latency: number | null;
  usage: Usage;
  error: string | null;
}

const SUGGESTIONS = [
  "What does ERR_VPN_809 mean and how do I fix it?",
  "Is there currently a VPN outage?",
  "My VPN keeps disconnecting. Check for an outage and check my laptop.",
  "What is the status of INC-10482?",
  "Create a high priority ticket for this VPN issue",
  "Show me every employee's device information",
  "What is the procedure for tape backup retrieval?",
];

export default function Chat({ onTurnComplete }: { onTurnComplete: () => void }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    setBusy(true);
    setInput("");
    setTurns((t) => [
      ...t,
      blank("user", text),
      blank("assistant", ""),
    ]);

    const idx = turns.length + 1;
    const patch = (fn: (t: Turn) => Turn) =>
      setTurns((all) => all.map((t, i) => (i === idx ? fn(t) : t)));

    try {
      for await (const ev of streamChat(text, conversationId)) {
        switch (ev.type) {
          case "start":
            setConversationId(ev.conversation_id);
            break;
          case "activity":
            patch((t) => ({ ...t, activity: [...t.activity, ev.step] }));
            break;
          case "token":
            patch((t) => ({ ...t, text: t.text + ev.text }));
            break;
          case "approval_required":
            patch((t) => ({ ...t, approval: ev.approval }));
            break;
          case "citations":
            patch((t) => ({ ...t, citations: ev.citations }));
            break;
          case "done":
            patch((t) => ({
              ...t,
              outcome: ev.outcome,
              latency: ev.latency_ms,
              usage: ev.usage,
            }));
            break;
          case "error":
            patch((t) => ({ ...t, error: ev.message }));
            break;
        }
      }
    } finally {
      setBusy(false);
      onTurnComplete();
    }
  }

  async function decide(i: number, token: string, approve: boolean) {
    try {
      // approve() returns an answer; decline() returns only a status. Narrow
      // to a plain string here so the Turn shape stays a single concrete type.
      let message: string;
      if (approve) {
        const r = await api.approve(token);
        message = r.answer || `Request ${r.status}.`;
      } else {
        const r = await api.decline(token);
        message = `Request ${r.status}.`;
      }
      setTurns((all) =>
        all.map((t, j) =>
          j === i ? { ...t, approval: null, approvalResult: message } : t,
        ),
      );
    } catch (e) {
      setTurns((all) =>
        all.map((t, j) => (j === i ? { ...t, error: (e as Error).message } : t)),
      );
    } finally {
      onTurnComplete();
    }
  }

  return (
    <>
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {turns.length === 0 && (
          <p className="pt-10 text-center text-sm text-[color:var(--mut)]">
            Ask about IT support, incidents or your tickets.
          </p>
        )}

        {turns.map((t, i) => (
          <div key={i} className="flex max-w-3xl gap-3">
            <span
              className={`grid h-7 w-7 shrink-0 place-items-center rounded-lg text-[11px] font-semibold text-white ${
                t.role === "user" ? "bg-[#3b4356]" : "bg-[color:var(--accent)]"
              }`}
            >
              {t.role === "user" ? "You" : "AI"}
            </span>

            <div
              className={`min-w-0 rounded-xl border border-[color:var(--line)] px-4 py-3 ${
                t.role === "user" ? "bg-[color:var(--panel2)]" : "bg-[color:var(--panel)]"
              }`}
            >
              {t.activity.length > 0 && (
                <div className="mb-2.5 border-l-2 border-[color:var(--line)] pl-2.5 text-[12.5px] text-[color:var(--mut)]">
                  {t.activity.map((a, k) => (
                    <div key={k}>✓ {a}</div>
                  ))}
                </div>
              )}

              {t.outcome && (
                <div
                  className="mb-2 flex items-center gap-2 text-[12.5px] font-medium"
                  style={{ color: `var(--${OUTCOME_META[t.outcome].tone})` }}
                >
                  <span>{OUTCOME_META[t.outcome].icon}</span>
                  {OUTCOME_META[t.outcome].label}
                </div>
              )}

              <div className="whitespace-pre-wrap break-words">{t.text}</div>

              {t.error && (
                <div className="mt-2 text-sm text-[color:var(--danger)]">{t.error}</div>
              )}

              {t.approval && (
                <div className="mt-3 rounded-lg border border-[color:var(--accent)]/40 bg-[color:var(--accent)]/10 p-3">
                  <p className="mb-2.5 text-[13.5px]">
                    <strong>Approval needed.</strong>
                    <br />
                    {t.approval.description}
                  </p>
                  <div className="flex gap-2">
                    <button
                      onClick={() => decide(i, t.approval!.token, true)}
                      className="rounded-lg bg-[color:var(--accent)] px-4 py-1.5 text-sm text-white"
                    >
                      Approve
                    </button>
                    <button
                      onClick={() => decide(i, t.approval!.token, false)}
                      className="rounded-lg border border-[color:var(--line)] bg-[color:var(--panel2)] px-4 py-1.5 text-sm text-[color:var(--mut)]"
                    >
                      Not now
                    </button>
                  </div>
                </div>
              )}

              {t.approvalResult && (
                <div className="mt-3 rounded-lg border border-[color:var(--ok)]/40 bg-[color:var(--ok)]/10 p-3 text-sm">
                  {t.approvalResult}
                </div>
              )}

              {t.citations.length > 0 && (
                <div className="mt-3 border-t border-[color:var(--line)] pt-2.5 text-[12.5px]">
                  <div className="mb-1.5 text-[color:var(--mut)]">Sources</div>
                  {t.citations.map((c) => (
                    <div key={c.index} className="py-0.5 text-[color:var(--mut)]">
                      <span className="font-semibold text-[color:var(--accent)]">
                        [{c.index}]
                      </span>{" "}
                      {c.title} — {c.section} · v{c.version} · {c.status}
                    </div>
                  ))}
                </div>
              )}

              {t.latency !== null && (
                <div className="mt-2.5 flex flex-wrap gap-1.5">
                  <Badge>{t.latency} ms</Badge>
                  {t.usage.llm_calls !== undefined && (
                    <Badge>{t.usage.llm_calls} LLM call(s)</Badge>
                  )}
                  {t.usage.cached && <Badge tone="ok">cache hit</Badge>}
                  {t.usage.prompt_tokens ? (
                    <Badge>{t.usage.prompt_tokens} prompt tok</Badge>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <div className="flex flex-wrap gap-2 px-6 pb-3">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => void send(s)}
            disabled={busy}
            className="rounded-full border border-[color:var(--line)] bg-[color:var(--panel)] px-3 py-1.5 text-xs text-[color:var(--mut)] hover:border-[color:var(--accent)] disabled:opacity-50"
          >
            {s}
          </button>
        ))}
      </div>

      <div className="flex gap-2.5 border-t border-[color:var(--line)] p-4">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && void send(input)}
          placeholder="Ask about IT support, incidents or your tickets…"
          className="flex-1 rounded-lg border border-[color:var(--line)] bg-[color:var(--panel)] px-3 py-2.5"
        />
        <button
          onClick={() => void send(input)}
          disabled={busy || !input.trim()}
          className="rounded-lg bg-[color:var(--accent)] px-5 font-medium text-white disabled:opacity-50"
        >
          {busy ? "…" : "Send"}
        </button>
      </div>
    </>
  );
}

function Badge({ children, tone }: { children: React.ReactNode; tone?: string }) {
  return (
    <span
      className="rounded-full border px-2.5 py-0.5 text-[11px]"
      style={{
        color: tone ? `var(--${tone})` : "var(--mut)",
        borderColor: tone ? `var(--${tone})` : "var(--line)",
      }}
    >
      {children}
    </span>
  );
}

function blank(role: "user" | "assistant", text: string): Turn {
  return {
    role,
    text,
    activity: [],
    citations: [],
    approval: null,
    approvalResult: null,
    outcome: null,
    latency: null,
    usage: {},
    error: null,
  };
}
