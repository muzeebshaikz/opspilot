/**
 * Typed client for the OpsPilot API.
 *
 * The types here mirror the FastAPI response models. They are hand-written
 * rather than generated so the contract is reviewable in one place; the
 * authoritative schema is the OpenAPI document at /docs.
 */

export type Outcome =
  | "answered"
  | "partial_evidence"
  | "abstained_no_evidence"
  | "denied_authorization"
  | "clarification";

export interface Principal {
  employee_id: string;
  name: string;
  role: "employee" | "IT_agent" | "IT_admin";
  department: string;
  is_staff: boolean;
}

export interface Account {
  employee_id: string;
  name: string;
  role: string;
  department: string;
  status: string;
}

export interface Citation {
  index: number;
  document_id: string;
  title: string;
  section: string;
  version: string;
  status: string;
  category: string;
  product: string;
  effective_date: string;
  score: number;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  llm_calls?: number;
  cached?: boolean;
  model?: string;
}

export interface ApprovalRequest {
  token: string;
  description: string;
  tool: string;
}

export interface Ticket {
  ticket_id: string;
  summary: string;
  category: string;
  priority: string;
  status: string;
  assigned_team: string;
  created_at: string;
}

export interface Incident {
  incident_id: string;
  service: string;
  severity: string;
  status: string;
  affected_users: number;
  started_at: string;
  description: string;
  workaround: string | null;
}

export interface TokenBudget {
  tokens_this_minute: number;
  tpm_limit: number;
  tokens_today: number;
  tpd_limit: number;
  daily_remaining: number;
  requests_this_minute: number;
}

export interface Metrics {
  requests_total: number;
  errors_total: number;
  latency_ms: { p50: number; p95: number; p99: number };
  cache: Record<string, number | string>;
  token_budget: TokenBudget | { status: string };
  mcp: Record<string, number | string>;
  audit_events: number;
  authorization_denials: number;
  index: { documents: number; chunks: number };
}

export interface ExperimentRow {
  experiment_id: string;
  label: string;
  arm: string;
  reranker: string;
  overall: { recall: number; precision: number; mrr: number; ndcg: number; n: number };
  per_difficulty: Record<string, { ndcg: number; recall: number; n: number }>;
  latency_ms: { p50: number; p95: number; p99: number };
}

export interface Comparison {
  a: string;
  b: string;
  delta: number;
  ci_low: number;
  ci_high: number;
  significant: boolean;
  verdict: string;
}

export interface EvaluationReport {
  status?: string;
  detail?: string;
  context?: Record<string, string | number>;
  experiments?: ExperimentRow[];
  comparisons?: Comparison[];
}

const TOKEN_KEY = "opspilot_token";

export const auth = {
  get token(): string | null {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(TOKEN_KEY);
  },
  set(token: string) {
    window.localStorage.setItem(TOKEN_KEY, token);
  },
  clear() {
    window.localStorage.removeItem(TOKEN_KEY);
  },
};

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const t = auth.token;
  return t ? { ...extra, Authorization: `Bearer ${t}` } : extra;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}) as Record<string, string>);
    throw new Error(body.detail ?? body.message ?? `HTTP ${res.status}`);
  }
  return (await res.json()) as T;
}

export const api = {
  directory: () => fetch("/api/v1/auth/directory").then(json<{ accounts: Account[] }>),

  login: (employee_id: string) =>
    fetch("/api/v1/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ employee_id }),
    }).then(json<{ token: string; expires_in: number }>),

  me: () => fetch("/api/v1/auth/me", { headers: headers() }).then(json<Principal>),

  tickets: () =>
    fetch("/api/v1/tickets", { headers: headers() }).then(json<{ tickets: Ticket[] }>),

  incidents: () =>
    fetch("/api/v1/incidents", { headers: headers() }).then(json<{ incidents: Incident[] }>),

  metrics: () => fetch("/metrics").then(json<Metrics>),

  evaluation: () => fetch("/api/v1/evaluation").then(json<EvaluationReport>),

  approve: (token: string) =>
    fetch(`/api/v1/approvals/${token}`, { method: "POST", headers: headers() }).then(
      json<{ status: string; answer: string; result_ref: string | null }>,
    ),

  decline: (token: string) =>
    fetch(`/api/v1/approvals/${token}/decline`, {
      method: "POST",
      headers: headers(),
    }).then(json<{ status: string }>),
};

/** Server-sent events emitted by POST /api/v1/chat/stream. */
export type ChatEvent =
  | { type: "start"; conversation_id: string }
  | { type: "activity"; step: string }
  | { type: "token"; text: string }
  | { type: "approval_required"; approval: ApprovalRequest }
  | { type: "citations"; citations: Citation[] }
  | { type: "done"; latency_ms: number; outcome: Outcome; usage: Usage }
  | { type: "error"; message: string };

/**
 * Stream a chat turn.
 *
 * SSE is consumed with fetch + ReadableStream rather than EventSource, because
 * EventSource cannot send a POST body or an Authorization header.
 */
export async function* streamChat(
  message: string,
  conversationId: string | null,
): AsyncGenerator<ChatEvent> {
  const res = await fetch("/api/v1/chat/stream", {
    method: "POST",
    headers: headers({ "Content-Type": "application/json" }),
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });

  if (!res.ok || !res.body) {
    const body = await res.json().catch(() => ({}) as Record<string, string>);
    yield {
      type: "error",
      message: body.message ?? body.detail ?? `Request failed (${res.status})`,
    };
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? "";

    for (const frame of frames) {
      const ev = /^event: (.*)$/m.exec(frame)?.[1];
      const raw = /^data: (.*)$/m.exec(frame)?.[1];
      if (!ev || !raw) continue;
      const data = JSON.parse(raw);
      switch (ev) {
        case "start":
          yield { type: "start", conversation_id: data.conversation_id };
          break;
        case "activity":
          yield { type: "activity", step: data.step };
          break;
        case "token":
          yield { type: "token", text: data.text };
          break;
        case "approval_required":
          yield { type: "approval_required", approval: data as ApprovalRequest };
          break;
        case "citations":
          yield { type: "citations", citations: data.citations };
          break;
        case "done":
          yield {
            type: "done",
            latency_ms: data.latency_ms,
            outcome: (data.outcome ?? "answered") as Outcome,
            usage: data.usage ?? {},
          };
          break;
        default:
          break; // ping and unknown events are ignored
      }
    }
  }
}

export const OUTCOME_META: Record<Outcome, { label: string; tone: string; icon: string }> = {
  answered: { label: "Answered from the knowledge base", tone: "ok", icon: "✓" },
  partial_evidence: {
    label: "Related material only — not a confirmed answer",
    tone: "warn",
    icon: "≈",
  },
  abstained_no_evidence: {
    label: "Abstained — not enough evidence to answer safely",
    tone: "warn",
    icon: "⚠",
  },
  denied_authorization: {
    label: "Denied — your role does not permit this",
    tone: "danger",
    icon: "🔒",
  },
  clarification: { label: "Needs clarification", tone: "mut", icon: "?" },
};
