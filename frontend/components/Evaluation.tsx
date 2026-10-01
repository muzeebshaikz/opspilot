"use client";

import { api } from "@/lib/api";
import { PanelState, Table, Td, usePanel } from "@/components/Panel";

/**
 * Spec §25: never render placeholder metrics. When no measured run exists the
 * API returns {status: "Not evaluated yet"} and this renders exactly that,
 * rather than zeros that look like results.
 */
export default function Evaluation() {
  const { data, error } = usePanel(() => api.evaluation());

  if (!data) return <div className="p-6"><PanelState error={error} /></div>;
  if (data.status) {
    return (
      <div className="p-6">
        <h3 className="mb-1 text-base font-semibold">Evaluation</h3>
        <PanelState empty={`${data.status} — ${data.detail ?? ""}`} />
      </div>
    );
  }

  const classes = Array.from(
    new Set((data.experiments ?? []).flatMap((e) => Object.keys(e.per_difficulty))),
  ).sort();

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h3 className="mb-1 text-base font-semibold">Retrieval evaluation</h3>
      <p className="mb-4 text-sm text-[color:var(--mut)]">
        Produced by an actual run. Nothing here is estimated.
      </p>

      <Table head={["Experiment", "Recall@10", "Precision@10", "MRR", "nDCG@10", "p95"]}>
        {(data.experiments ?? []).map((e) => (
          <tr key={e.experiment_id}>
            <Td>{e.experiment_id} {e.label}</Td>
            <Td>{e.overall.recall.toFixed(3)}</Td>
            <Td>{e.overall.precision.toFixed(3)}</Td>
            <Td>{e.overall.mrr.toFixed(3)}</Td>
            <Td><b>{e.overall.ndcg.toFixed(3)}</b></Td>
            <Td>{e.latency_ms.p95.toFixed(1)} ms</Td>
          </tr>
        ))}
      </Table>

      <h3 className="mb-1 mt-7 text-base font-semibold">nDCG@10 by difficulty class</h3>
      <p className="mb-4 text-sm text-[color:var(--mut)]">
        An aggregate number hides a technique that wins on one class and loses on another.
      </p>
      <Table head={["Experiment", ...classes]}>
        {(data.experiments ?? []).map((e) => (
          <tr key={e.experiment_id}>
            <Td>{e.experiment_id}</Td>
            {classes.map((c) => (
              <Td key={c}>{e.per_difficulty[c]?.ndcg.toFixed(3) ?? "—"}</Td>
            ))}
          </tr>
        ))}
      </Table>

      <h3 className="mb-1 mt-7 text-base font-semibold">Pairwise significance</h3>
      <p className="mb-4 text-sm text-[color:var(--mut)]">
        Paired bootstrap, 10,000 resamples. An improvement whose CI crosses zero is
        reported as no measurable difference — never as a win.
      </p>
      <Table head={["A", "B", "Δ nDCG", "95% CI", "Verdict"]}>
        {(data.comparisons ?? []).map((c) => (
          <tr key={`${c.a}-${c.b}`}>
            <Td>{c.a}</Td>
            <Td>{c.b}</Td>
            <Td>{c.delta >= 0 ? "+" : ""}{c.delta.toFixed(4)}</Td>
            <Td>[{c.ci_low.toFixed(4)}, {c.ci_high.toFixed(4)}]</Td>
            <Td>{c.significant ? <b>{c.verdict}</b> : c.verdict}</Td>
          </tr>
        ))}
      </Table>
    </div>
  );
}
