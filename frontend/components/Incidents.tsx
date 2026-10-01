"use client";

import { api } from "@/lib/api";
import { PanelState, Pill, Table, Td, usePanel } from "@/components/Panel";

export default function Incidents() {
  const { data, error } = usePanel(() => api.incidents());
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h3 className="mb-1 text-base font-semibold">Active incidents</h3>
      <p className="mb-4 text-sm text-[color:var(--mut)]">
        Live from the incident service, reached through the incident MCP server.
      </p>
      {!data ? (
        <PanelState error={error} />
      ) : data.incidents.length === 0 ? (
        <PanelState empty="No active incidents." />
      ) : (
        <Table head={["ID", "Service", "Severity", "Status", "Affected", "Workaround"]}>
          {data.incidents.map((i) => (
            <tr key={i.incident_id}>
              <Td>{i.incident_id}</Td>
              <Td>{i.service}</Td>
              <Td><Pill value={i.severity} /></Td>
              <Td><Pill value={i.status} /></Td>
              <Td>{i.affected_users}</Td>
              <Td>{(i.workaround ?? "").slice(0, 90)}</Td>
            </tr>
          ))}
        </Table>
      )}
    </div>
  );
}
