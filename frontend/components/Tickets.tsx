"use client";

import { api } from "@/lib/api";
import { PanelState, Pill, Table, Td, usePanel } from "@/components/Panel";

export default function Tickets() {
  const { data, error } = usePanel(() => api.tickets());
  return (
    <div className="flex-1 overflow-y-auto p-6">
      <h3 className="mb-1 text-base font-semibold">My tickets</h3>
      <p className="mb-4 text-sm text-[color:var(--mut)]">
        Scoped server-side to your own records — an employee cannot read another
        employee&apos;s ticket, regardless of what the model asks for.
      </p>
      {!data ? (
        <PanelState error={error} />
      ) : data.tickets.length === 0 ? (
        <PanelState empty="No tickets." />
      ) : (
        <Table head={["ID", "Summary", "Category", "Priority", "Status", "Team"]}>
          {data.tickets.map((t) => (
            <tr key={t.ticket_id}>
              <Td>{t.ticket_id}</Td>
              <Td>{t.summary}</Td>
              <Td>{t.category}</Td>
              <Td><Pill value={t.priority} /></Td>
              <Td><Pill value={t.status} /></Td>
              <Td>{t.assigned_team}</Td>
            </tr>
          ))}
        </Table>
      )}
    </div>
  );
}
