"use client";

import { useEffect, useState } from "react";

/** Shared data-loading wrapper: one place for loading and error states. */
export function usePanel<T>(load: () => Promise<T>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    load()
      .then((d) => alive && setData(d))
      .catch((e: Error) => alive && setError(e.message));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { data, error };
}

export function PanelState({ error, empty }: { error?: string | null; empty?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-[color:var(--line)] p-8 text-center text-sm text-[color:var(--mut)]">
      {error ?? empty ?? "Loading…"}
    </div>
  );
}

export function Pill({ value }: { value: string }) {
  const tone =
    ["Critical", "SEV1", "active", "open"].includes(value)
      ? "warn"
      : ["resolved", "closed"].includes(value)
        ? "ok"
        : null;
  return (
    <span
      className="rounded-full border px-2 py-0.5 text-[11px]"
      style={{
        color: tone ? `var(--${tone})` : "var(--mut)",
        borderColor: tone ? `var(--${tone})` : "var(--line)",
      }}
    >
      {value}
    </span>
  );
}

export function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return (
    <table className="w-full border-collapse text-[13.5px]">
      <thead>
        <tr>
          {head.map((h) => (
            <th
              key={h}
              className="border-b border-[color:var(--line)] px-3 py-2 text-left text-[11px] uppercase tracking-wide text-[color:var(--mut)]"
            >
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>{children}</tbody>
    </table>
  );
}

export function Td({ children }: { children: React.ReactNode }) {
  return <td className="border-b border-[color:var(--line)] px-3 py-2">{children}</td>;
}
