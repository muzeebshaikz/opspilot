"use client";

import { useEffect, useState } from "react";
import { api, auth, type Account } from "@/lib/api";

/**
 * Account picker.
 *
 * Deliberately honest about what it is: selecting an account mints a real
 * signed, expiring session token that the server verifies on every request --
 * but there is no password and no identity provider. It proves the SESSION
 * mechanism, not identity. The note at the bottom says so to the user rather
 * than letting the UI imply otherwise.
 */
export default function SignIn({ onSignedIn }: { onSignedIn: () => void }) {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api
      .directory()
      .then((d) => setAccounts(d.accounts))
      .catch((e: Error) => setError(e.message));
  }, []);

  async function signIn() {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      const { token } = await api.login(selected);
      auth.set(token);
      onSignedIn();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid h-screen place-items-center px-4">
      <div className="w-full max-w-md rounded-2xl border border-[color:var(--line)] bg-[color:var(--panel)] p-7">
        <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold">
          <span className="h-2.5 w-2.5 rounded-full bg-[color:var(--ok)]" />
          OpsPilot AI
        </h1>
        <p className="mb-5 text-sm text-[color:var(--mut)]">
          Sign in to the Acme Technologies IT copilot.
        </p>

        <div className="space-y-2">
          {accounts.map((a) => {
            const off = a.status === "offboarded";
            const on = selected === a.employee_id;
            return (
              <button
                key={a.employee_id}
                onClick={() => setSelected(a.employee_id)}
                className={`flex w-full items-center gap-3 rounded-lg border p-3 text-left transition
                  ${on ? "border-[color:var(--accent)]" : "border-[color:var(--line)]"}
                  ${off ? "opacity-55" : "hover:border-[color:var(--accent)]"}
                  bg-[color:var(--panel2)]`}
              >
                <span className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-[color:var(--accent)] text-xs font-semibold text-white">
                  {a.name.split(" ").map((p) => p[0]).join("").slice(0, 2)}
                </span>
                <span className="min-w-0">
                  <span className="block truncate font-medium">{a.name}</span>
                  <span className="block truncate text-xs text-[color:var(--mut)]">
                    {a.role} · {a.department}
                    {a.status !== "active" ? ` · ${a.status}` : ""}
                  </span>
                </span>
              </button>
            );
          })}
        </div>

        {error && <p className="mt-3 text-sm text-[color:var(--danger)]">{error}</p>}

        <button
          onClick={signIn}
          disabled={!selected || busy}
          className="mt-4 w-full rounded-lg bg-[color:var(--accent)] px-4 py-2.5 font-medium text-white disabled:opacity-50"
        >
          {busy ? "Signing in…" : "Sign in"}
        </button>

        <p className="mt-4 border-t border-[color:var(--line)] pt-3 text-xs leading-relaxed text-[color:var(--mut)]">
          <strong>Demo authentication.</strong> Selecting an account mints a real signed,
          expiring session token that the server verifies on every request — but there is no
          password and no identity provider, so this proves the <em>session</em> mechanism,
          not identity. Authorization (what each role may do) is fully enforced server-side.
        </p>
      </div>
    </main>
  );
}
