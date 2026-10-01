"use client";

import { useCallback, useEffect, useState } from "react";
import { api, auth, type Principal } from "@/lib/api";
import SignIn from "@/components/SignIn";
import Shell from "@/components/Shell";

export default function Home() {
  const [me, setMe] = useState<Principal | null>(null);
  const [checking, setChecking] = useState(true);

  const load = useCallback(async () => {
    if (!auth.token) {
      setChecking(false);
      return;
    }
    try {
      setMe(await api.me());
    } catch {
      auth.clear();
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (checking) {
    return <main className="grid h-screen place-items-center text-[color:var(--mut)]">Loading…</main>;
  }
  if (!me) return <SignIn onSignedIn={load} />;
  return <Shell me={me} onSignOut={() => { auth.clear(); setMe(null); }} />;
}
