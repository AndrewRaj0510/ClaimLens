"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { Wordmark } from "@/components/Shell";
import { postJson } from "@/lib/api";
import { safeNext } from "@/lib/nav";
import type { User } from "@/lib/types";

function LoginForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const user = await postJson<User>("/auth/login", { username, password });
      router.replace(safeNext(params.get("next"), user));
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="sheet rise w-full max-w-sm space-y-5 p-8" style={{ animationDelay: "120ms" }}>
      <div>
        <p className="label">Sign in</p>
        <h2 className="mt-1 font-serif text-3xl">Open your file</h2>
      </div>
      <label className="block space-y-1.5">
        <span className="text-sm text-graphite">Email or username</span>
        <input
          className="field"
          autoComplete="username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          required
        />
      </label>
      <label className="block space-y-1.5">
        <span className="text-sm text-graphite">Password</span>
        <input
          className="field"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </label>
      {error && (
        <p role="alert" className="border-l-2 border-signal pl-3 text-sm text-rust">
          {error}
        </p>
      )}
      <button className="btn btn-ink w-full" disabled={busy}>
        {busy ? "Checking…" : "Sign in"}
      </button>
      <p className="text-xs leading-relaxed text-faint">
        Demo accounts are created with <code className="font-mono">scripts/create_users.py</code>.
      </p>
    </form>
  );
}

export default function LoginPage() {
  return (
    <div className="grid min-h-dvh lg:grid-cols-[1.15fr_1fr]">
      <section className="relative hidden overflow-hidden border-r border-ink bg-ink p-12 text-sheet lg:flex lg:flex-col">
        <Wordmark large />
        <div className="rise mt-auto max-w-lg space-y-8">
          <h1 className="font-serif text-6xl leading-[0.95]">
            Every dent, measured.
            <br />
            <span className="italic text-signal">Every decision, human.</span>
          </h1>
          <ol className="grid grid-cols-3 gap-px border border-sheet/20 bg-sheet/20 font-mono text-xs">
            {[
              ["01", "Vision", "finds and measures damage"],
              ["02", "Agent", "checks policy, price, history"],
              ["03", "Adjuster", "approves, edits or rejects"],
            ].map(([n, title, text]) => (
              <li key={n} className="bg-ink p-4">
                <span className="text-signal">{n}</span>
                <p className="mt-3 text-sm text-sheet">{title}</p>
                <p className="mt-1 text-sheet/50">{text}</p>
              </li>
            ))}
          </ol>
        </div>
        {/* Corner brackets, the viewer's box style, as a quiet motif. */}
        <svg className="pointer-events-none absolute right-12 top-12 size-40 text-signal/60" viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden>
          <path d="M0 22V0h22M78 0h22v22M100 78v22H78M22 100H0V78" />
        </svg>
      </section>
      <section className="flex flex-col items-center justify-center gap-8 p-6">
        <div className="lg:hidden">
          <Wordmark large />
        </div>
        <Suspense>
          <LoginForm />
        </Suspense>
      </section>
    </div>
  );
}
