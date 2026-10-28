"use client";

import Link from "next/link";
import { useState } from "react";

import { Stamp } from "@/components/Stamp";
import { api } from "@/lib/api";
import { RECOMMENDATION_LABELS, day, money, shortId, time } from "@/lib/format";
import type { AdjusterClaimRow, ClaimStatus } from "@/lib/types";
import { usePoll } from "@/lib/usePoll";

const FILTERS: { status: ClaimStatus | null; label: string }[] = [
  { status: "awaiting_approval", label: "Needs decision" },
  { status: null, label: "All" },
  { status: "processing", label: "Processing" },
  { status: "awaiting_retake", label: "Awaiting photos" },
  { status: "failed", label: "Failed" },
  { status: "completed", label: "Completed" },
];

const FRAUD_TONE = { low: "text-moss", medium: "text-amber", high: "text-rust font-medium" } as const;

export default function Queue() {
  const [filter, setFilter] = useState<ClaimStatus | null>("awaiting_approval");
  const { data, error } = usePoll(
    () => api<{ claims: AdjusterClaimRow[] }>(`/claims${filter ? `?status=${filter}` : ""}`),
    () => true, // a queue: new claims arrive at any time
    5000,
  );
  // Rows from the previous filter stay visible until the new ones arrive.
  const rows = data?.claims.filter((c) => !filter || c.status === filter) ?? null;

  return (
    <div className="space-y-6">
      <header className="rise flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="label">Adjuster desk</p>
          <h1 className="font-serif text-5xl">Claims queue</h1>
        </div>
        <p className="font-mono text-sm text-graphite">{rows ? `${rows.length} shown` : ""}</p>
      </header>

      <nav className="flex flex-wrap gap-px border border-ink bg-ink" aria-label="Filter by status">
        {FILTERS.map((f) => (
          <button
            key={f.label}
            type="button"
            onClick={() => setFilter(f.status)}
            aria-pressed={filter === f.status}
            className={`flex-1 px-3 py-2 text-sm whitespace-nowrap ${filter === f.status ? "bg-ink text-sheet" : "bg-sheet hover:bg-paper"}`}
          >
            {f.label}
          </button>
        ))}
      </nav>

      {error && <p className="text-rust">{error.message}</p>}
      <div className="sheet overflow-x-auto">
        <table className="w-full min-w-[56rem] text-sm">
          <thead>
            <tr className="label border-b border-ink text-left">
              {["Claim", "Status", "Incident", "Damage", "Recommends", "Fraud", "Payable", "Updated"].map((h) => (
                <th key={h} className="px-4 py-3 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows?.map((c) => {
              const s = c.summary;
              return (
                <tr key={c.claim_id} className="group border-b border-rule last:border-0 hover:bg-paper">
                  <td className="px-4 py-3">
                    <Link href={`/adjuster/claims/${c.claim_id}`} className="font-mono group-hover:text-signal">
                      #{shortId(c.claim_id)}
                    </Link>
                    <span className="block text-xs text-faint">
                      {c.policy_id} · {c.customer_id ?? "internal"}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <Stamp status={c.status} role="adjuster" />
                  </td>
                  <td className="px-4 py-3">{day(s.incident_date)}</td>
                  <td className="max-w-56 px-4 py-3 text-graphite">{s.damages?.length ? s.damages.join(", ") : "—"}</td>
                  <td className="px-4 py-3">{s.recommendation ? RECOMMENDATION_LABELS[s.recommendation] : "—"}</td>
                  <td className={`px-4 py-3 font-mono ${s.fraud_level ? FRAUD_TONE[s.fraud_level] : ""}`}>
                    {s.fraud_score != null ? s.fraud_score.toFixed(2) : "—"}
                    {s.security_findings ? (
                      <span className="ml-2 border border-rust px-1 text-[10px] text-rust" title="Guardrails findings">
                        {s.security_findings} sec
                      </span>
                    ) : null}
                  </td>
                  <td className="px-4 py-3 text-right font-mono">{money(s.payable_expected)}</td>
                  <td className="px-4 py-3 font-mono text-xs text-faint">
                    {day(c.updated_at)}
                    <span className="block">{time(c.updated_at)}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {rows?.length === 0 && <p className="p-10 text-center font-serif text-xl text-graphite">Nothing here. The queue is clear.</p>}
        {!rows && !error && <p className="label animate-pulse p-10 text-center">Loading queue…</p>}
      </div>
    </div>
  );
}
