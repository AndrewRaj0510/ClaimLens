"use client";

import { useState } from "react";

import { money, time, titleCase } from "@/lib/format";
import type { ClaimState, Clause, Coverage, Estimate, Opinion, Range, TraceStep } from "@/lib/types";

export function Panel({
  title,
  aside,
  children,
}: {
  title: string;
  aside?: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <section className="sheet">
      <header className="flex items-baseline justify-between gap-3 border-b border-rule px-5 py-3">
        <h2 className="font-serif text-xl">{title}</h2>
        {aside}
      </header>
      <div className="p-5">{children}</div>
    </section>
  );
}

export function Tag({ children, tone = "faint" }: { children: React.ReactNode; tone?: "faint" | "amber" }) {
  return (
    <span className={`border px-1.5 py-px font-mono text-[10px] uppercase tracking-wider ${tone === "amber" ? "border-amber text-amber" : "border-rule text-faint"}`}>
      {children}
    </span>
  );
}

/** Where the LLM's answer came from: an LLM fallback is worth flagging. */
export function SourceTag({ source }: { source?: string }) {
  if (!source) return null;
  return <Tag tone={source === "fallback" ? "amber" : "faint"}>{source === "llm" ? "LLM" : source}</Tag>;
}

// ---- cost ----

/**
 * The estimate's range on one axis with the deductible and the claimed
 * amount marked, so "claimed is 3x the high estimate" is visible at a glance.
 */
export function CostRange({
  estimate,
  payable,
  deductible,
  claimed,
}: {
  estimate: Estimate;
  payable: Range;
  deductible: number | null;
  claimed: number | null;
}) {
  const max = Math.max(estimate.high, claimed ?? 0, deductible ?? 0) * 1.08 || 1;
  const at = (v: number) => `${(v / max) * 100}%`;
  return (
    <div className="space-y-5">
      <div className="grid grid-cols-3 gap-px border border-rule bg-rule text-center">
        {(
          [
            ["Low", estimate.low],
            ["Estimate", estimate.total],
            ["High", estimate.high],
          ] as const
        ).map(([label, value]) => (
          <div key={label} className="bg-sheet px-2 py-3">
            <p className="label">{label}</p>
            <p className={`font-mono ${label === "Estimate" ? "text-xl" : "text-base text-graphite"}`}>{money(value)}</p>
          </div>
        ))}
      </div>

      <div className="relative h-12" role="img" aria-label={`Estimate ${money(estimate.low)} to ${money(estimate.high)}`}>
        <div className="absolute inset-x-0 top-5 h-px bg-rule" />
        <div className="absolute top-3 h-5 bg-ink/15" style={{ left: at(estimate.low), width: `calc(${at(estimate.high)} - ${at(estimate.low)})` }} />
        <div className="absolute top-2 h-7 w-0.5 bg-ink" style={{ left: at(estimate.total) }} />
        {deductible != null && deductible > 0 && (
          <Marker x={at(deductible)} label={`ded. ${money(deductible)}`} className="text-graphite" />
        )}
        {claimed != null && <Marker x={at(claimed)} label={`claimed ${money(claimed)}`} className="text-signal" />}
      </div>

      <dl className="grid grid-cols-2 gap-y-1 text-sm">
        <dt className="text-graphite">Payable after deductible</dt>
        <dd className="text-right font-mono">
          {money(payable.low)} – {money(payable.high)}
        </dd>
        <dt className="text-graphite">Expected payout</dt>
        <dd className="text-right font-mono text-lg">{money(payable.expected)}</dd>
      </dl>

      <table className="w-full text-sm">
        <thead>
          <tr className="label text-left">
            <th className="pb-1 font-normal">Repair</th>
            <th className="pb-1 text-right font-normal">Parts</th>
            <th className="pb-1 text-right font-normal">Hours</th>
            <th className="pb-1 text-right font-normal">Cost</th>
          </tr>
        </thead>
        <tbody>
          {estimate.line_items.map((li, i) => (
            <tr key={i} className="border-t border-rule">
              <td className="py-1.5">
                {titleCase(li.damage_class)} <span className="text-faint">· {li.operation}</span>
              </td>
              <td className="text-right font-mono">{money(li.parts)}</td>
              <td className="text-right font-mono">{(li.labour_hours + li.paint_hours).toFixed(1)}</td>
              <td className="text-right font-mono">{money(li.cost)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-faint">
        Labour {money(estimate.labour_rate_per_hour)}/h. {estimate.assumptions.join(" ")}
      </p>
    </div>
  );
}

function Marker({ x, label, className }: { x: string; label: string; className: string }) {
  return (
    <div className={`absolute top-0 ${className}`} style={{ left: x }}>
      <div className="h-9 border-l border-dashed border-current" />
      <span className="absolute top-9 -translate-x-1/2 whitespace-nowrap font-mono text-[10px]">{label}</span>
    </div>
  );
}

// ---- fraud ----

const LEVEL_COLOR = { low: "text-moss", medium: "text-amber", high: "text-rust" } as const;

export function FraudPanel({ fraud }: { fraud: NonNullable<ClaimState["fraud"]> }) {
  return (
    <div className="space-y-4">
      <div className="flex items-end justify-between">
        <p className={`font-mono text-4xl ${LEVEL_COLOR[fraud.level]}`}>{fraud.score.toFixed(2)}</p>
        <p className={`font-mono text-sm uppercase tracking-widest ${LEVEL_COLOR[fraud.level]}`}>{fraud.level} risk</p>
      </div>
      {/* Zones match the agent's levels: medium from 0.3, high from 0.6. */}
      <div className="relative" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={fraud.score} aria-label="Fraud score">
        <div className="flex h-2">
          <div className="w-[30%] bg-moss/30" />
          <div className="w-[30%] bg-amber/35" />
          <div className="w-[40%] bg-rust/35" />
        </div>
        <div className="absolute -top-1.5 h-5 w-1 -translate-x-1/2 bg-ink" style={{ left: `${fraud.score * 100}%` }} />
        <div className="mt-1 flex justify-between font-mono text-[10px] text-faint">
          <span>0</span>
          <span className="ml-[22%]">0.3</span>
          <span className="ml-[22%]">0.6</span>
          <span className="ml-auto">1</span>
        </div>
      </div>
      {fraud.signals.length === 0 ? (
        <p className="text-sm text-graphite">No fraud signals.</p>
      ) : (
        <ul className="space-y-2">
          {fraud.signals.map((s) => (
            <li key={s.signal} className="grid grid-cols-[3rem_1fr] gap-2 text-sm">
              <span className="font-mono text-rust">+{s.weight.toFixed(2)}</span>
              <span>
                <span className="font-medium">{titleCase(s.signal)}</span>
                <span className="block text-graphite">{s.detail}</span>
              </span>
            </li>
          ))}
        </ul>
      )}
      <p className="border-t border-rule pt-3 text-xs text-faint">
        History: {fraud.history.claims_last_12_months} claim(s) in 12 months, {fraud.history.past_claims} on file,{" "}
        {money(fraud.history.total_paid)} paid. Signals combine as 1 − Π(1 − w): they add up but never pass 1.
      </p>
    </div>
  );
}

// ---- policy ----

function Check({ ok, label }: { ok: boolean | null; label: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={`grid size-5 place-items-center border font-mono text-xs ${ok === null ? "border-rule text-faint" : ok ? "border-moss text-moss" : "border-rust text-rust"}`}>
        {ok === null ? "?" : ok ? "✓" : "✗"}
      </span>
      {label}
    </li>
  );
}

const ASSESSMENT = {
  covered: ["Covered", "text-moss"],
  not_covered: ["Not covered", "text-rust"],
  unclear: ["Unclear", "text-amber"],
} as const;

export function PolicyPanel({ coverage, opinion, clauses }: { coverage: Coverage; opinion: Opinion; clauses: Clause[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const [label, color] = ASSESSMENT[opinion.assessment];
  return (
    <div className="space-y-5">
      <div>
        <p className="label mb-2">Checked by code</p>
        <ul className="space-y-1.5 text-sm">
          <Check ok={coverage.in_force} label="Policy in force on the incident date" />
          <Check ok={coverage.peril_covered} label="Peril covered by the policy" />
          <li className="pl-7 text-graphite">
            Deductible <span className="font-mono text-ink">{money(coverage.deductible)}</span>
          </li>
        </ul>
        {coverage.error && <p className="mt-2 text-sm text-rust">{coverage.error}</p>}
        {coverage.reasons.length > 0 && (
          <ul className="mt-2 list-inside list-disc text-sm text-graphite">
            {coverage.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>

      <div className="border-t border-rule pt-4">
        <div className="mb-2 flex items-center gap-2">
          <p className="label">Reading of the wording</p>
          <SourceTag source={opinion.source} />
        </div>
        <p className={`font-serif text-2xl ${color}`}>{label}</p>
        <p className="mt-1 text-sm leading-relaxed">{opinion.reasoning}</p>
        {opinion.concerns.length > 0 && (
          <ul className="mt-2 space-y-1 text-sm">
            {opinion.concerns.map((c) => (
              <li key={c} className="border-l-2 border-amber pl-2">
                {c}
              </li>
            ))}
          </ul>
        )}
        {opinion.citations.length > 0 && (
          <p className="mt-3 flex flex-wrap items-center gap-1.5 text-sm">
            <span className="text-graphite">Cites</span>
            {opinion.citations.map((id) => (
              <button
                key={id}
                type="button"
                onClick={() => setOpen(open === id ? null : id)}
                className="border border-ink px-1.5 font-mono text-xs hover:bg-ink hover:text-sheet"
              >
                {id}
              </button>
            ))}
          </p>
        )}
      </div>

      {clauses.length > 0 && (
        <div className="border-t border-rule pt-4">
          <p className="label mb-2">Clauses retrieved · {clauses.length}</p>
          <ul className="space-y-1">
            {clauses.map((c) => {
              const cited = opinion.citations.includes(c.id);
              const isOpen = open === c.id;
              return (
                <li key={c.id} className={cited ? "border-l-2 border-signal pl-2" : "pl-2.5"}>
                  <button
                    type="button"
                    onClick={() => setOpen(isOpen ? null : c.id)}
                    aria-expanded={isOpen}
                    className="flex w-full items-baseline gap-2 text-left text-sm hover:text-signal"
                  >
                    <span className="font-mono text-xs">{c.id}</span>
                    <span className="flex-1 truncate">{c.section}</span>
                    {c.withheld && <Tag tone="amber">withheld</Tag>}
                    <span className="font-mono text-[10px] text-faint">
                      p.{c.page_start}
                      {c.page_end !== c.page_start ? `–${c.page_end}` : ""}
                    </span>
                  </button>
                  {isOpen && (
                    <blockquote className="mt-1 mb-2 max-h-64 overflow-auto bg-paper p-3 font-serif text-[0.95rem] leading-snug whitespace-pre-line">
                      {c.text}
                      <footer className="mt-2 font-mono text-[10px] text-faint not-italic">{c.wording_id}</footer>
                    </blockquote>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}

// ---- trace ----

export function Trace({ steps, pending }: { steps: TraceStep[]; pending?: string | null }) {
  return (
    <ol className="relative space-y-4 border-l border-ink pl-5">
      {steps.map((s, i) => (
        <li key={i} className="relative">
          <span className="absolute -left-[1.6rem] top-1 size-2.5 border border-ink bg-sheet" />
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="font-mono text-xs text-faint">{time(s.at)}</span>
            <span className="font-mono text-sm">{s.node}</span>
            <SourceTag source={s.source} />
          </div>
          <p className="text-sm text-graphite">{s.summary}</p>
          {s.problems?.map((p) => (
            <p key={p} className="text-xs text-rust">
              {p}
            </p>
          ))}
        </li>
      ))}
      {pending && (
        <li className="relative">
          <span className="absolute -left-[1.6rem] top-1 size-2.5 animate-pulse bg-signal" />
          <p className="font-mono text-sm text-signal">{pending}</p>
        </li>
      )}
    </ol>
  );
}
