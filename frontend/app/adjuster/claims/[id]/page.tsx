"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { CostRange, FraudPanel, Panel, PolicyPanel, SourceTag, Trace } from "@/components/Assessment";
import { DecisionPanel } from "@/components/DecisionPanel";
import { PhotoViewer } from "@/components/PhotoViewer";
import { PhotoReviewPanel, SecurityPanel } from "@/components/Security";
import { Stamp } from "@/components/Stamp";
import { api } from "@/lib/api";
import { DECISION_LABELS, day, money, shortId, titleCase } from "@/lib/format";
import { classColor } from "@/lib/overlay";
import type { AdjusterClaim, ClaimStatus } from "@/lib/types";
import { usePoll } from "@/lib/usePoll";

const PENDING: Partial<Record<ClaimStatus, string>> = {
  processing: "running…",
  awaiting_retake: "waiting for the customer's photos",
  awaiting_approval: "human_approval · waiting for you",
};

export default function ReviewClaim() {
  const { id } = useParams<{ id: string }>();
  const { data: claim, error, restart } = usePoll(
    () => api<AdjusterClaim>(`/claims/${id}`),
    (c) => c.status === "processing",
  );

  if (error) {
    return (
      <div className="sheet p-10">
        <p className="font-serif text-2xl">Couldn&apos;t open this claim.</p>
        <p className="mt-2 text-graphite">{error.message}</p>
        <Link href="/adjuster" className="btn btn-line mt-6">
          ← Queue
        </Link>
      </div>
    );
  }
  if (!claim) return <p className="label animate-pulse">Loading claim…</p>;

  const s = claim.state;
  const approval = claim.question?.type === "approval" ? claim.question : null;
  const deductible = s.coverage?.deductible ?? null;

  return (
    <div className="space-y-6">
      <header className="rise flex flex-wrap items-start justify-between gap-4 border-b border-ink pb-6">
        <div>
          <Link href="/adjuster" className="label hover:text-signal">
            ← Queue
          </Link>
          <h1 className="mt-2 font-serif text-5xl">
            Claim <span className="font-mono text-4xl">#{shortId(claim.claim_id)}</span>
          </h1>
          <p className="mt-1 text-graphite">
            Policy {s.policy_id} · customer {s.customer_id ?? "internal"}
            {s.policy && ` · ${s.policy.vehicle.year} ${s.policy.vehicle.make} ${s.policy.vehicle.model} (${s.policy.vehicle.segment}), ${s.policy.state}`}
          </p>
          <p className="text-sm text-faint">
            Incident {day(s.incident_date)} · reported {day(s.reported_date)} · peril {s.peril ?? "—"}
            {s.retakes ? ` · ${s.retakes} retake(s)` : ""}
          </p>
        </div>
        <Stamp status={claim.status} role="adjuster" size="lg" />
      </header>

      {/* minmax(0, ...) and min-w-0: grid items otherwise grow to their widest
          content (a table, a long ID) and push the page sideways on phones. */}
      <div className="grid grid-cols-[minmax(0,1fr)] items-start gap-6 lg:grid-cols-[minmax(0,1fr)_24rem]">
        <div className="min-w-0 space-y-6">
          <div className="rise">
            <PhotoViewer claimId={claim.claim_id} count={s.photos.length} vision={s.vision ?? []} quality={s.quality ?? []} />
          </div>

          <div className="grid grid-cols-[minmax(0,1fr)] gap-6 md:grid-cols-2">
            <Panel title="Description" aside={<SourceTag source={claim.trace.find((t) => t.node === "intake")?.source} />}>
              <blockquote className="border-l-2 border-rule pl-4 font-serif text-lg italic leading-snug">{s.description}</blockquote>
              <p className="mt-2 text-xs text-faint">Customer&apos;s words: untrusted input, shown as text.</p>
              {s.intake?.screen && s.intake.screen !== "clean" && (
                <p className="mt-2 border-l-2 border-rust pl-2 text-sm text-rust">
                  {s.intake.screen === "flagged"
                    ? "Flagged as a possible prompt injection: withheld from the AI."
                    : "Couldn't be screened, so it was withheld from the AI."}
                </p>
              )}
              {s.intake && (
                <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <dt className="text-graphite">Peril read</dt>
                  <dd>{s.intake.peril}</dd>
                  <dt className="text-graphite">Date in text</dt>
                  <dd className="font-mono">{s.intake.mentioned_date ?? "—"}</dd>
                  <dt className="text-graphite">Other vehicle</dt>
                  <dd>{s.intake.other_vehicle_involved ? "yes" : "no"}</dd>
                  <dt className="text-graphite">Injuries</dt>
                  <dd className={s.intake.injuries_mentioned ? "text-rust" : ""}>{s.intake.injuries_mentioned ? "mentioned" : "no"}</dd>
                  <dt className="text-graphite">Claimed</dt>
                  <dd className="font-mono">{money(s.claimed_amount)}</dd>
                </dl>
              )}
            </Panel>

            <Panel title="Damage">
              {s.damages?.length ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="label text-left">
                      <th className="pb-1 font-normal">Class</th>
                      <th className="pb-1 font-normal">Severity</th>
                      <th className="pb-1 text-right font-normal">Area</th>
                      <th className="pb-1 text-right font-normal">Conf.</th>
                    </tr>
                  </thead>
                  <tbody>
                    {s.damages.map((d) => (
                      <tr key={d.damage_class} className="border-t border-rule">
                        <td className="py-1.5">
                          <span className="mr-2 inline-block size-2" style={{ background: classColor(d.damage_class) }} />
                          {titleCase(d.damage_class)}
                        </td>
                        <td>{d.severity}</td>
                        <td className="text-right font-mono">{d.area_pct.toFixed(1)}%</td>
                        <td className="text-right font-mono">{Math.round(d.confidence * 100)}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-sm text-graphite">{s.damages ? "No damage detected." : "Not analysed yet."}</p>
              )}
              <p className="mt-3 text-xs text-faint">
                Area is % of the photo (largest view per class). Severity thresholds are class-specific starting points.
              </p>
            </Panel>
          </div>

          {(s.photo_review || s.photo_checks?.length) && (
            <Panel title="Photo review" aside={<span className="text-xs text-faint">vision model + forensics</span>}>
              <PhotoReviewPanel review={s.photo_review} checks={s.photo_checks ?? []} />
            </Panel>
          )}

          {(s.estimate || s.estimate_error) && (
            <Panel title="Repair cost">
              {s.estimate && s.payable ? (
                <CostRange estimate={s.estimate} payable={s.payable} deductible={deductible} claimed={s.claimed_amount} />
              ) : (
                <p className="text-sm text-rust">{s.estimate_error}</p>
              )}
            </Panel>
          )}

          {s.coverage && s.opinion && (
            <Panel title="Policy" aside={s.policy?.wording && <span className="text-xs text-faint">{s.policy.wording.insurer} · {s.policy.wording.title}</span>}>
              <PolicyPanel coverage={s.coverage} opinion={s.opinion} clauses={s.clauses ?? []} />
            </Panel>
          )}
        </div>

        <aside className="min-w-0 space-y-6 lg:sticky lg:top-20">
          {approval && s.recommendation && s.payable && (
            <DecisionPanel claimId={claim.claim_id} recommendation={approval.recommendation} payable={s.payable} onDecided={restart} />
          )}
          {s.decision && (
            <Panel title="Decision">
              <p className="font-serif text-2xl">
                {DECISION_LABELS[s.decision.action]}{" "}
                {s.decision.action !== "reject" && <span className="font-mono text-xl text-moss">{money(s.decision.amount)}</span>}
              </p>
              <p className="mt-1 text-sm text-graphite">by {s.decision.adjuster}</p>
              {s.decision.notes && <p className="mt-2 text-sm">{s.decision.notes}</p>}
              {s.report && (
                <details className="mt-4 text-sm">
                  <summary className="label cursor-pointer">Report</summary>
                  <p className="mt-2 whitespace-pre-line">{s.report.adjuster_summary}</p>
                  <p className="label mt-3">Sent to customer</p>
                  <p className="mt-1 whitespace-pre-line text-graphite">{s.report.customer_message}</p>
                </details>
              )}
            </Panel>
          )}
          {(claim.status === "failed" || claim.status === "interrupted") && (
            <RetryPanel claimId={claim.claim_id} error={claim.error} onRetried={restart} />
          )}
          {s.security && (
            <Panel
              title="Security"
              aside={<span className="font-mono text-xs text-faint">{s.security.length} finding(s)</span>}
            >
              <SecurityPanel findings={s.security} photos={s.photos.length} clauses={s.clauses?.length ?? 0} />
            </Panel>
          )}
          {s.fraud && (
            <Panel title="Fraud signals">
              <FraudPanel fraud={s.fraud} />
            </Panel>
          )}
          <Panel title="Agent trace" aside={<span className="font-mono text-xs text-faint">{claim.trace.length} steps</span>}>
            <Trace steps={claim.trace} pending={PENDING[claim.status]} />
          </Panel>
        </aside>
      </div>
    </div>
  );
}

function RetryPanel({ claimId, error, onRetried }: { claimId: string; error: string | null; onRetried: () => void }) {
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);
  async function retry() {
    setBusy(true);
    try {
      await api(`/claims/${claimId}/retry`, { method: "POST" });
      onRetried();
    } catch (e) {
      setProblem((e as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="sheet space-y-3 border-rust p-5">
      <p className="label text-rust">Run stopped</p>
      <p className="font-mono text-xs break-words text-graphite">{error}</p>
      {problem && <p className="text-sm text-rust">{problem}</p>}
      <button type="button" className="btn btn-ink w-full" onClick={retry} disabled={busy}>
        {busy ? "Retrying…" : "Continue from last checkpoint"}
      </button>
    </section>
  );
}
