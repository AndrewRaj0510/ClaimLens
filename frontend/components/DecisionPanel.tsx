"use client";

import { useState } from "react";

import { postJson } from "@/lib/api";
import { RECOMMENDATION_LABELS, money } from "@/lib/format";
import type { DecisionAction, Range, Recommendation } from "@/lib/types";

const ACTIONS: { action: DecisionAction; label: string }[] = [
  { action: "approve", label: "Approve" },
  { action: "edit", label: "Edit amount" },
  { action: "reject", label: "Reject" },
];

const REC_TONE: Record<Recommendation, string> = {
  approve: "border-moss text-moss",
  review: "border-amber text-amber",
  deny: "border-rust text-rust",
  refer_to_investigator: "border-rust text-rust",
};

/**
 * The adjuster's decision: the only way a claim is ever decided. Two steps
 * (choose, then confirm), because a decision can't be changed afterwards.
 * The adjuster's name isn't sent: the gateway takes it from the session.
 */
export function DecisionPanel({
  claimId,
  recommendation,
  payable,
  onDecided,
}: {
  claimId: string;
  recommendation: { action: Recommendation; reasons: string[] };
  payable: Range;
  onDecided: () => void;
}) {
  const [action, setAction] = useState<DecisionAction | null>(null);
  const [amount, setAmount] = useState(String(payable.expected));
  const [notes, setNotes] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const parsed = Number(amount);
  const amountOk = amount.trim() !== "" && Number.isFinite(parsed) && parsed >= 0 && parsed <= 1_000_000;
  const valid = action !== null && (action !== "edit" || amountOk);
  const finalAmount = action === "edit" ? parsed : action === "approve" ? payable.expected : 0;

  async function submit() {
    if (!action) return;
    setBusy(true);
    setError(null);
    try {
      await postJson(`/claims/${claimId}/decision`, {
        action,
        notes,
        ...(action === "edit" ? { amount: parsed } : {}),
      });
      onDecided();
    } catch (e) {
      setError((e as Error).message);
      setConfirming(false);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="sheet border-ink" aria-label="Decision">
      <div className="space-y-3 border-b border-rule p-5">
        <p className="label">Agent recommends</p>
        <p className={`inline-block border-2 px-2 py-0.5 font-mono text-sm uppercase tracking-wider ${REC_TONE[recommendation.action]}`}>
          {RECOMMENDATION_LABELS[recommendation.action]}
        </p>
        <ul className="space-y-1 text-sm text-graphite">
          {recommendation.reasons.map((r) => (
            <li key={r}>— {r}</li>
          ))}
        </ul>
        <p className="text-xs text-faint">Advisory only. The agent can&apos;t approve or deny anything.</p>
      </div>

      <div className="space-y-4 p-5">
        <div className="grid grid-cols-3 gap-px border border-ink bg-ink" role="radiogroup" aria-label="Decision">
          {ACTIONS.map((a) => (
            <button
              key={a.action}
              type="button"
              role="radio"
              aria-checked={action === a.action}
              onClick={() => {
                setAction(a.action);
                setConfirming(false);
              }}
              className={`py-2 text-sm transition-colors ${
                action === a.action ? (a.action === "reject" ? "bg-rust text-white" : "bg-ink text-sheet") : "bg-sheet hover:bg-paper"
              }`}
            >
              {a.label}
            </button>
          ))}
        </div>

        {action === "edit" && (
          <label className="block space-y-1">
            <span className="text-sm text-graphite">Amount to pay</span>
            <input
              className="field font-mono"
              type="number"
              min="0"
              max="1000000"
              step="0.01"
              value={amount}
              onChange={(e) => {
                setAmount(e.target.value);
                setConfirming(false);
              }}
              aria-invalid={!amountOk}
            />
          </label>
        )}
        {action === "approve" && (
          <p className="text-sm text-graphite">
            Pays the expected amount, <span className="font-mono text-ink">{money(payable.expected)}</span>.
          </p>
        )}

        <label className="block space-y-1">
          <span className="text-sm text-graphite">Notes for the file</span>
          <textarea className="field min-h-20 text-sm" maxLength={2000} value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>

        {error && (
          <p role="alert" className="text-sm text-rust">
            {error}
          </p>
        )}

        {!confirming ? (
          <button type="button" className="btn btn-ink w-full" disabled={!valid} onClick={() => setConfirming(true)}>
            Review decision
          </button>
        ) : (
          <div className="space-y-2 border-2 border-signal p-3">
            <p className="text-sm">
              {action === "reject" ? "Reject this claim" : `Pay ${money(finalAmount)}`}? This can&apos;t be undone.
            </p>
            <div className="flex gap-2">
              <button type="button" className="btn btn-line flex-1" onClick={() => setConfirming(false)} disabled={busy}>
                Back
              </button>
              <button type="button" className="btn btn-ink flex-1" onClick={submit} disabled={busy}>
                {busy ? "Saving…" : "Confirm"}
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
