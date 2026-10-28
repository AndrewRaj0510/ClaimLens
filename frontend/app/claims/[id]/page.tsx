"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { PhotoPicker } from "@/components/PhotoPicker";
import { Stamp } from "@/components/Stamp";
import { api, photoUrl } from "@/lib/api";
import { DECISION_LABELS, customerStep, day, money, shortId } from "@/lib/format";
import { photoProblems } from "@/lib/photos";
import type { CustomerClaim, RetakeQuestion } from "@/lib/types";
import { usePoll } from "@/lib/usePoll";

const STEPS = ["Received", "Assessed", "With an adjuster", "Decided"];

export default function ClaimPage() {
  const { id } = useParams<{ id: string }>();
  const { data: claim, error, restart } = usePoll(
    () => api<CustomerClaim>(`/claims/${id}`),
    (c) => c.status === "processing",
  );

  if (error) {
    return (
      <div className="sheet p-10">
        <p className="font-serif text-2xl">We couldn&apos;t open this claim.</p>
        <p className="mt-2 text-graphite">{error.message}</p>
        <Link href="/claims" className="btn btn-line mt-6">
          ← My claims
        </Link>
      </div>
    );
  }
  if (!claim) return <p className="label animate-pulse">Loading claim…</p>;

  const step = customerStep(claim.status);
  return (
    <div className="mx-auto max-w-4xl space-y-8">
      <header className="rise flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/claims" className="label hover:text-signal">
            ← My claims
          </Link>
          <h1 className="mt-2 font-serif text-5xl">
            Claim <span className="font-mono text-4xl">#{shortId(claim.claim_id)}</span>
          </h1>
          <p className="mt-1 text-graphite">
            Policy {claim.policy_id} · incident {day(claim.incident_date)} · reported {day(claim.reported_date)}
          </p>
        </div>
        <Stamp status={claim.status} role="customer" size="lg" />
      </header>

      <ol className="rise grid grid-cols-4 border border-ink" style={{ animationDelay: "60ms" }}>
        {STEPS.map((label, i) => (
          <li
            key={label}
            className={`relative border-r border-ink p-3 last:border-r-0 ${i <= step ? "bg-ink text-sheet" : "bg-sheet text-faint"}`}
          >
            <span className={`font-mono text-xs ${i <= step ? "text-signal" : ""}`}>0{i + 1}</span>
            <p className="mt-1 text-sm">{label}</p>
            {i === step + 1 && claim.status === "processing" && (
              <span className="absolute inset-x-0 bottom-0 h-0.5 animate-pulse bg-signal" />
            )}
          </li>
        ))}
      </ol>

      {claim.status === "awaiting_retake" && claim.question && (
        <Retake claimId={claim.claim_id} question={claim.question} onSent={restart} />
      )}

      {(claim.status === "failed" || claim.status === "interrupted") && (
        <Notice title="Your claim is delayed">
          Something went wrong on our side while assessing it. Nothing is lost, and our team has been alerted.
        </Notice>
      )}
      {claim.status === "awaiting_approval" && (
        <Notice title="An adjuster is reviewing your claim">
          The assessment is done. A person makes every decision; you&apos;ll see the outcome here.
        </Notice>
      )}

      {claim.outcome && (
        <section className="sheet rise relative overflow-hidden p-8">
          <p className="label">Outcome</p>
          <p className="mt-2 font-serif text-4xl">
            {DECISION_LABELS[claim.outcome.action]}
            {claim.outcome.action !== "reject" && (
              <span className="ml-3 font-mono text-3xl text-moss">{money(claim.outcome.amount)}</span>
            )}
          </p>
          {claim.message && <p className="mt-4 max-w-2xl whitespace-pre-line leading-relaxed">{claim.message}</p>}
        </section>
      )}

      <section className="grid gap-6 sm:grid-cols-[1fr_1fr]">
        <div className="sheet p-6">
          <p className="label">Your description</p>
          <blockquote className="mt-3 border-l-2 border-rule pl-4 font-serif text-lg italic leading-snug">
            {claim.description}
          </blockquote>
          {claim.claimed_amount != null && (
            <p className="mt-4 text-sm text-graphite">
              You claimed <span className="font-mono text-ink">{money(claim.claimed_amount)}</span>
            </p>
          )}
        </div>
        <div className="sheet p-6">
          <p className="label">Photos on file · {claim.photo_count}</p>
          <ul className="mt-3 grid grid-cols-3 gap-2">
            {Array.from({ length: claim.photo_count }, (_, i) => (
              <li key={i} className="aspect-square overflow-hidden border border-rule bg-paper">
                <img src={photoUrl(claim.claim_id, i)} alt={`Photo ${i + 1}`} className="size-full object-cover" loading="lazy" />
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
}

function Notice({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rise border-l-4 border-ink bg-sheet p-5">
      <p className="font-medium">{title}</p>
      <p className="mt-1 text-graphite">{children}</p>
    </div>
  );
}

function Retake({
  claimId,
  question,
  onSent,
}: {
  claimId: string;
  question: RetakeQuestion;
  onSent: () => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function send() {
    setBusy(true);
    setError(null);
    const form = new FormData();
    for (const f of files) form.append("photos", f);
    try {
      await api(`/claims/${claimId}/photos`, { method: "POST", body: form });
      setFiles([]);
      onSent();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="sheet rise space-y-5 border-signal p-6" aria-labelledby="retake-title">
      <div>
        <p className="label text-signal">Action needed</p>
        <h2 id="retake-title" className="mt-1 font-serif text-3xl">
          Please retake {question.photos.length === 1 ? "one photo" : `${question.photos.length} photos`}
        </h2>
      </div>
      <ul className="space-y-3">
        {question.photos.map((p) => (
          <li key={p.photo_index} className="flex gap-4">
            <img
              src={photoUrl(claimId, p.photo_index)}
              alt={`Photo ${p.photo_index + 1}`}
              className="size-16 shrink-0 border border-rule object-cover grayscale"
            />
            <div>
              <p className="font-mono text-sm">Photo {String(p.photo_index + 1).padStart(2, "0")}</p>
              <ul className="text-sm text-graphite">
                {p.problems.map((problem) => (
                  <li key={problem}>{problem}</li>
                ))}
              </ul>
            </div>
          </li>
        ))}
      </ul>
      <PhotoPicker files={files} onChange={setFiles} hint="New photos replace the ones above, in order." />
      {error && <p className="text-sm text-rust">{error}</p>}
      <div className="flex justify-end">
        <button
          type="button"
          className="btn btn-ink"
          disabled={busy || photoProblems(files).length > 0}
          onClick={send}
        >
          {busy ? "Uploading…" : "Send new photos"}
        </button>
      </div>
    </section>
  );
}
