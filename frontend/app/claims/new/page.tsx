"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { PhotoPicker } from "@/components/PhotoPicker";
import { api } from "@/lib/api";
import { day } from "@/lib/format";
import { photoProblems } from "@/lib/photos";
import type { Policy } from "@/lib/types";

const MIN_DESCRIPTION = 10;
const MAX_DESCRIPTION = 4000;

function today() {
  const d = new Date();
  return new Date(d.getTime() - d.getTimezoneOffset() * 60_000).toISOString().slice(0, 10);
}

export default function NewClaim() {
  const router = useRouter();
  const [policies, setPolicies] = useState<Policy[] | null>(null);
  const [policyId, setPolicyId] = useState("");
  const [incidentDate, setIncidentDate] = useState("");
  const [description, setDescription] = useState("");
  const [amount, setAmount] = useState("");
  const [photos, setPhotos] = useState<File[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api<{ policies: Policy[] }>("/me/policies")
      .then(({ policies }) => {
        setPolicies(policies);
        const active = policies.find((p) => p.status === "active");
        if (active) setPolicyId(active.policy_id);
      })
      .catch((e) => setError(e.message));
  }, []);

  const ready =
    policyId &&
    incidentDate &&
    description.trim().length >= MIN_DESCRIPTION &&
    photoProblems(photos).length === 0;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.set("policy_id", policyId);
    form.set("incident_date", incidentDate);
    form.set("description", description.trim());
    if (amount) form.set("claimed_amount", amount);
    for (const p of photos) form.append("photos", p);
    try {
      const { claim_id } = await api<{ claim_id: string }>("/claims", { method: "POST", body: form });
      router.push(`/claims/${claim_id}`);
    } catch (err) {
      setError((err as Error).message);
      setBusy(false);
    }
  }

  const policy = policies?.find((p) => p.policy_id === policyId);

  return (
    <form onSubmit={submit} className="mx-auto max-w-3xl space-y-10">
      <header className="rise">
        <p className="label">New claim</p>
        <h1 className="font-serif text-5xl">What happened?</h1>
        <p className="mt-2 max-w-xl text-graphite">
          Tell us in your own words and add photos of the damage. We&apos;ll assess it and an adjuster will
          make the decision.
        </p>
      </header>

      <Section n="01" title="Policy and date">
        <div className="grid gap-4 sm:grid-cols-2">
          <label className="block space-y-1.5">
            <span className="text-sm text-graphite">Policy</span>
            <select className="field" value={policyId} onChange={(e) => setPolicyId(e.target.value)} required>
              {!policies && <option>Loading…</option>}
              {policies?.length === 0 && <option value="">No policies on file</option>}
              {policies?.map((p) => (
                <option key={p.policy_id} value={p.policy_id} disabled={p.status !== "active"}>
                  {p.policy_id} · {p.vehicle}
                  {p.status !== "active" ? ` (${p.status})` : ""}
                </option>
              ))}
            </select>
            {policy && (
              <span className="block text-xs text-faint">
                {day(policy.start_date)} – {day(policy.end_date)} · collision {policy.collision ? "✓" : "✗"} ·
                comprehensive {policy.comprehensive ? "✓" : "✗"}
              </span>
            )}
          </label>
          <label className="block space-y-1.5">
            <span className="text-sm text-graphite">Date of the incident</span>
            <input
              className="field"
              type="date"
              max={today()}
              value={incidentDate}
              onChange={(e) => setIncidentDate(e.target.value)}
              required
            />
          </label>
        </div>
      </Section>

      <Section n="02" title="Description">
        <textarea
          className="field min-h-36 resize-y"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          maxLength={MAX_DESCRIPTION}
          placeholder="Where and how it happened, which parts of the car are damaged, whether anyone else was involved."
          required
        />
        <div className="flex justify-between text-xs text-faint">
          <span>{description.trim().length < MIN_DESCRIPTION ? `At least ${MIN_DESCRIPTION} characters` : " "}</span>
          <span className="font-mono">
            {description.length}/{MAX_DESCRIPTION}
          </span>
        </div>
        <label className="block max-w-56 space-y-1.5">
          <span className="text-sm text-graphite">Amount you&apos;re claiming (optional)</span>
          <div className="flex items-center">
            <span className="border border-r-0 border-rule bg-paper px-3 py-2.5 font-mono text-sm">$</span>
            <input
              className="field font-mono"
              type="number"
              min="1"
              max="1000000"
              step="0.01"
              inputMode="decimal"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
            />
          </div>
        </label>
      </Section>

      <Section n="03" title="Photos">
        <PhotoPicker files={photos} onChange={setPhotos} />
      </Section>

      {error && (
        <p role="alert" className="border-l-2 border-signal pl-3 text-rust">
          {error}
        </p>
      )}
      <div className="flex items-center justify-end gap-4 border-t border-ink pt-6">
        <span className="text-sm text-faint">
          {photos.length} photo{photos.length === 1 ? "" : "s"}
        </span>
        <button className="btn btn-ink" disabled={!ready || busy}>
          {busy ? "Uploading…" : "Submit claim"}
        </button>
      </div>
    </form>
  );
}

function Section({ n, title, children }: { n: string; title: string; children: React.ReactNode }) {
  return (
    <section className="rise grid gap-4 sm:grid-cols-[5rem_1fr]">
      <div className="flex items-baseline gap-3 sm:block">
        <span className="font-mono text-sm text-signal">{n}</span>
        <h2 className="font-serif text-xl sm:mt-1">{title}</h2>
      </div>
      <div className="space-y-3">{children}</div>
    </section>
  );
}
