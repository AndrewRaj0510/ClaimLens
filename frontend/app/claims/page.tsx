"use client";

import Link from "next/link";

import { useUser } from "@/components/Shell";
import { Stamp } from "@/components/Stamp";
import { api } from "@/lib/api";
import { DECISION_LABELS, day, money, shortId } from "@/lib/format";
import type { CustomerClaimRow } from "@/lib/types";
import { usePoll } from "@/lib/usePoll";

export default function MyClaims() {
  const user = useUser();
  const { data, error } = usePoll(
    () => api<{ claims: CustomerClaimRow[] }>("/claims"),
    (d) => d.claims.some((c) => c.status === "processing"),
    4000,
  );

  return (
    <div className="space-y-8">
      <div className="rise flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="label">Policyholder · {user.customer_id}</p>
          <h1 className="font-serif text-5xl">Hello, {user.name.split(" ")[0]}.</h1>
        </div>
        <Link href="/claims/new" className="btn btn-ink">
          File a claim →
        </Link>
      </div>

      {error && <p className="text-rust">{error.message}</p>}
      {data && data.claims.length === 0 && (
        <div className="sheet rise p-10 text-center">
          <p className="font-serif text-2xl">No claims on file.</p>
          <p className="mt-2 text-graphite">If your car has been damaged, file a claim with a few photos.</p>
        </div>
      )}
      {data && data.claims.length > 0 && (
        <ul className="space-y-3">
          {data.claims.map((c, i) => (
            <li key={c.claim_id} className="rise" style={{ animationDelay: `${i * 50}ms` }}>
              <Link
                href={`/claims/${c.claim_id}`}
                className="sheet grid items-center gap-4 p-5 transition-transform hover:-translate-y-0.5 sm:grid-cols-[8rem_1fr_auto]"
              >
                <span className="font-mono text-sm">#{shortId(c.claim_id)}</span>
                <span>
                  <span className="block">
                    {c.damages.length ? c.damages.join(", ") : "Damage being assessed"}
                  </span>
                  <span className="text-sm text-graphite">
                    Incident {day(c.incident_date)} · policy {c.policy_id}
                    {c.decision && ` · ${DECISION_LABELS[c.decision.action]} ${c.decision.amount ? money(c.decision.amount) : ""}`}
                  </span>
                </span>
                <Stamp status={c.status} role="customer" />
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
