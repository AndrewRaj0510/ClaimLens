"use client";

import { SourceTag, Tag } from "@/components/Assessment";
import { day } from "@/lib/format";
import type { PhotoCheck, PhotoReview, SecurityFinding, SecurityKind } from "@/lib/types";

const KIND: Record<SecurityKind, [string, string]> = {
  injection: ["Prompt injection in the description", "border-rust"],
  visual_injection: ["Prompt injection in a photo", "border-rust"],
  rag_injection: ["Policy text withheld from the AI", "border-amber"],
  forensics: ["Photo forensics", "border-amber"],
  guardrails_unavailable: ["Check unavailable", "border-amber"],
};

/**
 * What the guardrails found. Flagged text is untrusted: it's rendered as
 * plain text (React escapes it), never as markup or a link.
 */
export function SecurityPanel({
  findings,
  photos,
  clauses,
}: {
  findings: SecurityFinding[];
  photos: number;
  clauses: number;
}) {
  if (findings.length === 0) {
    return (
      <p className="text-sm text-graphite">
        No findings. Screened the description, {photos} photo(s) and {clauses} policy clause(s).
      </p>
    );
  }
  return (
    <div className="space-y-3">
      <ul className="space-y-3">
        {findings.map((f, i) => {
          const [label, border] = KIND[f.kind] ?? [f.kind, "border-rule"];
          return (
            <li key={`${f.source}-${f.kind}-${i}`} className={`border-l-2 pl-3 ${border}`}>
              <p className="flex flex-wrap items-baseline gap-2 text-sm">
                <span className="font-medium">{label}</span>
                <Tag>{f.source}</Tag>
              </p>
              <p className="text-sm text-graphite">{f.detail}</p>
              {f.excerpt && (
                <pre className="mt-1 max-h-32 overflow-auto bg-paper p-2 font-mono text-xs break-words whitespace-pre-wrap">
                  {f.excerpt}
                </pre>
              )}
              {(f.rules?.length || f.score != null) && (
                <p className="mt-1 font-mono text-[10px] text-faint">
                  {f.rules?.length ? `rules: ${f.rules.join(", ")}` : ""}
                  {f.rules?.length && f.score != null ? " · " : ""}
                  {f.score != null ? `classifier ${f.score.toFixed(2)}` : ""}
                </p>
              )}
            </li>
          );
        })}
      </ul>
      <p className="border-t border-rule pt-3 text-xs text-faint">
        Flagged or unscanned input never reached an LLM; that step fell back to code. Excerpts are shown as plain text.
      </p>
    </div>
  );
}

const VERDICT = {
  yes: ["Photos match the description", "text-moss"],
  no: ["Photos don't match the description", "text-rust"],
  unclear: ["Unclear", "text-amber"],
} as const;

function describePhoto(c: PhotoCheck) {
  if (c.error) return c.error;
  const parts = [];
  const m = c.metadata;
  if (m?.camera) parts.push(m.camera);
  if (m?.taken_at) parts.push(`taken ${day(m.taken_at)}`);
  if (m && !m.has_exif) parts.push("no camera metadata");
  const r = c.redactions;
  if (r && (r.faces || r.text_regions)) parts.push(`blurred: ${r.faces} face(s), ${r.text_regions} text region(s)`);
  if (c.ai_generated) parts.push(`AI-image score ${c.ai_generated.score.toFixed(2)}`);
  return parts.join(" · ") || "Nothing notable";
}

/** The vision model's look at the redacted photos, and per-photo checks. */
export function PhotoReviewPanel({ review, checks }: { review?: PhotoReview; checks: PhotoCheck[] }) {
  return (
    <div className="space-y-4">
      {review && (
        <div>
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <p className={`font-serif text-2xl ${VERDICT[review.matches_description][1]}`}>
              {VERDICT[review.matches_description][0]}
            </p>
            <SourceTag source={review.source} />
          </div>
          <p className="mt-1 text-sm leading-relaxed">{review.summary}</p>
          {review.damage_seen.length > 0 && (
            <p className="mt-2 text-sm text-graphite">Seen: {review.damage_seen.join("; ")}</p>
          )}
          {review.concerns.length > 0 && (
            <ul className="mt-2 space-y-1 text-sm">
              {review.concerns.map((c) => (
                <li key={c} className="border-l-2 border-amber pl-2">
                  {c}
                </li>
              ))}
            </ul>
          )}
          <p className="mt-2 text-xs text-faint">
            {review.photos_shown} of {checks.length} photo(s) shown to the vision model, redacted: faces and text
            pixelated, no metadata.
          </p>
        </div>
      )}
      {checks.length > 0 && (
        <ul className="space-y-1.5 border-t border-rule pt-3 text-sm">
          {checks.map((c) => {
            const flagged = c.metadata?.findings.filter((f) => f.severity !== "info") ?? [];
            return (
              <li key={c.photo_index}>
                <span className="font-mono text-xs">Photo {c.photo_index + 1}</span>{" "}
                <span className="text-graphite">{describePhoto(c)}</span>
                {flagged.map((f) => (
                  <span key={f.code} className={`block pl-14 text-xs ${f.severity === "high" ? "text-rust" : "text-amber"}`}>
                    {f.detail}
                  </span>
                ))}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
