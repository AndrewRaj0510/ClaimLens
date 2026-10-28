import type { ClaimStatus, DecisionAction, Recommendation, Role } from "./types";

const usd = new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" });

export function money(value: number | null | undefined): string {
  return value == null ? "—" : usd.format(value);
}

/** "2026-09-01" or an ISO timestamp -> "Sep 1, 2026". Shown in UTC, as stored. */
export function day(value: string | null | undefined): string {
  if (!value) return "—";
  const d = new Date(value.length === 10 ? `${value}T00:00:00Z` : value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  });
}

export function time(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleTimeString("en-GB", { hour12: false, timeZone: "UTC" });
}

export function shortId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}

/** Customers and adjusters see the same statuses described differently. */
const STATUS_LABELS: Record<Role, Record<ClaimStatus, string>> = {
  customer: {
    processing: "Being assessed",
    awaiting_retake: "We need new photos",
    awaiting_approval: "With an adjuster",
    completed: "Decided",
    failed: "Delayed",
    interrupted: "Delayed",
  },
  adjuster: {
    processing: "Processing",
    awaiting_retake: "Awaiting photos",
    awaiting_approval: "Needs decision",
    completed: "Completed",
    failed: "Failed",
    interrupted: "Interrupted",
  },
};

export function statusLabel(status: ClaimStatus, role: Role): string {
  return STATUS_LABELS[role][status] ?? status;
}

export type Tone = "busy" | "attention" | "good" | "bad";

export function statusTone(status: ClaimStatus): Tone {
  switch (status) {
    case "processing":
      return "busy";
    case "awaiting_retake":
    case "awaiting_approval":
      return "attention";
    case "completed":
      return "good";
    default:
      return "bad";
  }
}

export const RECOMMENDATION_LABELS: Record<Recommendation, string> = {
  approve: "Approve",
  review: "Review",
  deny: "Deny",
  refer_to_investigator: "Refer to investigator",
};

export const DECISION_LABELS: Record<DecisionAction, string> = {
  approve: "Approved",
  edit: "Approved, amount edited",
  reject: "Not approved",
};

export function titleCase(text: string): string {
  return text.replace(/_/g, " ").replace(/(^|\s)\w/g, (m) => m.toUpperCase());
}

/** How far along the customer's four steps (received, assessed, with an adjuster, decided) a claim is. */
export function customerStep(status: ClaimStatus): number {
  const steps: Record<ClaimStatus, number> = {
    processing: 0,
    awaiting_retake: 0,
    failed: 0,
    interrupted: 0,
    awaiting_approval: 2,
    completed: 3,
  };
  return steps[status];
}
